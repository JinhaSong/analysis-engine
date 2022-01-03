import ast
import json

import requests
from torchvision.transforms import transforms as trn
from torchvision.datasets.folder import default_loader

import os
import torch
import torch.nn as nn
import time
from torch.utils.data import Dataset, DataLoader

from AnalysisEngine import settings
from Modules.food.darknet import Darknet, post_processing
from Modules.food.efficientnet import EfficientNet
from Modules.dummy.main import Dummy
from WebAnalyzer.utils.media import frames_to_timecode, timecode_to_frames
from utils import Logging

def load_class_names(namesfile):
    class_names = []
    with open(namesfile, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.rstrip()
        class_names.append(line)
    return class_names


class ListDataset(Dataset):
    def __init__(self, l, transform=None, load=True):
        self.l = l
        self.loader = default_loader  # self.feature_loader
        self.transform = trn.Compose([
            trn.Resize((224, 224)),
            trn.ToTensor(),
            trn.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.load = load
        if transform is not None:
            self.transform = transform

    def __getitem__(self, idx):
        path = self.l[idx]
        if self.load:
            frame = self.transform(self.loader(path))
            return path, frame
        else:
            frame = self.transform(path)
            return frame

    def __len__(self):
        return len(self.l)


class Food(Dummy):
    model = None
    result = None
    path = os.path.dirname(os.path.abspath(__file__))

    def __init__(self):
        super().__init__()
        start_time = time.time()

        yolo_weight_path = os.path.join(self.path, 'yolov4.weights')
        yolo_cfg_path = os.path.join(self.path, 'yolov4.cfg')
        classifier_weight_path = os.path.join(self.path, 'efficient.pt')
        food_classes_txt_path = os.path.join(self.path, 'classes.txt')

        self.coco_food_related_idx = [45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 60]
        self.coco_food_related_class = [
            "wine glass", "cup", "fork", "knife", "spoon", "bowl",
            "banana", "apple", "sandwich", "orange", "broccoli", "cake"
        ]
        self.food_classes = [i.strip() for i in open(food_classes_txt_path, 'r', encoding="utf-8").readlines()]

        self.detector = self.load_detector(yolo_cfg_path, yolo_weight_path)
        self.classifier = self.load_classifier(classifier_weight_path)
        self.prob = nn.Softmax(dim=1)

        self.detector_transform = trn.Compose([trn.Resize((608, 608)), trn.ToTensor()])

        self.classifer_transform = trn.Compose([trn.Resize(256),
                                                trn.CenterCrop(224),
                                                trn.ToTensor(),
                                                trn.Normalize([0.485, 0.456, 0.406],
                                                              [0.229, 0.224, 0.225])])

        end_time = time.time()
        print(Logging.i("Model is successfully loaded - {} sec".format(end_time - start_time)))

    def load_detector(self, yolo_cfg, yolo_weight, parallel=True):
        detector = Darknet(yolo_cfg)
        detector.load_weights(yolo_weight)
        detector = detector.cuda()
        if parallel:
            detector = torch.nn.DataParallel(detector)

        detector.eval()
        return detector

    def load_classifier(self, weight, parallel=True):
        classifier_ckpt = torch.load(weight)
        classifier = EfficientNet.from_pretrained('efficientnet-b0')
        classifier.load_state_dict(classifier_ckpt['model_state_dict'])
        classifier.cuda()
        if parallel:
            classifier = nn.DataParallel(classifier)
        classifier.eval()
        return classifier

    @torch.no_grad()
    def inference_by_image(self, image_path):
        conf_thr = 0.25
        nms_thr = .5
        cls_thr = .0
        img = default_loader(image_path)
        w, h = img.size[0], img.size[1]

        # detect bboxes
        image = self.detector_transform(img).unsqueeze(0)
        outputs = self.detector(image.cuda())
        boxes = post_processing(image, conf_thr, nms_thr, outputs)

        # filtering food related bbox
        food_boxes = [box for box in boxes[0] if box[-1] in self.coco_food_related_idx]

        result = {"frame_result": None}
        if len(food_boxes):
            # crop bbox
            crop_imgs = [img.crop((int(box[0] * w), int(box[1] * h), int(box[2] * w), int(box[3] * h))) for box
                         in food_boxes]
            food_loader = DataLoader(ListDataset(crop_imgs, transform=self.classifer_transform, load=False),
                                     batch_size=1,
                                     shuffle=False,
                                     num_workers=0)

            food_prob, food_indice = [], []
            for i, im in enumerate(food_loader):
                outputs = self.classifier(im.cuda())
                outputs = self.prob(outputs)
                prob, indice = torch.topk(outputs.cpu(), k=1)
                food_prob.extend(list(prob.numpy().flatten()))
                food_indice.extend(list(indice.numpy().flatten()))
            result["frame_result"] = [
                {
                    'label': [{
                        "description": self.food_classes[food_indice[i]],
                        "score": food_prob[i] * 100,
                    }],
                         "position": {"h": int(box[3] * h),
                                      "w": int(box[2] * w),
                                      "x": max(int(box[0] * w), 0),
                                      "y": max(int(box[1] * h), 0)}
                } for i, box in enumerate(food_boxes) if food_prob[i] > cls_thr
            ]
        return result

    @torch.no_grad()
    def inference_by_image_no_detection(self, image_path, object_info):
        cls_thr = .0
        img = default_loader(image_path)
        w, h = img.size[0], img.size[1]

        food_boxes = []
        for obj in object_info["frame_result"]:
            label = obj["label"][0]["description"]
            if label in self.coco_food_related_class :
                position = obj["position"]
                x1 = (position['x'])
                y1 = (position['y'])
                x2 = (x1 + position['w'])
                y2 = (y1 + position['h'])
                food_boxes.append((x1, y1, x2, y2))

        result = {"frame_result": None}
        if len(food_boxes):
            crop_imgs = [img.crop((int(box[0]), int(box[1]), int(box[2]), int(box[3]))) for box
                         in food_boxes]
            food_loader = DataLoader(ListDataset(crop_imgs, transform=self.classifer_transform, load=False),
                                     batch_size=1,
                                     shuffle=False,
                                     num_workers=0)

            food_prob, food_indice = [], []
            for i, im in enumerate(food_loader):
                outputs = self.classifier(im.cuda())
                outputs = self.prob(outputs)
                prob, indice = torch.topk(outputs.cpu(), k=1)
                food_prob.extend(list(prob.numpy().flatten()))
                food_indice.extend(list(indice.numpy().flatten()))
            result["frame_result"] = [
                {
                    'label': [{
                        "description": self.food_classes[food_indice[i]],
                        "score": food_prob[i] * 100,
                    }],
                         "position": {"h": int(box[3]),
                                      "w": int(box[2]),
                                      "x": max(int(box[0]), 0),
                                      "y": max(int(box[1]), 0)}
                } for i, box in enumerate(food_boxes) if food_prob[i] > cls_thr
            ]
        else:
            result["frame_result"] = []
        return result

    def inference_by_video(self, frame_path_list, infos):
        video_info = infos['video_info']
        object_info_text = infos['video_text']
        frame_urls = infos['frame_urls']
        start_timestamp = infos['start_time']
        fps = video_info['extract_fps']
        print(Logging.i("Start inference by video"))
        results = {
            "model_name": "food_detection",
            "analysis_time": 0,
            "frame_results": []
        }

        try:
            object_infos = ast.literal_eval(object_info_text)
            object_infos = object_infos["frame_results"]
        except:
            object_infos = None

        base_frame_number = timecode_to_frames(start_timestamp, fps)
        start_time = time.time()
        if object_infos is None:
            for idx, (frame_path, frame_url) in enumerate(zip(frame_path_list, frame_urls)):
                if idx % 10 == 0:
                    print(Logging.i("Processing(with detection)... (index: {}/{} / frame number: {} / path: {})".format(idx, len(frame_path_list), int((idx + 1) * fps), frame_path)))
                result = self.inference_by_image(frame_path)
                result["frame_url"] = settings.MEDIA_URL + frame_url[1:]
                result["frame_number"] = base_frame_number + int((idx + 1) * fps)
                result["timestamp"] = frames_to_timecode(result["frame_number"], fps)
                results["frame_results"].append(result)

            results["sequence_results"] = self.merge_sequence(results["frame_results"])
        else :
            for idx, (frame_path, frame_url) in enumerate(zip(frame_path_list, frame_urls)):
                if idx % 10 == 0:
                    print(Logging.i("Processing(no detection)... (index: {}/{} / frame number: {} / path: {})".format(idx, len(frame_path_list),int((idx + 1) * fps), frame_path)))
                result = self.inference_by_image_no_detection(frame_path, object_infos[idx])
                result["frame_url"] = settings.MEDIA_URL + frame_url[1:]
                result["frame_number"] = base_frame_number + int((idx + 1) * fps)
                result["timestamp"] = frames_to_timecode(result["frame_number"], fps)
                results["frame_results"].append(result)

            results["sequence_results"] = self.merge_sequence(results["frame_results"])

            end_time = time.time()
            results['analysis_time'] = end_time - start_time
            print(Logging.i("Processing time: {}".format(results['analysis_time'])))

        self.result = results

        return self.result

    def merge_sequence(self, result):
        sequence_results = []
        # TODO
        # - return format
        # [
        #     {
        #         "label": {
        #             "description": "class name",
        #             "score": 100 # 추가여부는 선택사항
        #         },
        #         "position": { # 추가여부는 선택사항
        #             "x": 100,
        #             "y": 100,
        #             "w": 100,
        #             "h": 100
        #         },
        #         "start_frame": 30,
        #         "end_frame": 300
        #     }
        #     ...
        # ]

        return sequence_results
