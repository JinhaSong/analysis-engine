import shutil
import string
import argparse

import os
import time

import cv2
import copy
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F

from AnalysisEngine import settings
from Modules.dummy.main import Dummy
from Modules.scenetext.utils import CTCLabelConverter, AttnLabelConverter
from Modules.scenetext.dataset import RawDataset, AlignCollate
from Modules.scenetext.model import Model

from WebAnalyzer.utils.media import frames_to_timecode, timecode_to_frames

from Modules.scenetext.CRAFT_pytorch import Craft_inference
from utils import Logging


class SceneText:
    model = None
    result = None
    path = os.path.dirname(os.path.abspath(__file__))
    charset = open(os.path.join(path, 'kr_charset.txt'), 'r')
    charset_str = charset.readline()
    character = charset_str + '0123456789abcdefghijklmnopqrstuvwxyz'

    def __init__(self):
        super().__init__()
        parser = argparse.ArgumentParser()
        self.opt = {
            'workers': 1,
            'batch_size': 16,
            'saved_model': os.path.join(self.path, 'weights/best_accuracy.pth'),
            'batch_max_length': 25,
            'imgH': 32,
            'imgW': 100,
            'rgb': True,
            'sensitive': False,
            'PAD': False,
            'Transformation': 'TPS',
            'FeatureExtraction': 'ResNet',
            'SequenceModeling': 'BiLSTM',
            'Prediction': 'Attn',
            'num_fiducial': 20,
            'input_channel': 1,
            'output_channel': 512,
            'hidden_size': 256
        }

        if self.opt["sensitive"]:
            self.opt["character"] = string.printable[:-6]
        self.opt["num_gpu"] = torch.cuda.device_count()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.recognition_confidence_threshold = 0.0

        cudnn.benchmark = True
        cudnn.deterministic = True

        self.detection_net, self.refine_net = Craft_inference.load_craft_weights()
        self.recognition_net, self.converter = self.load_reconize_weights()


    def detect_one_image(self, img, base_dir, current_frame=0) :
        bboxes, crop_images, detect_box_img = Craft_inference.craft_one_image(self.detection_net, self.refine_net, img, current_frame, base_dir)

        bboxes = sorted(bboxes,key = lambda x : (x[0][0],x[0][1]))
        return bboxes, crop_images, detect_box_img # coordinates / bboxCrop / original_image with bounding_box

    def recognize_one_image(self, model, base_dir, bboxes = None) :
        AlignCollate_demo = AlignCollate(imgH=self.opt['imgH'], imgW=self.opt['imgW'], keep_ratio_with_pad=self.opt['PAD'])
        demo_data = RawDataset(root=base_dir, opt=self.opt)  # use RawDataset

        demo_loader = torch.utils.data.DataLoader(
            demo_data, batch_size=self.opt['batch_size'],
            shuffle=False,
            num_workers=int(self.opt['workers']),
            collate_fn=AlignCollate_demo, pin_memory=False)
        # predict
        model.eval()
        results = []

        with torch.no_grad():
            iter_check = 0
            i = 0

            for image_tensors, image_path_list in demo_loader:
                iter_check = iter_check + 1
                batch_size = image_tensors.size(0)
                image = image_tensors.to(self.device)
                length_for_pred = torch.IntTensor([self.opt['batch_max_length']] * batch_size).to(self.device)
                text_for_pred = torch.LongTensor(batch_size, self.opt['batch_max_length'] + 1).fill_(0).to(self.device)

                if 'CTC' in self.opt['Prediction']:
                    preds = model(image, text_for_pred)

                    preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                    _, preds_index = preds.max(2)
                    preds_str = self.converter.decode(preds_index.data, preds_size.data)

                else:
                    preds = model(input=image, text=text_for_pred, is_train=False)
                    _, preds_index = preds.max(2)
                    preds_str = self.converter.decode(preds_index, length_for_pred)


                preds_prob = F.softmax(preds, dim=2)
                preds_max_prob, _ = preds_prob.max(dim=2)
                if len(bboxes) > 0:
                    for img_name, pred, pred_max_prob in zip(image_path_list, preds_str, preds_max_prob):
                        try :
                            if 'Attn' in self.opt['Prediction']:
                                pred_EOS = pred.find('[s]')
                                pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                                pred_max_prob = pred_max_prob[:pred_EOS]

                            confidence_score = pred_max_prob.cumprod(dim=0)[-1]

                            bbox = bboxes[i]
                            bbox_coordinate = bbox

                            confidence = confidence_score.detach().cpu().numpy()

                            left_top = bbox_coordinate[0]
                            left_top = list(map(int,left_top))
                            right_bottom = bbox_coordinate[2]
                            right_bottom = list(map(int,right_bottom))

                            x = int(round(bbox[0][0]))
                            w = int(round(bbox[2][0])) - x
                            y = int(round(bbox[0][1]))
                            h = int(round(bbox[2][1])) - y
                            score = float(confidence)

                            result = {
                                'label':[
                                    {
                                        'description': pred,
                                        'score': score
                                    }
                                ],
                                'position': {
                                    'x': x,
                                    'y': y,
                                    'w': w,
                                    'h': h
                                }
                            }
                            results.append(result)
                        except:
                            pass
                        i += 1

        return results

    def load_reconize_weights(self) :
        if 'CTC' in self.opt['Prediction']:
            converter = CTCLabelConverter(self.character)
        else:
            converter = AttnLabelConverter(self.character)
        self.opt["num_class"] = len(converter.character)

        model = Model(self.opt)
        model = torch.nn.DataParallel(model).to(self.device)
        model.load_state_dict(torch.load(self.opt['saved_model'], map_location=self.device))

        return model, converter

    def inference_by_image(self, image_path):
        image = cv2.imread(image_path)

        result = {}
        base_dir = image_path.replace(".jpg", "")
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)

        bboxes, crop_images, detect_box_img = self.detect_one_image(image, base_dir, 0)
        results = self.recognize_one_image(self.recognition_net, base_dir, bboxes=bboxes)
        result["frame_result"]  = results
        shutil.rmtree(base_dir)
        return result

    def inference_by_video(self, frame_path_list, infos):
        video_info = infos['video_info']
        frame_urls = infos['frame_urls']
        fps = video_info['extract_fps']
        start_time = infos['start_time']
        print(Logging.i("Start inference by video"))
        results = {
            "model_name": "scene_text_recognition",
            "analysis_time": 0,
            "frame_results": []
        }

        base_frame_number = timecode_to_frames(start_time, fps)
        start_time = time.time()
        for idx, (frame_path, frame_url) in enumerate(zip(frame_path_list, frame_urls)):
            if idx % 10 == 0:
                print(Logging.i("Processing... (index: {}/{} / frame number: {} / path: {})".format(idx, len(frame_path_list), int((idx + 1) * fps), frame_path)))
            result = self.inference_by_image(frame_path)
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
