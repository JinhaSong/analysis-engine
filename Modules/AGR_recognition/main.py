import time

import cv2
import os
import numpy as np
import copy
import torch
from torchvision import models
from torch.backends import cudnn

from AnalysisEngine import settings
from Modules.dummy.main import Dummy
from Modules.AGR_recognition.backbone import EfficientDetBackbone
from Modules.AGR_recognition.efficientdet.utils import BBoxTransform, ClipBoxes
from Modules.AGR_recognition.utils.utils import preprocess, invert_affine, postprocess, preprocess_video
from Modules.AGR_recognition.age_gender_model.resnet_3head_pretrained import ResNet152_three_head
from WebAnalyzer.utils.media import frames_to_timecode, timecode_to_frames
from utils import Logging


class AGR_recognition(Dummy):
    model = None
    result = None
    path = os.path.dirname(os.path.abspath(__file__))

    def __init__(self):
        super().__init__()
        start_time = time.time()
        efficient_det_model = 'weights/efficientdet-d1_43_1247.pth'
        self.obj_list = ['Human_face']

        compound_coef = 1
        force_input_size = None  # set None to use default size
        self.classifier_input_size = (448, 448)

        anchor_ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]
        anchor_scales = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]

        self.threshold = 0.50
        self.iou_threshold = 0.50

        cudnn.fastest = True
        cudnn.benchmark = True

        self.class_to_idx = {"Man_face": [0, 0], "Woman_face": [0, 1], "Boy_face": [1, 0], "Girl_face": [1, 1]}

        self.age_dict = {0: "Adult", 1: "Child"}
        self.gender_dict = {0: "Man", 1: "Woman"}
        self.race_dict = {0: "Asian", 1: "White", 2: "Black"}

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(self.obj_list), ratios=anchor_ratios, scales=anchor_scales)
        model_path = os.path.join(self.path, efficient_det_model)
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        
        model.requires_grad_(False)
        model.eval()

        model = model.cuda()

        self.input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
        self.input_size = 1536 if force_input_size is None else force_input_size
        self.regressBoxes = BBoxTransform()
        self.clipBoxes = ClipBoxes()
        classifier_model_name = 'weights/4_149.pth'
        classifier_model_path = os.path.join(self.path, classifier_model_name)

        self.classifier_mean = [0.485, 0.456, 0.40]
        self.classifier_std = [0.229, 0.224, 0.225]

        resnet = models.resnet152()
        classifier_model = ResNet152_three_head(old_model=resnet)
        classifier_model = torch.nn.DataParallel(classifier_model)
        classifier_model.load_state_dict(torch.load(classifier_model_path))
        classifier_model.to(self.device)
        classifier_model.requires_grad_(False)
        classifier_model.eval()

        self.model = model
        self.classifier_model = classifier_model
        end_time = time.time()

        print(Logging.i("Model is successfully loaded({}, {}) - {}".format(efficient_det_model, classifier_model_name, end_time - start_time)))

    def step_function(self, values) :
        new_values = []
        for value in values :
            if value < 0 :
                value = 0
            new_values.append(value)
        return new_values

    def inference_by_image(self, image):
        ori_imgs, framed_imgs, framed_metas = preprocess(image, max_size=self.input_size)
        image = cv2.imread(image)
        
        x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
        x = x.to(torch.float32).permute(0, 3, 1, 2)

        _, regression, classification, anchors = self.model(x)
        out = postprocess(x, anchors, regression, classification, self.regressBoxes, self.clipBoxes, self.threshold, self.iou_threshold)
        out = invert_affine(framed_metas, out)

        each_bbox = {
            # 1 bbox & 1 object
            'label':
                [{'description': 'None', 'score': 0.0}],

            'position': {
                'x': 0.0,
                'y': 0.0,
                'w': 0.0,
                'h': 0.0
            }
        }

        results = {"frame_result": []}
        inference_image = copy.deepcopy(image)

        if out != None :
            for i in range(len(out[0]['rois'])) :
                x1,y1,x2,y2 = out[0]['rois'][i].astype(np.int)
                
                obj = self.obj_list[out[0]['class_ids'][i]]
                score = float(out[0]['scores'][i])

                if obj == 'Human_face' and score > 0.75 :
                    face_crop_img = image[y1:y2,x1:x2]

                    width = x2-x1
                    height = y2-y1

                    width_margin = width * 0.2
                    height_margin = height * 0.2

                    x1 = int(x1 - width_margin)
                    x2 = int(x2 + width_margin)
                    y1 = int(y1 - height_margin)
                    y2 = int(y2 + height_margin)

                    x1,y1,x2,y2 = self.step_function([x1,y1,x2,y2])

                    face_crop_img = cv2.resize(face_crop_img, self.classifier_input_size)

                    face_crop_img = cv2.cvtColor(face_crop_img,cv2.COLOR_BGR2RGB)
                    cv2.imwrite("./a.png",face_crop_img)

                    face_crop_img = (face_crop_img / 255 - self.classifier_mean) / self.classifier_std

                    face_crop_img = np.expand_dims(face_crop_img,axis=0)

                    face_crop_img = torch.from_numpy(face_crop_img).permute(0, 3, 1, 2).float()
                    face_crop_img = face_crop_img.to(self.device)

                    output = self.classifier_model(face_crop_img)
                    
                    age_output = output[0].argmax(1)[0]
                    gender_output = output[1].argmax(1)[0]
                    race_output = output[2].argmax(1)[0]

                    softmax = torch.nn.Softmax(dim=1)
                    age_conf = float(softmax(output[0])[0][age_output])
                    gender_conf = float(softmax(output[1])[0][gender_output])
                    race_conf = float(softmax(output[2])[0][race_output])
    
                    age_obj = self.age_dict[int(age_output)]
                    gender_obj = self.gender_dict[int(gender_output)]
                    race_obj = self.race_dict[int(race_output)]

                    obj = ""
                    if age_conf > 0.80 :
                        obj = obj+age_obj + " "
                        age_text = age_obj
                        # cv2.putText(inference_image," age: " + age_text , (int(x1),int(y2)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, color=(0,0,255), fontScale=1.5, thickness= 3)
                    else :
                        obj = obj+"Unknown "
                        age_text = "Unknown "

                    if gender_conf > 0.80 :
                        obj = obj+gender_obj + " "
                        gender_text = gender_obj
                        # cv2.putText(inference_image," gender: " + gender_text , (int(x1),int(y2)+40), fontFace=cv2.FONT_HERSHEY_SIMPLEX, color=(0,255,0), fontScale=1.5, thickness= 3)
                    else :
                        obj = obj+"Unknown "
                        gender_text = "Unknown "

                    if race_conf > 0.80 :
                        obj = obj+race_obj + " "
                        race_text = race_obj
                        # cv2.putText(inference_image," race_obj: " + race_text , (int(x1),int(y2)+80), fontFace=cv2.FONT_HERSHEY_SIMPLEX, color=(255,0,0), fontScale=1.5, thickness= 3)
                    else :
                        obj = obj+"Unknown "
                        race_text = "Unknown "

                    classifier_score = float(age_conf + gender_conf + race_conf)
                    classifier_score = classifier_score / 3

                    score = score * classifier_score
                    # inference_image = cv2.rectangle(inference_image,(int(x1),int(y1)),(int(x2),int(y2)),(0,0,255),thickness=3)

                each_bbox['label'][0]['description'] = obj
                each_bbox['label'][0]['score'] = score
                each_bbox['position']['x'] = int(x1)
                each_bbox['position']['y'] = int(y1)
                each_bbox['position']['w'] = int(x2-x1)
                each_bbox['position']['h'] = int(y2-y1)

                deep_copy = copy.deepcopy(each_bbox)
                results["frame_result"].append(deep_copy)

        return results

    def inference_by_video(self, frame_path_list, infos):
        video_info = infos['video_info']
        frame_urls = infos['frame_urls']
        start_timestamp = infos['start_time']
        fps = video_info['extract_fps']

        print(Logging.i("Start inference by video"))
        results = {
            "model_name": "age_gender_race_recognition",
            "analysis_time": 0,
            "frame_results": []
        }

        base_frame_number = timecode_to_frames(start_timestamp, fps)
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