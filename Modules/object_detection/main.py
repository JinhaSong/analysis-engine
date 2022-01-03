import json
#from AnalysisModule import settings
#from Modules.dummy.example import test
#from WebAnalyzer.utils.media import frames_to_timecode

import torch
from torch.backends import cudnn
from matplotlib import colors

from AnalysisEngine import settings
from Modules.dummy.main import Dummy
from Modules.object_detection.backbone import EfficientDetBackbone
import cv2
import numpy as np
import os
import copy
import shutil
import time

from Modules.object_detection.efficientdet.utils import BBoxTransform, ClipBoxes
from Modules.object_detection.utils.utils import preprocess_video, invert_affine, postprocess, STANDARD_COLORS, standard_to_bgr, get_index_label, plot_one_box
from WebAnalyzer.utils.media import frames_to_timecode, timecode_to_frames
from utils import Logging


class ObjectDetection(Dummy):
    model = None
    result = None
    path = os.path.dirname(os.path.abspath(__file__))

    obj_list = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
           'fire hydrant', '', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
           'cow', 'elephant', 'bear', 'zebra', 'giraffe', '', 'backpack', 'umbrella', '', '', 'handbag', 'tie',
           'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
           'skateboard', 'surfboard', 'tennis racket', 'bottle', '', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
           'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
           'cake', 'chair', 'couch', 'potted plant', 'bed', '', 'dining table', '', '', 'toilet', '', 'tv',
           'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
           'refrigerator', '', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
           'toothbrush']

    def __init__(self):
        super().__init__()
        start_time = time.time()
        self.compound_coef = 4
        self.force_input_size = None  # set None to use default size

        # replace this part with your project's anchor config
        self.anchor_ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]
        self.anchor_scales = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]

        self.threshold = 0.40
        self.iou_threshold = 0.40

        cudnn.fastest = True
        cudnn.benchmark = True

        model = EfficientDetBackbone(compound_coef=self.compound_coef, num_classes=len(self.obj_list),
                                     ratios=self.anchor_ratios, scales=self.anchor_scales)

        model.load_state_dict(torch.load(os.path.join(self.path, 'model', 'efficientdet-d4.pth')))
        model.requires_grad_(False)
        model.eval()
        model = model.cuda()

        input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
        self.input_size = input_sizes[self.compound_coef] if self.force_input_size is None else self.force_input_size
        self.regressBoxes = BBoxTransform()
        self.clipBoxes = ClipBoxes()

        self.model = model
        end_time = time.time()
        print(Logging.i("Model is successfully loaded - {} sec".format(end_time - start_time)))
        

    def inference_by_image(self, image_path):
        image = cv2.imread(image_path)

        ori_imgs, framed_imgs, framed_metas = preprocess_video(image, max_size=self.input_size)
        

        x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)

        x = x.to(torch.float32).permute(0, 3, 1, 2)

        def inference():
            _, regression, classification, anchors = self.model(x)
            out = postprocess(x, anchors, regression, classification, self.regressBoxes, self.clipBoxes, self.threshold, self.iou_threshold)
            out = invert_affine(framed_metas, out)
            return out
        
        out = inference()

        each_bbox = {
            'label':
                [{'description': 'None', 'score': 0.0}],

            'position': {
                'x': 0.0,
                'y': 0.0,
                'w': 0.0,
                'h': 0.0
            }
        }

        result = {"frame_result": None}
        object_detection = []
        if out != None :
            for i in range(len(out[0]['rois'])) :
                x1,y1,x2,y2 = out[0]['rois'][i].astype(np.int)
                obj = self.obj_list[out[0]['class_ids'][i]]
                score = float(out[0]['scores'][i])

                each_bbox['label'][0]['description'] = obj
                each_bbox['label'][0]['score'] = score
                each_bbox['position']['x'] = int(x1)
                each_bbox['position']['y'] = int(y1)
                each_bbox['position']['w'] = int(x2-x1)
                each_bbox['position']['h'] = int(y2-y1)

                deep_copy = copy.deepcopy(each_bbox)
                object_detection.append(deep_copy)
        result["frame_result"] = object_detection
        self.result = result
        return self.result

    def inference_by_video(self, frame_path_list, infos):
        video_info = infos['video_info']
        frame_urls = infos['frame_urls']
        start_timestamp = infos['start_time']
        fps = video_info['extract_fps']
        print(Logging.i("Start inference by video"))
        results = {
            "model_name": "object_detection",
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
