import os
import time
import cv2

from AnalysisEngine import settings
from Modules.yolov4 import darknet
from WebAnalyzer.utils.media import timecode_to_frames, frames_to_timecode

from utils import Logging

class YOLOv4:
    model = None
    result = None
    prob_thresh = 0.0
    nms_thresh = 0.0
    path = os.path.dirname(os.path.abspath(__file__))

    def __init__(self, prob_thresh=0.5, nms_thresh=0.45):
        super().__init__()
        self.model_name = "yolov4-608"
        config_path = os.path.join(self.path, "configs", "yolov4-608.cfg")
        data_file_path = os.path.join(self.path, "configs", "coco.data")
        model_path = os.path.join(self.path, "weights", "yolov4.weights")
        self.model = None
        self.class_names = None
        self.class_colors = None
        self.prob_thresh = prob_thresh
        self.nms_thresh = nms_thresh

        try:
            start_time = time.time()
            self.model, self.class_names, self.class_colors = darknet.load_network(
                config_path,
                data_file_path,
                model_path,
                batch_size=1
            )
            end_time = time.time()
            print(Logging.i("yolov4-608 - Model successfully is loaded. ({} sec)".format(end_time - start_time)))
        except :
            self.model = None
            print(Logging.e("yolov4-608 - Model is failed to load."))

    def image_detection(self, image, model, class_names, class_colors, prob_thresh, nms_thresh):
        # Darknet doesn't accept numpy images.
        # Create one with image we reuse for each detect
        width = darknet.network_width(model)
        height = darknet.network_height(model)
        darknet_image = darknet.make_image(width, height, 3)

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, (width, height),
                                   interpolation=cv2.INTER_LINEAR)

        darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())
        detections = darknet.detect_image(model, class_names, darknet_image, thresh=prob_thresh, nms=nms_thresh)
        darknet.free_image(darknet_image)
        image = darknet.draw_boxes(detections, image_resized, class_colors)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), detections

    def bbox2points(self, bbox):
        """
        From bounding box yolo format
        to corner points cv2 rectangle
        """
        x, y, w, h = bbox
        xmin = int(round(x - (w / 2)))
        xmax = int(round(x + (w / 2)))
        ymin = int(round(y - (h / 2)))
        ymax = int(round(y + (h / 2)))
        return xmin, ymin, xmax, ymax

    def inference_by_image(self, image_path):
        image = cv2.imread(image_path)
        if self.model != None :
            result = []
            image_width = image.shape[1]
            image_height = image.shape[0]
            image, detections = self.image_detection(
                image,
                self.model,
                self.class_names,
                self.class_colors,
                self.prob_thresh,
                self.nms_thresh
            )

            # Drawing each detection on the image
            for label, confidence, bbox in detections:
                left, top, right, bottom = self.bbox2points(bbox)

                left = int(left / darknet.network_width(self.model) * image_width)
                top = int(top / darknet.network_height(self.model) * image_height)
                right = int(right / darknet.network_width(self.model) * image_width)
                bottom = int(bottom / darknet.network_height(self.model) * image_height)

                if left <= 0: left = 1
                if top <= 0: top = 1
                if right <= 0: right = 1
                if bottom <= 0: bottom = 1

                width = right - left
                height = bottom - top

                if left > image_width: left = image_width
                if top > image_height: top = image_height
                if width > image_width: width = image_width
                if height > image_height: height = image_height

                if height > 10 and width > 10:
                    result.append({
                        'label': [
                            {
                                'description': label,
                                'score': float(confidence)/100,
                            }
                        ],
                        'position': {
                            'x': left,
                            'y': top,
                            'w': width,
                            'h': height
                        }
                    })
        else :
            ret = []
        ret = {}
        ret["frame_result"] = result
        print(ret)
        return ret


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
