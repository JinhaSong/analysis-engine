import time

import torch
import sys
import os

import warnings
from transformers import BertTokenizer

from AnalysisEngine import settings
from Modules.image_captioning.datasets import coco
from Modules.image_captioning.configuration import Config
from PIL import Image

from WebAnalyzer.utils.media import frames_to_timecode
from utils import Logging


class ImageCaptioning:
    model = None
    result = None
    path = os.path.dirname(os.path.abspath(__file__))

    def __init__(self):
        start_time = time.time()
        self.config = Config()
        self.model = torch.hub.load('saahiluppal/catr', 'v3', pretrained=True)

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        self.start_token = self.tokenizer.convert_tokens_to_ids(self.tokenizer._cls_token)
        self.end_token = self.tokenizer.convert_tokens_to_ids(self.tokenizer._sep_token)
        end_time = time.time()
        print(Logging.i("Model is successfully loaded({} sec)".format(end_time - start_time)))


    def create_caption_and_mask(self, start_token, max_length):
        caption_template = torch.zeros((1, max_length), dtype=torch.long)
        mask_template = torch.ones((1, max_length), dtype=torch.bool)

        caption_template[:, 0] = start_token
        mask_template[:, 0] = False

        return caption_template, mask_template

    @torch.no_grad()
    def evaluate(self, img, caption, cap_mask):
        self.model.eval()
        for i in range(self.config.max_position_embeddings - 1):
            predictions = self.model(img, caption, cap_mask)
            predictions = predictions[:, i, :]
            predicted_id = torch.argmax(predictions, axis=1)

            if predicted_id[0] == 102:
                return caption

            caption[:, i + 1] = predicted_id[0]
            cap_mask[:, i + 1] = False

        return caption

    def inference(self, image):
        caption, cap_mask = self.create_caption_and_mask(self.start_token, self.config.max_position_embeddings)
        output = self.evaluate(image, caption, cap_mask)
        result = self.tokenizer.decode(output[0].tolist(), skip_special_tokens=True)
        result = result.capitalize()
        flag = True

        return result, flag

    def inference_by_image(self, image_path):
        image = Image.open(image_path)
        tmp_image = coco.val_transform(image).unsqueeze(0)
        result, flag = self.inference(tmp_image)

        frame_result = {"frame_result": []}
        frame_result["frame_result"].append({
            "label": [
                {
                    "description": result,
                    "score": None
                },
            ],
        })

        return frame_result

    def inference_by_video(self, frame_path_list, infos):
        video_info = infos['video_info']
        frame_urls = infos['frame_urls']
        fps = video_info['extract_fps']

        print(Logging.i("Start inference by video"))
        results = {
            "model_name": "image_captioning",
            "analysis_time": 0,
            "frame_results": []
        }

        start_time = time.time()
        for idx, (frame_path, frame_url) in enumerate(zip(frame_path_list, frame_urls)):
            if (idx % 50) == 0 and idx != 0:
                print(Logging.i("Processing...(frame number: {})".format(idx)))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = self.inference_by_image(frame_path)
            result["frame_url"] = settings.MEDIA_URL + frame_url[1:]
            result["frame_number"] = int((idx + 1) * fps)
            result["timestamp"] = frames_to_timecode((idx + 1) * fps, fps)
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