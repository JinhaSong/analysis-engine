import os
import time

import cv2
import torch
# import torchvision.models as models
import torchvision
from torch import nn

import torch
import os
import numpy as np
import cv2
from PIL import Image
from torch.autograd import Variable
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch.nn.functional as F

from AnalysisEngine import settings
from Modules.places import wideresnet
from Modules.dummy.main import Dummy
from WebAnalyzer.utils.media import frames_to_timecode
from utils import Logging
from Modules.places.metrictracker import label_map

class Places17(Dummy):
    path = os.path.dirname(os.path.abspath(__file__))

    def __init__(self):
        super().__init__()
        model_name = 'resnet101-0722.pth.tar'
        classes_name = "classes.txt"
        model_path = os.path.join(self.path, model_name)
        classes_path = os.path.join(self.path, classes_name)

        self.topk = 5
        self.result = None
        print(Logging.i("Start loading model({})".format(model_name)))
        start_time = time.time()
        self.device = torch.device("cuda:0")
        self.classes = label_map(classes_path)
        state = torch.load(model_path, map_location='cuda:0')
        self.model = wideresnet.resnet101()
        self.model.load_state_dict(state['state_dict'], strict=False)
        self.model.to(self.device)

        self.mean = np.array([0.4914, 0.4822, 0.4465])
        self.std = np.array([0.2023, 0.1994, 0.2010])
        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop((224, 224)),
            torchvision.transforms.ToTensor(),
        ])
        end_time = time.time()
        print(Logging.i("Model is successfully loaded - {} sec".format(end_time - start_time)))

    def inference_by_video(self, frame_path_list, infos):
        results = []
        video_info = infos['video_info']
        frame_urls = infos['frame_urls']
        frame_dir_path = infos['frame_dir_path']
        fps = video_info['extract_fps']
        print(Logging.i("Start inference by video"))

        datasets = ImageFolder(frame_dir_path, transform=self.transforms, target_transform=None)
        data_loader = DataLoader(datasets, batch_size=1, shuffle=False, num_workers=1)

        allFiles, _ = map(list, zip(*data_loader.dataset.samples))

        for i, (input, label) in enumerate(data_loader):
            self.model.eval()
            input = input.to(self.device)

            output = self.model(input)
            loss = F.softmax(output, dim=1).data.squeeze()
            probs, idx = loss.sort(0, True)
            probs = probs.cpu().numpy()
            idx = idx.cpu().numpy()
            file_path = str(allFiles[i]).replace("/workspace", "")
            result = {
                "frame_number": int((i + 1) * fps),
                "frame_url": file_path,
                "frame_result": []
            }
            for j in range(0, self.topk):
                label = {'label': {
                    'description': self.classes[idx[j]],
                    'score': float(probs[j]) * 100,
                }}
                result['frame_result'].append(label)
            results.append(result)

        # for idx, (frame_path, frame_url) in enumerate(zip(frame_path_list, frame_urls)):
        #     if idx % 10 == 0:
        #         print(Logging.i("inference frame - frame number: {} / path: {}".format(int((idx + 1) * fps), frame_path)))
        #     result = self.inference_by_image(frame_path)
        #     result["frame_url"] = settings.MEDIA_URL + frame_url[1:]
        #     result["frame_number"] = int((idx + 1) * fps)
        #     result["timestamp"] = frames_to_timecode((idx + 1) * fps, fps)
        #     results.append(result)

        self.result = results

        return self.result