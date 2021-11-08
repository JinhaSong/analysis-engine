from collections import deque
import numpy as np
from math import pi, sqrt, exp


def labelMap(label_map_path):
    # {0:amusementpark, 1:aquarium ```}
    label_map = {}
    with open(label_map_path, 'r', encoding='utf-8') as t:
        for i, line in enumerate(t.readlines()):
            line = line.rstrip('\n')
            label_map[i] = line
    return label_map


def labelScore(label_map_path):
    # Initialize scores to zero {amusementpark:0, aquarium:0 ```}
    label_score = {}
    with open(label_map_path, 'r', encoding='utf-8') as t:
        for i, line in enumerate(t.readlines()):
            line = line.rstrip('\n')
            label_score[line] = 0
    return label_score


class gaussianGrouping:
    def __init__(self, frame_results, label_map_path, video_fps=30, window_size=3):
        self.frame_results = frame_results
        self.label_map_path = label_map_path
        self.video_fps = video_fps
        self.window_size = window_size
        self.sigma = 1.5

    def makeGaussian(self):
        n = self.window_size * 2 + 1
        r = range(-int(n / 2), int(n / 2) + 1)
        weight = [1 / (self.sigma * sqrt(2 * pi)) * exp(-float(x) ** 2 / (2 * self.sigma ** 2)) for x in r]
        weight = np.reshape(weight, (n, 1))  # Adding dimension
        return weight

    def frameScore(self):
        # entering the real scores {aquarium:0.5, aquarium:0.2 ```}
        label_score = labelScore(label_map_path=self.label_map_path)
        frame_scores = []
        for i in range(len(self.frame_results)):
            for result in self.frame_results[0]['frame_result']:
                frame_score = label_score
                top = result["label"]["description"]
                score = round(result["label"]["score"], 2)
                frame_score[top] = score
                frame_scores.append(frame_score)

        return frame_scores

    def placesContext(self):
        weight = self.makeGaussian()
        label_map = labelMap(self.label_map_path)
        frame_scores = self.frameScore()
        inference_list = []
        dq = deque([])

        for i in range(len(frame_scores)):
            if i < self.window_size or i >= len(frame_scores) - self.window_size:
                a = max(frame_scores[i].keys())
                inference_list.append(a)

            dq.append(frame_scores[i])
            if len(dq) == len(weight):
                sumMatrix = []
                for j in dq:
                    arr1 = list(map(float, (k for k in j.values())))
                    sumMatrix.append(arr1)
                sumMatrix = np.array(sumMatrix).reshape((len(dq), 16))
                mul = np.add.reduce(np.multiply(sumMatrix, weight), axis=0)
                max_index = np.argmax(mul)
                inference_list.append(label_map[max_index])
                dq.popleft()
        return inference_list

    def smoothing(self):
        inference_list = self.placesContext()
        result = []
        file_list = []
        for i in range(len(inference_list)):
            if i == 0:
                file_list.append(inference_list[i])
                continue

            if inference_list[i] == inference_list[i - 1]:
                file_list.append(inference_list[i])

            else:
                result.append(file_list)
                file_list = []
                file_list.append(inference_list[i])

        sequence_result = []
        countLength = 0
        for i in range(len(result)):
            description = result[i][0]
            start_frame = countLength * self.video_fps
            length = len(result[i])
            countLength += length
            end_frame = (countLength - 1) * self.video_fps
            frame_result = {"start_frame": start_frame, "end_frame": end_frame, "label": {}}
            label = {"description": description}
            frame_result["label"] = label
            sequence_result.append(frame_result)
        return sequence_result