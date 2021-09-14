from Modules.audio.src.asc import preprob as asc_preprob
from Modules.audio.src.asc import process as asc_process
from Modules.audio.src.asc import feats as asc_feats
from utils import Logging

import os
import requests
import collections
import tensorflow as tf
import webrtcvad
import time

class AudioEventDetection:
    model = None
    result = None
    path = os.path.dirname(os.path.abspath(__file__))

    def __init__(self):
        model_dir = os.path.join(self.path, 'model')
        self.asc_model_path = os.path.join(model_dir, '3model-183-0.3535-0.9388.tflite')
        self.asc_model = tf.lite.Interpreter(model_path=self.asc_model_path)
        self.asc_model.allocate_tensors()

    def inference_by_audio(self, data, infos):
        paths = data['paths']
        sub_dirs = data['sub_dirs']
        file_list = data['file_lists']

        result = []

        # ASC(Audio Scene Classification)
        asc_result = self.inference_asc(paths[1], sub_dirs[1] + "/")
        result.append(asc_result)

        self.result = result

        return self.result

    def inference_asc(self, path, sub_dir):
        model_name = 'audio_scene_classification'
        asc_preprob(path, sub_dir, False)
        files = sorted([_ for _ in os.listdir(sub_dir) if _.endswith('.wav')])
        asc_results = {
            'model_name': model_name,
            'analysis_time': 0,
            'model_result': []
        }
        start_time = time.time()
        for i, file in enumerate(files):
            if i % 10 == 0:
                print(Logging.i("Processing... (index: {}/{})".format(i, len(files))))
            wavpath = os.path.join(sub_dir, file)
            logmel_data = asc_feats(wavpath)
            i = int(file.split('/')[-1].replace('.wav', ''))
            threshold = 0.5
            result = asc_process(i, threshold, logmel_data, self.asc_model, wavpath)
            asc_results['model_result'].append(result)
            os.remove(wavpath)
        end_time = time.time()
        asc_results['analysis_time'] = end_time - start_time
        print(Logging.i("Processing time: {}".format(asc_results['analysis_time'])))

        return asc_results
