from Modules.audio.src.aed import preprob as aed_preprob
from Modules.audio.src.aed import process as aed_process
from Modules.audio.src.aed import feats as aed_feats
from Modules.audio.src.asc import preprob as asc_preprob
from Modules.audio.src.asc import process as asc_process
from Modules.audio.src.asc import feats as asc_feats
from Modules.audio.src.vad import read_wave, write_wave
from Modules.audio.src.vad import frame_generator
from utils import Logging

import os
import requests
import collections
import tensorflow as tf
import webrtcvad

class AudioEventDetection:
    model = None
    result = None
    path = os.path.dirname(os.path.abspath(__file__))

    def __init__(self):
        model_dir = os.path.join(self.path, 'model')
        self.aed_model_path = os.path.join(model_dir, '11model-035-0.6208-0.8758.tflite')
        self.asc_model_path = os.path.join(model_dir, '3model-183-0.3535-0.9388.tflite')

        self.aed_model = tf.lite.Interpreter(model_path=self.aed_model_path)
        self.asc_model = tf.lite.Interpreter(model_path=self.asc_model_path)

        self.aed_model.allocate_tensors()
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
            'model_result': []
        }
        for i, file in enumerate(files):
            if i % 10 == 0:
                print(Logging.i("{} {}/{}".format(model_name, i, len(files))))
            wavpath = os.path.join(sub_dir, file)
            logmel_data = asc_feats(wavpath)
            i = int(file.split('/')[-1].replace('.wav', ''))
            threshold = 0.5
            result = asc_process(i, threshold, logmel_data, self.asc_model, wavpath)
            asc_results['model_result'].append(result)
            os.remove(wavpath)
        return asc_results
