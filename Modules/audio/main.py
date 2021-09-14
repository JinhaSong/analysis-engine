import ast
import json

from utils import Logging

import socket
import os
import time

class AudioEventDetection:
    model = None
    result = None
    path = os.path.dirname(os.path.abspath(__file__))

    def __init__(self):
        model_dir = os.path.join(self.path, 'model')

    def inference_by_audio(self, data, infos):
        paths = data['paths']
        sub_dirs = data['sub_dirs']
        file_list = data['file_lists']

        result = []

        # ASR(Automatic Speech Recognition)
        asr_result = self.inference_asr(paths[2], sub_dirs[2] + "/")
        result.append(asr_result)

        self.result = result

        return self.result

    def inference_asr(self, path, sub_dir):
        model_name = 'automatic_speech_recognition'
        asr_results = {
            'model_name': model_name,
            'analysis_time': 0,
            'model_result': None
        }
        print(Logging.i("Start preprocessing"))

        start_time = time.time()
        result = ''

        try :
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect(('0.0.0.0', 10000))
                data_transferred = 0

                with open(path, 'rb') as f:
                    try:
                        data = f.read()  # 1024바이트 읽는다
                        data_transferred = s.sendall(data)  # 1024바이트 보내고 크기 저장
                    except Exception as ex:
                        print(ex)
                print(Logging.i("file transfer is successfully complete"))
                rdata = s.recv(1024)
                data_transferred = 0

                try:
                    while rdata:
                        result += rdata.decode('utf-8')
                        data_transferred += len(rdata)
                        rdata = s.recv(1024)
                except Exception as ex:
                    print(ex)
            asr_results['model_result'] = ast.literal_eval(result)
            s.close()

        except :
            asr_results['model_result'] = []
        end_time = time.time()
        asr_results['analysis_time'] = end_time - start_time
        print(Logging.i("Processing time: {}".format(asr_results['analysis_time'])))

        return asr_results

