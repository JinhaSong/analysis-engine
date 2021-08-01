import datetime

import requests

from Modules.audio.src.vad import read_wave, write_wave
from Modules.audio.src.vad import frame_generator
from utils import Logging

import os
import collections
import tensorflow as tf
import webrtcvad

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
        vad = webrtcvad.Vad(int(3))
        asr_results = {
            'model_name': model_name,
            'model_result': []
        }
        audio, sample_rate = read_wave(path)
        frames = frame_generator(30, audio, sample_rate)
        frames = list(frames)
        segments = vad_collector(sample_rate, 30, 300, vad, frames)
        for i, segment in enumerate(segments):
            wavpath = os.path.join(sub_dir, '/chunk-%005d.wav'%(i))
            print(Logging.i("{} - {}".format(i, wavpath)))
            write_wave(wavpath, segment, sample_rate)
            with open(wavpath, 'rb') as wav:
                response = requests.post('https://speechapi.sogang.ac.kr:5013/client/dynamic/recognize', data=wav, verify=False,
                                    headers={'Connection': 'close'})
                result = ''.join(response.json()['hypotheses'][0]['utterance'])
                response.close()
            module_result = {
                'audio_url': wavpath,
                'audio_result': [{
                    'label': {
                        'description': result
                    }
                }]
            }
            asr_results['model_result'].append(module_result)
        return asr_results


def vad_collector(sample_rate, frame_duration_ms, padding_duration_ms, vad, frames):
    num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    triggered = False
    voiced_frames = []

    starttime = 0
    endtime = 0
    for frame in frames:
        is_speech = vad.is_speech(frame.bytes, sample_rate)

        if not triggered:
            ring_buffer.append((frame, is_speech))
            num_voiced = len([f for f, speech in ring_buffer if speech])
            if num_voiced > 0.9 * ring_buffer.maxlen:
                triggered = True
                starttime = ring_buffer[0][0].timestamp
                for f, s in ring_buffer:
                    voiced_frames.append(f)
                ring_buffer.clear()
        else:
            voiced_frames.append(frame)
            ring_buffer.append((frame, is_speech))
            num_unvoiced = len([f for f, speech in ring_buffer if not speech])
            if num_unvoiced > 0.9 * ring_buffer.maxlen:
                endtime = frame.timestamp + frame.duration
                triggered = False
                return b''.join([f.bytes for f in voiced_frames]), starttime, endtime
                ring_buffer.clear()
                voiced_frames = []
    if triggered:
        endtime = frame.timestamp + frame.duration
    if voiced_frames:
        return b''.join([f.bytes for f in voiced_frames]), starttime, endtime
