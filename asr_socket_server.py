import datetime
import json
import socket
import time

import webrtcvad
import collections
import requests
import urllib3
import os
import re
from utils import Logging

from AnalysisEngine import settings

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)



from Modules.audio.src.vad import read_wave, write_wave
from Modules.audio.src.vad import frame_generator

starttime=0.0
endtime=0.0

def vad_collector(sample_rate, frame_duration_ms,
                  padding_duration_ms, vad, frames):
    num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    triggered = False
    voiced_frames = []

    global starttime
    global endtime
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
                yield b''.join([f.bytes for f in voiced_frames])
                ring_buffer.clear()
                voiced_frames = []
    if triggered:
        endtime = frame.timestamp + frame.duration
    if voiced_frames:
        yield b''.join([f.bytes for f in voiced_frames])


print(Logging.i("================================================"))
print(Logging.s("Automatic Speech Recognition Socket Server Start"))
print(Logging.s("================================================"))

vad = webrtcvad.Vad(int(3))

dirpath = 'vad2'

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind(("0.0.0.0", 10000))
    s.listen()
    while True:
        conn, addr = s.accept()
        with conn:
            conn.settimeout(1)
            print(Logging.i("Connection info: {}:{} - {}".format(addr[0], addr[1],' connected')))

            data = conn.recv(1024)
            data_transferred = 0

            date_today = datetime.date.today().strftime("%Y%m%d")
            now = datetime.datetime.now().strftime("%H%M%S%f")
            directory_today = os.path.join(settings.MEDIA_ROOT, date_today)
            directory_time = os.path.join(directory_today, now)
            directory = os.path.join(directory_time, dirpath)

            if not os.path.exists(directory_today):
                os.makedirs(directory_today)
            if not os.path.exists(directory_time):
                os.makedirs(directory_time)
            if not os.path.exists(directory):
                os.makedirs(directory)

            inputfilepath = os.path.join(directory, 'input.wav'.format())
            processing_start = time.time()
            with open(inputfilepath, 'wb') as f:
                try:
                    while data:
                        f.write(data)
                        data_transferred += len(data)
                        data = conn.recv(1024)
                    f.close()
                except Exception as ex:
                    print(Logging.e(ex))
            print(Logging.i('Received info - file: %s / transfer amount: %d' %(inputfilepath, data_transferred)))
            audio, sample_rate = read_wave(inputfilepath)
            frames = frame_generator(30, audio, sample_rate)
            frames = list(frames)
            segments = vad_collector(sample_rate, 30, 300, vad, frames)

            results = []


            for i, segment in enumerate(segments):
                path = os.path.join(directory, 'chunk-%005d.wav' % (i,))
                print(Logging.i("Processing chunk - {}").format(path))
                write_wave(path, segment, sample_rate)
                with open(path, 'rb') as wav:
                    try:
                        res = requests.post('https://speechapi.sogang.ac.kr:5013/client/dynamic/recognize', data=wav,
                                            verify=False, headers={'Connection': 'close'})
                        result = ''.join(res.json()['hypotheses'][0]['utterance'])
                        res.close()
                    except:
                        result = ""
                start_time = time.strftime('%H:%M:%S', time.gmtime(float(starttime))) + '.' + format(int(float(starttime) * 100 % 100),"02")
                end_time = time.strftime('%H:%M:%S', time.gmtime(float(endtime))) + '.' + format(int(float(endtime) * 100 % 100),"02")
                audio_url = path.replace(settings.MEDIA_ROOT, "")
                result = str(result).replace("▁", "")

                segment_result = {
                    'audio_url': audio_url,
                    'start_time': start_time,
                    'end_time': end_time,
                    'audio_result': result
                }
                results.append(segment_result)
            processing_end = time.time()

            data_transferred = 0
            print(Logging.i("Processing complete - file: {} / time: {} sec".format(inputfilepath, processing_end - processing_start)))
            try:
                data_transferred = conn.sendall(json.dumps(results, indent='\t').encode('utf-8'))
            except Exception as ex:
                print(Logging.e(ex))
            print(Logging.i("Result transfer was successfully completed"))