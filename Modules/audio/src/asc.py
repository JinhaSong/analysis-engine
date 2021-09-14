import sys
import soundfile as sound
import numpy as np
import datetime
import librosa

from utils import Logging

sr = 48000
duration = 10
num_freq_bin = 128
num_fft = 2048
hop_length = int(num_fft / 2)
num_time_bin = int(np.ceil(duration * sr / hop_length))
num_channel = 1
use_norm = False
use_delta = False


classes = [['indoor'],['outdoor'],['transportation']]


def preprob(input_path, output_path, NORM):
    y, sr = sound.read(input_path)
    duration = y.shape[0]/sr
    
    seconds_division = 50
    window_length = int(sr/seconds_division)
    y_abs = np.absolute(y)
    y_mean = y_abs.mean()
    
    k = 5
    ratio = 0.3
    sec = 48000
    win = 10
    cnt = 0
    total_length = int(duration*seconds_division)/k
    
    if NORM :
        for n in range(5):
            normalized = []
            for i in range(n*int(total_length), (n+1)*int(total_length)):
                #if i % 1000 == 0:
                #    print (i)
                window = y[i*window_length : (i+1)*window_length]
                window_abs = np.absolute(window)
                if (window_abs.mean() > y_mean*ratio):
                    normalized = np.concatenate((normalized, window))
                else:
                    zero_window = np.zeros(window.shape)
                    normalized = np.concatenate((normalized, zero_window))
            print("iteration:",n, "duration:",duration/k, "normalized_duration:",normalized.shape[0]/sr)
            for i in range(int(normalized.shape[0]/sec)-3):
                cnt = cnt+1
                sound.write(output_path + '{:05}'.format(cnt) + '.wav', normalized[i*sec : (i+win)*sec], sr)
            print(Logging.i("number of accumulated files: {}".format(cnt)))
    else:
        for i in range(int(y.shape[0]/sec)):
            cnt = cnt+1
            sound.write(output_path + '{:05}'.format(cnt) + '.wav', y[i*sec : (i+win)*sec], sr)
        print(Logging.i("number of accumulated files: {}".format(cnt)))
        






def deltas(X_in):
    X_out = (X_in[:,2:,:]-X_in[:,:-2,:])/10.0
    X_out = X_out[:,1:-1,:]+(X_in[:,4:,:]-X_in[:,:-4,:])/5.0
    return X_out


def feats(wavpath):
    y, sr = sound.read(wavpath)
    logmel_data = np.zeros((num_freq_bin, num_time_bin, num_channel), 'float32')
    
    max_len = int(duration*sr)
    y_len = len(y)
    if y_len >= max_len:
        y = y[:max_len]
    else:
        num_repeats = round((max_len/y_len) + 1)
        y_repeat = y.repeat(num_repeats, 0)
        padded_y = y_repeat[:max_len]
        y = padded_y
    
    logmel_data[:,:,0] = librosa.feature.melspectrogram(y[:], 
                                                        sr=sr, 
                                                        n_fft=num_fft, 
                                                        hop_length=hop_length,
                                                        n_mels=num_freq_bin, 
                                                        fmin=0.0, 
                                                        fmax=sr/2, 
                                                        htk=True, 
                                                        norm=None)
    
    logmel_data = np.log(logmel_data+1e-8)
    if use_norm:
        logmel_data = (logmel_data - np.min(logmel_data)) / (np.max(logmel_data) - np.min(logmel_data))
    return logmel_data


def time(sec):
    return str(datetime.timedelta(seconds=sec))


def process(i, threshold, logmel_data, model, wavpath):
    unknown_flag = False
    
    if use_delta:
        logmel_data_deltas = deltas(logmel_data)
        logmel_data_deltas_deltas = deltas(logmel_data_deltas)
        logmel_data = np.concatenate((logmel_data[:,4:-4,:], logmel_data_deltas[:,2:-2,:], logmel_data_deltas_deltas), axis=-1)
        
    input_index = model.get_input_details()[0]["index"]
    output_index = model.get_output_details()[0]["index"]
    
    test_image = np.expand_dims(logmel_data, axis=0).astype(np.float32)
    model.set_tensor(input_index, test_image)
    model.invoke()
    softmax = model.get_tensor(output_index)
    result = np.argmax(softmax[0])
    
    if float(softmax[0][int(result)]) > threshold:
        unknown_flag = False
        out_softmax = softmax
        out_result = result
        out_classes = classes
    else:
        unknown_flag = True

    results = {
        'audio_url': '/media' + wavpath.split('/media')[1],
        'timestamp': str(time(round(i * 0.1, 1))),
        'result': []
    }
    if unknown_flag == True:
        results['result'].append({
            'label': {
                'score': str(0.0),
                'description': 'unknown'
            }
        })
    else:
        results['result'].append({
            'label': {
                'score': str(out_softmax[0][int(out_result)]),
                'description': out_classes[int(out_result)][0]
            }
        })
    return results
