import json
import glob
import os
import pickle
import joblib
import numpy as np
import pandas as pd
import requests

from Modules.dummy.main import Dummy
from Modules.Final.utils.utils import *

class Final(Dummy):
    path = os.path.dirname(os.path.abspath(__file__))
    module_names = [
        "places_recognition",
        "food_detection",
        "object_detection",
        # "image_captioning",
        "audio_event_detection",
        "audio_scene_classification",
        "automatic_speech_recognition",
        "age_gender_race_recognition",
        "scene_text_recognition",
        "text_rank",
        "human_object_interaction",
    ]

    tasks = ['where', 'race', 'age', 'gender', 'place', 'food', 'doing', 'etc', 'event']

    def __init__(self):
        super().__init__()
        with open(os.path.join(self.path, "data/all_words.pkl"), "rb") as f:
            self.all_word = pickle.load(f, encoding='utf-8')
        with open(os.path.join(self.path, "data/features.pkl"), "rb") as f:
            self.features = pickle.load(f, encoding='utf-8')
        with open(os.path.join(self.path, "data/idx2label.pkl"), "rb") as f:
            self.idx2labels = pickle.load(f, encoding='utf-8')
        with open(os.path.join(self.path, "data/scenetext.pkl"), "rb") as f:
            self.scenetext_words = pickle.load(f, encoding='utf-8')
        with open(os.path.join(self.path, "data/url_words.pkl"), "rb") as f:
            self.url_words = pickle.load(f, encoding='utf-8')

        self.all_model = []
        for task in self.tasks:
            self.all_model.append([joblib.load(open(os.path.join(self.path, f'model/{task}/{model_name}'), 'rb'))
                                   for model_name in [i for i in [i for i in os.listdir(os.path.join(self.path, f'model/{task}/')) if i != '.ipynb_checkpoints']]])


    def unnest(self, ls):
        return [item for sublist in ls for item in sublist]

    def make_df(self, place_data, food_data, objects_data, audio_event_data, audio_scene_data, person_data, scenetext_data, urltext_data, relation_data,
                scenetext_words, url_words, all_word, method='frame', scene_info=None):
        if method=='frame':
            place = find_place(place_data)
            food = find_food(food_data)
            objects = find_object(objects_data)
            audio_event = find_event_audio(audio_event_data, place_data)
            audio_scene = find_scene_audio(audio_scene_data, place_data)
            person = find_person(person_data)
            scenetext = find_scenetext(scenetext_data, scenetext_words)
            urltext = find_url(urltext_data, url_words)
            relation = find_relation(relation_data)
        elif method=='auto_scene':
            place = find_place(place_data, 'auto_scene', scene_info)
            food = find_food(food_data, 'auto_scene', scene_info)
            objects = find_object(objects_data, 'auto_scene', scene_info)
            audio_event = find_event_audio(audio_event_data, place_data, 'auto_scene', scene_info)
            audio_scene = find_scene_audio(audio_scene_data, place_data, 'auto_scene', scene_info)
            person = find_person(person_data, 'auto_scene', scene_info)
            scenetext = find_scenetext(scenetext_data, scenetext_words, 'auto_scene', scene_info)
            urltext = find_url(urltext_data, url_words)
            relation = find_relation(relation_data, 'auto_scene', scene_info)

        idx_name = []
        for i in range(len(place)):
            if len(str(i+1)) == 1:
                idx_name.append(f'frame0{i+1}')
            else:
                idx_name.append(f'frame{i+1}')

        df = pd.DataFrame({'name':idx_name})
        df[sorted(all_word)] = 0.0

        for idx, data in enumerate(zip(*(place, food, objects, audio_event, audio_scene, person, scenetext, relation))):
            row_data = self.unnest(data)
            for i in row_data:
                df.at[idx, i[0]] = i[1]
        for i in urltext:
            df[i[0]] = i[1]
        return df

    def scene_inference(self, results, method='frame'):

        place_data         = results[self.module_names[0]]
        food_data          = results[self.module_names[1]]
        object_data        = results[self.module_names[2]]
        audio_event_data   = results[self.module_names[4]]
        audio_scene_data   = results[self.module_names[5]]
        person_data        = results[self.module_names[7]]
        scenetext_data     = results[self.module_names[8]]
        urltext_data       = results[self.module_names[9]]
        relation_data      = results[self.module_names[10]]

        if method =='frame':
            df = self.make_df(place_data, food_data, object_data, audio_event_data, audio_scene_data, person_data,
                              scenetext_data, urltext_data, relation_data, self.scenetext_words, self.url_words, self.all_word)
            results = {}
            for models, task, idx2label in zip(self.all_model, self.tasks, self.idx2labels):
                prob_ls = []
                for model in models:
                    if task == 'race':
                        pred = model.predict(df[self.features[task]])
                        prob_ls.append(pred)
                    else:
                        pred = model.predict_proba(df[self.features[task]])
                        prob_ls.append(pred)
                if task == 'race':
                    pred = [max(i, key=i.count) for i in zip(*prob_ls)]
                    pred = max(pred, key=pred.count)
                else:
                    pred = np.mean(prob_ls, axis=0).tolist()
                    pred = [np.argmax(i) for i in pred]
                    pred = max(pred, key=pred.count)
                results[task] = idx2label[pred]
                final_results = {"results": results}

            return final_results

        elif method=='auto_scene':
            scene_ls = []
            for scene in place_data['result']['sequence_results']:
                if scene['end_frame'] - scene['start_frame'] == 0:
                    continue
                else:
                    scene_ls.append(scene)
            scene_results = {}
            for idx, scene_info in enumerate(scene_ls):
                start_frame, end_frame = scene_info['start_frame'], scene_info['end_frame']
                df = self.make_df(place_data, food_data, object_data, audio_event_data, audio_scene_data, person_data,
                              scenetext_data, urltext_data, relation_data, self.scenetext_words, self.url_words, self.all_word,
                              'auto_scene', scene_info)
                results = {}
                for models, task, idx2label in zip(self.all_model, self.tasks, self.idx2labels):
                    prob_ls = []
                    for model in models:
                        if task == 'race':
                            pred = model.predict(df[self.features[task]])
                            prob_ls.append(pred)
                        else:
                            pred = model.predict_proba(df[self.features[task]])
                            prob_ls.append(pred)
                    if task == 'race':
                        pred = [max(i, key=i.count) for i in zip(*prob_ls)]
                        pred = max(pred, key=pred.count)
                    else:
                        pred = np.mean(prob_ls, axis=0).tolist()
                        pred = [np.argmax(i) for i in pred]
                        pred = max(pred, key=pred.count)
                    results[task] = idx2label[pred]
                    final_results = {"results": results}
                    scene_results[f"scene{idx+1}"] = {'start_frame': start_frame, 'end_frame': end_frame, 'labels': final_results}
            return scene_results

    def inference_by_data(self, module_results_url):
        result_urls = json.loads(requests.get(module_results_url).content)
        results = {}

        for i, module_name in enumerate(self.module_names):
            result_url = result_urls[module_name]
            result = json.loads(requests.get(result_url).content)
            results[module_name] = result

        result = self.scene_inference(results)

        return result