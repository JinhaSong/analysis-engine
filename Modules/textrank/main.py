import json
import time
import os
import pickle
import re

import ast
from konlpy.tag import Komoran
from Modules.textrank.utils.summarizer import KeywordSummarizer
from utils import Logging


class TextRank:
    model = None
    result = None
    path = os.path.dirname(os.path.abspath(__file__))

    def __init__(self):
        start_time = time.time()
        self.userdic = []
        self.userdicfile = open(os.path.join(self.path, "data/dic.user"), "r", encoding='utf-8')
        lines = self.userdicfile.readlines()
        for line in lines:
            line = line.split('\t')
            self.userdic.append(line[0])

        with open(os.path.join(self.path, 'data/tourkeyword.pkl'), 'rb') as f2:
            self.tourAPI_dict = pickle.load(f2)

        self.tourAPI = []
        self.komoran = Komoran(userdic=os.path.join(self.path, 'data/dic.user'))

        for dic in self.tourAPI_dict:
            dic = self.clean_special_characters(dic)
            dic_word = self.komoran.pos(dic)
            self.tourAPI += [w + '/' + pos for w, pos in dic_word if
                             ('SL' in pos or 'NN' in pos or 'XR' in pos or 'VA' in pos or 'VV' in pos)]
        self.tourAPI = set(self.tourAPI)
        self.tourAPI = list(self.tourAPI)

        self.keyword_extractor = KeywordSummarizer(
            tokenize=self.komoran_tokenize,
            window=-1,
            verbose=False,
            min_count=2,
            min_cooccurrence=1
        )
        end_time = time.time()

        print(Logging.i("Text Rank model is successfully loaded(time: {}sec)".format(end_time - start_time)))


    def clean_special_characters(self, readData):
        text = re.sub('[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》]', '', readData)
        return text

    def komoran_tokenize(self, sent):
        words = self.komoran.pos(sent, join=True)
        words = [w for w in words if ('/SL' in w or '/NN' in w or '/XR' in w or '/VA' in w or '/VV' in w)]
        return words

    def clean_without_hangul(self, text):
        hangul = re.compile('[^1234567890 \u3131-\u3163\uac00-\ud7a3]+')
        return hangul.sub('', text)

    def get_strings(self, base_model_result, base_model_name, key):
        strings = []

        if base_model_name == "scene_text_recognition":
            for result in base_model_result:
                single_results = result[key + "_result"]
                for single_result in single_results:
                    str_label = single_result["label"][0]["description"]
                    if len(str_label) > 0:
                        strings.append(str_label)
        elif base_model_name == "automatic_speech_recognition":
            for audio_result in base_model_result:
                str_label = audio_result[key + "_result"]
                if len(str_label) > 0:
                    strings.append(str_label)

        return strings


    def inference_by_text(self, data, video_info):
        try :
            base_model_result = ast.literal_eval(data[0])
            base_model_name = base_model_result["model_name"]

            if base_model_name == "scene_text_recognition":
                key = "frame"
            elif base_model_name == "automatic_speech_recognition" or base_model_name == "audio_scene_classification":
                key = "audio"
            else :
                key = "frame"

            str_data = self.get_strings(base_model_result[key + "_results"], base_model_name, key)
            data = str_data.copy()
        except :
            base_model_name = None
            str_data = data.copy()

        results = {
            "text": str_data,
            "model_name": "text_rank",
            "base_model_name": base_model_name,
            "analysis_time": 0,
            "text_result": []
        }

        start_time = time.time()
        test_flag = 1
        keyword_test_list = []

        if data[0] != None :
            keyword_test = {}
            for i in range(len(data)):
                data[i] = self.clean_special_characters(self.clean_without_hangul(data[i]))

            key_test = ""
            try:
                key_test = self.keyword_extractor.summarize(data, topk=10)
            except:
                pass

            keyword_test['keyword'] = key_test
            keyword_test_list.append(keyword_test)

            f1 = 0
            total = 1
            location_list = ['성수', '이태원', '을지로', '남산', '잠실', '강남', '신촌', '명동', '여의도', '영등포', '동대문', '홍대', '신도림', '종로']

            for t_k in keyword_test_list:
                compare_list = []
                score_list = []
                list_count = 0
                dic_flag = 0
                text_json_result = {"label": []}

                for word, rank in t_k['keyword']:
                    word = word.split('/')
                    word = word[0]
                    if list_count < 10:
                        compare_list.append(word)
                        score_list.append(rank)
                        list_count += 1
                    if word in self.userdic or word in self.tourAPI:
                        compare_list.pop()
                        compare_list.insert(0, word)
                        score_list.pop()
                        score_list.insert(0, rank)
                        dic_flag = 1
                    if dic_flag == 0:
                        for ud in self.userdic:
                            if word in ud:
                                compare_list.pop()
                                compare_list.insert(0,word)
                                score_list.pop()
                                score_list.insert(0,rank)
                    ###
            ##LOC Priority
            LOCATION_PR = 1
            for c_i, com in enumerate(compare_list):
                ### sub location
                change_flag = 0
                if com in ['뚝섬역', '뚝섬', '광나루', '성수동']:
                    change_flag = 1
                    com1 = '성수'
                elif com in ['이태원']:
                    change_flag = 1
                    com1 = '이태원'
                elif com in ['을지로6가']:
                    change_flag = 1
                    com1 = '을지로'
                elif com in ['남산']:
                    change_flag = 1
                    com1 = '남산'
                elif com in ['문정동', '문정점', '방이', '방이동', '석촌']:
                    change_flag = 1
                    com1 = '잠실'
                elif com in ['탄천', '코엑스', '논현동', '신사', 'C27 가로수길', 'C27', '강남코엑스', '강남역', '역삼', '양재천']:
                    change_flag = 1
                    com1 = '강남'
                elif com in ['이대', '이대점']:
                    change_flag = 1
                    com1 = '신촌'
                elif com in ['명동']:
                    change_flag = 1
                    com1 = '명동'
                elif com in ['여의도']:
                    change_flag = 1
                    com1 = '여의도'
                elif com in ['영등포 신길동 홍어거리', '신길동', '대림동']:
                    change_flag = 1
                    com1 = '영등포'
                elif com in ['동대문 문구완구거리']:
                    change_flag = 1
                    com1 = '동대문'
                elif com in ['홍대역', '연남동', '연남', '합정역', '망원역']:
                    change_flag = 1
                    com1 = '홍대'
                elif com in ['신도림']:
                    change_flag = 1
                    com1 = '신도림'
                elif com in ['낙원동', '낙원동 아구찜 거리', '종각역', '서촌', '인사동', '대학로', '익선동', '토속촌', '종로3가', '북촌', '청계천', '혜화동', '혜화', '세종대로', '이화동']:
                    change_flag = 1
                    com1 = '종로'
                ###sub loc
                if LOCATION_PR == 1:
                    if com in location_list or change_flag == 1:
                        compare_list.remove(compare_list[c_i])
                        compare_list.insert(0,com)
                        tmp = score_list[c_i]
                        score_list.remove(tmp)
                        score_list.insert(0,tmp)
                        if change_flag == 1:
                            compare_list.pop()
                            compare_list.insert(0,com1)
                            score_list.pop()
                            score_list.insert(0,score_list[c_i])
                        break


                for comp_i in range(len(compare_list)):
                    json_result_element = {"description": str(compare_list[comp_i]), "score": score_list[comp_i]}
                    text_json_result["label"].append(json_result_element)
                results["text_result"].append(text_json_result)

                total += 1
                if test_flag == 0 or test_flag == 2:
                    answer_words = t_k['answer'].replace(" ", "")
                    answer_words = answer_words.split(";")
                    for word in compare_list:
                        if word in answer_words:
                            f1 += 1
                            break
            end_time = time.time()
            results['analysis_time'] = end_time - start_time
            print(Logging.i("Processing time: {}".format(results['analysis_time'])))

        self.result = results

        return self.result
