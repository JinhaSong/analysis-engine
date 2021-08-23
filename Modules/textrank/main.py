import time
import os
import pickle
import re
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

    def inference_by_text(self, data, video_info):
        result = {'text_result': []}

        test_flag = 1
        keyword_test_list = []

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
        text_json_result_list = {
            "text": data, "model_name": "text_rank", "model_result": []
        }

        for t_k in keyword_test_list:
            compare_list = []
            score_list = []
            list_count = 0
            dic_flag = 0
            text_json_result = {"label": []}

            for word, rank in t_k['keyword']:
                word = word.split('/')
                word = word[0]
                if list_count < 5:
                    compare_list.append(word)
                    score_list.append(rank)
                    list_count += 1
                if word in self.userdic:
                    compare_list.pop()
                    compare_list.insert(0, word)
                    score_list.pop()
                    score_list.insert(0, rank)
                    dic_flag = 1
                if dic_flag == 0:
                    for ud in self.userdic:
                        if word in ud:
                            compare_list.pop()
                            compare_list.append(word)
                            score_list.pop()
                            score_list.append(rank)
                result_element = {'label': [{'description': str(compare_list[0]), 'score': score_list[0]}]}
                result['text_result'].append(result_element)

            for comp_i in range(len(compare_list)):
                json_result_element = {"description": str(compare_list[comp_i]), "score": score_list[comp_i]}
                text_json_result["label"].append(json_result_element)
            text_json_result_list["model_result"].append(text_json_result)

            total += 1
            if test_flag == 0 or test_flag == 2:
                answer_words = t_k['answer'].replace(" ", "")
                answer_words = answer_words.split(";")
                for word in compare_list:
                    if word in answer_words:
                        f1 += 1
                        break

        self.result = text_json_result_list

        return self.result