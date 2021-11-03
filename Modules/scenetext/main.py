import shutil
import string
import argparse

import os
import time

import cv2
import copy
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F

from AnalysisEngine import settings
from Modules.dummy.main import Dummy
from Modules.scenetext.utils import CTCLabelConverter, AttnLabelConverter
from Modules.scenetext.dataset import RawDataset, AlignCollate
from Modules.scenetext.model import Model

from WebAnalyzer.utils.media import frames_to_timecode


from Modules.scenetext.CRAFT_pytorch import Craft_inference
from utils import Logging


class SceneText:
    model = None
    result = None
    path = os.path.dirname(os.path.abspath(__file__))
    character = '0123456789abcdefghijklmnopqrstuvwxyz가각간갇갈감갑값갓강갖같갚갛개객걀걔거걱건걷걸검겁것겉게겨격겪견결겹경곁계' \
                '고곡곤곧골곰곱곳공과관광괜괴굉교구국군굳굴굵굶굽궁권귀귓규균귤그극근글긁금급긋긍기긴길김깅깊까깍깎깐깔깜깝깡깥깨꺼꺾' \
                '껌껍껏껑께껴꼬꼭꼴꼼꼽꽂꽃꽉꽤꾸꾼꿀꿈뀌끄끈끊끌끓끔끗끝끼낌나낙낚난날낡남납낫낭낮낯낱낳내냄냇냉냐냥너넉넌널넓넘넣네' \
                '넥넷녀녁년념녕노녹논놀놈농높놓놔뇌뇨누눈눕뉘뉴늄느늑는늘늙능늦늬니닐님다닥닦단닫달닭닮담답닷당닿대댁댐댓더덕던덜덟덤' \
                '덥덧덩덮데델도독돈돌돕돗동돼되된두둑둘둠둡둥뒤뒷드득든듣들듬듭듯등디딩딪따딱딴딸땀땅때땜떠떡떤떨떻떼또똑뚜뚫뚱뛰뜨뜩' \
                '뜯뜰뜻띄라락란람랍랑랗래랜램랫략량러럭런럴럼럽럿렁렇레렉렌려력련렬렵령례로록론롬롭롯료루룩룹룻뤄류륙률륭르른름릇릎리' \
                '릭린림립릿링마막만많말맑맘맙맛망맞맡맣매맥맨맵맺머먹먼멀멈멋멍멎메멘멩며면멸명몇모목몬몰몸몹못몽묘무묵묶문묻물뭄뭇뭐' \
                '뭘뭣므미민믿밀밉밌및밑바박밖반받발밝밟밤밥방밭배백뱀뱃뱉버번벌범법벗베벤벨벼벽변별볍병볕보복볶본볼봄봇봉뵈뵙부북분불' \
                '붉붐붓붕붙뷰브븐블비빌빔빗빚빛빠빡빨빵빼뺏뺨뻐뻔뻗뼈뼉뽑뿌뿐쁘쁨사삭산살삶삼삿상새색샌생샤서석섞선설섬섭섯성세섹센셈' \
                '셋셔션소속손솔솜솟송솥쇄쇠쇼수숙순숟술숨숫숭숲쉬쉰쉽슈스슨슬슴습슷승시식신싣실싫심십싯싱싶싸싹싼쌀쌍쌓써썩썰썹쎄쏘쏟' \
                '쑤쓰쓴쓸씀씌씨씩씬씹씻아악안앉않알앓암압앗앙앞애액앨야약얀얄얇양얕얗얘어억언얹얻얼엄업없엇엉엊엌엎에엔엘여역연열엷염' \
                '엽엿영옆예옛오옥온올옮옳옷옹와완왕왜왠외왼요욕용우욱운울움웃웅워원월웨웬위윗유육율으윽은을음응의이익인일읽잃임입잇있' \
                '잊잎자작잔잖잘잠잡잣장잦재쟁쟤저적전절젊점접젓정젖제젠젯져조족존졸좀좁종좋좌죄주죽준줄줌줍중쥐즈즉즌즐즘증지직진질짐' \
                '집짓징짙짚짜짝짧째쨌쩌쩍쩐쩔쩜쪽쫓쭈쭉찌찍찢차착찬찮찰참찻창찾채책챔챙처척천철첩첫청체쳐초촉촌촛총촬최추축춘출춤춥춧' \
                '충취츠측츰층치칙친칠침칫칭카칸칼캄캐캠커컨컬컴컵컷케켓켜코콘콜콤콩쾌쿄쿠퀴크큰클큼키킬타탁탄탈탑탓탕태택탤터턱턴털텅' \
                '테텍텔템토톤톨톱통퇴투툴툼퉁튀튜트특튼튿틀틈티틱팀팅파팎판팔팝패팩팬퍼퍽페펜펴편펼평폐포폭폰표푸푹풀품풍퓨프플픔피픽' \
                '필핏핑하학한할함합항해핵핸햄햇행향허헌험헤헬혀현혈협형혜호혹혼홀홈홉홍화확환활황회획횟횡효후훈훌훔훨휘휴흉흐흑흔흘흙흡흥흩희흰히힘?!'

    def __init__(self):
        super().__init__()
        parser = argparse.ArgumentParser()
        self.opt = {
            'workers': 1,
            'batch_size': 16,
            'saved_model': os.path.join(self.path, 'weights/best_accuracy.pth'),
            'batch_max_length': 25,
            'imgH': 32,
            'imgW': 100,
            'rgb': True,
            'sensitive': True,
            'PAD': True,
            'Transformation': 'TPS',
            'FeatureExtraction': 'ResNet',
            'SequenceModeling': 'BiLSTM',
            'Prediction': 'Attn',
            'num_fiducial': 20,
            'input_channel': 1,
            'output_channel': 512,
            'hidden_size': 256
        }

        if self.opt["sensitive"]:
            self.opt["character"] = string.printable[:-6]
        self.opt["num_gpu"] = torch.cuda.device_count()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.recognition_confidence_threshold = 0.0

        cudnn.benchmark = True
        cudnn.deterministic = True

        self.detection_net, self.refine_net = Craft_inference.load_craft_weights()
        self.recognition_net, self.converter = self.load_reconize_weights()


    def detect_one_image(self, img, base_dir, current_frame=0) :
        bboxes, crop_images, detect_box_img = Craft_inference.craft_one_image(self.detection_net, self.refine_net, img, current_frame, base_dir)

        bboxes = sorted(bboxes,key = lambda x : (x[0][0],x[0][1]))
        return bboxes, crop_images, detect_box_img # coordinates / bboxCrop / original_image with bounding_box

    def recognize_one_image(self, model, base_dir, bboxes = None) :
        AlignCollate_demo = AlignCollate(imgH=self.opt['imgH'], imgW=self.opt['imgW'], keep_ratio_with_pad=self.opt['PAD'])
        demo_data = RawDataset(root=base_dir, opt=self.opt)  # use RawDataset

        demo_loader = torch.utils.data.DataLoader(
            demo_data, batch_size=self.opt['batch_size'],
            shuffle=False,
            num_workers=int(self.opt['workers']),
            collate_fn=AlignCollate_demo, pin_memory=False)
        # predict
        model.eval()
        results = []

        with torch.no_grad():
            iter_check = 0
            i = 0

            for image_tensors, image_path_list in demo_loader:
                iter_check = iter_check + 1
                batch_size = image_tensors.size(0)
                image = image_tensors.to(self.device)
                length_for_pred = torch.IntTensor([self.opt['batch_max_length']] * batch_size).to(self.device)
                text_for_pred = torch.LongTensor(batch_size, self.opt['batch_max_length'] + 1).fill_(0).to(self.device)

                if 'CTC' in self.opt['Prediction']:
                    preds = model(image, text_for_pred)

                    preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                    _, preds_index = preds.max(2)
                    preds_str = self.converter.decode(preds_index.data, preds_size.data)

                else:
                    preds = model(input=image, text=text_for_pred, is_train=False)
                    _, preds_index = preds.max(2)
                    preds_str = self.converter.decode(preds_index, length_for_pred)


                preds_prob = F.softmax(preds, dim=2)
                preds_max_prob, _ = preds_prob.max(dim=2)
                if len(bboxes) > 0:
                    for img_name, pred, pred_max_prob in zip(image_path_list, preds_str, preds_max_prob):
                        try :
                            if 'Attn' in self.opt['Prediction']:
                                pred_EOS = pred.find('[s]')
                                pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                                pred_max_prob = pred_max_prob[:pred_EOS]

                            confidence_score = pred_max_prob.cumprod(dim=0)[-1]

                            bbox = bboxes[i]
                            bbox_coordinate = bbox

                            confidence = confidence_score.detach().cpu().numpy()

                            left_top = bbox_coordinate[0]
                            left_top = list(map(int,left_top))
                            right_bottom = bbox_coordinate[2]
                            right_bottom = list(map(int,right_bottom))

                            x = left_top[0]
                            w = x + left_top[1]
                            y = right_bottom[0]
                            h = y + right_bottom[1]
                            score = float(confidence)

                            result = {
                                'label':[
                                    {
                                        'description': pred,
                                        'score': score
                                    }
                                ],
                                'position': {
                                    'x': x,
                                    'y': y,
                                    'w': w,
                                    'h': h
                                }
                            }
                            results.append(result)
                        except:
                            pass

        return results

    def load_reconize_weights(self) :
        if 'CTC' in self.opt['Prediction']:
            converter = CTCLabelConverter(self.character)
        else:
            converter = AttnLabelConverter(self.character)
        self.opt["num_class"] = len(converter.character)

        model = Model(self.opt)
        model = torch.nn.DataParallel(model).to(self.device)
        model.load_state_dict(torch.load(self.opt['saved_model'], map_location=self.device))

        return model, converter

    def inference_by_image(self, image_path):
        image = cv2.imread(image_path)

        result = {}
        base_dir = image_path.replace(".jpg", "")
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)

        bboxes, crop_images, detect_box_img = self.detect_one_image(image, base_dir, 0)
        results = self.recognize_one_image(self.recognition_net, base_dir, bboxes=bboxes)
        result["result"]  = results
        shutil.rmtree(base_dir)
        return result

    def inference_by_video(self, frame_path_list, infos):
        video_info = infos['video_info']
        frame_urls = infos['frame_urls']
        fps = video_info['extract_fps']
        print(Logging.i("Start inference by video"))
        results = {
            "model_name": "scene_text_recognition",
            "analysis_time": 0,
            "model_result": []
        }

        start_time = time.time()
        for idx, (frame_path, frame_url) in enumerate(zip(frame_path_list, frame_urls)):
            if idx % 10 == 0:
                print(Logging.i("Processing... (index: {}/{} / frame number: {} / path: {})".format(idx, len(frame_path_list), int((idx + 1) * fps), frame_path)))
            result = self.inference_by_image(frame_path)
            result["frame_url"] = settings.MEDIA_URL + frame_url[1:]
            result["frame_number"] = int((idx + 1) * fps)
            result["timestamp"] = frames_to_timecode((idx + 1) * fps, fps)
            results["model_result"].append(result)

        end_time = time.time()
        results['analysis_time'] = end_time - start_time
        print(Logging.i("Processing time: {}".format(results['analysis_time'])))

        self.result = results

        return self.result