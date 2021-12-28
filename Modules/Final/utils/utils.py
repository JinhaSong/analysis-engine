import os
import glob

def food_rename(word):
    dessert = ['기타/cheesecake', '기타/chocolate_cake', '기타/chocolate_mousse', '기타/club_sandwich', '기타/cup_cakes', '기타/donuts', '기타/pancakes', '기타/red_velvet_cake', '기타/macarons', '기타/tiramisu', '기타/waffles']
    steak = ['구이/baby_back_ribs', '구이/prime_rib', '구이/steak']
    chicken = ['구이/chicken_wings', '기타/양념치킨', '기타/후라이드치킨']
    japanese = ['기타/takoyaki', '면/ramen', '밥/sushi', ]
    k_noodle = ['면/라면', '면/막국수', '면/물냉면', '면/비빔냉면', '면/수제비', '면/열무국수', '면/잔치국수', '면/쫄면', '면/칼국수', '면/콩국수']
    c_noodle = ['면/짜장면', '면/짬뽕']
    boonsik = ['볶음/떡볶이', '볶음/라볶이', '적/떡꼬치']
    western = ['기타/eggs_benedict', '기타/french_toast', '기타/grilled_cheese_sandwich', '기타/hamburger', '기타/hot_dog', '기타/tacos', '면/spaghetti_bolognese', '면/spaghetti_carbonara', '밥/pizza', '밥/risotto', '튀김/french_fries', '기타/피자']
    k_rice = ['밥/fried_rice', '밥/김밥', '밥/김치볶음밥', '밥/누룽지', '밥/비빔밥', '밥/새우볶음밥', '밥/알밥', '밥/유부초밥', '밥/잡곡밥', '밥/주먹밥']
    k_grill = ['구이/갈비구이', '구이/갈치구이', '구이/고등어구이', '구이/곱창구이', '구이/닭갈비', '구이/더덕구이', '구이/떡갈비', '구이/불고기', '구이/삼겹살', '구이/장어구이', '구이/조개구이', '구이/조기구이', '구이/황태구이', '구이/훈제오리']
    k_fry = ['튀김/고추튀김', '튀김/새우튀김', '튀김/오징어튀김']
    etc = ['기타/과메기', '기타/젓갈', '기타/콩자반', '기타/편육']

    asitis = dict()
    asitis['soup'] = ['국/miso_soup', '국/계란국', '국/떡국_만두국', '국/무국', '국/미역국', '국/북엇국', '국/시래기국', '국/육개장', '국/콩나물국', '찌개/김치찌개', '찌개/닭계장', '찌개/동태찌개', '찌개/된장찌개', '찌개/순두부찌개', '탕/갈비탕', '탕/감자탕', '탕/곰탕_설렁탕', '탕/매운탕', '탕/삼계탕', '탕/추어탕'] # 그대로 (세개 합침)
    asitis['kimchi'] = ['김치/갓김치', '김치/깍두기', '김치/나박김치', '김치/무생채', '김치/배추김치', '김치/백김치', '김치/부추김치', '김치/열무김치', '김치/오이소박이', '김치/총각김치', '김치/파김치'] # 그대로
    asitis['herbs'] = ['나물/가지볶음', '나물/고사리나물', '나물/미역줄기볶음', '나물/숙주나물', '나물/시금치나물', '나물/애호박볶음'] # 그대로
    asitis['rice_cake'] = ['떡/경단', '떡/꿀떡', '떡/송편'] # 그대로
    asitis['dumpling'] = ['만두/dumplings', '만두/만두'] # 그대로
    asitis['moochim'] = ['무침/greek_salad', '무침/꽈리고추무침', '무침/도라지무침', '무침/도토리묵', '무침/잡채', '무침/콩나물무침', '무침/홍어무침', '무침/회무침'] # 그대로
    asitis['jjim'] = ['찜/갈비찜', '찜/계란찜', '찜/김치찜', '찜/꼬막찜', '찜/닭볶음탕', '찜/수육', '찜/순대', '찜/족발', '찜/찜닭', '찜/해물찜'] # 그대로
    asitis['k_sweets'] = ['한과/약과', '한과/약식', '한과/한과'] # 그대로
    asitis['seafood'] = ['해물/멍게', '해물/산낙지'] # 그대로
    asitis['sashimi'] = ['회/sashimi', '회/물회', '회/육회'] # 그대로
    asitis['jorim'] = ['조림/갈치조림', '조림/감자조림', '조림/고등어조림', '조림/꽁치조림', '조림/두부조림', '조림/땅콩조림', '조림/메추리알장조림', '조림/연근조림', '조림/우엉조림', '조림/장조림', '조림/코다리조림'] # 그대로
    asitis['jeon'] =  ['전/감자전', '전/계란말이', '전/계란후라이', '전/김치전', '전/동그랑땡', '전/생선전', '전/파전', '전/호박전'] # 그대로
    asitis['ssam'] = ['쌈/보쌈'] # 그대로
    asitis['jang'] = ['장/간장게장', '장/양념게장'] # 그대로
    asitis['jangajji'] = ['장아찌/깻잎장아찌'] # 그대로
    asitis['jeongol'] = ['전골/곱창전골'] # 그대로
    asitis['stir_fry'] = ['볶음/감자채볶음', '볶음/건새우볶음', '볶음/고추장진미채볶음', '볶음/두부김치', '볶음/멸치볶음', '볶음/소세지볶음', '볶음/어묵볶음', '볶음/오징어채볶음', '볶음/제육볶음', '볶음/주꾸미볶음']  # 그대로
    asitis['drink'] = ['음청류/수정과', '음청류/식혜'] # 그대로
    asitis['jook'] = ['죽/전복죽', '죽/호박죽'] # 그대로

    asitis_trans = dict((vv.split('/')[0], k) for k,v in asitis.items() for vv in v)
    
    first, second = word.split('/')
    if first in asitis_trans.keys():
        return asitis_trans[first]
    elif word in dessert:
        return 'dessert'
    elif word in steak:
        return 'steak'
    elif word in chicken:
        return 'chicken'
    elif word in japanese:
        return 'japanese'
    elif word in k_noodle:
        return 'k_noodle'
    elif word in c_noodle:
        return 'c_noodle'
    elif word in boonsik:
        return 'boonsik'
    elif word in western:
        return 'western'
    elif word in k_rice:
        return 'k_rice'
    elif word in k_grill:
        return 'k_grill'
    elif word in k_fry:
        return 'k_fry'
    elif word in etc:
        return 'etc_food'
    else:
        return False
    
def find_food(food_data, method='frame', scene_info=None):
    foods = []
    if scene_info != None:
        start_frame_idx = scene_info['start_frame'] if scene_info['start_frame'] != 0 else 30
        start_idx = int((start_frame_idx/30)-1)
        end_idx = int(scene_info['end_frame']/30)
    food_info = food_data['module_result']['frame_results'] if method == 'frame' else food_data['module_result']['frame_results'][start_idx:end_idx]
    for i in food_info:
        food = dict()
        if i['frame_result'] == None:
            food = []
        else:
            for j in i['frame_result']:
                word = food_rename(j['label'][0]['description']) + '_food'
                score = j['label'][0]['score']/100
                food[word] = max(food.get(word, 0.0), score)
            food = sorted(list(food.items()), key=lambda x: x[1], reverse=True)[:5]
        if len(food) != 5:
            while len(food) < 5:
                food.append(('padding_food', 0.0))
        foods.append(food)
    return foods

def find_place(place_data, method='frame', scene_info=None):
    places = []
    if scene_info != None:
        start_frame_idx = scene_info['start_frame'] if scene_info['start_frame'] != 0 else 30
        start_idx = int((start_frame_idx/30)-1)
        end_idx = int(scene_info['end_frame']/30)
    place_info = place_data['module_result']['frame_results'] if method == 'frame' else place_data['module_result']['frame_results'][start_idx:end_idx]
    for i in place_info:
        place = dict()
        for j in i['frame_result']:
            word = j['label']['description'].lower() + '_place'
            place[word] = j['label']['score']/100
        place = sorted(list(place.items()), key=lambda x: x[1], reverse=True)
        places.append(place)
    return places

def find_object(object_data, method='frame', scene_info=None):
    objects = []
    if scene_info != None:
        start_frame_idx = scene_info['start_frame'] if scene_info['start_frame'] != 0 else 30
        start_idx = int((start_frame_idx/30)-1)
        end_idx = int(scene_info['end_frame']/30)
    object_info = object_data['module_result']['frame_results'] if method == 'frame' else object_data['module_result']['frame_results'][start_idx:end_idx]
    for i in object_info:
        obj = dict()
        for j in i['frame_result']:
            word = j['label'][0]['description'].lower() + '_obj'
            score = j['label'][0]['score']
            obj[word] = max(obj.get(word, 0.0), score)
        obj = sorted(list(obj.items()), key=lambda x: x[1], reverse=True)[:5]
        if len(obj) != 5:
            while len(obj) < 5:
                obj.append(('padding_object', 0.0))
        objects.append(obj)
    return objects

def find_event_audio(audio_data, other_data, method='frame', scene_info=None):
    audios = []
    audio, word10, score10 = dict(), [], []
    if scene_info != None:
        start_frame_idx = scene_info['start_frame'] if scene_info['start_frame'] != 0 else 30
        start_idx = int((start_frame_idx/30)-1)
        end_idx = int(scene_info['end_frame']/30)
        if scene_info['start_frame'] == 0:
            start_idx, end_idx = scene_info['start_frame'], scene_info['end_frame']*10
        else:
            start_idx, end_idx = scene_info['start_frame']*10, scene_info['end_frame']*10
    other_data_info = other_data['module_result']['frame_results'] if method == 'frame' else other_data['module_result']['frame_results'][start_idx:end_idx]
    num_frames = len(other_data_info)
    audioevent_info = audio_data['module_result']['audio_results'] if method == 'frame' else audio_data['module_result']['audio_results'][start_idx:end_idx]
    for idx, i in enumerate(audioevent_info):
        word = i['audio_result'][0]['label']['description'].lower() + '_audio'
        score = float(i['audio_result'][0]['label']['score'])/10
        word10.append(word)
        score10.append(score)
        audio[word] = max(audio.get(word, 0.0), score)
        if (idx+1)%10 == 0 or (idx+1)==len(audioevent_info):
            top1 = max(word10, key=word10.count)
            audios.append([(top1, audio[top1])])
            audio, word10, score10 = dict(), [], []
        if len(audios) == num_frames: break
    return audios

def find_scene_audio(audio_data, other_data, method='frame', scene_info=None):
    audio_scene = []
    if scene_info != None:
        start_frame_idx = scene_info['start_frame'] if scene_info['start_frame'] != 0 else 30
        start_idx = int((start_frame_idx/30)-1)
        end_idx = int(scene_info['end_frame']/30)
        if scene_info['start_frame'] == 0:
            start_idx, end_idx = scene_info['start_frame'], scene_info['end_frame']*10
        else:
            start_idx, end_idx = scene_info['start_frame']*10, scene_info['end_frame']*10
    other_data_info = other_data['module_result']['frame_results'] if method == 'frame' else other_data['module_result']['frame_results'][start_idx:end_idx]
    audioscene_info = audio_data['module_result']['audio_results'] if method == 'frame' else audio_data['module_result']['audio_results'][start_idx:end_idx]
    for i in audioscene_info:
        word = i['audio_result'][0]['label']['description'].lower() +'_audio(sc)'
        score = float(i['audio_result'][0]['label']['score'])/8
        audio_scene.append([(word, score)])
    for _ in range(len(other_data_info)-len(audioscene_info)):
        audio_scene.append([('unknown_audio(sc)', score)])
    return audio_scene

def find_person(person_data, method='frame', scene_info=None):
    result = []
    if scene_info != None:
        start_frame_idx = scene_info['start_frame'] if scene_info['start_frame'] != 0 else 30
        start_idx = int((start_frame_idx/30)-1)
        end_idx = int(scene_info['end_frame']/30)
    person_info = person_data['module_result']['frame_results'] if method == 'frame' else person_data['module_result']['frame_results'][start_idx:end_idx]
    for i in person_info:
        person = dict()
        for j in i['frame_result']:
            if j['label'][0]['description'] == 'Human_face':
                continue
            person[j['label'][0]['description']] = j['label'][0]['score']
        person = sorted(list(person.items()), key=lambda x: x[1], reverse=True)
        if len(person) == 0:
            person += [('unknown', 0.0)]
        result.append(person)
    return result

def find_scenetext(scenetext_data, scenetext_word, method='frame', scene_info=None):
    scenels = []
    if scene_info != None:
        start_frame_idx = scene_info['start_frame'] if scene_info['start_frame'] != 0 else 30
        start_idx = int((start_frame_idx/30)-1)
        end_idx = int(scene_info['end_frame']/30)
    scenetext_info = scenetext_data['module_result']['frame_results'] if method == 'frame' else scenetext_data['module_result']['frame_results'][start_idx:end_idx]
    for i in scenetext_info:
        frame_ls = []
        for j in i['frame_result']:
            if j['label'][0]['description'] in scenetext_word:
                frame_ls.append((j['label'][0]['description'], j['label'][0]['score']))
        scenels.append(frame_ls)
    return scenels

def find_url(url_data, url_words):
    whole_urltext = {}
    if len(url_data['module_result']['text_result'][0]['label']) == 0:
        frame_ls = []
    else:
        for i in url_data['module_result']['text_result'][0]['label']:
            frame_ls = []
            if i['description'] in url_words:
                frame_ls.append((i['description'], i['score']))
    return frame_ls

def find_relation(relation_data, method='frame', scene_info=None):    
    obj_relations = []
    if scene_info != None:
        start_frame_idx = scene_info['start_frame'] if scene_info['start_frame'] != 0 else 30
        start_idx = int((start_frame_idx/30)-1)
        end_idx = int(scene_info['end_frame']/30)
    relation_info = relation_data['module_result']['frame_results'] if method == 'frame' else relation_data['module_result']['frame_results'][start_idx:end_idx]
    for i in relation_info:
        obj_relation = dict()
        for j in i['frame_result']:
            for ii in j['interaction']:
                obj_relation[ii['label'][0]['description']] = ii['label'][0]['score']/235.126078142783
        obj_relation = sorted(list(obj_relation.items()), key=lambda x: x[1], reverse=True)
        obj_relations.append(obj_relation)
    return obj_relations

def extract_word(lists):
    words = [j[0] for ls in lists for l in ls for j in l]
    return list(set(words))