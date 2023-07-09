import cv2
import numpy as np
import IMP
import requests
import uuid
import time
import json
import os
from PIL import ImageFont, ImageDraw, Image
import IMP
import re
import base64
import io
import pandas as pd
import Levenshtein
import random

# from clova_matching import scaling_matching
fontpath = "fonts/gulim.ttc"
# image_file = IMP.Tool.Image_Load("D:/test_image/img2.jpg",1)
font = ImageFont.truetype(fontpath, 21)

api_url = 'https://ha0mr4nm76.apigw.ntruss.com/custom/v1/23542/3ca2be5f04db2ff7d673eda501e04f496dcc82859b58f74c48f8ca4c797601a8/general'
secret_key = 'RnJxdkluR2xjTnlOZ3NyWnFjR3JxTkRWbkhTZ0RKSGQ='

# # image_file = 'input/picture1.png'
# file_name = '중간_작은거_Ref2.jpg'
# file_path ="D:/yolo/test_result_crop/"
# image_file =  file_path+ file_name
save_folder = 'D:/yolo/ocr_result/'

# IMP.Tool.create_folder(save_folder)

def array_to_binary(image_array):
    image_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    image_pil  = Image.fromarray(image_rgb)
    byte_io = io.BytesIO()
    image_pil .save(byte_io, format='JPEG')
    binary_data = byte_io.getvalue()
    return binary_data

def run(image_file) :
    # start = time.time()
    # img = array_to_binary(IMP.Tool.Image_Load(image_file,1))
    # img = array_to_binary(cv2.imread(image_file))
    img = IMP.Tool.Image_Load(image_file,1)
    request_json = {
        'images': [
            {
                'format': 'jpg',
                'name': 'demo'
                # 'data' : base64.b64encode(img).decode()
            }
        ],
        'requestId': str(uuid.uuid4()),
        'version': 'V2',
        # 'timestamp': 0
        'timestamp': int(round(time.time() * 1000))
    }

    payload = {'message': json.dumps(request_json).encode('UTF-8')}
    files = [('file', open(image_file,'rb'))]
    headers = {
    'X-OCR-SECRET': secret_key,
    # 'Content-Type': 'application/x-www-form-urlencoded'
    }

    # response = requests.request("POST", api_url, headers=headers, data = payload)
    # print(response.text)
    response = requests.post(api_url, headers=headers, files = files,data = payload)
  
    res = json.loads(response.text.encode('utf8'))
    
    def extract_numbers(text):
        numbers = re.findall(r'\d+', text)
        return numbers[0]
    
    def number_check(text) :
        if any(c.isdigit() for c in text)  :
            return 1
    
    def number_len(text) :
        number = [c for c in text if c.isdigit()]
        return len(number)
    
    def extract_english(text):
        english_only = re.sub('[^a-zA-Z]', ' ', text)
        english_only = english_only.strip()
        return english_only
    
    def korean_draw(img,color,bbox,text) :
        img_pil = Image.fromarray(img)
        draw = ImageDraw.Draw(img_pil)
        draw.text((bbox[0]['x'], bbox[0]['y']-25), text, font=font, fill=(color))
        return np.array(img_pil)
    
    def has_korean(text):
        korean = re.compile('[ㄱ-ㅣ가-힣]+')
        result = korean.search(text)
        if result:
            return 1
        else:
            return 0
    
    def check_zero(value):
        if value is None or value < 0:
            return 0
        else:
            return value
    
    def calculate_similarity(sentence1, sentence2):
        try :
            idx = sentence2.index(sentence1[:3])
            sentence2 = sentence2[idx:]
        except :
            pass
        # 문장의 유사도 계산
        similarity_ratio = Levenshtein.ratio(sentence1, sentence2)

        # 첫 번째 문장이 두 번째 문장에 포함되는지 확인
        is_contained = sentence1 in sentence2

        return similarity_ratio

    def remove_whitespace(text):
        try :
            if '.' in text :
                text = text.replace('.','')
            if '(냉장)' in text :
                text = text.replace('(냉장)','')
            text = text.replace(' ', '')
        except :
            pass
        return text

    ######## 표기 사항 오류 탐색 알고리즘#########
    df = pd.read_csv('d:/yolo/ref.csv',encoding='cp949')
    df = df.applymap(remove_whitespace)
    name = list(df['품목명'])
    target_count = list(df['계획'])
    order = list(df['주문처'])
    slaughter = list(df['도축장'])
    nums = list(df['묶음번호'])
    day = list(df['소비기한'])
    farm = list(df['농장명'])
    place = list(df['소재지'])
    c_num = list(df['인증번호'])

    def text_preprocess(string,mode) :
        lst = []
        scores = []
        indexs = []
        for idx,st in enumerate(name) :
            if mode == 0 and '바른농장' in st :
                continue
            score = calculate_similarity(string,st)
            if  score > 0.7 :
                scores.append(score)
                lst.append(st)
                indexs.append(idx)
        try : 
            return lst[scores.index(max(scores))],indexs[scores.index(max(scores))]
        except :
            return string

    
    
    ref_txt = ['내용량','기한','번호','용도','농장명','소재지','도축장','보관방법','부위','농장주','생산자','이력','제조원']
    # ref_txt_num = ['내용량','기한','번호']
    ref_txt_use = ['목심','다리','구이','갈비','찜','불고기','다짐육','롤링','안심','장조림','삼겹살','보쌈','사태','카레','잡채','찌개','등심','돈까스','항정살','대패']
    trigger = 0
    strings = []
    bboxs = []
    ref_bbox = []
    strings_ref = []
    extra_text1 = ''
    extra_text2 = ''
    extra_bbox1 = 0
    extra_bbox2 = 0
    berfore_text= ''
    trigger2 = 0
    
    ## 표기 오류
    true_name = ''
    true_string_num = ''
    true_day = ''
    true_nums = ''
    true_slaughter = ''
    true_farm = ''
    mode= 0
    fcount = 0
    f = open('D:/yolo/autolabel.txt','a')
    save_folder2 = 'D:/yolo/autolabel_image/'
    IMP.Tool.create_folder(save_folder2)
    if len(os.listdir(save_folder2)) > 1 : 
        # count = sorted(os.listdir(save_folder), key=int)[-1].split('.')[0].split('_')[-1]
        with open("D:/yolo/autolabel.txt", "r",encoding='cp949') as f:
            lines = f.readlines()
            last_str = lines[-1].split('_')[-1].split('.')[0]
            fcount = int(last_str)
            f.close()
    else :
        fcount = 0
    
    for idx,txt in enumerate(res['images'][0]['fields']) :
        text = res['images'][0]['fields'][idx]['inferText']
        confidence = res['images'][0]['fields'][idx]['inferConfidence']
        bbox = res['images'][0]['fields'][idx]['boundingPoly']['vertices']
        if confidence > 0.98 :
            with open("D:/yolo/autolabel.txt", "a",encoding='cp949') as f:
                fcount+= 1
                clip_image = img[int(bbox[0]['y']):int(bbox[2]['y']),int(bbox[0]['x']):int(bbox[2]['x'])]
                cv2.imwrite(save_folder2+f'/images_{fcount}.jpg',clip_image)
                f.write(save_folder2 + f'/images_{fcount}.jpg\t{text}\n')
        print(f"score : {confidence} / text : {text}")
        
        
        if len([lt for lt in ref_txt if lt in text]) == 1  :
            if len(text) < 6 :
                box_size = (abs(int(bbox[0]['x']) - int(bbox[2]['x'])))
            th = (abs(int(bbox[0]['y']) - int(bbox[2]['y'])/2)) 
            trigger = 1
            
         ## 2줄 텍스트에 대한 예외처리 클러스터링
        
        
        if trigger == 1 :
            string_trigger = [lt for lt in ref_txt if lt in text]
            text = text.replace(' ','')
            text = text.replace(':','')
            text = text.replace('.','')
            if '부위' in text or '용도' in text or '농장명' in text or '농장주' in text :
                if ('부위' in text or '용도' in text) and len(text) < 5:
                    extra_text1 += text
                    if '용도' in text or '부위' in text :
                        extra_bbox1 = bbox
                elif ('부위' in text or '용도' in text) and len(text) > 5:
                    extra_text1 = text
                    if '용도' in text or '부위' in text :
                        extra_bbox1 = bbox
                if ('농장명' in text or '농장주' in text) and len(text) < 5:
                    extra_text2 += text
                    if '농장주' in text or '농장명' in text :
                        extra_bbox2 = bbox
                elif ('농장명' in text or '농장주' in text) and len(text) > 5:
                    extra_text2 = text
                    if '농장주' in  text or '농장명' in text:
                        extra_bbox2 = bbox
            if '제조원' in text and idx < 10 :
                trigger2 += 1
            if berfore_text == '제조원' and '주' in text : ## 제조원의 경우 무조건 M이 아니라 다른 텍스트들도 오기 때문에 Trigger 방식으로 2가지 mode로 구현
                trigger2 += 1
            else :
                if len(string_trigger) > 0 :
                    ref_bbox.append(bbox)
                    strings_ref.append(text)
                    berfore_text = text
                    
                else :
                    strings.append(text)
                    bboxs.append(bbox)
    
    strings_ref.append(extra_text1)
    ref_bbox.append(extra_bbox1)
    strings_ref.append(extra_text2)
    ref_bbox.append(extra_bbox2)
    for idx, ref in enumerate(strings_ref) :
        count = 0
        ref = ref.replace(' ','')
        ref = ref.replace(':','')
        id = []
            
        if '내용량' in ref : 
            id = [[idx2,string] for idx2,string in enumerate(strings) if 'g' in string or 'G' in string]
            id2 = [[idx2,string] for idx2,string in enumerate(strings) if '300' == string or '500' == string or '1' == string]
            if len(id2) > 0 :
                true_string_num = id2[0][1] + id[0][1]
            else :
                true_string_num = id[0][1]
            if len(id) > 0 :
                if abs((bboxs[id[0][0]])[0]['y']-ref_bbox[idx][0]['y']) > th :
                    cv2.rectangle(img,(int(ref_bbox[idx][0]['x']),int(ref_bbox[idx][0]['y'])),(int(ref_bbox[idx][2]['x']),int(ref_bbox[idx][2]['y'])),(255,0,0),3)
            else :
                    cv2.rectangle(img,(int(ref_bbox[idx][0]['x']),int(ref_bbox[idx][0]['y'])),(int(ref_bbox[idx][2]['x']),int(ref_bbox[idx][2]['y'])),(255,0,0),3)

        if '기한' in ref : 
            if number_check(ref) :
                count = number_len(ref)
            id = [[idx2,string] for idx2,string in enumerate(strings) if number_len(string) == 8-count]
            true_day = id[0][1]
            if len(id) > 0 :
                if abs((bboxs[id[0][0]])[0]['y']-ref_bbox[idx][0]['y']) > th :
                    cv2.rectangle(img,(int(ref_bbox[idx][0]['x']),int(ref_bbox[idx][0]['y'])),(int(ref_bbox[idx][2]['x']),int(ref_bbox[idx][2]['y'])),(255,0,0),3)
            else :
                    cv2.rectangle(img,(int(ref_bbox[idx][0]['x']),int(ref_bbox[idx][0]['y'])),(int(ref_bbox[idx][2]['x']),int(ref_bbox[idx][2]['y'])),(255,0,0),3)
        
        if '인증' in ref and '생산자' not in ref :
            if number_check(ref) :
                count = number_len(ref)
            id = [[idx2,string] for idx2,string in enumerate(strings) if '동물복지' in string and number_len(string) >= check_zero(6-count)]
            if len(id) > 0 :
                if abs((bboxs[id[0][0]])[0]['y']-ref_bbox[idx][0]['y']) > th :
                    cv2.rectangle(img,(int(ref_bbox[idx][0]['x']),int(ref_bbox[idx][0]['y'])),(int(ref_bbox[idx][2]['x']),int(ref_bbox[idx][2]['y'])),(255,0,0),3)
            else :
                    cv2.rectangle(img,(int(ref_bbox[idx][0]['x']),int(ref_bbox[idx][0]['y'])),(int(ref_bbox[idx][2]['x']),int(ref_bbox[idx][2]['y'])),(255,0,0),3)    

        if '이력' in ref :
            if abs(ref_bbox[idx][0]['x']-ref_bbox[idx][2]['x']) > box_size * 4 :
                pass
            else :
                if number_check(ref) :
                    count = number_len(ref)
                if count > 12 :
                    continue
                id = [[idx2,string] for idx2,string in enumerate(strings) if number_len(string) > 12-count]
                true_nums = id[0][1]
                if len(id) > 0 :
                    if abs((bboxs[id[0][0]])[0]['y']-ref_bbox[idx][0]['y']) > th :
                        cv2.rectangle(img,(int(ref_bbox[idx][0]['x']),int(ref_bbox[idx][0]['y'])),(int(ref_bbox[idx][2]['x']),int(ref_bbox[idx][2]['y'])),(255,0,0),3)
                else :
                        cv2.rectangle(img,(int(ref_bbox[idx][0]['x']),int(ref_bbox[idx][0]['y'])),(int(ref_bbox[idx][2]['x']),int(ref_bbox[idx][2]['y'])),(255,0,0),3)
            
        if '소재지' in ref :
            ## 1번째 예외사항
            if len(ref) > 3 :
                if '도' in ref or '시' in ref :
                    pass
            else :
                id = [[idx2,string] for idx2,string in enumerate(strings) if ('도' in string or '시' in string) and len(string) < 8]
                for i in range(len(id)) :
                    if i < len(id) -1 :
                       if abs(id[i][0] - id[i+1][0]) > 5 :
                            del id[i]
                if len(id) > 0 :
                    if abs((bboxs[id[0][0]])[0]['y']-ref_bbox[idx][0]['y']) > th :
                        cv2.rectangle(img,(int(ref_bbox[idx][0]['x']),int(ref_bbox[idx][0]['y'])),(int(ref_bbox[idx][2]['x']),int(ref_bbox[idx][2]['y'])),(255,0,0),3)
                else :
                        cv2.rectangle(img,(int(ref_bbox[idx][0]['x']),int(ref_bbox[idx][0]['y'])),(int(ref_bbox[idx][2]['x']),int(ref_bbox[idx][2]['y'])),(255,0,0),3)    
        
        if '농장' in ref :
            ## 1번째 예외사항
            id = [[idx2,string] for idx2,string in enumerate(strings) if ('농장' in string or '동장' in string) and len(string) > 3]
            if len(id) > 0 :
                true_farm = id[0][1]
                mode= 1
                if abs((bboxs[id[0][0]])[0]['y']-ref_bbox[idx][0]['y']) > th :
                    cv2.rectangle(img,(int(ref_bbox[idx][0]['x']),int(ref_bbox[idx][0]['y'])),(int(ref_bbox[idx][2]['x']),int(ref_bbox[idx][2]['y'])),(255,0,0),3)
            else :
                    cv2.rectangle(img,(int(ref_bbox[idx][0]['x']),int(ref_bbox[idx][0]['y'])),(int(ref_bbox[idx][2]['x']),int(ref_bbox[idx][2]['y'])),(255,0,0),3)
                    
        if '용도' in ref :
            ## 1번째 예외사항
            for ref_use in ref_txt_use :
                if len([[idx2,string] for idx2,string in enumerate(ref) if ref_use in ref]) > 0 :
                    strings.append(ref[ref.index('용도')+2:])
                    bboxs.append(ref_bbox[idx])
                    ref = ref[:ref.index('용도')+2]
                if len([[idx2,string] for idx2,string in enumerate(strings) if ref_use in string]) > 0 :
                    id = [[idx2,string] for idx2,string in enumerate(strings) if ref_use in string]
            true_name = id[0][1]
            if len(id) > 0 :
                if abs((bboxs[id[0][0]])[0]['y']-ref_bbox[idx][0]['y']) > th :
                    cv2.rectangle(img,(int(ref_bbox[idx][0]['x']),int(ref_bbox[idx][0]['y'])),(int(ref_bbox[idx][2]['x']),int(ref_bbox[idx][2]['y'])),(255,0,0),3)
            else :
                    cv2.rectangle(img,(int(ref_bbox[idx][0]['x']),int(ref_bbox[idx][0]['y'])),(int(ref_bbox[idx][2]['x']),int(ref_bbox[idx][2]['y'])),(255,0,0),3)                  
        
        if '도축장' in ref :
            ## 1번째 예외사항
            if len(ref) > 3 :
                pass
            else :
                id = [[idx2,string] for idx2,string in enumerate(strings) if '도드람' in string or '엘피' in string or '축협' in string or '애프엠' in string or '에프엠' in string or '공판장' in string or '피씨' in string]
                true_slaughter = id[0][1]
                if len(id) > 0 :
                    if abs((bboxs[id[0][0]])[0]['y']-ref_bbox[idx][0]['y']) > th :
                        cv2.rectangle(img,(int(ref_bbox[idx][0]['x']),int(ref_bbox[idx][0]['y'])),(int(ref_bbox[idx][2]['x']),int(ref_bbox[idx][2]['y'])),(255,0,0),3)
                else :
                        cv2.rectangle(img,(int(ref_bbox[idx][0]['x']),int(ref_bbox[idx][0]['y'])),(int(ref_bbox[idx][2]['x']),int(ref_bbox[idx][2]['y'])),(255,0,0),3)
        
        if '제조원' in ref and len(ref) == 3:
            ## 1번째 예외사항
            if trigger2 == 0 :
                id = [[idx2,string] for idx2,string in enumerate(strings) if 'M' in string]
                if len(id) > 0 :
                    if abs((bboxs[id[0][0]])[0]['y']-ref_bbox[idx][0]['y']) > th :
                        cv2.rectangle(img,(int(ref_bbox[idx][0]['x']),int(ref_bbox[idx][0]['y'])),(int(ref_bbox[idx][2]['x']),int(ref_bbox[idx][2]['y'])),(255,0,0),3)
                else :
                        cv2.rectangle(img,(int(ref_bbox[idx][0]['x']),int(ref_bbox[idx][0]['y'])),(int(ref_bbox[idx][2]['x']),int(ref_bbox[idx][2]['y'])),(255,0,0),3)
        
        if '생산자' in ref :
            if number_check(ref) :
                count = number_len(ref)
            id = [[idx2,string] for idx2,string in enumerate(strings) if number_len(string) == 8-count]
            if len(id) > 0 :
                if abs((bboxs[id[0][0]])[0]['y']-ref_bbox[idx][0]['y']) > th :
                    cv2.rectangle(img,(int(ref_bbox[idx][0]['x']),int(ref_bbox[idx][0]['y'])),(int(ref_bbox[idx][2]['x']),int(ref_bbox[idx][2]['y'])),(255,0,0),3)
            else :
                    cv2.rectangle(img,(int(ref_bbox[idx][0]['x']),int(ref_bbox[idx][0]['y'])),(int(ref_bbox[idx][2]['x']),int(ref_bbox[idx][2]['y'])),(255,0,0),3)
       
    
    # IMP.Tool.Show(img)
    cv2.imwrite(save_folder+image_file.split('/')[-1],img)
    ## 표기 에러 검출
    # true_name = true_name + true_string_num
    # if len(true_farm) > 0 :
    #     true_name = '바른농장'+true_name
    # true_name = true_name.replace('[','')
    # true_name = true_name.replace(']','')
    # true_name,index = text_preprocess(true_name,mode)
    # print("제품 명 : ",true_name)
    # if order[index] == '푸드머스' :
    #     if mode == 0 :
    #         if slaughter[index] != true_slaughter :
    #             print("도축장 표기 오류")
    #         print("실제 값 : ", slaughter[index])
    #         print("표기 값 : ", true_slaughter)
    #     elif mode == 1 :
    #         if farm[index] != true_farm :
    #             print("농장 표기 오류")
    #         print("실제 값 : ", slaughter[index])
    #         print("표기 값 : ", true_slaughter)
    # else :
    #     if nums[index] != true_nums :
    #         print("이력번호 표기 오류")
    #     print("실제 값 : ", slaughter[index])
    #     print("표기 값 : ", true_slaughter)
    # if day[index] != extract_numbers(true_day) :
    #     print("소비기한 표기 오류")
    # print("실제 값 : ", day[index])
    # print("표기 값 : ", extract_numbers(true_day))
        


run("d:/yolo/test_result_crop/2023_07_05_10_23_56_crop.jpg")
