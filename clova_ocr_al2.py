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
# from clova_matching import scaling_matching
fontpath = "fonts/gulim.ttc"
# image_file = IMP.Tool.Image_Load("D:/test_image/img2.jpg",1)
font = ImageFont.truetype(fontpath, 21)

api_url = 'https://s2cm7yuquo.apigw.ntruss.com/custom/v1/22188/fa24acd3ef4dc26539dd12962a443577b9fb7f5a91929f3923a7917aee1f1b13/general'
secret_key = 'dk53Zk9HelJGcUdBS0Fhd0lHa3hvRWJQR0dvWkdjTWw='
start = time.time()
# image_file = 'input/picture1.png'
file_name = '2023_06_20_14_20_08.jpg'
file_path ="D:/yolo/test_result_crop/"
image_file =  file_path+ file_name
save_folder = 'D:/yolo/ocr_result/'
IMP.Tool.create_folder(save_folder)

def run(image_file) :
    img = IMP.Tool.Image_Load(image_file,1)
    img = IMP.Tool.Rotate(img,-2)
    
    # img = cv2.resize(img,None,Non)
    request_json = {
        'images': [
            {
                'format': 'jpg',
                'name': 'test'
            }
        ],
        'requestId': str(uuid.uuid4()),
        'version': 'V2',
        'timestamp': 0
        # 'timestamp': int(round(time.time() * 1000))
    }

    payload = {'message': json.dumps(request_json).encode('UTF-8')}
    files = [('file', open(image_file,'rb'))]
    headers = {
    'X-OCR-SECRET': secret_key,
    }

    response = requests.request('POST',api_url, headers=headers, data = payload, files = files)
    # response = requests.post(api_url, headers=headers, files = files)
  
    res = json.loads(response.text.encode('utf8'))
    
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
    
    
    label_txt = ['내용량','기한','용도','농장명','소재지','인증번호','도축장','보관방법','부위','농장주','생산자','이력']
    num_txt = ['내용량','기한','번호']
    except1_txt = ['부위','용도']
    except1_num = [0 for i in range(len(except1_txt)) ]
    except2_txt = ['농장명','농장주']
    except2_num = [0 for i in range(len(except2_txt)) ]
    except3_txt = ['또는','번호','품목','보고','품종','및','축종','반품','품목','부위','용도','식품','이하','까지']
    except4_txt = ['C','L','k','g','K']
    
    before_label = ''
    b_bbox = []
    nums = ''
    cut_box = []
    strings = ''
    except1_box =[]
    except2_box = []
    n_string= ''
    th = 30
    flag= 0
    flag2 = 0
    flag3 = 0

    
    for idx,txt in enumerate(res['images'][0]['fields']) :
        text = res['images'][0]['fields'][idx]['inferText']
        confidence = res['images'][0]['fields'][idx]['inferConfidence']
        bbox = res['images'][0]['fields'][idx]['boundingPoly']['vertices']
        
        
        ## 농장명(농장주)가 서로 붙어있고 그 다음 글자가 '농장'이라는 글자가 오지 않으면 에러로 분류
        try :
            if '농장명' in n_string and '농장주' in n_string and '농장' in text :
                flag2 = 1
            elif '농장명' in n_string and '농장주' in n_string and '농장' not in text and flag2 == 0:
                cv2.rectangle(img,(int(cut_box[0]['x']),int(cut_box[0]['y'])),(int(cut_box[2]['x']),int(cut_box[2]['y'])),(255,0,0),3)
                img = korean_draw(img,(255,0,0,0),cut_box,text='텍스트 누락')
        except :
            pass
        
        ## 부위및용도가 서로 붙어있고 그 다음 글자가 인쇄 항목이 나오고 다른 텍스트가 오면 예외적으로 에러로 분류
        try :
            if '부위' in n_string and '용도' in n_string and len([lt for lt in label_txt if lt in text]) == 0 and len(text) > 3 :
                flag3 = 1
            elif '부위' in n_string and '용도' in n_string and flag3 == 0 :
                cv2.rectangle(img,(int(cut_box[0]['x']),int(cut_box[0]['y'])),(int(cut_box[2]['x']),int(cut_box[2]['y'])),(255,0,0),3)
                img = korean_draw(img,(255,0,0,0),cut_box,text='텍스트 누락')
                
        except :
            pass
        
        if n_string != before_label :
            flag2 = 1
        try :
            if '내용량' in n_string :
                if  len([lt for lt in label_txt if lt in text]) > 0 and len(extract_english(nums)) == 0 :
                    cv2.rectangle(img,(int(cut_box[0]['x']),int(cut_box[0]['y'])),(int(cut_box[2]['x']),int(cut_box[2]['y'])),(255,255,0),3)
                    img = korean_draw(img,(255,255,0,0),cut_box,text='텍스트 일부 누락')
        except :
            pass
        ## 초기값 설정 
        if len(before_label) > 0  :      ## before_label의 길이가 0이 아니고, 12글자 아래일 때
            cut_box = b_bbox
            n_string = before_label
            test = 0

        elif len(before_label)== 0 and len([lt for lt in label_txt if lt in text]) > 0 and len(text) < 12 : ## 현재 text가 인쇄 항목에 포함되고 12글자 이상일때
            cut_box = bbox
            n_string = text
            test = 0

        ## 예외 처리
        if len(before_label) >= 10 and len([lt for lt in label_txt if lt in text]) > 0 and len(text) >= 8 :  ## before_label 8글자 이상이고, 현재 text도 8글자 이상이면
            cut_box = bbox
            n_string = text
            test = 0
        if '제조원' in n_string and len(text) == 0 :
            cv2.rectangle(img,(int(bbox[0]['x']),int(bbox[0]['y'])),(int(bbox[2]['x']),int(bbox[2]['y'])),(255,0,0),3)
            img = korean_draw(img,(255,0,0,0),bbox,text='텍스트 누락')
            
        ## 예외 처리 : 두 줄로 이루어져 있는 항목들에 대해서 예외처리를 위해 선행정보 입력
        if len(n_string) > 0 :  
            except_text = ([lt for lt in except1_txt if lt in n_string])
            except_text2 = ([lt for lt in except2_txt if lt in n_string])
            if len([lt for lt in except1_txt if lt in n_string]) == 1 and len(cut_box) > 0 :
                except1_box = cut_box
                
            if len([lt for lt in except2_txt if lt in n_string]) == 1 and len(cut_box) > 0 :
                except2_box = cut_box
            
            
        ## 마지막 글자가 인쇄 항목에 포함되고 text의 길이가 적으면 불량으로 판정
        if idx == len(res['images'][0]['fields'])-1 and len([lt for lt in label_txt if lt in text]) > 0 and len(text) < 12: 
            cv2.rectangle(img,(int(bbox[0]['x']),int(bbox[0]['y'])),(int(bbox[2]['x']),int(bbox[2]['y'])),(255,0,0),3)
            img = korean_draw(img,(255,0,0,0),bbox,text='텍스트 누락')
            flag = 1
            print("마지막글자 에러")
        else :
            pass

        
    
                
        ## 좌표 활용 if 문
        if len(cut_box) > 1 :
            ## 같은 (세로)줄에 해당 되며(위 아래로 20픽셀 차이), 이전의 텍스트와 다음 텍스트의 (가로) 차이가 이미지의 1/2사이즈 -40보다 작으면 실행
            if abs(cut_box[0]['y'] - bbox[0]['y']) < th and abs(cut_box[2]['y'] - bbox[2]['y']) < th and abs(cut_box[0]['x'] - bbox[0]['x']) < (img.shape[1]/3) :
                for num in num_txt :
                    if num in n_string :         ## 숫자가 포함 되어있는 인쇄 항목일경우
                        nums += text
                if len(nums) == 0 :       ## 숫자가 포함 되어있지 않은 인쇄 항목일경우
                    strings += text
                ## 예외 처리 : 두 줄로 이루어져 있는 항목들에 대해서 예외처리를 위해 선행정보 입력
                if len([lt for lt in except1_txt if lt in n_string]) == 1 :
                    if '부위' in n_string or '용도' in n_string:
                        if len(strings) > 4 :
                            except1_num[except1_txt.index(except_text[0])] = 1
                        
                if len([lt for lt in except2_txt if lt in n_string]) == 1 and len([lt for lt in except2_txt if lt in strings]) == 0 :
                    if len(text) > 4 :
                        except2_num[except2_txt.index(except_text2[0])] = 1
                    
                ## 텍스트가 잘림으로 발생하는 상황 재현 후 불량으로 처리 -> 문장에서 2글자 또는 1글자가 되는 현상, 이상한 영문자로 나오는 현상 구현
                if len(text) < 3 and '내용량' not in n_string and len([lt for lt in except3_txt if lt in text]) == 0 :
                    cv2.rectangle(img,(int(bbox[0]['x']),int(bbox[0]['y'])),(int(bbox[2]['x']),int(bbox[2]['y'])),(255,255,0),3)
                    img = korean_draw(img,(255,255,0,0),bbox,text='텍스트 일부 누락')
                if len([lt for lt in except4_txt if lt in text]) == 0 and len(extract_english(text)) != 0 :
                    cv2.rectangle(img,(int(bbox[0]['x']),int(bbox[0]['y'])),(int(bbox[2]['x']),int(bbox[2]['y'])),(255,255,0),3)
                    img = korean_draw(img,(255,255,0,0),bbox,text='텍스트 일부 누락')
            ## 항목들에 대한 x좌표 또는 y좌표가 설정한 값 이상 차이나면 그 줄에 해당하는 텍스트들을 가지고 정상 및 불량 판별
            else :
                
                if len([lt for lt in num_txt if lt in n_string]) > 0 and number_len(nums) == 0 :  ## 숫자가 들어가야하는 항목에 대해 숫자가 없을 경우 불량
                    cv2.rectangle(img,(int(cut_box[0]['x']),int(cut_box[0]['y'])),(int(cut_box[2]['x']),int(cut_box[2]['y'])),(255,255,0),3)
                    img = korean_draw(img,(255,255,0,0),cut_box,text='숫자 누락')
                elif '기한' in n_string and number_len(nums) < 8 :             ## 유통기한의 숫자가 8글자가 아니면 불량
                    cv2.rectangle(img,(int(cut_box[0]['x']),int(cut_box[0]['y'])),(int(cut_box[2]['x']),int(cut_box[2]['y'])),(255,255,0),3)
                    img = korean_draw(img,(255,255,0,0),cut_box,text='숫자 일부 누락')
                elif '묶음' in n_string and number_len(nums) != 14 :        ## 묶음번호의 숫자가 14글자가 아니면 불량
                    cv2.rectangle(img,(int(cut_box[0]['x']),int(cut_box[0]['y'])),(int(cut_box[2]['x']),int(cut_box[2]['y'])),(255,255,0),3)
                    img = korean_draw(img,(255,255,0,0),cut_box,text='숫자 일부 누락')
                elif number_check(nums) == 1 and len(extract_english(nums)) == 0 and len(extract_english(text)) == 0 and has_korean(text) == 0:
                    cv2.rectangle(img,(int(cut_box[0]['x']),int(cut_box[0]['y'])),(int(cut_box[2]['x']),int(cut_box[2]['y'])),(255,0,0),3)
                elif '생산자' in n_string and number_len(nums) < 7 :    ## 생산자인증번호에서 숫자가 7글자 이상이 아니라면 불량
                    cv2.rectangle(img,(int(cut_box[0]['x']),int(cut_box[0]['y'])),(int(cut_box[2]['x']),int(cut_box[2]['y'])),(255,255,0),3)
                    img = korean_draw(img,(255,255,0,0),cut_box,text='숫자 일부 누락')   
                elif len(nums) == 0 and len(strings) < 4 and len(n_string) < 10 :
                    if '용도' not in n_string and '부위' not in n_string and '농장명' not in n_string and '농장주' not in n_string :
                        cv2.rectangle(img,(int(cut_box[0]['x']),int(cut_box[0]['y'])),(int(cut_box[2]['x']),int(cut_box[2]['y'])),(255,0,0),3)
                        img = korean_draw(img,(255,0,0,0),cut_box,text='텍스트 누락') 
                elif len(nums) == 0 and '용도' in strings and len(strings[strings.index('용도'):]) < 6 :
                    cv2.rectangle(img,(int(cut_box[0]['x']),int(cut_box[0]['y'])),(int(cut_box[2]['x']),int(cut_box[2]['y'])),(255,0,0),3)
                    img = korean_draw(img,(255,0,0,0),cut_box,text='텍스트 누락')
                elif len(nums) == 0 and '농장주' in strings and len(strings[strings.index('농장주'):]) < 6 :
                    cv2.rectangle(img,(int(cut_box[0]['x']),int(cut_box[0]['y'])),(int(cut_box[2]['x']),int(cut_box[2]['y'])),(255,0,0),3)
                    img = korean_draw(img,(255,0,0,0),cut_box,text='텍스트 누락')
                if '농장주' in n_string and '농장명' in n_string and len(strings) < 3 :
                    cv2.rectangle(img,(int(cut_box[0]['x']),int(cut_box[0]['y'])),(int(cut_box[2]['x']),int(cut_box[2]['y'])),(255,0,0),3)
                    img = korean_draw(img,(255,0,0,0),cut_box,text='텍스트 누락')
                if '부위' in n_string and '용도' in strings and len(strings[strings.index('용도'):]) < 7 :
                    cv2.rectangle(img,(int(cut_box[0]['x']),int(cut_box[0]['y'])),(int(cut_box[2]['x']),int(cut_box[2]['y'])),(255,0,0),3)
                    img = korean_draw(img,(255,0,0,0),cut_box,text='텍스트 누락')
                nums = ''
                strings = ''
                cut_box = []
        ### 초기화 ###
        before_label = ''
        b_bbox = []

        ## th 값 조정 파라미터 // 기준 : 내용량의 글자크기
        if '내용량' in text :
            th = abs(int(bbox[0]['y']) - int(bbox[2]['y'])) - 15
        

        
        ## 마지막 줄의 n_stirng(인쇄 항목)이지만, 같은 줄에 마지막 항목이 하나 더 있을 경우 예외처리
        try :
            if len(n_string) > 0 and flag == 1 and len([lt for lt in label_txt if lt in text]) > 0 and len(strings) < 4 :
                cv2.rectangle(img,(int(cut_box[0]['x']),int(cut_box[0]['y'])),(int(cut_box[2]['x']),int(cut_box[2]['y'])),(255,0,0),3)
                img = korean_draw(img,(255,0,0,0),cut_box,text='텍스트 누락')
        except :
            pass
            
        
        

        print(f"score : {confidence} / text : {text}")


        # Confidence Score가 낮으면 불량으로 검출 ##
        if confidence < 0.7 and text != '.' and text != '·' :       ## '·'으로 잘못 검출 되는 사례 제외
            cv2.rectangle(img,(int(bbox[0]['x']),int(bbox[0]['y'])),(int(bbox[2]['x']),int(bbox[2]['y'])),(0,255,0),3)
            img = korean_draw(img,(0,255,0,0),bbox,text='작업자 검수 요망')
            print("Score가 낮아서 검출")
        else :
            pass
        if len([lt for lt in label_txt if lt in text]) > 0 : 
            before_label = text
            b_bbox = bbox
            
        
    try :
        if sum(except1_num) == 0 :
            cv2.rectangle(img,(int(except1_box[0]['x']),int(except1_box[0]['y'])),(int(except1_box[2]['x']),int(except1_box[2]['y'])),(255,0,0),3)
            img = korean_draw(img,(255,0,0,0),except1_box,text='텍스트 누락')
        if sum(except2_num) == 0 :
            cv2.rectangle(img,(int(except2_box[0]['x']),int(except2_box[0]['y'])),(int(except2_box[2]['x']),int(except2_box[2]['y'])),(255,0,0),3)
            img = korean_draw(img,(255,0,0,0),except2_box,text='텍스트 누락')
    except :
        pass
    endtime = time.time()
    print("처리 시간 : ",round(endtime-start,3))
    IMP.Tool.Show(img)
    cv2.imwrite(save_folder+file_name,img)


run(image_file)


    


# with open(output_file, 'w', encoding='utf-8') as outfile:
#     json.dump(res, outfile, indent=4, ensure_ascii=False)


