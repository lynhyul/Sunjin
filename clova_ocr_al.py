import cv2
import numpy as np
import IMP
import requests
import uuid
import time
import json
import os
# from clova_matching import scaling_matching

# image_file = IMP.Tool.Image_Load("D:/test_image/img2.jpg",1)


api_url = 'https://s2cm7yuquo.apigw.ntruss.com/custom/v1/22188/fa24acd3ef4dc26539dd12962a443577b9fb7f5a91929f3923a7917aee1f1b13/general'
secret_key = 'U2llcExObEFhcnBQZnNLc0Z2Z3pvdXRyZmlYTnRwZVA='
start = time.time()
# image_file = 'input/picture1.png'
image_file = "D:/ref_image/ref_test.jpg"
save_folder = 'D:/ref_image/data/'

def run(image_file) :
    img = IMP.Tool.Image_Load(image_file,1)
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
    endtime = time.time()
    print("호출 시간 : ",round(endtime-start,3))
    res = json.loads(response.text.encode('utf8'))
    
    def number_check(text) :
        if any(c.isdigit() for c in text) and not all(c.isalpha() for c in text) :
            return 1
    
    label_txt = ['내용량','기한','용도','농장명','소재지','번호','도축장','제조원','보관방법','원재료명']
    num_txt = ['내용량','기한','번호']
    
    before_label= ''
    
    for idx,txt in enumerate(res['images'][0]['fields']) :
        text = res['images'][0]['fields'][idx]['inferText']
        confidence = res['images'][0]['fields'][idx]['inferConfidence']
        bbox = res['images'][0]['fields'][idx]['boundingPoly']['vertices']
        if len(before_label) > 1 and before_label != '번호' and '판매원' not in before_label:      ## before string의 길이가 1개 이상 일경우 진행
            ### 숫자 필터링 ###
            for num in num_txt :
                if num in before_label and number_check(text) != 1:
                    cv2.rectangle(img,(int(b_bbox[0]['x']),int(b_bbox[0]['y'])),(int(b_bbox[2]['x']),int(b_bbox[2]['y'])),(0,0,255),3)
                for lt in label_txt :
                    if lt in text and len(before_label) < 15:
                        cv2.rectangle(img,(int(b_bbox[0]['x']),int(b_bbox[0]['y'])),(int(b_bbox[2]['x']),int(b_bbox[2]['y'])),(0,0,255),3)
                
                # elif abs(int(b_bbox[0]['x']) - int(bbox[0]['x'])) > (img.shape[1]/2)-20 :
                #     cv2.rectangle(img,(int(b_bbox[0]['x']),int(b_bbox[0]['y'])),(int(b_bbox[2]['x']),int(b_bbox[2]['y'])),(0,0,255),3)
        else :
            pass
        ### 초기화 ###
        before_label = ''
        b_bbox = []

        print(f"score : {confidence} / text : {text}")
        if confidence < 0.8 and len(text) > 1 : 
            cv2.rectangle(img,(int(bbox[0]['x']),int(bbox[0]['y'])),(int(bbox[2]['x']),int(bbox[2]['y'])),(0,0,255),3)
        else :
            pass
        for lt in label_txt :
            if lt in text :
                before_label = text
                b_bbox = bbox

    IMP.Tool.Show(img)


run(image_file)

# with open(output_file, 'w', encoding='utf-8') as outfile:
#     json.dump(res, outfile, indent=4, ensure_ascii=False)


