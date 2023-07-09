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
secret_key = 'dk53Zk9HelJGcUdBS0Fhd0lHa3hvRWJQR0dvWkdjTWw='
start = time.time()
# image_file = 'input/picture1.png'
image_file = "D:/ref_image/ref7.jpg"
# image_file = "D:/ref14.jpg"
save_folder = 'D:/ref_image/data/'

def run(save_folder,image_file) :
    img = IMP.Tool.Image_Load(image_file,1)
    if len(os.listdir(save_folder)) > 1 : 
        # count = sorted(os.listdir(save_folder), key=int)[-1].split('.')[0].split('_')[-1]
        with open("D:/ref_image/label.txt", "r",encoding='cp949') as f:
            lines = f.readlines()
            last_str = lines[-1].split('_')[-1].split('.')[0]
        count = int(last_str)
    else :
        count = 0


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
    'X-OCR-SECRET': secret_key
    }

    response = requests.request('POST',api_url, headers=headers, data = payload, files = files)
    # response = requests.post(api_url, headers=headers, files = files)
    endtime = time.time()
    print("호출 시간 : ",round(endtime-start,3))
    res = json.loads(response.text.encode('utf8'))

    f = open('D:/ref_image/label.txt','a')

    for idx,txt in enumerate(res['images'][0]['fields']) :
        count +=1
        text = res['images'][0]['fields'][idx]['inferText']
        confidence = res['images'][0]['fields'][idx]['inferConfidence']
        bbox = res['images'][0]['fields'][idx]['boundingPoly']['vertices']
        print(f"score : {confidence} / text : {text}")
        # cv2.rectangle(img,(int(bbox[0]['x']),int(bbox[0]['y'])),(int(bbox[2]['x']),int(bbox[2]['y'])),(0,0,255),3)
        clip_image = img[int(bbox[0]['y']):int(bbox[2]['y']),int(bbox[0]['x']):int(bbox[2]['x'])]
        cv2.imwrite(save_folder+f'/images_{count}.jpg',clip_image)
        f.write(f'D:/ref_image/data/images_{count}.jpg\t{text}\n')
        test = 0
    end = time.time()
    print(round(end-start,3))

run(save_folder,image_file)

# with open(output_file, 'w', encoding='utf-8') as outfile:
#     json.dump(res, outfile, indent=4, ensure_ascii=False)


