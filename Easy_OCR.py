from matplotlib import pyplot as plt
from imutils.perspective import four_point_transform
from imutils.contours import sort_contours
import imutils
from easyocr import Reader
import cv2
import requests
import numpy as np
from PIL import ImageFont, ImageDraw, Image
import pytesseract


def plt_imshow(title='image', img=None, figsize=(8 ,5)):
    plt.figure(figsize=figsize)
 
    if type(img) == list:
        if type(title) == list:
            titles = title
        else:
            titles = []
 
            for i in range(len(img)):
                titles.append(title)
 
        for i in range(len(img)):
            if len(img[i].shape) <= 2:
                rgbImg = cv2.cvtColor(img[i], cv2.COLOR_GRAY2RGB)
            else:
                rgbImg = cv2.cvtColor(img[i], cv2.COLOR_BGR2RGB)
 
            plt.subplot(1, len(img), i + 1), plt.imshow(rgbImg)
            plt.title(titles[i])
            plt.xticks([]), plt.yticks([])
 
        plt.show()
    else:
        if len(img.shape) < 3:
            rgbImg = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            rgbImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
 
        plt.imshow(rgbImg)
        plt.title(title)
        plt.xticks([]), plt.yticks([])
        plt.show()
 
 
def make_scan_image(image, width, ksize=(5,5), min_threshold=75, max_threshold=200):
  image_list_title = []
  image_list = []
 
  image = imutils.resize(image, width=width)
  ratio = business_card_image.shape[1] / float(image.shape[1])
 
  # 이미지를 grayscale로 변환하고 blur를 적용
  # 모서리를 찾기위한 이미지 연산
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  blurred = cv2.GaussianBlur(gray, ksize, 0)
  edged = cv2.Canny(blurred, min_threshold, max_threshold)
 
  image_list_title = ['gray', 'blurred', 'edged']
  image_list = [gray, blurred, edged]
 
  # contours를 찾아 크기순으로 정렬
  cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  cnts = imutils.grab_contours(cnts)
  cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
 
  findCnt = None
 
  # 정렬된 contours를 반복문으로 수행하며 4개의 꼭지점을 갖는 도형을 검출
  for c in cnts:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
 
    # contours가 크기순으로 정렬되어 있기때문에 제일 첫번째 사각형을 영역으로 판단하고 break
    if len(approx) == 4:
      findCnt = approx
      break
 
 
  # 만약 추출한 윤곽이 없을 경우 오류
  if findCnt is None:
    raise Exception(("Could not find outline."))
 
 
  output = image.copy()
  cv2.drawContours(output, [findCnt], -1, (0, 255, 0), 2)
  
  image_list_title.append("Outline")
  image_list.append(output)
 
  # 원본 이미지에 찾은 윤곽을 기준으로 이미지를 보정
  transform_image = four_point_transform(business_card_image, findCnt.reshape(4, 2) * ratio)
 
  plt_imshow(image_list_title, image_list)
  plt_imshow("Transform", transform_image)
 
  return transform_image


def putText(cv_img, text, x, y, color=(0, 0, 0), font_size=22):
  # Colab이 아닌 Local에서 수행 시에는 gulim.ttc 를 사용하면 됩니다.
  # font = ImageFont.truetype("fonts/gulim.ttc", font_size)
  font = ImageFont.truetype('/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf', font_size)
  img = Image.fromarray(cv_img)
   
  draw = ImageDraw.Draw(img)
  draw.text((x, y), text, font=font, fill=color)
 
  cv_img = np.array(img)
  
  return cv_img


# url = 'https://user-images.githubusercontent.com/69428232/155486780-55525c3c-8f5f-4313-8590-dd69d4ce4111.jpg'
from IMP import Tool
import os

path = 'd:/'
# file = os.listdir(path)[3]
 
business_card_image = Tool.Image_Load(path+'클립 이미지.jpg',1)
# business_card_image = cv2.resize(business_card_image,(200,200))
# ref_img = Tool.Image_Load(path+'ref.png',1)
# image_nparray = np.asarray(bytearray(requests.get(url).content), dtype=np.uint8)
# org_image = cv2.imdecode(image_nparray, cv2.IMREAD_COLOR) 

 
import IMP
# business_card_image = make_scan_image(org_image, width=480, ksize=(5, 5), min_threshold=75, max_threshold=200)
# business_card_image = IMP.Matching.pattern_match_rotation(org_image,ref_img,5)
# plt_imshow("business_card_image", business_card_image)



langs = ['ko', 'en']
 
print("[INFO] OCR'ing input image...")
from time import time
start_time = time()
reader = Reader(lang_list=langs, gpu=True)
# reader = pytesseract.image_to_string(business_card_image,lang='kor')
# results = pytesseract.image_to_boxes(business_card_image,lang='kor')
print(reader)
# # results = reader.readtext(business_card_image)
end_time = time()
print("모델 로딩 시간 : ",start_time-end_time)
results = reader.readtext(business_card_image)
start_time = time()
results = reader.readtext(business_card_image)
print(results)
end_time = time()
print(start_time-end_time)

# # results


# simple_results = reader.readtext(business_card_image, detail = 0)
# simple_results


# loop over the results
# for bbox in results:
#     # print("[INFO] {:.4f}: {}".format(prob, text))

#     (tl, tr, br, bl) = bbox
#     tl = (int(tl[0]), int(tl[1]))
#     tr = (int(tr[0]), int(tr[1]))
#     br = (int(br[0]), int(br[1]))
#     bl = (int(bl[0]), int(bl[1]))

#     # 추출한 영역에 사각형을 그리고 인식한 글자를 표기합니다.
#     cv2.rectangle(business_card_image, tl, br, (0, 255, 0), 2)
#     # print()
#     # business_card_image = putText(business_card_image, text, tl[0], tl[1] - 60, (0, 255, 0), 50)
#     business_card_image = Image.fromarray(business_card_image)
#     fontpath = "fonts/gulim.ttc"
#     font = ImageFont.truetype(fontpath, 10)
#     b,g,r,a = 0,0,0,255 # 빨간색 글자
#     draw = ImageDraw.Draw(business_card_image, 'RGBA')
#     # draw.text((tl[0]-10, tl[1]-10), text, font=font, fill=(b,g,r,a))
#     business_card_image= np.array(business_card_image)
#     # # putText(business_card_image, text, tl[0], tl[1] - 60, (0, 255, 0), 2)
#     # cv2.putText(business_card_image, text, (tl[0], tl[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
 
# plt_imshow("Image", business_card_image, figsize=(8,5))