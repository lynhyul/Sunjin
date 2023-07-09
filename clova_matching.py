import IMP
import cv2
import os
import numpy as np
import time

# path = 'd:/test_image/'
path = 'd:/ref_image/'
# file = os.listdir(path)[-1]

def scaling_matching(target_src,template,resize_ratio = 1/5) :
    ## 원본 이미지 Load
    start = time.time()

    ## 처리시간 단축을 위한 Resize
    if resize_ratio != 1 :      ## resize scale이 1이 아닐 때 template 이미지 size에 2배수 만큼 곱하기
        target = cv2.resize(target_src,None,None,resize_ratio,resize_ratio)
        template = cv2.resize(template,None,None,resize_ratio*2,resize_ratio*2)
    else :
        target = target_src
    # template = cv2.medianBlur(template,3)
    sift = cv2.SIFT_create()
    # kp1, des1 = sift.detectAndCompute(template, None)
    kp2, des2 = sift.detectAndCompute(target, None)

    # BF(Brute-Force) 매처 생성
    bf = cv2.BFMatcher()

    # 템플릿 이미지를 다양한 크기로 변환하여 매칭
    good_matches = []  # Good Match 저장할 리스트
    scale = 5
    scaled_template = cv2.resize(template, None, fx=scale, fy=scale)
    kp1_scaled, des1_scaled = sift.detectAndCompute(scaled_template, None)
    matches = bf.knnMatch(des1_scaled, des2, k=2)

    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    # 키포인트 좌표 추출
    src_pts = np.float32([kp1_scaled[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Similarity Transform 행렬 계산
    similarity_transform_matrix, _ = cv2.estimateAffine2D(src_pts, dst_pts)

    # 템플릿 이미지를 대상 이미지에 매핑
    h, w = scaled_template.shape[:2]
    warped_template = cv2.warpAffine(scaled_template, similarity_transform_matrix, (w, h))


    contours, _ = cv2.findContours(warped_template, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # # 최대 contour의 경계상자 좌표 추출
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))

    # 검은 부분을 제외한 이미지 슬라이싱
    image_roi = warped_template[y:y+h, x:x+w]
    
    angle_rad = np.arctan2(similarity_transform_matrix[1, 0], similarity_transform_matrix[0, 0])
    angle_deg = np.degrees(angle_rad)
    print(angle_deg)

    ## 템플릿 매칭
    top_left,score, clip_image = IMP.Matching.Template_Matching(target,image_roi,5)
    ## 얻은 좌표를 원본 배율로 다시 확대해서 적용
    h,w = image_roi.shape
    # if h > template * 2/3 :
    ratio= int(1/resize_ratio)
    # clip_image= target_src[top_left[1]*ratio:(top_left[1]+h)*ratio,top_left[0]*ratio:(top_left[0]+w)*ratio]

    
    clip_image = IMP.Tool.Rotate(target_src,angle_deg)

    end = time.time()
    t_time = abs(round(start - end,2))
    print(t_time,"s")
    print("각도 보정 : ",angle_deg)
    print(scale)
    return clip_image

file_name = '2023_06_20_14_00_09.jpg'
target_src = IMP.Tool.Image_Load(path+file_name,0)
# target_src = IMP.Tool.Rotate(target_src,90)
# target_src = IMP.Tool.Image_Load(path+'중간_불량.jpg',0)
# target_src = IMP.Tool.Image_Load(path+'test2.png',0)
template = IMP.Tool.Image_Load(path+'ref.jpg',0)
# template = IMP.Tool.Image_Load(path+'ref6.png',0)
clip_image= scaling_matching(target_src,template, resize_ratio=1)
cv2.imwrite(f'{path}/ref8.jpg',clip_image)
# # 결과 이미지 출력
# target_src = cv2.resize(target_src,None,None,1/5,1/5)
# cv2.imshow(f'target', target_src)
cv2.imshow(f'image_roi', template)
cv2.imshow(f'Clip_image', clip_image)
cv2.waitKey(0)
cv2.destroyAllWindows()