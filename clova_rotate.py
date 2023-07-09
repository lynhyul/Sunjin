import IMP
import cv2
import os
import numpy as np
import time
import imutils
# path = 'd:/test_image/'
path = 'd:/ref_image/test/'
# file = os.listdir(path)[-1]

def scaling_matching(target_src,template,resize_ratio = 1/5) :
    ## 원본 이미지 Load
    start = time.time()

    ## 처리시간 단축을 위한 Resize
    if resize_ratio != 1 :      ## resize scale이 1이 아닐 때 template 이미지 size에 2배수 만큼 곱하기
        target = cv2.resize(target_src,None,None,resize_ratio,resize_ratio)
        template = cv2.resize(template,None,None,resize_ratio,resize_ratio)
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
    scale = 1
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
    
    angle_rad = np.arctan2(similarity_transform_matrix[1, 0], similarity_transform_matrix[0, 0])
    angle_deg = np.degrees(angle_rad)
    print(angle_deg)
    print("각도 보정 : ",angle_deg)
    print(scale)
    rotate_image= imutils.rotate_bound(target_src,-angle_deg)
    rotate_image = cv2.resize(rotate_image,None,None,1/5,1/5)
    end = time.time()
    t_time = abs(round(start - end,2))
    print(t_time,"s")
    return rotate_image

flist = os.listdir(path)

for file in flist :
    fpath = path + file
    # file_name = '2023-05-24-804.jpg'
    file_name = file
    target_src = IMP.Tool.Image_Load(path+file_name,0)

    template = IMP.Tool.Image_Load('d:/ref_image/n_ref5.jpg',0)
    rotate_image= scaling_matching(target_src,template, resize_ratio=0.5)
    target_src = cv2.resize(target_src,None,None,1/5,1/5)
    # cv2.imshow(f'target', target_src)
    cv2.imshow(f'target_src', target_src)
    cv2.imshow(f'rotate_image', rotate_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()