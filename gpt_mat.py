import cv2
import numpy as np
import IMP
import os
import time


path = 'd:/test_image/'
file = os.listdir(path)[-1]

def sift_matching(template_img, target_img_src, threshold=0.7, resize_ratio=0.2):
    # 템플릿 이미지와 원본 이미지를 리사이징
    start = time.time()

    template_img = cv2.resize(template_img, None, fx=resize_ratio, fy=resize_ratio)
    target_img = cv2.resize(target_img_src, None, fx=resize_ratio, fy=resize_ratio)

    # SIFT 초기화
    sift = cv2.xfeatures2d.SIFT_create()

    # 키포인트 및 디스크립터 추출
    kp1, des1 = sift.detectAndCompute(template_img, None)
    kp2, des2 = sift.detectAndCompute(target_img, None)

    # FLANN 기반 매칭
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # 좋은 매칭점 선택
    good_matches = []
    for m, n in matches:
        if m.distance < threshold * n.distance:
            good_matches.append(m)

    # 좋은 매칭점들의 좌표 추출
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # 유사도 변환 행렬 추정
    M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # 템플릿 이미지를 원본 이미지에 매핑
    h, w = template_img.shape[:2]
    template_corners = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
    dst_corners = cv2.perspectiveTransform(template_corners, M)
    dst_corners = np.int32(dst_corners)
    top_left = dst_corners[0][0]
    botoom_right = dst_corners[2][0]
    h = abs(top_left[1] - botoom_right[1])
    w = abs(top_left[0] - botoom_right[0])
    target_with_box = target_img_src[top_left[1]*5:(top_left[1]+h)*5,top_left[0]*5:(top_left[0]+w)*5]
    # 테두리 그리기 
    # target_with_box = cv2.polylines(target_img, [dst_corners], True, (0, 255, 0), 3)
    end = time.time()
    t_time = abs(round(start - end,2))
    print(t_time)
    return target_with_box

# 템플릿 이미지와 원본 이미지 로딩
target_src = IMP.Tool.Image_Load(path+'test1.png',0)
# target_src = IMP.Tool.Image_Load(path+'넓은거_정상.bmp',0)
template = IMP.Tool.Image_Load(path+'ref4.png',0)
# template = IMP.Tool.Image_Load(path+'tt.png',0)

# SIFT 매칭
result_img = sift_matching(template, target_src, threshold=0.7, resize_ratio=0.5)

# 결과 이미지 출력
cv2.imshow('Result', result_img)
cv2.waitKey(0)
cv2.destroyAllWindows()