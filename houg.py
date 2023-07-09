import cv2
import numpy as np

def detect_bottom_line(image):
    # Convert the image to grayscale
    h,w = image.shape[:2]
    # 가장자리 검출
    edges = cv2.Canny(image, 0, 150, apertureSize=3)

    # Hough 변환 적용
    lines = cv2.HoughLines(edges,1,np.pi/180,200,w/2+30)

    # 가로 형태의 최하단 직선을 찾기 위한 변수 초기화
    max_rho = -1
    max_line = None

    # lines 내 각 선에 대해
    for line in lines:
        for rho, theta in line:
            # 가로 형태의 선 찾기 (θ가 0 또는 π에 가깝게)
            if np.abs(np.pi/2 - theta) < np.pi/4:
                if theta < np.pi/4:
                    theta += np.pi/2
                elif theta > 3*np.pi/4:
                    theta -= np.pi/2

                # Update the maximum rho value if the current rho is larger
                if rho > max_rho:
                    max_rho = rho
                    max_line = line
                    
    center = (image.shape[1] / 2, image.shape[0] / 2)

    # theta를 radian에서 degree로 변환
    # 이미지 회전
    

    # 가로 형태의 최하단 직선을 찾았다면
    if max_line is not None:
        for rho, theta in max_line:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            # 이미지 회전을 위한 변환 매트릭스 계산
            rotation_matrix = cv2.getRotationMatrix2D(center, -theta, 1)
            print(-theta, "보정")
            rotated_image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))
            image = rotated_image[:int(y0)+10,:]
    return image

import IMP
img = IMP.Tool.Image_Load('D:/yolo/test_result_crop/2023_06_20_14_25_46.jpg',1)

rotated_image = detect_bottom_line(img)

IMP.Tool.Show(rotated_image)

