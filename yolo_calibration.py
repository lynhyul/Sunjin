import cv2
import numpy as np

# 왜곡 계수
k1 = 0.0
k2 = 0.0
k3 = 0.0
k4 = 0.0

# 카메라 매트릭스 계산
focal_length = 8.0  # 렌즈 초점 거리 (mm)
sensor_width = 5.76  # 센서 너비 (mm)
image_width = 4096  # 이미지 너비 (픽셀)
image_height = 3000  # 이미지 높이 (픽셀)
fx = fy = focal_length * sensor_width / image_width
cx = cy = image_width / 2
camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

# 왜곡 보정 함수
def undistort_image(image):
    # 이미지 크기
    image_height, image_width = image.shape[:2]

    # 왜곡 계수
    distortion_coeffs = np.array([k1, k2, k3, k4], dtype=np.float64)

    # 왜곡 보정
    undistorted_image = cv2.undistort(image, camera_matrix, distortion_coeffs)

    return undistorted_image

# 이미지 로드
image = cv2.imread('d:/yolo/test_data/test_result/2023_06_29_09_41_35.jpg')

# 왜곡 보정 적용
undistorted_image = undistort_image(image)

h,w = undistorted_image.shape[:2]

cv2.imwrite("d:/yolo/2023_06_29_09_41_35_1.jpg",undistorted_image)
cv2.imwrite("d:/yolo/2023_06_29_09_41_35.jpg",image)

# import IMP
# IMP.Tool.Show_Resize(undistorted_image,int(w/5),int(h/5))

# 결과 이미지 출력
# cv2.imshow('Undistorted Image', undistorted_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()