import cv2
import numpy as np
import IMP


def high_pass_filter(image, cutoff_freq):
    # 이미지를 그레이스케일로 변환
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 이미지를 float32 자료형으로 변환
    gray = np.float32(gray)

    # 이미지에 대해 2D 푸리에 변환 수행
    dft = cv2.dft(gray, flags=cv2.DFT_COMPLEX_OUTPUT)

    # 주파수 영역으로 이동
    dft_shift = np.fft.fftshift(dft)

    # 고역 통과 필터 생성
    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.ones((rows, cols, 2), np.uint8)
    mask[crow - cutoff_freq:crow + cutoff_freq, ccol - cutoff_freq:ccol + cutoff_freq] = 0

    # 필터링 수행
    fshift = dft_shift * mask

    # 주파수 영역에서 다시 이미지 영역으로 이동
    f_ishift = np.fft.ifftshift(fshift)

    # 역 푸리에 변환을 통해 이미지 얻기
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

    # 최소값과 최대값 정규화
    img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    return img_back

img = IMP.Tool.Image_Load('D:/2023_07_01_08_42_18_crop.jpg',1)
cutoff_frequency = 10
image2 = high_pass_filter(img,cutoff_frequency)


# clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
# enhanced = clahe.apply(gray)
# enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
# result = cv2.addWeighted(img, 0.7, enhanced, 0.3, 0)

# IMP.Tool.Show_Resize(img,int(img.shape[1]/3),int(img.shape[0]/3))
IMP.Tool.Show(image2)
# IMP.Tool.Show_Resize(image2,int(img.shape[1]/3),int(img.shape[0]/3))

# ## Track Bar Laplacian

# def onChange(pos):
#     pass

# cv2.namedWindow("Trackbar Windows")

# cv2.createTrackbar("ksize", "Trackbar Windows", 0, 255, lambda x : x)

# cv2.setTrackbarPos("ksize", "Trackbar Windows", 1)

# while cv2.waitKey(1) != ord('q'):

#     ksize = cv2.getTrackbarPos("ksize", "Trackbar Windows")
#     cl
#     # if ksize % 2 != 0 :
#         # laplacian = cv2.Laplacian(gray_img, cv2.CV_8U, ksize=ksize)

#     cv2.imshow("Trackbar Windows", laplacian)

# cv2.destroyAllWindows()
