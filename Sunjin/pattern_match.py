import cv2
import numpy as np


def pattern_match_rotation(image, template,th=0):
    # 회전 변환 함수
    def rotate(image, angle):
        h, w = image.shape[:2]
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h))
        return rotated
    
    # 이미지 및 패턴 정보 가져오기
    if len(image.shape) > 2 :  
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if len(template.shape) > 2 :
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    t_h, t_w = template_gray.shape[:2]
    
    img_gray = cv2.medianBlur(img_gray,3)
    template_gray = cv2.medianBlur(template_gray,3)
    scores = []
    locs = []
    locs2 = []
    angles = []
    scores2 = []
    # 이미지 회전 및 패턴 매칭
    for angle in range(0, 360, 15):
        rotated = rotate(img_gray, angle)
        result = cv2.matchTemplate(rotated, template_gray, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        scores.append(max_val)
        locs.append(max_loc)
        
        
        #if max_val > 0.25:  # 일치하는 패턴이 존재하는 경우
            # 회전된 이미지에서 패턴 위치 계산
    idx = scores.index(max(scores)) ## 매칭 score가 가장 높은 부분을 추출
    loc = locs[idx]
    for angle in range(0, 360, 1):
        if angle % 15 != 0 :
            rotated = rotate(img_gray, angle)
            result = cv2.matchTemplate(rotated, template_gray, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            if max_val > max(scores) and angle < 300  :
                # 원본 이미지에서 패턴 위치 계산
                image = rotate(image,angle)
                print(f"score : {max(scores)}")
                print(f"앵글 보정 : {angle}")
                image = image[max_loc[1]-th:max_loc[1]+t_h+th,max_loc[0]-th:max_loc[0]+t_w+th]
                return image
            elif max_val > max(scores) and angle > 350  :
                scores2.append(max_val)
                angles.append(angle)
                locs2.append(max_loc)
                if angle == 359 :
                    idx = scores2.index(max(scores2))
                    max_loc = locs2[idx]
                    angle = angles[idx]
                    image = rotate(image,angle)
                    print(f"score : {max(scores)}")
                    print(f"앵글 보정 : {angle}")
                    image = image[max_loc[1]-th:max_loc[1]+t_h+th,max_loc[0]-th:max_loc[0]+t_w+th]
                    try :
                        return image
                    except :
                        pass
            
    return image[loc[1]-th:loc[1]+t_h+th,loc[0]-th:loc[0]+t_w+th]




src = cv2.imread('d:/test_image/test1.png')
ref = cv2.imread('d:/test_image/ref.png')
# src = rotate(src,90)


image = pattern_match_rotation(src,ref,15)

cv2.imshow('src',src)
cv2.imshow('ref',ref)
cv2.imshow('result',image)
cv2.waitKey(0)
cv2.destroyAllWindows()
