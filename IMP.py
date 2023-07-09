import cv2
import numpy as np
import copy
# import imutils
import os

class Binary() :
    def IntensityThresholding(img,th,mode=1) :
        if mode == 'inv' :
            ret, dst = cv2.threshold(img, th, 255, cv2.THRESH_BINARY_INV)
        elif mode == 'otsu' :
            ret, dst = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        else : 
            ret, dst = cv2.threshold(img, th, 255, cv2.THRESH_BINARY)
        print("Threshold : ",ret)
        return dst,ret
    

    def MultiIntensityThresholding(img,min,max) :
        ret, dst = cv2.threshold(img, min, 255, cv2.THRESH_BINARY)
        ret2, dst2 = cv2.threshold(img, max, 255, cv2.THRESH_BINARY)
        dst = dst + dst2
        return dst

class Tool() :
    def create_folder(folder_path):
        try:
            os.makedirs(folder_path)
            print(f"Folder created: {folder_path}")
        except FileExistsError:
            print(f"Folder already exists: {folder_path}")
            
    def Save(img,path) :
        cv2.imwrite(path,img)
        return 1

    def GrayToRGB(img) :
        img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
        return img

    def RGBToGray(img) :
        img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        return img

    def Inverse(img) :
        img = cv2.bitwise_not(img)
        return img

    def Image_Load(path,mode) :
        img_array = np.fromfile(path, np.uint8)
        img = cv2.imdecode(img_array, mode)
        return img

    def Clip(img,sx,ex,sy,ey) :
        img= img[sy:ey,sx:ex]
        return img

    def Rotate(img,degree,scale=1.0) :
        (h, w) = img.shape[:2]
        (cX, cY) = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D((cX, cY), degree, scale)
        img = cv2.warpAffine(img, M, (w, h))
        return img

    def flip(img,mode) :
        img = cv2.flip(img,mode)
        return img
    
    def Resize(img,fx,fy) :
        img = cv2.resize(img,(fx,fy))
        return img
    
    def Show(img) :
        cv2.imshow("img", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    def Show_Resize(img,size_x,size_y) :
        img = cv2.resize(img,(size_x,size_y))
        cv2.imshow("img", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    def Copy(img) :
        img = copy.deepcopy(img)
        return img
        
class Morphology() :
    def Closing(img,mask_w,mask_h, iterations=1) :
        kernel = np.ones((mask_h,mask_w))
        img = cv2.morphologyEx(img,cv2.MORPH_CLOSE,kernel,iterations = iterations)
        return img

    def Opening(img,mask_w,mask_h,iterations= 1) :
        kernel = np.ones((mask_h,mask_w))
        img = cv2.morphologyEx(img,cv2.MORPH_OPEN,kernel,iterations = iterations)
        return img

    def Erosion(img,mask_w,mask_h,iterations=1) :
        kernel = np.ones((mask_h,mask_w))
        img = cv2.erode(img,kernel,iterations=iterations)
        return img

    def Dilation(img,mask_w,mask_h,iterations=1) :
        kernel = np.ones((mask_h,mask_w))
        img = cv2.dilate(img,kernel,iterations=iterations)
        return img
    
class Filter() :
    def Blur(img,filter_size=7) :
        img = cv2.blur(img,(filter_size,filter_size))
        return img
    
    def Median(img,filter_size=7) :
        img = cv2.medianBlur(img,filter_size)
        return img
    
    def Bilateral(img,d=10, sigmaColor=75, sigmaSpace=75) :
        img = cv2.bilateralFilter(img,10,75,75)
        return img
    
    def Gausian_Blur(img,filter_size=5,sig_xy=0):
        img = cv2.GaussianBlur(img,(filter_size,filter_size),sig_xy)
        return img
    
    def Laplacian(img) :
        img = cv2.Laplacian(img,cv2.CV_64F)
        return img
        
    def Sobel(img,filter_size=7,mode='x') :
        if mode == 'x' :
            img = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=filter_size)
        elif mode == 'y' :
            img = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=filter_size)
        elif mode == 'combine' :
            sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=filter_size)
            sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=filter_size)
            img = cv2.bitwise_and(sobelx, sobely)
        return img
    
    
class Matching() :
    def pattern_match_rotation(image, template,th=0,gamma = 1):
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
        else :
            img_gray = image
        
        if len(template.shape) > 2 :
            template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
            
        else :
            template_gray= template
        t_h, t_w = template_gray.shape[:2]
        
        # img_gray = cv2.medianBlur(img_gray,5)
        # template_gray = cv2.medianBlur(template_gray,5)
        scores = []
        locs = []
        locs2 = []
        angles2 = []
        angles = []
        scores2 = []
        # 이미지 회전 및 패턴 매칭
        for angle in range(0, 360, gamma):
            rotated = rotate(img_gray, angle)
            result = cv2.matchTemplate(rotated, template_gray, cv2.TM_CCOEFF_NORMED)
            result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            if max_val > 0.9 :
                print(max_val)
                return image[max_loc[1]-th:max_loc[1]+t_h+th,max_loc[0]-th:max_loc[0]+t_w+th]
            scores.append(max_val)
            locs.append(max_loc)
            angles.append(angle)
            
            #if max_val > 0.25:  # 일치하는 패턴이 존재하는 경우
                # 회전된 이미지에서 패턴 위치 계산
        idx = scores.index(max(scores)) ## 매칭 score가 가장 높은 부분을 추출
        loc = locs[idx]
        for angle in range(0, 360, gamma):
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
                    angles2.append(angle)
                    locs2.append(max_loc)
                    if angle == 359 :
                        idx = scores2.index(max(scores2))
                        max_loc = locs2[idx]
                        angle = angles2[idx]
                        image = rotate(image,angle)
                        print(f"score : {max(scores)}")
                        print(f"앵글 보정 : {angle}")
                        image = image[max_loc[1]-th:max_loc[1]+t_h+th,max_loc[0]-th:max_loc[0]+t_w+th]
                        try :
                            return image
                        except :
                            pass
        angle = angles[idx]
        image = rotate(image,angle)
        print(f"score : {max(scores)}")
        print(f"앵글 보정 : {angle}")
        return image[loc[1]-th:loc[1]+t_h+th,loc[0]-th:loc[0]+t_w+th]
    
    
    def Pattern_Matching(target, template, th=15):
        # ORB 디스크립터 추출기를 생성한다
        orb = cv2.ORB_create()

        # 템플릿 이미지와 대상 이미지에서 특징점과 디스크립터를 추출한다
        kp1, des1 = orb.detectAndCompute(template, None)
        kp2, des2 = orb.detectAndCompute(target, None)

        # BFMatcher 객체를 생성하여 매칭을 수행한다
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)

        # 매칭 결과를 거리 기준으로 오름차순 정렬한다
        matches = sorted(matches, key=lambda x: x.distance)

        # 가장 매칭이 잘 된 상위 N개의 특징점을 이용하여 변환 행렬을 계산한다
        N = 10
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches[:N]]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches[:N]]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)


        # 변환 행렬을 이용하여 템플릿 이미지의 위치를 대상 이미지 상에 찾는다
        h, w = template.shape[:2]
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)

        xs = []
        ys = []
        for x in np.int32(dst):
            xs.append(x[0][0])
            ys.append(x[0][1])

        sx = np.min(xs)
        ex = np.max(xs)
        sy = np.min(ys)
        ey = np.max(ys)
        
        # 회전각도 계산을 위해 이미지 중심점 구하기
        rows, cols = target.shape[:2]
        center_x, center_y = cols // 2, rows // 2

        # 변환 행렬을 이용하여 회전 각도 계산
        M_inv = np.linalg.inv(M)
        theta = np.arctan2(M_inv[1, 0], M_inv[0, 0])

        # 회전 변환 행렬 계산
        rot_mat = cv2.getRotationMatrix2D((center_x, center_y), np.degrees(-theta), 1)
        rotated_target = cv2.warpAffine(target, rot_mat, (cols, rows))
        
        clip_image = target[sy - th:ey + th, sx - th:ex + th]

        if len(target.shape) > 2:
            result_image = copy.deepcopy(target)
            # result_image[sy-th:ey+th, sx-th:ex+th] = rotated_target
        else:
            result_image = cv2.cvtColor(target, cv2.COLOR_GRAY2RGB)
            # result_image[sy-th:ey+th, sx-th:ex+th] = rotated_target

        img = cv2.polylines(result_image, [np.int32(dst)], True, (0, 255, 0), 2, cv2.LINE_AA)
        return clip_image

    
    
    def Template_Matching(src_img, ref_img,thresh = 10) :
        res = cv2.matchTemplate(src_img, ref_img, cv2.TM_CCOEFF_NORMED) # 여기서 최댓값 찾기
        # 최솟값 0, 최댓값 255 지정하여 결과값을 그레이스케일 영상으로 만들기
        res_norm = cv2.normalize(res, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        # 최댓값을 찾아야하므로 minmaxloc 사용, min, max, min좌표, max좌표 반환
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        # th, tw = ref_img.shape[:2]
        # # if src_img.shape 
        # if len(src_img.shape) < 3 :
        #     dst = cv2.cvtColor(src_img, cv2.COLOR_GRAY2BGR)
        # else :
        #     dst = src_img
        top_left = max_loc
        # bottom_right = (top_left[0] + tw, top_left[1] + th)
        # dst = dst[top_left[1]-thresh:top_left[1]+th+thresh,top_left[0]-thresh:top_left[0]+tw+thresh]
        # cv2.rectangle(dst, top_left, bottom_right, (0,0,255),2)
        print(f"score : {max_val}")
        return top_left,max_val, res  
    
    
class Countour() :
    def blob(img,min,max,mode) :
        # img2 = Tool.Inverse(img)
        if len(img.shape) != 3 :
            rgbimg  = Tool.GrayToRGB(img)
        else :
            img = Tool.RGBToGray(img)
        contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # contours = imutils.grab_contours(contours)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        contoursize = len(contours)
        x = [0 for i in range(contoursize)]
        y = [0 for i in range(contoursize)]
        w = [0 for i in range(contoursize)]
        h = [0 for i in range(contoursize)]
        ContourArea = [0 for i in range(contoursize)]
        filtered_area = []
        filtered_area2 = []
        for i in range(contoursize) :
            ContourArea[i] = cv2.contourArea(contours[i])
            x[i], y[i] , w[i] , h[i] = cv2.boundingRect(contours[i])
            if min < ContourArea[i] and ContourArea[i] < max and mode == 'area' :
                cv2.rectangle(rgbimg,(x[i], y[i]) , (x[i]+w[i] , y[i]+h[i]),(0,0,255),3)
                filtered_area.append([x[i], y[i] , w[i] , h[i]])
                print("Area : ",ContourArea[i])
            elif min < h[i] and h[i] < max and mode == 'h' :
                cv2.rectangle(rgbimg,(x[i], y[i]) , (x[i]+w[i] , y[i]+h[i]),(0,0,255),3)
                filtered_area.append([x[i], y[i] , w[i] , h[i]])  
            elif min < w[i] and w[i] < max and mode == 'w' :
                cv2.rectangle(rgbimg,(x[i], y[i]) , (x[i]+w[i] , y[i]+h[i]),(0,0,255),3)
                filtered_area.append([x[i], y[i] , w[i] , h[i]])  
                
        print(f"Count : {len(filtered_area)}")
        Count = len(filtered_area)
        return rgbimg,Count,filtered_area


    
    