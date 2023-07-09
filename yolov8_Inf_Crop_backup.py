from ultralytics import YOLO
import IMP
import os
import cv2
import gc
import numpy as np
import torch
import imutils
# from clova_ocr_al3 import run
import time

gc.collect()
torch.cuda.empty_cache()

flist = os.listdir("d:/yolo/test/")

result_path = f"d:/yolo/test_result_crop/"
IMP.Tool.create_folder(result_path)

def get_bbox_from_polygon(polygon):
    x_coords = [p[0] for p in polygon]
    y_coords = [p[1] for p in polygon]

    sx = min(x_coords)
    sy = min(y_coords)

    w = max(x_coords) - sx
    h = max(y_coords) - sy

    return sx, sy, w, h

def crop_image(image, polygon,th=20):
    # 이미지를 불러옵니다.
    # 다각형의 좌표를 이용해 시작점, 너비, 높이를 얻습니다.
    sx, sy, w, h = get_bbox_from_polygon(polygon)
    try :
    # 이미지를 자릅니다.
        if w > h * 2 :
            cropped_image = image[sy-th:sy+h+th, sx-th-60:sx+w+th+60]
        elif h > w * 2 :
            cropped_image = image[sy-th-60:sy+h+th+60, sx-th:sx+w+th]
        else :
            cropped_image = image[sy-th:sy+h+th, sx-th:sx+w+th]
    except :
        cropped_image = image[sy-th:sy+h+th, sx-th:sx+w+th]
    return cropped_image


def read_points_from_text_file(src,points,cls):
    label_points = []
    # 각 라인에서 x, y 좌표를 읽어와서 포인트로 변환
    for i in range(0, len(points)):
        x = float(points[i][0]) * src.shape[1]
        y = float(points[i][1]) * src.shape[0]
        point = (x, y)
        label_points.append(point) 
        # 다각형 좌표를 NumPy 배열로 변환
    
    polygon = np.array(label_points, dtype=np.float32)
    rect = cv2.minAreaRect(polygon)
    replace_box = cv2.boxPoints(rect)
    box = np.intp(replace_box)
    angle = rect[-1]
    ex  = check_polygon_in_boundary(src,box,20)
    if ex == 1 :
        return polygon,angle
    else :
        return [],0


### 라벨지와 바코드 위치 비교
def compare_bbox_positions(image,bbox1_center, bbox2_center):
    ## bbox1 = 바코드, bbox2 = 라벨지 
    bbox1_x, bbox1_y = bbox1_center
    bbox2_x, bbox2_y = bbox2_center

    if bbox2_x > bbox1_x and abs(bbox2_x-bbox1_x) > abs(bbox2_y-bbox1_y):
        position = -90
    elif bbox2_x < bbox1_x and abs(bbox2_x-bbox1_x) > abs(bbox2_y-bbox1_y):
        position = 90
    elif bbox2_y > bbox1_y and abs(bbox2_y-bbox1_y) > abs(bbox2_x-bbox1_x):
        position = 180
    else :
        position = 0

    return position

## 라벨지와 바코드 위치 비교(바코드 위치가 라벨지 오른쪽에 위치한 제품의 예외처리)
def compare_bbox_positions_except(image,bbox1_center, bbox2_center):
    ## bbox1 = 바코드, bbox2 = 라벨지 
    bbox1_x, bbox1_y = bbox1_center
    bbox2_x, bbox2_y = bbox2_center
    h,w = image.shape[:2]

    if abs(bbox2_x - bbox1_x) > 400 and h > w :
        position = -90
    elif abs(bbox2_x - bbox1_x) < 400 and h > w  :
        position = 90
    elif abs(bbox2_y - bbox1_y) > 400 and w > h :
        position = 0
    else :
        position = 180

    return position

import math

## 라벨지가 카메라 영상 외곽에 걸쳐서 촬영 된 경우 제외 처리
def check_polygon_in_boundary(image, polygon, threshold):
    # Get the dimensions of the image
    height, width = image.shape[:2]

    # Convert the polygon coordinates to contour format
    contour_polygon = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))

    # Create a mask with zeros
    mask = np.zeros((height, width), dtype=np.uint8)

    # Draw the contour polygon on the mask
    cv2.drawContours(mask, [contour_polygon], -1, 255, thickness=cv2.FILLED)

    # Calculate the distance of each point in the polygon to the image boundary
    distances = [min(point[0][0], width - point[0][0], point[0][1], height - point[0][1]) for point in contour_polygon]

    # Check if the minimum distance is below the threshold
    if min(distances) < threshold:
        return 0
    else:
        return 1

## Polygon의 시작 좌표 추정
def calculate_start_coordinates(polygon_points):
    sx = polygon_points[0][0]  # 첫 번째 점의 x 좌표를 시작점의 x 좌표로 설정
    sy = polygon_points[0][1]  # 첫 번째 점의 y 좌표를 시작점의 y 좌표로 설정
    
    return sx, sy



## Polygon의 Center 좌표 추정
def calculate_polygon_center(polygon_points):
    # Calculate the center coordinates of a polygon
    num_points = len(polygon_points)

    # Calculate the sum of x-coordinates and y-coordinates
    sum_x = sum([point[0] for point in polygon_points])
    sum_y = sum([point[1] for point in polygon_points])

    # Calculate the average of x-coordinates and y-coordinates
    center_x = sum_x / num_points
    center_y = sum_y / num_points

    return center_x, center_y

import math

def find_closest_coordinates(lst, index,th):
    if len(lst) < 2:
        return None
    
    min_distance = math.inf
    closest_coordinates = None
    
    for i in range(len(lst[0])):
        x1, y1 = lst[0][i]
        
        for j in range(len(lst[index])):
            x2, y2 = lst[index][j]
            distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            
            if abs(distance - th) < abs(min_distance - th):
                min_distance = distance
                closest_coordinates = ((x1, y1), (x2, y2))
    
    return closest_coordinates

import itertools

def calculate_distance(point1, point2):
    # 두 좌표 간의 유클리드 거리 계산
    x1, y1 = point1
    x2, y2 = point2
    distance = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
    return distance

def clahe_image(img) :
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    result = cv2.addWeighted(img, 0.7, enhanced, 0.3, 0)
    return result


def extract_positive_indices(lst,th):
    positive_indices = [index for index, value in enumerate(lst) if value >= th]
    return positive_indices

def find_closest_lists(center_lists, threshold):
    closest_distance = float('inf')
    closest_lists = []

    # 모든 가능한 조합을 비교하여 가장 근접한 두 리스트를 추출
    for list1, list2 in itertools.combinations(center_lists, 2):
        for point1 in list1:
            for point2 in list2:
                distance = calculate_distance(point1, point2)
                if distance < closest_distance:
                    closest_distance = distance
                    closest_lists = [list1, list2]

    # 최소 거리가 threshold 이내인 경우 결과 반환
    if closest_distance <= threshold:
        return closest_lists
    else:
        return None

def main() :
    model = YOLO("D:/yolo/best.pt")  # load a pretrained model (recommended for training)
    for file in flist :
        start = time.time()
        file_name = file
        src = f'//193.202.10.241/디지털혁신센터/디지털기술팀/01.5_스마트팩토리/03_식육라벨지검사/05_데이터/{file_name}'
        gray_path = f'd:/yolo/test/{file_name}'
        src = IMP.Tool.Image_Load(src,1)
        results = model.predict(gray_path,save=False, imgsz=640, conf=0.6,name=f'd:/yolo/Inf_result/',save_txt = False,device=0)
        
        # results = model.predict("D:/yolo/test/2023_06_15_08_41_42.jpg",save=False, imgsz=640, conf=0.6,name=f'd:/yolo/Inf_result/',save_txt = False,device=0)
        boxes = [[],[],[]]
        box_count = 0
        cnt = [0,0,0]
        count = 0
        centers = []
        angles = []
        starts = [[],[],[]]
        try : 
            for result in results :
                clss = result.boxes.cls
                mask_results = result.masks.xyn
                for i in range(len(clss)) :
                    cls = int(clss[i])
                    mask_result = mask_results[i]
                    polygon,a = read_points_from_text_file(src,mask_result,cls)
                    if len(polygon) > 0:
                        boxes[cls].append(polygon)
                        p = calculate_start_coordinates(boxes[cls][cnt[cls]])
                        starts[cls].append(p)
                        cnt[cls] += 1
                        if cls == 0 :
                            angles.append(a)
                            
            ### 제품이 쌓여서 라벨지나 바코드가 2개 이상 일 경우
            if sum(cnt) > 2 :
                if cnt[1] > 0 and cnt[2] > 0 :
                    boxes[1] = []
                    cnt[1] = 0
                else : 
                    index_0 = extract_positive_indices(cnt,1)[0]
                    index_1 = extract_positive_indices(cnt,1)[1]
                    index0 = starts[index_0].index(find_closest_coordinates(starts,index_0,400)[0])
                    index1 = starts[index_1].index(find_closest_coordinates(starts,index_1,400)[1])
                    boxes[0] = [boxes[0][index0]]
                    boxes[index_1] = [boxes[index_1][index1]]
            endtime = time.time()
            print("polygon 1차 가공 처리 시간 : ",round(endtime-start,3))

            image_center = (src.shape[1] // 2, src.shape[0] // 2)
            if len(angles) == 1 :
                angle = angles[0]
            elif len(angles) > 1 :
                angle = angles[index0]
            for idx,box in enumerate(boxes) :
                if cnt[idx] > 0 :
                    ## box Count
                    count += 1
                    rotation_matrix = cv2.getRotationMatrix2D(image_center, angle, 1.0)
                    # rotation_matrix = imutils.rotate_bound(image_center,angle)
                    rotated_polygon = cv2.transform(np.array([box[0]]), rotation_matrix)[0]
                    # ri,rotated_polygon = rotate_image_with_polygon(src,angle,box)
                    ## polygon 전체를 감싸는 사각형 그리기 : 좌표 4쌍으로 압축
                    rect = cv2.minAreaRect(rotated_polygon)
                    replace_box = cv2.boxPoints(rect)
                    replace_box = np.intp(replace_box)
                    ## polygon의 좌표로 센터 좌표 추정
                    p = calculate_polygon_center(replace_box)
                    centers.append(p)
                    if idx == 0 :
                        rotate_image = IMP.Tool.Rotate(src,angle)
                        # IMP.Tool.Show_Resize(rotate_image,520,520)
                        rotate_image = crop_image(rotate_image,replace_box,20)

                        # cv2.imwrite(f"{result_path}{file}",rotate_image)     
                else :
                    pass
            endtime = time.time()
            print("polygon 2차 가공 처리 시간 : ",round(endtime-start,3))
            
            if len(centers) == 2 :
                if rotate_image.shape[0] > rotate_image.shape[1] * 3 or rotate_image.shape[1] > rotate_image.shape[0] * 3 :
                    pos = compare_bbox_positions_except(rotate_image,centers[1],centers[0])
                else :
                    pos = compare_bbox_positions(rotate_image,centers[1],centers[0])
                rotate_image= imutils.rotate_bound(rotate_image,pos)
                endtime = time.time()
                print("회전 보정 처리 시간 : ",round(endtime-start,3))
                
            ## OCR
            rotate_image = clahe_image(rotate_image)
            endtime = time.time()
            print("대비 보정 처리 시간 : ",round(endtime-start,3))
            cv2.imwrite(f"{result_path}{file}",src)
            cv2.imwrite(f"{result_path}{file[:-4]}_crop.jpg",rotate_image)
            # run(f"{result_path}{file}")
        except :
            cv2.imwrite(f"{result_path}{file}",src)
            pass
        endtime = time.time()
        print("총 처리 시간 : ",round(endtime-start,3))
        # read_points_from_text_file(boxes)
    


if __name__ == '__main__':
    main()

            