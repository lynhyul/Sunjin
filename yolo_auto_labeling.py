import cv2
import numpy as np
import IMP

file_name = '2023_06_08_08_46_38'

src = f'//193.202.10.241/디지털혁신센터/디지털기술팀/01.5_스마트팩토리/03_식육라벨지검사/05_데이터/{file_name}.jpg'
gray = f'd:/yolo/data/06_14/images/{file_name}.jpg'
text_file = f"D:/yolo/Inf_result2/labels/{file_name}.txt"

src = IMP.Tool.Image_Load(src,1)
gray = IMP.Tool.Image_Load(gray,0)

def read_points_from_text_file(file_path):
    label_points = []
    name = file_path.split('/')[-1]
    new_file = open(f"d:/yolo/autolabeling/{name}",'w')
    with open(file_path, 'r') as file:
        line = file.readline()
        # 각 라인에서 x, y 좌표를 읽어와서 포인트로 변환
        coordinates = line.strip().split(' ')[1:]
        class_id = line.strip().split(' ')[0]
        if class_id == '0' :
            for i in range(0, len(coordinates), 2):
                x = int(float(coordinates[i]))
                y = int(float(coordinates[i + 1]))
                point = (x, y)
                label_points.append(point)
        
    return label_points

points = read_points_from_text_file(text_file)


def get_min_area_rect(points,src):
    # 다각형 좌표를 NumPy 배열로 변환
    polygon = np.array(points, dtype=np.float32)

    # 다각형의 최소 영역 사각형 구하기
    rect = cv2.minAreaRect(polygon)

    # 사각형 좌표 반환
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(src, [box], 0, (0,0,255), 1)
    return box
rect_coords =get_min_area_rect(points,src)

