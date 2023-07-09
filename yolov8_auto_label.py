from ultralytics import YOLO
import IMP
import os
import cv2
import gc
import numpy as np
import torch

gc.collect()
torch.cuda.empty_cache()

flist = os.listdir("d:/yolo/test/")

def read_points_from_text_file(src,points,cls,text_file):
    label_points = []
    boxes = f'{int(cls)}'
    if int(cls) == 0 :
        color = (255,0,0)
    else :
        color = (0,0,255)
    # 각 라인에서 x, y 좌표를 읽어와서 포인트로 변환
    for i in range(0, len(points)):
        x = float(points[i][0]) * src.shape[1]
        y = float(points[i][1]) * src.shape[0]
        point = (x, y)
        label_points.append(point) 
        # 다각형 좌표를 NumPy 배열로 변환
    
    polygon = np.array(label_points, dtype=np.float32)
    
    # 다각형의 최소 영역 사각형 구하기
    rect = cv2.minAreaRect(polygon)
    
    # 사각형 좌표 반환
    box = cv2.boxPoints(rect)
    box = np.intp(box)
    
    for i in range(len(box)) :
        boxes += f' {box[i][0] / src.shape[1]} {box[i][1]/ src.shape[0]}'
        if i == len(box)-1 :
            boxes += '\n'
    text_file.write(boxes)
    if int(cls) == 0 :
        test = check_polygon_in_boundary(src,box,50)
        if test == 1 :
            cv2.drawContours(src, [box], 0, color, 5)
            return box
        else :
            return []
    else :
        cv2.drawContours(src, [box], 0, color, 5)
        return box


def calculate_angle(box1, box2, image_width, image_height):
    # Calculate the center coordinates of box1
    x1, y1, w1, h1 = box1
    box1_center_x = x1 + (w1 / 2)
    box1_center_y = y1 + (h1 / 2)

    # Calculate the center coordinates of box2
    x2, y2, w2, h2 = box2
    box2_center_x = x2 + (w2 / 2)
    box2_center_y = y2 + (h2 / 2)

    # Calculate the center coordinates of the image
    image_center_x = image_width / 2
    image_center_y = image_height / 2

    # Calculate the vectors from the image center to the box centers
    vector1 = np.array([box1_center_x - image_center_x, box1_center_y - image_center_y])
    vector2 = np.array([box2_center_x - image_center_x, box2_center_y - image_center_y])

    # Calculate the angle between the two vectors using the dot product
    dot_product = np.dot(vector1, vector2)
    magnitudes = np.linalg.norm(vector1) * np.linalg.norm(vector2)
    cos_theta = dot_product / magnitudes
    angle_radians = np.arccos(cos_theta)
    angle_degrees = np.degrees(angle_radians)

    return angle_degrees

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


def main() :
    model = YOLO("D:/yolo/07_06/weights/best.pt")  # load a pretrained model (recommended for training)
    # model = YOLO("D:/yolo/best.pt")  # load a pretrained model (recommended for training)
    IMP.Tool.create_folder('d:/yolo/test_result/')
    for file in flist :
        file_name = file
        text_file = open(f"D:/yolo/autolabelling/{file_name[:-4]}.txt",'w')
        src = f'//193.202.10.241/디지털혁신센터/디지털기술팀/01.5_스마트팩토리/03_식육라벨지검사/05_데이터/{file_name}'
        gray_path = f'd:/yolo/test/{file_name}'
        src = IMP.Tool.Image_Load(src,1)
        results = model.predict(gray_path,save=False, imgsz=640, conf=0.8,name=f'd:/yolo/Inf_result/',save_txt = False,device=0)
        for result in results :
            try :
                clss = result.boxes.cls
                mask_results = result.masks.xyn
                for i in range(len(clss)) :
                    cls = clss[i]
                    mask_result = mask_results[i]
                    box = read_points_from_text_file(src,mask_result,cls,text_file)
            except :
                print("error")
                pass
        cv2.imwrite(f"d:/yolo/test_result/{file}",src)
        # read_points_from_text_file(boxes)
    


if __name__ == '__main__':
    main()
    import yolo_YOLO2CVAT