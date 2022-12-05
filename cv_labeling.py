import numpy as np
import cv2
import json
from pathlib import Path #https://hleecaster.com/python-pathlib/
from os import listdir, path, remove

# 모니터 해상도 얻기, 원래의 DPI 고려
import ctypes
user32 = ctypes.windll.user32
user32.SetProcessDPIAware()
# ctypes.windll.user32.SetProcessDPIAware()
# width, height = user32.GetSystemMetrics(0)//2, user32.GetSystemMetrics(1)//2
width, height = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
print('monitor resolution :', width, height)

font_scale = 0
drawing = False # Mouse가 클릭된 상태 확인용
mode = True # True 면 사각형, False 면 폴리곤
nclick = 0 # Mouse 클릭횟수
gx, gy = [], [] # 도형 그리기
GX, GY = [], [] # 모든 도형의 좌표정보 저장
pimg = [] # 클릭하기 전/후 이미지 저장
img = []
classes = [] # 라벨링 클래스
CLASSES = [] # 라벨링 한 모든 클래스 인덱스 정보 저장
iclass = 0 # 라벨링할 클래스 인덱스
iobject = -1 # object(labelled region) 인덱스
event_mouse_wheel = 0 # 마우스 휠 : +1 up, -1 down, 0 no event

#For coco panoptic annotaion
anno_images = {}
anno_instances = {}
# categories = []

def init(x, y, gx, gy, GX, GY, CLASSES, iclass, mode) : 
    #print('class_name :', class_name)
    #if mode == True : 
    if True:
        gx.append(x)
        gy.append(y)
    GX.append(gx)
    GY.append(gy)
    CLASSES.append(iclass)
    return [], [], False, 0

def limit_xy(x, limit) : 
    return (x<0)*0 + (0<=x<limit)*x + (limit<=x)*(limit-1)

# x, y 좌표 정보 별로 sorting
def check_xy(img, X, Y) : 
    x1, y1, x2, y2 = X[0], Y[0], X[1], Y[1]
    xy = [x1, y1, x2, y2]
    h, w = img.shape[:2]
    limit = [w, h, w, h]
    for i in range(4) : 
        xy[i] = limit_xy(xy[i], limit[i])
    X2, Y2 = list(np.sort([xy[0], xy[2]])), list(np.sort([xy[1], xy[3]]))
    return X2, Y2

def xy_to_normxy(img, X, Y) : 
    height, width = img.shape[:2]
    X2, Y2 = [], []
    #X, Y = check_xy(img, X, Y)
    if len(X)==2 : 
        X, Y = check_xy(img, X, Y)
        x1, y1, x2, y2 = X[0], Y[0], X[1], Y[1]
        cx, cy, w, h = (x1+x2)/2./width, (y1+y2)/2./height, (x2-x1)/width, (y2-y1)/height
        X2 = [cx, w]
        Y2 = [cy, h]
        return X2, Y2
    elif len(X)>=3 : 
        X2 = np.array(X)/width
        Y2 = np.array(Y)/height
    return X2, Y2

def normxy_to_xy(img, X, Y) : 
    height, width = img.shape[:2]
    X2, Y2 = [], []
    if len(X)==2 : 
        cx, cy, w, h = X[0], Y[0], X[1], Y[1]
        #x1, x2, y1, y2 = int((cx-w/2)*width), int((cx+w/2)*width), int((cy-h/2)*height), int((cy+h/2)*height)
        x1, x2, y1, y2 = round((cx-w/2)*width), round((cx+w/2)*width), round((cy-h/2)*height), round((cy+h/2)*height)
        X2 = [x1, x2]
        Y2 = [y1, y2]
        X2, Y2 = check_xy(img, X2, Y2)
        return X2, Y2
    elif len(X)>=3 : 
        #X2 = (np.array(X)*width).astype('int')
        #Y2 = (np.array(Y)*height).astype('int')
        X2 = np.round(np.array(X)*width, 0).astype('int')
        Y2 = np.round(np.array(Y)*height, 0).astype('int')
    return X2, Y2

# Annotation 정보를 파일에 저장
def file_anno_writing(img, file_img_name, file_anno_name, GX, GY, CLASSES, img_h, img_w, anno_images, anno_instances) : 
    #print('GX, GY :', GX, GY)
    img_h2, img_w2 = img.shape[:2]
    if len(GX)>=1 and len(GY)>=1 : 
        file_name = Path(file_img_name).name
        file_img_name_temp = file_img_name.replace('\\','/')
        anno_images[file_img_name] = {'file_name':file_name, 'height':img_h, 'width':img_w, 'id':file_img_name_temp}
        #print('anno_images :', anno_images)
        all_seg_in_img = []
        for i in range(len(GX)) : 
            one_seg = {}
            #print('i', i, ':', GX[i], GY[i])
            GX2 = np.round(np.array(GX[i])/img_w2*img_w, 6)
            GY2 = np.round(np.array(GY[i])/img_h2*img_h, 6)
            XY = np.zeros(len(GX[i])*2)
            XY[::2] = GX2
            XY[1::2] = GY2
            XY = [list(XY)]
            xmin, xmax = np.min(GX2), np.max(GX2)
            ymin, ymax = np.min(GY2), np.max(GY2)
            bbox = [xmin, ymin, xmax, ymax]
            #bbox = [(xmax-xmin)/2, (ymax-ymin)/2, xmax-xmin, ymax-ymin]
            
            # https://detectron2.readthedocs.io/en/latest/tutorials/datasets.html
            one_seg['segmentation'] = XY
            if len(GX[i])==2:
                XY = [[xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax]]
                one_seg['segmentation'] = XY
                bbox2 = np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]])
                bbox2 = np.round(bbox2, 0).astype('int')
                one_seg['area'] = int(cv2.contourArea(bbox2))
            elif len(GX[i])>=3:
                XY2 = np.array(XY[0]).reshape(-1,2)
                XY2 = np.round(XY2, 0).astype('int')
                one_seg['area'] = int(cv2.contourArea(XY2))
            one_seg['iscrowd'] = 0
            one_seg['image_id'] = file_img_name_temp
            one_seg['bbox'] = bbox
            one_seg['bbox_mode'] = 0
            one_seg['category_id'] = CLASSES[i]
            one_seg['id'] = file_img_name_temp + '_%d'%i
            all_seg_in_img.append(one_seg)
        anno_instances[file_img_name] = all_seg_in_img
        
        file_anno = open(file_anno_name,'w')
        txt = ''
        for i in range(len(GX)) : 
            #print(i, GX[i], GY[i])
            txt = txt + str(CLASSES[i])
            GX[i], GY[i] = xy_to_normxy(img, GX[i], GY[i])
            for j in range(len(GX[i])) : 
                txt = txt + ' %.7f %.7f'%(GX[i][j], GY[i][j])
            txt = txt + '\n'
        file_anno.write(txt)
        file_anno.close()

# Annotation 파일 불러오기/시각화
def file_anno_loading(img, files, file_index, file_anno_name, class_names, iclass, font_scale) : 
    #file_name = file_anno_name.split('/')[-1]
    #file_name = file_anno_name.split('\\')[-1]
    file_name = Path(file_anno_name).name
    height, width = img.shape[:2]
    cv2.putText(img, '(%d/%d) %s'%(file_index+1, len(files), file_name[:-4]), (10,height-round(7*font_scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.6*font_scale, (0,0,0), round(3*font_scale))
    cv2.putText(img, '(%d/%d) %s'%(file_index+1, len(files), file_name[:-4]), (10,height-round(7*font_scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.6*font_scale, (255,255,255), round(2*font_scale))
    pimg = [img.copy()]
    gx, gy, GX, GY, CLASSES = [], [], [], [], []
    file_anno = None
    if path.isfile(file_anno_name) == True : 
        file_anno = open(file_anno_name,'r')
        lines = file_anno.readlines()
        for i in lines : 
            #xy = i[2:-1].split()
            info = i.split()
            class_index = int(info[0])
            class_name = class_names[class_index]
            xy = info[1:] # 클래스 제외 좌표정보만 선택
            ix, iy = xy[::2], xy[1::2]
            ix = [float(i) for i in ix]
            iy = [float(i) for i in iy]
            ix, iy = normxy_to_xy(img, ix, iy)
            if len(ix)==2 : 
                cv2.rectangle(img, (ix[0], iy[0]), (ix[1], iy[1]), (0,255,0), 2)
                cv2.putText(img, class_name, (ix[0], iy[0]-round(4*font_scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.6*font_scale, (0,0,0), round(2*font_scale))
                cv2.putText(img, class_name, (ix[0], iy[0]-round(4*font_scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.6*font_scale, (255,255,255), round(1*font_scale))
            elif len(ix)>=3 : 
                pts = list(zip(ix, iy))
                pts = np.array(pts).reshape(-1,1,2)
                cv2.polylines(img, [pts], True, (0,255,0), 2)
                cv2.putText(img, class_name, (ix[0], iy[0]-round(4*font_scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.6*font_scale, (0,0,0), round(2*font_scale))
                cv2.putText(img, class_name, (ix[0], iy[0]-round(4*font_scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.6*font_scale, (255,255,255), round(1*font_scale))
            GX.append(ix)
            GY.append(iy)
            CLASSES.append(class_index)
        file_anno.close()
        pimg.append(img.copy())
    return pimg, img, gx, gy, GX, GY, CLASSES

def box_scan(pimg, img, GX, GY, x, y, masking=True) : 
    idx_fire = -1
    d = 1
    if len(pimg)>=1 : 
        img = pimg[-1].copy()
    for i in range(len(GX)) : 
        if len(GX[i])==2:
            x1, y1, x2, y2 = GX[i][0], GY[i][0], GX[i][1], GY[i][1]
            #if ((x1<=x<=x2) and (y==y1 or y==y2)) or ((y1<=y<=y2) and (x==x1 or x==x2)) : 
            if ((x1-d<=x<=x2+d) and (y1-d<=y<=y1+d or y2-d<=y<=y2+d)) or ((y1-d<=y<=y2+d) and (x1-d<=x<=x1+d or x2-d<=x<=x2+d)) : 
                if masking:
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0,0,255), 2)
                idx_fire = i
                break
    return img, idx_fire

# 마우스 포인터가 labelling 영역 안에 있는지 감지
def labelling_region_scan(pimg, img, GX, GY, x, y, masking=True):
    #if len(pimg)>=1 : 
    #    img = pimg[-1].copy()
    idx_fire = -1
    masking_ratio = 0.25
    for i in range(len(GX)):
        if len(GX[i])==2:
            x1, y1, x2, y2 = GX[i][0], GY[i][0], GX[i][1], GY[i][1]
            d = 1
            if x1+d<x<x2-d and y1+d<y<y2-d:
                idx_fire = i
                if masking:
                    img[y1+d:y2-d, x1+d:x2-d, :] = (img[y1+d:y2-d, x1+d:x2-d, :]*(1-masking_ratio) + np.array([0,255*masking_ratio,0])).astype('int')
                break
        else:
            # https://stackoverflow.com/questions/13786088/determine-if-a-point-is-inside-or-outside-of-a-shape-with-opencv
            cnt = np.array(list(zip(GX[i], GY[i])))
            index2 = cv2.pointPolygonTest(cnt, [x,y], False)
            if index2==1:
                idx_fire = i
                if masking:
                    img_temp = np.zeros(img.shape[:2])
                    cv2.drawContours(img_temp, [cnt], 0, 1, -1)
                    cut = img_temp==1
                    img[cut] = (img[cut]*(1-masking_ratio) + np.array([0,255*masking_ratio,0])).astype('int')
                break
    return img, idx_fire

def img_loading(dir_data, files, file_index, img_scale, class_names, iclass, font_scale) : 
    # img = cv2.imread(file)
    # 그림파일 가져오기, cv2.imread 한글경로 오류 해결
    img_array = np.fromfile(path.join(dir_data, files[file_index]), np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    img_h, img_w = img.shape[:2]
    img = cv2.resize(img, dsize=(0,0), fx=img_scale, fy=img_scale, interpolation=cv2.INTER_AREA)
    file_anno_name = path.join(dir_data, files[file_index][:-3] + 'txt')
    pimg, img, gx, gy, GX, GY, CLASSES = file_anno_loading(img, files, file_index, file_anno_name, class_names, iclass, font_scale)
    return pimg, img, gx, gy, GX, GY, CLASSES, img_h, img_w

def draw_label(pimg, iobject, GX, GY, class_names, CLASSES, font_scale):
    img = pimg[0].copy()
    for i in range(len(GX)) : 
        if len(GX[i])==2 : 
            cv2.rectangle(img, (GX[i][0],GY[i][0]), (GX[i][1],GY[i][1]), (0,255,0), 2)
            cv2.putText(img, class_names[CLASSES[i]], (GX[i][0],GY[i][0]-round(4*font_scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.6*font_scale, (0,0,0), round(2*font_scale))
            cv2.putText(img, class_names[CLASSES[i]], (GX[i][0],GY[i][0]-round(4*font_scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.6*font_scale, (255,255,255), round(1*font_scale))
        elif len(GX[i])>=3 : 
            pts = list(zip(GX[i], GY[i]))
            pts = np.array(pts).reshape(-1,1,2)
            cv2.polylines(img, [pts], True, (0,255,0), 2)
            cv2.putText(img, class_names[CLASSES[i]], (GX[i][0],GY[i][0]-round(4*font_scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.6*font_scale, (0,0,0), round(2*font_scale))
            cv2.putText(img, class_names[CLASSES[i]], (GX[i][0],GY[i][0]-round(4*font_scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.6*font_scale, (255,255,255), round(1*font_scale))
    pimg = [pimg[0]] + [img.copy()]
    return pimg, img

# callback함수
def draw_circle(event, x, y, flags, param):
    global drawing, mode, nclick, gx, gy, pimg, img, GX, GY, CLASSES, iclass, class_names, font_scale, iobject, event_mouse_wheel
    if event == cv2.EVENT_LBUTTONDOWN: #마우스 왼버튼 누른 상태
        img = pimg[-1].copy()
        if mode==False  and nclick>=1 : 
            cv2.line(img, (gx[-1], gy[-1]), (x, y), (0,255,0), 2)
        pimg.append(img.copy())
        gx.append(x)
        gy.append(y)
        drawing = True
        nclick = nclick + 1
        #print('마우스 왼쪽버튼 누름')
    elif event == cv2.EVENT_MOUSEMOVE: #마우스 이동중
        if drawing == True : 
            img = pimg[-1].copy()
            if mode==True : 
                cv2.rectangle(img, (gx[-1], gy[-1]), (x, y), (0,0,255), 1)
            else : 
                cv2.line(img, (gx[-1], gy[-1]), (x, y), (0,0,255), 1)
        else : 
            #img = pimg[-1].copy()
            #cv2.line(img, (x, 0), (x, img.shape[0]-1), (255,255,255), 1)
            #cv2.line(img, (0, y), (img.shape[1]-1, y), (255,255,255), 1)
            #if mode==True  : 
            if True:
                img, iobject = box_scan(pimg, img, GX, GY, x, y) # 도형 외곽선 감지
                # 마우스 포인터에 수직 교차선 추가
                cv2.line(img, (x, 0), (x, img.shape[0]-1), (255,255,255), 1)
                cv2.line(img, (0, y), (img.shape[1]-1, y), (255,255,255), 1)
            img, iobject = labelling_region_scan(pimg, img, GX, GY, x, y)
    elif event == cv2.EVENT_LBUTTONUP: #마우스 왼버튼 눌렀다가 뗀 상태
        if mode==True and drawing==True : 
            cv2.rectangle(img, (gx[-1], gy[-1]), (x, y), (0,255,0), 2)
            cv2.putText(img, class_names[iclass], (gx[-1], gy[-1]-round(4*font_scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.6*font_scale, (0,0,0), round(2*font_scale))
            cv2.putText(img, class_names[iclass], (gx[-1], gy[-1]-round(4*font_scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.6*font_scale, (255,255,255), round(1*font_scale))
            gx, gy, drawing, nclick = init(x, y, gx, gy, GX, GY, CLASSES, iclass, mode)
            pimg.append(img.copy())
        #print('마우스 왼쪽버튼 뗌')
    elif event == cv2.EVENT_RBUTTONDOWN: #마우스 우측버튼 누른 상태
        #if mode==False : 
        if mode==False and drawing==True: 
            img = pimg[-1].copy()
            cv2.line(img, (gx[0], gy[0]), (gx[-1], gy[-1]), (0,255,0), 2)
            cv2.putText(img, class_names[iclass], (gx[0], gy[0]-round(4*font_scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.6*font_scale, (0,0,0), round(2*font_scale))
            cv2.putText(img, class_names[iclass], (gx[0], gy[0]-round(4*font_scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.6*font_scale, (255,255,255), round(1*font_scale))
            gx, gy, drawing, nclick = init(x, y, gx, gy, GX, GY, CLASSES, iclass, mode)
            pimg.append(img.copy())

        # 마우스 우클릭시 도형 제거
        #_, iobject = box_scan(pimg, img, GX, GY, x, y)
        img_temp, iobject = labelling_region_scan(pimg, img, GX, GY, x, y)
        if iobject>=0 : 
            del GX[iobject]
            del GY[iobject]
            del CLASSES[iobject]
            pimg, img = draw_label(pimg, iobject, GX, GY, class_names, CLASSES, font_scale)
            gx, gy, drawing, nclick = [], [], False, 0
        #print('마우스 오른쪽버튼 누름')
    elif event == cv2.EVENT_MOUSEWHEEL:
        #print('mouse wheel')
        #print('flags :', flags)
        if flags>0:
            event_mouse_wheel = 1
            #print('up')
        elif flags<0:
            event_mouse_wheel = -1
            #print('down')

def run(dir_data, img_scale, file_index=1, class_names=['0', '1']) : 
    global drawing, mode, nclick, gx, gy, pimg, img, GX, GY, CLASSES, iclass, font_scale, iobject, event_mouse_wheel
    # data 폴더에 있는 그림파일들 목록 가져오기
    files = listdir(dir_data)
    files = [i for i in files if i[-3:]=='png' or i[-3:]=='jpg']
    #file_index -= 1
    file_index = np.max((0, file_index-1))
    if file_index>=len(files) or file_index<0 : 
        print('file_index error!!!')
        return 0

    pimg, img, gx, gy, GX, GY, CLASSES, img_h, img_w = img_loading(dir_data, files, file_index, img_scale, class_names, iclass, font_scale)
    img2 = img.copy()
    
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', draw_circle)
    
    key_map = {}
    for i in range(1,min(len(class_names)+1, 11)):
        key_map[ord('%s'%i)] = i-1

    while(1):
        try : 
            file_img_name = path.join(dir_data, files[file_index])
            file_anno_name = path.join(dir_data, files[file_index][:-3] + 'txt')

            k = cv2.waitKey(1) & 0xFF
            if k == 27 : # esc를 누르면 종료
                cv2.imwrite('test.png', img2)
                file_anno_writing(img, file_img_name, file_anno_name, GX, GY, CLASSES, img_h, img_w, anno_images, anno_instances)
                print('last file_index :', file_index+1)
                categories = []
                for i in range(len(class_names)):
                    categories.append({'supercategory':'cow',
                                       'id':i,
                                       'name':class_names[i]
                                      })
                anno_instances2 = list(anno_instances.values())
                anno_instances3 = []
                for i in anno_instances2:
                    anno_instances3 += i
                #temp = [i['id']=n for n,i in enumerate(anno_instances3)]
                for n,i in enumerate(anno_instances3):
                    i['id'] = n
                panoptic_anno = {'images':list(anno_images.values()), 
                                 'annotations':anno_instances3, 
                                 'categories':categories
                                }
                #with open(path.join(dir_data, 'panoptic_annotation.json'),'w') as file_json:
                with open(path.join(dir_data, 'instance_annotation.json'),'w') as file_json:
                    #print('panoptic_anno :', panoptic_anno)
                    #re_json = json.dumps(panoptic_anno, indent=2, ensure_ascii=False)
                    re_json = json.dumps(panoptic_anno, indent=2)
                    #print('\n\n')
                    #print('re_json :', re_json)
                    file_json.write(re_json)
                    #print('anno_instances :', list(anno_instances.values()))
                    #print('categories :', categories)
                print('categories :', categories)
                break
            elif k == ord('q') : # 이미지 및 annotation 파일 삭제
                file_anno_name2 = file_img_name[:-3] + 'xml'
                for file in [file_img_name, file_anno_name, file_anno_name2] : 
                    if path.isfile(file) : 
                        remove(file)
                del files[file_index]
                pimg, img, gx, gy, GX, GY, CLASSES, img_h, img_w = img_loading(dir_data, files, file_index, img_scale, class_names, iclass, font_scale)
            elif k == ord('m') : # mode 바꾸기
                mode = not mode
                img = pimg[-1].copy()
            elif k == ord('d') or k == ord('a') : # 이전/다음 이미지 불러오기
                file_anno_writing(img, file_img_name, file_anno_name, GX, GY, CLASSES, img_h, img_w, anno_images, anno_instances)
                if k == ord('d') : 
                    file_index = (file_index+1)%len(files)
                elif k == ord('a') : 
                    file_index = file_index - 1
                    file_index = np.min((file_index, file_index+len(files)))%len(files)
                pimg, img, gx, gy, GX, GY, CLASSES, img_h, img_w = img_loading(dir_data, files, file_index, img_scale, class_names, iclass, font_scale)
            elif k == ord('e') : # 이전 도형 지우기
                if len(pimg)>=3 : 
                    n_point = len(GX[-1]) + (not mode)*1
                    img = pimg[-n_point]
                    pimg = pimg[:-n_point]
                    if (len(pimg)%2==1 and len(pimg)>=1) or (len(pimg)%2==0 and len(pimg)>=2) : 
                        GX.pop()
                        GY.pop()
                        CLASSES.pop()
            elif k == ord('c') : # 모든 도형 지우기
                if len(pimg)>=1 : 
                    img = pimg[0].copy()
                    if path.isfile(file_anno_name) : 
                        remove(file_anno_name)
                    pimg, img, gx, gy, GX, GY, CLASSES = file_anno_loading(img, files, file_index, file_anno_name, class_names, iclass, font_scale)
            elif k in key_map:
                #print('iclass : before', iclass, ', after', key_map[k])
                iclass = key_map[k]
                #print('iclass :', iclass)
                if 0 <= iobject < len(CLASSES) : 
                    #print(iobject, len(CLASSES), CLASSES)
                    CLASSES[iobject] = iclass
                    pimg, img = draw_label(pimg, iobject, GX, GY, class_names, CLASSES, font_scale)
                    drawing, nclick = False, 0
                    
            if event_mouse_wheel!=0:
                scale = 0.05
                if event_mouse_wheel==1:
                    img_scale *= 1.0 + scale
                    font_scale *= 1.0 + scale
                elif event_mouse_wheel==-1:
                    img_scale *= 1.0 - scale
                    font_scale *= 1.0 - scale
                #print(img_scale, font_scale)
                file_anno_writing(img, file_img_name, file_anno_name, GX, GY, CLASSES, img_h, img_w, anno_images, anno_instances)
                pimg, img, gx, gy, GX, GY, CLASSES, img_h, img_w = img_loading(dir_data, files, file_index, img_scale, class_names, iclass, font_scale)
                event_mouse_wheel = 0

            #cv2.imshow('image', img)
            img2 = img.copy()
            cv2.putText(img2, 'class : %s'%(class_names[iclass]), (10,round(30*font_scale)), cv2.FONT_HERSHEY_SIMPLEX, 1*font_scale, (0,0,0), round(3*font_scale))
            cv2.putText(img2, 'class : %s'%(class_names[iclass]), (10,round(30*font_scale)), cv2.FONT_HERSHEY_SIMPLEX, 1*font_scale, (255,255,255), round(2*font_scale))
            cv2.imshow('image', img2)

        except Exception as e : 
            print(str(e))
            break

    cv2.destroyAllWindows()
    return file_index+1

# dir_data = r'\\193.202.10.241\디지털혁신센터\디지털기술팀\01.4_스마트팜\02_축우섭취량모니터링시스템\3.2 Label\220725_우농목장_이미지_선별\6월16일 한우\01010001952000000'

# dir_data = r'C:\python\code\yolov7\img\unong_hanwoo'
# dir_data = r'C:\python\code\yolov7\img\unong_holstein'
# dir_data = r'C:\python\code\yolov7\img\unong_train'
# dir_data = r'C:\python\code\yolov7\img\unong_val'
dir_data = r'C:\python\code\yolov7\img\labelling_test'

img_scale = 1.35
font_scale = 3

file_index = 1 # 첫번째로 오픈할 그림파일 인덱스 (1부터 시작)
# class_names = ['standing', 'lying', 'bulky_feed', 'TMR']
# class_names = ['standing', 'lying', 'bulky_feed']
class_names = ['standing', 'lying', 'feedL', 'feedM', 'feedS']
# super_category = ['cow', 'cow', 'feed']

# dir_data = r'C:\python\code\Computer_Vision\data'
# font_scale = 3
# scale = 0.5
# width, height = 960, 540
# class_names = ['cow', 'cow']

file_index = run(dir_data, img_scale, file_index, class_names)
