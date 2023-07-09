from easyocr.easyocr import *
import time
import IMP
import os
import re
# from t_mat

# GPU 설정
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'


def str2number(string,result_list) :
    for s in string :
        try :
            int(s)
            result_list.append(s)
        except :
            pass
    return result_list

def get_files(path):
    file_list = []
    files = [f for f in os.listdir(path) if not f.startswith('.')]  # skip hidden file
    files.sort()
    abspath = os.path.abspath(path)
    for file in files:
        file_path = os.path.join(abspath, file)
        file_list.append(file_path)
    return file_list, len(file_list)

def number_len(text) :
    number = [c for c in text if c.isdigit()]
    return len(number)

if __name__ == '__main__':

    # # Using default model
    # reader = Reader(['ko'], gpu=True)

    # Using custom model
    start = time.time()
    reader = Reader(['ko'], gpu=True,
                    model_storage_directory='d:/VGG/',
                    user_network_directory='d:/VGG/',
                    recog_network='custom')

    
    end = time.time()
    # print("숫자 추출 알고리즘 적용 : ",result_list)
    print(round(end-start,3))
    while True :
        files, count = get_files('d:/ref_image/test/')
        if len(files) > 0 :
            try :
                for idx, file in enumerate(files):
                    result_list = []
                    start = time.time()
                    filename = os.path.basename(file)
                    img = cv2.imread(file,1)
                    # gray = IMP.Tool.RGBToGray(img)
                    # _,th = IMP.Binary.IntensityThresholding(gray,40,'otsu')
                    # gray,th = IMP.Binary.IntensityThresholding(gray,th-20,'inv')
                    # gray = IMP.Morphology.Closing(gray,3,3,1)
                    # img = cv2.resize(img,None,None,1/3,1/3)
                    allowlist = ['1','2','3','4','5','6','7','8','9','0','.']
                    result = reader.readtext(img,link_threshold=0.6,detail=1,width_ths = 0.2,low_text=0.5, text_threshold=0.7,allowlist= allowlist,add_margin=0.3)
                    for (bbox, string, confidence) in result:
                        top_left = (int(bbox[0][0]),int(bbox[0][1])) 
                        bottom_right = (int(bbox[-2][0]),int(bbox[-2][1]))
                        if number_len(string) > 1 and confidence > 0.6:
                            cv2.rectangle(img,top_left,bottom_right,(255,0,0),1)
                            cv2.putText(img, string, (int(bbox[0][0]), int(bbox[0][1])-30), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 1, cv2.LINE_AA)
                            print("filename: '%s', confidence: %.4f, string: '%s'" % (filename, confidence, string))
                        
                            # print('bbox: ', bbox)
                        # except :
                        #     pass
                    end = time.time()
                    print(round(end-start,3))
                    IMP.Tool.Show(img)
                    
                    os.remove(f'{file}')
                # print("숫자 추출 알고리즘 적용 : ",result_list)
                    
            except :
                pass
            # cv2.imwrite(file[:-4]+'_result.jpg',img)
