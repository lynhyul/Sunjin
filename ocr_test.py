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
                    print("높이 : ", img.shape[0])
                    if img.shape[0] < 200 :
                        ratio = 3
                    if img.shape[0] < 500 :
                        ratio = 2
                    elif img.shape[0] < 600 :
                        ratio = 1.5
                    else :
                        ratio = 1.2
                    img = cv2.resize(img,None,None,ratio,ratio)
                    block_list = ['요','히','후']
                    result = reader.readtext(img,beamWidth=1,text_threshold=0.7,ycenter_ths=0.2,low_text=0.55,add_margin=0.12 ,width_ths=0.6,height_ths=0.6,min_size=40 /ratio ,link_threshold=0.32,detail=1,blocklist=block_list)
                    for (bbox, string, confidence) in result:
                        top_left = (int(bbox[0][0]),int(bbox[0][1]))
                        bottom_right = (int(bbox[-2][0]),int(bbox[-2][1]))
                        result_list = str2number(string,result_list)
                        for i in range(2) :
                            string = re.sub(r'_\s*$', ' ', string)
                        string = re.sub(r'드\s*$', ' ', string)
                        if '교환' not in string : 
                            string = re.sub(r'는\s*$', ' ', string)
                        string = re.sub(r'트\s*$', ' ', string)
                        string = re.sub(r'암\s*$', '명', string)
                        string = re.sub(r'소\s*$', ':', string)
                        string = re.sub(r'::\s*$', ':', string)
                        if '용도' in string and len(string[string.index('용도'):]) > 3 :
                            string = string[:string.index('용도')]+'용도'
                        # if '[' in string and ']' not in string :
                        #     string = string.replace('[','')
                        # if '(' in string and ')' not in string :
                        #     string = string.replace('(','')
                        string = string.replace(' ','')
                        string = string.replace('품헤렌/','포장재질(내면)')
                        string = string.replace('리에렌','리에틸렌')
                        if '후)' in string :
                            string = '(농장명)'
                        if '주)' in string and len(string) < 6:
                            string = '(농장주)' 
                        string = string.replace('묶용','묶음')
                        string = string.replace('등회','등급')
                        string = string.replace('묶광','묶음')
                        string = string.replace('받을주','받을수')
                        string = string.replace('푸조','품종')
                        string = string.replace('겹장','겹살')
                        string = string.replace('경처도','경기도')
                        if '한주' in string :
                            string = '도축장:(주)해드림엘피씨'    
                        if len(string) == 1 and len(string) > 7 :
                            string = string.replace('.','')
                        string = string.replace('c1','')
                        if len(string) == 1 :
                            string = re.sub(r'[ㄱ-ㅎㅏ-ㅣ가-힣]\s*$', '', string)
                        if len(string) > 5 :
                            string = re.sub(r'[a-z A-Z]\s*$', '', string)
                        string = string.replace(':품',':')
                        if '폴리' not in string :
                            string = string.replace('리에틸렌','폴리에틸렌')
                        print("filename: '%s', confidence: %.4f, string: '%s'" % (filename, confidence, string))  
    
                        # print(bbox)
                        cv2.rectangle(img,top_left,bottom_right,(255,0,0),1)
                        
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
