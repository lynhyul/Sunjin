from pyclovaocr import ClovaOCR
import cv2
import time
import IMP
import os
ocr = ClovaOCR()
file_path = 'D:/ref_image/test2/'
# file_path2 = 'D:/ref_image/ref.jpg'
# img = cv2.imread(file_path)
# img = IMP.Tool.Image_Load(file_path,1)
    # reimg = cv2.resize(img,None,None,1/1.5,1/1)
    # cv2.imwrite(file_path2,img)
flist = os.listdir("d:/ref_image/test2/") 
    
for file in flist :
    img = IMP.Tool.Image_Load(file_path+file,1)
    start = time.time()

    result = ocr.run_ocr(
        image_path=file_path+file,
        language_code='ko',
        ocr_mode='general'
    )


    label_txt = ['내용량','기한','용도','농장명','소재지','번호','도축장','보관방법','부위','농장주']
    num_txt = ['내용량','기한','번호']
    string_txt = ['kg','Kg','동물','보관','g']
    double = ['부위','용도','농장명','농장주','번호']
    double_n = [0 for i in range(len(double))]

    ### 문장에 숫자가 포함되어 있는지 확인하는 함수 ###
    def number_check(text) :
        if any(c.isdigit() for c in text)  :
            return 1
    
    def number_len(text) :
        number = [c for c in text if c.isdigit()]
        return len(number)
        
    ### 초기값 설정 ###
    before_label = ''
    b_bbox = []
    nums = ''
    cut_box = []
    strings = ''
    flag = 0
    bboxs = []
    b_triger = 0
    triger_bbox = []
    n_triger = 0
    triger_nbox=[]
    dbn = 0

    
    string = result['words']
    for idx,txt in enumerate(string) :

            text = string[idx]['text']
            confidence = string[idx]['confidence']
            bbox = string[idx]['boundingBox']
            
            ## 초기값 설정 
            if len(before_label) > 0  :      ## before_label의 길이가 0이 아니고, 12글자 아래일 때
                cut_box = b_bbox
                n_string = before_label
                test = 0

            elif len(before_label)== 0 and len([lt for lt in label_txt if lt in text]) > 0 and len(text) >= 8 : ## 현재 text가 인쇄 항목에 포함되고 12글자 이상일때
                cut_box = bbox
                n_string = text
                test = 0
            
            ## 예외 처리
            if len(before_label) >= 8 and len([lt for lt in label_txt if lt in text]) > 0 and len(text) >= 8 :  ## before_label 8글자 이상이고, 현재 text도 8글자 이상이면
                cut_box = bbox
                n_string = text
                test = 0
                
            ## 마지막 글자가 인쇄 항목에 포함되고 text의 길이가 적으면 불량으로 판정
            if idx == len(string)-1 and len([lt for lt in label_txt if lt in text]) > 0 and len(text) < 12: 
                cv2.rectangle(img,(int(bbox[0][0]),int(bbox[0][1])),(int(bbox[2][0]),int(bbox[2][1])),(255,0,0),3)
                print("마지막글자 에러")
            else :
                pass
            
            ## 좌표 활용 if 문
            if len(cut_box) > 1 :
                ## 같은 (세로)줄에 해당 되며(위 아래로 20픽셀 차이), 이전의 텍스트와 다음 텍스트의 (가로) 차이가 이미지의 1/2사이즈 -40보다 작으면 실행
                if abs(cut_box[0][1] - bbox[0][1]) < 30 and abs(cut_box[2][1] - bbox[2][1]) < 30 and abs(cut_box[0][0] - bbox[0][0]) < (img.shape[1]/2 - 40) :
                    for num in num_txt :
                        if num in n_string :         ## 숫자가 포함 되어있는 인쇄 항목일경우
                            nums += text
                    if len(nums) == 0 :       ## 숫자가 포함 되어있지 않은 인쇄 항목일경우
                        strings += text
                    doubleb_str = [lt for lt in double if lt in n_string]
                    if len(doubleb_str) == 1  and len(text) > 3 :
                        double_n[double.index(doubleb_str[0])] += 1
                        bboxs.append(cut_box)
                        
                else :
                    if len(nums) < 12 and number_check(nums) != 1 and len(nums) > 0  :
                        cv2.rectangle(img,(int(cut_box[0][0]),int(cut_box[0][1])),(int(cut_box[2][0]),int(cut_box[2][1])),(255,0,0),3)
                    elif '기한' in n_string and number_len(nums) != 8 : 
                        cv2.rectangle(img,(int(cut_box[0][0]),int(cut_box[0][1])),(int(cut_box[2][0]),int(cut_box[2][1])),(255,0,0),3)
                    elif '내용량' in n_string and len([lt for lt in string_txt if lt in nums]) == 0 :
                        cv2.rectangle(img,(int(cut_box[0][0]),int(cut_box[0][1])),(int(cut_box[2][0]),int(cut_box[2][1])),(255,0,0),3)
                    elif '묶음' in n_string and number_len(nums) != 14 :
                        cv2.rectangle(img,(int(cut_box[0][0]),int(cut_box[0][1])),(int(cut_box[2][0]),int(cut_box[2][1])),(255,0,0),3)
                    elif number_len(nums) > 10 and number_check(nums) != 1 and len(nums) > 0 :       ## 품목 보고 번호
                        cv2.rectangle(img,(int(cut_box[0][0]),int(cut_box[0][1])),(int(cut_box[2][0]),int(cut_box[2][1])),(255,0,0),3)      
                    elif len(nums) == 0 and len(strings) < 4 and len(n_string) < 10 and len([lt for lt in double if lt in n_string]) == 0 :
                        cv2.rectangle(img,(int(cut_box[0][0]),int(cut_box[0][1])),(int(cut_box[2][0]),int(cut_box[2][1])),(255,0,0),3)
                    elif '!' in text or 'i' in text or '|' in text :
                        cv2.rectangle(img,(int(cut_box[0][0]),int(cut_box[0][1])),(int(cut_box[2][0]),int(cut_box[2][1])),(255,0,0),3)
                    nums = ''
                    strings = ''
                    cut_box = []
            ### 초기화 ###
            before_label = ''
            b_bbox = []
            
            print(f"score : {confidence} / text : {text}")
            
            
            # Confidence Score가 낮으면 불량으로 검출 ##
            if confidence < 0.9 and text != '.' and text != '·' :       ## '·'으로 잘못 검출 되는 사례 제외
                cv2.rectangle(img,(int(bbox[0][0]),int(bbox[0][1])),(int(bbox[2][0]),int(bbox[2][1])),(0,255,0),3)
                print("Score가 낮아서 검출")
            else :
                pass
            if len([lt for lt in label_txt if lt in text]) > 0 : 
                before_label = text
                b_bbox = bbox

    cv2.imwrite("d:/result/"+file,img)
    end = time.time()
    print(round(end-start,2))
    IMP.Tool.Show(img)
    cv2.imwrite("d:/"+file,img)
    