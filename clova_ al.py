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


    label_txt = ['내용량','기한','용도','농장명','소재지','번호','도축장','제조원','보관방법','원재료명','품목']
    num_txt = ['내용량','기한','번호']

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

    string = result['words']
    for idx,txt in enumerate(string) :
        try :
            text = string[idx]['text']
            confidence = string[idx]['confidence']
            bbox = string[idx]['boundingBox']
            if idx == len(string)-1 :           ## 마지막 글자가 인쇄 해야하는 항목에 해당 될 경우 다음 글자가 비어있다고 판단
                if len([lt for lt in label_txt if lt in text]) > 0 : 
                        cv2.rectangle(img,(int(bbox[0][0]),int(bbox[0][1])),(int(bbox[2][0]),int(bbox[2][1])),(255,0,0),3)
                        print("마지막글자 에러")
            else :
                pass
            if len(before_label) > 1 and before_label != '번호' and '판매원' not in before_label:      ## before string의 길이가 1개 이상 일경우 진행
                ### 숫자 필터링 ###image.png
                for num in num_txt :
                    if len(text) > 8 : 
                        if num in text and number_check(text) != 1 :
                            cv2.rectangle(img,(int(b_bbox[0][0]),int(b_bbox[0][1])),(int(b_bbox[2][0]),int(b_bbox[2][1])),(255,0,0),3)
                            print("숫자가 포함되어야 하는데 8글자 이상이고 숫자가 없을때 검출")
                    else : 
                        if num in before_label and number_check(text) != 1 :
                            cv2.rectangle(img,(int(b_bbox[0][0]),int(b_bbox[0][1])),(int(b_bbox[2][0]),int(b_bbox[2][1])),(255,0,0),3)
                            print("숫자가 들어가야 하는 곳에 숫자검출이 안될때")
                        if '기한' in before_label and number_len(text) != 8:
                            cv2.rectangle(img,(int(b_bbox[0][0]),int(b_bbox[0][1])),(int(b_bbox[2][0]),int(b_bbox[2][1])),(255,0,0),3)
                            print("유통기한이 8글자가 아닐 때 검출")
                        for lt in label_txt :
                            if lt in text and len(before_label) :       ## 인쇄 내용없이 다음 항목이 오면 불량으로 검출
                                cv2.rectangle(img,(int(b_bbox[0][0]),int(b_bbox[0][1])),(int(b_bbox[2][0]),int(b_bbox[2][1])),(255,0,0),3)
                                print("인쇄 내용 다음에 바로 인쇄 내용이 올 때")
            else :
                pass
            ### 초기화 ###
            before_label = ''
            b_bbox = []
            
            print(f"score : {confidence} / text : {text}")
            
            ## Confidence Score가 낮으면 불량으로 검출 ##
            if confidence < 0.88 and text != '.' and text != '·' :       ## '·'으로 잘못 검출 되는 사례 제외
                cv2.rectangle(img,(int(bbox[0][0]),int(bbox[0][1])),(int(bbox[2][0]),int(bbox[2][1])),(255,0,0),3)
                print("Score가 낮아서 검출")
            else :
                pass
            if len([lt for lt in label_txt if lt in text]) > 0 : 
                before_label = text
                b_bbox = bbox
        except :
           pass
        
    cv2.imwrite("d:/result/"+file,img)
    end = time.time()
    print(round(end-start,2))
    IMP.Tool.Show(img)
    cv2.imwrite("d:/"+file,img)
    