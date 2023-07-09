from pyclovaocr import ClovaOCR
import cv2
import time
import IMP
ocr = ClovaOCR()
file_path = 'D:/ref_image/ref_test.jpg'
file_path2 = 'D:/ref_image/ref.jpg'
# img = cv2.imread(file_path)
img = IMP.Tool.Image_Load(file_path,1)
# reimg = cv2.resize(img,None,None,1/1.5,1/1)
# cv2.imwrite(file_path2,img)

start = time.time()

result = ocr.run_ocr(
    image_path=file_path,
    language_code='ko',
    ocr_mode='general'
)


label_txt = ['내용량','기한','용도','농장명','소재지','인증번호']
label_mat = ['','20','살','농장','','동물']




before_label = ''

string = result['words']
for idx,txt in enumerate(string) :
    try :
        text = string[idx]['text']
        confidence = string[idx]['confidence']
        bbox = string[idx]['boundingBox']
        if len(before_label) > 1 :
            if len(text) == 0 :
                cv2.rectangle(img,(int(b_bbox[0][0]),int(b_bbox[0][1])),(int(b_bbox[2][0]),int(b_bbox[2][1])),(0,0,255),3)
            if '내용량' in text and any(c.isdigit() for c in text) and not all(c.isalpha() for c in text):
                pass
            else :
                cv2.rectangle(img,(int(b_bbox[0][0]),int(b_bbox[0][1])),(int(b_bbox[2][0]),int(b_bbox[2][1])),(0,0,255),3)
            if label_mat[bidx] in text :
                pass
            else :
                cv2.rectangle(img,(int(b_bbox[0][0]),int(b_bbox[0][1])),(int(b_bbox[2][0]),int(b_bbox[2][1])),(0,0,255),3)
        before_label = ''
        bidx = 0
        b_bbox = []
        print(f"score : {confidence} / text : {text}")
        if confidence < 0.8 and len(text) > 1 : 
            cv2.rectangle(img,(int(bbox[0][0]),int(bbox[0][1])),(int(bbox[2][0]),int(bbox[2][1])),(0,0,255),3)
        else :
            pass
        for lt in label_txt :
            if lt in text :
                before_label = text
                bidx = label_txt.index(lt)
                b_bbox = bbox
    except :
        pass

end = time.time()
print(round(end-start,2))
IMP.Tool.Show(img)

# cv2.imwrite(file_path[:-4]+'_test.jpg',img)



# for id,bbox, isVertical,confidence, string in result:
#     try :
#         print("filename: '%s', confidence: %.4f, string: '%s'" % ('img2', confidence, string))
#         # cv2.rectangle(img,bbox[0],bbox[-2],(255,0,0),1)
#         # print('bbox: ', bbox)
#     except :
#         pass
    
# {'id': 21, 'boundingBox': [[904, 2095], [943, 2095], [943, 2118], [904, 2118]], 'isVertical': False, 'text': '850', 'confidence': 0.4355}