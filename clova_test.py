import json
import time

with open('d:/label_test/output.json', 'r') as f:
    json_data = json.load(f)



start = time.time()
# print(json_data['images'][0])
for idx,txt in enumerate(json_data['images'][0]['fields']) :
    text = json_data['images'][0]['fields'][idx]['inferText']
    confidence = json_data['images'][0]['fields'][idx]['inferConfidence']
    bbox = json_data['images'][0]['fields'][idx]['boundingPoly']['vertices']
    print(f"score : {confidence} / text : {text}")
    print(bbox[0]['x'])
    
end = time.time()
print(round(end-start,2))
# for idx,txt in enumerate(json_data) :
