import os
import shutil
import random

p = random.randint(1,10)        ## 1/10 확률

replace_text = '\n'
text_name = 'gt_train'
text_file_path = f'd:/{text_name}.txt'
new_file = f'd:/{text_name}2.txt'
with open(text_file_path,'r',encoding='UTF8') as f:
    lines = f.readlines()
with open(new_file,'w',encoding='UTF8') as f:
    for i, l in enumerate(lines):
        check_blank = l.split('\t')[-1]
        if check_blank == replace_text and p == 5 :
            pass
        else :
            f.write(l)
# with open(text_file_path,'w') as f:
#     f.write(new_text_content)