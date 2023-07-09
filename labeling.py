import os


image_path = ''
txt_path = ''



img_lst = os.listdir(image_path)
txt_lst = os.listdir(txt_path)


for img in img_lst :
    if img[:-4] in txt_lst :
        pass
    else :
        f = open(f'{txt_path}{img[:-4]}.txt', 'w')
        f.close()
        

