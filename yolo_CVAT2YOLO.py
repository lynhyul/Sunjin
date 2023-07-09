import xml.etree.ElementTree as ET
from IMP import Tool

day = '06_30'
tree = ET.parse(f'd:/yolo/data/{day}/annotations.xml')
root = tree.getroot() # 해당 트리의 root를 반환

# for child in root:
#     print(child.tag, child.attrib) 

# root[0].attrib
# print(root[4][0].attrib)
# print(len(root[4]))

labels = ['Label','BC','number']

target_folder = f'd:/yolo/data/{day}/labels/'
target_folder2 = f'd:/yolo/data/{day}/images/'

Tool.create_folder(target_folder)
Tool.create_folder(target_folder2)

def normal(w,h) :
    return float(w)/640, float(h)/640


for i in range(len(root)) :
    file_name = root[i].attrib.get('name')
    if file_name is not None :
        f = open(f"{target_folder}/{file_name[:-4]}.txt",'w')
        if len(root[i]) > 0 :
            for j in range(len(root[i])) :
                if j > 0:
                    text = '\n'
                else :
                    text = ''
                label = str(labels.index(root[i][j].attrib.get('label')))
                text += label
                points = root[i][j].attrib.get('points')
                for point in points.split(';') :
                    x,y = normal(point.split(',')[0],point.split(',')[1])
                    text += f' {x} {y}'
                f.write(text)
        else :
            pass

                
        # print(points)
            # f.write(f'{label} {points}\n')
        # print(points)

    # print(points)

        
        # f.write(text+'\n')
        # text_file_path = f'd:/yolo/data/label/{file_name}.txt'
        