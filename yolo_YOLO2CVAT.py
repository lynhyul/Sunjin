from xml.etree import ElementTree as ET
from xml.dom import minidom
import os
import glob

def yolo_to_cvat_polygon(yolo_annotation_dir, img_width, img_height, output_xml_file):
    # Get all text files in the directory
    yolo_annotation_files = glob.glob(os.path.join(yolo_annotation_dir, '*.txt'))
    class_ids =['Label','BC','number']

    # Create root element
    root = ET.Element('annotations')
    version = ET.SubElement(root, 'version')
    version.text = '1.1'

    meta = ET.SubElement(root, 'meta')
    job = ET.SubElement(meta, 'job')
    id_ = ET.SubElement(job, 'id')
    id_.text = '4'  # Modify as per your data
    size = ET.SubElement(job, 'size')
    size.text = str(sum([len(open(f).readlines()) for f in yolo_annotation_files]))
    mode = ET.SubElement(job, 'mode')
    mode.text = 'annotation'

    for i, yolo_annotation_file in enumerate(yolo_annotation_files):
        with open(yolo_annotation_file, 'r') as f:
            lines = f.readlines()

        # Create image
        image = ET.SubElement(root, 'image', {'id': str(i), 'name': os.path.basename(yolo_annotation_file).replace('.txt', '.jpg'), 'width': str(img_width), 'height': str(img_height)})

        for line in lines:
            line = line.strip().split()
            class_id = int(line[0])
            points = line[1:]

            # Convert YOLO polygon to pixel coordinates
            points = [str(float(p) * img_width if i % 2 == 0 else float(p) * img_height) for i, p in enumerate(points)]
            points = ';'.join([','.join(points[i:i + 2]) for i in range(0, len(points), 2)])

            # Create polygon
            polygon = ET.SubElement(image, 'polygon', {'label': f'{class_ids[class_id]}', 'source': 'manual', 'occluded': '0', 'points': points, 'z_order': '0'})

    # Save as XML file
    xmlstr = minidom.parseString(ET.tostring(root)).toprettyxml(indent='  ')
    with open(output_xml_file, 'w') as f:
        f.write(xmlstr)
            
yolo_to_cvat_polygon('D:/yolo/autolabelling/',640,640,"D:/yolo/auto_label/annotations.xml")