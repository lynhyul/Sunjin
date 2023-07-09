import cv2
import numpy as np
import os
import random

flist = os.listdir('e:/xray/1/')
random.shuffle(flist)


for file in flist[:11000] :
    os.remove(f'e:/xray/1/{file}')
