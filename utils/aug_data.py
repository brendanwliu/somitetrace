import numpy as np
from PIL import Image
from os import listdir
import os

dir_img = 'datasets/SomiteTraceLibrary/input/last_frames/'
dir_mask = 'datasets/SomiteTraceLibrary/input/masks/'

imgs = sorted(listdir(dir_img))
skip_img = [178,179,358,359,538,539,718,719,898,899,1078,1079,1258,1259,1438,1439,1618,1619,1798,1799,1978,1979,2158,2159]

for i in range(len(imgs)):
    if i in skip_img:
        continue
    else:
        im1 = Image.open(dir_img + imgs[i]).convert("L")
        im2 = Image.open(dir_img + imgs[i+1]).convert("L")
        im3 = Image.open(dir_img + imgs[i+2]).convert("L")


        w, h = im1.size
        ima = Image.new('RGB', (w,h))
        data = zip(im1.getdata(), im2.getdata(), im3.getdata())
        ima.putdata(list(data))

        ima.save("./datasets/SomiteTraceLibrary/input/last_frames_layer/" + imgs[i])
