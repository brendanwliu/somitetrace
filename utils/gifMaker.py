from os import listdir
import numpy as np
from glob import glob
from PIL import Image, ImageSequence
import argparse
import cv2
from torchvision.transforms import CenterCrop

def animate(dir):
    images = []
    for f in sorted(glob(dir + "*.tif")):
        images.append(CenterCrop([128,128])(Image.open(f)))
    images[0].save(dir + 'out.gif', save_all = True, append_images = images[1:], duration = 100, loop = 0)
    print("Saved!")

def get_args():
    parser = argparse.ArgumentParser(description='Get animated gif from output tifs',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--dir', metavar='d', type=str, default=None,
                        help='Directory where the output tifs are', dest='directory')

    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    animate(args.directory)