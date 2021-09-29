from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
import numpy as np
import scipy.io as sio
import skimage.io
import json
import math
import glob

json.encoder.FLOAT_REPR = lambda o: format(o, '.2f')
random.seed(9001)

import sys
sys.path.append('../src')
from utils import mask_generation_with_BB

class TextMaskData():
    def __init__(self, IMG_DIR, BG_DIR, name, charBB, wordBB):
      self.img_dir = IMG_DIR
      self.img_shape = None
      self.bgimg_dir = BG_DIR
      self.imname = name
      self.wordBB = wordBB
      self.charBB = charBB
      self.wordBB_ratio = None
      self.charBB_ratio = None
      self.isValid = True
      self._fix_dim()
      self._fix_BB_format()
      self._validate()
      self._get_ratio()

    def _fix_dim(self):
        """Dimension fix"""
        if len(self.wordBB.shape) == 2:
            self.wordBB = np.expand_dims(self.wordBB, axis=-1)
        if len(self.charBB.shape) == 2:
            self.charBB = np.expand_dims(self.charBB, axis=-1)

    def _get_img_dir(self):
        adir = os.path.join(self.img_dir, self.imname + '.jpg')
        if os.path.exists(adir):
            return adir
        else:
            print("image %s does not exist." % (self.imname))
            self.isValid = False
            return -1

    def _get_bgimg_dir(self):
        # Todo: change the code for bgimg
        if os.path.exists(os.path.join(self.bgimg_dir, self.imname + '.jpg')):
            return os.path.join(self.bgimg_dir, self.imname + '.jpg')
        else:
            print("background image%s does not exist." % (self.imname))
            self.isValid = False
            return -1

    def _validate(self):
        self.img_dir = self._get_img_dir()
        self.bgimg_dir = self._get_bgimg_dir()
        self.img_shape = skimage.io.imread(self.img_dir).shape

    def _get_ratio(self):
        if self.isValid:
            self.wordBB_ratio = int(self._get_mask_ratio(self.wordBB) * 100)
            self.charBB_ratio = int(self._get_mask_ratio(self.charBB) * 100)
        else:
            return -1

    def _fix_BB_format(self):
        self.wordBB = self.wordBB.astype(int)
        self.charBB = self.charBB.astype(int)

    # def isFileValid(self):
    #     try:
    #         bgimg_dir = self._get_bgimg_dir()
    #         img_dir = self._get_img_dir()
    #         _ = skimage.io.imread(bgimg_dir)
    #         self.img_shape = skimage.io.imread(img_dir).shape[:2]
    #     except:
    #         self.isValid =False

    def get_word(self, type, pos):
        assert type in ['x', 'y']
        if type=='x':
            return self.wordBB[0, pos, :]
        else :
            return self.wordBB[1, pos, :]

    def get_char(self, type, pos):
        assert type in ['x', 'y']
        if type=='x':
            return self.charBB[0, pos, :]
        else :
            return self.charBB[1, pos, :]

    def _get_mask_ratio(self, BB):
        return np.mean(mask_generation_with_BB(self.img_shape, BB))

    def get_info(self):
        return { "dir": self.img_dir, "gt_dir": self.bgimg_dir,\
                "word_bb": self.wordBB.tolist(), "char_bb": self.charBB.tolist(), \
                "word_percent": self.wordBB_ratio, "char_percent": self.charBB_ratio}

def convert_dataset():
    """Convert dataset"""

    IMG_DIR = '/media/osman/megamind/SynthText/EnsNet_dataset/real/jpg/img'
    BG_IMG_DIR = '/media/osman/megamind/SynthText/EnsNet_dataset/real/jpg/label'
    BBOX_DIR = '/media/osman/megamind/SynthText/EnsNet_dataset/real/bbox'
    DATASET_DIR = './output'
    if not os.path.exists(DATASET_DIR):
        os.mkdir(DATASET_DIR)

    all_files = glob.glob(IMG_DIR + '/*.jpg')
    print("Total %d files are found" % (len(all_files)))
    data = []
    for item in all_files:
        img_name = os.path.splitext(os.path.basename(item))[0]
        super_boxes = []
        for l in open(os.path.join(BBOX_DIR, img_name + '.txt')):
            x0, y0, x1, y1 = [int(ii) for ii in l.rstrip().split(',')[:4]]
            super_boxes.append(np.array([[x0, x1, x1, x0],
                                [y0, y0, y1, y1]]))

        super_boxes = np.asarray(super_boxes)
        # print(super_boxes.shape)
        if len(super_boxes.shape)==3:
            super_boxes = np.transpose(super_boxes, [1, 2, 0])
        item = TextMaskData(IMG_DIR, BG_IMG_DIR, img_name, np.array([]), np.array(super_boxes))
        if item.isValid:
            data.append(item)

    with open(os.path.join(DATASET_DIR, 'output.json'), 'w') as f:
        output = []
        for item in data:
            output.append(item.get_info())
        json.dump(output, f)

def main():
    convert_dataset()
    print('\nFinish!')

if __name__ == "__main__":
  main()