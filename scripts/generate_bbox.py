import glob2
import os
import cv2
import numpy as np
from skimage.measure import compare_ssim
import imutils

input_dir =  './syn_train'
output_dir = os.path.join(input_dir, 'bbox')

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

images = glob2.glob(os.path.join(input_dir, 'img', '*.png'))
print("#item: ", len(images))
DEGBUG = False

for i, item in enumerate(images):
    label_img_path = os.path.join(input_dir, 'label', os.path.basename(item))
    img = cv2.imread(item)
    label_img = cv2.imread(label_img_path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_label_img = cv2.cvtColor(label_img, cv2.COLOR_BGR2GRAY)

    diff = np.abs(gray_img.astype(np.float32) - gray_label_img.astype(np.float32))
    #diff = np.mean(diff, axis=2)
    diff_threshold = 50
    diff[diff < diff_threshold] = 1
    diff[diff >= diff_threshold] = 0
    diff = (diff * 255).astype("uint8")

    # Taking a matrix of size 5 as the kernel 
    # kernel = np.ones((5,5), np.uint8) 
    # diff = cv2.morphologyEx(diff, cv2.MORPH_OPEN,kernel, iterations=1)

    # print(diff)
    # cv2.imshow('img', gray_img)
    # cv2.imshow('label img', gray_label_img)
    # cv2.waitKey(0)

    # (score, diff) = compare_ssim(gray_img, gray_label_img, full=True)
    # diff = (diff * 255).astype("uint8")
    # print("SSIM: {}".format(score))

    # threshold the difference image, followed by finding contours to
    # obtain the regions of the two input images that differ
    thresh = cv2.threshold(diff, 0, 255,
        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # loop over the contours
    cnt_list = []
    bbox_size_threshold = 50
    for c in cnts:
        # compute the bounding box of the contour and then draw the
        # bounding box on both input images to represent where the two
        # images differ
        (x, y, w, h) = cv2.boundingRect(c)
        if w*h < bbox_size_threshold:
            continue
        cnt_list.append((x, y, x + w, y + h))
        if DEGBUG:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.rectangle(label_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
    
            # show the output images
            # cv2.imshow("Original", gray_img)
            # cv2.imshow("Modified", gray_label_img)
    if DEGBUG:
        cv2.imshow("Original", img)
        cv2.imshow("Modified", label_img)
        cv2.imshow("Diff", diff)
        cv2.imshow("Thresh", thresh)
        cv2.waitKey(0)

    with open(os.path.join(output_dir, os.path.basename(item).split('.')[0] + ".txt"), 'w') as f:
        for item in cnt_list:
            f.write(','.join([str(i) for i in item]) + '\n')


