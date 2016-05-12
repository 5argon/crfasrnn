# -*- coding: utf-8 -*-
"""
This package contains code for the "CRF-RNN" semantic image segmentation method, published in the 
ICCV 2015 paper Conditional Random Fields as Recurrent Neural Networks. Our software is built on 
top of the Caffe deep learning library.
 
Contact:
Shuai Zheng (szheng@robots.ox.ac.uk), Sadeep Jayasumana (sadeep@robots.ox.ac.uk), Bernardino Romera-Paredes (bernard@robots.ox.ac.uk)

Supervisor: 
Philip Torr (philip.torr@eng.ox.ac.uk)

For more information about CRF-RNN, please vist the project website http://crfasrnn.torr.vision.
"""

caffe_root = '../caffe/'
import sys
sys.path.insert(0, caffe_root + 'python')

import os
import cPickle
import logging
import numpy as np
import pandas as pd
from PIL import Image as PILImage
#import Image
import cStringIO as StringIO
import caffe
import matplotlib.pyplot as plt
import sys
import time


MODEL_FILE = 'TVG_CRFRNN_new_deploy.prototxt'
PRETRAINED = 'TVG_CRFRNN_COCO_VOC.caffemodel'
IMAGE_FILE = 'input.jpg'

if len(sys.argv) != 4:
    print("How to use : python batchDetect.py folder imageNamePrefix numberOfImages")
    sys.exit()

#caffe.set_mode_gpu()
#caffe.set_device(1)
net = caffe.Segmenter(MODEL_FILE, PRETRAINED)


numberOfImages = int(sys.argv[3])
for i in range(0,numberOfImages):
    processing = "./" + sys.argv[1] + "/" + sys.argv[2] + str(i).zfill(4) + ".jpg"
    print processing;

    input_image = 255 * caffe.io.load_image(processing)

    width = input_image.shape[0]
    height = input_image.shape[1]
    maxDim = max(width,height)

    image = PILImage.fromarray(np.uint8(input_image))
    image = np.array(image)

    pallete = [0,0,0,
                0,0,0,
                0,0,0,
                0,0,0,
                0,0,0,
                0,0,0,
                0,0,0,
                0,0,0,
                0,0,0,
                0,0,0,
                0,0,0,
                0,0,0,
                0,0,0,
                0,0,0,
                0,0,0,
                255,255,255,
                0,0,0,
                0,0,0,
                0,0,0,
                0,0,0,
                0,0,0,
                0,0,0,
                0,0,0,
                0,0,0,
                0,0,0,
                0,0,0,
                0,0,0,
                0,0,0]

    mean_vec = np.array([103.939, 116.779, 123.68], dtype=np.float32)
    reshaped_mean_vec = mean_vec.reshape(1, 1, 3);

# Rearrange channels to form BGR
    im = image[:,:,::-1]
# Subtract mean
    im = im - reshaped_mean_vec

# Pad as necessary
    cur_h, cur_w, cur_c = im.shape
    pad_h = 500 - cur_h
    pad_w = 500 - cur_w
    im = np.pad(im, pad_width=((0, pad_h), (0, pad_w), (0, 0)), mode = 'constant', constant_values = 0)
# Get predictions
    t0 = time.time()

    segmentation = net.predict([im])

    t1 = time.time()

    print("Time elapsed : " + str(t1-t0))
    segmentation2 = segmentation[0:cur_h, 0:cur_w]
    output_im = PILImage.fromarray(segmentation2)
    output_im.putpalette(pallete)

    plt.imshow(output_im)
    plt.axis('off')
    plt.savefig('./' + sys.argv[1] + '/detected' + str(i).zfill(4) + '.png')
