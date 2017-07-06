# ----------------------------------------------------------------------------
# Copyright 2017 Nervana Systems Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------

"""
Example based on CNTK_201A_CIFAR-10_DataLoader tutorial.
"""
from __future__ import division, print_function

import os
import pickle as cp
import sys
import tarfile
import xml.dom.minidom
import xml.etree.cElementTree as et

import numpy as np
from PIL import Image

try:
    from urllib.request import urlretrieve
except ImportError:
    from urllib import urlretrieve


def readBatch(src):
    with open(src, 'rb') as f:
        if sys.version_info[0] < 3:
            d = cp.load(f)
        else:
            d = cp.load(f, encoding='latin1')
        data = d['data']
        feat = data
    res = np.hstack((feat, np.reshape(d['labels'], (len(d['labels']), 1))))
    return res.astype(np.int)


def loadData(src):
    print('Downloading ' + src)
    fname, h = urlretrieve(src, './delete.me')
    print('Done.')
    try:
        print('Extracting files...')
        with tarfile.open(fname) as tar:
            tar.extractall()
        print('Done.')
        print('Preparing train set...')
        trn = np.empty((0, numFeature + 1), dtype=np.int)
        for i in range(5):
            batchName = './cifar-10-batches-py/data_batch_{0}'.format(i + 1)
            trn = np.vstack((trn, readBatch(batchName)))
        print('Done.')
        print('Preparing test set...')
        tst = readBatch('./cifar-10-batches-py/test_batch')
        print('Done.')
    finally:
        os.remove(fname)
    return (trn, tst)


def saveTxt(filename, ndarray):
    with open(filename, 'w') as f:
        labels = list(map(' '.join, np.eye(10, dtype=np.uint).astype(str)))
        for row in ndarray:
            row_str = row.astype(str)
            label_str = labels[row[-1]]
            feature_str = ' '.join(row_str[:-1])
            f.write('|labels {} |features {}\n'.format(label_str, feature_str))


def saveImage(fname, data, label, mapFile, regrFile, pad, **key_parms):
    # data in CIFAR-10 dataset is in CHW format.
    pixData = data.reshape((3, imgSize, imgSize))
    if ('mean' in key_parms):
        key_parms['mean'] += pixData

    if pad > 0:
        pixData = np.pad(
            pixData, ((0, 0), (pad, pad), (pad, pad)), mode='constant', constant_values=128
        )

    img = Image.new('RGB', (imgSize + 2 * pad, imgSize + 2 * pad))
    pixels = img.load()
    for x in range(img.size[0]):
        for y in range(img.size[1]):
            pixels[x, y] = (pixData[0][y][x], pixData[1][y][x], pixData[2][y][x])
    img.save(fname)
    mapFile.write("%s\t%d\n" % (fname, label))

    # compute per channel mean and store for regression example
    channelMean = np.mean(pixData, axis=(1, 2))
    regrFile.write(
        "|regrLabels\t%f\t%f\t%f\n" % (
            channelMean[0] / 255.0,
            channelMean[1] / 255.0,
            channelMean[2] / 255.0
        )
    )


def saveMean(fname, data):
    root = et.Element('opencv_storage')
    et.SubElement(root, 'Channel').text = '3'
    et.SubElement(root, 'Row').text = str(imgSize)
    et.SubElement(root, 'Col').text = str(imgSize)
    meanImg = et.SubElement(root, 'MeanImg', type_id='opencv-matrix')
    et.SubElement(meanImg, 'rows').text = '1'
    et.SubElement(meanImg, 'cols').text = str(imgSize * imgSize * 3)
    et.SubElement(meanImg, 'dt').text = 'f'
    et.SubElement(meanImg, 'data').text = ' '.join(
        ['%e' % n for n in np.reshape(data, (imgSize * imgSize * 3))]
    )

    tree = et.ElementTree(root)
    tree.write(fname)
    x = xml.dom.minidom.parse(fname)
    with open(fname, 'w') as f:
        f.write(x.toprettyxml(indent='  '))


def saveTrainImages(filename, foldername):
    if not os.path.exists(foldername):
        os.makedirs(foldername)
    data = {}
    dataMean = np.zeros((3, imgSize, imgSize))  # mean is in CHW format.
    with open('train_map.txt', 'w') as mapFile:
        with open('train_regrLabels.txt', 'w') as regrFile:
            for ifile in range(1, 6):
                file_path = os.path.join('./cifar-10-batches-py', 'data_batch_' + str(ifile))
                with open(file_path, 'rb') as f:
                    if sys.version_info[0] < 3:
                        data = cp.load(f)
                    else:
                        data = cp.load(f, encoding='latin1')
                    for i in range(10000):
                        fname = os.path.join(
                            os.path.abspath(foldername),
                            ('%05d.png' % (i + (ifile - 1) * 10000))
                        )
                        saveImage(
                            fname, data['data'][i, :], data['labels'][i],
                            mapFile, regrFile, 4, mean=dataMean
                        )
    dataMean = dataMean / (50 * 1000)
    saveMean('CIFAR-10_mean.xml', dataMean)


def saveTestImages(filename, foldername):
    if not os.path.exists(foldername):
        os.makedirs(foldername)
    with open('test_map.txt', 'w') as mapFile:
        with open('test_regrLabels.txt', 'w') as regrFile:
            with open(os.path.join('./cifar-10-batches-py', 'test_batch'), 'rb') as f:
                if sys.version_info[0] < 3:
                    data = cp.load(f)
                else:
                    data = cp.load(f, encoding='latin1')
                for i in range(10000):
                    fname = os.path.join(os.path.abspath(foldername), ('%05d.png' % i))
                    saveImage(fname, data['data'][i, :], data['labels'][i], mapFile, regrFile, 0)


if __name__ == "__main__":
    # CIFAR Image data
    imgSize = 32
    numFeature = imgSize * imgSize * 3

    # URLs for the train image and labels data
    url_cifar_data = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'

    # Paths for saving the text files
    data_dir = '/tmp/data/CIFAR-10/'
    train_filename = data_dir + '/Train_cntk_text.txt'
    test_filename = data_dir + '/Test_cntk_text.txt'

    train_img_directory = data_dir + '/Train'
    test_img_directory = data_dir + '/Test'

    root_dir = os.getcwd()

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    try:
        os.chdir(data_dir)
        trn, tst = loadData('http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz')
        print('Writing train text file...')
        saveTxt(r'./Train_cntk_text.txt', trn)
        print('Done.')
        print('Writing test text file...')
        saveTxt(r'./Test_cntk_text.txt', tst)
        print('Done.')
        print('Converting train data to png images...')
        saveTrainImages(r'./Train_cntk_text.txt', 'train')
        print('Done.')
        print('Converting test data to png images...')
        saveTestImages(r'./Test_cntk_text.txt', 'test')
        print('Done.')
    finally:
        os.chdir("../..")
