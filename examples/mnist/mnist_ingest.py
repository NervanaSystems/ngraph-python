import gzip
import pickle
import os, sys, getopt, errno
from tqdm import tqdm
import numpy as np
from PIL import Image
import argparse

def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

def create_manifest(dataset, output_dir, set_name):
    manifest_path = os.path.join(output_dir,'manifest_'+set_name+'.csv')
    print('processing {0} set').format(output_dir)
    output_dir = os.path.join(output_dir, set_name)
    make_sure_path_exists(output_dir)
    images = dataset[0]
    targets = dataset[1]

    for idx in range(0,10):
        tgt_path = os.path.join(output_dir, 'target_' + str(idx) + '.txt')
        file = open(tgt_path, 'w')
        file.write(str(idx))
        file.close()

    records = []
    for idx, data in enumerate(tqdm(images)):
        target = targets[idx]
        img_path = os.path.join(output_dir, 'image_' + str(idx) + '.png')
        tgt_path = os.path.join(output_dir, 'target_' + str(target) + '.txt')
        im = Image.fromarray(data)
        im.save(img_path, format='PNG')
        records.append((img_path, tgt_path))
    np.savetxt(manifest_path, records, fmt='%s,%s')

def ingest_mnist(input_file, output_dir):
    with gzip.open(input_file, 'rb') as f:
        train_set, valid_set = pickle.load(f)
        create_manifest(train_set, output_dir, 'train')
        create_manifest(valid_set, output_dir, 'valid')

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Ingest MNIST from pkl to pngs')
    parser.add_argument('-i', '--inputfile', default='', help='MNIST pkl file')
    parser.add_argument('-o', '--outputdir', default='', help='Output directory')
    args = parser.parse_args()
    ingest_mnist(args.inputfile, args.outputdir)