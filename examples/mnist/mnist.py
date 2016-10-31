import gzip
from ngraph.util.persist import ensure_dirs_exist, pickle_load, valid_path_append, fetch_file
import os, sys, getopt, errno
from tqdm import tqdm
import numpy as np
from PIL import Image
import argparse


class MNIST(object):
    """
    MNIST data set from https://www.cs.toronto.edu/~kriz/cifar.html

    Arguments:
        path (str): Local path to copy data files.
    """
    def __init__(self, path='.'):
        self.path = path
        self.url = 'https://s3.amazonaws.com/img-datasets'
        self.filename = 'mnist.pkl.gz'
        self.size = 15296311

    def load_data(self):
        """
        Fetch the CIFAR-10 dataset and load it into memory.

        Arguments:
            path (str, optional): Local directory in which to cache the raw
                                  dataset.  Defaults to current directory.
            normalize (bool, optional): Whether to scale values between 0 and 1.
                                        Defaults to True.

        Returns:
            tuple: Both training and test sets are returned.
        """
        workdir, filepath = valid_path_append(self.path, '', self.filename)
        if not os.path.exists(filepath):
            fetch_file(self.url, self.filename, filepath, self.size)

        with gzip.open(filepath, 'rb') as f:
            train_set, valid_set = pickle_load(f)

        return train_set, valid_set


def ingest_mnist(out_dir, overwrite=False):
    '''
    Save MNIST dataset as PNG files
    '''
    set_names = ('train', 'val')
    manifest_files = [os.path.join(out_dir, setn + '-index.csv') for setn in set_names]

    if (all([os.path.exists(manifest) for manifest in manifest_files]) and not overwrite):
        return manifest_files

    dataset = {k:s for k, s in zip(set_names, MNIST(path=out_dir).load_data())}

    manifest_list_cfg = ', '.join([k+':'+v for k, v in zip(set_names, manifest_files)])

    # Write out label files and setup directory structure
    lbl_paths, img_paths = dict(), dict(train=dict(), val=dict())
    for lbl in range(10):
        lbl_paths[lbl] = ensure_dirs_exist(os.path.join(out_dir, 'labels', str(lbl) + '.txt'))
        np.savetxt(lbl_paths[lbl], [lbl], fmt='%d')
        for setn in ('train', 'val'):
            img_paths[setn][lbl] = ensure_dirs_exist(os.path.join(out_dir, setn, str(lbl) + '/'))

    # Now write out image files and manifests
    for setn, manifest in zip(set_names, manifest_files):
        records = []
        for idx, (img, lbl) in enumerate(tqdm(zip(*dataset[setn]))):
            img_path = os.path.join(img_paths[setn][lbl], str(idx) + '.png')
            im = Image.fromarray(img)
            im.save(os.path.join(out_dir, img_path), format='PNG')
            records.append((os.path.relpath(img_path, out_dir),
                            os.path.relpath(lbl_paths[lbl], out_dir)))
        np.savetxt(manifest, records, fmt='%s,%s')

    return manifest_files

# def create_manifest(dataset, output_dir, set_name):
#     manifest_path = os.path.join(output_dir,'manifest_'+set_name+'.csv')
#     print('processing {0} set').format(output_dir)
#     output_dir = os.path.join(output_dir, set_name)
#     ensure_dirs_exist(output_dir)
#     images = dataset[0]
#     targets = dataset[1]

#     for idx in range(0,10):
#         tgt_path = os.path.join(output_dir, 'target_' + str(idx) + '.txt')
#         file = open(tgt_path, 'w')
#         file.write(str(idx))
#         file.close()

#     records = []
#     for idx, data in enumerate(tqdm(images)):
#         target = targets[idx]
#         img_path = os.path.join(output_dir, 'image_' + str(idx) + '.png')
#         tgt_path = os.path.join(output_dir, 'target_' + str(target) + '.txt')
#         im = Image.fromarray(data)
#         im.save(img_path, format='PNG')
#         records.append((img_path, tgt_path))
#     np.savetxt(manifest_path, records, fmt='%s,%s')
#     return manifest_path

# def ingest_mnist(output_dir, overwrite=False):
#     set_names = ('train', 'val')
#     manifest_files = [os.path.join(out_dir, setn + '-index.csv') for setn in set_names]

#     if (all([os.path.exists(manifest) for manifest in manifest_files]) and not overwrite):
#         return manifest_files


#     with gzip.open(input_file, 'rb') as f:
#         train_set, valid_set = pickle_load(f)
#         create_manifest(train_set, output_dir, 'train')
#         create_manifest(valid_set, output_dir, 'valid')
