import tempfile

import numpy as np
from PIL import Image as PILImage
import random
import struct
import pytest


def random_image(filename):
    """
    generate a small random image
    """
    a = np.random.random((28, 28, 1)).astype('uint8')
    img = PILImage.fromarray(a)
    img.save(filename)


def invalid_image(filename):
    """
    write an empty file to filename to trigger invalid image file exceptions
    """
    with open(filename, 'w') as f:
        pass


def random_target(filename):
    target = int(random.random() * 1024)

    with open(filename, 'wb') as f:
        f.write(struct.pack('i', target))

    return filename


def random_manifest(num_lines, invalid_image_index=None):
    manifest = tempfile.NamedTemporaryFile(mode='w')

    # generate a manifest of filenames with an invalid image on the 3rd line
    for i in range(num_lines):
        img_filename = tempfile.mkstemp(suffix='.jpg')[1]
        if i == invalid_image_index:
            invalid_image(img_filename)
        else:
            random_image(img_filename)

        target_filename = tempfile.mkstemp(suffix='.jpg')[1]
        random_target(target_filename)

        manifest.write("{},{}\n".format(img_filename, target_filename))
    manifest.flush()

    return manifest
