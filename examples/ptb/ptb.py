import gzip
from ngraph.util.persist import ensure_dirs_exist, pickle_load, valid_path_append, fetch_file
import os
import numpy as np


class PTB(object):
    """
    Penn Treebank data set from http://arxiv.org/pdf/1409.2329v5.pdf

    Arguments:

    """
    def __init__(self, path='.', reverse_target=False):
        self.path = path
        self.url = 'https://raw.githubusercontent.com/wojzaremba/lstm/master/data'
        self.filemap = dict(train=dict(filename='ptb.train.txt', size=5101618),
                            test=dict(filename='ptb.test.txt', size=449945),
                            valid=dict(filename='ptb.valid.txt', size=399782))
        self.reverse_target = reverse_target

    def load_data(self):
        self.data_dict = {}
        self.vocab = None
        for phase in ['train', 'test', 'valid']:
            filename, filesize = self.filemap[phase]['filename'], self.filemap[phase]['size']
            workdir, filepath = valid_path_append(self.path, '', filename)
            if not os.path.exists(filepath):
                fetch_file(self.url, filename, filepath, filesize)

            tokens = open(filepath).read()  # add tokenization here if necessary

            self.vocab = sorted(set(tokens if self.vocab is None else self.vocab))

            # vocab dicts
            self.token_to_index = dict((t, i) for i, t in enumerate(self.vocab))
            self.index_to_token = dict((i, t) for i, t in enumerate(self.vocab))

            # map tokens to indices
            X = np.asarray([self.token_to_index[t] for t in tokens], dtype=np.uint32)
            if self.reverse_target:
                y = X.copy()
            else:
                y = np.concatenate((X[1:], X[:1]))

            self.data_dict[phase] = (X, y)

        return self.data_dict
