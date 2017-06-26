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
from ngraph.util.persist import valid_path_append, fetch_file, pickle_load
import os
import numpy as np


def preprocess_text(X, vocab_size, oov=2, start=1, index_from=3):
    """
    Preprocess the text by adding start and offset indices given padding. Also
    limit the max index using

    Arguments:
        X (List): dataset
        vocab_size (int): a given vocab_size limit. From a dataset. Words less frequent can
                    get capped when they are beyond the vocab size.
        oov (int): the index used for oov word, typically 2
        start (int): the index used as the start of a sentence, typically 1
        index_from (int): the index offset, typically 3, given having 0 (padding),
                          1 (start), 2 (OOV).
    """

    if start is not None:
        X = [[start] + [w + index_from for w in x] for x in X]
    else:
        X = [[w + index_from for w in x] for x in X]

    if vocab_size is None:
        vocab_size = max([max(x) for x in X])

    if oov is not None:
        X = [[oov if w >= vocab_size else w for w in x] for x in X]

    return X


def pad_sentences(sentences, pad_idx, pad_to_len=None, pad_from='left'):
    """
    pad sentences to the same length. When the length is not given,
    use the max length from the set.

    Arguments:
        sentences (List): list of sentences
        pad_idx (int): the index value used for padding
        pad_to_len (int): the maximum length to pad to
        pad_from (string): either "left" or "right"
    """
    nsamples = len(sentences)

    if pad_to_len is None:
        lengths = [len(sent) for sent in sentences]
        pad_to_len = np.max(lengths)

    X = (np.ones((nsamples, pad_to_len)) * pad_idx).astype(dtype=np.int32)
    for i, sent in enumerate(sentences):
        trunc = sent[-pad_to_len:]
        if pad_from is 'left':
            X[i, -len(trunc):] = trunc
        else:
            X[i, :len(trunc)] = trunc
    return X


class IMDB(object):
    """
    IMDB data set from http://www.aclweb.org/anthology/P11-1015..

    Arguments:
        path (string): Data directory to find the data, if not existing, will
                       download the data
        vocab_size (int): vocabulary size limite
        sentence_length (int): the max sentence length to pad the data to
        pad_idx (int): the index value used for padding
        shuffle (bool): whether to shuffle the data
    """

    def __init__(self, path='.', vocab_size=20000, sentence_length=128,
                 pad_idx=0, shuffle=False):
        self.path = path
        self.url = 'https://s3.amazonaws.com/text-datasets'
        self.filename = 'imdb.pkl'
        self.filesize = 33213513
        self.vocab_size = vocab_size
        self.sentence_length = sentence_length
        self.shuffle = shuffle
        self.pad_idx = pad_idx

    def load_data(self, test_split=0.2):
        self.data_dict = {}
        self.vocab = None
        workdir, filepath = valid_path_append(self.path, '', self.filename)
        if not os.path.exists(filepath):
            fetch_file(self.url, self.filename, filepath, self.filesize)

        with open(filepath, 'rb') as f:
            X, y = pickle_load(f)

        X = preprocess_text(X, self.vocab_size)
        X = pad_sentences(
            X, pad_idx=self.pad_idx, pad_to_len=self.sentence_length, pad_from='left')

        if self.shuffle:
            np.random.shuffle(X)
            np.random.shuffle(y)

        # split the data
        X_train = X[:int(len(X) * (1 - test_split))]
        y_train = y[:int(len(X) * (1 - test_split))]

        X_test = X[int(len(X) * (1 - test_split)):]
        y_test = y[int(len(X) * (1 - test_split)):]

        y_train = np.array(y_train)
        y_test = np.array(y_test)

        self.nclass = 1 + max(np.max(y_train), np.max(y_test))

        self.data_dict['train'] = {'review': {'data': X_train,
                                              'axes': ('batch', 'REC')},
                                   'label': {'data': y_train,
                                             'axes': ('batch',)}}
        self.data_dict['valid'] = {'review': {'data': X_test,
                                              'axes': ('batch', 'REC')},
                                   'label': {'data': y_test,
                                             'axes': ('batch',)}}
        return self.data_dict
