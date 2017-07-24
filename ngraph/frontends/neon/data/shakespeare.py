# ----------------------------------------------------------------------------
# Copyright 2016 Nervana Systems Inc.
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
from ngraph.util.persist import valid_path_append, fetch_file
import os
import numpy as np


class Shakespeare(object):
    """
    Shakespeare Dataset from http://cs.stanford.edu/people/karpathy/char-rnn/shakespeare_input.txt
    Arguments:
        path (string): Data directory to find the data, if does not exist, will
                       download the data
        url (string, optional): path to the text file to be downloaded
        filename (string, optional): name of the text file
        train_split (float): Value between 0 and 1
                             Ratio of the text to set aside for training

    """
    def __init__(self, path='./data/', url=None, filename=None, train_split=.9):
        self.path = path
        self.vocab = None
        if(url is None):
            self.url = 'http://cs.stanford.edu/people/karpathy/char-rnn/'
            self.filename = 'shakespeare_input.txt'
        else:
            self.url = url
            self.filename = filename

        self.train_split = train_split

        # Load the text and split to train and test
        self.train, self.test = self.load_data()

        # Digitize the train set (convert letters to integers)
        self.train = self.digitize(text=self.train)
        # Digitize the test set using train set vocab (convert letters to integers)
        self.test = self.digitize(text=self.test, vocab=self.vocab)

    def load_data(self):
        self.data_dict = {}
        workdir, filepath = valid_path_append(self.path, '', self.filename)
        if not os.path.exists(filepath):
            fetch_file(self.url, self.filename, filepath)

        tokens = open(filepath).read()

        train_samples = int(self.train_split * len(tokens))
        train = tokens[:train_samples]
        test = tokens[train_samples:]

        return train, test

    def build_vocab(self, text):
        '''
            Build a vocabulary from given text and store as the object's vocab
            If no text given, build the vocab from self.train
        '''
        if text is None:
            self.vocab = sorted(set(self.train))
        else:
            self.vocab = sorted(set(text))

        # vocab dicts
        self.token_to_index = dict((t, i + 1) for i, t in enumerate(self.vocab))
        self.index_to_token = dict((i + 1, t) for i, t in enumerate(self.vocab))

        # Add zero as unknown token
        self.index_to_token[0] = '<UNK>'
        self.token_to_index['<UNK>'] = 0

    def digitize(self, text, vocab=None):
        '''
            Convert given text to a sequence of integers (indices) using the given vocab
            If no vocab given, it is built from the text
        '''
        if self.vocab is None:
            self.build_vocab(text=text)

        # map tokens to indices
        # if the token is not in the vocabulary, put a zero (unknown)
        text_dig = np.asarray([self.token_to_index[t] if t in self.vocab else 0 for t in text],
                              dtype=np.uint32)
        return text_dig
