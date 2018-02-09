# ******************************************************************************
# Copyright 2017-2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ******************************************************************************
import string
import csv
import numpy as np
from ngraph.util.persist import valid_path_append


class CrepeDataset(object):
    def __init__(self,
                 path='.',
                 sentence_length=1014,
                 use_uppercase=False):
        self.path = path
        self.sentence_length = sentence_length
        self.use_uppercase = use_uppercase

        self.token_to_idx, self.idx_to_token, self.vocab_size = self.make_vocab()

    def make_vocab(self):
        if self.use_uppercase:
            alphabet = string.ascii_letters
        else:
            alphabet = string.ascii_lowercase
        alphabet = alphabet + u"0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
        token_to_idx = {token: idx + 1 for idx, token in enumerate(alphabet)}
        idx_to_token = {idx + 1: token for idx, token in enumerate(alphabet)}
        return token_to_idx, idx_to_token, len(token_to_idx)

    def preprocess_text(self, text):
        if not self.use_uppercase:
            text = text.lower()
        text = text[::-1]
        text = text[:self.sentence_length]
        text = [self.token_to_idx.get(token, 0) for token in text]
        text = text + [0] * (self.sentence_length - len(text))
        return np.array(text, dtype=np.int32)

    def load_split(self, filepath):
        texts = []
        labels = []
        with open(filepath, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                label = row[0]
                text = ''.join(' '.join(row[-1:0:-1]))

                text = self.preprocess_text(text)
                label = int(label) - 1

                texts.append(text)
                labels.append(label)

        texts, labels = np.stack(texts), np.stack(labels)
        return texts, labels

    def make_dict(self, texts, labels):
        return {
            'text': {'data': texts, 'axes': ('N', 'REC')},
            'label': {'data': labels, 'axes': ('N')}
        }

    def load_data(self):

        train_filepath = valid_path_append(self.path, 'train.csv')
        test_filepath = valid_path_append(self.path, 'test.csv')

        train_texts, train_labels = self.load_split(train_filepath)
        test_texts, test_labels = self.load_split(test_filepath)

        train_dict = self.make_dict(train_texts, train_labels)
        test_dict = self.make_dict(test_texts, test_labels)
        data_dict = {'train': train_dict, 'test': test_dict}

        self.data_dict = data_dict
        return data_dict
