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
from ngraph.util.persist import valid_path_append
from tqdm import tqdm
import numpy as np


class TSP(object):
    """
    Traveling Salesman Problem dataset from https://arxiv.org/pdf/1506.03134.pdf

    Arguments:
        path (string): Data directory to find the data.
    """
    def __init__(self, train_filename, test_filename, path='.'):
        self.path = path
        self.filemap = dict(train=dict(filename=train_filename),
                            test=dict(filename=test_filename))

    def load_data(self):
        self.data_dict = {}
        for phase in ['train', 'test']:
            filename = self.filemap[phase]['filename']
            workdir, filepath = valid_path_append(self.path, '', filename)

            with open(filepath, 'r') as f:
                X, y, y_teacher = [], [], []
                for i, line in tqdm(enumerate(f)):
                    inputs, outputs = line.split('output')
                    X.append(np.array([float(j) for j in inputs.split()]).reshape([-1, 2]))
                    y.append(np.array([int(j) - 1 for j in outputs.split()])[:-1])  # delete last
                    # teacher forcing array as decoder's input while training
                    y_teacher.append([X[i][j - 1] for j in y[i]])
            X = np.array(X)
            y = np.array(y)
            y_teacher = np.array(y_teacher)
            self.data_dict[phase] = {'inp_txt': X, 'tgt_txt': y, 'teacher_tgt': y_teacher}

        return self.data_dict
