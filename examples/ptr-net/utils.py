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
<<<<<<< HEAD
import numpy as np

def first_example(tsp_data):
    return tsp_data['train']['inp_txt'][0], tsp_data['train']['tgt_txt'][0]

def avg_travel_distance(inputs_array, pred_travel_index_array):
    """on going function"""

    avg_travel_distance = 0
    num_of_examples = 0
    for i, (inputs, pred_travel_index) in enumerate(zip(inputs_array, pred_travel_index_array)):
        avg_travel_distance += travel_distance(inputs, pred_travel_index)
        num_of_examples = i+1

    print(num_of_examples)
    return avg_travel_distance



def travel_distance(inputs, pred_travel_index):
    """
    Sum up the total travel distance per example.

    Arguements:
        inputs (array): an array of coordinates

        pred_travel_index (array): the predicted order to travel

    Return:
        travel_dist (float)
    """
    pred_travel_coordinate = inputs[pred_travel_index-1]
    num_points = len(pred_travel_coordinate)

    travel_dist = 0
    for i, coord in enumerate(pred_travel_coordinate):
        if i+1 == num_points:
            # distance between the last point visited and the start point
            travel_dist += distance(pred_travel_coordinate[i], pred_travel_coordinate[0])
        else:
            # distance between two successive points
            travel_dist += distance(pred_travel_coordinate[i], pred_travel_coordinate[i+1])

    return travel_dist

def distance(coord_a, coord_b):
    assert len(coord_a) == len(coord_b)
    dim = len(coord_a)
    sum_square_dist = 0

    for a, b in zip(coord_a,coord_b):
        sum_square_dist += (a-b)**2

    dist = sum_square_dist**(1./dim)

    return dist


if __name__ == '__main__':
    # test case
    inputs = np.array([[0.59764034, 0.81147296],
     [0.37839254, 0.1881658],
     [0.44284963, 0.29181517],
     [0.36377419, 0.59624465],
     [0.95521193, 0.94703186]])


    pred_travel_index = np.array([1, 4, 2, 3, 5])

    # test case
    # inputs = np.array([[0, 0], [1, 1], [1, 0], [0, 1]])
    # pred_travel_index = np.array([1, 3, 2, 4])

    travel_dist = travel_distance(inputs, pred_travel_index)
    print(travel_dist)
=======
def get_first_example(tsp_data):
    return tsp_data['train']['inp_txt'][0], tsp_data['train']['tgt_txt'][0]
>>>>>>> 4b7eb0159d671ed0226fe820069655d9a2aeb6d4
