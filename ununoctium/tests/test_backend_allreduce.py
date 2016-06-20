# Copyright 2016 Nervana Systems Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
To test AllReduce integration in Graphiti
Things to consider: 
1- Set the inputs: Either fixed of through the command line
2- Define a model
3- Call the model without mpi
4- Call the model with mpi for different thread_nums
5- Compare the results

Functions:
  setup(): set the arguments 
"""

from pprint import pprint
import pytest

import numpy as np

import argparse
import os
from subprocess import Popen, PIPE, call
import time


# TODO: Add more verbose description for arguments' help
def set_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n',type=int, default=0, help='Number of MPI threads')
    parser.add_argument('-r', type=int, default=0, help='Model parameter')
    parser.add_argument('--epochs', type=int, default=1, help='Model parameter')
    parser.add_argument('-mpi_flag', action='store_true', help='Run through MPI')
    parser.add_argument('--eval_freq', type=int, default=1, help='Model parameter')
    parser.add_argument('--testmodel', default='', help='Test model to run by mpi')
    parser.add_argument('--data_dir', default="../data/", help='Model\'s data')
    parser.add_argument('--save_path', default='~/', help='Model\'s data')
    args = parser.parse_args()
    return args


def make_run_cmd(args):
    cmd = "python {} --data_dir {} --save_path {}".format(args.testmodel, 
              args.data_dir, args.save_path)
    cmd += " -r {} -vv  --eval_freq {} --epochs {} ".format(str(args.r), 
              str(args.eval_freq), str(args.epochs))
    
    if args.n > 0 and args.mpi_flag:
        cmd = "mpirun -n {} {} -mpi_flag".format(str(args.n), cmd, '-mpi_flag')                    
    return cmd


def parse_result(results):
    '''
    results format: 
      Epoch 0
      Training error: 2.05111741959
      Test error: 1.97437227074
    '''
    training_errors = []
    test_errors = []
    for res in results:
        print res
        if res.find('Training error'):
            training_errors.append(float(res.strip('\n').split(':')[1]))
        elif es.find('Test error'):
            test_errors.append(float(res.strip('\n').split(':')[1]))
    

def run_test(args):
    # run stand-alone test 
    args.n = 0
    cmd = make_run_cmd(args)
    print 'running: ', cmd
    proc = Popen(cmd, stdout=PIPE, stderr=PIPE,shell=True)
    proc_out, proc_err = proc.communicate()
    
    print proc_err
    print proc_out

    print parse_result(proc_out)
    
    # run MPI test
    args.n = 2
    cmd = make_run_cmd(args)
    print 'running: ', cmd
    proc = Popen(cmd, stdout=PIPE, stderr=PIPE,shell=True)
    proc_out, proc_err = proc.communicate()
    
    # # print proc_err
    print proc_out
    

if __name__ == '__main__':
    # check for venv activations
    cmd = 'if [ -z "$VIRTUAL_ENV" ];then exit 1;else exit 0;fi'
    if call(cmd, shell=True) > 0:
        raise IOError('Need to activate the virtualenv')

    # cehck for the PYTHONPATH env_variable setup
    PYTHON_PATH = os.path.normpath(os.getenv("PYTHONPATH"))
    if PYTHON_PATH and os.path.isdir(PYTHON_PATH): 
        ## sanity check , ugly .. 
        dir1, remain= os.path.basename(PYTHON_PATH), os.path.dirname(PYTHON_PATH)
        dir2 = os.path.basename(remain)
        if dir1 != 'ununoctium' or dir2 != 'graphiti': 
            raise IOError("Need to set PYTHONPATH to \'~/graphiti/ununoctium\'")      
    else: 
        raise IOError("Need to set PYTHONPATH")

    # make and parse arguments 
    args = set_arguments()

    # check for the model: currently, only works for the model example located in 
    # "~/graphiti/ununoctium/tests/models.py", so I overwrite the argument
    # We can later define a genral template for AllReduce type of models 
    model = os.path.normpath(os.path.join(os.getenv("PYTHONPATH"), "tests/test_model_init.py"))
    if not os.path.isfile(model): 
        raise IOError("Can not find the model")
    args.testmodel = model
    
    # run test
    test_results = run_test(args)




