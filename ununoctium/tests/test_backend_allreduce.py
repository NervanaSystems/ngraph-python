# ----------------------------------------------------------------------------
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
# ----------------------------------------------------------------------------
"""
This script uses the model in ./test_model_init.py to test AllReduce performance
Command Example for running the mode:
    python test_model_init.py --data_dir /Path_to_data/ --save_path ~./results/ -r 0\
     -vv  --eval_freq 1 --epochs 2 -mpi_flag
The output of the model script should have the following format:
    epoch: 0 time: 5.83s train_error: 71.70 test_error: 68.75 train_loss: 2.907
Use test_model_init.py as the template for developing any other models
"""
import os
import argparse
import time
import numpy as np
from subprocess import Popen, PIPE, call
import itertools as itt

# TODO: Add more verbose description for arguments' help
def set_arguments():
    """
    Add the MPI and model arguments. 
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--n',type=int, default=0, help='Number of MPI threads')
    parser.add_argument('-mpi_flag', action='store_true', help='Run through MPI')
    parser.add_argument('-r', type=int, default=0, help='Model parameter: Random seed')
    parser.add_argument('--epochs', type=int, default=1, help='Model parameter: number of epochs')
    parser.add_argument('--eval_freq', type=int, default=1, help='Model parameter: Evaluation frequency')
    parser.add_argument('--testmodel', default='', help='Path to the test model: default -> \
      ununoctium/tests/test_model_init.py')
    parser.add_argument('--data_dir', default="../data/", help='Path to the model\'s data')
    parser.add_argument('--save_path', default='~/results/', help='Path to the result directory')
    args = parser.parse_args()
    return args


def make_run_cmd(args):
    """
    Make a command
    """
    cmd = "python {} --data_dir {} --save_path {}".format(args.testmodel, 
              args.data_dir, args.save_path)
    cmd += " -r {} -vv  --eval_freq {} --epochs {} ".format(str(args.r), 
              str(args.eval_freq), str(args.epochs))
    
    if args.n > 0 and args.mpi_flag:
        cmd = "mpirun -n {} {} -mpi_flag".format(str(args.n), cmd, '-mpi_flag')                    
    return cmd


def parse_cmd_outputs(results):
    '''
    Parse the output of stdout 
    Sample output: 'epoch: 0 time: 5.72s train_error: 71.70 test_error: 68.75 train_loss: 2.907'
    '''
    train_info = []
    results = [res for res in results.split('\n') if res.strip()]
    try: 
      for res in results:
          res = res.strip('\n').split()
          # train_info: (epoch, train_time, train_error, test_error, train_loss) 
          train_info.append((res[1], res[3][:-1], res[5], res[7], res[9]))    
      return train_info
    except IndexError:
        raise IOError('Invalid output foramat!')
    
def analyze_test_results(test_results):
    """
    Analyzing the results for different MPI threads. 
    Each result tuple is: (epoch, train_time, train_error, test_error, train_loss) 
    """
    # take average for each of the test's output
    average_test_resuts = dict()
    for test, res in test_results.items():
        numeric_res = [tuple(map(float,itt.compress(i, (0,1,1,1,1)))) for i in res]
        avg_numeric_res = tuple(round(np.mean(i),3) for i in zip(*numeric_res))
        average_test_resuts[test] = avg_numeric_res

    # print the results
    def print_analyzed_test_results(results):
        """
        To print the analyzed results. 
        Format: {"MPI_thread_num":(train_time, train_error, test_error, train_loss)}
        """
        print "\t\tAnalyzed results:"

        headers = ['MPI_thrds', 'train_time', 'train_error', 'test_error', 'train_loss']
        table_content = []

        row_format ="{:>16}" * (len(headers) + 1)
        print row_format.format("", *headers)    

        for thrds, res in sorted(results.items()):
            row_content = [thrds] + list(res)
            table_content.append(row_content)
            print row_format.format("", *row_content)
    
    print_analyzed_test_results(average_test_resuts)


def run_tests(args):
    """
    Run tests  
    """
    # sets of the experiments with different MPI thread nums
    threads_set = set([0,1,2])
    threads_set.add(args.n)
    tests = dict((thrd,[]) for thrd in threads_set)
    for test,res in tests.items():
        args.n = test
        args.mpi_flag = (test > 0)
        cmd = make_run_cmd(args)
        # run a blocking process
        print cmd
        proc = Popen(cmd, stdout=PIPE, stderr=PIPE,shell=True)
        proc.wait()
        proc_out, proc_err = proc.communicate()
        res[:] = parse_cmd_outputs(proc_out)
    # analyzing the results
    analyze_test_results(tests)
    

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
    # "~/graphiti/ununoctium/tests/test_model_init.py", so I overwrite the argument
    # We can later define a genral template for AllReduce type of models 
    model = os.path.normpath(os.path.join(os.getenv("PYTHONPATH"), "tests/test_model_init.py"))
    if not os.path.isfile(model): 
        raise IOError("Can not find the model")
    args.testmodel = model
    # run tests
    run_tests(args)