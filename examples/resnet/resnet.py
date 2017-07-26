#!/usr/bin/env python
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
import ngraph as ng
import numpy as np
from ngraph.frontends.neon import NgraphArgparser
from data import make_aeon_loaders

#Command Line Parser
parser = NgraphArgparser(description="Resnet trainer for Imagenet and Cifar10")
parser.add_argument('--dataset', type=str, default='cifar10', help="Enter cifar10 or i1k")
parser.add_argument('--size', type=int, default=56, help="Enter size of resnet")
args=parser.parse_args()

#Command line args
cifar_sizes = [20, 32, 44, 56, 200]
i1k_sizes   = [18, 34, 50, 101, 152] 
resnet_size=args.size
resnet_dataset=args.dataset

#Checking Command line args are proper
if resnet_dataset=='cifar10':
    if resnet_size in cifar_sizes:
        train_set,valid_set=make_aeon_loaders(args.data_dir,args.batch_size,args.num_iterations)
        num_resnet_layers=(args.size-2)/6
        print("Completed loading CIFAR10 dataset")                
    else:
        print("Invalid cifar10 size.Select from 20,32,44,56,200")
        exit()
elif resnet_dataset=='i1k':
    if resnet_size in i1k_sizes:
        print("i1k is still under construction")
        exit()
    else:
        print("Invalid i1k size.Select from 18,34,50,101,152")
        exit()
else:
    print("Invalid Dataset. Dataset should be either cifar10 or i1k")
    exit()
print("Passed through basic screening") 

#Make placeholders
input_ph=train_set.make_placeholders(include_iteration=True)


