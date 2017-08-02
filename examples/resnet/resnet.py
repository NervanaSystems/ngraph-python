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
from __future__ import division, print_function
from builtins import range
import numpy as np
import ngraph as ng
from ngraph.frontends.neon import Layer, Sequential
from ngraph.frontends.neon import Affine, Preprocess, Convolution, Pool2D, BatchNorm, Activation
from ngraph.frontends.neon import KaimingInit, Rectlin, Softmax, GradientDescentMomentum
from ngraph.frontends.neon import ax, NgraphArgparser
from ngraph.frontends.neon import make_bound_computation, make_default_callbacks, loop_train  # noqa
from tqdm import tqdm
import ngraph.transformers as ngt
import ngraph.op_graph.tensorboard.tensorboard as tb
from data import make_aeon_loaders

#Helpers
def mean_subtract(x):
    bgr_mean = ng.persistent_tensor(
        axes=[x.axes.channel_axis()],
        initial_value=np.array([104., 119., 127.]))
    return (x - bgr_mean) / 255.

#Returns dict of convolution layer parameters
def conv_params(fil_size, num_fils, strides=1,en_relu=True, en_batchnorm=True):
   return dict(fshape=(fil_size,fil_size,num_fils),
                strides=strides,
                padding=(1 if fil_size > 1 else 0),
                activation=(Rectlin() if en_relu else None),
                filter_init=KaimingInit(),
                batch_norm=en_batchnorm)

#Class for Residual Module
class ResidualModule(object):
    def __init__(self,num_fils,first=False,strides=1):

        self.side_path =None
        #Trunk represents the operation after addition
        self.trunk =None

        if en_bottleneck:
            print("Oops you should not be here until i1k is done");
            exit()
        else:
            #This is for CIFAR10 and Resnet18 and Resnet34 for i1K
            main_path=([Convolution(**conv_params(fil_size=3,num_fils=num_fils,strides=strides)),
                        Convolution(**conv_params(fil_size=3,num_fils=num_fils))])

            #Add a 1x1 Convolution with stride 2 for dimension reduction to allow proper addition
            #Add a 1x1 Conv with stride 1 for the very first residual module
            if first or strides == 2:
                self.side_path=Convolution(
                        **conv_params(1,num_fils,strides=strides,en_relu=False,en_batchnorm=False))
            else:
                main_path=[BatchNorm(), Activation(Rectlin())]+main_path
            
            #Relu after addition
            if strides == 2:
                self.trunk=Sequential([BatchNorm(),Activation(Rectlin())])

            self.main_path=Sequential(main_path)

    def __call__(self,in_obj):
        if en_bottleneck:
            print("Comeback when i1k is implemented")
            exit()
        else:
            #Process through trunk if it exists
            trunk_val=self.trunk(in_obj) if self.trunk else in_obj
            #Divide input half for size matching
            identity_conn=self.side_path(trunk_val) if self.side_path else trunk_val
            #Calculate outputs of convolution
            convs=self.main_path(trunk_val)
        return identity_conn+convs

#Class for constructing the network
class BuildResnet(Sequential):
    def __init__(self,net_type):
        num_fils=[16,32,64]
        if net_type=='cifar10' or 'cifar100':
            layers=[#Subtracting mean as suggested in paper
                    Preprocess(functor=mean_subtract),
                    #First Conv with 3x3 and stride=1
                    Convolution(**conv_params(fil_size=3,num_fils=16))]
            first=True
            #Lay out residual layers. Hardcoding 3 as there are only 3 sets of filters
            for fil in range(3):
                for resmods in range(num_resnet_mods):
                    if(resmods==0):
                        if(first):
                            layers.append(ResidualModule(num_fils[fil],strides=1,first=first))
                            first=False
                        else:
                            layers.append(ResidualModule(num_fils[fil],strides=2))
                    else:
                        layers.append(ResidualModule(num_fils[fil],strides=1))
            layers.append(BatchNorm())
            layers.append(Activation(Rectlin()))
            #Do average pooling --> fully connected--> softmax.8 since final layer output size is 8
            layers.append(Pool2D(8,op='avg'))
            #Axes are 10 as number of classes are 10
            layers.append(Affine(axes=ax.Y,weight_init=KaimingInit(),activation=Softmax()))
        elif net_type=='i1k':
            print("Seriously how did we come this far.")
            exit()
        else:
            print("Unknown network")
            exit()
        #Sequential.__init__(self,layers=layers)
        super(BuildResnet, self).__init__(layers=layers)

#Result collector
def loop_eval(dataset, computation, metric_names):
    dataset._dataloader.reset()
    all_results = None
    for data in dataset:

        feed_dict = {input_ph[k]: data[k] for k in data.keys()}
        results = computation(feed_dict=feed_dict)
        if all_results is None:
            all_results = {name: list(res) for name, res in zip(metric_names, results)}
        else:
            for name, res in zip(metric_names, results):
                all_results[name].extend(list(res))

    reduced_results = {k: np.mean(v[:dataset._dataloader.ndata]) for k, v in all_results.items()}
    return reduced_results


if __name__ == "__main__":
    #Command Line Parser
    parser = NgraphArgparser(description="Resnet for Imagenet and Cifar10")
    parser.add_argument('--dataset', type=str, default="cifar10", help="Enter cifar10 or i1k")
    parser.add_argument('--size', type=int, default=56, help="Enter size of resnet")
    args=parser.parse_args()

    #Command line args
    cifar_sizes = [8, 14, 20, 32, 44, 56, 200]
    i1k_sizes   = [18, 34, 50, 101, 152]
    resnet_size=args.size
    resnet_dataset=args.dataset

    #Checking Command line args are proper
    if resnet_dataset=='cifar10':
        if resnet_size in cifar_sizes:
            #Create training and validation set objects
            train_set,valid_set=make_aeon_loaders(args.data_dir,args.batch_size,args.num_iterations)
            #Num of resnet modules required for cifar10
            num_resnet_mods=(args.size-2)//6
            #Only 2 layers for one resnet module
            en_bottleneck=False
            ax.Y.length=10
            print("Completed loading CIFAR10 dataset")
            #Randomize Seed
            np.random.seed(args.rng_seed)
            #Make placeholders
            input_ph=train_set.make_placeholders(include_iteration=True)
            ax.Y.length = 10
            resnet = BuildResnet(resnet_dataset)
            learning_rate_policy = {'name': 'schedule',
                                    'schedule': [32000, 48000],
                                    'gamma': 0.1,
                                    'base_lr': 0.1}
            optimizer=GradientDescentMomentum(learning_rate=learning_rate_policy,
                                              momentum_coef=0.9,
                                              wdecay=0.0001,
                                              iteration=input_ph['iteration'])

        else:
            print("Invalid cifar10 size.Select from "+str(cifar_sizes))
            exit()
    elif resnet_dataset=='i1k':
        if resnet_size in i1k_sizes:
            if(resnet_size in [18,34]):
                en_bottleneck=False
            else:
                en_bottleneck=True
            print("i1k is still under construction")
            exit()
        else:
            print("Invalid i1k size. Select from "+str(i1k_sizes))
            exit()
    else:
        print("Invalid Dataset. Dataset should be either cifar10 or i1k")
        exit()

label_indices=input_ph['label']
label_indices=ng.cast_role(ng.flatten(label_indices),label_indices.axes.batch_axis())
#Put 1 for tensorboard tracking
if(0):
    seq1=BuildResnet(resnet_dataset)
    train=seq1(input_ph['image'])
    print(type(train))
    test=tb.ngraph_to_tensorboard(train,True)
    exit()
train_loss=ng.cross_entropy_multi(resnet(input_ph['image']),ng.one_hot(label_indices,axis=ax.Y))
batch_cost=ng.sequential([optimizer(train_loss),ng.mean(train_loss,out_axes=())])
train_computation=ng.computation(batch_cost,"all")

#Inference Parameters
with Layer.inference_mode_on():
    inference_prob = resnet(input_ph['image'])
    errors = ng.not_equal(ng.argmax(inference_prob, out_axes=[ax.N]), label_indices)
    eval_loss = ng.cross_entropy_multi(inference_prob, ng.one_hot(label_indices, axis=ax.Y))
    eval_loss_names = ['cross_ent_loss', 'misclass']
    eval_computation = ng.computation([eval_loss, errors], "all")

#Trainig the network by calling transformer
transformer=ngt.make_transformer()
train_function=transformer.add_computation(train_computation)
eval_function=transformer.add_computation(eval_computation)
tpbar = tqdm(unit="batches", ncols=100, total=args.num_iterations)
interval_cost = 0.0

for step, data in enumerate(train_set):
    data['iteration'] = step
    feed_dict = {input_ph[k]: data[k] for k in input_ph.keys()}
    output = train_function(feed_dict=feed_dict)

    tpbar.update(1)
    tpbar.set_description("Training {:0.4f}".format(output[()]))
    interval_cost += output[()]
    if (step + 1) % args.iter_interval == 0 and step > 0:
        tqdm.write("Interval {interval} Iteration {iteration} complete. "
                   "Avg Train Cost {cost:0.4f}".format(
                       interval=step // args.iter_interval,
                       iteration=step,
                       cost=interval_cost / args.iter_interval))
        interval_cost = 0.0
        eval_losses = loop_eval(valid_set, eval_function, eval_loss_names)
        tqdm.write("Avg losses: {}".format(eval_losses))

print("Training complete.")
