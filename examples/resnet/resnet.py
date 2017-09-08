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
import numpy as np
import ngraph as ng
from ngraph.frontends.neon import Layer, Sequential, Dropout, XavierInit
from ngraph.frontends.neon import Affine, Preprocess, Convolution, Pool2D, BatchNorm, Activation
from ngraph.frontends.neon import KaimingInit, Rectlin, Softmax, GradientDescentMomentum
from ngraph.frontends.neon import ax, NgraphArgparser
from ngraph.frontends.neon import make_bound_computation, make_default_callbacks, loop_train  # noqa
from tqdm import tqdm
import ngraph.transformers as ngt
from ngraph.op_graph.tensorboard.tensorboard import TensorBoard
from ngraph.op_graph.tensorboard.graph_def import ngraph_to_tf_graph_def
from data import make_aeon_loaders
try:
    import matplotlib
except ImportError:
    matplotlib_available = False
    print("matplotlib is not available disabling plotting")
else:
    matplotlib_available = True
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt


#Hyperparameters
#Optimizer
base_lr=0.1
gamma=0.1
momentum_coef=0.9
wdecay=0.0001
#BatchNorm
rho_val=0.2
#Initializer
weight_init=KaimingInit()

print("HyperParameters")
print("Learning Rate:     "+str(base_lr))
print("Momentum:          "+str(momentum_coef))
print("Weight Decay:      "+str(wdecay))
print("Rho:               "+str(rho_val))

#Helpers
#TODO: Fix mean_subtract for imagenet
def mean_subtract(x):
    bgr_mean = ng.persistent_tensor(
    axes=[x.axes.channel_axis()],
    initial_value=np.array([127.0, 119.0, 104.0]))
    return (x - bgr_mean)

#Returns dict of convolution layer parameters
def conv_params(fil_size, num_fils,rho=rho_val,strides=1,batch_norm=True,relu=True):
    return dict(fshape=(fil_size,fil_size,num_fils),
                rho=rho,
                strides=strides,
                padding=(1 if fil_size > 1 else 0),
                batch_norm=batch_norm,
                activation=(Rectlin() if relu else None),
                filter_init=weight_init)

#Class for Residual Module
class ResidualModule(object):
    def __init__(self,num_fils,direct=True,strides=1):
        self.direct=direct
        #This is for i1k Resnet50 and above
        if en_bottleneck:
            print("Oops you should not be here until i1k is done");
            exit()
        #This is for CIFAR10 and Resnet18 and Resnet34 for i1K
        else:
            #Main path always does two 3x3 convs with second one not doing activation
            self.main_path=Sequential([
                Convolution(**conv_params(3,num_fils,strides=strides)),
                Convolution(**conv_params(3,num_fils,relu=False))])
            
            #Side path will either have a 1x1 Conv to match shape or direct connection
            if(direct):
                self.side_path=None
            else:
                self.side_path=Convolution(**conv_params(1,num_fils,strides=strides,relu=False))

    def __call__(self,in_obj):
        #Computes the output for main path. Parallel path 1
        mp=self.main_path(in_obj)
        #Computes the output for side path. Parallel path 2
        sp=in_obj if self.direct else self.side_path(in_obj)
        #Sum both the paths        
        return mp+sp

#Class for constructing the network
class BuildResnet(Sequential):
    def __init__(self,net_type):
        #For CIFAR10 dataset
        if net_type=='cifar10':
            #Number of Filters
            num_fils=[16,32,64]
            #Network Layers
            layers=[#Subtracting mean as suggested in paper
                    Preprocess(functor=mean_subtract),
                    #First Conv with 3x3 and stride=1
                    Convolution(**conv_params(3,16))]

            first_resmod=True #Indicates the first residual module

            #Loop 3 times for each filter.
            for fil in range(3):
                #Lay out n residual modules so that we have 2n layers.
                for resmods in range(num_resnet_mods):
                    if(resmods==0):
                        if(first_resmod):
                            #Strides=1 and convolution side path
                            layers.append(ResidualModule(num_fils[fil],direct=False)) 
                            layers.append(Activation(Rectlin()))
                            first_resmod=False
                        else:
                            #Strides=2 and Convolution side path
                            layers.append(ResidualModule(num_fils[fil],strides=2,direct=False))
                            layers.append(Activation(Rectlin()))
                    else:
                        #Strides=1 and direction connection
                        layers.append(ResidualModule(num_fils[fil]))
                        layers.append(Activation(Rectlin()))
            #Do average pooling --> fully connected--> softmax.8 since final layer output size is 8
            layers.append(Pool2D(8,op='avg'))
            layers.append(Affine(axes=ax.Y,weight_init=weight_init))
            layers.append(Activation(Softmax()))
        else:
            print("Unknown dataset")
            exit()
        super(BuildResnet, self).__init__(layers=layers)

#Result collector
def loop_eval(dataset, computation, metric_names):
    if(use_aeon):
         dataset._dataloader.reset()
    else:
        dataset.reset()
    all_results = None
    for data in dataset:
        feed_dict = {input_ph[k]: data[k] for k in data.keys()}
        results = computation(feed_dict=feed_dict)
        if all_results is None:
            all_results = {name: list(res) for name, res in zip(metric_names, results)}
        else:
            for name, res in zip(metric_names, results):
                all_results[name].extend(list(res))
    ndata=dataset._dataloader.ndata if use_aeon else dataset.ndata
    reduced_results = {k: np.mean(v[:ndata]) for k, v in all_results.items()}
    return reduced_results

if __name__ == "__main__":
    #Command Line Parser
    parser = NgraphArgparser(description="Resnet for Imagenet and Cifar10")
    parser.add_argument('-dataset', type=str, default="cifar10", help="Enter cifar10 or i1k")
    parser.add_argument('-size', type=int, default=56, help="Enter size of resnet")
    parser.add_argument('-tb',type=int,default=0,help="1- Enables tensorboard")
    if matplotlib_available:
        parser.add_argument('-name',type=str,help="Name of experiment for graph", required=True)
    args=parser.parse_args()
    args.iter_interval=50000//args.batch_size 
    learning_schedule=[84*args.iter_interval,120 *args.iter_interval]
    print("Learning Schedule: "+str(learning_schedule))

    #Command line args
    cifar_sizes = [20, 32, 44, 56, 200]
    i1k_sizes   = [18, 34, 50, 101, 152]
    resnet_size=args.size
    resnet_dataset=args.dataset
    use_aeon=True
    #Checking Command line args are proper
    if resnet_dataset=='cifar10':
        if resnet_size in cifar_sizes:
            #Create training and validation set objects
            train_set,valid_set=make_aeon_loaders(args.data_dir,args.batch_size,args.num_iterations,dataset=resnet_dataset)
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
        else:
            print("Invalid cifar10 size.Select from "+str(cifar_sizes))
            exit()
    elif resnet_dataset=='i1k':
        if resnet_size in i1k_sizes:
            #Enable or disable bottleneck depending on resnet size
            if(resnet_size in [18,34]):
                en_bottleneck=False
                num_resnet_mods=(args.size-2)//6
            else:
                en_bottleneck=True
                num_resnet_mods=(args.size-2)//9
            #Creating training and validation set objects
            train_set,valid_set=make_aeon_loaders(args.data_dir,args.batch_size,args.num_iterations,dataset=resnet_dataset)
            # of Classes
            ax.Y.length=1000
            print("Completed loading Imagenet dataset")
            #Randomize seed
            np.random.seed(args.rng_seed)
            #Make placeholders
            input_ph=train_set.make_placeholders(include_iteration=True)
        else:
            print("Invalid i1k size. Select from "+str(i1k_sizes))
            exit()
    else:
        print("Invalid Dataset. Dataset should be either cifar10 or i1k")
        exit()

#Build the network
resnet = BuildResnet(resnet_dataset)
#Optimizer
learning_rate_policy = {'name': 'schedule',
                        'schedule': learning_schedule,
                        'gamma': gamma,
                        'base_lr': base_lr}

optimizer=GradientDescentMomentum(learning_rate=learning_rate_policy,
                                  momentum_coef=momentum_coef,
                                  wdecay=wdecay,
                                  nesterov=True,
                                  iteration=input_ph['iteration'])
label_indices=input_ph['label']
label_indices=ng.cast_role(ng.flatten(label_indices),label_indices.axes.batch_axis())
#Tensorboard
if(args.tb):
    seq1=BuildResnet(resnet_dataset)
    train=seq1(input_ph['image'])
    tb=TensorBoard("./")
    tb.add_graph(train)
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

#Training the network by calling transformer
transformer=ngt.make_transformer()
#Trainer
train_function=transformer.add_computation(train_computation)
#Inference
eval_function=transformer.add_computation(eval_computation)
#Progress bar
tpbar = tqdm(unit="batches", ncols=100, total=args.num_iterations)
interval_cost = 0.0
train_result=[]
test_result=[]
err_result=[]
diff_result=[]
for step, data in enumerate(train_set):
    data['iteration'] = step
    feed_dict = {input_ph[k]: data[k] for k in input_ph.keys()}
    output = train_function(feed_dict=feed_dict)
    tpbar.update(1)
    tpbar.set_description("Training {:0.4f}".format(output[()]))
    interval_cost += output[()]
    if (step + 1) % args.iter_interval == 0 and step > 0:
        eval_losses = loop_eval(valid_set, eval_function, eval_loss_names)
        tqdm.write("Interval {interval} Iteration {iteration} complete. "
                   "Avg Train Cost {cost:0.4f} Test Avg loss:{tcost} Diff:{diff}".format(
                       interval=step // args.iter_interval,
                       iteration=step,
                       cost=interval_cost / args.iter_interval,tcost=eval_losses,diff=abs(eval_losses['cross_ent_loss']-(interval_cost/args.iter_interval))))
        #For graph plotting
        train_result.append(interval_cost/args.iter_interval)
        test_result.append(eval_losses['cross_ent_loss'])
        err_result.append(eval_losses['misclass'])
	diff_result.append(abs(eval_losses['cross_ent_loss']-(interval_cost/args.iter_interval)))
        interval_cost = 0.0
        #tqdm.write("VALID Avg losses: {}".format(eval_losses))
if matplotlib_available:
    plt.figure(1)
    plt.subplot(211)
    plt.plot(train_result,label="Training Cost")
    plt.plot(test_result,label="Test Cost")
    plt.legend()
    plt.subplot(212)
    plt.plot(err_result,label="Error")
    plt.legend()
    plt.savefig(args.name+"_graph.png")
    plt.figure(2)
    plt.plot(diff_result)
    plt.savefig(args.name+"_diffgraph.png")
    print("Training complete.")

