import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import os.path
import copy

#Ngraph imports
from builtins import range
import numpy as np
import ngraph as ng
from ngraph.frontends.neon import Layer, Sequential, Dropout, XavierInit
from ngraph.frontends.neon import Affine, Preprocess, Convolution, Pool2D, BatchNorm, Activation
from ngraph.frontends.neon import KaimingInit, Rectlin, Softmax, GradientDescentMomentum
from ngraph.frontends.neon import ax, NgraphArgparser
from tqdm import tqdm
from ngraph.op_graph.tensorboard.tensorboard import TensorBoard
from ngraph.op_graph.tensorboard.graph_def import ngraph_to_tf_graph_def
import ngraph.transformers as ngt
from ngraph.frontends.neon import ArrayIterator  

#Neon imports
from neon.layers import Conv, Pooling, GeneralizedCost
from neon.layers import Affine as Naffine
from neon.transforms import Rectlin as Nrectlin
from neon.transforms import Softmax as Nsoftmax
from neon.transforms import CrossEntropyMulti
from neon.models import Model
from neon.util.argparser import NeonArgparser
from neon.backends import gen_backend 
from neon.optimizers import GradientDescentMomentum as NeSGD
from neon.initializers import Kaiming
from neon.initializers.initializer import Initializer

#1. Copy initilizer from ngrpah to neon
class copyInit(Initializer): 
    def __init__(self, initTensor, name="copyInit"):
        super(copyInit, self).__init__(name=name)
        self.initTensor = initTensor 
    def fill(self, param):
        param[:] = self.be.array(self.initTensor)

#2. Use random data as input images if you cannot load same image on both FWs
learning_rate=0.1

def test_ngraph_resnet(iters,input_data,labels):
    #Make axes for input data
    C_len,D_len,H_len,W_len,N_len=input_data.shape
    C=ng.make_axis(length=C_len,name='C')
    D=ng.make_axis(length=D_len,name='D')
    H=ng.make_axis(length=H_len,name="H")
    W=ng.make_axis(length=W_len,name='W')
    N=ng.make_axis(length=N_len,name='N')
    Y=ng.make_axis(length=3,name="Classes") #There are only 3 classes
    #Make Axes
    image_axes=ng.make_axes([C,D,H,W,N])
    label_axes=ng.make_axes([Y,N])
    #Make Placeholder
    ng_input_ph={}
    ng_input_ph['image'] = ng.placeholder(image_axes)
    ng_input_ph['label'] = ng.placeholder(label_axes)
    #Define layers
    layers=[
            Convolution((3,3,16),filter_init=KaimingInit(),activation=Rectlin(),batch_norm=False,name="Conv0"),
            Affine(axes=Y,weight_init=KaimingInit(),activation=Softmax(),batch_norm=False)
    ]
    #Define Model
    model=Sequential(layers)
    #Prediction
    prediction=model(ng_input_ph['image'])
    #Calculate training loss
    train_loss=ng.cross_entropy_multi(prediction,ng_input_ph['label'])
    #Optimizer
    optimizer=GradientDescentMomentum(learning_rate,momentum_coef= 0, wdecay = 0)
    #Update weights using training loss
    update=optimizer(train_loss)
    #Caluclate mean batch cost
    batch_cost=ng.sequential([update,ng.mean(train_loss,out_axes=())])
    #Define transformer
    transformer=ngt.make_transformer()
    #Define Computation
    loss_comp=transformer.computation(train_loss,"all")
    batch_comp=transformer.computation(batch_cost,"all")
    #These computations are for init copy
    #Conv1 Layer
    conv0params_comp=transformer.computation(model.layers[0].variables.items()[0][1])
    conv0bias_comp=transformer.computation(model.layers[0].variables.items()[1][1])
    #Affine1 Layer
    affine0param_comp=transformer.computation(model.layers[1].variables.items()[0][1])
    affine0bias_comp=transformer.computation(model.layers[1].variables.items()[1][1])
    #Compute output of first Conv layer
    opt_comp=transformer.computation(model.scopes['Conv0'].outputs['Rectlin/Add'],"all")
    #Feed Dict
    feed_dict={
        ng_input_ph['image']:copy.deepcopy(input_data),
        ng_input_ph['label']:copy.deepcopy(labels.transpose())}
    #Initialize Transformer
    transformer.initialize()
    #Copy initializations 
    conv0w=copy.deepcopy(conv0params_comp(feed_dict=feed_dict))
    conv0b=copy.deepcopy(conv0bias_comp(feed_dict=feed_dict))
    affine0w=copy.deepcopy(affine0bias_comp(feed_dict=feed_dict))
    affine0b=copy.deepcopy(affine0bias_comp(feed_dict=feed_dict))
    #Iterate
    for i in range(iters):
        #loss=loss_comp(feed_dict=feed_dict)
        #print("LOSS: "+str(loss))
        opt=opt_comp(feed_dict=feed_dict)
        print("OPT"+opt.sum())
        output=batch_comp(feed_dict=feed_dict)
        print(output)
    print("Completed Ngraph computations")
    return [[conv0w,conv0b],[affine0w,affine0b]]

def test_neon_resnet(iters,input_data,labels,ne_init_data):
    parser = NeonArgparser()
    args = parser.parse_args()
    #Backend
    be=gen_backend(args.backend,batch_size=4,rng_seed=0)
    #Initialization from ngraph
    #Conv Layer
    c0w=copyInit(ne_init_data[0][0].squeeze())
    c0b=copyInit(ne_init_data[0][1])
    #Affine Layer
    a0w=copyInit(ne_init_data[1][0].squeeze())
    a0b=copyInit(ne_init_data[1][1].squeeze())
    #Define layers
    layers=[
        Conv((3,3,16),init=c0w,bias=c0b,activation=Nrectlin(),batch_norm=False),
        Naffine(3,init=a0w,bias=a0b,activation=Nsoftmax(),batch_norm=False)
    ]
    #Define model
    neData = be.array(input_data.reshape(-1,be.bsz))
    model=Model(layers=layers)
    #Cost
    neon_cost = GeneralizedCost(costfunc=CrossEntropyMulti())
    #Optimizer
    model.initialize(input_data.shape,neon_cost)
    neon_optimizer=NeSGD(learning_rate,momentum_coef= 0, wdecay = 0)
    #Iterate
    for i in range(iters):
        result=model.fprop(neData)
        loss=neon_cost.get_cost(result,be.array(labels).transpose())
        print(loss)
        delta=neon_cost.get_errors(result,be.array(labels).transpose())
        model.bprop(delta)
        neon_optimizer.optimize(model.layers_to_optimize,epoch=i)
    print("Completed NEON")
   
    
if __name__ == "__main__":
    iters=5
    np.random.seed(0) 

    #Generate Input test data. Formate is C,D,H,W,N
    input_data_shape=(3,1,32,32,4)
    input_data=np.random.random(input_data_shape)

    #Generate labels
    labels=np.array([[0,0,1],[0,1,0],[1,0,0],[1,0,0]])
    assert(labels.shape[0]==input_data_shape[4])
    print("Completed generating data and labels")

    ne_init_data=test_ngraph_resnet(iters,input_data,labels)
    test_neon_resnet(iters,input_data,labels,ne_init_data)
    

    

    

    