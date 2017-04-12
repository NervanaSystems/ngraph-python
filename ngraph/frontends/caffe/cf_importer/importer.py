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
from __future__ import print_function
import ngraph as ng
import ngraph.transformers as ngt
from ngraph.frontends.caffe.cf_importer.ops_bridge import OpsBridge
import ngraph.frontends.caffe.cf_importer.ops_constant
import ngraph.frontends.caffe.cf_importer.ops_binary
import argparse

from google.protobuf import text_format
try:
    import caffe_pb2
except:
    raise ImportError('Must be able to import Caffe modules to use this module')

supported_layers = ["Eltwise","DummyData"] 

#TBD Adding support for below layers   
#AbsVal, Accuracy, ArgMax, 
#BNLL, BatchNorm, BatchReindex, Bias, Concat, ContrastiveLoss, Convolution, 
#Crop, Deconvolution, DetectionEvaluate, DetectionOutput, Dropout,ELU, 
#Eltwise, Embed, EuclideanLoss, Exp, Filter, Flatten,HDF5Output, HingeLoss,
#Im2col, InfogainLoss, InnerProduct, Input, LRN, LSTM, LSTMUnit, Log, MVN,
#MultiBoxLoss, MultinomialLogisticLoss, Normalize, PReLU, Parameter, Permute, 
#Pooling, Power, PriorBox, RNN, ReLU, Reduction, Reshape,SPP, Scale, Sigmoid,
#SigmoidCrossEntropyLoss, Silence, Slice, SmoothL1Loss,Softmax, 
#SoftmaxWithLoss, Split, TanH, Threshold, Tile, 
#"Data","AnnotatedData","HDF5Data","ImageData","MemoryData","VideoData","WindowData"


    
def parse_prototxt(model_txt=None,solver_txt=None,caffemodel=None,verbose=False):
    """
    This function parses and creates a graph of ngraph ops corresponding to each layer 
    in the prototxt  
    Arguments:
        model_txt: prototxt file of the Neural net topology
        solver_txt: protoxt file of the solver to train the neural net
        caffemodel: parameters (weights/biases) to be loded into the model
    return : 
        Dictionary of the ngraph ops whose keys are the layer names of the prototxt
    """
    
    ops_bridge = OpsBridge() #opsBridge constructor
    data_layers = [ l for l in supported_layers if "Data" in l]
    name_op_map = {} #graph data structure

    if model_txt is None and solver_txt is None:
        raise ValueError ("Either model prototxt or solver prototxt is needed")

    model_def = caffe_pb2.NetParameter()
    solver_def = caffe_pb2.SolverParameter()

    #TBD: Addding support to load weights from .caffemodel

    if solver_txt is not None:
        with open(solver_txt, 'r') as fid:
            text_format.Merge(fid.read(), solver_def)
    
        if not solver_def.HasField("net"):
            raise ValueError ('model prototxt is not available in the solver prototxt')
        else:
            modelFile = solver_def.net
    else:
        with open(model_txt, 'r') as fid:
            text_format.Merge(fid.read(),model_def)

    netLayers = model_def.layer

    for layer in netLayers:
        if verbose:
            print("\nLayer: ",layer.name," Type: ",layer.type)
        if layer.type not in supported_layers:
            raise ValueError ('layer type', layer.type ,' is not supported')
        if len(layer.top) > 1 and layer.type not in data_layers:
            raise ValueError ('only "Data" layers can have more than one output (top)')

        input_ops = [] 
        for name in layer.bottom:
            if name_op_map.has_key(name):
                input_ops.append(name_op_map[name])
            elif layer.type not in data_layers:
                raise ValueError ("Bottom layer:",name ," is missing in the prototxt") 
        #get the ngraph op from bridge
        out_op = ops_bridge(layer,input_ops)

        if out_op is None:
            print("!!! Unknown Operation '{}' of type '{}' !!!"
                  .format(layer.name, layer.type))
        if verbose:
            print("input Ops:",input_ops)
            print("output Op:",[out_op])

        if name_op_map.has_key(layer.name):
            raise ValueError('Layer ',Layer.name,' already exists. Layer name should be unique')

        #update dictionary
        name_op_map[layer.name] = out_op

        # handle special cases like relu,dropout etc
        if layer.top == layer.bottom:
            if name_op_map.has_key(layer.top):
                name_op_map[layer.top] = out_op

    return name_op_map

class CaffeCLI:
    """
    A class to manage the CLI arguments
    """
    def __init__(self):
        self.caffe_cli_emulator()
        self.validate_cmdline_args()

    def validate_cmdline_args(self):
        """
        To validate the required arguments given for a task
        """
        args = self._cmdargs

        if args['verbose']:
            print((args))

        if args['mode'] in ['train'] and args['solver'] is None:
                raise ValueError ("solver prototxt is required")

        if args['mode'] in ['test','time','compute'] and args['model'] is None:
                raise ValueError ("model prototxt is required")

        if args['mode'] in ['test'] and args['weights'] is None:
                raise ValueError ("file .caffemodel is required")

        if args['mode'] in ['compute'] and args['name'] is None:
                raise ValueError ("Layer name is required to compute")
        
    def get_cmd_args(self):
        return self._cmdargs

    def compute(self,ops):
        """
        To compute the value for the given layer 
        Arguments:
            ops : ngraph ops of the caffe layers to compute
        Return:
            return the value of the given layers
        """
        return ngt.make_transformer().computation(ops)()

    def caffe_cli_emulator(self): 
        """
        This function handles the cmd line arguments given by the user
        """
        parser = argparse.ArgumentParser()

        parser.add_argument("mode", help="Set the mode to run. \
            'compute' is not available in original caffe",
            choices=['train','test','time','compute'])
        parser.add_argument("-weights",
            help="Pretrained .caffemodel file")
        parser.add_argument("-engine",help="HW Engine to run sequence ",default="CPU",
            choices=['CPU','GPU','KNL','HETR'])
        parser.add_argument("-forward_only", 
            help="Execute only forward pass.",type=bool)
        parser.add_argument("-gpu",
            help="Run in GPU mode on given device IDs separated by ','")
        parser.add_argument("-iterations",type=int,default=50,
            help="The number of iterations to run. Dafault:50")
        parser.add_argument("-solver",
            help="path to the solver definition .prototxt file")
        parser.add_argument("-model",
            help="path to the model definition .prototxt file")
        parser.add_argument("-phase",
            help="network phase (TRAIN or TEST). Only used for 'time'",
            default='TRAIN',choices=['TRAIN','TEST'])
        parser.add_argument("-name",
            help="layer names to compute",type=str)
        parser.add_argument("-verbose",help = " debug prints",default=False)

        self._cmdargs = vars(parser.parse_args())
        

if __name__ == '__main__':

    cli = CaffeCLI()
    args = cli.get_cmd_args()

    solver_txt = None
    model_txt = None
    caffemodel = args['weights']

    if args['mode'] == 'train':
        solver_txt= args['solver']
    else:
        model_txt = args['model']

    op_map = parse_prototxt(model_txt,solver_txt,caffemodel,verbose=args['verbose'])

    if args['mode'] == 'compute':
        layers = args['name'].split(',')
        ops =[]
        for l in layers:
            op = op_map.get(l)
            if not op:
                print("Layer ",l," does not exists in the prototxt")
            else:
                ops.append(op)
        result = cli.compute(ops)

    for out in result:
        print(out)
