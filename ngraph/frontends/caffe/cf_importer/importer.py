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
from ngraph.frontends.caffe.cf_importer.ops_bridge import OpsBridge
import ngraph as ng
import ngraph.transformers as ngt
from ngraph.testing import ExecutorFactory as ef
import os
import argparse

from google.protobuf import text_format
try:
    import caffe_pb2
except:
    raise ImportError('Must be able to import Caffe modules to use this module')

class CaffeImporter:
    """
    Importer for Caffe prototxt 
    Arguments:
        args : cmd line arguments for a task
    Returns:
        instant of CaffeImporter
    """

    def __init__(self,args):
        self.reset(args)
        self.init_args()
        self.validate_cmdline_args()
        self.parse_net_def()

    def reset(self,args):
        """
        Resets importer states and handles the command line options
        """
        self._name_op_map = dict()
        self._ops_bridge = OpsBridge()
        self._net_def = None
        self._solver_def = None
        self._cmdargs = args
        self._supported_layers = ["Eltwise","DummyData"] 
        self._data_layers = [ l for l in self._supported_layers if "Data" in l]

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
    
    def init_args(self):
        """
        User can provide the args as a dictionary to the importer.
        This function is to provide the flexibility to initialize 
        to defalut values for arguments not provided by user
        """
        args = self._cmdargs

        if not args.has_key('name'):
            args['name'] = None
        if not args.has_key('solver'):
            args['solver'] = None
        if not args.has_key('model'):
            args['model'] = None
        if not args.has_key('forward_only'):
            args['forward_only'] = None
        if not args.has_key('weights'):
            args['weights'] = None
        if not args.has_key('gpu'):
            args['gpu'] = None
        if not args.has_key('verbose'):
            args['verbose'] = False

    def validate_cmdline_args(self):
        """
        To validate whether all the required arguments given for a task
        """
        args = self._cmdargs

        if args['mode'] in ['train'] and args['solver'] is None:
                raise ValueError ("solver prototxt is required")

        if args['mode'] in ['test','time','compute'] and args['model'] is None:
                raise ValueError ("model prototxt is required")

        if args['mode'] in ['test'] and args['weights'] is None:
                raise ValueError ("file .caffemodel is required")

        if args['mode'] in ['compute'] and args['name'] is None:
                raise ValueError ("Layer name is required to compute")
        
    def parse_net_def(self):
        """
        Imports a net_def to ngraph. Creates a graph with ngraph ops
        corresponding to each layer in the given prototxt 
        """
        
        self._net_def = caffe_pb2.NetParameter()
        self._solver_def = caffe_pb2.SolverParameter()
        #TBD: Addding support to load weights from .caffemodel

        args = self._cmdargs

        if args['mode'] == 'train':
            with open(args["solver"], 'r') as fid:
                text_format.Merge(fid.read(), self._solver_def)
        
            if not self._solver_def.HasField("net"):
                raise ValueError ('model prototxt is not available in the solver prototxt')
            else:
                modelFile = self._solver_def.net
        else:
            modelFile = args['model']

        with open(modelFile, 'r') as fid:
            text_format.Merge(fid.read(), self._net_def)

        netLayers = self._net_def.layer

        for layer in netLayers:

            if args['verbose']:
                print("\nLayer: ",layer.name," Type: ",layer.type)

            if layer.type not in self._supported_layers:
                raise ValueError ('layer type', layer.type ,' is not supported')
            if len(layer.top) > 1 and layer.type not in self._data_layers:
                raise ValueError ('only "Data" layers can have more than one output (top)')

            input_ops = [] 
            for name in layer.bottom:
                if self._name_op_map.has_key(name):
                    input_ops.append(self._name_op_map[name])
                elif layer.type not in self._data_layers:
                    raise ValueError ("Bottom layer:",name ," is missing in the prototxt") 

            out_op = self._ops_bridge(layer,input_ops)

            if out_op is None:
                print("!!! Unknown Operation '{}' of type '{}' !!!"
                      .format(layer.name, layer.type))
            if args['verbose']:
                print("input Ops:",input_ops)
                print("output Op:",[out_op])

            if self._name_op_map.has_key(layer.name):
                raise ValueError('Layer ',Layer.name,' already exists. Layer name should be unique')

            self._name_op_map[layer.name] = out_op

            # handle special cases like relu,dropout etc
            if layer.top == layer.bottom:
                if self._name_op_map.has_key(layer.top):
                    self._name_op_map[layer.top] = out_op

    def compute(self,name):
        """
        To compute the value for the given layer 
        Arguments:
            name : name of the layers to compute
        Return:
            return the final value of the given layer
        """
        layers = name.split(',')
        ops =[]
        for l in layers:
            if not self._name_op_map.has_key(l) :
                print("Layer ",l," does not exists in the prototxt")
            else:
                ops.append(self._name_op_map[l])

        with ef() as ex:
            return ex.executor(ops)()
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
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

    args = vars(parser.parse_args())
    if args['verbose']:
        print((args))

    importer = CaffeImporter(args)

    if args['mode'] == 'compute':
            print("\n",importer.compute(args['name']))

