# Caffe importer for ngraph

    Two options were supported by this importer
        1. cmd line mode => give all the options in the cmdline just like "caffe"
        2. calling importer through python script 

## Sum example
    
    option1: 
        python cf_importer/importer.py compute -model examples/sum.prototxt  -name C,D
    option2: 
        using the below script in a python file and calling it

```python
from __future__ import print_function
import ngraph.transformers as ngt
from ngraph.frontends.caffe.cf_importer.importer import parse_prototxt
#path to the topology file
model = "sum.prototxt"
#import graph from the prototxt
op_map = parse_net_def(model,verbose=True)
#get the op handle for any layer
op = op_map.get("D")
#execute the op handle
res = ngt.make_transformer().computation(op)()
print("Result is:",res)

```

## Example models

- MNIST MLP - TODO
- Logistic regression - TODO
