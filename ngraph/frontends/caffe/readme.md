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
from ngraph.frontends.caffe.cf_importer.importer import CaffeImporter
import ngraph.transformers as ngt

model = "sum.prototxt"
#import graph from the prototxt
importer = CaffeImporter()
importer.parse_net_def(model,verbose=True)
#get the op handle for any layer
op = importer.get_op_by_name("D")
#execute the op handle
res = ngt.make_transformer().computation(op)()
print("Result is:",res)

```

## Example models

- MNIST MLP - TODO
- Logistic regression - TODO
