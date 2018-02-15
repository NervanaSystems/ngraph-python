# Intel® nGraph™ 

### An Intermediate Representation, Compiler, and Executor for Deep Learning

*Updated: February 13, 2018* 

Welcome to the Intel nGraph repo. While we're transitioning our main project 
from Python and [preparing to open-source our C++ code base] to the community, 
you can browse here to learn a bit about the roots of the [legacy] project.  


## Why did we build nGraph?

When Deep Learning (DL) frameworks first emerged as the vehicle for training and 
inference models, they were designed around kernels optimized for a particular 
platform. As a result, many backend details were being exposed in the model 
definitions, making the adaptability and portability of DL models to other or 
more advanced backends inherently complex and expensive.

The traditional approach means that an algorithm developer cannot easily adapt 
his or her model to different backends. Making a model run on a different 
framework is also problematic because the developer must separate the essence of 
the model from the performance adjustments made for the backend, translate to 
similar ops in the new framework, and finally make the necessary changes for 
the preferred backend configuration on the new framework.

We designed the Intel nGraph project to substantially reduce these kinds of 
engineering complexities. While optimized kernels for deep-learning primitives 
are provided through the project and via libraries like Intel® Math Kernel Library 
for Deep Neural Networks (Intel® MKL-DNN), there are several compiler-inspired 
ways in which performance can be further optimized. 


## How does it work in practice?

Install the nGraph library and write or compile a framework with the library, 
in order to run training and inference models. Using the command line on any 
supported system (currently Linux) specify the backend you want to use. Our 
Intermediate Representation (IR) layer handles all the hardware abstraction 
details and frees developers to focus on their data science, algorithms and 
models, rather than on machine code.

At a more granular level of detail: 

* The **nGraph core** creates a strongly-typed and platform-neutral stateless 
  graph representation of computations. Each node, or *op*, in the graph 
  corresponds to one step in a computation, where each step produces zero or 
  more tensor outputs from zero or more tensor inputs.

* We've developed a **framework bridge** for each supported framework; it acts 
  as an intermediary between the *ngraph core* and the framework. A **transformer** 
  then plays a similar role between the ngraph core and the various execution 
  platforms.

* **Transformers** handle the hardware abstraction; they compile the graph with 
  a combination of generic and platform-specific graph transformations. The 
  result is a function that can be executed from the framework bridge. 
  Transformers also allocate and deallocate, as well as read and write tensors 
  under direction of the bridge.
  
You can read more about design decisions and what is tentatively in the pipeline 
for backends and development in our [ARXIV abstract and conference paper]:


[preparing to open-source our C++ code base]:http://ngraph.nervanasys.com/docs/cpp/ 
[legacy]:legacy_README.md
[ARXIV abstract and conference paper]:https://arxiv.org/pdf/1801.08058.pdf
