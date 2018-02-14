# Intel® nGraph™ 

### An Intermediate Representation, Compiler, and Executor for Deep Learning

*Updated: February 13, 2018* 

Welcome to the Intel nGraph repo. While we're transitioning our main project 
from Python and [preparing to open-source our C++ code base] to the community, 
you can browse here to learn a bit about the roots of the [legacy] project.  


## Why did we build nGraph?

When Deep Learning (DL) frameworks first emerged as the vehicle for training and
inference models, they were designed around kernels optimized for a particular 
platform. As a result, many backend details -- which normally should get 
encapsulated within the kernel-framework implementation -- were getting muddied 
up in the frontend framework, and sometimes even in the the model itself. This 
problem, which remains largely unchanged today, makes the adaptability and 
portability of DL models to new frameworks or more advanced backends inherently 
complex and expensive. 

The traditional approach means that an algorithm developer cannot easily adapt 
his or her model to other frameworks. Nor does the developer have the freedom to 
experiment or test a model with different backends or on better hardware; teams 
get locked into a framework and their model either has to be entirely rewritten 
for the new framework, or re-optimized with the newer hardware and kernel in 
mind. Furthermore, any hard-earned optimizations of the model (usually focused 
on only one aspect, such as training performance) from its original topology 
break with a change, update, or upgrade to the platform.  

We designed the Intel nGraph project to substantially reduce these kinds of 
engineering complexities. While optimized kernels for deep-learning primitives 
are provided through the project and via libraries like Intel® Math Kernel Library 
for Deep Neural Networks (Intel® MKL-DNN), there are several compiler-inspired 
ways in which performance can be further optimized. 


## How does it work in practice?

Install the nGraph library and write or compile a framework with the library, 
in order to run training and inference models. Using the command line, 
specify the backend you want to use. Our Intermediate Representation (IR) layer 
handles all the hardware abstraction details and frees developers to focus on 
their data science, algorithms and models, rather than on machine code.  

At a more granular level of detail: 

* The **nGraph core** uses a strongly-typed and platform-neutral stateless graph 
  representation for computations. Each node, or *op*, in the graph corresponds
  to one step in a computation, where each step produces zero or more tensor
  outputs from zero or more tensor inputs.

* There is a **framework bridge** for each supported framework which acts as an 
  intermediary between the *ngraph core* and the framework. A **transformer** 
  then plays a similar role between the ngraph core and the various execution 
  platforms.

* **Transformers** handle the hardware abstraction; they compile the graph with 
  a combination of generic and platform-specific graph transformations. The result 
  is a function that can be executed from the framework bridge. Transformers also 
  allocate and deallocate, as well as read and write tensors under direction of the
  bridge.
  
You can read more about design decisions and what is tentatively in the 
pipeline for backends and development in our [ARXIV abstract and conference paper]:


[preparing to open-source our C++ code base]:http://ngraph.nervanasys.com/docs/cpp/ 
[legacy]:legacy_README.md
[ARXIV abstract and conference paper]:https://arxiv.org/pdf/1801.08058.pdf
