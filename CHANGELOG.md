# ChangeLog

## v0.4.0 (2016-11-16):

Nervana graph is a library for developing frameworks that can efficiently run deep learning computations on a variety of compute platforms. In this preview release we introduce three primary API components:
- An API for creating computational `Nervana Graphs`.
- Two higher level frontend APIs (TensorFlow and Neon) utilizing the `Nervana Graph` API for common deep learning workflows
- A transformer API for compiling these graphs and executing them.

### Frontends
- The neon frontend offers an improved interface for increased composability/flexibility while leaving common use cases easy. We demonstrate this with MLP, convolutional, and RNN network examples on MNIST, CIFAR10, and Penn Treebank datasets.
- The tensorflow importer allows users to import existing tensorflow graphs and execute them using Nervana Graph transformers/runtimes. This importer currently only supports a subset of the tensorflow API, but this will be expanded over time.

### Nervana Graph API
- The Nervana Graph API consists of a collection of graph building functions all exposed in the `ngraph` module/namespace. (eg: `ngraph.sum(...)`)
- We include walkthrough examples to use this API for logistic regression and multilayer perceptron classification of MNIST digit images.
- With the introduction of named `Axes` we lay the foundation for frontend writers to reason about tensor axis without concern of memory layout or order (for future optimization against hardware targets which often have differing and specific requirements for batch axis orderings for example).

### Transformer API
- This release ships with two example transformers targetting CPU and GPU hardware targets.
- Both transformers support memory usage optimization passes.
- The GPU transformer also includes preliminary support for automatic kernel fusion/compounding for increased performance.
- Transformers allow users to register an included set of optional compiler passes for debug and visualization.
- The compiler pass infrastructure is slated to offer frontends/users similar flexibility to  what LLVM library offers for general purpose compilation.

### Known Issues
These are known issues which are being addressed:

- The transformer fusion and memory sharing optimizations are currently hampered by some of the tensor dimension reshaping introduced by the existing lowering passes. Thus both are turned off by default.
- Nervana Graph still requires a neon installation as a dependency.
- RNNs don't work well with longer sequences (longer than 30).
