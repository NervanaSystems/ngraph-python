# Graphitti: Computation Graphs

## What are we trying to do?

The term "Computation graph" is pretty vague, and means different things to different groups 
and people.  It is hoped that a common structure can serve as a basis for the varied needs.

We want to make it easier for users to correctly define networks that can be trained and run efficiently.
In frameworks such as Theano, TensorFlow, and MXNet, a computation is defined using python tricks
to construct something like an abstract syntax tree/graph for a tensor computation and a loss function
on it.  "Autodiff" can augment the computation graph with its derivative.

An alternative approach is to provide parameterized computations, layers in Neon, that can be chained.  The
implementation of each layer type includes hand-coded allocation, forward and backward propagation, and
layer to layer interfacing.  Layers make it easy for users to set up standard systems, but since each
layer is implemented independently of the others, there is no opportunity for cross-layer optimization.  
Furthermore, over time layers can pick up rather obscure options as they adapt to ever-changind demands.
Because propagation is hand-coded, there is room for error, and the more hand-coded layer code there is, the
harder it will be to make fundamental changes to the backends.  Finally, it would be very difficult for a
user to create a new kind of layer because of all the internal knowledge required, not to mention the
additional problems that would make for us if customer-implemented layers made it even more difficult
to change back-end organizations.  

In TensorFlow, some of the computations correspond directly to graph nodes, 
while others correspond to multiple graph nodes.  We can use the same approach with Neon by reimplementing
the current layers as graph operators, so that a model specified in terms of layers defines a graph. It would
be much easier for a user to define a new kind of layer as a graph transformation.  We could also look into
similarly transforming other layer-like frameworks, such as CNTK, into our graphs, as well as transforming
computations defined in other frameworks into our graphs.

## Ways we can use graphs

We want to ensure that the representation we choose for graphs makes it easier for users to define
computations and makes it easier for us to make those computations efficient.  For example, we would
like to be able to compute data flow, efficient batch sizes, etc.

# Specifying the graph

We could invent our own graph language and have a parser construct the graph.  Because python provides hooks for
letting objects handle many builtin operators, we can use the python parser to construct the graph for us.  For example,
`o+value` turns into `o.__add__(value)` and `value+o` turns into `o__radd__(value)` if `o` has the corresponding
methods.  Thus, if you manage to get one graph object into an expression such as `neon.tanh(W*x+b)` it is relatively
easy to have the entire expression construct a graph rather than execute a computation.

# Autodiff

Autodiff techniques have been around for some time.  For example,
http://www.qucosa.de/fileadmin/data/qucosa/documents/827/1206719130404-2230.pdf
and [Autograd](https://github.com/HIPS/autograd).



