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

If each node in our graph has a single output and zero or more inputs, and the nodes correspond to differentiable values,
it is relatively easy to compute the derivative, although also more complicated than it will first appear.

Consider the simple case `w*x+b`.  We have zero-input nodes for `w`, `x` and `b`.  A `*` node has `w` and `x` as inputs, and
this node and the `b` node are inputs to a `+` node; five nodes in total.  Each node represents its value in a computation.
To compute the derivative with respect to `v` for a `+` node, we obtain nodes for the derivative with respect to `v` for its
two inputs and then construct a `+` node that has these as inputs.  But, with pythons arithmetic on objects constructing
graphs for us, we can simply say that for `a+b` we get `deriv(a,v)+deriv(b,v)`.  Similarly, for `a*b` we get
`deriv(a,v)*b+a*deriv(b,v)`.  For variables, we get 0 or 1, depending on whether `v` is the variable.

Of course it's not really quite that simple.  Even in the scalar case, you will end up with a graph containing
many constants that can be propagated and eliminated to produce a much simpler expression.

We will probably want to begin with a graph transformation, which converts generic nodes into the actual tensor-variant
of `+` that is occurring, i.e. make whatever broadcasting is happening explicit before computing their derivatives,
rather than having every operation that can broadcast attempt to deal with it.  Some nodes may have conditional
behavior, to support things like drop-out.




