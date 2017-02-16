# Containers -- A Proposal
Currently in ngraph, graphs are represented by a single output Op (or list of Ops) that the user 
wants computed (and implicitly all of the ops pointed to by its `args`, `initializers`, 
`control_deps`, and `forwards` attributes).

## Motivation
This low level representation of the computational graph has worked so far, but we see an 
opportunity to provide more powerful graph abstractions making it easier for users/frontend authors 
to reason about, compose, and interact with computations.

In addition, it is common for deep learning frameworks (Neon, Keras, etc) to offer _layers_ as a 
primary abstraction. These layers are then combined together using either containers (Neon and 
Keras) or functional APIs (Keras functional). This is a powerful and effective API, but why should 
this interface not extend to the lower level graph composition and (as yet, largely unrealized) 
higher level multi-model compositions?

Here are a list of challenges that we want measure any proposal against:

### Challenge 1 (C1) - Container Composition (Dataflow)
We want composition of computional graphs to extend straightforwardly to composition of layers, 
optimizers, and models.

### C2 - Computation Composition in the face of side effecting containers
Because op graphs (currently) support side-effects through `SetItemOp`s,
pseudo-random number generators, and initializers what does the composition of
computations look like (in addition to dataflow outlined in `C1`)?  It's
reasonable to see how inputs and outputs can be stitched together automatically
by name, but what about computations?

### C3 - Skip Connections
Consider you may have already existing layers connected in a sequential chain.
You then load this chain into your model and now want to connect skip
connections between layers. Certainly you can modify the connections and add
sum ops between layers, but can we do this in a non-mutating way with tombstone
edges etc?

### C4 - Batchnorm
In a batchnorm layer you want to describe two computations (one for training
and one for inference), that both represent a batchnorm, but each occurs during
a different phase of training. How do we represent this in a single logical
container? Do you force the choice at construction time, or allow 'late
binding' for the choice to be made later?

### C5 - Op Fusion
A transformer can replace some ops with a container to indicate a fused kernel. It'd be nice to be 
able to represent this fused kernel as a single container in a way that it retains its subsumed ops.

### C6 - Metadata
A container can contain metadata that applies to all children ops/containers. This helps with 
querying, debugging, etc.

### C7 - Serialized weight pairings
When you serialize some trained weights and then want to reuse those in a similar 

### C8 - Ergonomics
Almost without saying, any proposal requires that the abstraction is easy to work with or at least 
easy to build simple interfaces to. This includes composition (building up), encapsulation (fusion, 
rnn cells, etc), and querying (selecting out). 

## Proposal 1 (P1)
Considering that we are defining computational graphs for _computations_, we will use the concept of 
a _function_ as our starting place. Functions and computational graphs take one or more input 
arguments and transform them to one or more outputs:
```
Inputs                 Outputs
         +-----------+
         |           |
   +----->           |
         |           +----->
   +----->           |
         |           |
         +-----------+
```
We will call these computational units _containers_. As with functions, containers only describe a 
computation, and therefore the construction/instantiation of a container is independent from its 
execution. No surprises there.

In order to reason about the set of inputs and outputs, we must make many of the same choices that 
programming language (PL) designers have grappled with for decades. Named arguments, positional 
arguments, default arguments, variadic arguments, etc... Keras makes the choice of positional 
arguments in its functional API, but I think named arguments are probably better if we can get the 
syntax easy enough to work with.

To make this model a little more flexible we introduce the notion of a container's `ports`. We have
already witnessed two types of ports so far: input and output ports. They simply refer to some type
of handle of a container (in this case, data input and output handles). Input and output ports are a
natural place for tensor-like attributes to belong such as axes, shape information, striding, etc.

And finally, computations have attributes in the form of string key value pairs. This metadata is 
understood to apply to all children (which will be explained shortly) recursively. Examples include:
- `debug: true`
- `model_name: adversarial`
- `recurrent_cell_unroll: 4`
- `cudnn_recurrent_cell_standard_type: lstm`

### Nesting
Previous versions of containers proposed that containers would "own" children ops (drawn large boxes around smaller boxes inside). This was deemed to be a bit heavy handed of a representation, and instead we have containers point to 'children' ops through `CONTAINER` typed edges. This way containers can be a less restrictive way of labeleing several ops. There are two ways that containers could be represented (forgive the bad ascii art here). Semantic proposal 1:
```
                         +--------+
                         |        |
                         | LSTM   |
                         |        |
                     +--->        +-+
                     |   +--------| |
                     |              |
                     |              |
                 +---v--+      +----v--+
                 |      |      |       |
Input   +-------^+      +------>       +---------->  Output
                 |      |      |       |
                 +------+      +-------+
```

Semantic proposal 2:
```
                         +--------+
                         |        |
    Input   +------------> LSTM   +---------->   Output
                     +--->        +-+
                     |   +--------+ |
                     |              |
                     |              |
                     |              |
                 +---v--+      +----v--+
                 |      |      |       |
                 |      +------>       |
                 |      |      |       |
                 +------+      +-------+
```

This semantic level decision can be made later since both are equally easily represented by the graph structure.

### Querying
Given the nested design of containers, it seems logical that users will want to grab things out of 
them (variables, debug ops, RNN cells etc). We propose a querying mechanism similar to JSONPath, CSS 
selectors, or XPath that allows a lightweight combination of querying by position (child of), type, 
or metadata (key value attributes). Therefore queries such as (given in pure english):
- Give me all containers of type RNN (ie has metadata entry `rnn_cell: true`)
- Give me all `Dot` ops
- Give me all ops flagged for debug in the inference path.
- Give me all top level ops that aren't initializers


### Implementation Strategy
To proceed iteratively we're thinking of breaking containers into the following 
PRs:

- Clean up `op_graph.py` by:
  - Moving the adjoints into a separate transformer pass.
  - Removing One/Two/Zero D version of ops (Ops at the core ngraph level should imply computational intention and not worry about the dimensionality of their arguments).
  - Moving shape inference/broadcasting into a pass (out of constructor)
- Replace the `Op` class with a much simpler class with a `op_type` string parameter identifying the type of Op it is.
  - Along with this, simplify edge types into `DATAFLOW`, `CONTROL`, `INITIALIZER` and `CONTAINER`. This simplifies graph traversal and representation.
- Generalize compiler passes to match on these `op_type`s and other attributes.
