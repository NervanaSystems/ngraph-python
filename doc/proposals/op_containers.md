# Containers -- A Proposal
Currently in ngraph, graphs are represented by a single output Op (or list of Ops) that the user 
wants computed (and implicitly all of the ops pointed to by its `args`, `initializers`, 
`other_deps`, and `forwards` attributes).

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
syntax easy enough to work with. We can consider the addressing of input/output arguments as 
*DECISION 1 (D1)*.

To make this model a little more flexible we introduce the notion of a container's `ports`. We have
already witnessed two types of ports so far: input and output ports. They simply refer to some type
of handle of a container (in this case, data input and output handles). Input and output ports are a
natural place for tensor-like attributes to belong such as axes, shape information, striding, etc.

We can also consider another container port type: a `control` port. A control port is a handle to
reason about a set of side-effecting computations that can occur inside a container. Thus a
BatchNorm container would have a single output port (the input tensor after normalization) but have
one control ports for the updates to its internal state during training. Then when a user or
framework executed this layer during training, it would request the output (for the next block in
the dataflow graph) and then execute the training control port to update its state. This way the
ordering between dataflow computations and side effecting computations _can_ be explicitly
controlled (and defaults can be applied otherwise) and reasoned about independently (such as weight
reinitialization etc).

Also because of we need to reason about side effects (C2 or imperative mutations if you aren't a 
functional programming person), these computations can include subsets of ops as well.
```
       Control ports

     (train)  (inference)
         +       +
         |       |
      +-----------------+
      |  |       v      |
      |  | side effect  +-->  1
      |  |              |
+----->  |              +-->  2
      |  v              |
      | side effect     +-->  3
      |                 |
      +-----------------+
```

Additionally, instead of a container maintaining the dependency edges to its upstream parents, we
find it more natural and powerful for a container to own the connectivity information (edges)
between all of its children. These edges themselves can contain information such as type and
orderings.

And finally, computations have attributes/metadata in the form of string key value pairs. This metadata is 
understood to apply to all children (which will be explained shortly) recursively. Examples include:
- `debug: true`
- `model_name: adversarial`
- `recurrent_cell_unroll: 4`
- `cudnn_recurrent_cell_standard_type: lstm`

#### Addressing Challenges:

##### Challenge 1 (C1) - Container Composition (Dataflow)
Containers naturally address this.

##### C2 - Computation Composition in the face of side effecting containers
Control ports allow for explicit control and reasoning about side effecting computations occuring
inside a container.

##### C3 - Skip Connections
By replacing the chain container, you can maintain all of the individual layers while adding skip
connections. This is reasonably nice.

##### C4 - Batchnorm
See C2.

##### C5 - Op Fusion
This is naturally handled by containers.

##### C6 - Metadata
Check.

##### C7 - Serialized weight pairings
By giving containers UUIDs, we should be able to match things up pretty easily.

##### C8 - Ergonomics
This is work in progress.

### Implementation

_NOTE: Most of this is now deprecated by work with Evren on the serialization/containers joint effort_

Let's look at an example container implementation in Python (or at least, Python if Python had 
proper datatypes and types)::

```python
class Container(object):
    ports: dict<str, Port>,
    children: [Container],
    edges: [Edge],
    metadata: dict<str, str>,
```
Here the inputs and outputs are merely string handles to Ops, and the computations are handles to 
lists of Ops. This gives us named arguments and outputs (for decision D1). This allows for 
computations to point to combinations of output Ops and SetItem ops:
```
       Computations

     (train)  (inference)
         +       +
         |       |
      +-----------------+
      |  |       v      |
      |  |  1 and       +-->  1
      |  |  SetItemOp 2 |
+----->  |              +-->  2
      |  v              |
      | (2 & 3) and     +-->  3
      | SetItemOp 1     |
      +-----------------+
```

But how can we make a Sequential container that itself contains multiple layer containers? This is 
the role of the ``children`` attribute. A container can have zero or more children containers. The 
outputs, inputs, and computations are then pointing to input, output and computation ops of their 
children (recursively). 

Then a container can be built out of smaller containers:
```
                         CC1       CC2
                          +         +
                          |         |
                          |         |
                          |         |
         +----------------+---------+------------+
         |                                       |
         |         C1   C2        C3             |
CI1 +---->          +   +          +             |
         |          |   |          |             +----> CO1
         |       +--+---+-+    +---+---+         |
         | I1 +-->        +---->       +--> O1   |
         |       +--------+    +-------+         |
         |                                       |
         +---------------------------------------+

```
Here a linear chain of two containers creates a larger container with each sub container as a child.  
Here it is obvious that the singular output of the first subcontainer should be connected to the 
singular input of the second subcontainer leaving the parent container with a single input (CI1 == 
I1) and a single output (CO1 == O1). This could be achieved with a simple syntax. Anything like:
```
layer1 = Affine()
layer2 = Tanh()

parent = layer1.chain(layer2)
```
or a more functional
```
parent = Tanh()(Affine())
```
And allows for renaming the parent input and output ports with something like (in yet another syntax 
proposal):
```python
parent = Container.chain(
           inputs=['image'],
           outputs=['class_label'],
           Affine( ),
           Tanh( )
        )
```

Though the trickier question is what to do with the computations, it's not clear that you want to 
create the cartesian product of childrens' computations to produce the parent's computations. I 
think the thing to do here is have some default behavior for simple cases: create an 'all' 
computation for the parent that encompasses all computations of the children and allow frontends to 
override this behavior (for cases like Batchnorm where you want to produce a training and inference 
computation which contains all the childrens' training and inference computations respectively.

We call the computation composition question *DECISION 2 (D2)*. 

In general, users can compose arbitrarily complex sets of containers:
```
                         CC1       CC2
                          +         +
                          |         |
         +----------------+---------+------------+
         |       C1   C2        C3               |
         |        +   +          +               |
         |        |   |        +-+-----+         |
         |        |   |   +---->   2   +--> O1   +----> CO1
CI1 +---->      +-+---+-+ |    +-------+         |
         | I1   |       +-+                      |
         |  +--->   1   |----->  O2              +----> CO2
         |      |       +-+                      |
         |      +-------+ |         C4           |
         |                |         +            +----> CO3
         |                |     +---+---+        |
CI2 +---->                +----->       |   O3   |
         |                      |   3   +-->     |
         |             I2 +----->       |        |
         |                      +-------+        |
         |                                       |
         +---------------------------------------+
```
Here is a parent container with three children containers. This requires an API to allow for 
'wiring' up the input and output ports arbitrarily between containers. This can 
be done in this example with the following input output label 'patch' pairs:
```python
Container.named_merge(
    patches=[('layer_1_o1', 'layer2_i1'),
             ('layer_1_o3', 'layer_3_i1')],
    containers=[
        layer_1,
        layer_2,
        layer_3
        ]
    )
```

I propose that a container itself actually houses these edges, to avoid mutating the Ops in the 
children itself.

### Problem: Enforcing Invariants

One potential problem I forsee with this design is that there's no way to enforce that the Op an 
output of a container points to actually exists inside that container, or that the inputs properly 
corresponds to the set of outputs. There are also other invariants/properties we'd like to ensure:
- Computations only point to output ops that are also `output`s of that container.
- An unbound output of a child container should be exposed as an output of the parent container 
    (although perhaps this 'masking' effect could be useful (if confusing)?)

If our containers are immutable, then this makes this problem much simpler, as we don't need to 
worry about mutation breaking containers or the invariants that are setup at creation time.

### Querying
Given the nested design of containers, it seems logical that users will want to grab things out of 
them (variables, debug ops, RNN cells etc). We propose a querying mechanism similar to JSONPath, CSS 
selectors, or XPath that allows a lightweight combination of querying by position (child of), type, 
or metadata (key value attributes). Therefore queries such as (given in pure english):
- Give me all containers of type RNN (ie has metadata entry `rnn_cell: true`)
- Give me all `Dot` ops
- Give me all ops flagged for debug in the inference path.
- Give me all top level ops that aren't initializers

### Traversal
For graph passes we want to expose convenient ways to traverse the graph of containers. At the simplest, we want to make it easy to do things like:
```python
# In Python
for op in my_container:
  # do something with every op

for op in my_container.sorted():
  # every op starting with the inputs

for op in my_container.sorted(skip_containers=lambda c: if 'rnn' in c.metadata):
  # All ops in topological sorted order, but don't traverse into RNN cells
```

### Implementation Strategy
To proceed iteratively I'm thinking of breaking containers into the following 
PRs:

#### Add Container class with tests
Use it with MNIST (but without tie-ins to transformers or the factory functions 
by taking the ngraph out of the container before passing to transformer.

We'll leave ops connected across container boundaries for now instead of having 
containers harbor edges between children. We can handle connectivity later.

#### Traversal and graph passes
Show how traversal works and have the graph passes use it.

From there we'll decide the next most important steps.

# Proposal 2
A further extension of proposal 1 is that we unify the notion of an `Op` and `Container`. Op's after 
all can be considered as Containers with a single output, a single computation, and no Children. 

This has a very nice unifying property to computational graphs, but the full ramifications are 
unclear.

After further discussion, this is underway in our serialization/container joint
implementation plan (since both affect the op_graph definition so much).

# Proposal 3

_NOTE: This has not garnered much interest and is not currently being pursued._

A further extension of proposal 1 (independent of proposal 2) is to stop storing edge information in 
the Op's themselves, but instead store them in the Containers. Combining this with `tombstone` edges 
(phantom edges marking deletion) we could support non mutating graph modifications (just layer 
another container on top of a model and delete edges you no longer want (with tombstones) and add in 
whatever you do want.

I haven't thought this out completely, but it would allow for things like adding ResNet connections 
by adding a single wrapping container around a Container chain while still allowing that container 
to be un-modified. This has nice properties for weight saving/resaving etc..
