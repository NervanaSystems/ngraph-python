# ngraph modularity and C APIs

```
         1                             2                                    3

+-------------------+      +-------------------------+        +--------------------------+
|      C API        |      |C API (Generic interface)|        | C API (generic interface)|
+-------------------+      +-------------------------+        +--------------------------+
|                   |      |                         |        |                          |
|     Op Graph      |      |      Transformer        |        |     Computation          |
|     (ngraph)      |      |      Implementation     |        |     Implementation       |
+-------------------+      |                         |        |                          |
                           +------------+------------+        +-------------+------------+
                                        |              .                    |
                                        |               .                   |
                                        |                .  6               |     5
                                        |   4             .                 |
                                        |                   . . +-----------+-----------+
                              +---------+---------+             |                       |
                              |      C API        |             |   Execution Runtime   |
                              |  (for transformer |             |                       |
                              |  implementors)    |             +-----------------------+
                              +-------------------+
                              |                   |
                              |  Common           |
                              |  Transformer      |
                              |  Passes           |
                              |  Library          |
                              |                   |
                              +-------------------+
```

ngraph API (1):
- Graph/container construction
- Serialiaztion/Deserialization

Transformer interface (2):
- Instantiation
- def compile(op_graph) -> computation
- Control of passes used by transformer (advanced use)

Computation interface (3):
- Data input/output
- Runtime init/teardown
- Weight Serialization/deserialization
- Checkpoint/resume

Common Transformer Library (4):
- def pass(op graph) -> op graph
- Standardized C API makes it easy for passes/transformers written in other languages

Execution Runtime (5)
- Split away from the Transformer for runtime concerns (vs compilation)
- Allows for compilation and/or training to occur separately from training/deployment
- Integration point (potentially) with DL SDK inference engine
- No C API standardized because we leave the details up to the implementor to hide behind the transformer/computation interface.

Transformer and Execution Linkage (6): A particular transformer implementation
(GPU) will almost always come with a paired runtime to service the computations
produced by that transformer. This is not set in stone, more to emphasize that we expect coupling between transformers and runtimes.

Given this coupling that exists between transformers and runtimes, why split these functionalities? We get to this below in `Motivations`.

### Example: Lattice Search
For the graph search transformation idea that Maciej proposed, this would all be implementable as a single transformer. From the API user's perspective, they still are merely inputting an ngraph into a transformer and then obtaining a computation as a result (the transformation process hiding the complicated search with database of computation costs and the runtime hiding multiple devices behind it). 

### Typical user interaction
1. User creates ngraph
2. Instantiates transformer
3. Compiles ngraph using transformer and receives a computation
4. Interacts with computation object, saves it out etc..

### Motivations behind splitting Transformer and Runtime
1. Separation of concerns (compiling/lowering the graph vs allocation of memory, shuttling data etc)
2. First steps for deployment story
3. Potentially exposes imperative interface for advanced/research users.
4. Lowers barrier to new transformers: if I want to try a completely new way of compiling ngraphs (the lattice method perhaps) then its entirely possible I could reuse an existing runtime (GPU?) while writing a completely new transformer.

To say more about the imperative interface: computation implementations could optionally support a `get_runtime_context` method. This would then give undocumented (and unsupported?) access to the underlying execution runtime for calls to kernels etc.. Would this be enough for research users? Or would it feel too hacky?

### Python Libraries
The python ngraph library should be written to use the C-API itself even if/when components themselves are written in Python. This dog-fooding of the public API will keep us honest about the usability/functionality of our public API. This includes neon and the other frontends.
