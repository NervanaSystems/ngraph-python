# ----------------------------------------------------------------------------
# Copyright 2016 Nervana Systems Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------
from builtins import range, zip
import tempfile

from ngraph.op_graph.axes import TensorDescription
from ngraph.transformers.gpu.util import _get_sm_count
from ngraph.transformers.flex2 import Flex, flex16

from pycuda.compiler import SourceModule

import numpy as np

flex_verbose = False


_op_templates = {
    "assign": r"%(out)s = %(x)s;",
    "finite": None,
    "neg": r"%(out)s = -%(x)s;",
    "abs": r"%(out)s = abs(%(x)s);",
    "sqrt": r"%(out)s = sqrtf(%(x)s);",
    "sqr": r"%(out)s = %(x)s * %(x)s;",
    "exp": r"%(out)s = expf(%(x)s);",
    "log": r"%(out)s = logf(%(x)s);",
    "rcp": r"%(out)s = 1.0f / %(x)s;",
    "exp2": r"%(out)s = exp2f(%(x)s);",
    "log2": r"%(out)s = log2f(%(x)s);",
    "sig": r"%(out)s = 1.0f / (1.0f + expf(-%(x)s));",
    "sig2": r"%(out)s = 1.0f / (1.0f + exp2f(-%(x)s));",
    "tanh": r"%(out)s = tanhf(%(x)s);",
    "tanh2": r"%(out)s = (exp2f(2.0f * %(x)s) - 1.0f) / (exp2f(2.0f * %(x)s) + 1.0f);",
    "transpose": None,
    "safelog": r"%(out)s = (%(x)s > 0.0f) ? logf(%(x)s) : -50.0f;",
    "add": r"%(out)s = %(x)s + %(y)s;",
    "sub": r"%(out)s = %(x)s - %(y)s;",
    "mul": r"%(out)s = %(x)s * %(y)s;",
    "div": r"%(out)s = %(x)s / %(y)s;",
    "mod": r"%(out)s = int(%(x)s) %% int(%(y)s);",
    "eq": r"%(out)s = %(x)s == %(y)s;",
    "ne": r"%(out)s = %(x)s != %(y)s;",
    "lt": r"%(out)s = %(x)s < %(y)s;",
    "le": r"%(out)s = %(x)s <= %(y)s;",
    "gt": r"%(out)s = %(x)s > %(y)s;",
    "ge": r"%(out)s = %(x)s >= %(y)s;",
    "pow": r"%(out)s = powf(%(x)s, %(y)s);",
    "minimum": r"%(out)s = fminf(%(x)s, %(y)s);",
    "maximum": r"%(out)s = fmaxf(%(x)s, %(y)s);",
    "onehot": r"%(out)s = (%(index)s == %(x)s);",
    "dot": None
}

_redop_templates = {
    "prod": r"%(out)s = %(out)s * %(x)s;",
    "sum": r"%(out)s = %(out)s + %(x)s;",
    "max": r"%(out)s = fmaxf(%(out)s, %(x)s);",
    "min": r"%(out)s = fminf(%(out)s, %(x)s);",
    "argmax": r"if(%(x)s > %(y)s) {%(out)s = %(index)s; %(y)s = %(x)s;}",
    "argmin": r"if(%(x)s < %(y)s) {%(out)s = %(index)s; %(y)s = %(x)s;}"
}

_redop32_templates = {
    "prod": r"%(out)s = %(out)s * __shfl_xor(%(out)s, i);",
    "sum": r"%(out)s = %(out)s + __shfl_xor(%(out)s, i);",
    "max": r"%(out)s = fmaxf(%(out)s, __shfl_xor(%(out)s, i));",
    "min": r"%(out)s = fminf(%(out)s, __shfl_xor(%(out)s, i));",
    "argmax": r"""temp_idx = __shfl_xor(%(out)s, i);
%(indent)stemp_val = __shfl_xor(%(y)s, i);
%(indent)sif(temp_val > %(y)s) {%(out)s = temp_idx; %(y)s = temp_val;}""",
    "argmin": r"""temp_idx = __shfl_xor(%(out)s, i);
%(indent)stemp_val = __shfl_xor(%(y)s, i);
%(indent)sif(temp_val < %(y)s) {%(out)s = temp_idx; %(y)s = temp_val;}""",
}

# Key for conversion template is (src_fmt, dst_fmt)
_conversion_templates = {
    ("half", "float"): r"%(out)s = __half2float(%(in)s);",
    ("float", "half"): r"%(out)s = __float2half(%(in)s);",
    ("flex", "float"): r"%(out)s = %(scale)s * %(in)s;",
    ("float", "flex"): r"%(out)s = fp32_to_int16(%(scale)s * %(in)s);",
}
_default_conversion = r"%(out)s = %(in)s;"

_redop_inits = {
    "prod": "1.0f",
    "sum": "0.0f",
    "max": "-FLT_MAX",
    "min": "FLT_MAX",
    "argmax": "0",
    "argmin": "0"
}

_item_loop_template = "for(int item = idx%(loopidx)s; item < loopmax; item += blockDim.x)"

_index_template1 = r"%(index)s = %(item)s * %(stridea)s;"
_index_template20 = r"%(index)s = %(item)s * %(stridea)s + idx1 * %(strideb)s;"
_index_template21 = r"%(index)s = idx0 * %(stridea)s + %(item)s * %(strideb)s;"
_index_template30 = \
    r"%(index)s = %(item)s * %(stridea)s + idx1 * %(strideb)s + idx2 * %(stridec)s;"
_index_template31 = \
    r"%(index)s = idx0 * %(stridea)s + %(item)s * %(strideb)s + idx2 * %(stridec)s;"
_index_template32 = \
    r"%(index)s = idx0 * %(stridea)s + idx1 * %(strideb)s + %(item)s * %(stridec)s;"

_load_template = r"%(buffer)s[%(index)s]"

_store_template = r"%(buffer)s[%(index)s]"

_redstore_template = r"if(idx%(loopidx)s == 0) {%(store)s}"

_red32_template = r"""
    #pragma unroll
    for (int i = 16; i > 0; i >>= 1)
    {
        %(statement)s
    }
"""

_red_template = r"""
    // Reduce within warp
    #pragma unroll
    for (int i = 16; i > 0; i >>= 1)
    {
        %(statement)s
    }
    if (!(threadIdx.x & 0x1f))
    {
        %(shared_buffer)s[threadIdx.x >> 5] = %(out)s;
    }

    __syncthreads();

    // Reduce between warps (max of 32 warps since block has max 1024 threads)
    if (threadIdx.x < 32)
    {
        %(out)s = %(shared_buffer)s[threadIdx.x];

        #pragma unroll
        for (int i = 16; i > 0; i >>= 1)
        {
            %(statement)s
        }
    }

    if (threadIdx.x == 0)
    {
        %(shared_buffer)s[0] = %(out)s;
    }

    __syncthreads();

    %(out)s = %(shared_buffer)s[0];
"""

_reg_decl_template = r"""
    %(type)s %(regname)s = %(initval)s;"""

_smem_decl_template = r"""
    __shared__ float %(sbuf)s[32];"""

_smem_init_template = r"""
        %(sbuf)s[threadIdx.x] = 0.0f;"""

_thread_index_template1 = r"""unsigned int idx0 = threadIdx.%(dim0)s + blockIdx.%(dim0)s * ITEMS_PER_BLOCK0_%(id)s;
    unsigned int loopmax = min(shape%(loop_axis)s, (blockIdx.x + 1) * ITEMS_PER_BLOCK0_%(id)s);
"""

_thread_index_template2 = r"""unsigned int idx0 = threadIdx.%(dim0)s + blockIdx.%(dim0)s * ITEMS_PER_BLOCK0_%(id)s;
    unsigned int idx1 = threadIdx.%(dim1)s + blockIdx.%(dim1)s * ITEMS_PER_BLOCK1_%(id)s;
    unsigned int loopmax = min(shape%(loop_axis)s, (blockIdx.x + 1) * ITEMS_PER_BLOCK0_%(id)s);
"""

_thread_index_template3 = r"""unsigned int idx0 = threadIdx.%(dim0)s + blockIdx.%(dim0)s * ITEMS_PER_BLOCK0_%(id)s;
    unsigned int idx1 = threadIdx.%(dim1)s + blockIdx.%(dim1)s * ITEMS_PER_BLOCK1_%(id)s;
    unsigned int idx2 = threadIdx.%(dim2)s + blockIdx.%(dim2)s * ITEMS_PER_BLOCK2_%(id)s;
    unsigned int loopmax = min(shape%(loop_axis)s, (blockIdx.x + 1) * ITEMS_PER_BLOCK0_%(id)s);
"""

_exit_condition_template = r"(idx%(axis)s >= shape%(axis_letter)s)"
_early_exit_template = r"""
    if(%(condition)s)
        return;
"""

_init_template = r"""%(smem_decl)s

    %(index_calc)s
    unsigned int index = 0;
    %(reg_decl)s
    if (threadIdx.x < 32)
    {%(smem_init)s
    }
"""

_init_template_noshare = r"""
    %(index_calc)s
    unsigned int index = 0;
    %(reg_decl)s
"""

_defines_template1 = r"""#define ITEMS_PER_BLOCK0_%(id)s %(blksize0)s
"""

_defines_template2 = r"""#define ITEMS_PER_BLOCK0_%(id)s %(blksize0)s
#define ITEMS_PER_BLOCK1_%(id)s %(blksize1)s
"""

_defines_template3 = r"""#define ITEMS_PER_BLOCK0_%(id)s %(blksize0)s
#define ITEMS_PER_BLOCK1_%(id)s %(blksize1)s
#define ITEMS_PER_BLOCK2_%(id)s %(blksize2)s
"""

_header_template = r"""

%(defines)s
__global__ void %(kernel_name)s(%(args)s)
{"""

_includes_template = r"""#include <float.h>
#include <cuda_fp16.h>
"""
# flex functions taken from neon flexsim neon/backends/cuda_templates.py
_flex_includes_template = r"""
__device__ __forceinline__ short fp32_to_int16(float val)
{
    short ret;
    asm("cvt.rni.s16.f32 %0, %1;" : "=h"(ret) : "f"(val));
    return ret;
}

__device__ __forceinline__ float max_abs(int max_abs, int val)
{
    asm("{\n\t"
        ".reg .s32 abs_val;\n\t"
        "abs.s32 abs_val, %1;\n\t"
        "max.s32 %0, %0, abs_val;\n\t"
        "}" : "+r"(max_abs) : "r"(val));
    return max_abs;
}
"""

# flex templates
_flex_maxabs_atomicmax = "atomicMax(flex_stats, flex_max);"

indent_str = "    "

MAX_AXES = 3
THREADS_PER_BLOCK = 1024


class TensorDescriptionWrapper:
    """
    Wraps a TensorDescription and handles broadcasting dimensions by altering
    shape and strides.
    """
    def __init__(self, tensor_description, max_dims=1, gemm=False):
        self.dtype = tensor_description.dtype
        self.strides = tensor_description.strides
        self.shape = tensor_description.shape
        self.td = tensor_description

        if len(self.strides) == 0:
            self.strides = (0, )

        if len(self.shape) == 0:
            self.shape = (1, )

        if len(self.shape) < max_dims:
            if gemm:
                self.shape = tuple(list(self.shape) + [1])
                self.strides = tuple(list(self.strides) + [1])
            else:
                self.shape = tuple([1] + list(self.shape))
                self.strides = tuple([0] + list(self.strides))

        self.strides = [s // self.dtype.itemsize for s in self.strides]
        self.strides = tuple(self.strides)

    @property
    def is_trans(self):
        return (len(self.shape) == 2 and self.strides[0] < self.strides[1])

    def is_flex(self):
        return hasattr(self.td.buffer.data.device_tensor(self.td), 'flex_entry')

    def flex_entry(self):
        return self.td.buffer.data.device_tensor(self.td).flex_entry


class GenerationContext:
    def __init__(self):
        self.register_mapping = None
        self.register_inits = None
        self.register_types = None
        self.buffers = None
        self.constants = None
        self.last_write = None
        self.has_argmaxmin = None
        self.shared_buffers = None


class FlexScaleDescription:
    def __init__(self, flex_entry, is_output):
        self.flex_entry = flex_entry
        self.is_output = is_output

def _are_flex_params(params):
    return any([isinstance(p, FlexScaleDescription) for p in params])

def _is_buffer(value):
    """
    When looking at an op in the buffer, there are several fields for inputs
    and outputs which can be either memory buffers, constants, or registers.
    This function returns true if the value is a memory buffer (tensor).

    Arguments:
        value: Object to check type of

    Returns: True if the input is a buffer in memory
    """
    if isinstance(value, TensorDescriptionWrapper) and value.td.buffer is not None:
        return True

    return False


def _compress_axes(ops):
    """
    Called to homogenize the axes of tensors used in a kernel. Also finds the
    reduction axis.
    TODO: If this logic is moved up into the graph, this function may not be
    necessary. Currently with all of the flattening spliced into the graph
    and limited fusion, this function is not expected to do much.

    Arguments:
        ops (list): List of tuples describing ops to compile into kernel

    Returns: New list of ops with tensors reshaped as needed.
    """
    reduction_axis = None
    num_axes = 0

    # Find reduction axis if reduction ops are part of this function
    for op in ops:
        if op[0] in _redop_templates:
            assert reduction_axis is None or reduction_axis == op[4]
            reduction_axis = op[4]

        for t in op[1:4]:
            if _is_buffer(t):
                num_axes = max(num_axes, len(t.shape))

    if num_axes <= 3:
        return ops

    # Combine non-reduction axes
    if reduction_axis == 0 or reduction_axis is None:
        new_axes = [[0], range(1, num_axes)]
    elif reduction_axis == (num_axes - 1):
        new_axes = [range(num_axes - 1), [num_axes - 1]]
    else:
        new_axes = [range(reduction_axis), [reduction_axis], range(reduction_axis + 1, num_axes)]

    # Reshape tensors
    new_ops = []
    for op in ops:
        new_op = list(op)

        for index in range(1, 4):
            if _is_buffer(op[index]):
                new_shape = [np.prod([t.shape[d] for d in compress]) for compress in new_axes]
                new_op[index] = op[index].reshape(tuple(new_shape))

        new_ops.append(tuple(new_op))

    return new_ops


def _optimize_loop_axis(dim):
    """
    Chooses kernel parameters including CUDA block size, grid size, and
    number of elements to compute per thread for the loop axis. The loop
    axis is the axis of the tensor for which a thread can compute multiple
    outputs. Uses a simple heuristic which tries to get at least 4 warps
    per block and 8 items per thread to hide latencies. Prefers a higher
    item-per-thread to launching many blocks for very large axes since
    blocks are serialized by the GPU after all SMs are filled.

    Arguments:
        dim (int): Size of the tensor on the loop axis.

    Returns: tuple of grid dimension, block dimension, and items per thread
    """
    sm_count = _get_sm_count()

    griddim = min(sm_count, -((-dim) // 32))
    items_per_block = -((-dim) // griddim)

    items_per_thread = 1
    warps = -((-items_per_block) // (32 * items_per_thread))

    while (warps > 4 and items_per_thread < 8) or (warps > 32):
        items_per_thread = items_per_thread + 1
        warps = -((-items_per_block) // (32 * items_per_thread))

    blockdim = warps * 32

    return (griddim, blockdim, items_per_thread)


def _get_axes_mapping(ops):
    """
    Maps the axes of tensors involved in the computation to CUDA block and grid
    dimensions. Also finds block and grid sizes. The strategy here is to choose
    one axis as the "loop axis" where the kernel will loop over multiple values.
    The loop axis is always the reduction axis if a reduction op is involved,
    since the entire axis must be computed in a single CUDA block. If no reduction
    is involved, we try to choose the most contiguous axis as the loop axis to
    improve memory load contiguity and cache hit rate. This mapping function
    supports a maximum of 3 axes so that no compounding of axes into CUDA
    dimensions is required.

    Arguments:
        ops (list): List of tuples describing ops to compute in the kernel

    Returns: Grid, block, item-per-thread for each axis and its CUDA dimension
        mapping along with number of dimensions used by the kernel.
    """
    max_shape = [1] * MAX_AXES
    axes = range(MAX_AXES)
    reduction_axis = None

    # Find maximum shape and check for reductions
    for op in ops:
        if op[0] in _redop_templates or op[0]:
            assert reduction_axis is None or reduction_axis == op[4]
            reduction_axis = op[4]

        for t in op[1:4]:
            if _is_buffer(t):
                shape = t.shape
                assert len(shape) <= MAX_AXES

                for axis in axes:
                    if axis < len(shape) and shape[axis] > max_shape[axis]:
                        max_shape[axis] = shape[axis]

    # Determine which axis/axes map to block
    axes_mapping = [(None, None, None, None, None, False)] * MAX_AXES
    dims = ['x', 'y', 'z']
    blocksize = 1
    if reduction_axis is not None:
        blockdim = -((-max_shape[reduction_axis]) // 256)
        blockdim = min(THREADS_PER_BLOCK, max(32, blockdim * 32))
        items_per_thread = -((-max_shape[reduction_axis]) // blockdim)
        axes_mapping[reduction_axis] = ('x', blockdim, 1, items_per_thread,
                                        max_shape[reduction_axis], False)

        blocksize = blockdim
        dims.remove('x')
    elif max_shape[0] == 1 and np.prod(max_shape) != 1:
        if max_shape[1] == 1:
            axis = 2
        else:
            axis = 1

        (griddim, blockdim, items_per_thread) = _optimize_loop_axis(max_shape[axis])
        blocksize = blockdim
        axes_mapping[axis] = (dims.pop(0), blockdim, griddim, items_per_thread, max_shape[axis],
                              False)

    # TODO: consider contiguity in axis mapping
    for axis in axes:
        if axes_mapping[axis][0] is not None:
            continue

        if len(dims) == MAX_AXES:
            (griddim, blockdim, items_per_thread) = _optimize_loop_axis(max_shape[axis])
            blocksize = blockdim
            exit_condition = False
        else:
            items_per_thread = 1
            blockdim = 1
            while ((blockdim * blocksize * 2) <= THREADS_PER_BLOCK and
                   (blockdim * 2) < max_shape[axis]):
                blockdim = blockdim * 2
            blocksize = blocksize * blockdim
            griddim = -((-max_shape[axis]) // (blockdim * items_per_thread))

            # Build exit condition if partial block exists
            if (blockdim * griddim) != max_shape[axis]:
                exit_condition = True
            else:
                exit_condition = False

        axes_mapping[axis] = (dims.pop(0), blockdim, griddim, items_per_thread, max_shape[axis],
                              exit_condition)

    # Prune unused axes
    dims = MAX_AXES
    while (axes_mapping[dims - 1][1] * axes_mapping[dims - 1][2] * axes_mapping[dims - 1][3]) == 1:
        dims = dims - 1

    return (axes_mapping, dims)


def _preprocess_ops(ops, loop_axis_len):
    """
    Breaks ops into stages, where stages are terminated by reduction ops,
    since reductions must be completed before their results can be used
    by a subsequent op. Also since we don't store elementwise results in
    registers between stages (due to limited register space), any elementwise
    results which are needed after a reduction must be re-computed and are
    therefore duplicated by this function.
    TODO: There is probably some work which can be done at the graph traversal
    stage to order ops optimally to have the fewest stages and fewest re-
    calculations.

    Arguments:
        ops (list): List of tuples describing ops to compute in the kernel

    Returns: New list of ops with stages and duplicated ops as needed.
    """
    updaters = {}
    dependencies = {}

    out_ops = [[]]
    last_evaluated_stage = {}

    def add_dep(dep_index):
        for dep in dependencies[dep_index]:
            if dep not in last_evaluated_stage or last_evaluated_stage[dep] != (len(out_ops) - 1):
                if ops[dep][0] not in _redop_templates:
                    add_dep(dep)

        if ops[dep_index][0] in _redop_templates and loop_axis_len == 1:
            # Replace no-op reduction with assign
            new_op = list(ops[dep_index])
            new_op[0] = "assign"
            out_ops[-1].append(tuple(new_op))
        else:
            out_ops[-1].append(ops[dep_index])

        last_evaluated_stage[dep_index] = len(out_ops) - 1

    # Find dependencies for each operation
    for op, index in zip(ops, range(len(ops))):
        dependencies[index] = []

        for inval in op[1:3]:
            if inval is not None and inval in updaters:
                dependencies[index].append(updaters[inval])

        updaters[op[3]] = index

    # Replicate any ops where dependencies cross boundary of a reduction
    for op, index in zip(ops, range(len(ops))):
        if op[0] in _op_templates:
            if out_ops[-1] and out_ops[-1][-1][0] in _redop_templates:
                # New stage
                out_ops.append([])

        # Check that op's dependencies are evaluated in this stage
        add_dep(index)

    return out_ops


def _get_register_type(dtype, memory=False):
    if isinstance(dtype, Flex):
        # short buffers will be converted to float registers by flex scale
        if memory:
            if dtype == flex16:
                return "short"
            else:
                raise NotImplementedError
        else:
            return "float"
        # FLEX TODO:
        # need a case to return "flex" string for _conversion_templates
        # or push this case to calling code
            #return dtype.dtype_name
    if dtype == np.float32:
        return "float"
    elif dtype == np.float16:
        if memory:
            return "half"
        else:
            return "float"
    elif dtype == np.int32:
        return "int"
    elif dtype == np.int16:
        return "short"
    elif dtype == np.int8:
        return "char"
    else:
        raise TypeError("Unsupported type")


def _wrap_tensor_descriptions(ops):
    max_dims = 1
    for op in ops:
        new_op = list(op)
        for index in range(1, 4):
            if isinstance(new_op[index], TensorDescription):
                max_dims = max(max_dims, len(new_op[index].shape))

    new_ops = []
    for op in ops:
        new_op = list(op)
        for index in range(1, 4):
            if isinstance(new_op[index], TensorDescription):
                new_op[index] = TensorDescriptionWrapper(new_op[index], max_dims)

        new_ops.append(tuple(new_op))

    return new_ops


def _build_register_mapping(stages):
    """
    Maps buffers, constants, and intermediate values to variables in the kernel
    which should be stored as registers. Also determines types and init values
    for each register.

    Arguments:
        stages (list): List of stages each containing descriptions of ops to
            execute in the kernel

    Returns: GenerationContext containing information about register mapping
    """
    # Build lists of registers for each input/output
    register_mapping = {None: "None"}
    reg_count = 0
    register_inits = {}
    register_types = {}
    buffers = {}
    last_write = {}
    constants = {}
    has_argmaxmin = False
    # flex_scale: register name --> (kernel arg name, flex entry, whether is output)
    flex_scale = {}
    flex_stats_ptr = None

    for stage, stage_index in zip(stages, range(len(stages))):
        for op, op_index in zip(stage, range(len(stage))):
            if op[0] == "argmin" or op[0] == "argmax":
                has_argmaxmin = True

            for inval in op[1:3]:
                if inval not in register_mapping:
                    if isinstance(inval, (np.float16, np.float32, np.float64)):
                        regname = "constant" + str(len(constants))
                        register_mapping[inval] = regname
                        constants[regname] = inval
                        register_types[regname] = _get_register_type(type(inval), False)
                    else:
                        regname = "reg" + str(reg_count)
                        sclname = "scale" + str(reg_count)
                        reg_count = reg_count + 1
                        register_mapping[inval] = regname
                        register_types[regname] = _get_register_type(inval.dtype, False)

                        # FLEX TODO: other ops without scale?
                        if (op[0] == "argmax" or op[0] == "argmin") and inval is op[2]:
                            register_inits[regname] = \
                                "FLT_MAX" if op[0] == "argmin" else "-FLT_MAX"
                        else:
                            register_inits[regname] = "0.0f"

                        if _is_buffer(inval):
                            buffername = "buf" + str(len(buffers))
                            buffers[inval] = buffername

                        from ngraph.transformers.gputransform import GPURegister
                        if isinstance(inval, GPURegister) and not (op[0] == "argmax" or op[0] == "argmin"):
                            print "according to Stewart, this should not happen in current graph without fusing"
                            import ipdb; ipdb.set_trace()

                        # flex
                        # for argmax and argmin, inval is GPURegister, not TensorDescriptionWrapper
                        if not (op[0] == "argmax" or op[0] == "argmin") and inval.is_flex():
                            flex_entry = inval.flex_entry()
                            flex_scale[regname] = (sclname, flex_entry, False)

            if op[3] not in register_mapping:
                regname = "reg" + str(reg_count)
                sclname = "scale" + str(reg_count)
                reg_count = reg_count + 1
                register_mapping[op[3]] = regname
                register_types[regname] = _get_register_type(op[3].dtype, False)

                if op[0] in _redop_templates:
                    register_inits[regname] = _redop_inits[op[0]]
                else:
                    register_inits[regname] = "0.0f"

                if _is_buffer(op[3]):
                    buffername = "buf" + str(len(buffers))
                    buffers[op[3]] = buffername

                # flex
                if op[3].is_flex():
                    flex_entry = op[3].flex_entry()
                    flex_scale[regname] = (sclname, flex_entry, True)
                    flex_stats_ptr = flex_entry.ptr

            if _is_buffer(op[3]):
                last_write[op[3]] = (stage_index, op_index)

    ctx = GenerationContext()
    ctx.register_mapping = register_mapping
    ctx.register_inits = register_inits
    ctx.register_types = register_types
    ctx.buffers = buffers
    ctx.constants = constants
    ctx.last_write = last_write
    ctx.has_argmaxmin = has_argmaxmin
    ctx.flex_scale = flex_scale
    ctx.flex_stats_ptr = flex_stats_ptr
    return ctx


def _generate_stage_code(broadcast_loads, loop_loads, loop_stores, op_statements,
                         loop_axis, warp_reductions, reduction_stores):
    """
    Generates CUDA C code for a single stage which can contain any number of
    elementwise operations followed by one or more reductions. This code takes
    the form of a for loop over elements optionally followed by a warp and/or
    block wide reduction using shared memory.

    Arguments:
        broadcast_loads (list): List of buffer loads which are broadcast along
            the loop axis and only need to be loaded once
        loop_loads (list): List of buffer loads which must be done on each
            iteration of the loop
        loop_stores (list): List of buffer stores which must be done on each
            iteration of the loop
        op_statements (list): List of operation evaluations which are done in
            the loop.
        loop_axis (int): Axis of the tensor which the thread loops over
        warp_reductions (list): List of reduction operations done within the
            warp or block
        reduction_stores (list): List of buffer stores which are the result of
            reduction operations and only need to be done for the first thread
            in the dimension.

    Returns: CUDA C code string for this stage
    """
    code = ""

    # Add broadcast loads
    for load in broadcast_loads:
        code = code + "\n" + indent_str + load

    # Add op statements
    if len(loop_loads) == 0 and len(loop_stores) == 0:
        # All tensors are reduced, no item loop needed
        for statement in op_statements:
            code = code + "\n" + indent_str + statement
    else:
        # Build item loop
        item_loop_code = _item_loop_template % {
            "loopidx": loop_axis
        }
        code = code + "\n" + indent_str + item_loop_code + "\n" + indent_str + "{"

        for load in loop_loads:
            code = code + "\n" + (indent_str * 2) + load

        for statement in op_statements:
            code = code + "\n" + (indent_str * 2) + statement

        for store in loop_stores:
            code = code + "\n" + (indent_str * 2) + store

        code = code + "\n" + indent_str + "}"

    # Add warp reductions
    for warp_red in warp_reductions:
        code = code + warp_red

    # Add reduction stores
    for store in reduction_stores:
        code = code + "\n" + indent_str + store

    return code


def _generate_kernel_code(ctx, code, _defines_template, _thread_index_template,
                          kernel_name, argstring, axes_mapping, loop_axis,
                          kernel_identifier):
    """
    Generates entire kernel code which can be passed to the CUDA C compiler.
    Takes care of function header, defines, and initialization code.

    Arguments:
        ctx (GenerationContext): Context containing kernel specific data
            structures for register mapping, etc
        code (string): Generated CUDA C stage code
        _defines_template (string): Template for #defines
        _thread_index_template (string): Template for computing thread indices
        kernel_name (string): Name for function
        argstring (string): String containing list of kernel arguments
        axes_mapping (list): Mapping between tensor axes and kernel block
            dimensions
        loop_axis (int): Axis which the thread loops over to compute multiple
            elements

    Returns: String containing entire kernel source code
    """
    defines = _defines_template % {
        "blksize0": axes_mapping[0][1] * axes_mapping[0][3],
        "blksize1": axes_mapping[1][1] * axes_mapping[1][3],
        "blksize2": axes_mapping[2][1] * axes_mapping[2][3],
        "id": kernel_identifier
    }

    header = _header_template % {
        "defines": defines,
        "kernel_name": kernel_name,
        "args": argstring
    }

    # Initialization code
    reg_decls = ""
    for reg in ctx.register_mapping.values():
        if reg != "None" and reg not in ctx.constants:
            reg_decls = reg_decls + _reg_decl_template % {
                "regname": reg,
                "initval": ctx.register_inits[reg],
                "type": ctx.register_types[reg]
            }

    if ctx.has_argmaxmin:
        reg_decls = reg_decls + "\n    float temp_val = 0.0f;"
        reg_decls = reg_decls + "\n    unsigned int temp_idx = 0;"

    if ctx.flex_stats_ptr is not None:
       reg_decls = reg_decls + "\n    int flex_max = 0;"
       reg_decls = reg_decls + "\n    short reg_out = 0;"

    smem_decls = ""
    smem_inits = ""
    for sbuf in ctx.shared_buffers:
        smem_decls = smem_decls + _smem_decl_template % {
            "sbuf": sbuf
        }
        smem_inits = smem_inits + _smem_init_template % {
            "sbuf": sbuf
        }

    loop_axis_letters = ['a', 'b', 'c']
    index_calc = _thread_index_template % {
        "dim0": axes_mapping[0][0],
        "dim1": axes_mapping[1][0],
        "dim2": axes_mapping[2][0],
        "loop_axis": loop_axis_letters[loop_axis],
        "id": kernel_identifier
    }

    # Build exit conditions of block size does not line up exactly with tensor dimensions
    exit_conditions = []
    for axis_mapping, axis_index in zip(axes_mapping, range(3)):
        if axis_mapping[5]:
            condition = _exit_condition_template % {
                "axis": axis_index,
                "axis_letter": loop_axis_letters[axis_index]
            }
            exit_conditions.append(condition)

    if len(exit_conditions) != 0:
        total_condition = " && ".join(exit_conditions)
        code = _early_exit_template % {
            "condition": total_condition
        } + code

    if ctx.shared_buffers:
        code = _init_template % {
            "smem_decl": smem_decls,
            "reg_decl": reg_decls,
            "smem_init": smem_inits,
            "index_calc": index_calc
        } + code
    else:
        code = _init_template_noshare % {
            "reg_decl": reg_decls,
            "index_calc": index_calc
        } + code

    code = header + code + "\n}"
    return code


def _generate_kernel_args(ctx, axes_mapping, dims):
    """
    Generates a list of parameters which need to be passed to the CUDA kernel
    at runtime along with strings to represent them in C

    Arguments:
        ctx (GenerationContext): Context containing kernel specific data
            structures for register mapping, etc
        axes_mapping (list): Mapping between tensor axes and kernel block
            dimensions
        dims (int): Number of dimensions used by the kernel

    Returns: List of parameters and arguments and descriptor string for
        pycuda kernel compiler
    """
    # List arguments to kernel
    args = ["unsigned int shapea"]
    arg_desc = "I"
    params = [axes_mapping[0][4]]
    if dims == 2:
        args.append("unsigned int shapeb")
        arg_desc = arg_desc + "I"
        params.append(axes_mapping[1][4])
    elif dims == 3:
        args.extend(["unsigned int shapeb", "unsigned int shapec"])
        arg_desc = arg_desc + "II"
        params.extend([axes_mapping[1][4], axes_mapping[2][4]])

    for constant in ctx.constants.keys():
        args.append("float " + constant)
        arg_desc = arg_desc + "f"
        params.append(ctx.constants[constant])

    for buf in ctx.buffers.keys():
        args.append(_get_register_type(buf.dtype, True) + "* " + ctx.buffers[buf])
        args.append("unsigned int stridea_" + ctx.buffers[buf])
        arg_desc = arg_desc + "PI"
        params.append(buf.td)
        params.append(buf.strides[0])

        if dims == 2:
            args.append("unsigned int strideb_" + ctx.buffers[buf])
            arg_desc = arg_desc + "I"
            params.append(buf.strides[1])
        elif dims == 3:
            args.append("unsigned int strideb_" + ctx.buffers[buf])
            args.append("unsigned int stridec_" + ctx.buffers[buf])
            arg_desc = arg_desc + "II"
            params.append(buf.strides[1])
            params.append(buf.strides[2])

    # flex scale arguments
    for argname, flex_entry, is_output in ctx.flex_scale.values():
        args.append("float " + argname)
        arg_desc = arg_desc + "f"
        # create description of flex scale parameters that will be bound later
        params.append(FlexScaleDescription(flex_entry, is_output))

    if ctx.flex_stats_ptr is not None:
        args.append("int * flex_stats")
        arg_desc = arg_desc + "P"
        params.append(ctx.flex_stats_ptr)

    return (args, arg_desc, params)

def _get_compound_kernel(ops, axes_mapping, dims, kernel_identifier=''):
    """
    Generates a kernel which compounds multiple elementwise and reduction
    operations.

    Arguments:
        ops (list): List of tuples describing each operation
        axes_mapping (list): Mapping between tensor axes and kernel block
            dimensions
        dims (int): Number of dimensions used by the kernel

    Returns: pycuda kernel function object and parameters to pass
    """
    # Find axis which thread will loop over
    loop_axis = 0
    for axis in range(len(axes_mapping)):
        if axes_mapping[axis][0] == 'x':
            loop_axis = axis
            loop_axis_len = axes_mapping[axis][4]

    # Choose templates based on number of axes
    if dims == 1:
        _defines_template = _defines_template1
        _index_template = _index_template1
        _thread_index_template = _thread_index_template1
    elif dims == 2:
        _defines_template = _defines_template2
        if loop_axis == 0:
            _index_template = _index_template20
        else:
            _index_template = _index_template21
        _thread_index_template = _thread_index_template2
    elif dims == 3:
        _defines_template = _defines_template3
        if loop_axis == 0:
            _index_template = _index_template30
        elif loop_axis == 1:
            _index_template = _index_template31
        else:
            _index_template = _index_template32
        _thread_index_template = _thread_index_template3
    else:
        assert False

    # Pre-process ops so that we don't need to store intermediate results in registers
    # Also remove any no-op reductions
    stages = _preprocess_ops(ops, loop_axis_len)

    # Build lists of registers, buffers, and constants
    ctx = _build_register_mapping(stages)

    buffers_in_reg = [set() for stage in stages]
    code = ""
    arg_desc = ""
    shared_buffers = []
    for stage, stage_index in zip(stages, range(len(stages))):
        # Collect all load, op, and store statements for this stage
        broadcast_loads = []
        reduction_stores = []
        loop_loads = []
        loop_stores = []
        op_statements = []
        warp_reductions = []
        for op, op_index in zip(stage, range(len(stage))):
            for inval in op[1:3]:
                if _is_buffer(inval) and inval not in buffers_in_reg[stage_index]:
                    load_code = _load_template % {
                        "index": "index",
                        "buffer": ctx.buffers[inval]
                    }

                    # Check if explicit type conversion is needed for load because ALU
                    # doesn't support data format
                    reg_name = ctx.register_mapping[inval]
                    if isinstance(inval.dtype, Flex):
                        type_key = (inval.dtype.dtype_name,  # FLEX TODO: see _conversion_template note
                                    ctx.register_types[reg_name])
                    else:
                        type_key = (_get_register_type(inval.dtype, True),
                                    ctx.register_types[reg_name])
                    if op[0] == 'argmax' or op[0] == 'argmin':  # FLEX TODO: others?
                        # there should not be a conversion performed, even though type_key is currently (flex, float)
                        # FLEX TODO: fix this more systematically
                        type_key = (float, float)
                    else:
                        scale = ctx.flex_scale[reg_name][0] if inval.is_flex() else None
                    if type_key in _conversion_templates:
                        load_code = _conversion_templates[type_key] % {
                            "out": reg_name,
                            "in": load_code,
                            "scale": scale
                        }
                    else:
                        load_code = _default_conversion % {
                            "out": reg_name,
                            "in": load_code
                        }

                    if inval.strides[loop_axis] == 0 or inval.shape[loop_axis] == 1:
                        index_code = _index_template % {
                            "index": "index",
                            "stridea": "stridea_" + ctx.buffers[inval],
                            "strideb": "strideb_" + ctx.buffers[inval],
                            "stridec": "stridec_" + ctx.buffers[inval],
                            "item": "idx" + str(loop_axis)
                        }
                        broadcast_loads.append(index_code)
                        broadcast_loads.append(load_code)
                    else:
                        index_code = _index_template % {
                            "index": "index",
                            "stridea": "stridea_" + ctx.buffers[inval],
                            "strideb": "strideb_" + ctx.buffers[inval],
                            "stridec": "stridec_" + ctx.buffers[inval],
                            "item": "item"
                        }
                        loop_loads.append(index_code)
                        loop_loads.append(load_code)

                    buffers_in_reg[stage_index].add(inval)

            if op[0] in _op_templates:
                if op[0] == "onehot" and op[4] != loop_axis:
                    index = "idx" + str(loop_axis)
                else:
                    index = "item"
                op_code = _op_templates[op[0]] % {
                    "x": ctx.register_mapping[op[1]],
                    "y": ctx.register_mapping[op[2]],
                    "out": ctx.register_mapping[op[3]],
                    "index": index
                }
            else:
                op_code = _redop_templates[op[0]] % {
                    "x": ctx.register_mapping[op[1]],
                    "y": ctx.register_mapping[op[2]],
                    "out": ctx.register_mapping[op[3]],
                    "index": "item"
                }
                redop_code = _redop32_templates[op[0]] % {
                    "out": ctx.register_mapping[op[3]],
                    "y": ctx.register_mapping[op[2]],
                    "indent": (2 * indent_str)
                }

                if axes_mapping[loop_axis][1] <= 32:
                    warp_red_code = _red32_template % {
                        "statement": redop_code
                    }
                else:
                    sbuf = "sbuffer" + str(len(shared_buffers))
                    shared_buffers.append(sbuf)
                    warp_red_code = _red_template % {
                        "statement": redop_code,
                        "out": ctx.register_mapping[op[3]],
                        "shared_buffer": sbuf
                    }

                warp_reductions.append(warp_red_code)

            op_statements.append(op_code)

            if _is_buffer(op[3]):
                buffers_in_reg[stage_index].add(op[3])
                if op[0] in _redop_templates:
                    for subsequent_stage in buffers_in_reg[stage_index + 1:]:
                        subsequent_stage.add(op[3])

                if ctx.last_write[op[3]] == (stage_index, op_index):
                    store_code = _store_template % {
                        "index": "index",
                        "buffer": ctx.buffers[op[3]]
                    }

                    reg_name = ctx.register_mapping[op[3]]
                    if isinstance(op[3].dtype, Flex):
                        type_key = (ctx.register_types[reg_name],
                                    op[3].dtype.dtype_name)  # FLEX TODO: see conversion_template note
                    else:
                        type_key = (ctx.register_types[reg_name],
                                    _get_register_type(op[3].dtype, True))

                    # Check if explicit type conversion is needed for store because ALU
                    # doesn't support data format
                    if op[3].is_flex():
                        # for flex, store conversion in reg_out, which is reused for
                        # max_abs besides loop or reduction store
                        flex_stores = []  # flex statements for both loop and reduction stores
                        flex_conversion = _conversion_templates[type_key] % {
                            "out": "reg_out",
                            "in": reg_name,
                            "scale": ctx.flex_scale[reg_name][0]
                        }
                        flex_stores.append(flex_conversion)
                        flex_stores.append("flex_max = max_abs(flex_max, reg_out);")
                        store_code = _default_conversion % {
                            "out": store_code,
                            "in": "reg_out"
                        }
                    elif type_key in _conversion_templates:
                        store_code = _conversion_templates[type_key] % {
                            "out": store_code,
                            "in": reg_name
                        }
                    else:
                        store_code = _default_conversion % {
                            "out": store_code,
                            "in": reg_name
                        }

                    # reduction or loop store
                    if (op[0] in _redop_templates or op[3].strides[loop_axis] == 0
                            or op[3].shape[loop_axis] == 1):
                        store_code = _redstore_template % {
                            "store": store_code,
                            "loopidx": loop_axis
                        }
                        index_code = _index_template % {
                            "index": "index",
                            "stridea": "stridea_" + ctx.buffers[op[3]],
                            "strideb": "strideb_" + ctx.buffers[op[3]],
                            "stridec": "stridec_" + ctx.buffers[op[3]],
                            "item": "idx" + str(loop_axis)
                        }
                        if op[3].is_flex():
                            reduction_stores.extend(flex_stores)
                        reduction_stores.append(index_code)
                        reduction_stores.append(store_code)
                    else:
                        index_code = _index_template % {
                            "index": "index",
                            "stridea": "stridea_" + ctx.buffers[op[3]],
                            "strideb": "strideb_" + ctx.buffers[op[3]],
                            "stridec": "stridec_" + ctx.buffers[op[3]],
                            "item": "item"
                        }
                        if op[3].is_flex():
                            loop_stores.extend(flex_stores)
                        loop_stores.append(index_code)
                        loop_stores.append(store_code)
                    # flex collect max_abs across threads
                    if op[3].is_flex():
                        reduction_stores.append(_flex_maxabs_atomicmax)

        # Build stage code from collected statements
        code = code + _generate_stage_code(broadcast_loads, loop_loads, loop_stores,
                                           op_statements, loop_axis, warp_reductions,
                                           reduction_stores)

    # Construct kernel name
    kernel_name = "float_ew" + kernel_identifier + "_"
    if len(ops) > 4:
        op_names = [op[0] for op in ops[:5]]
    else:
        op_names = [op[0] for op in ops]
    kernel_name = kernel_name + '_'.join(op_names)

    # Compute arguments, parameters, and descriptor string
    args, arg_desc, params = _generate_kernel_args(ctx, axes_mapping, dims)
    argstring = ', '.join(args)

    # Construct header and join with code
    ctx.shared_buffers = shared_buffers
    code = _generate_kernel_code(ctx, code, _defines_template, _thread_index_template,
                                 kernel_name, argstring, axes_mapping, loop_axis,
                                 kernel_identifier)

    return (code, kernel_name, arg_desc, params)


def _prepare_compound_kernel(ops):
    """
    Generate and return a kernel given a set of ops.

    ops (list): List of tuples describing ops to execute in kernel. Each tuple
        should be of the format (op_name, input0, input1, output, axis)
    """
    # Take care tensor dimensionality
    ops = _wrap_tensor_descriptions(ops)
    ops = _compress_axes(ops)

    # Generate kernel source code and block/grid mapping
    (axes_mapping, dims) = _get_axes_mapping(ops)
    code, kernel_name, arg_desc, params = _get_compound_kernel(ops, axes_mapping, dims)

    # Compile kernel
    if _are_flex_params(params):
        code = _includes_template + _flex_includes_template + code
    else:
        code = _includes_template + code
    module = SourceModule(code, options=[])
    kernel = module.get_function(kernel_name)
    kernel.name = kernel_name
    kernel.prepare(arg_desc)

    # Calculate block and grid dims
    blockdim = [1, 1, 1]
    griddim = [1, 1, 1]
    for axis in axes_mapping:
        if axis[0] == 'x':
            blockdim[0] = axis[1]
            griddim[0] = axis[2]
        elif axis[0] == 'y':
            blockdim[1] = axis[1]
            griddim[1] = axis[2]
        elif axis[0] == 'z':
            blockdim[2] = axis[1]
            griddim[2] = axis[2]

    params = [tuple(griddim), tuple(blockdim), None] + params
    return (kernel, params, 128)


def _call_compound_kernel(ops):
    """
    Generate and call a kernel given a set of ops.

    ops (list): List of tuples describing ops to execute in kernel. Each tuple
        should be of the format (op_name, input0, input1, output, axis)
    """
    kernel, params, shared_size = _prepare_compound_kernel(ops)
    kernel.prepared_async_call(*params, shared_size=shared_size)


class CudaSourceFile:
    def __init__(self, name):
        self.num_kernels = 0
        self.module = None
        self.functions = dict()
        self.arg_descs = dict()

        self.compiled = False

        # Open file and add header
        self.f = tempfile.NamedTemporaryFile(mode='w', suffix='.c', prefix=name, delete=False)
        self.filename = self.f.name
        self.f.write(_includes_template)
        self.f.flush()

        if flex_verbose: print "CudaSourceFile temporary file", self.filename
        self.flex_includes_written = False  # FLEX TODO hack - relying on _get_compound_kernel processing of params to know if we have a flex kernel

    def add_kernel(self, ops):
        assert not self.compiled

        # Take care tensor dimensionality
        ops = _wrap_tensor_descriptions(ops)
        ops = _compress_axes(ops)

        # Generate kernel source code and block/grid mapping
        (axes_mapping, dims) = _get_axes_mapping(ops)
        code, kernel_name, arg_desc, params = _get_compound_kernel(ops, axes_mapping, dims,
                                                                   str(self.num_kernels))

        # Calculate block and grid dims
        blockdim = [1, 1, 1]
        griddim = [1, 1, 1]
        for axis in axes_mapping:
            if axis[0] == 'x':
                blockdim[0] = axis[1]
                griddim[0] = axis[2]
            elif axis[0] == 'y':
                blockdim[1] = axis[1]
                griddim[1] = axis[2]
            elif axis[0] == 'z':
                blockdim[2] = axis[1]
                griddim[2] = axis[2]

        params = [tuple(griddim), tuple(blockdim), None] + params

        # check if flex includes are needed/have been written
        if _are_flex_params(params) and self.flex_includes_written is not True:
            self.f.write(_flex_includes_template)
            self.flex_includes_written = True

        # Add kernel code to source file
        self.f.write(code)

        # Save arg_desc in dict
        self.arg_descs[kernel_name] = arg_desc

        # Increment number of kernels
        self.num_kernels = self.num_kernels + 1

        # Return kernel name and params
        return (kernel_name, params)

    def compile(self):
        assert not self.compiled
        self.f.close()

        # Create source module and compile
        sourcefile = open(self.filename, 'r')
        code = sourcefile.read()
        self.module = SourceModule(code, options=[])
        sourcefile.close()

        self.compiled = True

    def get_kernel(self, name):
        assert self.compiled
        assert name in self.arg_descs

        # Check if kernel is prepared in functions and return if so
        if name not in self.functions:
            # Prepare function using arg_desk
            kernel = self.module.get_function(name)
            kernel.name = name
            kernel.prepare(self.arg_descs[name])
            self.functions[name] = kernel

        # Return kernel function
        return self.functions[name]
