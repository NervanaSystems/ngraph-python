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
from future.utils import native_str
from pycuda.tools import context_dependent_memoize

from ngraph.flex.base import Flex
from ngraph.transformers.gpu.float_ew2 import _get_register_type
from ngraph.transformers.gpu.float_ew2 import NvrtcSourceModule as SourceModule

from ngraph.transformers.gpu.kernels.cuda.cuda_templates import _common_max_abs

"""
CUDA kernels for lookup table layers. Kernels are only given for bprop, since
fprop is just a take operation. There is a deterministic kernel and
non-deterministic kernel (using atomics) provided. Sorting kernels are also
provided to help with the deterministic version.
"""
flex_prefix = "i"


_flex_includes_template = r"""
__device__ __forceinline__ short fp32_to_int16(float val)
{
    short ret;
    asm("cvt.rni.s16.f32 %0, %1;" : "=h"(ret) : "f"(val));
    return ret;
}
"""


@context_dependent_memoize
def _get_lut_bprop_kernel(dtype, in_dtype, deterministic=False):
    """
    Builds the bprop kernel for lookup table layers based on templated code.
    If the deterministic version is requested, an index buffer must be passed
    as an argument. This index buffer re-orders items in the input tensor
    so that word_ids are sorted. This is required since we need to be sure that
    each thread only updates weights for one word id.

    Arguments:
        dtype (np.dtype): The data which the kernel will operate on.
        deterministic (boolean): Builds the deterministic kernel when this is
            set to True.
    """
    if not deterministic:
        code = r"""
__global__ void lut_bprop(
    %(in_dtype)s* inputs, %(type)s* dW, %(type)s* errors, const int nin,
    const int embedding_dim, const int vocab_size, const int pad_idx)
{
    const int tid  = threadIdx.x;
    const int bid  = blockIdx.x;

    int word_id = inputs[bid];
    int error_row = bid * embedding_dim;
    int output_row = word_id * embedding_dim;

    if(word_id != pad_idx)
    {
        for(int i = tid; i < embedding_dim; i += blockDim.x)
        {
            atomicAdd(&dW[output_row + i], errors[error_row + i]);
        }
    }
}
"""

        code = code % {
            "type": _get_register_type(dtype)
        }

        module = SourceModule(code, options=["--use_fast_math"])
        kernel = module.get_function("lut_bprop")
        kernel.prepare("PPPIIIi")
    else:
        code = r"""
%(common)s
        
__global__ void lut_bprop(
    %(in_dtype)s* inputs, int* index_buffer, %(type)s* dW, %(type)s* errors,
    const int nin, const int embedding_dim, const int vocab_size,
    const int pad_idx %(stats_args)s)
{
    const int tid  = threadIdx.x;
    const int bid  = blockIdx.x;

    int index_position = bid;
    int index = index_buffer[index_position];
    int word_id = inputs[index] %(scale_i_multiply)s;
    int intermediate_max = 0;

    if((bid == 0 || word_id != (inputs[index_buffer[bid - 1]] %(scale_i_multiply)s)) && word_id != pad_idx)
    {
        int output_row = word_id * embedding_dim;

        do {
            int error_row = index * embedding_dim;

            for(int i = tid; i < embedding_dim; i += blockDim.x)
            {
                %(compute_dW_code)s
            }
            index_position++;
            if(index_position == gridDim.x)
            {
                break;
            }
            index = index_buffer[index_position];
        } while((inputs[index] %(scale_i_multiply)s) == word_id);
    }
    %(atomic_max)s
}
"""
        template_vals = dict()
        _configure_template_vals(template_vals, in_dtype, dtype)
        code = code % template_vals

        module = SourceModule(code, options=["--use_fast_math"])
        kernel = module.get_function("lut_bprop")
        flex_sig = lambda prefix: "Pfff" if (prefix == flex_prefix) else ""
        kernel.prepare("PPPPIIIi" + flex_sig(in_dtype.str[1]))

    kernel.name = "lut_bprop"
    return kernel


def _get_sorting_kernel(kernel_id, block_size, in_dtype):
    """
    Builds kernels used for sorting inputs. There are several kernels here
    corresponding to the steps in the algorithm. The algorithm works by
    determining the sorted position for each input item. This is done with
    a bucket sort algorithm, where each word_id is a bucket. The first step
    determines the size of each bucket (number of occurences of each word_id).
    Next, a prefix some is computed over the list of bucket sizes to find
    where each bucket will be placed in the output buffer. Finally, each thread
    places it's index into the correct sorted position based on the bucket
    start index (computed from the prefix sum) and that thread's offset into
    the bucket (which is taken from the output of the atomic add done in the
    first step.)

    Arguments:
        kernel_id (Integer): Which step to build the kernel for [0, 4]
        block_size (Integer): Number of threads per block for the prefix sum
            kernels.
    """
    code = r"""
#define THREADS %(threads)s
#define STORE_BLOCKSUM %(store_blocksum)s
__global__ void sort_inputs0(
        %(in_dtype)s* inputs, int* index_buffer, int* offset_buffer, int* word_counts,
        const int vocab_size, const int input_length %(stats_args)s)
{
    const int tid = threadIdx.x + (blockDim.x * blockIdx.x);
    int word_id;

    if(tid < input_length)
    {
        word_id = inputs[tid] %(scale_i_multiply)s;
        offset_buffer[tid] = atomicAdd(&word_counts[word_id], 1);
    }
}

__device__ void scan(int* buffer, int* blocksum, int global_length)
{
    const int tid = (threadIdx.x << 1) + 1;
    const int gid = ((threadIdx.x + (blockIdx.x * blockDim.x)) << 1) + 1;

    __shared__ int local_counts[THREADS * 2];
    local_counts[tid] = buffer[gid];
    local_counts[tid - 1] = buffer[gid - 1];

    #pragma unroll
    for(int skip = 1; skip <= THREADS; skip <<= 1)
    {
        int mask = (skip << 1) - 1;
        if((tid & mask) == mask)
        {
            local_counts[tid] += local_counts[tid - skip];
        }

        __syncthreads();
    }

    if(tid == (THREADS * 2 - 1))
    {
#if STORE_BLOCKSUM
        blocksum[blockIdx.x] = local_counts[tid];
#endif
        local_counts[tid] = 0;
    }

    #pragma unroll
    for(int skip = THREADS; skip > 0; skip >>= 1)
    {
        int mask = (skip << 1) - 1;
        if((tid & mask) == mask)
        {
            int temp = local_counts[tid - skip];
            local_counts[tid - skip] = local_counts[tid];
            local_counts[tid] += temp;
        }

        __syncthreads();
    }

    if(gid < global_length)
    {
        buffer[gid] = local_counts[tid];
        buffer[gid - 1] = local_counts[tid - 1];
    }
}

__global__ void sort_inputs1(
        %(in_dtype)s* inputs, int* index_buffer, int* offset_buffer, int* word_counts,
        const int vocab_size, const int input_length %(stats_args)s)
{
    scan(word_counts, word_counts + vocab_size, vocab_size);
}

__global__ void sort_inputs2(
        %(in_dtype)s* inputs, int* index_buffer, int* offset_buffer, int* word_counts,
        const int vocab_size, const int input_length %(stats_args)s)
{
    scan(word_counts + vocab_size, 0, blockDim.x);
}

__global__ void sort_inputs3(
        %(in_dtype)s* inputs, int* index_buffer, int* offset_buffer, int* word_counts,
        const int vocab_size, const int input_length %(stats_args)s)
{
    const int gid = (threadIdx.x + (blockIdx.x * blockDim.x)) << 1;

    if(gid < vocab_size)
    {
        word_counts[gid] += word_counts[vocab_size + blockIdx.x];
        word_counts[gid + 1] += word_counts[vocab_size + blockIdx.x];
    }
}

__global__ void sort_inputs4(
        %(in_dtype)s* inputs, int* index_buffer, int* offset_buffer, int* word_counts,
        const int vocab_size, const int input_length %(stats_args)s)
{
    const int tid = threadIdx.x + (blockDim.x * blockIdx.x);
    int word_id;

    if(tid < input_length)
    {
        word_id = inputs[tid] %(scale_i_multiply)s;
        int sorted_position = word_counts[word_id] + offset_buffer[tid];
        index_buffer[sorted_position] = tid;
    }
}
"""
    template_vals = {
        "threads": block_size,
        "store_blocksum": (1 if kernel_id == 1 else 0),
        "in_dtype": _get_register_type(in_dtype, memory=True)
    }

    for key in ("stats_args", "scale_i_multiply"):
        template_vals[key] = ""

    if isinstance(in_dtype, Flex):
        template_vals["stats_args"] = ", float scaleI"
        template_vals["scale_i_multiply"] = "* scaleI"

    code = code % template_vals
    module = SourceModule(code, options=["--use_fast_math"])

    function_name = "sort_inputs" + native_str(kernel_id)
    kernel = module.get_function(function_name)
    flex_sig = lambda prefix: "f" if (prefix == flex_prefix) else ""
    kernel.prepare("PPPPII" + flex_sig(in_dtype.str[1]))

    kernel.name = "sort_inputs"
    return kernel


def _configure_template_vals(template_vals, in_dtype, dtype):
    template_vals["in_dtype"] = _get_register_type(in_dtype, memory=True)
    template_vals["type"] = _get_register_type(dtype, memory=True)
    template_vals["compute_dW_code"] = r"""dW[output_row + i] += errors[error_row + i];"""

    for key in ("stats_args", "atomic_max", "scale_i_multiply", "common"):
        template_vals[key] = ""
    if isinstance(dtype, Flex):
        template_vals["stats_args"] = ", int* maxabs, float scaleO, float scaleI, float scaleE"
        template_vals["atomic_max"] = r"""atomicMax(maxabs, intermediate_max);"""
        template_vals["scale_i_multiply"] = "* scaleI"
        template_vals["common"] = _common_max_abs + _flex_includes_template

        template_vals["compute_dW_code"] = r"""
                float dW_flex = dW[output_row + i] * scaleO;
                float error_flex = errors[error_row + i] * scaleE;
                
                dW_flex += error_flex;
                
                int dW_out = fp32_to_int16(dW_flex / scaleO);
                intermediate_max = max_abs(intermediate_max, dW_out);
                
                dW[output_row + i] = dW_out;
                """
