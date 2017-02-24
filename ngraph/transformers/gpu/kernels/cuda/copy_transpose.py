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

from functools import reduce
from future.utils import native_str
from operator import mul

from ngraph.transformers.gpu.util import _ceil_div, _magic64
from ngraph.transformers.gpu.float_ew2 import _get_register_type
from ngraph.flex.base import Flex

from pycuda.compiler import SourceModule
from pycuda.tools import context_dependent_memoize


_div64 = r"""
__device__ __forceinline__ int div64(int value, int magic, int shift)
{
    // if the divisor is a power of 2 the magic will be 1 and it's just a simple right shift
    // Otherwise multiply by magic and right shift just the high bits
    int result;
    asm("{\n\t"
        ".reg .pred p;\n\t"
        ".reg .u64 res64;\n\t"
        ".reg .u32 lo32, hi32;\n\t"
        "setp.ne.s32 p, %2, 1;\n\t"
        "mul.wide.u32 res64, %1, %2;\n\t"
        "mov.b64 {lo32, hi32}, res64;\n\t"
        "selp.u32 hi32, hi32, %1, p;\n\t"
        "shr.u32 %0, hi32, %3;\n\t"
        "}" : "=r"(result) : "r"(value), "r"(magic), "r"(shift));
    return result;
}
"""


def _get_oned_copy_kernel(dtype, shape):
    copy = r"""
__global__ void copy_oned(%(type)s* out, const %(type)s* in, int dim, long long src_str,
                          long long dst_str)
{
    int tid_x = threadIdx.x;
    int idx = blockIdx.x;

    idx = (idx << 5) + tid_x;

    const %(type)s* in0 = in + (src_str * idx);
    %(type)s* out0 = out + (dst_str * idx);

    if(idx < dim) *out0 = *in0;
}
"""
    code = copy % dict(
        type=_get_register_type(dtype, memory=True)
    )

    # print code
    module = SourceModule(code)
    kernel = module.get_function("copy_oned")
    kernel.prepare("PPIqq")

    kernel.grid = (_ceil_div(shape[0], 32), 1, 1)
    kernel.block = (32, 1, 1)
    kernel.args = (shape[0], )

    return kernel


@context_dependent_memoize
def _get_copy_transpose_kernel(dtype, shape, axes=None):
    if len(shape) == 1:
        return _get_oned_copy_kernel(dtype, shape)

    src = list(range(len(shape)))
    dst = list(axes)

    src_dim = src[-1]
    dst_dim = dst[-1]

    # If the inner dim is the same for both, no need for shared memory tile
    # Then map the outer source dim to the threadIdx.y values
    if src_dim == dst_dim:
        dst_dim = src[0]
        shared_tile = False
    else:
        shared_tile = True

    src_offset = []
    dst_offset = []
    params = []
    values = []
    magic = ""

    # add dims for bounds checking
    for dim in (src_dim, dst_dim):
        params.append("int dim_%s" % dim)
        values.append(shape[dim])

    # collapse src and dst shape by 32
    grid_shape = list(shape)
    grid_shape[src_dim] = _ceil_div(shape[src_dim], 32)
    grid_shape[dst_dim] = _ceil_div(shape[dst_dim], 32)

    # get a src list without dst dim
    src2 = [s for s in src if s != dst_dim]

    # get the name of the first compound index
    blkx_name = compound_idx = "".join(native_str(x) for x in src2)

    # generate the magic number math to extract all indeces
    while len(src2) > 1:

        idx1 = src2[0]
        del src2[0]
        idx2 = "".join(native_str(i) for i in src2)
        div = reduce(mul, (grid_shape[i] for i in src2), 1)

        params.extend(p % idx2 for p in ("int magic_%s", "int shift_%s", "int div_%s"))
        values.extend(_magic64(div))
        values.append(div)

        magic += r"""
    int idx_{1} = div64(idx_{0}, magic_{2}, shift_{2});
    int idx_{2} = idx_{0} - idx_{1}*div_{2};
""".format(compound_idx, idx1, idx2)

        compound_idx = idx2

    # Add params for src strides and generate src offset
    # The param values will be added externally
    for s in src:
        params.append("long long src_str_%d" % s)
        src_offset.append("src_str_%d*idx_%d" % (s, s))

    # Add params for dst strides and generate dst offset
    for d in dst:
        params.append("long long dst_str_%d" % d)
        dst_offset.append("dst_str_%d*idx_%d" % (d, d))

    num_strides = len(src) + len(dst)

    if shared_tile:
        copy_transpose = r"""
%(common)s

__global__ void copy_transpose(%(type)s* out, const %(type)s* in, %(params)s)
{
    __shared__ %(type)s tile[32][33];

    int tid_x = threadIdx.x;
    int tid_y = threadIdx.y;
    int idx_%(blk)s = blockIdx.x;
    int idx_%(dst)s = blockIdx.y;

    %(magic)s

    idx_%(src)s = (idx_%(src)s << 5) + tid_x;
    idx_%(dst)s = (idx_%(dst)s << 5) + tid_y;

    const %(type)s* in00 = in   + %(src_offset)s;
    const %(type)s* in08 = in00 + src_str_%(dst)s*8;
    const %(type)s* in16 = in08 + src_str_%(dst)s*8;
    const %(type)s* in24 = in16 + src_str_%(dst)s*8;

    bool b%(src)s = idx_%(src)s < dim_%(src)s;

    if (idx_%(dst)s +  0 < dim_%(dst)s && b%(src)s) tile[tid_y +  0][tid_x] = *in00;
    if (idx_%(dst)s +  8 < dim_%(dst)s && b%(src)s) tile[tid_y +  8][tid_x] = *in08;
    if (idx_%(dst)s + 16 < dim_%(dst)s && b%(src)s) tile[tid_y + 16][tid_x] = *in16;
    if (idx_%(dst)s + 24 < dim_%(dst)s && b%(src)s) tile[tid_y + 24][tid_x] = *in24;

    __syncthreads();

    %(type)s val00 = tile[tid_x][tid_y +  0];
    %(type)s val08 = tile[tid_x][tid_y +  8];
    %(type)s val16 = tile[tid_x][tid_y + 16];
    %(type)s val24 = tile[tid_x][tid_y + 24];

    idx_%(src)s += tid_y - tid_x;
    idx_%(dst)s += tid_x - tid_y;

    bool b%(dst)s = idx_%(dst)s < dim_%(dst)s;

    %(type)s* out00 = out   + %(dst_offset)s;
    %(type)s* out08 = out00 + dst_str_%(src)s*8;
    %(type)s* out16 = out08 + dst_str_%(src)s*8;
    %(type)s* out24 = out16 + dst_str_%(src)s*8;

    if (idx_%(src)s +  0 < dim_%(src)s && b%(dst)s) *out00 = val00;
    if (idx_%(src)s +  8 < dim_%(src)s && b%(dst)s) *out08 = val08;
    if (idx_%(src)s + 16 < dim_%(src)s && b%(dst)s) *out16 = val16;
    if (idx_%(src)s + 24 < dim_%(src)s && b%(dst)s) *out24 = val24;
}
"""
    else:
        copy_transpose = r"""
%(common)s

__global__ void copy_transpose(%(type)s* out, const %(type)s* in, %(params)s)
{
    int tid_x = threadIdx.x;
    int tid_y = threadIdx.y;
    int idx_%(blk)s = blockIdx.x;
    int idx_%(dst)s = blockIdx.y;

    %(magic)s

    idx_%(src)s = (idx_%(src)s << 5) + tid_x;
    idx_%(dst)s = (idx_%(dst)s << 5) + tid_y;

    bool b%(src)s    = idx_%(src)s      < dim_%(src)s;
    bool b%(dst)s_00 = idx_%(dst)s +  0 < dim_%(dst)s && b%(src)s;
    bool b%(dst)s_08 = idx_%(dst)s +  8 < dim_%(dst)s && b%(src)s;
    bool b%(dst)s_16 = idx_%(dst)s + 16 < dim_%(dst)s && b%(src)s;
    bool b%(dst)s_24 = idx_%(dst)s + 24 < dim_%(dst)s && b%(src)s;

    %(type)s val00 = 0;
    %(type)s val08 = 0;
    %(type)s val16 = 0;
    %(type)s val24 = 0;

    const %(type)s* in00 = in   + %(src_offset)s;
    const %(type)s* in08 = in00 + src_str_%(dst)s*8;
    const %(type)s* in16 = in08 + src_str_%(dst)s*8;
    const %(type)s* in24 = in16 + src_str_%(dst)s*8;

    if (b%(dst)s_00) val00 = *in00;
    if (b%(dst)s_08) val08 = *in08;
    if (b%(dst)s_16) val16 = *in16;
    if (b%(dst)s_24) val24 = *in24;

    %(type)s* out00 = out   + %(dst_offset)s;
    %(type)s* out08 = out00 + dst_str_%(dst)s*8;
    %(type)s* out16 = out08 + dst_str_%(dst)s*8;
    %(type)s* out24 = out16 + dst_str_%(dst)s*8;

    if (b%(dst)s_00) *out00 = val00;
    if (b%(dst)s_08) *out08 = val08;
    if (b%(dst)s_16) *out16 = val16;
    if (b%(dst)s_24) *out24 = val24;
}
"""
    code = copy_transpose % dict(
        common=_div64,
        type=_get_register_type(dtype, memory=True),
        params=", ".join(params),
        blk=blkx_name,
        src=src_dim,
        dst=dst_dim,
        magic=magic,
        src_offset=" + ".join(src_offset),
        dst_offset=" + ".join(dst_offset)
    )
    # print code
    module = SourceModule(code)
    kernel = module.get_function("copy_transpose")
    kernel.prepare("PP" + ("I" * (len(params) - num_strides)) + "q" * num_strides)

    grid_x = grid_shape[src_dim]
    grid_y = grid_shape[dst_dim]
    for s in src:
        if s not in (src_dim, dst_dim):
            grid_x *= grid_shape[s]

    kernel.grid = (grid_x, grid_y, 1)
    kernel.block = (32, 8, 1)
    kernel.args = tuple(values)

    return kernel
