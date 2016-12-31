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

from builtins import zip


_init_rand_func = r"""
    unsigned lfsr0, lfsr1, lfsr2;
    unsigned idx = bid * THREADS + tid;
    rand_state += idx % RAND_POOL_SIZE;
    lfsr0 = *(rand_state + 0*RAND_POOL_SIZE);
    lfsr1 = *(rand_state + 1*RAND_POOL_SIZE);
    lfsr2 = *(rand_state + 2*RAND_POOL_SIZE);
"""


_init_rand_round_func = r"""
    int i_rand_scale = (127 - 32 - mantissa_bits) << 23;
    float rand_scale = *(float*)&i_rand_scale;
    unsigned rand_mask = 0xffffffff << (23 - mantissa_bits);
"""


_finish_rand_func = r"""
    *(rand_state + 0*RAND_POOL_SIZE) = lfsr0;
    *(rand_state + 1*RAND_POOL_SIZE) = lfsr1;
    *(rand_state + 2*RAND_POOL_SIZE) = lfsr2;
"""

_common_kepler = r"""
#define __ldg(x) (*(x))
"""

_common_urand_gen = r"""
__device__ unsigned urand_gen(unsigned& lfsr0, unsigned& lfsr1, unsigned& lfsr2)
{
    lfsr0 = ((lfsr0 & 0xfffffffe) << 12) ^ (((lfsr0 << 13) ^ lfsr0) >> 19);
    lfsr1 = ((lfsr1 & 0xfffffff8) <<  4) ^ (((lfsr1 << 2)  ^ lfsr1) >> 25);
    lfsr2 = ((lfsr2 & 0xfffffff0) << 11) ^ (((lfsr2 << 3)  ^ lfsr2) >> 11);
    return lfsr0 ^ lfsr1 ^ lfsr2;
}
"""


_common_frand = r"""
__device__ __forceinline__ float frand(unsigned& lfsr0, unsigned& lfsr1, unsigned& lfsr2)
{
    unsigned urand = urand_gen(lfsr0, lfsr1, lfsr2);
    float val;
    asm("cvt.rn.f32.u32 %0, %1;\n\t"
        "mul.f32 %0, %0, 0F2f800000;"
        : "=f"(val) : "r"(urand));
    return val;
}
"""


_common_round = {

    "random": {

        "f4": r"""
__device__ float fp32_to_fp32_rand(
    float val, unsigned& lfsr0, unsigned& lfsr1, unsigned& lfsr2, float rand_scale,
          unsigned rand_mask)
{
    unsigned urand = urand_gen(lfsr0, lfsr1, lfsr2);

    float ret;
    asm("{\n\t"
        ".reg .f32 exponent, frand, result;\n\t"
        "and.b32 exponent, %1, 0xff800000;\n\t"
        "mul.f32 exponent, exponent, %2;\n\t"
        "cvt.rz.f32.u32 frand, %3;\n\t"
        "fma.rz.f32 result, exponent, frand, %1;\n\t"
        "and.b32 %0, result, %4;\n\t"
        "}" : "=f"(ret) : "f"(val), "f"(rand_scale), "r"(urand), "r"(rand_mask));

    return ret;
}
""",
        "f2": r"""
__device__ unsigned short fp32_to_fp16_rand(
    float val, unsigned& lfsr0, unsigned& lfsr1, unsigned& lfsr2, float rand_scale,
          unsigned rand_mask)
{
    unsigned urand = urand_gen(lfsr0, lfsr1, lfsr2);

    unsigned short half;
    asm("{\n\t"
        ".reg .f16 result16;\n\t"
        ".reg .f32 exponent, frand, result32;\n\t"
        "and.b32 exponent, %1, 0xff800000;\n\t"
        "mul.f32 exponent, exponent, %2;\n\t"
        "cvt.rz.f32.u32 frand, %3;\n\t"
        "fma.rz.f32 result32, exponent, frand, %1;\n\t"
        "and.b32 result32, result32, %4;\n\t"
        "cvt.rz.f16.f32 result16, result32;\n\t"
        "mov.b16 %0, result16;\n\t"
        "}" : "=h"(half) : "f"(val), "f"(rand_scale), "r"(urand), "r"(rand_mask));

    return half;
}
""",
        "i4": r"""
__device__ __forceinline__ int fp32_to_int32_rand(
    float val, unsigned& lfsr0, unsigned& lfsr1, unsigned& lfsr2)
{
    unsigned urand = urand_gen(lfsr0, lfsr1, lfsr2);
    int ret;
    asm("{\n\t"
        ".reg .f32 frand, result32;\n\t"
        "cvt.rz.f32.u32 frand, %2;\n\t"
        "copysign.f32 frand, %1, frand;\n\t"
        "mul.rz.f32 frand, frand, 0F2f800000;\n\t"
        "add.rz.f32 result32, frand, %1;\n\t"
        "cvt.rzi.s32.f32 %0, result32;\n\t"
        "}" : "=r"(ret) : "f"(val), "r"(urand));
    return ret;
}
""",
        "i2": r"""
__device__ __forceinline__ short fp32_to_int16_rand(
    float val, unsigned& lfsr0, unsigned& lfsr1, unsigned& lfsr2)
{
    unsigned urand = urand_gen(lfsr0, lfsr1, lfsr2);
    short half;
    asm("{\n\t"
        ".reg .f32 frand, result32;\n\t"
        "cvt.rz.f32.u32 frand, %2;\n\t"
        "copysign.f32 frand, %1, frand;\n\t"
        "mul.rz.f32 frand, frand, 0F2f800000;\n\t"
        "add.rz.f32 result32, frand, %1;\n\t"
        "cvt.rzi.s16.f32 %0, result32;\n\t"
        "}" : "=h"(half) : "f"(val), "r"(urand));
    return half;
}
""",
        "i1": r"""
__device__ __forceinline__ char fp32_to_int8_rand(
    float val, unsigned& lfsr0, unsigned& lfsr1, unsigned& lfsr2)
{
    unsigned urand = urand_gen(lfsr0, lfsr1, lfsr2);
    int ret;
    asm("{\n\t"
        ".reg .f32 frand, result32;\n\t"
        "cvt.rz.f32.u32 frand, %2;\n\t"
        "copysign.f32 frand, %1, frand;\n\t"
        "mul.rz.f32 frand, frand, 0F2f800000;\n\t"
        "add.rz.f32 result32, frand, %1;\n\t"
        "cvt.rzi.s8.f32 %0, result32;\n\t"
        "}" : "=r"(ret) : "f"(val), "r"(urand));
    return ret;
}
""",
    },
    "nearest": {

        "f2": r"""
__device__ __forceinline__ unsigned short fp32_to_fp16(float val)
{
    unsigned short ret;
    asm("{\n\t"
        ".reg .f16 f16;\n\t"
        "cvt.rn.f16.f32 f16, %1;"
        "mov.b16 %0, f16;\n\t"
        "}" : "=h"(ret) : "f"(val));
    return ret;
}
""",
        "i4": r"""
__device__ __forceinline__ int fp32_to_int32(float val)
{
    int ret;
    asm("cvt.rni.s32.f32 %0, %1;" : "=r"(ret) : "f"(val));
    return ret;
}
""",
        "u4": r"""
__device__ __forceinline__ unsigned fp32_to_uint32(float val)
{
    unsigned ret;
    asm("cvt.rni.u32.f32 %0, %1;" : "=r"(ret) : "f"(val));
    return ret;
}
""",
        "i2": r"""
__device__ __forceinline__ short fp32_to_int16(float val)
{
    short ret;
    asm("cvt.rni.s16.f32 %0, %1;" : "=h"(ret) : "f"(val));
    return ret;
}
""",
        "u2": r"""
__device__ __forceinline__ unsigned short fp32_to_uint16(float val)
{
    unsigned short ret;
    asm("cvt.rni.u16.f32 %0, %1;" : "=h"(ret) : "f"(val));
    return ret;
}
""",
        "i1": r"""
__device__ __forceinline__ char fp32_to_int8(float val)
{
    int ret;
    asm("cvt.rni.s8.f32 %0, %1;" : "=r"(ret) : "f"(val));
    return ret;
}
""",
        "u1": r"""
__device__ __forceinline__ unsigned char fp32_to_uint8(float val)
{
    unsigned ret;
    asm("cvt.rni.u8.f32 %0, %1;" : "=r"(ret) : "f"(val));
    return ret;
}
""",
    },
}
# random rounding not yet used for these types
for dtype in ("u4", "u2", "u1"):
    _common_round["random"][dtype] = _common_round["nearest"][dtype]

for mode in ("random", "nearest"):
    for xtype, itype in zip(("x4", "x2", "x1"), ("i4", "i2", "i1")):
        _common_round[mode][xtype] = _common_round[mode][itype]


_common_fp16_to_fp32 = r"""
__device__ __forceinline__ float fp16_to_fp32(unsigned short val)
{
    float ret;
    asm("{\n\t"
        ".reg .f16 f16;\n\t"
        "mov.b16 f16, %1;\n\t"
        "cvt.f32.f16 %0, f16\n\t;"
        "}" : "=f"(ret) : "h"(val));
    return ret;
}
"""

_common_max_abs = r"""
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

_ew_types = {
    "f4": {
        "type": "float",
        "type4": "float4",
        "cvt": "",
        "cvt_out": "",
    },
    "f2": {
        "type": "unsigned short",
        "type4": "ushort4",
        "cvt": "fp16_to_fp32",
        "cvt_out": "fp32_to_fp16",
    },
    "i4": {
        "type": "int",
        "cvt": "(float)",
    },
    "u4": {
        "type": "unsigned int",
        "cvt": "(float)",
    },
    "i2": {
        "type": "short",
        "cvt": "(float)",
    },
    "u2": {
        "type": "unsigned short",
        "cvt": "(float)",
    },
    "i1": {
        "type": "char",
        "cvt": "(float)",
    },
    "u1": {
        "type": "unsigned char",
        "cvt": "(float)",
    },
    "x4": {
        "type": "int",
        "cvt": "scale{0} * (float)",
    },
    "x2": {
        "type": "short",
        "cvt": "scale{0} * (float)",
    },
    "x1": {
        "type": "char",
        "cvt": "scale{0} * (float)",
    },
}
