/*******************************************************************************
* Copyright 2016 Nervana Systems Inc.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*      http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#ifndef MKLDNN_UTIL_H_
#define MKLDNN_UTIL_H_

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>

#include "mkldnn.h"

#define MKLDNN_NETLIST_MAX_SIZE 10
#define MKLDNN_MAX_ARGS 8

typedef struct {
    int ndims;
    int sizes[TENSOR_MAX_DIMS];     // TENSOR_MAX_DIMS defined in mkldnn.h
    mkldnn_memory_desc_t     md;   // Memory Descriptor - Non Opaque
    mkldnn_primitive_desc_t  desc; // Primitive Descriptor - Non Opaque
    mkldnn_primitive_t       prim; // Bound Primitive - Opaque
    void* buffer;
} mkldnn_tensor;

struct mkldnn_opkernel {
    int id;   
    int num_inputs;
    int num_outputs;

    mkldnn_tensor inputs[MKLDNN_MAX_ARGS];
    mkldnn_tensor outputs[MKLDNN_MAX_ARGS];
    mkldnn_tensor internal_inputs[MKLDNN_MAX_ARGS];  
    mkldnn_tensor internal_outputs[MKLDNN_MAX_ARGS];  
    
    mkldnn_primitive_desc_t op_desc;
    mkldnn_primitive_t      op_prim;
    mkldnn_primitive_t      reorder_i[MKLDNN_MAX_ARGS];
    mkldnn_primitive_t      reorder_o[MKLDNN_MAX_ARGS];

    int net_size;
    mkldnn_stream_t stream;
    mkldnn_primitive_t net[MKLDNN_MAX_ARGS];
};

typedef struct mkldnn_opkernel* mkldnn_opkernel_t;

struct mkldnn_netlist {
  mkldnn_stream_t stream;
  int net_size;
  mkldnn_primitive_t net[MKLDNN_NETLIST_MAX_SIZE];
  int prim_desc_count;
  int prim_count;
  int buffer_count;
  int prim_layouts_count;
  mkldnn_primitive_desc_t prim_desc_list[3 * MKLDNN_NETLIST_MAX_SIZE];
  mkldnn_primitive_t prim_list[3 * MKLDNN_NETLIST_MAX_SIZE];
  mkldnn_primitive_desc_t prim_layouts[3*MKLDNN_NETLIST_MAX_SIZE];
  float* buffer_list[MKLDNN_NETLIST_MAX_SIZE];
  mkldnn_primitive_desc_t fwd_desc;
  float * fprop_src_addr;
};

typedef struct mkldnn_netlist* mkldnn_netlist_t;

#define MKL_CHECK(f)                                                       \
  do {                                                                     \
    mkldnn_status_t s = f;                                                 \
    if (s != mkldnn_success) {                                             \
      printf("[%s:%d] error: %s returns %d\n", __FILE__, __LINE__, #f, s); \
      exit(2);                                                             \
    }                                                                      \
  } while (0)

#define MKL_CHECK_TRUE(expr)                                    \
  do {                                                          \
    int e_ = expr;                                              \
    if (!e_) {                                                  \
      printf("[%s:%d] %s failed\n", __FILE__, __LINE__, #expr); \
      exit(2);                                                  \
    }                                                           \
  } while (0)

#endif  // MKLDNN_UTIL_H_
