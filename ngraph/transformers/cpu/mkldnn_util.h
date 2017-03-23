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

#include "mkldnn.h"

#define MKLDNN_NETLIST_MAX_SIZE 10

struct mkldnn_netlist {
    mkldnn_stream_t         stream;
    int                     net_size;
    mkldnn_primitive_t      net[MKLDNN_NETLIST_MAX_SIZE];
    int                     prim_desc_count;
    int                     prim_count;
    int                     buffer_count;
    mkldnn_primitive_desc_t prim_desc_list[3*MKLDNN_NETLIST_MAX_SIZE];
    mkldnn_primitive_t      prim_list[3*MKLDNN_NETLIST_MAX_SIZE];
    float*                  buffer_list[MKLDNN_NETLIST_MAX_SIZE];
};

typedef struct mkldnn_netlist *mkldnn_netlist_t;


#define MKL_CHECK(f) do { \
    mkldnn_status_t s = f; \
    if (s != mkldnn_success) { \
        printf("[%s:%d] error: %s returns %d\n", __FILE__, __LINE__, #f, s); \
        exit(2); \
    } \
} while(0)

#define MKL_CHECK_TRUE(expr) do { \
    int e_ = expr; \
    if (!e_) { \
        printf("[%s:%d] %s failed\n", __FILE__, __LINE__, #expr); \
        exit(2); \
    } \
} while(0)

#endif // MKLDNN_UTIL_H_
