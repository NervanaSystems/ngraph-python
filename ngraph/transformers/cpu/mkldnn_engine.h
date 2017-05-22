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

#ifndef MKLDNN_ENGINE_H
#define MKLDNN_ENGINE_H

#include "mkldnn.h"
#include "mkldnn_util.h"

size_t product(int *arr, size_t size);

void* alloc_memory(size_t size, mkldnn_data_type_t data_type);

void set_mkl_dimensions(char *primitive_name, int *primitive_src_sizes,
                        int *primitive_dst_sizes, int *primitive_weights_sizes,
                        int *primitive_strides, int *primitive_padding,
                        int *mkl_src_sizes, int *mkl_dst_sizes,
                        int *mkl_weights_sizes, int *mkl_strides,
                        int *mkl_padding);

void print_mkl_shape(int *mkl_src_sizes, int *mkl_dst_sizes,
                     int *mkl_weights_sizes, int mkl_src_dims,
                     int mkl_weights_dims, int mkl_dst_dims);

void create_mkldnn_memory_primitive(uint32_t n_dim, const int *dims,
                                    mkldnn_memory_format_t user_fmt,
                                    mkldnn_data_type_t data_type,
                                    mkldnn_engine_t engine, float *data,
                                    mkldnn_primitive_t *memory);

void create_mkldnn_reorder_primitive(
    mkldnn_primitive_t *user_memory,               /** in */
    const_mkldnn_primitive_desc_t *prim_memory_pd, /** in */
    int dir_is_user_to_prim,         /** in: user -> prim or prim -> user */
    mkldnn_primitive_t *prim_memory, /** out: memory primitive created */
    mkldnn_primitive_t *reorder      /** out: reorder primitive created */
    );

mkldnn_netlist_t create_mkldnn_netlist(void);

void destroy_mkldnn_netlist(mkldnn_netlist_t mkldnn_net);

void destroy_mkldnn_engine(mkldnn_engine_t engine);

void run_mkldnn_netlist(mkldnn_netlist_t mkldnn_net, int verbose);

void cleanup_mkldnn(mkldnn_netlist_t mkldnn_net);

#endif
