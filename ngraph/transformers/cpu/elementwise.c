/*******************************************************************************
* Copyright 2016 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/
#include "mkldnn_engine.h"
#include "mkldnn_util.h"

/* Create list of mkldnn primitives to run elelment Wise add primitive  */
mkldnn_netlist_t create_mkldnn_add_primitives(mkldnn_engine_t engine, float *add_src1, float *add_src2, float *add_dst,\
                                              int src1_sizes, int src2_sizes, int dst_sizes, int Num_matrix_to_add){

    mkldnn_netlist_t mkldnn_net = create_mkldnn_netlist();
    mkldnn_primitive_t add;
    int mkl_src1_dims = 1;
    int mkl_src2_dims = 1;
    int mkl_dst_dims = 1;
    int mkl_src1_size[1];
    int mkl_src2_size[1];
    int mkl_dst_size[1];

    mkl_src1_size[0] = src1_sizes;
    mkl_src2_size[0] = src2_sizes;
    mkl_dst_size[0] = dst_sizes;

    double scale_vector[] = {1,1};

    // create memory primitive descriptor
     mkldnn_memory_desc_t prim_md1;
     mkldnn_primitive_desc_t user_pd1;
     MKL_CHECK(mkldnn_memory_desc_init(&prim_md1, mkl_src1_dims, mkl_src1_size, mkldnn_f32, mkldnn_x));
     MKL_CHECK(mkldnn_memory_primitive_desc_create(&user_pd1, &prim_md1, engine));

     mkldnn_memory_desc_t prim_md2;
     mkldnn_primitive_desc_t user_pd2;
     MKL_CHECK(mkldnn_memory_desc_init(&prim_md2, mkl_src2_dims, mkl_src2_size, mkldnn_f32, mkldnn_x));
     MKL_CHECK(mkldnn_memory_primitive_desc_create(&user_pd2, &prim_md2, engine));

     mkldnn_primitive_desc_t input_pds[] = {user_pd1, user_pd2};

    // create a memory primitive for input and output
    mkldnn_primitive_t mkldnn_memory_prim_src1, mkldnn_memory_prim_src2, mkldnn_memory_prim_dst;
    create_mkldnn_memory_primitive(mkl_src1_dims, mkl_src1_size, mkldnn_x, mkldnn_f32, engine, add_src1, &mkldnn_memory_prim_src1);
    create_mkldnn_memory_primitive(mkl_src2_dims, mkl_src2_size, mkldnn_x, mkldnn_f32, engine, add_src2, &mkldnn_memory_prim_src2);
    create_mkldnn_memory_primitive(mkl_dst_dims, mkl_dst_size, mkldnn_x, mkldnn_f32, engine, add_dst, &mkldnn_memory_prim_dst);

    // create a Sum primitive descriptor 
    mkldnn_primitive_desc_t add_primitive_desc;
    MKL_CHECK(mkldnn_sum_primitive_desc_create(&add_primitive_desc, NULL, Num_matrix_to_add, scale_vector, input_pds));

    // create sum primitive
    const_mkldnn_primitive_t add_prim_dsts[] = { mkldnn_memory_prim_dst };
    mkldnn_primitive_at_t add_prim_srcs[] = {
                                        mkldnn_primitive_at(mkldnn_memory_prim_src1, 0),
                                        mkldnn_primitive_at(mkldnn_memory_prim_src2, 0),
                                        };

    MKL_CHECK(mkldnn_primitive_create(&add, add_primitive_desc, add_prim_srcs, add_prim_dsts));

    // account MKLDNN resources for clean up
    mkldnn_net->prim_list[mkldnn_net->prim_count++] = add;
    mkldnn_net->prim_list[mkldnn_net->prim_count++] = mkldnn_memory_prim_src1;
    mkldnn_net->prim_list[mkldnn_net->prim_count++] = mkldnn_memory_prim_src2;
    mkldnn_net->prim_list[mkldnn_net->prim_count++] = mkldnn_memory_prim_dst;
    mkldnn_net->prim_desc_list[mkldnn_net->prim_desc_count++] = add_primitive_desc;
    mkldnn_net->prim_desc_list[mkldnn_net->prim_desc_count++] = user_pd1;
    mkldnn_net->prim_desc_list[mkldnn_net->prim_desc_count++] = user_pd2;
    mkldnn_net->net[mkldnn_net->net_size++] = add;

    return mkldnn_net;
}
