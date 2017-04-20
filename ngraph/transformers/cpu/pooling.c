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
/* Create list of mkldnn primitives to run pooling fprop */
mkldnn_netlist_t create_mkldnn_pool_fprop_primitives(
            mkldnn_engine_t engine,
            int src_dims, int dst_dims, int stride_dims, int pad_dims,
            int* pool_src_sizes, int* pool_kernel_sizes, int* pool_dst_sizes,
            float* pool_src, float* pool_out,
            int* pool_strides, int* pool_padding, int pool_type
        )
{
    mkldnn_netlist_t mkldnn_net = create_mkldnn_netlist();
    mkldnn_primitive_t pool;

    int mkl_src_dims = 4;
    int mkl_kernel_dims = 2;
    int mkl_dst_dims = 4;
    int mkl_src_sizes[4];
    int mkl_kernel_sizes[2];
    int mkl_dst_sizes[4];
    int mkl_strides[2];
    int mkl_padding[2];

    /* Flatten out the depth (D, M) dimension and reorder logical dimensions to match MKLDNN */
    set_mkl_dimensions("pooling", pool_src_sizes, pool_dst_sizes,
                        pool_kernel_sizes, pool_strides, pool_padding,
                        mkl_src_sizes, mkl_dst_sizes, mkl_kernel_sizes,
                        mkl_strides, mkl_padding);

    /* create data descriptors for pooling w/ no specified format */
    mkldnn_memory_desc_t mkldnn_memory_desc_src_md, mkldnn_memory_desc_dst_md;
    MKL_CHECK(mkldnn_memory_desc_init(&mkldnn_memory_desc_src_md, mkl_src_dims,
                                      mkl_src_sizes, mkldnn_f32, mkldnn_chwn));
    MKL_CHECK(mkldnn_memory_desc_init(&mkldnn_memory_desc_dst_md, mkl_dst_dims,
                                       mkl_dst_sizes, mkldnn_f32, mkldnn_chwn));

    /* create a pooling descriptor  - logical description of pooling */
    mkldnn_pooling_desc_t pool_any_desc;
    if(pool_type == 0) {
        MKL_CHECK(mkldnn_pooling_forward_desc_init (&pool_any_desc,
                  mkldnn_forward_training, mkldnn_pooling_max,
                  &mkldnn_memory_desc_src_md, &mkldnn_memory_desc_dst_md,
                  mkl_strides, mkl_kernel_sizes, mkl_padding, mkl_padding,
                  mkldnn_padding_zero));
    }
    else {
        MKL_CHECK(mkldnn_pooling_forward_desc_init (&pool_any_desc, 
                  mkldnn_forward_training, mkldnn_pooling_avg,
                  &mkldnn_memory_desc_src_md, &mkldnn_memory_desc_dst_md,
                  mkl_strides, mkl_kernel_sizes, mkl_padding, mkl_padding,
                  mkldnn_padding_zero));
    }

    /* create a pooling primitive descriptor - pooling descriptor bound to the CPU engine */
    mkldnn_primitive_desc_t pool_fwd_pd;
    MKL_CHECK(mkldnn_primitive_desc_create(&pool_fwd_pd, &pool_any_desc,
                                           engine, NULL));

    /* create memory primitives for input and output data in user format */
    mkldnn_primitive_t mkldnn_memory_prim_user_src, mkldnn_prim_argmax,
                       mkldnn_memory_prim_user_dst;
    create_mkldnn_memory_primitive(mkl_src_dims, mkl_src_sizes, mkldnn_chwn,
                                   mkldnn_f32, engine, pool_src,
                                   &mkldnn_memory_prim_user_src);
    create_mkldnn_memory_primitive(mkl_dst_dims, mkl_dst_sizes, mkldnn_chwn,
                                   mkldnn_f32, engine, pool_out,
                                   &mkldnn_memory_prim_user_dst);

    float *pool_argmax_buffer = (float*)calloc(product(pool_dst_sizes, dst_dims),
                                               sizeof(float));
    if(pool_type == 0) {
       const_mkldnn_primitive_desc_t argmax_pd = 
           mkldnn_primitive_desc_query_pd(pool_fwd_pd, mkldnn_query_workspace_pd, 0);
       mkldnn_primitive_create(&mkldnn_prim_argmax, argmax_pd, NULL, NULL);
       mkldnn_memory_set_data_handle(mkldnn_prim_argmax, pool_argmax_buffer);
    }

    const_mkldnn_primitive_t pool_dsts[2];
    if(pool_type == 0) {
        pool_dsts[0]  = mkldnn_memory_prim_user_dst;
        pool_dsts[1]  = mkldnn_prim_argmax;
    }
    else {
        pool_dsts[0] = mkldnn_memory_prim_user_dst;
    }
    mkldnn_primitive_at_t pool_srcs[] = 
        { mkldnn_primitive_at(mkldnn_memory_prim_user_src, 0)};

    /* create a pooling primitive */
    MKL_CHECK(mkldnn_primitive_create(&pool, pool_fwd_pd, pool_srcs, pool_dsts));
    mkldnn_net->fwd_desc = pool_fwd_pd;
    mkldnn_net->fprop_src_addr = pool_argmax_buffer;

    /* Remember MKLDNN resources for cleanup */
    mkldnn_net->prim_list[mkldnn_net->prim_count++] = pool;
    mkldnn_net->prim_list[mkldnn_net->prim_count++] = mkldnn_memory_prim_user_src;
    mkldnn_net->prim_list[mkldnn_net->prim_count++] = mkldnn_memory_prim_user_dst;
    if (pool_type == 0) {
        mkldnn_net->prim_list[mkldnn_net->prim_count++] = mkldnn_prim_argmax;
    }
    mkldnn_net->buffer_list[mkldnn_net->buffer_count++] = pool_argmax_buffer; 
    mkldnn_net->prim_desc_list[mkldnn_net->prim_desc_count++] = pool_fwd_pd;
    mkldnn_net->net[mkldnn_net->net_size++] = pool;

    return mkldnn_net;
}


/* Create list of mkldnn primitives to run pooling bprop */
mkldnn_netlist_t create_mkldnn_pool_bprop_primitives(
            mkldnn_engine_t engine,
            int src_dims, int dst_dims, int stride_dims, int pad_dims,
            int* pool_src_sizes, int* pool_kernel_sizes, int* pool_dst_sizes,
            float* pool_src, float* pool_out,
            int* pool_strides, int* pool_padding, int pool_type,
            mkldnn_netlist_t mkldnn_fprop_net
        )
{

    mkldnn_netlist_t mkldnn_net = create_mkldnn_netlist();
    mkldnn_primitive_t pool_back;

    int mkl_src_dims = 4;
    int mkl_kernel_dims = 2;
    int mkl_dst_dims = 4;
    int mkl_src_sizes[4];
    int mkl_kernel_sizes[2];
    int mkl_dst_sizes[4];
    int mkl_strides[2];
    int mkl_padding[2];

    /* Flatten out the depth (D, M) dimension and reorder logical dimensions to match MKLDNN */
   set_mkl_dimensions("pooling", pool_src_sizes, pool_dst_sizes,  pool_kernel_sizes,
                      pool_strides, pool_padding, mkl_src_sizes, mkl_dst_sizes,
                      mkl_kernel_sizes, mkl_strides, mkl_padding);


    /* create data descriptors for pooling w/ no specified format */
    mkldnn_memory_desc_t mkldnn_memory_desc_src_md, mkldnn_memory_desc_dst_md;
    MKL_CHECK(mkldnn_memory_desc_init(&mkldnn_memory_desc_src_md, mkl_src_dims,
        mkl_src_sizes, mkldnn_f32, mkldnn_chwn));
    MKL_CHECK(mkldnn_memory_desc_init(&mkldnn_memory_desc_dst_md, mkl_dst_dims,
        mkl_dst_sizes, mkldnn_f32, mkldnn_chwn));

    /* create a pooling descriptor  - logical description of pooling */
    mkldnn_pooling_desc_t pool_any_desc;
    if(pool_type == 0) {
       MKL_CHECK(mkldnn_pooling_backward_desc_init (&pool_any_desc,
                 mkldnn_pooling_max, &mkldnn_memory_desc_dst_md,
                 &mkldnn_memory_desc_src_md, mkl_strides, mkl_kernel_sizes,
                 mkl_padding, mkl_padding, mkldnn_padding_zero));
    }
    else {
       MKL_CHECK(mkldnn_pooling_backward_desc_init (&pool_any_desc,
                 mkldnn_pooling_avg, &mkldnn_memory_desc_dst_md,
                 &mkldnn_memory_desc_src_md, mkl_strides, mkl_kernel_sizes,
                 mkl_padding, mkl_padding, mkldnn_padding_zero));
    }
    /* create a pooling primitive descriptor - pooling descriptor bound to the CPU engine */
    mkldnn_primitive_desc_t pool_pd;

    MKL_CHECK(mkldnn_primitive_desc_create(&pool_pd, &pool_any_desc, engine,
                                           mkldnn_fprop_net->fwd_desc));


    /* create memory primitives for input and output data in user format */
    mkldnn_primitive_t mkldnn_memory_prim_user_src, mkldnn_memory_prim_user_dst,
                       mkldnn_prim_argmax;
    create_mkldnn_memory_primitive(mkl_src_dims, mkl_src_sizes, mkldnn_chwn,
                                   mkldnn_f32, engine, pool_src,
                                   &mkldnn_memory_prim_user_src);
    create_mkldnn_memory_primitive(mkl_dst_dims, mkl_dst_sizes, mkldnn_chwn,
                                   mkldnn_f32, engine, pool_out,
                                   &mkldnn_memory_prim_user_dst);

    if(pool_type == 0) {
       const_mkldnn_primitive_desc_t argmax_pd = 
           mkldnn_primitive_desc_query_pd(pool_pd, mkldnn_query_workspace_pd, 0);
       mkldnn_primitive_create(&mkldnn_prim_argmax, argmax_pd, NULL, NULL);
       mkldnn_memory_set_data_handle(mkldnn_prim_argmax, 
                                     mkldnn_fprop_net->fprop_src_addr);
   }

    mkldnn_primitive_at_t pool_srcs[2];
    if(pool_type == 0) {
       pool_srcs[0]  = mkldnn_primitive_at(mkldnn_memory_prim_user_src,0);
       pool_srcs[1]  = mkldnn_primitive_at(mkldnn_prim_argmax,0);
    }
    else {
       pool_srcs[0] = mkldnn_primitive_at(mkldnn_memory_prim_user_src,0);
    }

    /* create a pooling primitive */
    const_mkldnn_primitive_t pool_dsts[] = {mkldnn_memory_prim_user_dst};

    MKL_CHECK(mkldnn_primitive_create(&pool_back, pool_pd, pool_srcs, pool_dsts));

    mkldnn_net->prim_list[mkldnn_net->prim_count++] = pool_back;
    mkldnn_net->prim_list[mkldnn_net->prim_count++] = mkldnn_memory_prim_user_src;
    mkldnn_net->prim_list[mkldnn_net->prim_count++] = mkldnn_memory_prim_user_dst;
    if (pool_type == 0) {
        mkldnn_net->prim_list[mkldnn_net->prim_count++] = mkldnn_prim_argmax;
    }
    
    mkldnn_net->prim_desc_list[mkldnn_net->prim_desc_count++] = pool_pd;
    mkldnn_net->net[mkldnn_net->net_size++] = pool_back;

    return mkldnn_net;
}
