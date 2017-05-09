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

#include "mkldnn_engine.h"
#include "mkldnn_util.h"

/* Create list of mkldnn primitives to run batch norm fprop */
mkldnn_netlist_t create_mkldnn_batchnorm_fprop_primitives(
    mkldnn_engine_t engine, float *batch_norm_src, float *batch_norm_dst,
    float *weights, float *mean, float *variance, int *src_sizes,
    int *dst_sizes, int *weights_size, int mean_sizes, int variance_sizes,
    double epsilon) {

    mkldnn_netlist_t mkldnn_net = create_mkldnn_netlist();
    mkldnn_primitive_t batch_norm;

    // TODO - src_dims and dst_dims are incorrect should figure out the right format from the previous layer

    int mkl_src_dims = 4;
    int mkl_dst_dims = 4;
    int mkl_weight_dims = 2;
    int mkl_mean_dims = 1;
    int mkl_variance_dims = 1;
    int mkl_src_sizes[4];
    int mkl_dst_sizes[4];
    int mkl_weight_sizes[2];
    int mkl_mean_sizes[1];
    int mkl_variance_sizes[1];


    mkl_src_sizes[0] = src_sizes[1];
    mkl_src_sizes[1] = src_sizes[0];
    mkl_src_sizes[2] = src_sizes[2];
    mkl_src_sizes[3] = src_sizes[3];

    mkl_dst_sizes[0] = dst_sizes[1];
    mkl_dst_sizes[1] = dst_sizes[0];
    mkl_dst_sizes[2] = dst_sizes[2];
    mkl_dst_sizes[3] = dst_sizes[3];

    mkl_dst_sizes[0] = dst_sizes;
    mkl_weight_sizes[0] = weights_size[0];
    mkl_weight_sizes[1] = weights_size[1];
    mkl_mean_sizes[0] = mean_sizes;
    mkl_variance_sizes[0] = variance_sizes;

    mkldnn_memory_desc_t mkldnn_memory_desc_src_md, mkldnn_memory_desc_dst_md;

    MKL_CHECK(mkldnn_memory_desc_init(&mkldnn_memory_desc_src_md, mkl_src_dims,
                                      mkl_src_sizes, mkldnn_f32, mkldnn_chwn));

    /* create a batch norm descriptor - logical descriptor of the batch norm */
    mkldnn_batch_normalization_desc_t batch_norm_desc;
    MKL_CHECK(mkldnn_batch_normalization_forward_desc_init(&batch_norm_desc, mkldnn_forward_training,
                                                &mkldnn_memory_desc_src_md, epsilon, mkldnn_use_global_stats));

    /* create a batch norm primitive descriptor - convolution descriptor bound to the CPU engine */
    mkldnn_primitive_desc_t batch_norm_pd;
    MKL_CHECK(mkldnn_primitive_desc_create(&batch_norm_pd, &batch_norm_desc, engine, NULL));

    /* create memory primitives for input and output data in user format */
    mkldnn_primitive_t mkldnn_memory_prim_user_src, mkldnn_memory_prim_user_dst, \
    mkldnn_memory_prim_user_mean, mkldnn_memory_prim_user_variance, mkldnn_memory_prim_user_weight;

    create_mkldnn_memory_primitive(mkl_src_dims, mkl_src_sizes, mkldnn_chwn,
                                   mkldnn_f32, engine, batch_norm_src,
                                   &mkldnn_memory_prim_user_src);
    create_mkldnn_memory_primitive(mkl_dst_dims, mkl_dst_sizes, mkldnn_chwn,
                                   mkldnn_f32, engine, batch_norm_dst,
                                   &mkldnn_memory_prim_user_dst);
    create_mkldnn_memory_primitive(mkl_weight_dims, mkl_weight_sizes, mkldnn_nc,
                                   mkldnn_f32, engine, weights,
                                   &mkldnn_memory_prim_user_weight);
    create_mkldnn_memory_primitive(mkl_mean_dims, mkl_mean_sizes, mkldnn_x,
                                   mkldnn_f32, engine, mean,
                                   &mkldnn_memory_prim_user_mean);
    create_mkldnn_memory_primitive(mkl_variance_dims, mkl_variance_sizes, mkldnn_x,
                                   mkldnn_f32, engine, variance,
                                   &mkldnn_memory_prim_user_variance);

    /* create batch norm primitive */
    const_mkldnn_primitive_t batch_norm_prim_dsts[] = { mkldnn_memory_prim_user_dst};
    mkldnn_primitive_at_t batch_norm_prim_srcs[] = {
                                            mkldnn_primitive_at(mkldnn_memory_prim_user_src,0),
                                            mkldnn_primitive_at(mkldnn_memory_prim_user_mean,0),
                                            mkldnn_primitive_at(mkldnn_memory_prim_user_variance,0),
                                            mkldnn_primitive_at(mkldnn_memory_prim_user_weight,0)};

    MKL_CHECK(mkldnn_primitive_create(&batch_norm, batch_norm_pd, batch_norm_prim_srcs, batch_norm_prim_dsts));

    /* account resources for clean up */
    mkldnn_net->prim_list[mkldnn_net->prim_count++] = batch_norm;
    mkldnn_net->prim_list[mkldnn_net->prim_count++] = mkldnn_memory_prim_user_src;
    mkldnn_net->prim_list[mkldnn_net->prim_count++] = mkldnn_memory_prim_user_dst;
    mkldnn_net->prim_list[mkldnn_net->prim_count++] = mkldnn_memory_prim_user_mean;
    mkldnn_net->prim_list[mkldnn_net->prim_count++] = mkldnn_memory_prim_user_variance;
    mkldnn_net->prim_list[mkldnn_net->prim_count++] = mkldnn_memory_prim_user_weight;
    mkldnn_net->prim_desc_list[mkldnn_net->prim_desc_count++] = batch_norm_pd;
    mkldnn_net->net[mkldnn_net->net_size++] = batch_norm;

    return mkldnn_net;

}
