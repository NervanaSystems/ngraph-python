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

/* Create list of mkldnn primitives to run relu fprop */
mkldnn_netlist_t create_mkldnn_relu_fprop_primitives(
            mkldnn_engine_t engine,
            float* relu_src, float* relu_out, double slope,
            int src_size
        )
{
    mkldnn_netlist_t mkldnn_net = create_mkldnn_netlist();
    mkldnn_primitive_t relu;

    int mkl_src_dims = 1;
    int mkl_dst_dims = 1;
    int mkl_src_sizes[1];
    int mkl_dst_sizes[1];

    mkl_src_sizes[0] = src_size;
    mkl_dst_sizes[0] = src_size;

    mkldnn_memory_desc_t mkldnn_memory_desc_src_md, mkldnn_memory_desc_dst_md;
    MKL_CHECK(mkldnn_memory_desc_init(&mkldnn_memory_desc_src_md, mkl_src_dims,
                                      mkl_src_sizes, mkldnn_f32, mkldnn_x));
    MKL_CHECK(mkldnn_memory_desc_init(&mkldnn_memory_desc_dst_md, mkl_dst_dims,
                                      mkl_dst_sizes, mkldnn_f32, mkldnn_x));  ///specify format of previous mkl layer

    mkldnn_relu_desc_t relu_desc;
    MKL_CHECK(mkldnn_relu_forward_desc_init(&relu_desc, mkldnn_forward_training,
              &mkldnn_memory_desc_src_md, slope));
    mkldnn_primitive_desc_t relu_pd;
    MKL_CHECK(mkldnn_primitive_desc_create(&relu_pd, &relu_desc, engine, NULL));

    mkldnn_primitive_t mkldnn_memory_prim_user_src, mkldnn_memory_prim_user_dst;
    create_mkldnn_memory_primitive(mkl_src_dims, mkl_src_sizes, mkldnn_x,
                                   mkldnn_f32, engine, relu_src,
                                   &mkldnn_memory_prim_user_src);
    create_mkldnn_memory_primitive(mkl_dst_dims, mkl_dst_sizes, mkldnn_x,
                                   mkldnn_f32, engine, relu_out,
                                   &mkldnn_memory_prim_user_dst);

    const_mkldnn_primitive_t relu_dsts[] = { mkldnn_memory_prim_user_dst };

    mkldnn_primitive_at_t relu_srcs[] =
        { mkldnn_primitive_at(mkldnn_memory_prim_user_src, 0) };

    MKL_CHECK(mkldnn_primitive_create(&relu, relu_pd, relu_srcs, relu_dsts));

     /* Remember MKLDNN resources for cleanup */
    mkldnn_net->prim_list[mkldnn_net->prim_count++] = relu;
    mkldnn_net->prim_list[mkldnn_net->prim_count++] = mkldnn_memory_prim_user_src;
    mkldnn_net->prim_list[mkldnn_net->prim_count++] = mkldnn_memory_prim_user_dst;

    mkldnn_net->prim_desc_list[mkldnn_net->prim_desc_count++] = relu_pd;
    mkldnn_net->net[mkldnn_net->net_size++] = relu;

    return mkldnn_net;
}
