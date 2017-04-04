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

/* Create list of mkldnn primitives to run innerproduct fprop */
mkldnn_netlist_t create_mkldnn_innerproduct_fprop_primitives(
            mkldnn_engine_t engine,
            int src_dims, int weights_dims, int mkl_bias_dims, int dst_dims,
            int* src_sizes, int* weights_sizes, int* mkl_bias_sizes, int* dst_sizes,
            float* ip_src, float* ip_weights, float* ip_bias, float* ip_out
        )
{
  mkldnn_netlist_t mkldnn_net = create_mkldnn_netlist();
  mkldnn_primitive_t inner_product;

  int mkl_src_dims = 2;
  int mkl_weights_dims = 2;
  int mkl_dst_dims = 2;
  int mkl_src_sizes[2];
  int mkl_weights_sizes[2];
  int mkl_dst_sizes[2];

  mkl_src_sizes[0] = src_sizes[0];
  mkl_src_sizes[1] = src_sizes[1];
  mkl_weights_sizes[0] = weights_sizes[1];
  mkl_weights_sizes[1] = weights_sizes[0];
  mkl_dst_sizes[0] = dst_sizes[0];
  mkl_dst_sizes[1] = dst_sizes[1];

  /* create data descriptors for dot product */
  mkldnn_memory_desc_t mkldnn_memory_desc_src_md,
                       mkldnn_memory_desc_weights_md,
                       mkldnn_memory_desc_bias_md, mkldnn_memory_desc_dst_md;
  MKL_CHECK(mkldnn_memory_desc_init(&mkldnn_memory_desc_src_md, mkl_src_dims,
                                     mkl_src_sizes, mkldnn_f32, mkldnn_nc));
  MKL_CHECK(mkldnn_memory_desc_init(&mkldnn_memory_desc_weights_md,
                                    mkl_weights_dims, mkl_weights_sizes,
                                    mkldnn_f32, mkldnn_io));
  if (ip_bias) {
      // TODO - support bias
      MKL_CHECK(mkldnn_memory_desc_init(&mkldnn_memory_desc_bias_md,
                                        mkl_bias_dims, mkl_bias_sizes,
                                        mkldnn_f32, mkldnn_any));
  }
  MKL_CHECK(mkldnn_memory_desc_init(&mkldnn_memory_desc_dst_md,
                                    mkl_dst_dims, mkl_dst_sizes,
                                    mkldnn_f32, mkldnn_nc));

    /* create an inner product descriptor  - logical description of inner product */
  mkldnn_inner_product_desc_t ip_any_desc;
  if (ip_bias) {
      MKL_CHECK(mkldnn_inner_product_forward_desc_init(&ip_any_desc,
                mkldnn_forward_inference, &mkldnn_memory_desc_src_md,
                &mkldnn_memory_desc_weights_md, &mkldnn_memory_desc_bias_md,
                &mkldnn_memory_desc_dst_md));
  } else {
      MKL_CHECK(mkldnn_inner_product_forward_desc_init(&ip_any_desc,
                mkldnn_forward_inference, &mkldnn_memory_desc_src_md,
                &mkldnn_memory_desc_weights_md, NULL,
                &mkldnn_memory_desc_dst_md));
  }

  /* create an inner product primitive descriptor - inner product descriptor bound to the CPU engine */
  mkldnn_primitive_desc_t ip_pd;
  MKL_CHECK(mkldnn_primitive_desc_create(&ip_pd, &ip_any_desc, engine, NULL));

  /* create memory primitives for input and output data in user format */
  mkldnn_primitive_t mkldnn_memory_prim_user_src,
                     mkldnn_memory_prim_user_weights,
                     mkldnn_memory_prim_user_bias, mkldnn_memory_prim_user_dst;
  create_mkldnn_memory_primitive(mkl_src_dims, mkl_src_sizes, mkldnn_nc,
                                 mkldnn_f32, engine, ip_src,
                                 &mkldnn_memory_prim_user_src);
  create_mkldnn_memory_primitive(mkl_weights_dims, mkl_weights_sizes,
                                 mkldnn_io, mkldnn_f32, engine, ip_weights,
                                 &mkldnn_memory_prim_user_weights);
  if (ip_bias) {
      create_mkldnn_memory_primitive(1, mkl_bias_sizes, mkldnn_x,
                                     mkldnn_f32, engine, ip_bias,
                                     &mkldnn_memory_prim_user_bias);
  }
  create_mkldnn_memory_primitive(mkl_dst_dims, mkl_dst_sizes, mkldnn_nc,
                                 mkldnn_f32, engine, ip_out,
                                 &mkldnn_memory_prim_user_dst);

  const_mkldnn_primitive_t ip_dsts[] = { mkldnn_memory_prim_user_dst };

  /* create an inner product primitive */
  if (ip_bias) {
      mkldnn_primitive_at_t ip_srcs[] = {
          mkldnn_primitive_at(mkldnn_memory_prim_user_src, 0),
          mkldnn_primitive_at(mkldnn_memory_prim_user_weights, 0),
          mkldnn_primitive_at(mkldnn_memory_prim_user_bias, 0)
      };

      MKL_CHECK(mkldnn_primitive_create(&inner_product, ip_pd, ip_srcs,
                                        ip_dsts));
  } else {
      mkldnn_primitive_at_t ip_srcs[] = {
          mkldnn_primitive_at(mkldnn_memory_prim_user_src, 0),
          mkldnn_primitive_at(mkldnn_memory_prim_user_weights, 0)
      };

      MKL_CHECK(mkldnn_primitive_create(&inner_product, ip_pd, ip_srcs,
                                        ip_dsts));
  }

  /* Remember MKLDNN resources for cleanup */
  mkldnn_net->prim_list[mkldnn_net->prim_count++] = inner_product;
  mkldnn_net->prim_list[mkldnn_net->prim_count++] = mkldnn_memory_prim_user_src;
  mkldnn_net->prim_list[mkldnn_net->prim_count++] = 
         mkldnn_memory_prim_user_weights;
  mkldnn_net->prim_list[mkldnn_net->prim_count++] = mkldnn_memory_prim_user_dst;
  if (ip_bias)
      mkldnn_net->prim_list[mkldnn_net->prim_count++] = mkldnn_memory_prim_user_bias;
  mkldnn_net->prim_desc_list[mkldnn_net->prim_desc_count++] = ip_pd;

  mkldnn_net->net[mkldnn_net->net_size++] = inner_product;

  return mkldnn_net;
}
