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

/* Create list of mkldnn primitives to run convolution fprop */
mkldnn_netlist_t create_mkldnn_conv_fprop_primitives(
    mkldnn_engine_t engine, int src_dims, int weights_dims, int bias_dims,
    int dst_dims, int stride_dims, int pad_dims, int* conv_src_sizes,
    int* conv_weights_sizes, int* conv_bias_sizes, int* conv_dst_sizes,
    float* conv_src, float* conv_weights, float* conv_bias, float* conv_out,
    int* conv_strides, int* conv_padding) {
  mkldnn_netlist_t mkldnn_net = create_mkldnn_netlist();
  mkldnn_primitive_t conv;

  int mkl_src_dims = 4;
  int mkl_weights_dims = 4;
  int mkl_dst_dims = 4;
  int mkl_src_sizes[4];
  int mkl_weights_sizes[4];
  int mkl_dst_sizes[4];
  int mkl_strides[2];
  int mkl_padding[2];

  /* Flatten out the depth (D, M) dimension and reorder logical dimensions to
   * match MKLDNN */
  set_mkl_dimensions("convolution", conv_src_sizes, conv_dst_sizes,
                     conv_weights_sizes, conv_strides, conv_padding,
                     mkl_src_sizes, mkl_dst_sizes, mkl_weights_sizes,
                     mkl_strides, mkl_padding);

  mkldnn_memory_desc_t mkldnn_memory_desc_src_md, mkldnn_memory_desc_weights_md,
      mkldnn_memory_desc_bias_md, mkldnn_memory_desc_dst_md;
  MKL_CHECK(mkldnn_memory_desc_init(&mkldnn_memory_desc_src_md, mkl_src_dims,
                                    mkl_src_sizes, mkldnn_f32, mkldnn_any));
  MKL_CHECK(mkldnn_memory_desc_init(&mkldnn_memory_desc_weights_md,
                                    mkl_weights_dims, mkl_weights_sizes,
                                    mkldnn_f32, mkldnn_any));
  MKL_CHECK(mkldnn_memory_desc_init(&mkldnn_memory_desc_dst_md, mkl_dst_dims,
                                    mkl_dst_sizes, mkldnn_f32, mkldnn_any));
  if (conv_bias) {
    MKL_CHECK(mkldnn_memory_desc_init(&mkldnn_memory_desc_bias_md, 1,
                                      conv_bias_sizes, mkldnn_f32, mkldnn_x));
  }

  /* create a convolution descriptor  - logical description of convolution */
  mkldnn_convolution_desc_t conv_desc;
  if (conv_bias) {
    MKL_CHECK(mkldnn_convolution_forward_desc_init(
        &conv_desc, mkldnn_forward, mkldnn_convolution_direct,
        &mkldnn_memory_desc_src_md, &mkldnn_memory_desc_weights_md,
        &mkldnn_memory_desc_bias_md, &mkldnn_memory_desc_dst_md, mkl_strides,
        mkl_padding, mkl_padding, mkldnn_padding_zero));
  } else {
    MKL_CHECK(mkldnn_convolution_forward_desc_init(
        &conv_desc, mkldnn_forward, mkldnn_convolution_direct,
        &mkldnn_memory_desc_src_md, &mkldnn_memory_desc_weights_md, NULL,
        &mkldnn_memory_desc_dst_md, mkl_strides, mkl_padding, mkl_padding,
        mkldnn_padding_zero));
  }

  /* create a convolution primitive descriptor - convolution descriptor bound to
   * the CPU engine */
  mkldnn_primitive_desc_t conv_pd;
  MKL_CHECK(mkldnn_primitive_desc_create(&conv_pd, &conv_desc, engine, NULL));

  /* create memory primitives for input and output data in user format */
  mkldnn_primitive_t mkldnn_memory_prim_user_src,
      mkldnn_memory_prim_user_weights, mkldnn_memory_prim_user_bias,
      mkldnn_memory_prim_user_dst;
  create_mkldnn_memory_primitive(mkl_src_dims, mkl_src_sizes, mkldnn_chwn,
                                 mkldnn_f32, engine, conv_src,
                                 &mkldnn_memory_prim_user_src);
  create_mkldnn_memory_primitive(mkl_weights_dims, mkl_weights_sizes,
                                 mkldnn_ihwo, mkldnn_f32, engine, conv_weights,
                                 &mkldnn_memory_prim_user_weights);
  create_mkldnn_memory_primitive(mkl_dst_dims, mkl_dst_sizes, mkldnn_chwn,
                                 mkldnn_f32, engine, conv_out,
                                 &mkldnn_memory_prim_user_dst);
  if (conv_bias) {
    create_mkldnn_memory_primitive(1, conv_bias_sizes, mkldnn_x, mkldnn_f32,
                                   engine, conv_bias,
                                   &mkldnn_memory_prim_user_bias);
  }

  /* create memory and reorder primitives for internal conversions */
  mkldnn_primitive_t mkldnn_memory_prim_internal_src,
      mkldnn_memory_prim_internal_weights, mkldnn_memory_prim_internal_dst;
  mkldnn_primitive_t mkldnn_reorder_prim_src, mkldnn_reorder_prim_weights,
      mkldnn_reorder_prim_dst;
  float *conv_src_buffer, *conv_weights_buffer, *conv_dst_buffer;

  const_mkldnn_primitive_desc_t src_pd =
      mkldnn_primitive_desc_query_pd(conv_pd, mkldnn_query_src_pd, 0);
  create_mkldnn_reorder_primitive(&mkldnn_memory_prim_user_src, &src_pd, 1,
                                  &mkldnn_memory_prim_internal_src,
                                  &mkldnn_reorder_prim_src);
  const_mkldnn_primitive_desc_t weights_pd =
      mkldnn_primitive_desc_query_pd(conv_pd, mkldnn_query_weights_pd, 0);
  create_mkldnn_reorder_primitive(&mkldnn_memory_prim_user_weights, &weights_pd,
                                  1, &mkldnn_memory_prim_internal_weights,
                                  &mkldnn_reorder_prim_weights);
  const_mkldnn_primitive_desc_t dst_pd =
      mkldnn_primitive_desc_query_pd(conv_pd, mkldnn_query_dst_pd, 0);
  create_mkldnn_reorder_primitive(&mkldnn_memory_prim_user_dst, &dst_pd, 0,
                                  &mkldnn_memory_prim_internal_dst,
                                  &mkldnn_reorder_prim_dst);

  /* Allocate memory for internal format conversions */
  if (mkldnn_memory_prim_internal_src) {
    conv_src_buffer =
        (float*)calloc(product(conv_src_sizes, src_dims), sizeof(float));
    MKL_CHECK(mkldnn_memory_set_data_handle(mkldnn_memory_prim_internal_src,
                                            conv_src_buffer));
  }
  if (mkldnn_memory_prim_internal_weights) {
    conv_weights_buffer = (float*)calloc(
        product(conv_weights_sizes, weights_dims), sizeof(float));
    MKL_CHECK(mkldnn_memory_set_data_handle(mkldnn_memory_prim_internal_weights,
                                            conv_weights_buffer));
  }
  if (mkldnn_memory_prim_internal_dst) {
    conv_dst_buffer =
        (float*)calloc(product(conv_dst_sizes, dst_dims), sizeof(float));
    MKL_CHECK(mkldnn_memory_set_data_handle(mkldnn_memory_prim_internal_dst,
                                            conv_dst_buffer));
  }

  /* select input and output primitives for convolution */
  mkldnn_primitive_t mkldnn_memory_prim_src =
      mkldnn_memory_prim_internal_src ? mkldnn_memory_prim_internal_src
                                      : mkldnn_memory_prim_user_src;
  mkldnn_primitive_t mkldnn_memory_prim_weights =
      mkldnn_memory_prim_internal_weights ? mkldnn_memory_prim_internal_weights
                                          : mkldnn_memory_prim_user_weights;
  mkldnn_primitive_t mkldnn_memory_prim_dst =
      mkldnn_memory_prim_internal_dst ? mkldnn_memory_prim_internal_dst
                                      : mkldnn_memory_prim_user_dst;

  const_mkldnn_primitive_t conv_dsts[] = {mkldnn_memory_prim_dst};

  /* create a convolution primitive */
  if (conv_bias) {
    mkldnn_primitive_at_t conv_srcs[] = {
        mkldnn_primitive_at(mkldnn_memory_prim_src, 0),
        mkldnn_primitive_at(mkldnn_memory_prim_weights, 0),
        mkldnn_primitive_at(mkldnn_memory_prim_user_bias, 0)};

    MKL_CHECK(mkldnn_primitive_create(&conv, conv_pd, conv_srcs, conv_dsts));
  } else {
    mkldnn_primitive_at_t conv_srcs[] = {
        mkldnn_primitive_at(mkldnn_memory_prim_src, 0),
        mkldnn_primitive_at(mkldnn_memory_prim_weights, 0)};

    MKL_CHECK(mkldnn_primitive_create(&conv, conv_pd, conv_srcs, conv_dsts));
  }

  /* Remember MKLDNN resources for cleanup */
  mkldnn_net->prim_list[mkldnn_net->prim_count++] = conv;
  mkldnn_net->prim_list[mkldnn_net->prim_count++] = mkldnn_memory_prim_user_src;
  mkldnn_net->prim_list[mkldnn_net->prim_count++] =
      mkldnn_memory_prim_user_weights;
  mkldnn_net->prim_list[mkldnn_net->prim_count++] = mkldnn_memory_prim_user_dst;
  if (conv_bias) {
    mkldnn_net->prim_list[mkldnn_net->prim_count++] =
        mkldnn_memory_prim_user_bias;
  }
  if (mkldnn_memory_prim_internal_src) {
    mkldnn_net->prim_list[mkldnn_net->prim_count++] =
        mkldnn_memory_prim_internal_src;
    mkldnn_net->prim_list[mkldnn_net->prim_count++] = mkldnn_reorder_prim_src;
    mkldnn_net->buffer_list[mkldnn_net->buffer_count++] = conv_src_buffer;
  }
  if (mkldnn_memory_prim_internal_weights) {
    mkldnn_net->prim_list[mkldnn_net->prim_count++] =
        mkldnn_memory_prim_internal_weights;
    mkldnn_net->prim_list[mkldnn_net->prim_count++] =
        mkldnn_reorder_prim_weights;
    mkldnn_net->buffer_list[mkldnn_net->buffer_count++] = conv_weights_buffer;
  }
  if (mkldnn_memory_prim_internal_dst) {
    mkldnn_net->prim_list[mkldnn_net->prim_count++] =
        mkldnn_memory_prim_internal_dst;
    mkldnn_net->prim_list[mkldnn_net->prim_count++] = mkldnn_reorder_prim_dst;
    mkldnn_net->buffer_list[mkldnn_net->buffer_count++] = conv_dst_buffer;
  }
  mkldnn_net->prim_desc_list[mkldnn_net->prim_desc_count++] = conv_pd;

  /* Create MKLDNN netlist */
  if (mkldnn_reorder_prim_src)
    mkldnn_net->net[mkldnn_net->net_size++] = mkldnn_reorder_prim_src;
  if (mkldnn_reorder_prim_weights)
    mkldnn_net->net[mkldnn_net->net_size++] = mkldnn_reorder_prim_weights;
  mkldnn_net->net[mkldnn_net->net_size++] = conv;
  if (mkldnn_reorder_prim_dst)
    mkldnn_net->net[mkldnn_net->net_size++] = mkldnn_reorder_prim_dst;

  return mkldnn_net;
}

/* Create list of mkldnn primitives to run convolution bprop */
// Variable name convention : _bw --> back prop wrt to weights & _bd --> back
// prop wrt to data
mkldnn_netlist_t create_mkldnn_conv_bprop_primitives(
    mkldnn_engine_t engine, int src_dims, int weights_dims, int bias_dims,
    int dst_dims, int stride_dims, int pad_dims, int* conv_src_sizes,
    int* conv_weights_sizes, int* conv_bias_sizes, int* conv_dst_sizes,
    float* conv_src, float* conv_weights, float* conv_bias, float* conv_out,
    int* conv_strides, int* conv_padding) {
  mkldnn_netlist_t mkldnn_net = create_mkldnn_netlist();
  mkldnn_primitive_t conv_bwd_data;

  int mkl_src_dims = 4;
  int mkl_weights_dims = 4;
  int mkl_dst_dims = 4;
  int mkl_src_sizes[4];
  int mkl_weights_sizes[4];
  int mkl_dst_sizes[4];
  int mkl_strides[2];
  int mkl_padding[2];

  /* Flatten out the depth (D, M) dimension and reorder logical dimensions to
   * match MKLDNN */
  set_mkl_dimensions("convolution", conv_src_sizes, conv_dst_sizes,
                     conv_weights_sizes, conv_strides, conv_padding,
                     mkl_src_sizes, mkl_dst_sizes, mkl_weights_sizes,
                     mkl_strides, mkl_padding);

  mkldnn_memory_desc_t mkldnn_memory_desc_src_md, mkldnn_memory_desc_weights_md,
      mkldnn_memory_desc_bias_md, mkldnn_memory_desc_dst_md;
  MKL_CHECK(mkldnn_memory_desc_init(&mkldnn_memory_desc_src_md, mkl_src_dims,
                                    mkl_src_sizes, mkldnn_f32, mkldnn_any));
  MKL_CHECK(mkldnn_memory_desc_init(&mkldnn_memory_desc_weights_md,
                                    mkl_weights_dims, mkl_weights_sizes,
                                    mkldnn_f32, mkldnn_any));
  if (conv_bias) {
    MKL_CHECK(mkldnn_memory_desc_init(&mkldnn_memory_desc_bias_md, 1,
                                      conv_bias_sizes, mkldnn_f32, mkldnn_x));
  }
  MKL_CHECK(mkldnn_memory_desc_init(&mkldnn_memory_desc_dst_md, mkl_dst_dims,
                                    mkl_dst_sizes, mkldnn_f32, mkldnn_any));

  /* create a convolution descriptor  - logical description of convolution. i/p
   * --> diff_dst(src) & o/p --> diff_src */
  mkldnn_convolution_desc_t conv_desc_data;
  MKL_CHECK(mkldnn_convolution_backward_data_desc_init(
      &conv_desc_data, mkldnn_convolution_direct, &mkldnn_memory_desc_dst_md,
      &mkldnn_memory_desc_weights_md, &mkldnn_memory_desc_src_md, mkl_strides,
      mkl_padding, mkl_padding, mkldnn_padding_zero));

  /* create a convolution primitive descriptor - convolution descriptor bound to
   * the CPU engine */
  mkldnn_primitive_desc_t conv_pd_data;
  MKL_CHECK(mkldnn_primitive_desc_create(&conv_pd_data, &conv_desc_data, engine,
                                         NULL));

  /* create memory primitives for input and output data in user format */
  mkldnn_primitive_t mkldnn_memory_prim_user_src,
      mkldnn_memory_prim_user_weights, mkldnn_memory_prim_user_bias,
      mkldnn_memory_prim_user_dst;
  create_mkldnn_memory_primitive(mkl_src_dims, mkl_src_sizes, mkldnn_chwn,
                                 mkldnn_f32, engine, conv_src,
                                 &mkldnn_memory_prim_user_src);
  create_mkldnn_memory_primitive(mkl_weights_dims, mkl_weights_sizes,
                                 mkldnn_ihwo, mkldnn_f32, engine, conv_weights,
                                 &mkldnn_memory_prim_user_weights);
  if (conv_bias) {
    create_mkldnn_memory_primitive(1, conv_bias_sizes, mkldnn_x, mkldnn_f32,
                                   engine, conv_bias,
                                   &mkldnn_memory_prim_user_bias);
  }
  create_mkldnn_memory_primitive(mkl_dst_dims, mkl_dst_sizes, mkldnn_chwn,
                                 mkldnn_f32, engine, conv_out,
                                 &mkldnn_memory_prim_user_dst);

  /* CONV BPROP DATA */
  /* create memory and reorder primitives for internal conversions w.r.t DATA */
  mkldnn_primitive_t mkldnn_memory_prim_internal_bd_src,
      mkldnn_memory_prim_internal_bd_weights,
      mkldnn_memory_prim_internal_bd_dst;
  mkldnn_primitive_t mkldnn_reorder_prim_bd_src, mkldnn_reorder_prim_bd_weights,
      mkldnn_reorder_prim_bd_dst;
  float *conv_bd_src_buffer, *conv_bd_weights_buffer, *conv_bd_dst_buffer;

  const_mkldnn_primitive_desc_t src_pd_data =
      mkldnn_primitive_desc_query_pd(conv_pd_data, mkldnn_query_diff_dst_pd, 0);
  create_mkldnn_reorder_primitive(&mkldnn_memory_prim_user_src, &src_pd_data, 1,
                                  &mkldnn_memory_prim_internal_bd_src,
                                  &mkldnn_reorder_prim_bd_src);
  const_mkldnn_primitive_desc_t weights_pd_data =
      mkldnn_primitive_desc_query_pd(conv_pd_data, mkldnn_query_weights_pd, 0);
  create_mkldnn_reorder_primitive(
      &mkldnn_memory_prim_user_weights, &weights_pd_data, 1,
      &mkldnn_memory_prim_internal_bd_weights, &mkldnn_reorder_prim_bd_weights);
  const_mkldnn_primitive_desc_t dst_pd_data =
      mkldnn_primitive_desc_query_pd(conv_pd_data, mkldnn_query_diff_src_pd, 0);
  create_mkldnn_reorder_primitive(&mkldnn_memory_prim_user_dst, &dst_pd_data, 0,
                                  &mkldnn_memory_prim_internal_bd_dst,
                                  &mkldnn_reorder_prim_bd_dst);

  /* Allocate memory for internal format conversions */
  if (mkldnn_memory_prim_internal_bd_src) {
    conv_bd_src_buffer =
        (float*)calloc(product(conv_src_sizes, src_dims), sizeof(float));
    MKL_CHECK(mkldnn_memory_set_data_handle(mkldnn_memory_prim_internal_bd_src,
                                            conv_bd_src_buffer));
  }
  if (mkldnn_memory_prim_internal_bd_weights) {
    conv_bd_weights_buffer = (float*)calloc(
        product(conv_weights_sizes, weights_dims), sizeof(float));
    MKL_CHECK(mkldnn_memory_set_data_handle(
        mkldnn_memory_prim_internal_bd_weights, conv_bd_weights_buffer));
  }
  if (mkldnn_memory_prim_internal_bd_dst) {
    conv_bd_dst_buffer =
        (float*)calloc(product(conv_dst_sizes, dst_dims), sizeof(float));
    MKL_CHECK(mkldnn_memory_set_data_handle(mkldnn_memory_prim_internal_bd_dst,
                                            conv_bd_dst_buffer));
  }

  /* select input and output primitives for convolution w.r.t DATA*/
  mkldnn_primitive_t mkldnn_memory_prim_bd_src =
      mkldnn_memory_prim_internal_bd_src ? mkldnn_memory_prim_internal_bd_src
                                         : mkldnn_memory_prim_user_src;
  mkldnn_primitive_t mkldnn_memory_prim_bd_weights =
      mkldnn_memory_prim_internal_bd_weights
          ? mkldnn_memory_prim_internal_bd_weights
          : mkldnn_memory_prim_user_weights;
  mkldnn_primitive_t mkldnn_memory_prim_bd_dst =
      mkldnn_memory_prim_internal_bd_dst ? mkldnn_memory_prim_internal_bd_dst
                                         : mkldnn_memory_prim_user_dst;

  const_mkldnn_primitive_t conv_data_dsts[] = {
      mkldnn_memory_prim_bd_dst};  // For conv w.r.t data o/p -> diff_src
  /* create a convolution primitive w.r.t DATA*/
  mkldnn_primitive_at_t conv_data_srcs[] = {
      mkldnn_primitive_at(
          mkldnn_memory_prim_bd_src,
          0),  // For conv w.r.t data i/p -> diff_dst and weights
      mkldnn_primitive_at(mkldnn_memory_prim_bd_weights, 0),
  };

  MKL_CHECK(mkldnn_primitive_create(&conv_bwd_data, conv_pd_data,
                                    conv_data_srcs, conv_data_dsts));

  /* Remember MKLDNN resources for cleanup */
  mkldnn_net->prim_list[mkldnn_net->prim_count++] = conv_bwd_data;
  mkldnn_net->prim_list[mkldnn_net->prim_count++] = mkldnn_memory_prim_user_src;
  mkldnn_net->prim_list[mkldnn_net->prim_count++] =
      mkldnn_memory_prim_user_weights;
  mkldnn_net->prim_list[mkldnn_net->prim_count++] = mkldnn_memory_prim_user_dst;
  if (conv_bias) {
    mkldnn_net->prim_list[mkldnn_net->prim_count++] =
        mkldnn_memory_prim_user_bias;
  }
  if (mkldnn_memory_prim_internal_bd_src) {
    mkldnn_net->prim_list[mkldnn_net->prim_count++] =
        mkldnn_memory_prim_internal_bd_src;
    mkldnn_net->prim_list[mkldnn_net->prim_count++] =
        mkldnn_reorder_prim_bd_src;
    mkldnn_net->buffer_list[mkldnn_net->buffer_count++] = conv_bd_src_buffer;
  }
  if (mkldnn_memory_prim_internal_bd_weights) {
    mkldnn_net->prim_list[mkldnn_net->prim_count++] =
        mkldnn_memory_prim_internal_bd_weights;
    mkldnn_net->prim_list[mkldnn_net->prim_count++] =
        mkldnn_reorder_prim_bd_weights;
    mkldnn_net->buffer_list[mkldnn_net->buffer_count++] =
        conv_bd_weights_buffer;
  }
  if (mkldnn_memory_prim_internal_bd_dst) {
    mkldnn_net->prim_list[mkldnn_net->prim_count++] =
        mkldnn_memory_prim_internal_bd_dst;
    mkldnn_net->prim_list[mkldnn_net->prim_count++] =
        mkldnn_reorder_prim_bd_dst;
    mkldnn_net->buffer_list[mkldnn_net->buffer_count++] = conv_bd_dst_buffer;
  }
  mkldnn_net->prim_desc_list[mkldnn_net->prim_desc_count++] = conv_pd_data;

  if (mkldnn_reorder_prim_bd_src)
    mkldnn_net->net[mkldnn_net->net_size++] = mkldnn_reorder_prim_bd_src;
  if (mkldnn_reorder_prim_bd_weights)
    mkldnn_net->net[mkldnn_net->net_size++] = mkldnn_reorder_prim_bd_weights;
  mkldnn_net->net[mkldnn_net->net_size++] = conv_bwd_data;
  if (mkldnn_reorder_prim_bd_dst)
    mkldnn_net->net[mkldnn_net->net_size++] = mkldnn_reorder_prim_bd_dst;

  return mkldnn_net;
}

mkldnn_netlist_t create_mkldnn_conv_bprop_weights_primitives(
    mkldnn_engine_t engine, int diff_dst_dims, int src_dims, int bias_dims, int diff_weights_dims,
    int stride_dims, int pad_dims, int* conv_diff_dst_sizes, int *conv_src_sizes,
    int* conv_bias_sizes, int* conv_diff_weights_sizes,
    float* conv_diff_dst, float *conv_src, float* conv_bias, float* conv_diff_weights,
    int* conv_strides, int* conv_padding) {

/* For Conv bprop weights:
   inputs :
   1. diff_dst = error/delta
   2. src = input data
   output:
   1. diff_weights = Updated weights/filters
   2. bias = bias
*/

 mkldnn_netlist_t mkldnn_net = create_mkldnn_netlist();
 mkldnn_primitive_t conv_update_weights;

  int mkl_diff_dst_dims = 4;
  int mkl_diff_weights_dims = 4;
  int mkl_src_dims = 4;
  int mkl_diff_dst_sizes[4];
  int mkl_diff_weights_sizes[4];
  int mkl_src_sizes[4];
  int mkl_strides[2];
  int mkl_padding[2];

  /* Flatten out the depth (D, M) dimension and reorder logical dimensions to
   * match MKLDNN */
  set_mkl_dimensions("convolution", conv_diff_dst_sizes, conv_src_sizes,
                     conv_diff_weights_sizes, conv_strides, conv_padding,
                     mkl_diff_dst_sizes, mkl_src_sizes, mkl_diff_weights_sizes,
                     mkl_strides, mkl_padding);

  mkldnn_memory_desc_t mkldnn_memory_desc_diff_dst_md, mkldnn_memory_desc_diff_weights_md,
      mkldnn_memory_desc_bias_md, mkldnn_memory_desc_src_md;
  MKL_CHECK(mkldnn_memory_desc_init(&mkldnn_memory_desc_diff_dst_md, mkl_diff_dst_dims,
                                    mkl_diff_dst_sizes, mkldnn_f32, mkldnn_any));
  MKL_CHECK(mkldnn_memory_desc_init(&mkldnn_memory_desc_diff_weights_md,
                                    mkl_diff_weights_dims, mkl_diff_weights_sizes,
                                    mkldnn_f32, mkldnn_any));
  if (conv_bias) {
    MKL_CHECK(mkldnn_memory_desc_init(&mkldnn_memory_desc_bias_md, 1,
                                      conv_bias_sizes, mkldnn_f32, mkldnn_x));
  }
  MKL_CHECK(mkldnn_memory_desc_init(&mkldnn_memory_desc_src_md, mkl_src_dims,
                                    mkl_src_sizes, mkldnn_f32, mkldnn_any));

// create a convolution descriptor  - logical description of convolution.
  mkldnn_convolution_desc_t conv_desc_weights;
  if (conv_bias) {
    MKL_CHECK(mkldnn_convolution_backward_weights_desc_init(
        &conv_desc_weights, mkldnn_convolution_direct,
        &mkldnn_memory_desc_src_md, &mkldnn_memory_desc_diff_weights_md,
        &mkldnn_memory_desc_bias_md, &mkldnn_memory_desc_diff_dst_md, mkl_strides,
        mkl_padding, mkl_padding, mkldnn_padding_zero));
  } else {
    MKL_CHECK(mkldnn_convolution_backward_weights_desc_init(
        &conv_desc_weights, mkldnn_convolution_direct,
        &mkldnn_memory_desc_src_md, &mkldnn_memory_desc_diff_weights_md, NULL,
        &mkldnn_memory_desc_diff_dst_md, mkl_strides, mkl_padding, mkl_padding,
        mkldnn_padding_zero));
  }

  /* create a convolution primitive descriptor - convolution descriptor bound to
   * the CPU engine */
  mkldnn_primitive_desc_t conv_pd_weights;
  MKL_CHECK(mkldnn_primitive_desc_create(&conv_pd_weights, &conv_desc_weights,
                                         engine, NULL));

  /* create memory primitives for input and output data in user format */
  mkldnn_primitive_t mkldnn_memory_prim_user_diff_dst,
      mkldnn_memory_prim_user_diff_weights, mkldnn_memory_prim_user_bias,
      mkldnn_memory_prim_user_src;
  create_mkldnn_memory_primitive(mkl_diff_dst_dims, mkl_diff_dst_sizes, mkldnn_chwn,
                                 mkldnn_f32, engine, conv_diff_dst,
                                 &mkldnn_memory_prim_user_diff_dst);
  create_mkldnn_memory_primitive(mkl_diff_weights_dims, mkl_diff_weights_sizes,
                                 mkldnn_ihwo, mkldnn_f32, engine, conv_diff_weights,
                                 &mkldnn_memory_prim_user_diff_weights);
if (conv_bias) {
    create_mkldnn_memory_primitive(1, conv_bias_sizes, mkldnn_x, mkldnn_f32,
                                   engine, conv_bias,
                                   &mkldnn_memory_prim_user_bias);
  }
  create_mkldnn_memory_primitive(mkl_src_dims, mkl_src_sizes, mkldnn_chwn,
                                 mkldnn_f32, engine, conv_src,
                                 &mkldnn_memory_prim_user_src);

  /* CONV BPROP WEIGHTS */
  /* create memory and reorder primitives for internal conversions w.r.t
   * WEIGHTS*/
  mkldnn_primitive_t mkldnn_memory_prim_internal_diff_dst,
      mkldnn_memory_prim_internal_diff_weights,
      mkldnn_memory_prim_internal_src;
  mkldnn_primitive_t mkldnn_reorder_prim_diff_dst, mkldnn_reorder_prim_diff_weights,
      mkldnn_reorder_prim_src;
  float *conv_diff_dst_buffer, *conv_diff_weights_buffer, *conv_src_buffer;

  const_mkldnn_primitive_desc_t mkl_diff_dst_pd = mkldnn_primitive_desc_query_pd(
      conv_pd_weights, mkldnn_query_diff_dst_pd, 0);
  create_mkldnn_reorder_primitive(&mkldnn_memory_prim_user_diff_dst, &mkl_diff_dst_pd,
                                  1, &mkldnn_memory_prim_internal_diff_dst,
                                  &mkldnn_reorder_prim_diff_dst);
  const_mkldnn_primitive_desc_t mkl_diff_weights_pd =
      mkldnn_primitive_desc_query_pd(conv_pd_weights,
                                     mkldnn_query_diff_weights_pd, 0);
  create_mkldnn_reorder_primitive(
      &mkldnn_memory_prim_user_diff_weights, &mkl_diff_weights_pd, 0,
      &mkldnn_memory_prim_internal_diff_weights, &mkldnn_reorder_prim_diff_weights);
  const_mkldnn_primitive_desc_t mkl_src_pd =
      mkldnn_primitive_desc_query_pd(conv_pd_weights, mkldnn_query_src_pd, 0);
  create_mkldnn_reorder_primitive(&mkldnn_memory_prim_user_src, &mkl_src_pd,
                                  1, &mkldnn_memory_prim_internal_src,
                                  &mkldnn_reorder_prim_src);

  /* Allocate memory for internal format conversions */
  if (mkldnn_memory_prim_internal_diff_dst) {
    conv_diff_dst_buffer =
        (float*)calloc(product(conv_diff_dst_sizes, diff_dst_dims), sizeof(float));
    MKL_CHECK(mkldnn_memory_set_data_handle(mkldnn_memory_prim_internal_diff_dst,
                                            conv_diff_dst_buffer));
  }
  if (mkldnn_memory_prim_internal_diff_weights) {
    conv_diff_weights_buffer = (float*)calloc(
        product(conv_diff_weights_sizes, diff_weights_dims), sizeof(float));
    MKL_CHECK(mkldnn_memory_set_data_handle(
        mkldnn_memory_prim_internal_diff_weights, conv_diff_weights_buffer));
  }
 if (mkldnn_memory_prim_internal_src) {
    conv_src_buffer =
        (float*)calloc(product(conv_src_sizes, src_dims), sizeof(float));
    MKL_CHECK(mkldnn_memory_set_data_handle(mkldnn_memory_prim_internal_src,
                                            conv_src_buffer));
  }

  /* select input and output primitives for convolution w.r.t WEIGHTS*/
  mkldnn_primitive_t mkldnn_memory_prim_diff_dst =
      mkldnn_memory_prim_internal_diff_dst ? mkldnn_memory_prim_internal_diff_dst
                                         : mkldnn_memory_prim_user_diff_dst;
  mkldnn_primitive_t mkldnn_memory_prim_diff_weights =
      mkldnn_memory_prim_internal_diff_weights
          ? mkldnn_memory_prim_internal_diff_weights
          : mkldnn_memory_prim_user_diff_weights;
  mkldnn_primitive_t mkldnn_memory_prim_src =
      mkldnn_memory_prim_internal_src ? mkldnn_memory_prim_internal_src
                                         : mkldnn_memory_prim_user_src;

  const_mkldnn_primitive_t conv_weights_dsts[2];

  if (conv_bias) {
    conv_weights_dsts[0] = mkldnn_memory_prim_diff_weights;
    conv_weights_dsts[1] = mkldnn_memory_prim_user_bias;
  } else {
    conv_weights_dsts[0] = mkldnn_memory_prim_diff_weights;
  }
  mkldnn_primitive_at_t conv_weights_srcs[] = {
      mkldnn_primitive_at(mkldnn_memory_prim_src, 0),
      mkldnn_primitive_at(mkldnn_memory_prim_diff_dst, 0),
  };

  /* create a convolution primitive w.r.t WEIGHTS*/
  MKL_CHECK(mkldnn_primitive_create(&conv_update_weights, conv_pd_weights,
                                    conv_weights_srcs, conv_weights_dsts));
 /* Remember MKLDNN resources for cleanup */
  mkldnn_net->prim_list[mkldnn_net->prim_count++] = conv_update_weights;
  mkldnn_net->prim_list[mkldnn_net->prim_count++] = mkldnn_memory_prim_user_diff_dst;
  mkldnn_net->prim_list[mkldnn_net->prim_count++] =
      mkldnn_memory_prim_user_diff_weights;
  mkldnn_net->prim_list[mkldnn_net->prim_count++] = mkldnn_memory_prim_user_src;
  if (conv_bias) {
    mkldnn_net->prim_list[mkldnn_net->prim_count++] =
        mkldnn_memory_prim_user_bias;
  }
  if (mkldnn_memory_prim_internal_diff_dst) {
    mkldnn_net->prim_list[mkldnn_net->prim_count++] =
        mkldnn_memory_prim_internal_diff_dst;
    mkldnn_net->prim_list[mkldnn_net->prim_count++] =
        mkldnn_reorder_prim_diff_dst;
    mkldnn_net->buffer_list[mkldnn_net->buffer_count++] = conv_diff_dst_buffer;
  }
  if (mkldnn_memory_prim_internal_diff_weights) {
    mkldnn_net->prim_list[mkldnn_net->prim_count++] =
        mkldnn_memory_prim_internal_diff_weights;
    mkldnn_net->prim_list[mkldnn_net->prim_count++] =
        mkldnn_reorder_prim_diff_weights;
    mkldnn_net->buffer_list[mkldnn_net->buffer_count++] =
        conv_diff_weights_buffer;
  }
  if (mkldnn_memory_prim_internal_src) {
    mkldnn_net->prim_list[mkldnn_net->prim_count++] =
        mkldnn_memory_prim_internal_src;
    mkldnn_net->prim_list[mkldnn_net->prim_count++] =
        mkldnn_reorder_prim_src;
    mkldnn_net->buffer_list[mkldnn_net->buffer_count++] = conv_src_buffer;
  }
  mkldnn_net->prim_desc_list[mkldnn_net->prim_desc_count++] = conv_pd_weights;

  if (mkldnn_reorder_prim_src)
    mkldnn_net->net[mkldnn_net->net_size++] = mkldnn_reorder_prim_src;
  if (mkldnn_reorder_prim_diff_dst)
    mkldnn_net->net[mkldnn_net->net_size++] = mkldnn_reorder_prim_diff_dst;
  mkldnn_net->net[mkldnn_net->net_size++] = conv_update_weights;
  if (mkldnn_reorder_prim_diff_weights)
    mkldnn_net->net[mkldnn_net->net_size++] = mkldnn_reorder_prim_diff_weights;

  return mkldnn_net;
}
