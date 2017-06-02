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

void create_mkldnn_innerproduct_fprop_kernel(
    mkldnn_engine_t engine, int src_dims, int weights_dims, int bias_dims,
    int dst_dims, int* src_sizes, int* weights_sizes, int* bias_sizes,
    int* dst_sizes, mkldnn_primitive_desc_t src_pd,
    mkldnn_primitive_desc_t weights_pd, mkldnn_primitive_desc_t bias_pd,
    mkldnn_data_type_t data_type, mkldnn_opkernel_t opkernel) {
  // assert(src_dims == 2);
  // assert(weights_dims == 2);
  // assert(bias_dims == src_dims);

  // Let MKL pick the best format (mkldnn_any)
  mkldnn_memory_desc_t src_md, weights_md, bias_md, dst_md;
  MKL_CHECK(mkldnn_memory_desc_init(&src_md, src_dims, src_sizes, data_type,
                                    mkldnn_any));
  MKL_CHECK(mkldnn_memory_desc_init(&weights_md, weights_dims, weights_sizes,
                                    data_type, mkldnn_any));
#if 0
    MKL_CHECK(mkldnn_memory_desc_init(&bias_md, bias_dims,
                                       bias_sizes, data_type,
                                       mkldnn_x));
#endif
  MKL_CHECK(mkldnn_memory_desc_init(&dst_md, dst_dims, dst_sizes, data_type,
                                    mkldnn_any));

  // -
  /* create an inner product descriptor  - logical description of inner product
   */
  mkldnn_inner_product_desc_t ip_any_desc;
  MKL_CHECK(mkldnn_inner_product_forward_desc_init(
      &ip_any_desc, mkldnn_forward_inference, &src_md, &weights_md,
      /* &bias_md */ NULL, &dst_md));

  /* create an inner product primitive descriptor - inner product descriptor
     bound to the CPU engine */
  MKL_CHECK(mkldnn_primitive_desc_create(&opkernel->op_desc, &ip_any_desc,
                                         engine, NULL));

  // ------------------------------------------------------------------------
  // Query primitive chosen layouts.
  const_mkldnn_primitive_desc_t kernel_src_pd =
      mkldnn_primitive_desc_query_pd(opkernel->op_desc, mkldnn_query_src_pd, 0);
  const_mkldnn_primitive_desc_t kernel_weights_pd =
      mkldnn_primitive_desc_query_pd(opkernel->op_desc, mkldnn_query_weights_pd,
                                     0);
  // TODO: what about bias?
  const_mkldnn_primitive_desc_t kernel_dst_pd =
      mkldnn_primitive_desc_query_pd(opkernel->op_desc, mkldnn_query_dst_pd, 0);
  // ------------------------------------------------------------------------

  /* Create memory descriptors that are demanded from this op. */
  if (src_pd) {
    mkldnn_memory_desc_t md = *(mkldnn_primitive_desc_query_memory_d(
        (const_mkldnn_primitive_desc_t)src_pd));
    create_mkldnn_tensor_from_pd(src_dims, src_sizes, &md, engine,
                                 &(opkernel->inputs[0]));
  } else {
    create_mkldnn_tensor(src_dims, src_sizes, data_type, mkldnn_nc, engine,
                         &(opkernel->inputs[0]));
  }

  if (weights_pd) {
    mkldnn_memory_desc_t md = *(mkldnn_primitive_desc_query_memory_d(
        (const_mkldnn_primitive_desc_t)weights_pd));
    create_mkldnn_tensor_from_pd(weights_dims, weights_sizes, &md, engine,
                                 &(opkernel->inputs[1]));
  } else {
    create_mkldnn_tensor(weights_dims, weights_sizes, data_type, mkldnn_oi,
                         engine, &(opkernel->inputs[1]));
  }
  mkldnn_memory_desc_t output_md =
      *mkldnn_primitive_desc_query_memory_d(kernel_dst_pd);
  create_mkldnn_tensor_from_pd(dst_dims, dst_sizes, &output_md, engine,
                               &(opkernel->outputs[0]));

#if 0
    // TODO: add support for bias
    if (bias_pd) {
        mkldnn_memory_desc_t md = *(
            mkldnn_primitive_desc_query_memory_d(
                (const_mkldnn_primitive_desc_t) bias_pd));
        create_mkldnn_tensor_from_pd(bias_dims, bias_sizes, &md,
                            engine, &(opkernel->inputs[0]));
    } else {
        create_mkldnn_tensor(bias_dims, bias_sizes, data_type, mkldnn_x,
                            engine, &(opkernel->inputs[0]));
    }
#endif
  // ------------------------------------------------------------------------

  opkernel->num_inputs = 2;
  opkernel->num_outputs = 1;

  // ------------------------------------------------------------------------
  // Reorder inputs

  if (!mkldnn_memory_primitive_desc_equal(opkernel->inputs[0].desc,
                                          kernel_src_pd)) {
    mkldnn_memory_desc_t md =
        *mkldnn_primitive_desc_query_memory_d(kernel_src_pd);
    create_mkldnn_tensor_from_pd(src_dims, src_sizes, &md, engine,
                                 &(opkernel->internal_inputs[0]));
    mkldnn_primitive_desc_t reorder_pd;
    MKL_CHECK(mkldnn_reorder_primitive_desc_create(
        &reorder_pd, opkernel->inputs[0].desc, kernel_src_pd));
    mkldnn_primitive_at_t inputs[] = {
        mkldnn_primitive_at(opkernel->inputs[0].prim, 0)};
    const_mkldnn_primitive_t outputs[] = {opkernel->internal_inputs[0].prim};
    MKL_CHECK(mkldnn_primitive_create(&(opkernel->reorder_i[0]), reorder_pd,
                                      inputs, outputs));
  } else {
    opkernel->reorder_i[0] = NULL;
  }

  if (!mkldnn_memory_primitive_desc_equal(opkernel->inputs[1].desc,
                                          kernel_weights_pd)) {
    mkldnn_memory_desc_t md =
        *mkldnn_primitive_desc_query_memory_d(kernel_weights_pd);
    create_mkldnn_tensor_from_pd(weights_dims, weights_sizes, &md, engine,
                                 &(opkernel->internal_inputs[1]));
    mkldnn_primitive_desc_t reorder_pd;
    MKL_CHECK(mkldnn_reorder_primitive_desc_create(
        &reorder_pd, opkernel->inputs[1].desc, kernel_weights_pd));
    mkldnn_primitive_at_t inputs[] = {
        mkldnn_primitive_at(opkernel->inputs[1].prim, 0)};
    const_mkldnn_primitive_t outputs[] = {opkernel->internal_inputs[1].prim};
    MKL_CHECK(mkldnn_primitive_create(&(opkernel->reorder_i[1]), reorder_pd,
                                      inputs, outputs));
  } else {
    opkernel->reorder_i[1] = NULL;
  }

  if (!mkldnn_memory_primitive_desc_equal(opkernel->outputs[0].desc,
                                          kernel_dst_pd)) {
    mkldnn_memory_desc_t md =
        *mkldnn_primitive_desc_query_memory_d(kernel_dst_pd);
    create_mkldnn_tensor_from_pd(src_dims, src_sizes, &md, engine,
                                 &(opkernel->internal_outputs[0]));
    mkldnn_primitive_desc_t reorder_pd;
    MKL_CHECK(mkldnn_reorder_primitive_desc_create(
        &reorder_pd, opkernel->outputs[0].desc, kernel_dst_pd));
    mkldnn_primitive_at_t inputs[] = {
        mkldnn_primitive_at(opkernel->internal_outputs[0].prim, 0)};
    const_mkldnn_primitive_t outputs[] = {opkernel->outputs[0].prim};
    MKL_CHECK(mkldnn_primitive_create(&(opkernel->reorder_o[0]), reorder_pd,
                                      inputs, outputs));
  } else {
    opkernel->reorder_o[0] = NULL;
  }

  // ------------------------------------------------------------------------

  /* Allocate memory for internal format conversions */
  if (opkernel->reorder_i[0]) {
    void* tmp_buf = alloc_memory(product(src_sizes, src_dims), data_type);
    opkernel->internal_inputs[0].buffer = tmp_buf;
    MKL_CHECK(mkldnn_memory_set_data_handle(opkernel->internal_inputs[0].prim,
                                            tmp_buf));
  }
  if (opkernel->reorder_i[1]) {
    void* tmp_buf =
        alloc_memory(product(weights_sizes, weights_dims), data_type);
    opkernel->internal_inputs[1].buffer = tmp_buf;
    MKL_CHECK(mkldnn_memory_set_data_handle(opkernel->internal_inputs[1].prim,
                                            tmp_buf));
  }
  if (opkernel->reorder_o[0]) {
    void* tmp_buf = alloc_memory(product(dst_sizes, dst_dims), data_type);
    opkernel->internal_outputs[0].buffer = tmp_buf;
    MKL_CHECK(mkldnn_memory_set_data_handle(opkernel->internal_outputs[0].prim,
                                            tmp_buf));
  }

  // ------------------------------------------------------------------------

  /* select input and output primitives for convolution */
  mkldnn_primitive_t mkldnn_memory_prim_src =
      opkernel->reorder_i[0] ? opkernel->internal_inputs[0].prim
                             : opkernel->inputs[0].prim;
  mkldnn_primitive_t mkldnn_memory_prim_weights =
      opkernel->reorder_i[1] ? opkernel->internal_inputs[1].prim
                             : opkernel->inputs[1].prim;
  mkldnn_primitive_t mkldnn_memory_prim_dst =
      opkernel->reorder_o[0] ? opkernel->internal_outputs[0].prim
                             : opkernel->outputs[0].prim;

  // ------------------------------------------------------------------------

  const_mkldnn_primitive_t ip_dsts[] = {mkldnn_memory_prim_dst};

  /* create a convolution primitive */
  mkldnn_primitive_at_t ip_srcs[] = {
      mkldnn_primitive_at(mkldnn_memory_prim_src, 0),
      mkldnn_primitive_at(mkldnn_memory_prim_weights, 0)};

  MKL_CHECK(mkldnn_primitive_create(&opkernel->op_prim, opkernel->op_desc,
                                    ip_srcs, ip_dsts));

  if (opkernel->reorder_i[0])
    opkernel->net[opkernel->net_size++] = opkernel->reorder_i[0];
  if (opkernel->reorder_i[1])
    opkernel->net[opkernel->net_size++] = opkernel->reorder_i[1];
  opkernel->net[opkernel->net_size++] = opkernel->op_prim;
  if (opkernel->reorder_o[0])
    opkernel->net[opkernel->net_size++] = opkernel->reorder_o[0];
}
