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

#include "mkldnn.h"
#include "mkldnn_engine.h"
#include "mkldnn_util.h"

mkldnn_engine_t init_mkldnn_engine(void) {
  mkldnn_engine_t engine;
  MKL_CHECK(mkldnn_engine_create(&engine, mkldnn_cpu, 0 /* idx */));
  return engine;
}

size_t product(int *arr, size_t size) {
  size_t prod = 1;
  for (size_t i = 0; i < size; ++i) prod *= arr[i];
  return prod;
}

void set_mkl_dimensions(char *primitive_name, int *primitive_src_sizes,
                        int *primitive_dst_sizes, int *primitive_weights_sizes,
                        int *primitive_strides, int *primitive_padding,
                        int *mkl_src_sizes, int *mkl_dst_sizes,
                        int *mkl_weights_sizes, int *mkl_strides,
                        int *mkl_padding) {
  /* Flatten out the depth (D, M) dimension and reorder logical dimensions to
   * match MKLDNN */

  /* Input: C, D, H, W, N -> N, C, H, W */
  mkl_src_sizes[0] = primitive_src_sizes[4];
  mkl_src_sizes[1] = primitive_src_sizes[0];
  mkl_src_sizes[2] = primitive_src_sizes[2];
  mkl_src_sizes[3] = primitive_src_sizes[3];

  /* Output: K, M, P, Q, N -> N, K, P, Q */
  mkl_dst_sizes[0] = primitive_dst_sizes[4];
  mkl_dst_sizes[1] = primitive_dst_sizes[0];
  mkl_dst_sizes[2] = primitive_dst_sizes[2];
  mkl_dst_sizes[3] = primitive_dst_sizes[3];

  if (!strcmp(primitive_name, "convolution")) {
    /* Weights: C, T, R, S, K ->  O, I, H, W */
    mkl_weights_sizes[0] = primitive_weights_sizes[4];
    mkl_weights_sizes[1] = primitive_weights_sizes[0];
    mkl_weights_sizes[2] = primitive_weights_sizes[2];
    mkl_weights_sizes[3] = primitive_weights_sizes[3];
  }

  if (!strcmp(primitive_name, "pooling")){
     /* Kernel: J, T, R, S -> R, S */
    mkl_weights_sizes[0] = primitive_weights_sizes[2];
    mkl_weights_sizes[1] = primitive_weights_sizes[3];
  }

  mkl_strides[0] = primitive_strides[1];
  mkl_strides[1] = primitive_strides[2];

  mkl_padding[0] = primitive_padding[1];
  mkl_padding[1] = primitive_padding[2];
}

void destroy_mkldnn_engine(mkldnn_engine_t engine) {
  MKL_CHECK(mkldnn_engine_destroy(engine));
}


void create_mkldnn_tensor(int ndims, const int* dim_sizes,
                          mkldnn_data_type_t data_type,
                          mkldnn_memory_format_t fmt,
                          mkldnn_engine_t engine,
                          mkldnn_tensor* tensor) {
    tensor->ndims = ndims;
    for (int i = 0; i < ndims; i++) tensor->sizes[i] = dim_sizes[i];

    mkldnn_memory_desc_t md;
    MKL_CHECK(mkldnn_memory_desc_init(&md, ndims, dim_sizes, data_type, fmt));
    MKL_CHECK(mkldnn_memory_primitive_desc_create(&(tensor->desc), &md, engine));
    MKL_CHECK(mkldnn_primitive_create(&(tensor->prim), tensor->desc, NULL, NULL));
}

void create_mkldnn_tensor_from_pd(int ndims, const int* dim_sizes,
                        mkldnn_memory_desc_t* md,
                        mkldnn_engine_t engine,
                        mkldnn_tensor* tensor) {
    tensor->ndims = ndims;
    for (int i = 0; i < ndims; i++) tensor->sizes[i] = dim_sizes[i];

    MKL_CHECK(mkldnn_memory_primitive_desc_create(&(tensor->desc), md, engine));
    MKL_CHECK(mkldnn_primitive_create(&(tensor->prim), tensor->desc, NULL, NULL));
}

/* Create MKLDNN memory primitives */
void create_mkldnn_memory_primitive(uint32_t n_dim, const int *dims,
                                    mkldnn_memory_format_t user_fmt,
                                    mkldnn_data_type_t data_type,
                                    mkldnn_engine_t engine, float *data,
                                    mkldnn_primitive_t *memory) {
  mkldnn_memory_desc_t prim_md;
  mkldnn_primitive_desc_t user_pd;
  MKL_CHECK(
      mkldnn_memory_desc_init(&prim_md, n_dim, dims, data_type, user_fmt));
  MKL_CHECK(mkldnn_memory_primitive_desc_create(&user_pd, &prim_md, engine));
  MKL_CHECK(mkldnn_primitive_create(memory, user_pd, NULL, NULL));
  MKL_CHECK(mkldnn_memory_set_data_handle(*memory, data));
  MKL_CHECK(mkldnn_primitive_desc_destroy(user_pd));
}

void create_mkldnn_reorder_primitive(
    mkldnn_primitive_t *user_memory,               /** in */
    const_mkldnn_primitive_desc_t *prim_memory_pd, /** in */
    int dir_is_user_to_prim,         /** in: user -> prim or prim -> user */
    mkldnn_primitive_t *prim_memory, /** out: memory primitive created */
    mkldnn_primitive_t *reorder      /** out: reorder primitive created */
    ) {
  const_mkldnn_primitive_desc_t user_memory_pd;
  mkldnn_primitive_get_primitive_desc(*user_memory, &user_memory_pd);

  if (!mkldnn_memory_primitive_desc_equal(user_memory_pd, *prim_memory_pd)) {
    MKL_CHECK(
        mkldnn_primitive_create(prim_memory, *prim_memory_pd, NULL, NULL));
    mkldnn_primitive_desc_t reorder_pd;
    if (dir_is_user_to_prim) {
      MKL_CHECK(mkldnn_reorder_primitive_desc_create(
          &reorder_pd, user_memory_pd, *prim_memory_pd));
      mkldnn_primitive_at_t inputs = {*user_memory};
      const_mkldnn_primitive_t outputs[] = {*prim_memory};
      MKL_CHECK(mkldnn_primitive_create(reorder, reorder_pd, &inputs, outputs));
    } else {
      MKL_CHECK(mkldnn_reorder_primitive_desc_create(
          &reorder_pd, *prim_memory_pd, user_memory_pd));
      mkldnn_primitive_at_t inputs = {*prim_memory};
      const_mkldnn_primitive_t outputs[] = {*user_memory};
      MKL_CHECK(mkldnn_primitive_create(reorder, reorder_pd, &inputs, outputs));
    }
  } else {
    *prim_memory = NULL;
    *reorder = NULL;
  }
}

mkldnn_opkernel_t create_empty_kernel(void) {
    mkldnn_opkernel_t op_kernel =
        (mkldnn_opkernel_t) malloc(sizeof(struct mkldnn_opkernel));
    op_kernel->num_inputs = 0;
    op_kernel->num_outputs = 0;
    op_kernel->net_size = 0;

    return op_kernel;
}

mkldnn_netlist_t create_mkldnn_netlist(void) {
  mkldnn_netlist_t mkldnn_net =
      (mkldnn_netlist_t)malloc(sizeof(struct mkldnn_netlist));
  mkldnn_net->net_size = 0;
  mkldnn_net->prim_desc_count = 0;
  mkldnn_net->prim_layouts_count = 0;
  mkldnn_net->prim_count = 0;
  mkldnn_net->buffer_count = 0;

  return mkldnn_net;
}

void destroy_mkldnn_netlist(mkldnn_netlist_t mkldnn_net) {
  for (int i = 0; i < mkldnn_net->prim_desc_count; i++) {
    MKL_CHECK(mkldnn_primitive_desc_destroy(mkldnn_net->prim_desc_list[i]));
  }

  for (int i = 0; i < mkldnn_net->prim_count; i++) {
    MKL_CHECK(mkldnn_primitive_destroy(mkldnn_net->prim_list[i]));
  }

  for (int i = 0; i < mkldnn_net->buffer_count; i++) {
    free(mkldnn_net->buffer_list[i]);
  }

  free(mkldnn_net);
}

void delete_mkldnn_tensor(mkldnn_tensor* tensor) {
    MKL_CHECK(mkldnn_primitive_desc_destroy(tensor->desc));
    MKL_CHECK(mkldnn_primitive_destroy(tensor->prim));
}

void delete_mkldnn_opkernel(mkldnn_opkernel_t opkernel) {
    for (int i = 0; i < opkernel->num_inputs; i++) {
        delete_mkldnn_tensor(&opkernel->inputs[i]);
        if (opkernel->reorder_i[i]) {
            delete_mkldnn_tensor(&opkernel->internal_inputs[i]);
            MKL_CHECK(mkldnn_primitive_destroy(opkernel->reorder_i[i]));
            free(opkernel->internal_inputs[i].buffer);
        }
    }
    for (int i = 0; i < opkernel->num_outputs; i++) {
        delete_mkldnn_tensor(&opkernel->outputs[i]);
        if (opkernel->reorder_o[i]) {
            delete_mkldnn_tensor(&opkernel->internal_outputs[i]);
            MKL_CHECK(mkldnn_primitive_destroy(opkernel->reorder_o[i]));
            free(opkernel->internal_outputs[i].buffer);
        }
    }
    MKL_CHECK(mkldnn_primitive_desc_destroy(opkernel->op_desc));
    MKL_CHECK(mkldnn_primitive_destroy(opkernel->op_prim));
}

void run_mkldnn_opkernel(mkldnn_opkernel_t opkernel) {
  MKL_CHECK(mkldnn_stream_create(&opkernel->stream, mkldnn_eager));
  mkldnn_primitive_t error_primitive;
  mkldnn_status_t s =
      mkldnn_stream_submit(opkernel->stream, opkernel->net_size,
                           opkernel->net, &error_primitive);
  if (s != mkldnn_success) {
    printf(
        "[%s:%d] error: mkldnn_stream_submit returns %d, error_primitive: %p\n",
        __FILE__, __LINE__, s, error_primitive);
    exit(2);
  }
  MKL_CHECK(mkldnn_stream_wait(opkernel->stream, opkernel->net_size, NULL));
  MKL_CHECK(mkldnn_stream_destroy(opkernel->stream));
}

void run_mkldnn_netlist(mkldnn_netlist_t mkldnn_net) {
  MKL_CHECK(mkldnn_stream_create(&mkldnn_net->stream, mkldnn_eager));
  mkldnn_primitive_t error_primitive;
  mkldnn_status_t s =
      mkldnn_stream_submit(mkldnn_net->stream, mkldnn_net->net_size,
                           mkldnn_net->net, &error_primitive);
  if (s != mkldnn_success) {
    printf(
        "[%s:%d] error: mkldnn_stream_submit returns %d, error_primitive: %p\n",
        __FILE__, __LINE__, s, error_primitive);
    exit(2);
  }
  MKL_CHECK(mkldnn_stream_wait(mkldnn_net->stream, mkldnn_net->net_size, NULL));
  MKL_CHECK(mkldnn_stream_destroy(mkldnn_net->stream));
}

void cleanup_mkldnn(mkldnn_netlist_t mkldnn_net) {
  destroy_mkldnn_netlist(mkldnn_net);
}

mkldnn_primitive_desc_t query_opkernel_layout(mkldnn_opkernel_t opkernel, int index) {
    assert (index < opkernel->num_outputs);
    mkldnn_memory_desc_t md = *mkldnn_primitive_desc_query_memory_d(opkernel->outputs[index].desc);
    if (md.format == mkldnn_x) { 
        return NULL;
    } else {
        return opkernel->outputs[index].desc;
    }
}

mkldnn_primitive_desc_t query_prim_layout(mkldnn_netlist_t mkldnn_net, int index) {
  return mkldnn_net->prim_layouts[index];
}

int compare_layouts(mkldnn_primitive_desc_t a, mkldnn_primitive_desc_t b) {
  if (mkldnn_memory_primitive_desc_equal(a, b))
    return 1;
  else
    return 0;
}

void create_reorder_kernel(
    mkldnn_engine_t engine,
    const_mkldnn_primitive_desc_t input_pd_const,
    int ndims, int* sizes,
    mkldnn_netlist_t mkldnn_net
  ) {
  mkldnn_primitive_desc_t reorder_pd;
  mkldnn_memory_desc_t output_memory_desc;
  mkldnn_primitive_desc_t output_pd, input_pd;

  int mkl_dims = 4;
  int mkl_sizes[4];
  /* Input: C, D, H, W, N -> N, C, H, W */
  mkl_sizes[0] = sizes[4];
  mkl_sizes[1] = sizes[0];
  mkl_sizes[2] = sizes[2];
  mkl_sizes[3] = sizes[3];

  mkldnn_primitive_desc_clone(&input_pd, input_pd_const);
  mkldnn_memory_desc_init(&output_memory_desc, mkl_dims, mkl_sizes, mkldnn_f32, mkldnn_chwn);
  mkldnn_memory_primitive_desc_create(&output_pd, &output_memory_desc, engine);
  mkldnn_reorder_primitive_desc_create(
          &reorder_pd, input_pd, output_pd);
  mkldnn_net->prim_desc_list[mkldnn_net->prim_desc_count++] = reorder_pd;
  mkldnn_net->prim_desc_list[mkldnn_net->prim_desc_count++] = input_pd;
  mkldnn_net->prim_desc_list[mkldnn_net->prim_desc_count++] = output_pd;
}

void alloc_reorder_kernel(
    mkldnn_engine_t engine,
    float* src, float* dst,
    mkldnn_netlist_t mkldnn_net) {

    mkldnn_primitive_t src_prim, dst_prim, reorder;
    mkldnn_primitive_desc_t reorder_pd = mkldnn_net->prim_desc_list[0];
    MKL_CHECK(mkldnn_primitive_create(&src_prim, mkldnn_net->prim_desc_list[1], NULL, NULL));
    MKL_CHECK(mkldnn_primitive_create(&dst_prim, mkldnn_net->prim_desc_list[2], NULL, NULL));
    MKL_CHECK(mkldnn_memory_set_data_handle(src_prim, src));
    MKL_CHECK(mkldnn_memory_set_data_handle(dst_prim, dst));

    mkldnn_primitive_at_t inputs = {src_prim};
    const_mkldnn_primitive_t outputs[] = {dst_prim};
    MKL_CHECK(mkldnn_primitive_create(&reorder, reorder_pd, &inputs, outputs));
    mkldnn_net->prim_list[mkldnn_net->prim_count++] = reorder;
    mkldnn_net->net[mkldnn_net->net_size++] = reorder;
}
