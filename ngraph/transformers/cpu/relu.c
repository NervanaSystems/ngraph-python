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

void create_mkldnn_relu_fprop_kernel(
        mkldnn_engine_t engine,
        int src_size, double slope,
        mkldnn_primitive_desc_t input_src_pd,
        mkldnn_opkernel_t opkernel) {
    
    int mkl_src_dims = 1;
    int mkl_dst_dims = 1;
    int mkl_src_sizes[1];
    int mkl_dst_sizes[1];

    mkl_src_sizes[0] = src_size;
    mkl_dst_sizes[0] = src_size;
    
    mkldnn_memory_desc_t mkldnn_memory_desc_src_md;
    if (input_src_pd) {
        mkldnn_memory_desc_src_md = *(mkldnn_primitive_desc_query_memory_d((const_mkldnn_primitive_desc_t)input_src_pd));
    } else {
        MKL_CHECK(mkldnn_memory_desc_init(&mkldnn_memory_desc_src_md, mkl_src_dims,
                                          mkl_src_sizes, mkldnn_f32, mkldnn_x));
    }
    mkldnn_relu_desc_t relu_desc;
    MKL_CHECK(mkldnn_relu_forward_desc_init(&relu_desc, mkldnn_forward_training,
              &mkldnn_memory_desc_src_md, slope));
    MKL_CHECK(mkldnn_primitive_desc_create(&opkernel->op_desc, &relu_desc, engine, NULL));
    
    if (input_src_pd) {
        mkldnn_memory_desc_t md = *(mkldnn_primitive_desc_query_memory_d((const_mkldnn_primitive_desc_t)input_src_pd));
        create_mkldnn_tensor_from_pd(mkl_src_dims, mkl_src_sizes, &md,
                            engine, &(opkernel->inputs[0]));
    } else {
        create_mkldnn_tensor(mkl_src_dims, mkl_src_sizes, mkldnn_f32, mkldnn_x,
                            engine, &(opkernel->inputs[0]));
    }
    mkldnn_memory_desc_t dst_md = mkldnn_memory_desc_src_md;
    create_mkldnn_tensor_from_pd(mkl_dst_dims, mkl_dst_sizes, &dst_md,
                        engine, &(opkernel->outputs[0]));
    opkernel->num_inputs = 1;
    opkernel->num_outputs = 1;

    // No reorders required
    opkernel->reorder_i[0] = NULL;
    opkernel->reorder_o[0] = NULL;

    const_mkldnn_primitive_t relu_dsts[] = { opkernel->outputs[0].prim };
    mkldnn_primitive_at_t relu_srcs[] =
        { mkldnn_primitive_at(opkernel->inputs[0].prim, 0) };

    MKL_CHECK(mkldnn_primitive_create(&opkernel->op_prim, opkernel->op_desc, relu_srcs, relu_dsts));
    opkernel->net[opkernel->net_size++] = opkernel->op_prim;
}

void run_mkldnn_relu_fprop_kernel(
        void* relu_src, void* relu_dst,
        mkldnn_opkernel_t opkernel
        ) {
    MKL_CHECK(mkldnn_memory_set_data_handle(opkernel->inputs[0].prim, relu_src));
    MKL_CHECK(mkldnn_memory_set_data_handle(opkernel->outputs[0].prim, relu_dst));
    run_mkldnn_opkernel(opkernel);
}

void create_mkldnn_relu_bprop_kernel(
        mkldnn_engine_t engine,
        int src_size, double slope,
        mkldnn_primitive_desc_t input_fprop_src_pd,
        mkldnn_primitive_desc_t input_error_pd, 
        mkldnn_opkernel_t opkernel) {
    int mkl_src_dims = 1;
    int mkl_dst_dims = 1;
    int mkl_src_sizes[1];
    int mkl_dst_sizes[1];

    mkl_src_sizes[0] = src_size;
    mkl_dst_sizes[0] = src_size;

    mkldnn_memory_desc_t mkldnn_memory_desc_src_md, mkldnn_memory_desc_fprop_src_md;
    if (input_fprop_src_pd && input_error_pd) {
        mkldnn_memory_desc_fprop_src_md = *(mkldnn_primitive_desc_query_memory_d((const_mkldnn_primitive_desc_t)input_fprop_src_pd));
        mkldnn_memory_desc_src_md = *(mkldnn_primitive_desc_query_memory_d((const_mkldnn_primitive_desc_t)input_error_pd));
    } else if (input_fprop_src_pd) {
        // fprop_src - MKL 5-D, error - 1D, dst - 1D
        mkldnn_memory_desc_t md = *(mkldnn_primitive_desc_query_memory_d((const_mkldnn_primitive_desc_t)input_fprop_src_pd));
        mkldnn_memory_desc_fprop_src_md = md;
        MKL_CHECK(mkldnn_memory_desc_init(&mkldnn_memory_desc_src_md, md.ndims,
                                          md.dims, mkldnn_f32, mkldnn_chwn));
    } else if (input_error_pd) {
        // fprop_src - 1D, error - 5D MKL, dst - 5D MKL
        mkldnn_memory_desc_t md = *(mkldnn_primitive_desc_query_memory_d((const_mkldnn_primitive_desc_t)input_error_pd));
        MKL_CHECK(mkldnn_memory_desc_init(&mkldnn_memory_desc_fprop_src_md, md.ndims,
                                          md.dims, mkldnn_f32, mkldnn_chwn));
        mkldnn_memory_desc_src_md = md;
    } else {
        MKL_CHECK(mkldnn_memory_desc_init(&mkldnn_memory_desc_src_md, mkl_src_dims,
                                          mkl_src_sizes, mkldnn_f32, mkldnn_x));
        MKL_CHECK(mkldnn_memory_desc_init(&mkldnn_memory_desc_fprop_src_md, mkl_src_dims,
                                          mkl_src_sizes, mkldnn_f32, mkldnn_x));
    }

    mkldnn_relu_desc_t relu_desc;
    MKL_CHECK(mkldnn_relu_backward_desc_init(&relu_desc, &mkldnn_memory_desc_src_md,
                                             &mkldnn_memory_desc_src_md, slope));
    MKL_CHECK(mkldnn_primitive_desc_create(&opkernel->op_desc, &relu_desc, engine, NULL));

    create_mkldnn_tensor_from_pd(mkl_src_dims, mkl_src_sizes, &mkldnn_memory_desc_fprop_src_md, 
            engine, &(opkernel->inputs[0]));
    create_mkldnn_tensor_from_pd(mkl_src_dims, mkl_src_sizes, &mkldnn_memory_desc_src_md, 
            engine, &(opkernel->inputs[1]));

    mkldnn_memory_desc_t dst_md = mkldnn_memory_desc_src_md;
    create_mkldnn_tensor_from_pd(mkl_dst_dims, mkl_dst_sizes, &dst_md, 
            engine, &(opkernel->outputs[0]));
    opkernel->num_inputs = 2;
    opkernel->num_outputs = 1;

    // If layouts dont match, convert fprop_src to match error. Dst matches error
    if (!mkldnn_memory_primitive_desc_equal(opkernel->inputs[0].desc, opkernel->inputs[1].desc)) {
        mkldnn_memory_desc_t md = *mkldnn_primitive_desc_query_memory_d(opkernel->inputs[1].desc);
        create_mkldnn_tensor_from_pd(mkl_src_dims, mkl_src_sizes, &md, engine, &(opkernel->internal_inputs[0]));
        mkldnn_primitive_desc_t reorder_pd;
        MKL_CHECK(mkldnn_reorder_primitive_desc_create(&reorder_pd,
                    opkernel->inputs[0].desc, opkernel->inputs[1].desc));
        mkldnn_primitive_at_t inputs[] = {opkernel->inputs[0].prim};
        const_mkldnn_primitive_t outputs[] = {opkernel->internal_inputs[0].prim};
        MKL_CHECK(mkldnn_primitive_create(&(opkernel->reorder_i[0]), reorder_pd, inputs, outputs));
    } else {
        opkernel->reorder_i[0] = NULL;
    }

    opkernel->reorder_i[1] = NULL;
    opkernel->reorder_o[0] = NULL;

    if (opkernel->reorder_i[0]) {
        float* tmp_buf =
            (float*)calloc(src_size, sizeof(float));
        opkernel->internal_inputs[0].buffer = tmp_buf;
        MKL_CHECK(mkldnn_memory_set_data_handle(opkernel->internal_inputs[0].prim,
                                                tmp_buf));
    }
    
    const_mkldnn_primitive_t relu_dsts[] = { opkernel->outputs[0].prim };
    mkldnn_primitive_at_t relu_srcs[] =
        { mkldnn_primitive_at(opkernel->inputs[0].prim, 0),
          mkldnn_primitive_at(opkernel->inputs[1].prim, 0),
        };

    MKL_CHECK(mkldnn_primitive_create(&opkernel->op_prim, opkernel->op_desc, relu_srcs, relu_dsts));
    opkernel->net[opkernel->net_size++] = opkernel->op_prim;
}

void run_mkldnn_relu_bprop_kernel(
        void* relu_fprop_src,
        void* relu_error_src, void* relu_dst,
        mkldnn_opkernel_t opkernel
        ) {
    MKL_CHECK(mkldnn_memory_set_data_handle(opkernel->inputs[0].prim, relu_fprop_src));
    MKL_CHECK(mkldnn_memory_set_data_handle(opkernel->inputs[1].prim, relu_error_src));
    MKL_CHECK(mkldnn_memory_set_data_handle(opkernel->outputs[0].prim, relu_dst));
    run_mkldnn_opkernel(opkernel);
}

/* Create list of mkldnn primitives to run relu bprop */
mkldnn_netlist_t create_mkldnn_relu_bprop_primitives(
            mkldnn_engine_t engine,
            float* relu_src, float* relu_out, double slope,
            float* relu_fprop_src, int src_size
        )
{
    mkldnn_netlist_t mkldnn_net = create_mkldnn_netlist();
    mkldnn_primitive_t relu_back;

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
    MKL_CHECK(mkldnn_relu_backward_desc_init(&relu_desc, &mkldnn_memory_desc_src_md,
                                             &mkldnn_memory_desc_src_md, slope));
    mkldnn_primitive_desc_t relu_pd;
    MKL_CHECK(mkldnn_primitive_desc_create(&relu_pd, &relu_desc, engine, NULL));

    mkldnn_primitive_t mkldnn_memory_prim_user_src, mkldnn_memory_prim_user_dst,
                       mkldnn_memory_prim_user_fprop_src;
    create_mkldnn_memory_primitive(mkl_src_dims, mkl_src_sizes, mkldnn_x,
                                   mkldnn_f32, engine, relu_src,
                                   &mkldnn_memory_prim_user_src);
    create_mkldnn_memory_primitive(mkl_dst_dims, mkl_dst_sizes, mkldnn_x,
                                   mkldnn_f32, engine, relu_out,
                                   &mkldnn_memory_prim_user_dst);
    create_mkldnn_memory_primitive(mkl_src_dims, mkl_src_sizes, mkldnn_x,
                                   mkldnn_f32, engine,
                                   relu_fprop_src,
                                   &mkldnn_memory_prim_user_fprop_src);

    const_mkldnn_primitive_t relu_dsts[] = { mkldnn_memory_prim_user_dst };

    mkldnn_primitive_at_t relu_srcs[] =
        { mkldnn_primitive_at(mkldnn_memory_prim_user_fprop_src, 0),
                              mkldnn_primitive_at(mkldnn_memory_prim_user_src, 0)
        };

    MKL_CHECK(mkldnn_primitive_create(&relu_back, relu_pd, relu_srcs, relu_dsts));

     /* Remember MKLDNN resources for cleanup */
    mkldnn_net->prim_list[mkldnn_net->prim_count++] = relu_back;
    mkldnn_net->prim_list[mkldnn_net->prim_count++] = mkldnn_memory_prim_user_src;
    mkldnn_net->prim_list[mkldnn_net->prim_count++] = mkldnn_memory_prim_user_dst;
    mkldnn_net->prim_list[mkldnn_net->prim_count++] =
        mkldnn_memory_prim_user_fprop_src;

    mkldnn_net->prim_desc_list[mkldnn_net->prim_desc_count++] = relu_pd;

    mkldnn_net->net[mkldnn_net->net_size++] = relu_back;

    return mkldnn_net;
}
