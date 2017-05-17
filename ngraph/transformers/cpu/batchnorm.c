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
    mkldnn_engine_t engine,
    int src_dims, int weights_dims, int dest_dims,int mean_sizes, int variance_sizes,
    int *batchnorm_src_sizes, int *batchnorm_weights_sizes,
    int *batchnorm_dst_sizes, double epsilon, mkldnn_primitive_desc_t input_src_pd,
    mkldnn_primitive_desc_t input_weights_pd, mkldnn_primitive_desc_t input_mean_pd,
    mkldnn_primitive_desc_t input_variance_pd,  
    mkldnn_data_type_t data_type,
    mkldnn_opkernel_t opkernel) {

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

    // C,H,W,N -> N, C, H, W
    mkl_src_sizes[0] = batchnorm_src_sizes[3];
    mkl_src_sizes[1] = batchnorm_src_sizes[0];
    mkl_src_sizes[2] = batchnorm_src_sizes[1];
    mkl_src_sizes[3] = batchnorm_src_sizes[2];

    mkl_dst_sizes[0] = batchnorm_src_sizes[3];
    mkl_dst_sizes[1] = batchnorm_src_sizes[0];
    mkl_dst_sizes[2] = batchnorm_src_sizes[1];
    mkl_dst_sizes[3] = batchnorm_src_sizes[2];


    mkl_weight_sizes[0] = batchnorm_weights_sizes[0];
    mkl_weight_sizes[1] = batchnorm_weights_sizes[1];
    mkl_mean_sizes[0] = mean_sizes;
    mkl_variance_sizes[0] = variance_sizes;


    /* create a batch norm descriptor - logical descriptor of the batch norm */
     mkldnn_memory_desc_t mkldnn_memory_desc_src_md;
     if (input_src_pd) {
          mkldnn_memory_desc_src_md = *(mkldnn_primitive_desc_query_memory_d((const_mkldnn_primitive_desc_t)input_src_pd));
     } else {
         MKL_CHECK(mkldnn_memory_desc_init(&mkldnn_memory_desc_src_md, mkl_src_dims,
                                           mkl_src_sizes, data_type, mkldnn_chwn));
     }

    mkldnn_batch_normalization_desc_t batch_norm_desc;
    MKL_CHECK(mkldnn_batch_normalization_forward_desc_init(&batch_norm_desc, mkldnn_forward_training,
                                                &mkldnn_memory_desc_src_md, epsilon, mkldnn_use_global_stats | mkldnn_use_scaleshift));

    /* create a batch norm primitive descriptor - convolution descriptor bound to the CPU engine */
    MKL_CHECK(mkldnn_primitive_desc_create(&opkernel->op_desc, &batch_norm_desc, engine, NULL));

    if (input_src_pd) {
            mkldnn_memory_desc_t md = *(mkldnn_primitive_desc_query_memory_d((const_mkldnn_primitive_desc_t)input_src_pd));
            create_mkldnn_tensor_from_pd(mkl_src_dims, mkl_src_sizes, &md,
                                engine, &(opkernel->inputs[0]));
        } else {
            create_mkldnn_tensor(mkl_src_dims, mkl_src_sizes, data_type, mkldnn_chwn,
                                engine, &(opkernel->inputs[0]));
        }

    if (input_mean_pd) {
            mkldnn_memory_desc_t md = *(mkldnn_primitive_desc_query_memory_d((const_mkldnn_primitive_desc_t)input_mean_pd));
            create_mkldnn_tensor_from_pd(mkl_mean_dims, mkl_mean_sizes, &md,
                                engine, &(opkernel->inputs[1]));
        } else {
            create_mkldnn_tensor(mkl_mean_dims, mkl_mean_sizes, data_type, mkldnn_x,
                                engine, &(opkernel->inputs[1]));
        }

    if (input_variance_pd) {
            mkldnn_memory_desc_t md = *(mkldnn_primitive_desc_query_memory_d((const_mkldnn_primitive_desc_t)input_variance_pd));
            create_mkldnn_tensor_from_pd(mkl_variance_dims, mkl_variance_sizes, &md,
                                engine, &(opkernel->inputs[2]));
        } else {
            create_mkldnn_tensor(mkl_variance_dims, mkl_variance_sizes, data_type, mkldnn_x,
                                engine, &(opkernel->inputs[2]));
        }

    if (input_weights_pd) {
            mkldnn_memory_desc_t md = *(mkldnn_primitive_desc_query_memory_d((const_mkldnn_primitive_desc_t)input_weights_pd));
            create_mkldnn_tensor_from_pd(mkl_weight_dims, mkl_weight_sizes, &md,
                                engine, &(opkernel->inputs[3]));
        } else {
            create_mkldnn_tensor(mkl_weight_dims, mkl_weight_sizes, data_type, mkldnn_nc,
                                engine, &(opkernel->inputs[3]));
        }

    mkldnn_memory_desc_t dst_md = mkldnn_memory_desc_src_md;
    create_mkldnn_tensor_from_pd(mkl_dst_dims, mkl_dst_sizes, &dst_md,
                          engine, &(opkernel->outputs[0]));

    opkernel->num_inputs = 4;
    opkernel-> num_outputs = 1;

    // No reorders required
    opkernel->reorder_i[0] = NULL;
    opkernel->reorder_i[1] = NULL;
    opkernel->reorder_i[2] = NULL;
    opkernel->reorder_i[3] = NULL;
    opkernel->reorder_o[0] = NULL;

    /* create batch norm primitive */
    const_mkldnn_primitive_t batch_norm_prim_dsts[] = { opkernel->outputs[0].prim};
    mkldnn_primitive_at_t batch_norm_prim_srcs[] = {
                                            mkldnn_primitive_at(opkernel->inputs[0].prim,0),
                                            mkldnn_primitive_at(opkernel->inputs[1].prim,0),
                                            mkldnn_primitive_at(opkernel->inputs[2].prim,0),
                                            mkldnn_primitive_at(opkernel->inputs[3].prim,0)};

    MKL_CHECK(mkldnn_primitive_create(&opkernel->op_prim, opkernel->op_desc, batch_norm_prim_srcs, batch_norm_prim_dsts));
    opkernel->net[opkernel->net_size++] = opkernel->op_prim;

}
