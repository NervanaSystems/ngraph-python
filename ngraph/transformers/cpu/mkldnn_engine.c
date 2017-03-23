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

mkldnn_engine_t init_mkldnn_engine(void) 
{
    mkldnn_engine_t engine;
    MKL_CHECK(mkldnn_engine_create(&engine, mkldnn_cpu, 0 /* idx */));
    return engine;
}

size_t product(int *arr, size_t size) {
    size_t prod = 1;
    for (size_t i = 0; i < size; ++i) prod *= arr[i];
    return prod;
}

void set_mkl_dimensions(char *primitive_name, int *primitive_src_sizes, int *primitive_dst_sizes, int *primitive_weights_sizes, int *primitive_strides, int *primitive_padding, \
                                    int *mkl_src_sizes, int *mkl_dst_sizes, int *mkl_weights_sizes, int *mkl_strides , int *mkl_padding){

   /* Flatten out the depth (D, M) dimension and reorder logical dimensions to match MKLDNN */

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

    if (!strcmp(primitive_name ,"convolution")){
        /* Weights: C, T, R, S, K ->  O, I, H, W */
        mkl_weights_sizes[0] = primitive_weights_sizes[4];
        mkl_weights_sizes[1] = primitive_weights_sizes[0];
        mkl_weights_sizes[2] = primitive_weights_sizes[2];
        mkl_weights_sizes[3] = primitive_weights_sizes[3];
    }

    mkl_strides[0] = primitive_strides[1];
    mkl_strides[1] = primitive_strides[2];

    mkl_padding[0] = primitive_padding[1];
    mkl_padding[1] = primitive_padding[2];

}

void destroy_mkldnn_engine(mkldnn_engine_t engine)
{
    MKL_CHECK(mkldnn_engine_destroy(engine));
}

/* Create MKLDNN memory primitives */
void create_mkldnn_memory_primitive(uint32_t n_dim, const int *dims,
        mkldnn_memory_format_t user_fmt, mkldnn_data_type_t data_type,
        mkldnn_engine_t engine, float *data, 
        mkldnn_primitive_t *memory)
{
    mkldnn_memory_desc_t prim_md;
    mkldnn_primitive_desc_t user_pd;
    MKL_CHECK(mkldnn_memory_desc_init(&prim_md, n_dim, dims, data_type, user_fmt));
    MKL_CHECK(mkldnn_memory_primitive_desc_create(&user_pd, &prim_md, engine));
    MKL_CHECK(mkldnn_primitive_create(memory, user_pd, NULL, NULL));
    MKL_CHECK(mkldnn_memory_set_data_handle(*memory, data));
    MKL_CHECK(mkldnn_primitive_desc_destroy(user_pd));
}


void create_mkldnn_reorder_primitive(
        mkldnn_primitive_t *user_memory, /** in */
        const_mkldnn_primitive_desc_t *prim_memory_pd, /** in */
        int dir_is_user_to_prim, /** in: user -> prim or prim -> user */
        mkldnn_primitive_t *prim_memory, /** out: memory primitive created */
        mkldnn_primitive_t *reorder /** out: reorder primitive created */
        )
{
    const_mkldnn_primitive_desc_t user_memory_pd;
    mkldnn_primitive_get_primitive_desc(*user_memory, &user_memory_pd);

    if (!mkldnn_memory_primitive_desc_equal(user_memory_pd, *prim_memory_pd)) {
        MKL_CHECK(mkldnn_primitive_create(prim_memory, *prim_memory_pd, NULL, NULL));
        mkldnn_primitive_desc_t reorder_pd;
        if (dir_is_user_to_prim) {
            MKL_CHECK(mkldnn_reorder_primitive_desc_create(&reorder_pd, user_memory_pd, *prim_memory_pd));
            mkldnn_primitive_at_t inputs = { *user_memory };
            const_mkldnn_primitive_t outputs[] = { *prim_memory };
            MKL_CHECK(mkldnn_primitive_create(reorder, reorder_pd, &inputs, outputs));
        } else {
            MKL_CHECK(mkldnn_reorder_primitive_desc_create(&reorder_pd, *prim_memory_pd, user_memory_pd));
            mkldnn_primitive_at_t inputs = { *prim_memory };
            const_mkldnn_primitive_t outputs[] = { *user_memory };
            MKL_CHECK(mkldnn_primitive_create(reorder, reorder_pd, &inputs, outputs));
        }
    } else {
        *prim_memory = NULL;
        *reorder = NULL;
    }
}

mkldnn_netlist_t create_mkldnn_netlist(void)
{
    mkldnn_netlist_t mkldnn_net = (mkldnn_netlist_t) malloc(sizeof(struct mkldnn_netlist));
    mkldnn_net->net_size = 0;
    mkldnn_net->prim_desc_count = 0;
    mkldnn_net->prim_count = 0;
    mkldnn_net->buffer_count = 0;

    return mkldnn_net;
}

void destroy_mkldnn_netlist(mkldnn_netlist_t mkldnn_net)
{

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

void run_mkldnn_netlist(mkldnn_netlist_t mkldnn_net)
{
    MKL_CHECK(mkldnn_stream_create(&mkldnn_net->stream, mkldnn_eager));
    mkldnn_primitive_t error_primitive;
    mkldnn_status_t s = mkldnn_stream_submit(mkldnn_net->stream, mkldnn_net->net_size, mkldnn_net->net, &error_primitive);
    if (s != mkldnn_success) {
        printf("[%s:%d] error: mkldnn_stream_submit returns %d, error_primitive: %p\n", __FILE__, __LINE__, s, error_primitive);
        exit(2);
    }
    MKL_CHECK(mkldnn_stream_wait(mkldnn_net->stream, mkldnn_net->net_size, NULL));
    MKL_CHECK(mkldnn_stream_destroy(mkldnn_net->stream));
}

void cleanup_mkldnn(mkldnn_netlist_t mkldnn_net)
{
    destroy_mkldnn_netlist(mkldnn_net);
}
