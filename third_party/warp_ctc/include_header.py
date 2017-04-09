#!/usr/bin/env python                                                                                   
# ----------------------------------------------------------------------------                          
# Copyright 2015 Nervana Systems Inc. All rights reserved.                                              
# Unauthorized copying or viewing of this file outside Nervana Systems Inc.,                            
# via any medium is strictly prohibited. Proprietary and confidential.                                  
# ----------------------------------------------------------------------------                          
header = """
typedef struct {
int loc; 
unsigned int num_threads;
void* stream;
int blank_label;
}ctcOptions;

int get_workspace_size(const int* const label_lengths,
                       const int* const input_lengths,
                       int alphabet_size, 
                       int minibatch,
                       ctcOptions options,
                       size_t* size_bytes);

int compute_ctc_loss(const float* const activations,
                     float* gradients,
                     const int* const flat_labels,
                     const int* const label_lengths,
                     const int* const input_lengths,
                     int alphabet_size,
                     int minibatch,
                     float *costs,
                     void *workspace, 
                     ctcOptions options);
"""

def ctc_header():
    return header
