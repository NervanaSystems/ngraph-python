# ----------------------------------------------------------------------------
# Copyright 2017 Nervana Systems Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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
