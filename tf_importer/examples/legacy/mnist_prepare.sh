#!/bin/bash

# tensorflow fetch data and train for 2 epochs. this will dump the training
# graph and model checkpoints
python ../3rd_party/mnist/fully_connected_feed.py

# freeze the model checkpoints to protobuf. this is ued for the mnist mlp
# inference example
python ../3rd_party/freeze_graph.py \
    --input_graph=graph.pb.txt \
    --input_checkpoint=model.ckpt \
    --output_graph=graph_froze.pb \
    --output_node_names=softmax_linear/add
