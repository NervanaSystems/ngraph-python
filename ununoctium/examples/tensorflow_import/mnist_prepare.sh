#!/bin/bash

# tensorflow fetch data and train for 2 epochs, this will dump the training
# graph and model checkpoints
python 3rd_party/mnist/fully_connected_feed.py

# free the model checkpoints to protobuf
python 3rd_party/freeze_graph.py \
    --input_graph=variable_graph.pb.txt \
    --input_checkpoint=model.ckpt \
    --output_graph=variable_graph_froze.pb \
    --output_node_names=add
