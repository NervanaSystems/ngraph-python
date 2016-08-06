python ../freeze_graph.py \
  --input_graph=graph.pb.txt \
  --input_checkpoint=model.ckpt \
  --output_graph=graph_froze.pb \
  --output_node_names=softmax_linear/add
