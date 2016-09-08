python ../freeze_graph.py \
  --input_graph=variable_graph.pb.txt \
  --input_checkpoint=model.ckpt \
  --output_graph=variable_graph_froze.pb \
  --output_node_names=add
