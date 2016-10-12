## Protobuf

```protobuf  
message GraphDef {
  repeated NodeDef node = 1;
  VersionDef versions = 4;
  int32 version = 3 [deprecated = true];
  FunctionDefLibrary library = 2;
};

message NodeDef {
  string name = 1;
  string op = 2;
  repeated string input = 3;
  string device = 4;
  map<string, AttrValue> attr = 5;
};

message FunctionDefLibrary {
  repeated FunctionDef function = 1;
  repeated GradientDef gradient = 2;
}

message FunctionDef {
  OpDef signature = 1;
  repeated Node node = 2;  // function.node.ret are unique.
  message Node {
    repeated string ret = 1;
    string op = 2;
    repeated string arg = 3;
    repeated string dep = 4;
    map<string, AttrValue> attr = 5;
  }
}

message GradientDef {
  string function_name = 1;  // The function name.
  string gradient_func = 2;  // The gradient function's name.
}

message AttrValue {
  message ListValue {
    repeated bytes s = 2;                        // "list(string)"
    repeated int64 i = 3 [packed = true];        // "list(int)"
    repeated float f = 4 [packed = true];        // "list(float)"
    repeated bool b = 5 [packed = true];         // "list(bool)"
    repeated DataType type = 6 [packed = true];  // "list(type)"
    repeated TensorShapeProto shape = 7;         // "list(shape)"
    repeated TensorProto tensor = 8;             // "list(tensor)"
  }
  oneof value {
    bytes s = 2;                 // "string"
    int64 i = 3;                 // "int"
    float f = 4;                 // "float"
    bool b = 5;                  // "bool"
    DataType type = 6;           // "type"
    TensorShapeProto shape = 7;  // "shape"
    TensorProto tensor = 8;      // "tensor"
    ListValue list = 1;          // any "list(...)"
    NameAttrList func = 10;
    string placeholder = 9;
  }
}

message NameAttrList {
  string name = 1;
  map<string, AttrValue> attr = 2;
}
```

## Python

```python
tf.AggregationMethod
tf.Assert
tf.AttrValue
tf.ConfigProto
tf.DType
tf.DeviceSpec
tf.Dimension
tf.Event
tf.FIFOQueue
tf.FixedLenFeature
tf.FixedLenSequenceFeature
tf.FixedLengthRecordReader
tf.GPUOptions
tf.GRAPH_DEF_VERSION
tf.GRAPH_DEF_VERSION_MIN_CONSUMER
tf.GRAPH_DEF_VERSION_MIN_PRODUCER
tf.Graph
tf.GraphDef
tf.GraphKeys
tf.GraphOptions
tf.HistogramProto
tf.IdentityReader
tf.IndexedSlices
tf.InteractiveSession
tf.LogMessage
tf.NameAttrList
tf.NoGradient
tf.NodeDef
tf.OpError
tf.Operation
tf.OptimizerOptions
tf.PaddingFIFOQueue
tf.Print
tf.QUANTIZED_DTYPES
tf.QueueBase
tf.RandomShuffleQueue
tf.ReaderBase
tf.RegisterGradient
tf.RegisterShape
tf.RunMetadata
tf.RunOptions
tf.Session
tf.SessionLog
tf.SparseTensor
tf.SparseTensorValue
tf.Summary
tf.TFRecordReader
tf.Tensor
tf.TensorArray
tf.TensorShape
tf.TextLineReader
tf.VarLenFeature
tf.Variable
tf.VariableScope
tf.WholeFileReader
tf.__builtins__
tf.__doc__
tf.__file__
tf.__name__
tf.__package__
tf.__path__
tf.__version__
tf.abs
tf.absolute_import
tf.accumulate_n
tf.acos
tf.add
tf.add_check_numerics_ops
tf.add_n
tf.add_to_collection
tf.all_variables
tf.app
tf.arg_max
tf.arg_min
tf.argmax
tf.argmin
tf.as_dtype
tf.asin
tf.assert_equal
tf.assert_integer
tf.assert_less
tf.assert_less_equal
tf.assert_negative
tf.assert_non_negative
tf.assert_non_positive
tf.assert_positive
tf.assert_proper_iterable
tf.assert_rank
tf.assert_rank_at_least
tf.assert_type
tf.assert_variables_initialized
tf.assign
tf.assign_add
tf.assign_sub
tf.atan
tf.audio_summary
tf.batch_cholesky
tf.batch_cholesky_solve
tf.batch_fft
tf.batch_fft2d
tf.batch_fft3d
tf.batch_ifft
tf.batch_ifft2d
tf.batch_ifft3d
tf.batch_matmul
tf.batch_matrix_band_part
tf.batch_matrix_determinant
tf.batch_matrix_diag
tf.batch_matrix_diag_part
tf.batch_matrix_inverse
tf.batch_matrix_solve
tf.batch_matrix_solve_ls
tf.batch_matrix_triangular_solve
tf.batch_self_adjoint_eig
tf.batch_to_space
tf.bfloat16
tf.bfloat16_ref
tf.bitcast
tf.bool
tf.bool_ref
tf.boolean_mask
tf.bytes
tf.case
tf.cast
tf.ceil
tf.check_numerics
tf.cholesky
tf.cholesky_solve
tf.clip_by_average_norm
tf.clip_by_global_norm
tf.clip_by_norm
tf.clip_by_value
tf.compat
tf.complex
tf.complex128
tf.complex128_ref
tf.complex64
tf.complex64_ref
tf.complex_abs
tf.concat
tf.cond
tf.conj
tf.constant
tf.constant_initializer
tf.contrib
tf.control_dependencies
tf.convert_to_tensor
tf.convert_to_tensor_or_indexed_slices
tf.core
tf.cos
tf.count_up_to
tf.create_partitioned_variables
tf.cross
tf.decode_csv
tf.decode_json_example
tf.decode_raw
tf.delete_session_tensor
tf.depth_to_space
tf.deserialize_many_sparse
tf.device
tf.diag
tf.diag_part
tf.digamma
tf.div
tf.division
tf.double
tf.double_ref
tf.dynamic_partition
tf.dynamic_stitch
tf.edit_distance
tf.equal
tf.erf
tf.erfc
tf.errors
tf.exp
tf.expand_dims
tf.extract_image_patches
tf.fft
tf.fft2d
tf.fft3d
tf.fill
tf.flags
tf.float16
tf.float16_ref
tf.float32
tf.float32_ref
tf.float64
tf.float64_ref
tf.floor
tf.floordiv
tf.foldl
tf.foldr
tf.gather
tf.gather_nd
tf.get_collection
tf.get_collection_ref
tf.get_default_graph
tf.get_default_session
tf.get_seed
tf.get_session_handle
tf.get_session_tensor
tf.get_variable
tf.get_variable_scope
tf.gfile
tf.global_norm
tf.gradients
tf.greater
tf.greater_equal
tf.group
tf.half
tf.half_ref
tf.histogram_fixed_width
tf.histogram_summary
tf.identity
tf.ifft
tf.ifft2d
tf.ifft3d
tf.igamma
tf.igammac
tf.imag
tf.image
tf.image_summary
tf.import_graph_def
tf.initialize_all_tables
tf.initialize_all_variables
tf.initialize_local_variables
tf.initialize_variables
tf.int16
tf.int16_ref
tf.int32
tf.int32_ref
tf.int64
tf.int64_ref
tf.int8
tf.int8_ref
tf.inv
tf.invert_permutation
tf.is_finite
tf.is_inf
tf.is_nan
tf.is_non_decreasing
tf.is_numeric_tensor
tf.is_strictly_increasing
tf.is_variable_initialized
tf.lbeta
tf.less
tf.less_equal
tf.lgamma
tf.lin_space
tf.linspace
tf.list_diff
tf.listdiff
tf.load_file_system_library
tf.load_op_library
tf.local_variables
tf.log
tf.logging
tf.logical_and
tf.logical_not
tf.logical_or
tf.logical_xor
tf.make_template
tf.map_fn
tf.matching_files
tf.matmul
tf.matrix_determinant
tf.matrix_inverse
tf.matrix_solve
tf.matrix_solve_ls
tf.matrix_triangular_solve
tf.maximum
tf.merge_all_summaries
tf.merge_summary
tf.minimum
tf.mod
tf.moving_average_variables
tf.mul
tf.multinomial
tf.name_scope
tf.neg
tf.nn
tf.no_op
tf.no_regularizer
tf.not_equal
tf.one_hot
tf.ones
tf.ones_initializer
tf.ones_like
tf.op_scope
tf.pack
tf.pad
tf.parse_example
tf.parse_single_example
tf.parse_single_sequence_example
tf.placeholder
tf.placeholder_with_default
tf.polygamma
tf.pow
tf.print_function
tf.py_func
tf.python
tf.python_io
tf.qint16
tf.qint16_ref
tf.qint32
tf.qint32_ref
tf.qint8
tf.qint8_ref
tf.quint16
tf.quint16_ref
tf.quint8
tf.quint8_ref
tf.random_crop
tf.random_normal
tf.random_normal_initializer
tf.random_shuffle
tf.random_uniform
tf.random_uniform_initializer
tf.range
tf.rank
tf.read_file
tf.real
tf.reduce_all
tf.reduce_any
tf.reduce_join
tf.reduce_max
tf.reduce_mean
tf.reduce_min
tf.reduce_prod
tf.reduce_sum
tf.register_tensor_conversion_function
tf.report_uninitialized_variables
tf.reset_default_graph
tf.reshape
tf.resource_loader
tf.reverse
tf.reverse_sequence
tf.round
tf.rsqrt
tf.saturate_cast
tf.scalar_mul
tf.scalar_summary
tf.scan
tf.scatter_add
tf.scatter_sub
tf.scatter_update
tf.segment_max
tf.segment_mean
tf.segment_min
tf.segment_prod
tf.segment_sum
tf.select
tf.self_adjoint_eig
tf.serialize_many_sparse
tf.serialize_sparse
tf.set_random_seed
tf.shape
tf.shape_n
tf.sigmoid
tf.sign
tf.sin
tf.size
tf.slice
tf.space_to_batch
tf.space_to_depth
tf.sparse_add
tf.sparse_concat
tf.sparse_fill_empty_rows
tf.sparse_mask
tf.sparse_matmul
tf.sparse_merge
tf.sparse_placeholder
tf.sparse_reduce_sum
tf.sparse_reorder
tf.sparse_reset_shape
tf.sparse_retain
tf.sparse_segment_mean
tf.sparse_segment_mean_grad
tf.sparse_segment_sqrt_n
tf.sparse_segment_sqrt_n_grad
tf.sparse_segment_sum
tf.sparse_softmax
tf.sparse_split
tf.sparse_tensor_dense_matmul
tf.sparse_tensor_to_dense
tf.sparse_to_dense
tf.sparse_to_indicator
tf.split
tf.sqrt
tf.square
tf.squared_difference
tf.squeeze
tf.stop_gradient
tf.string
tf.string_ref
tf.string_to_hash_bucket
tf.string_to_hash_bucket_fast
tf.string_to_hash_bucket_strong
tf.string_to_number
tf.sub
tf.sysconfig
tf.tan
tf.tanh
tf.test
tf.tile
tf.to_bfloat16
tf.to_double
tf.to_float
tf.to_int32
tf.to_int64
tf.trace
tf.train
tf.trainable_variables
tf.transpose
tf.truediv
tf.truncated_normal
tf.truncated_normal_initializer
tf.tuple
tf.uint16
tf.uint16_ref
tf.uint8
tf.uint8_ref
tf.uniform_unit_scaling_initializer
tf.unique
tf.unique_with_counts
tf.unpack
tf.unsorted_segment_sum
tf.user_ops
tf.variable_axis_size_partitioner
tf.variable_op_scope
tf.variable_scope
tf.verify_tensor_all_finite
tf.where
tf.while_loop
tf.zeros
tf.zeros_initializer
tf.zeros_like
tf.zeta
```
