Relay2ONNX
----------
We provide 74 operator conversions and tested 15 onnx models from [ONNX Model Zoo](https://github.com/onnx/models) and [EfficientNet PyTorch](https://github.com/lukemelas/EfficientNet-PyTorch) in [model_test.py](https://github.com/itri-tvm/Relay2ONNX/blob/relay2onnx/tests/python/converter/onnx/model_test.py).

Model List
----------
|Model|
|----------|
|MobileNet|
|ResNet-50|
|SqueezeNet|
|VGG-16|
|AlexNet|
|GoogleNet|
|CaffeNet|
|RCNN_ILSVRC13|
|DenseNet-121|
|Inception_V1|
|Inception_V2|
|ShuffleNet_V1|
|ShuffleNet_V2|
|ZFNet-512|
|Efficientnet|

Operator List
-------------
|Relay|ONNX|
|------|------|
|abs|Abs|
|add|Add|
|logical_and|And|
|argmax|ArgMax|
|argmin|ArgMin|
|nn.avg_poolid|AveragePool|
|nn.batch_norm|BatchNormalization|
|cast|Cast|
|ceil|Ceil|
|clip|Clip|
|concatenate|Concat|
|Constant|Constant|
|full_like, full|ConstantOfShape|
|nn.conv1d, nn.conv2d, nn.conv3d, fused_nn_bias_add_nn_conv...|Conv|
|nn.conv1d_transpose, nn.conv2d_transpose, fused_nn_bias_add_nn_conv...|ConvTranspose|
|nn.depth_to_space|DepthToSpace|
|divide|Div|
|nn.dropout|Dropout|
|equal|Equal|
|erf|Erf|
|exp|Exp|
|broadcast_to|Expand|
|nn.batch_flatten|Flatten|
|floor|Floor|
|take|Gather|
|fused_gemm|Gemm|
|nn.global_avg_poolid|GlobalAveragePool|
|nn.global_max_poolid|GlobalMaxPool|
|greater|Greater|
|copy|Identity|
|nn.instance_norm|InstanceNormalization|
|nn.leaky_relu|LeakyRelu|
|less|Less|
|log|Log|
|nn.log_softmax|LogSoftmax|
|nn.lrn|LRN|
|nn.batch_norm, nn.dense, fused_matmul|MatMul|
|maximun|Max|
|nn.max_pool|MaxPool|
|mean|Mean|
|min|Min|
|multiply|Mul|
|negative|Neg|
|argwhere, fused_nonzero|NonZero|
|logical_not|Not|
|one_hot|OneHot|
|logical_or|Or|
|nn.pad|Pad|
|power|Pow|
|nn.prelu|PRelu|
|max|ReduceMax|
|mean|ReduceMean|
|min|ReduceMin|
|prod|ReduceProd|
|sum|ReduceSum|
|nn.relu|Relu|
|reshape|Reshape|
|image.resize|Resize|
|shape_of|Shape|
|sigmoid|Sigmoid|
|sign|Sign|
|strided_slice|Slice|
|nn.softmax|Softmax|
|nn.space_to_space|SpaceToDepth|
|split|Split|
|sqrt|Sqrt|
|squeeze|Squeeze|
|sub|Sub|
|tanh|Tanh|
|tile|Tile|
|transpose|Transpose|
|fused_unsqueeze|Unsqueeze|
|upsampling|Upsample|
|where|Where|
