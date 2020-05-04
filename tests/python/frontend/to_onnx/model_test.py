import onnx
import numpy as np
import tvm
from tvm import relay
from tvm.contrib import graph_runtime
import logging
import torch
from onnx import numpy_helper
from efficientnet_pytorch import EfficientNet
from PIL import Image
from torchvision import transforms
import netron
from tvm.relay.frontend.common import infer_shape, infer_value_simulated
from tvm.relay import testing
from tvm.relay.op.nn.nn import batch_flatten
from topi.transform import shape

def get_rt_mod(inputs, mod, params):
    target = {"cpu": "llvm"}   
    cpu_ctx = tvm.context("cpu")
    with relay.build_config(opt_level=2):
        graph, lib, params = relay.build(mod, target=target, params=params)
    rt_mod = graph_runtime.create(graph, lib, ctx=[cpu_ctx])
    for k, v in inputs.items():
        rt_mod.set_input(key = k, value = v)
    rt_mod.set_input(**params)
    rt_mod.run()
    return rt_mod
def print_different(before, after):
    before_lines = []
    after_lines = []
    before = before.split('\n')
    after = after.split('\n')
    i, j = 0, 0
    while i < len(before) or j < len(after):
        if i >= len(before):
            after_lines.append(after[j])
            i, j = i-1, j+1
            continue
        if j >= len(after):
            before_lines.append(before[i])
            i, j = i+1, j-1
            continue
        if before[i] == after[j]:
            i, j = i+1, j+1
        else:
            ti, tj = i, j
            while ti < len(before) and before[ti] != after[j]:
                ti += 1
            while tj < len(after) and before[i] != after[tj]:
                tj += 1
            if ti == len(before) and tj == len(after):
                before_lines.append(before[i])
                after_lines.append(after[j])
                i, j = i+1, j+1
            else:
                i, j = (ti, j) if ti - i < tj - j else (i, tj)
    print('before:')
    for line in before_lines:
        print(line)
    print('after:')
    for line in after_lines:
        print(line)
def load_img_and_run(model_path, data_path, model_name, show_netron =False):
    print('{}...'.format(model_name))
    onnx_model = onnx.load(model_path)
    if show_netron:
        netron.start(model_path, port=9930)
    before_opset = onnx_model.opset_import[0].version
    # Preprocess image
    params_name = [param.name for param in onnx_model.graph.initializer]
    for input_tensor in onnx_model.graph.input:
        if input_tensor.name not in params_name:
            input_name = input_tensor.name
            input_shape = tuple(dim.dim_value for dim in input_tensor.type.tensor_type.shape.dim)
            break
    if model_name=='efficientnet-b1':
        tfms = transforms.Compose([transforms.Resize((input_shape[-2],input_shape[-1])), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])
        data = np.array(tfms(Image.open(data_path)), "float32")
    else:
        data = np.array(Image.open(data_path), "float32")
    new_shape = [1 for s in range(len(input_shape)-len(data.shape))]
    new_shape.extend(data.shape)
    data = data.reshape(new_shape)
    inputs = {input_name:data}
    shape_dict = {k: v.shape for k,v in inputs.items()}
    output_shape = [dim.dim_value for dim in onnx_model.graph.output[0].type.tensor_type.shape.dim]
    mod, params = relay.frontend.from_onnx(onnx_model, shape=shape_dict)
    rt_mod = get_rt_mod(inputs, mod, params)
    before_output = rt_mod.get_output(0, tvm.nd.empty(output_shape, 'float32')).asnumpy()
    after_opset = before_opset
    print('Opset from',before_opset, 'to', after_opset)
    onnx_model = relay.frontend.to_onnx(mod, params,model_name, opset=after_opset)
    onnx.save(onnx_model, 'model.onnx')
    if show_netron:
        netron.start('model.onnx', port=3030)
    onnx_model = onnx.load('model.onnx')
    mod, params = relay.frontend.from_onnx(onnx_model, shape=shape_dict)
    rt_mod = get_rt_mod(inputs, mod, params)
    after_output = rt_mod.get_output(0, tvm.nd.empty(output_shape, 'float32')).asnumpy()
    assert np.array_equal(before_output, after_output), 'The outputs of are different!'
def load_pb_and_run(model_path, data_path, model_name, show_netron = False):
    print('{}...'.format(model_name))
    onnx_model = onnx.load_model(model_path)
    if show_netron:
        netron.start(model_path)
    new_tensor = onnx.TensorProto()
    with open(data_path, 'rb') as f:
        new_tensor.ParseFromString(f.read())
    before_opset = onnx_model.opset_import[0].version
    params_name = [param.name for param in onnx_model.graph.initializer]
    shape_list = []
    data = numpy_helper.to_array(new_tensor) 
    data_shape =  data.shape
    for input_tensor in onnx_model.graph.input:
        if input_tensor.name not in params_name:
            input_name = input_tensor.name
            input_shape =[]
            for i, dim in enumerate(input_tensor.type.tensor_type.shape.dim):
                x = dim.dim_value if dim.dim_value != 0 else data_shape[i]
                input_shape.append(x)
            shape_list.append((input_name, input_shape))
    if len(shape_list) ==1:
        data = data.reshape(shape_list[0][1])
        inputs = {shape_list[0][0]:data}
    elif len(shape_list) ==2:
        data = data.reshape(shape_list[0][1])
        inputs = {shape_list[0][0]:data, shape_list[1][0]: np.array(data.shape[-2:])}    
    output_shape = [dim.dim_value for dim in onnx_model.graph.output[0].type.tensor_type.shape.dim]
    shape_dict = shape={k:v for k,v in shape_list}
    mod, params = relay.frontend.from_onnx(onnx_model, shape=shape_dict)
    type = infer_shape(mod['main']).ret_type 
    rt_mod = get_rt_mod(inputs, mod, params)
    before_output = rt_mod.get_output(0, tvm.nd.empty(type.shape, type.dtype)).asnumpy()
    after_opset = before_opset
    print('Opset from', before_opset, 'to', after_opset)
    onnx_model = relay.frontend.to_onnx(mod, params,model_name, opset=after_opset)
    onnx.save(onnx_model, 'model.onnx')
    if show_netron:
        netron.start('model.onnx', port=3030)
    onnx_model = onnx.load('model.onnx')
    mod, params = relay.frontend.from_onnx(onnx_model, shape=shape_dict)
    type = infer_shape(mod['main']).ret_type 
    rt_mod = get_rt_mod(inputs, mod, params)
    after_output = rt_mod.get_output(0, tvm.nd.empty(type.shape, type.dtype)).asnumpy()
    assert np.array_equal(before_output, after_output), 'The outputs of are different!'
def lenet5(show_netron = False):
    model_name = 'lenet5'
    model_path = '../../model/lenet5/self/lenet5_0.onnx'
    data_path = '../../model/lenet5/self/225.png'
    load_img_and_run(model_path, data_path, model_name, show_netron)
def mobilenetv2(show_netron = False):
    model_name = 'mobilenetv2'
    model_path = '../../model/mobilenet_v2/mobilenetv2-1.0/mobilenetv2-1.0.onnx'
    data_path = '../../model/mobilenet_v2/mobilenetv2-1.0/test_data_set_0/input_0.pb'
    load_pb_and_run(model_path, data_path, model_name, show_netron)
def resnet50(show_netron = False):
    model_name = 'resnet50'
    model_path = '../../model/resnet/resnet50v2/resnet50v2.onnx'
    data_path = '../../model/resnet/resnet50v2/test_data_set_0/input_0.pb'
    load_pb_and_run(model_path, data_path, model_name, show_netron)
def squeezenet(show_netron = False):
    model_name = 'squeezenet'
    model_path = '../../model/squeezenet/squeezenet1.1/squeezenet1.1.onnx'
    data_path = '../../model/squeezenet/squeezenet1.1/test_data_set_0/input_0.pb'
    load_pb_and_run(model_path, data_path, model_name, show_netron)
def vgg19_bn(show_netron = False):
    model_name = 'vgg19_bn'
    model_path = '../../model/vgg/vgg19-bn/vgg19-bn.onnx'
    data_path = '../../model/vgg/vgg19-bn/test_data_set_0/input_0.pb'
    load_pb_and_run(model_path, data_path, model_name, show_netron)
def alexnet(show_netron = False):
    model_name = 'alexnet'
    model_path = '../../model/alexnet/bvlc_alexnet/model.onnx'
    data_path = '../../model/alexnet/bvlc_alexnet/test_data_set_0/input_0.pb'
    load_pb_and_run(model_path, data_path, model_name, show_netron)
def googlenet(show_netron = False):
    model_name = 'googlenet'
    model_path = '../../model/googlenet/bvlc_googlenet/model.onnx'
    data_path = '../../model/googlenet/bvlc_googlenet/test_data_set_0/input_0.pb'
    load_pb_and_run(model_path, data_path, model_name, show_netron)
def caffenet(show_netron = False):
    model_name = 'caffenet'
    model_path = '../../model/caffenet/bvlc_reference_caffenet/model.onnx'
    data_path = '../../model/caffenet/bvlc_reference_caffenet/test_data_set_0/input_0.pb'
    load_pb_and_run(model_path, data_path, model_name, show_netron)
def rcnn(show_netron = False):
    model_name = 'rcnn'
    model_path = '../../model/rcnn/bvlc_reference_rcnn_ilsvrc13/model.onnx'
    data_path = '../../model/rcnn/bvlc_reference_rcnn_ilsvrc13/test_data_set_0/input_0.pb'
    load_pb_and_run(model_path, data_path, model_name, show_netron)
def densenet121(show_netron = False):
    model_name = 'densenet121'
    model_path = '../../model/densenet/densenet121/model.onnx'
    data_path = '../../model/densenet/densenet121/test_data_set_0/input_0.pb'
    load_pb_and_run(model_path, data_path, model_name, show_netron)
def inception_v1(show_netron = False):
    model_name = 'inception_v1'
    model_path = '../../model/inception_v1/inception_v1/model.onnx'
    data_path = '../../model/inception_v1/inception_v1/test_data_set_0/input_0.pb'
    load_pb_and_run(model_path, data_path, model_name, show_netron)
def inception_v2(show_netron = False):
    model_name = 'inception_v2'
    model_path = '../../model/inception_v2/inception_v2/model.onnx'
    data_path = '../../model/inception_v2/inception_v2/test_data_set_0/input_0.pb'
    load_pb_and_run(model_path, data_path, model_name, show_netron)
def shufflenet_v1(show_netron = False):
    model_name = 'shufflenet_v1'
    model_path = '../../model/shufflenet_v1/shufflenet/model.onnx'
    data_path = '../../model/shufflenet_v1/shufflenet/test_data_set_0/input_0.pb'
    load_pb_and_run(model_path, data_path, model_name, show_netron)
def shufflenet_v2(show_netron = False):
    model_name = 'shufflenet_v2'
    model_path = '../../model/shufflenet_v2/test_shufflenetv2/model.onnx'
    data_path = '../../model/shufflenet_v2/test_shufflenetv2/test_data_set_0/input_0.pb'
    load_pb_and_run(model_path, data_path, model_name, show_netron)
def zfnet512(show_netron = False):
    model_name = 'zfnet512'
    model_path = '../../model/zfnet/zfnet512/model.onnx'
    data_path = '../../model/zfnet/zfnet512/test_data_set_0/input_0.pb'
    load_pb_and_run(model_path, data_path, model_name, show_netron)
def save_efficientnet_from_pytorch(model_name,model_path):
    image_size = EfficientNet.get_image_size(model_name)
    # Load model
    model = EfficientNet.from_pretrained(model_name)
    model.set_swish(memory_efficient=False) # swish->x*sigmoid(x)
    model.eval()
    # Dummy input for ONNX
    dummy_input = torch.randn(1, 3, 224, 224)  
    # Export with ONNX
    torch.onnx.export(model, dummy_input, model_path, verbose=True)
def efficientnet(show_netron = False):
    model_name = 'efficientnet-b1'
    model_path = '../../model/efficientnet/pytorch/efficientnet-b1.onnx'
    data_path = '../../model/efficientnet/pytorch/img2.jpg'
    # save_efficientnet_from_pytorch(model_name,model_path)
    load_img_and_run(model_path,data_path, model_name, show_netron)
def  unit_test(mod, model_name, params, show_netron = False, opset = None, shape_dict = {}):
    before_print = str(mod)
    expr = mod['main'].body
    before_output = infer_value_simulated(expr, params).asnumpy()
    onnx_model = relay.frontend.to_onnx(mod, params, model_name, opset)
    onnx.save(onnx_model, 'model.onnx')
    if show_netron:
        netron.start('model.onnx')
    for v in relay.analysis.free_vars(expr):
        if v not in params:
            shape_dict[v.name_hint] = v.checked_type.shape
    mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)
    affter_print = str(mod)
    after_output = infer_value_simulated(expr, params).asnumpy()
    # print_different(before_print, affter_print)
    assert np.array_equal(before_output, after_output), 'The outputs of are different:\nBefore:\n{}\nAfter:\n{}'.format(before_output, after_output)
def strided_slice_test(show_netron = False):
    model_name = 'strided_slice'
    print('{}...'.format(model_name))
    data = relay.var('data_weight',shape = (10,1))
    p0 = relay.strided_slice(data, [1],[3], strides = [2])
    mod, params =testing.create_workload(p0)
    unit_test(mod, model_name, params, show_netron)
def one_hot_test(show_netron = False):
    model_name = 'one_hot'
    print('{}...'.format(model_name))
    indices = relay.var('indices_weight',shape = (3,1))
    values = relay.var('values_weight',shape = (2,1))
    off_value, on_value = relay.take(values,relay.const(0)),relay.take(values, relay.const(1))
    p0 = relay.one_hot(indices, on_value, off_value, depth = 2, axis=-1, dtype='float32')
    mod, params =testing.create_workload(p0)
    unit_test(mod, model_name, params, show_netron)
def full_like_test(show_netron = False):
    model_name = 'full_like'
    print('{}...'.format(model_name))
    data = relay.var('data_weight', shape=(10,1), dtype = 'int8')
    fill_value = relay.const(1.0)
    #fill_value = relay.var('1_weight', shape=(1))
    p0 = relay.full_like(data, fill_value)
    mod, params =testing.create_workload(p0)
    unit_test(mod, model_name, params, show_netron)
def full_test(show_netron = False):
    model_name = 'full'
    print('{}...'.format(model_name))
    data = relay.const(10)
    p0 = relay.full(data, shape=(10,), dtype = 'int8')
    mod, params =testing.create_workload(p0)
    unit_test(mod, model_name, params, show_netron, opset=9)
def dense_test(show_netron = False):
    model_name = 'nn.dense'
    print('{}...'.format(model_name))
    data = relay.var('data_weight',shape = (1, 2, 3, 3))
    w0 = relay.var('0_weight', shape = (1, 18))
    b0 = relay.var('0_bias', shape = (1,))
    p0 = relay.nn.batch_flatten(data)
    p1 = relay.nn.dense(p0, w0)
    p2 = relay.nn.bias_add(p1, b0, axis=0)
    mod, params =testing.create_workload(p2)
    unit_test(mod, model_name, params, show_netron, opset=9)    
def matmul_test(show_netron = False):
    model_name = 'fused_matmul'
    print('{}...'.format(model_name))
    data = relay.var('data_weight',shape = (1, 1,5))
    w0 = relay.var('0_weight', shape = (1, 5, 1))
    b0 = relay.var('0_bias', shape = (1,))
    p0 = relay.transpose(w0, axes=(0,2,1))
    p1 = relay.nn.batch_matmul(data, p0)
    p2 = relay.nn.bias_add(p1, b0, axis=0)
    mod, params =testing.create_workload(p2)
    unit_test(mod, model_name, params, show_netron, opset=9)    

def split_test(show_netron = False):
    model_name = 'split'
    print('{}...'.format(model_name))
    data = relay.var('data_weight',shape = (1, 10, 3, 3))
    p0 = relay.split(data, indices_or_sections=2, axis =1)
    p1 = p0[0]
    mod, params =testing.create_workload(p1)
    unit_test(mod, model_name, params, show_netron, opset=9)
def nonzero_test(show_netron = False):
    model_name = 'nonzero'
    print('{}...'.format(model_name))
    a = relay.var('a_weight',shape = (2,2))
    b = relay.var('b_weight',shape = (2,2))
    p0 = relay.greater(a, b)
    p1 = relay.argwhere(p0)
    p2 = relay.transpose(p1, (1,0))
    mod, params =testing.create_workload(p2)
    unit_test(mod, model_name, params, show_netron, opset=9)
if __name__ == "__main__":
    logging.disable(logging.WARNING)
#     lenet5()         
#     mobilenetv2()
#     resnet50()
#     squeezenet()
#     vgg19_bn()
#     alexnet() 
#     googlenet()
#     caffenet()
#     rcnn()
#     densenet121()
#     inception_v1()
#     inception_v2() 
#     shufflenet_v1()
#     shufflenet_v2()
#     zfnet512()
#     efficientnet()
    strided_slice_test()
    one_hot_test()
    full_like_test()
    full_test()
    dense_test()
    matmul_test()
    split_test()
    nonzero_test()
    print('Finish!')