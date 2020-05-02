# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=invalid-name, import-self, len-as-condition, unused-argument, too-many-lines
# pylint: disable=import-outside-toplevel
import logging
import numpy as np
import tvm
from tvm.ir import IRModule

from ... import nd as _nd
from .. import analysis
from .. import expr as _expr
from .. import op as _op
from ..ty import TensorType, TupleType
from ..function import Function
from ..expr import Var, Call, Constant, TupleGetItem, Tuple
from ..op import Op
from ..expr_functor import ExprVisitor
from onnx import numpy_helper, helper, NodeProto
from onnx.mapping import NP_TYPE_TO_TENSOR_TYPE
import onnx
from ..transform import Sequential, PartitionGraphInOrder, PartitionGraphInUnorder, PartitionGraphByExpr
from .common import new_var, infer_type, infer_value_simulated
from torch._jit_internal import ignore
from mxnet.ndarray.numpy._op import nonzero
from matplotlib.pyplot import axes
__all__ = ['to_onnx']

class ONNXNode(object):
    """A simply renamer for ops.

    Parameters
    ----------
    new_name : str
        The new name for the op
    """
    def __init__(self, op_name, counts_out = 1):
        self._op_name = op_name
        self._counts_out = counts_out

    def __call__(self, graph, inputs, attrs={}, args=None):
        outputs = graph.get_name(self._op_name, self._counts_out)
        node = helper.make_node(self._op_name, inputs, outputs, **attrs)
        graph.nodes.append(node)
        return outputs
class AttrCvt(ONNXNode):
    """Common attribute converter. An AttrConverter instance is a callable:
    ```
    attr_converter = AttrConverter(op_name, transforms={'a':'b', 'c':('d', 1)})
    new_op_name, new_attr = attr_converter(attrs)
    ```

    Parameters
    ----------
    op_name : str or callable
        If set as str, returned op name is the str.
        If set as callable, returned op is the str returned by calling:
        `op_name = func(attr)`

    transforms : dict of `new_name, or (new_name, default_value, transform function)`
        If only a new_name is provided, it's like renaming the attribute name.
        If default_value if provided, then the attribute is considered as optional.
        If transform function is provided, the original attribute value is handled
        by transform function.

    excludes : list
        A list of excluded attributes that should `NOT` appear.
        Raise NotImplementedError if occurred.

    disables : list
        A list of attributes that is disabled in relay. Log warnings.

    ignores : list
        A list of attributes that is ignoresd in relay. Debug level logging.

    extras : dict
        A series of additional attributes should be added anyway to the returned
        attribute dict.

    custom_check : callable
        A custom function takes attribute, and return True/False.
        Raise RuntimeError if not bool(True) returned.
    """
    def __init__(self, op_name, transforms=None,
                 excludes=None, disables=None, ignores=None,
                 extras=None, custom_check=None, counts_out = 1):      
        super(AttrCvt, self).__init__(op_name, counts_out)
        self._transforms = transforms if transforms else {}
        self._excludes = excludes if excludes else []
        self._disables = disables if disables else []
        self._ignores = ignores if ignores else []
        self._extras = extras if extras else {}
        self._custom_check = custom_check

    def __call__(self, graph, inputs, attrs = {}, args=None):
        if self._custom_check:
            func, msg = self._custom_check
            if not func(attrs):
                raise RuntimeError("Check failed: {}".format(msg))
        # get new op_name
        if isinstance(self._op_name, str):
            op_name = self._op_name
        else:
            assert callable(self._op_name), "op_name can either be string or callable"
            op_name = self._op_name(attrs)
        
        new_attrs = {}
        for k in attrs.keys():
            if k in self._excludes:
                raise NotImplementedError('Attribute %s in op %s is not' +
                                          ' supported.', k, op_name)
            if k in self._disables:
                logging.warning("Attribute %s is disabled in op %s", k, op_name)
            elif k in self._ignores:
                if k != 'tvm_custom':
                    logging.warning("Attribute %s is ignoresd in op %s", k, op_name)
            elif k in self._transforms:
                new_name, defaults, transform = self._parse_default(self._transforms[k])
                if defaults is None:
                    new_attr = self._required_attr(attrs, k)
                else:
                    new_attr = attrs.setdefault(k, None)
                if new_attr is None:
                    new_attrs[new_name] = defaults
                else:
                    new_attrs[new_name] = transform(new_attr)
            else:
                # copy
                new_attrs[k] = attrs[k]
        # add extras
        new_attrs.update(self._extras)
        return super(AttrCvt, self).__call__(graph, inputs, new_attrs)
    def _parse_default(self, target):
        """Helper function to parse default values."""
        if not isinstance(target, (list, tuple)):
            k, v, t = target, None, lambda x: x
        elif len(target) == 1:
            k, v, t = target[0], None, lambda x: x
        elif len(target) == 2:
            k, v, t = target[0], target[1], lambda x: x
        elif len(target) > 2:
            k, v, t = target[0], target[1], target[2]
        else:
            k = None  # should raise
        if not isinstance(k, str):
            msg = "{} is not a valid target, (name, default) expected.".format(target)
            raise ValueError(msg)
        return k, v, t

    def _parse_bool(self, value):
        """Helper function to parse default boolean values."""
        if isinstance(value, str):
            return value.strip().lower() in ['true', '1', 't', 'y', 'yes']
        return bool(value)

    def _required_attr(self, attr, key):
        """Wrapper for getting required attributes."""
        assert isinstance(attr, dict)
        if key not in attr:
            raise AttributeError("Required attribute {} not found.".format(key))
        return attr[key]
   
class ToOnnxOpConverter(object):
    """ A helper class for holding onnx op converters.
    """
    opset = 0
    @classmethod
    def get_impl(cls, opset):
        """ Get converter matches given opset.

        Parameters
        ----------
        opset: int
            opset from model.

        Returns
        -------
        converter, which should be `_impl_vx`. Number x is the biggest
            number smaller than or equal to opset belongs to all support versions.
        """
        cls.opset = opset
        versions = [
            int(d.replace('_impl_v', '')) for d in dir(cls) if '_impl_v' in d
        ]
        versions = sorted(versions + [opset])
        version = versions[
            max([i for i, v in enumerate(versions) if v == opset]) - 1]
        if hasattr(cls, '_impl_v{}'.format(version)):
            return getattr(cls, '_impl_v{}'.format(version))
        raise NotImplementedError(
            'opset version {} of {} not implemented'.format(
                version, cls.__name__))

class Constant(ToOnnxOpConverter):
    name = 'Constant'
    @classmethod
    def _impl_v1(cls, graph, name, data):
        outputs = graph.get_name(name)
        if type(data) in (list,tuple):
            dtype = type(data[0]).__name__
            shape = np.array(data).shape
        elif type(data) in (int, float):
            dtype = type(data).__name__
            shape = ()
            data = [data]
        elif type(data) is np.ndarray:
            dtype = data.dtype
            shape = data.shape
        else:
            dtype = type(data).__name__
            shape = np.array(data, dtype=dtype).shape
        assert dtype not in (np.float16, np.float, np.double), 'Constant with dtype "{}" is not supported in opset=1.'.format(dtype)
        const_node = onnx.helper.make_node(
            cls.name,
            inputs=[],
            outputs=outputs,
            value = onnx.helper.make_tensor(
                name=outputs[0],
                data_type=NP_TYPE_TO_TENSOR_TYPE[np.dtype(dtype)],
                dims=shape,
                vals=data
            )
        )
        graph.nodes.append(const_node)
        return outputs
    @classmethod
    def _impl_v9(cls, graph, name, data):
        outputs = graph.get_name(name)
        if type(data) in (list,tuple):
            dtype = type(data[0]).__name__
            shape = np.array(data).shape
        elif type(data) in (int, float):
            dtype = type(data).__name__
            shape = ()
            data = [data]
        elif type(data) is np.ndarray:
            dtype = data.dtype
            shape = data.shape
        else:
            dtype = type(data).__name__
            shape = np.array(data, dtype=dtype).shape
        const_node = onnx.helper.make_node(
            cls.name,
            inputs=[],
            outputs=outputs,
            value = onnx.helper.make_tensor(
                name=outputs[0],
                data_type=NP_TYPE_TO_TENSOR_TYPE[np.dtype(dtype)],
                dims=shape,
                vals=data
            )
        )
        graph.nodes.append(const_node)
        return outputs
    @classmethod
    def _impl_v11(cls, graph, name, data):
        outputs = graph.get_name(name)
        if type(data) in (list,tuple):
            dtype = type(data[0]).__name__
            shape = np.array(data).shape
        elif type(data) in (int, float):
            dtype = type(data).__name__
            shape = ()
            data = [data]
        elif type(data) is np.ndarray:
            dtype = data.dtype
            shape = data.shape
        else:
            dtype = type(data).__name__
            shape = np.array(data, dtype=dtype).shape
        const_node = onnx.helper.make_node(
            cls.name,
            inputs=[],
            outputs=outputs,
            value = onnx.helper.make_tensor(
                name=outputs[0],
                data_type=NP_TYPE_TO_TENSOR_TYPE[np.dtype(dtype)],
                dims=shape,
                vals=data
            )
        )
        graph.nodes.append(const_node)
        return outputs
    @classmethod
    def _impl_v12(cls, graph, name, data):
        outputs = graph.get_name(name)
        if type(data) in [list,tuple] and len(data)>0:
            dtype = type(data[0]).__name__
            value = {'value_{}s'.format(dtype): data}
        else:
            dtype = type(data).__name__
            value = {'value_{}'.format(dtype): data}
        const_node = onnx.helper.make_node(
            cls.name,
            inputs=[],
            outputs=outputs,
            **value
        )
        graph.nodes.append(const_node)
        return outputs

class BatchNorm(ToOnnxOpConverter):
    """ Operator converter for nn.batch_norm op."""
    name = 'BatchNormalization'
    @classmethod
    def convert(cls, graph, inputs, attrs):
        make_batch_norm = AttrCvt(cls.name, ignores=['axis', 'center', 'scale'])
        if attrs['axis'] == 3:
            make_trans = ONNXNode('Transpose')
            in_trans = make_trans(graph, [inputs[0]],{'perm': [0, 3, 1, 2]}),
            batch_norm = make_batch_norm(graph, in_trans + inputs[1:]), 
            return make_trans(graph, batch_norm, {'perm': [0, 2, 3, 1]})
        else: 
            return make_batch_norm(graph, inputs, attrs)
    @classmethod
    def _impl_v1(cls, graph, inputs, attrs={}, args=None):
        consumed_inputs = [0]*5
        if attrs['scale'] == False:
            consumed_inputs[1]=1
        if attrs['center'] == False:
            consumed_inputs[2]=1
        return cls.convert(graph, inputs, attrs)
    @classmethod
    def _impl_v7(cls, graph, inputs, attrs={}, args=None):
        assert attrs.setdefault('scale', 1) == True and attrs.setdefault('center', 1) == True, 'Scale(gamma) and bias(beta) in opset 12 of BatchNorm must be reserved.'
        return cls.convert(graph, inputs, attrs)

    @classmethod
    def _impl_v12(cls, graph, inputs, attrs={}, args=None):
        assert attrs.setdefault('scale', 1) == True and attrs.setdefault('center', 1) == True, 'Scale(gamma) and bias(beta) in opset 12 of BatchNorm must be reserved.'
        return cls.convert(graph, inputs, attrs)
class BatchFlatten(ToOnnxOpConverter):
    """ Operator converter for nn.batch_flatten op."""
    name = 'Flatten'
    @classmethod
    def _impl_v1(cls, graph, inputs, attrs={}, args=None):
        return AttrCvt(cls.name, extras={'axis': 1})(graph, inputs, attrs)
    @classmethod
    def _impl_v11(cls, graph, inputs, attrs={}, args=None):
        return AttrCvt(cls.name, extras={'axis': 1})(graph, inputs, attrs)
    
class MaxPool(ToOnnxOpConverter):
    """ Operator converter for nn.max_poolid op."""
    name = 'MaxPool'
    @classmethod
    def _impl_v1(cls, graph, inputs, attrs={}, args=None):
        assert attrs.setdefault('ceil_mode', 0) == False, 'Value of attr "ceil_mode" must be False.'
        transforms ={
            'dilation':'dilations',
            'pool_size': 'kernel_shape',
            'padding':'pads'
        }
        ignores=['layout','ceil_mode']
        return AttrCvt(cls.name, transforms = transforms ,ignores = ignores)(graph, inputs, attrs)
    @classmethod
    def _impl_v11(cls, graph, inputs, attrs={}, args=None):
        transforms ={
            'dilation':'dilations',
            'pool_size': 'kernel_shape',
            'padding':'pads'
        }
        ignores=['layout']
        return AttrCvt(cls.name, transforms = transforms ,ignores = ignores)(graph, inputs, attrs)
    @classmethod
    def _impl_v12(cls, graph, inputs, attrs={}, args=None):
        transforms ={
            'dilation':'dilations',
            'pool_size': 'kernel_shape',
            'padding':'pads'
        }
        ignores=['layout']
        return AttrCvt(cls.name, transforms = transforms ,ignores = ignores)(graph, inputs, attrs)
class MaxPool1D(MaxPool):
    """ Operator converter for nn.max_pool1d op."""
class MaxPool2D(MaxPool):
    """ Operator converter for nn.max_pool2d op."""
class MaxPool3D(MaxPool):
    """ Operator converter for nn.max_pool3d op."""
class ConvTranspose(ToOnnxOpConverter):
    """ Operator converter for nn.convid_transpose op."""
    name = 'ConvTranspose'
    @classmethod
    def _impl_v1(cls, graph, inputs, attrs={}, args=None):
        transforms ={
            'dilation':'dilations',
            'groups':'group',
            'kernel_size': 'kernel_shape',
            'padding':'pads'
        }
        ignores=['channels', 'data_layout','kernel_layout','out_layout','out_dtype']
        extras={'output_shape': c.checked_type.concrete_shape()}
        make_trans = AttrCvt(cls.name, transforms = transforms ,ignores = ignores, extras=extras)
        return make_trans(graph, inputs, attrs)
    @classmethod
    def _impl_v11(cls, graph, inputs, attrs={}, args=None):
        transforms ={
            'dilation':'dilations',
            'groups':'group',
            'kernel_size': 'kernel_shape',
            'padding':'pads'
        }
        ignores=['channels', 'data_layout','kernel_layout','out_layout','out_dtype']
        extras={'output_shape': c.checked_type.concrete_shape()}
        make_trans = AttrCvt(cls.name, transforms = transforms ,ignores = ignores, extras=extras)
        return make_trans(graph, inputs, attrs)
class Conv1DTranspose(ConvTranspose):
    """ Operator converter for nn.conv1d_transpose op."""
class Conv2DTranspose(ConvTranspose):
    """ Operator converter for nn.conv2d_transpose op."""
class Conv(ToOnnxOpConverter):
    """ Operator converter for nn.convid op."""
    name = 'Conv'
    @classmethod
    def _impl_v1(cls, graph, inputs, attrs={}, args=None):
        transforms ={
            'dilation':'dilations',
            'groups':'group',
            'kernel_size': 'kernel_shape',
            'padding':'pads'
        }
        ignores=['channels','data_layout','kernel_layout','out_layout','out_dtype']
        return AttrCvt(cls.name, transforms = transforms ,ignores = ignores)(graph, inputs, attrs)
    @classmethod
    def _impl_v11(cls, graph, inputs, attrs={}, args=None):
        transforms ={
            'dilation':'dilations',
            'groups':'group',
            'kernel_size': 'kernel_shape',
            'padding':'pads'
        }
        ignores=['channels','data_layout','kernel_layout','out_layout','out_dtype']
        return AttrCvt(cls.name, transforms = transforms ,ignores = ignores)(graph, inputs, attrs)
class Conv1D(Conv):
    """ Operator converter for nn.conv1d op."""
class Conv2D(Conv):
    """ Operator converter for nn.conv2d op."""
class Conv3D(Conv):
    """ Operator converter for nn.conv3d op."""
class Dropout(ToOnnxOpConverter):
    """ Operator converter for nn.dropout op."""
    name='Dropout'
    @classmethod
    def _impl_v1(cls, graph, inputs, attrs={}, args=None):
        transforms ={
            'consumed_inputs':[0],
            'rate':'ratio',
            'is_test':1
        }
        return AttrCvt('Dropout',transforms = transforms)(graph, inputs, attrs)
    @classmethod
    def _impl_v10(cls, graph, inputs, attrs={}, args=None):
        transforms ={
            'rate':'ratio'
        }
        return AttrCvt('Dropout',transforms = transforms)(graph, inputs, attrs)
    @classmethod
    def _impl_v12(cls, graph, inputs, attrs={}, args=None):
        rate_data = attrs['rate']
        make_const = Constant.get_impl(cls.opset)
        rate = make_const(graph, 'rate', rate_data)
        return AttrCvt(cls.name, ignores=['rate'])(graph, inputs + rate, attrs)
class Reshape(ToOnnxOpConverter):
    """ Operator converter for reshape op."""
    name = 'Reshape'
    @classmethod
    def _impl_v1(cls, graph, inputs, attrs={}, args=None):
        transforms ={
            'newshape':'shape'
        }
        ignores=['reverse']
        return AttrCvt(cls.name, ignores=ignores, transforms=transforms)(graph, inputs, attrs)
    @classmethod
    def _impl_v5(cls, graph, inputs, attrs={}, args=None): 
        shape_data = attrs['newshape']
        make_const = Constant.get_impl(cls.opset)
        shape = make_const(graph, 'shape', shape_data)
        return AttrCvt(cls.name, ignores=['newshape', 'reverse'])(graph, inputs + shape, attrs)
class AvgPool(ToOnnxOpConverter):
    """ Operator converter for nn.avg_poolid op."""
    name='AveragePool'
    @classmethod
    def _impl_v1(cls, graph, inputs, attrs={}, args=None):
        assert attrs.setdefault('ceil_mode', 0) == False, 'Value of attr "ceil_mode" must be False.'
        assert attrs.setdefault('count_include_pad', 0) == False, 'Value of attr "count_include_pad" must be False.'
        transforms ={
            'pool_size': 'kernel_shape',
            'padding':'pads'
        }
        ignores=['layout', 'ceil_mode', 'count_include_pad']
        return AttrCvt(cls.name, transforms = transforms ,ignores = ignores)(graph, inputs, attrs)
    @classmethod
    def _impl_v11(cls, graph, inputs, attrs={}, args=None):
        transforms ={
            'pool_size': 'kernel_shape',
            'padding':'pads'
        }
        ignores=['layout']
        return AttrCvt(cls.name, transforms = transforms ,ignores = ignores)(graph, inputs, attrs)
class AvgPool1D(AvgPool):
    """ Operator converter for nn.avg_pool1d op."""
class AvgPool2D(AvgPool):
    """ Operator converter for nn.avg_pool2d op."""
class AvgPool3D(AvgPool):
    """ Operator converter for nn.avg_pool3d op."""
class Pad(ToOnnxOpConverter):
    """ Operator converter for nn.pad op."""
    name = 'Pad'
    @classmethod
    def _convert_pads(cls, pads):
        l = len(pads)
        new_pads = [None]*(2*l)
        for i, pad in enumerate(pads):
            new_pads[i], new_pads[i+l] = pad[0], pad[1]
        return new_pads
    @classmethod
    def _impl_v1(cls, graph, inputs, attrs={}, args=None):
        transforms ={
            'pad_value': 'value',
            'pad_mode': 'mode',
            'pad_width': ('paddings', None, cls._convert_pads)
        }
        return AttrCvt(cls.name, transforms = transforms)(graph, inputs, attrs)

    @classmethod
    def _impl_v2(cls, graph, inputs, attrs={}, args=None):
        transforms ={
            'pad_value': 'value',
            'pad_mode': 'mode',
            'pad_width': ('pads', None, cls._convert_pads)
        }
        return AttrCvt(cls.name, transforms = transforms)(graph, inputs, attrs)
    @classmethod
    def _impl_v11(cls, graph, inputs, attrs={}, args=None):
        assert attrs.setdefault('pad_mode', 'constant') is 'constant', 'For opset=11, "pad_mode" must be "constant".'
        make_const = Constant.get_impl(cls.opset)
        pads_data = cls._convert_pads(attrs['pad_width'])
        constant_value_data = attrs['pad_value']
        pads = make_const(graph, 'pads', pads_data)
        value = make_const(graph, 'constant_value', constant_value_data)
        ignores = ['pad_mode', 'pad_width', 'pad_value']
        return AttrCvt(cls.name, ignores=ignores)(graph, inputs+pads+value, attrs)

class BroadcastTo(ToOnnxOpConverter):
    """ Operator converter for broadcast_to op."""
    name = 'Expand'
    @classmethod
    def _impl_v8(cls, graph, inputs, attrs={}, args=None):
        shape_data = attrs['shape']
        ignores = ['shape', 'dtype']
        make_const = Constant.get_impl(cls.opset)
        shape = make_const(graph, 'shape', shape_data)
        return AttrCvt(cls.name, ignores=ignores)(graph, inputs+shape, attrs)
class Split(ToOnnxOpConverter):
    """ Operator converter for split op."""
    name = 'Split'
    @classmethod
    def convert_split(cls, indices, size_axis):
        if isinstance(indices, int):
            split =[size_axis//indices]*(indices)
        else:
            split = [None]*(len(indices)+1)
            split[0] = indices[0] - 0
            for i in range(len(indices)-1):
                split[i+1] = indices[i+1] - indices[i]
            split[-1] = size_axis - indices[-1]
        return split
    @classmethod
    def _impl_v1(cls, graph, inputs, attrs={}, args=None):
        size_axis = expr.args[0].checked_type.concrete_shape[attrs['axis']]
        split = cls.convert_split(attrs['indices_or_sections'], size_axis)
        return AttrCvt(cls.name, extras={'split': split}, ignores=['indices_or_sections'], counts_out=len(split))(graph, inputs, attrs)
    @classmethod
    def _impl_v2(cls, graph, inputs, attrs={}, args=None):
        size_axis = args[0].checked_type.concrete_shape[attrs['axis']]
        split = cls.convert_split(attrs['indices_or_sections'], size_axis)
        return AttrCvt(cls.name, extras={'split': split}, ignores=['indices_or_sections'], counts_out=len(split))(graph, inputs, attrs)
class Tile(ToOnnxOpConverter):
    """ Operator converter for tile op."""
    name = 'Tile'
    @classmethod
    def _impl_v1(cls, graph, inputs, attrs={}, args=None):
        return AttrCvt(cls.name, {'reps':'repeats'})(graph, inputs, attrs)
    @classmethod
    def _impl_v6(cls, graph, inputs, attrs={}, args=None):
        repeats_data = attrs['reps']
        make_const = Constant.get_impl(cls.opset)
        repeats = make_const(graph, 'repeats', repeats_data)
        return AttrCvt(cls.name, ignores = ['reps'])(graph, inputs+repeats, attrs)
class Resize(ToOnnxOpConverter):
    """Operator converter for image.resize op."""
    name = 'Resize'
    @classmethod
    def _convert_mode(cls, method):
        if method== 'nearest_neighbor':
            return 'nearest'
        elif method == 'bilinear':
            return 'linear'
        else:
            raise tvm.error.OpAttributeInvalid(
                'Value {} in attribute "method" of op image.resize is not valid.'.format(method))
    @classmethod
    def _impl_v11(cls, graph, inputs, attrs={}, args=None):
        roi_data = np.array([],dtype='int8')
        scales_data = np.array([],dtype='float32')
        sizes_data = attrs['sizes']
        make_const = Constant.get_impl(cls.opset)
        roi = make_const(graph, 'roi', roi_data)
        scales = make_const(graph, 'scales', scales_data)
        size = make_const(graph, 'sizes', sizes_data)
        convert = AttrCvt(cls.name,
                transforms = {'method':('mode', None, cls._convert_mode)},
                ignores = ['size', 'layout', 'out_dtype'],
                extras ={'sizes': expr.checked_type.concrete_shape})
        return convert(graph, inputs+roi+scales+size, attrs)
        
class Reduce(ToOnnxOpConverter):
    """ Operator converter for reduce op."""
    name = ''
    @classmethod
    def _impl_v1(cls, graph, inputs, attrs={}, args=None):
        assert attrs.setdefault('exclude', 0) == False, 'Value "exclude" can not be True.'
        return AttrCvt(cls.name,transforms={'axis':'axes'}, ignores=['exclude'])(graph, inputs, attrs)

class Max(Reduce):
    """ Operator converter for max op."""
    name = 'ReduceMax'
class Min(Reduce):
    """ Operator converter for min op."""
    name = 'ReduceMin'
class Sum(Reduce):
    """ Operator converter for sum op."""
    name = 'ReduceSum'
class Mean(Reduce):
    """ Operator converter for sum op."""
    name = 'ReduceMean'
class Prod(Reduce):
    """ Operator converter for prod op."""
    name = 'ReduceProd'
class Upsampling(ToOnnxOpConverter):
    """ Operator converter for upsample (nearest mode) op."""
    name = 'Upsample'
    @classmethod
    def _convert_mode(cls, method):
        if method == 'nearest_neighbor':
            return 'nearest'
        elif method == 'bilinear':
            return 'linear'
        else:
            raise tvm.error.OpAttributeInvalid(
                'Value {} in attribute "method" of op Upsample is not valid.'.format(method))
    @classmethod
    def _impl_v1(cls, graph, inputs, attrs={}, args=None):
        assert attrs.setdefault('align_corners', 1) == True, 'Value "align_corners" must be True.'
        transforms={'method':('mode',None,cls._convert_mode),
                    'scale_h':'height_scale',
                    'scale_w':'weight_scale'}
        ignores=['layout', 'align_corners']
        return AttrCvt(cls.name,
                    transforms=transforms,
                    ignores=ignores
                    )(graph, inputs, attrs)
    @classmethod
    def _impl_v9(cls, graph, inputs, attrs={}, args=None):
        assert attrs.setdefault('align_corners', 1) == True, 'Value "align_corners" must be True.'
        scales_data = [1.0, 1.0, attrs['scale_h'],attrs['scale_w']]
        make_const = Constant.get_impl(cls.opset)
        scales = make_const(graph, 'scales', scales_data)
        return AttrCvt(cls.name,
            transforms={'method':('mode',None,cls._convert_mode)},
            ignores=['scale_h', 'scale_w', 'layout', 'align_corners']
            )(graph, inputs+scales, attrs)
class StridedSlice(ToOnnxOpConverter):
    """ Operator converter for strided_slice op."""
    name = 'Slice'
    @classmethod
    def _impl_v1(cls, graph, inputs, attrs={}, args=None):
        assert not attrs.setdefault('strides',[]), 'Value of "strides" must be empty.'
        return AttrCvt(cls.name,
                       transforms={'begin': 'starts',
                                   'end': 'ends'},
                       extras={'axes':[i for i in range(0, len(attrs['begin']))]},
                       ignores=['strides'])(graph, inputs, attrs) 
    @classmethod
    def _impl_v10(cls, graph, inputs, attrs={}, args=None):
        make_const = Constant.get_impl(cls.opset)
        starts_data = attrs['begin']
        ends_data = attrs['end']
        axes_data = [i for i in range(0, len(attrs['begin']))]
        steps_data = attrs['strides']
        starts = make_const(graph, 'starts', starts_data)
        ends = make_const(graph, 'ends', ends_data)
        axes = make_const(graph, 'axes', axes_data)
        steps = make_const(graph, 'steps', steps_data)
        inputs = inputs + starts + ends + axes + steps
        return AttrCvt(cls.name,ignores=['begin', 'end', 'strides'])(graph, inputs, attrs)
    @classmethod
    def _impl_v11(cls, graph, inputs, attrs={}, args=None):
        return cls._impl_v10(graph, inputs, attrs, args)
class SubPixel(ToOnnxOpConverter):
    """ Operator converter for sub pixel ops."""
    name = ''
    @classmethod
    def _impl_v1(cls, graph, inputs, attrs={}, args=None):
        assert attrs.setdefault('mode', 'DCR') is 'DCR', 'Value of attr "mode" must be "DCR"'
        return AttrCvt('DepthToSpace', ignores=['layout','mode'])(graph, inputs, attrs)
    @classmethod
    def _impl_v11(cls, graph, inputs, attrs={}, args=None):
        return AttrCvt('DepthToSpace', ignores=['layout'])(graph, inputs, attrs)
class DepthToSpace(SubPixel):
    """ Operator converter for nn.depth_to_space op."""
    name = 'DepthToSpace'
class SpaceToDepth(SubPixel):
    """ Operator converter for nn.space_to_depth op."""
    name = 'SpaceToDepth'
class OneHot(ToOnnxOpConverter):
    """ Operator converter for one_hot op."""
    name = 'OneHot'
    @classmethod
    def get_new_attrs(cls, func_attrs):
        for call, attrs in func_attrs.items():
            if call.op.name == 'one_hot':
                return attrs
    @classmethod
    def _impl_v9(cls, graph, inputs, attrs={}, args=None):
        attrs = cls.get_new_attrs(attrs)
        depth_data = attrs.setdefault('depth', 1)
        make_const = Constant.get_impl(cls.opset)
        depth = make_const(graph, 'depth', depth_data)
        inputs.insert(1,depth[0])
        return AttrCvt(cls.name, ignores=['dtype','depth'])(graph, inputs, attrs)
    @classmethod
    def _impl_v11(cls, graph, inputs, attrs={}, args=None):
        attrs = cls.get_new_attrs(attrs)
        depth_data = attrs.setdefault('depth', 1)
        depth = Constant.get_impl(cls.opset)(graph, 'depth', depth_data)
        inputs.insert(1,depth[0])
        return AttrCvt(cls.name, ignores=['dtype','depth'])(graph, inputs, attrs)
class Broadcast(ToOnnxOpConverter):
    """ Operator converter for binary op whose pattern is broadcast."""
    name = ''
    @classmethod
    def _impl_v1(cls, graph, inputs, attrs={}, args=None):
        extras = {'axis':0,'broadcast':1, 'consumed_inputs':[0,0]}
        return AttrCvt(cls.name, extras = extras)(graph, inputs, attrs)
    @classmethod
    def _impl_v6(cls, graph, inputs, attrs={}, args=None):
        extras = {'axis':0,'broadcast':1}
        return AttrCvt(cls.name, extras = extras)(graph, inputs, attrs)
    @classmethod
    def _impl_v7(cls, graph, inputs, attrs={}, args=None):
        return AttrCvt(cls.name)(graph, inputs, attrs)
class Add(Broadcast):
    """ Operator converter for add op."""
    name = 'Add'
class Sub(Broadcast):
    """ Operator converter for sub op."""
    name = 'Sub'
class Multiply(Broadcast):
    """ Operator converter for multiply op."""
    name = 'Mul'
class Divide(Broadcast):
    """ Operator converter for divide op."""
    name = 'Div'
class FullLike(ToOnnxOpConverter):
    """ Operator converter for full like op."""
    @classmethod
    def _impl_v9(cls, graph, inputs, attrs={}, args=None):
        # full -> ConstantOfShape
        # full_like -> Shape + ConstantOfShape
        # Convert fill_value(node) to value(TensorProto)
        value = infer_value_simulated(args[1], graph.params)
        value = helper.make_tensor(
            name='value',
            data_type=NP_TYPE_TO_TENSOR_TYPE[np.dtype(value.dtype)],
            dims=(1,),
            vals=[value.asnumpy().item()]
        )
        # Create a Shape node
        data = inputs[0]
        shape = ONNXNode('Shape')(graph, [data])
        # Drop out fill_value argument
        return AttrCvt('ConstantOfShape', ignores=['shape','dtype'], extras={'value':value})(graph, shape, attrs)

class Full(ToOnnxOpConverter):
    """ Operator converter for full op."""
    name = 'ConstantOfShape'
    @classmethod
    def _impl_v9(cls, graph, inputs, attrs={}, args=None):
        # full -> ConstantOfShape
        value = infer_value_simulated(args[0], graph.params)
        value = helper.make_tensor(
                name='value',
                data_type=NP_TYPE_TO_TENSOR_TYPE[np.dtype(value.dtype)],
                dims=(1,),
                vals=[value.asnumpy().item()]
            )
        new_args = [] # Drop out value argument
        make_const = Constant.get_impl(cls.opset)
        shape = make_const(graph, 'input', attrs['shape'])
        return AttrCvt(cls.name, ignores=['shape','dtype'], extras={'value':value})(graph, shape, attrs)
class MatMul(ToOnnxOpConverter):
    """ Operator converter for fused_matmul op."""
    name = 'MatMul'
    @classmethod
    def _impl_v1(cls, graph, inputs, attrs={}, args=None):
        attrs.clear()
        return AttrCvt(cls.name)(graph, inputs, attrs)
class Gemm(ToOnnxOpConverter):
    """ Operator converter for fused_gemm op."""
    name = 'Gemm'
    @classmethod
    def get_new_attrs(cls, func_attrs):
        new_attrs={'alpha': 1.0, 'beta':1.0, 'transA':0, 'transB':1}
        for call, attr in func_attrs.items():
            if call.op.name == 'nn.dense':
                if isinstance(call.args[0], Call) and call.args[0].op.name == 'multiply':
                    new_attrs['alpha'] = np.asscalar(call.args[0].args[0].data.asnumpy())
                    if isinstance(call.args[0].args[1], Call) and  call.args[0].args[1].op.name is 'transpose':
                        new_attrs['transB'] = 0
            elif call.op.name == 'nn.batch_flatten':
                if isinstance(call.args[0], Call) and call.args[0].op.name == 'transpose':
                    new_attrs['transA'] = 1
            elif call.op.name == 'nn.bias_add':
                if isinstance(call.args[1], Call) and  call.args[1].op.name == 'multiply':
                    new_attrs['beta'] = np.asscalar(call.args[1].args[0].data.asnumpy())
        return new_attrs
    @classmethod
    def _impl_v1(cls, graph, inputs, attrs={}, args=None):
        assert len(inputs) == 3, 'Intputs of Gemm with opset <= 7 must has bias.'
        attrs = cls.get_new_attrs(attrs)
        return AttrCvt(cls.name, extras={'broadcast': 1})(graph, inputs, attrs)
    @classmethod
    def _impl_v7(cls, graph, inputs, attrs={}, args=None):
        assert len(inputs) == 3, 'Intputs of Gemm with opset <= 7 must has bias.'
        attrs = cls.get_new_attrs(attrs)
        return AttrCvt(cls.name)(graph, inputs, attrs)
    @classmethod
    def _impl_v11(cls, graph, inputs, attrs={}, args=None):
        attrs = cls.get_new_attrs(attrs)
        return AttrCvt(cls.name)(graph, inputs, attrs)
class Dense(ToOnnxOpConverter):
    """ Operator converter for nn.dense op."""
    @classmethod
    def _impl_v1(cls, graph, inputs, attrs={}, args=None):
        a, b = [inputs[0]], [inputs[1]]
        b_t = ONNXNode('Transpose')(graph, b, attrs={'perm':(1,0)})
        return AttrCvt('MatMul', ignores=['units', 'out_dtype'])(graph, a + b_t, attrs)

class BiasAdd(ToOnnxOpConverter):
    """ Operator converter for nn.bias_add op."""
    @classmethod
    def _impl_v1(cls, graph, inputs, attrs={}, args=None):
        ndim = len(args[0].checked_type.shape)
        axis = attrs.setdefault('axis', 1)
        if axis < 0:
            axis = axis + ndim
        new_axes = ndim - axis - 1
        nodes = []
        in_add = inputs
        if new_axes:
            unsqueeze = Unsqueeze.get_impl(cls.opset)(graph, [inputs[1]], {'axes':tuple(range(1, new_axes + 1))})
            in_add = [inputs[0], unsqueeze[0]]
        return Add.get_impl(cls.opset)(graph, in_add)
class BatchMatMul(ToOnnxOpConverter):
    """ Operator converter for nn.batch_matmul op."""
    @classmethod
    def _impl_v1(cls, graph, inputs, attrs={}, args=None):
        a, b = input[0], input[1]
        b_t = '{}_transpose'.format(inputs[1])
        b_t = ONNXNode('Transpose')([b], [b_t], attrs={'perm':(0,2,1)})
        return AttrCvt('MatMul', ignores=['units', 'out_dtype'])([a,b_t], attrs)
class Unsqueeze(ToOnnxOpConverter):
    """ Operator converter for fused_unsqueeze op."""
    @classmethod
    def get_new_attrs(cls, func_attrs):
        if not func_attrs or isinstance([*func_attrs][0], str):
            return func_attrs
        axes = []
        for call, attrs in func_attrs.items():
            axes.append(attrs['axis'])
        axes.reverse()
        return {'axes':axes}
    @classmethod
    def _impl_v1(cls, graph, inputs, attrs={}, args=None):
        attrs = cls.get_new_attrs(attrs)
        return ONNXNode('Unsqueeze')(graph, inputs , attrs)
class Argwhere(ToOnnxOpConverter):
    """ Operator converter for argwhere op."""
    @classmethod
    def _impl_v1(cls, graph, inputs, attrs={}, args=None):
        non = ONNXNode('NonZero')(graph, inputs)
        return ONNXNode('Transpose')(graph, non, {'perm':(1,0)})
class NonZero(ToOnnxOpConverter):
    """ Operator converter for fused_nonzero op."""
    @classmethod
    def _impl_v1(cls, graph, inputs, attrs={}, args=None):
        return ONNXNode('NonZero')(graph, inputs)
# compatible ops that do NOT require any conversion.
_identity_list = []


# _convert_map defines maps of name to converter functor(callable)
# for 1 to 1 mapping, use Renamer if nothing but name is different
# use AttrCvt if attributes must be converted
# for 1 to N mapping(composed), use custom callable functions
# for N to 1 mapping, currently not supported(?)
def _get_impl_map(opset):
    return {
        # defs/experimental
        'copy': ONNXNode('Identity'),
        #'Affine': has been simplified.
        #'ThresholdedRelu': has been simplified.
        'full_like': FullLike.get_impl(opset),
        #'ScaledTanh': has been simplified.
        #'ParametricSoftplus': has been simplified.
        'full': Full.get_impl(opset),
        #'GivenTensorFill'
        'nn.dense': Dense.get_impl(opset), # FC has been removed on ONNX
        #'Scale': has been simplified.
        # 'GRUUnit'
        # 'ATen'
        # 'ImageScaler'
        # 'MeanVarianceNormalization'
        # 'Crop'
        # 'Embedding'
        'nn.upsampling': Upsampling.get_impl(opset),
        #'SpatialBN': has been simplified.

        # defs/generator
        'Constant': Constant.get_impl(opset), # Implemented
        # 'RandomUniform'
        # 'RandomNormal'
        # 'RandomUniformLike'
        # 'RandomNormalLike'

        # defs/logical

        # defs/math
        'add': Add.get_impl(opset),
        'sub':Sub.get_impl(opset),
        'multiply': Multiply.get_impl(opset),
        'divide': Divide.get_impl(opset),
        'negative': ONNXNode('Neg'),
        'abs': ONNXNode('Abs'),
        #'Reciprocal': has been simplified.
        'floor': ONNXNode('Floor'),
        'ceil': ONNXNode('Ceil'),
        'sqrt': ONNXNode('Sqrt'),
        'nn.relu': ONNXNode('Relu'),
        'nn.leaky_relu': ONNXNode('LeakyRelu'),
        #'Selu': has been simplified.
        #'Elu': has been simplified.
        'exp': ONNXNode('Exp'),
        'greater': ONNXNode('Greater'),
        'less': ONNXNode('Less'),
        'log': ONNXNode('Log'),
        'tanh': ONNXNode('Tanh'),
        'power': ONNXNode('Pow'),
        'nn.prelu': ONNXNode('PRelu'),
        'sigmoid': ONNXNode('Sigmoid'),
        #'HardSigmoid': has been simplified.
        'maximum': ONNXNode('Max'),
        'minimum':  ONNXNode('Min'),
        'clip': AttrCvt('Clip', {'a_min': 'min', 'a_max': 'max'}),
        # softmax default axis is different in onnx
        'nn.softmax': ONNXNode('Softmax'),
        'nn.log_softmax': ONNXNode('LogSoftmax'),
        'fused_onehot': OneHot.get_impl(opset),
        # 'Hardmax'
        #'Softsign': has been simplified.
        #'SoftPlus': has been simplified.
        'fused_gemm': Gemm.get_impl(opset),
        'fused_matmul': MatMul.get_impl(opset),
        'nn.batch_matmul':BatchMatMul.get_impl(opset),
        # defs/nn
        'nn.bias_add': BiasAdd.get_impl(opset),
        'nn.avg_pool1d':AvgPool1D.get_impl(opset),
        'nn.avg_pool2d':AvgPool2D.get_impl(opset),
        'nn.avg_pool3d':AvgPool3D.get_impl(opset),
        'nn.max_pool1d': MaxPool1D.get_impl(opset),
        'nn.max_pool2d': MaxPool2D.get_impl(opset),
        'nn.max_pool3d': MaxPool3D.get_impl(opset),
        'nn.conv1d': Conv1D.get_impl(opset),
        'nn.conv2d': Conv2D.get_impl(opset),
        'nn.conv3d': Conv3D.get_impl(opset),
        'nn.conv1d_transpose': Conv1DTranspose.get_impl(opset),
        'nn.conv2d_transpose': Conv2DTranspose.get_impl(opset),
        'nn.global_avg_pool2d': AttrCvt(op_name = 'GlobalAveragePool', ignores=['layout']),
        'nn.global_max_pool2d': ONNXNode('GlobalMaxPool'),
        'nn.batch_norm': BatchNorm.get_impl(opset),
        'nn.instance_norm': ONNXNode('InstanceNormalization'), # _impl_v1
        # 'LpNormalization'
        'nn.dropout': Dropout.get_impl(opset),
        'nn.batch_flatten': BatchFlatten.get_impl(opset),
        'nn.lrn': AttrCvt(op_name = 'LRN', ignores=['axis']),
        # Recurrent Layers
        #'LSTM': has been simplified.

        # defs/reduction
        'max': Max.get_impl(opset),
        'min': Min.get_impl(opset),
        'sum': Sum.get_impl(opset),
        'mean': Mean.get_impl(opset),
        'prod': Prod.get_impl(opset),
        # 'ReduceLogSumExp'
        'argmax':ONNXNode('ArgMax'),
        'argmin':ONNXNode('ArgMin'),

        # defs/tensor
        'cast': AttrCvt('Cast', {'dtype': 'to'}), # _impl_v1
        'reshape': Reshape.get_impl(opset),
        'broadcast_to': BroadcastTo.get_impl(opset),  # Expand
        'concatenate': ONNXNode('Concat'),
        'split': Split.get_impl(opset),
        'strided_slice': StridedSlice.get_impl(opset),
        'transpose': AttrCvt('Transpose', {'axes': 'perm'}),
        'nn.depth_to_space': DepthToSpace.get_impl(opset),
        'nn.space_to_depth': SpaceToDepth.get_impl(opset),
        'take': ONNXNode('Gather'), # Gather
        'squeeze': AttrCvt('Squeeze', {'axis':'axes'}),
        'fused_unsqueeze': Unsqueeze.get_impl(opset),
        'nn.pad': Pad.get_impl(opset),
        'shape_of': AttrCvt('Shape', ignores=['dtype']),
        'sign': ONNXNode('Sign'),
        'equal': ONNXNode('Equal'),
        'logical_not':ONNXNode('Not'),
        'logical_and':ONNXNode('And'),
        'tile': Tile.get_impl(opset),
        'erf':ONNXNode('Erf'),
        'where': ONNXNode('Where'),
        'logical_or':ONNXNode('Or'),
        'image.resize': Resize.get_impl(opset),
        'argwhere': Argwhere.get_impl(opset),
        'fused_nonzero':NonZero.get_impl(opset)
    }
def tvm_array_to_list(object):
    """ Convert tvm.ir.container.Array to List."""
    if isinstance(object, tvm.ir.container.Array) or isinstance(object, list):
        new_list=[]
        for value in object:
            new_list.append(tvm_array_to_list(value))
        return new_list
    elif isinstance(object, tvm.ir.PrimExpr):
        return object.value
    else:
        return object
def convert_attrs_to_dict(attrs):
    """Convert attrs of op to ONNX format and delete the input params"""
    def convert_object_for_py(object):
        """ Convert tvm object for Python."""
        if isinstance(object, tvm.ir.attrs.Attrs):
            new_map={}
            for key in object.keys():  
                new_value = tvm_array_to_list(object[key])
                new_map[key] = new_value
            return new_map
        elif isinstance(object, tvm.ir.container.Array):
            return tvm_array_to_list(object)
        else:
            return object
    new_attrs = convert_object_for_py(attrs)
    return new_attrs if new_attrs else {}
def make_tensor_info(name, tensortype):
    shape = tensortype.concrete_shape
    shape = tuple('?' if isinstance(x, tvm.tir.Any) else x for x in shape)
    elem_type = NP_TYPE_TO_TENSOR_TYPE[np.dtype(tensortype.dtype)]
    return helper.make_tensor_value_info(name, elem_type, shape)
class ONNXGraph():
    """ A container class to hold different attributes of ONNX model graph"""
    def __init__(self, params, opset, graph_name, doc_string):
        self.params = params
        self.opset = opset
        self.graph_name = graph_name
        self.doc_string = doc_string
        self.id = 0
        self.inputs = []
        self.nodes = []
        self.initializer = []
        self.outputs = []
    def get_name(self, name, counts_out = 1):
        if counts_out == 1:
            outputs = ['%d_%s'%(self.id, name)]
        else:
            outputs = ['']*counts_out
            for i in range(counts_out):
                outputs[i] = '%d_%d_%s'%(self.id, i, name)
        self.id += 1
        return outputs
    def add_const(self, const):
        outputs = self.get_name('const')
        data = const.data.asnumpy()
        node = onnx.helper.make_node(
            'Constant',
            inputs=[],
            outputs=outputs,
            value=onnx.helper.make_tensor(
                name=outputs[0],
                data_type=NP_TYPE_TO_TENSOR_TYPE[data.dtype],
                dims=data.shape,
                vals=data.flatten().astype(data.dtype),
                ),
        )
        self.nodes.append(node)
        return outputs
    def add_nodes(self, c, op_name, inputs, attrs):
        outputs = self.convert_op(op_name, inputs, attrs, c.args)
        return outputs
    def add_var(self, var):
        name = var.name_hint
        if name in self.params:
            node = numpy_helper.from_array(self.params[name].asnumpy(), name)
            self.initializer.append(node)
        else:
            info = make_tensor_info(name, var.checked_type)
            self.inputs.append(info)
    def add_outputs(self, name, tensortype):
        node = make_tensor_info(name, tensortype)
        self.outputs.append(node)
    def make_model(self):
        graph = helper.make_graph(self.nodes, self.graph_name, self.inputs, self.outputs, self.initializer, self.doc_string)
        onnx_id = helper.make_opsetid("", self.opset)
        return helper.make_model(graph, opset_imports=[onnx_id])
    def convert_op(self, 
                    op_name,
                    inputs,
                    attrs,
                    args):
        """Convert Relay op into a ONNX op.
        Parameters
        ----------
        args : list of tvm.relay.Expr
            The argument of relay call node
        op_name : str
            Name of operator, such as nn.conv2d, nn.dense
        inputs : list of str
            Inputs of operator
        outputs : list of str
            Outputs of operator
        attrs : dict
            Attribute of operator 
        Returns
        -------
        sym : onnx.NodeProto
            Converted ONNX node.
        """
        convert_map = _get_impl_map(self.opset)
        if op_name in convert_map:
            outputs = convert_map[op_name](self, inputs, attrs, args)
        else:
            raise NotImplementedError(
                "Operator {} not implemented.".format(op_name))
        return outputs
class ONNXGenerator(ExprVisitor):
    """A visitor to traverse relay function and generate ONNX model."""
    def __init__(self, params, opset, graph_name, doc_string):
        super(ONNXGenerator, self).__init__()
        self.graph = ONNXGraph(params, opset, graph_name, doc_string)
        self.name_map = {}
        self.func_attrs = {}
        self.parent_map = {}
        self.in_func = False
    def to_onnx(self, func):
        assert isinstance(func, Function), 'func must be tvm.relay.Function.'
        # Traverse relay function
        self.visit(func)
        # Add output
        output_name = self.graph.nodes[-1].output[0]
        self.graph.add_outputs(output_name, func.ret_type)
        # Make model
        return self.graph.make_model()
    def get_func_attrs(self, name, func_attrs):
        new_attrs={}
        def get_func_major_op(func_name):
            return func_name.replace('_nn_', '_nn.').split('_')[1]
        new_name = get_func_major_op(name)
        if func_attrs is None:
            return new_name, new_attrs
        else:    
            for call, attrs in func_attrs.items():
                if call.op.name == new_name:
                    new_attrs.update(attrs)
                    return new_name, new_attrs
            new_attrs.update(func_attrs)
            return name, new_attrs
    def visit_constant(self, const):
        if not self.in_func:
            self.name_map[const] = self.graph.add_const(const)
    def visit_var(self, v):
        if not self.in_func:
            self.graph.add_var(v)
            self.name_map[v] = [v.name_hint]
    def visit_tuple(self, t):
        self.name_map[t] = []
        if not self.in_func:
            for x in t.fields:
                self.visit(x)
                self.name_map[t].extend(self.name_map[x])
        else:
            super(GraphVisitor, self).visit_tuple(t)
    def visit_tuple_getitem(self, op):
        if not self.in_func:
            self.visit(op.tuple_value)
            self.name_map[op] = [self.name_map[op.tuple_value][op.index]]       
        else:
            super(GraphVisitor, self).visit_tuple_getitem(t)    
    def visit_call(self, c):
        if not self.in_func:
            node_params={'inputs':[]}
            name = ''
            if isinstance(c.op, Op):
                name = c.op.name
                node_params['op_name'] = name
                attrs = convert_attrs_to_dict(c.attrs)
                node_params['attrs']=attrs
            elif isinstance(c.op, Function):
                func_name = str(c.op.attrs['Composite'])
                self.in_func = True
                self.visit(c.op)
                self.in_func = False
                name, attrs = self.get_func_attrs(func_name, self.func_attrs)
                node_params['op_name'] = name
                node_params['attrs']=attrs
                self.func_attrs.clear()
            for a in c.args:
                self.visit(a)
                node_params['inputs'].extend(self.name_map.get(a, []))
            self.name_map[c] = self.graph.add_nodes(c, **node_params)    
        else:
            self.func_attrs[c] = convert_attrs_to_dict(c.attrs)
            for a in c.args:
                self.visit(a)
def fused_gemm():
    """Fuse operatprs to gemm op"""
    def gemm(transA, transB, bias):
        A = new_var('A', shape=(1,1)) # m, k
        B = new_var('B', shape=(1,1)) # n, k
        alpha = _op.const(1.0)
        _A, _B = A, B
        if transA:
            _A = _op.transpose(_A, axes=(1, 0))
        if transB:
            _B = _op.transpose(_B, axes=(1, 0)) # k, n
        _A = _op.nn.batch_flatten(_A) # 1, m
        out = _op.nn.dense(alpha * _A, _B) # m. k * k, n = m, n
        if bias:
            C = new_var('C', shape=(1,1))
            beta = _op.const(1.0)
            out = _op.nn.bias_add(out,beta*C)
            return tvm.relay.Function([A,B,C],out)
        else:
            return tvm.relay.Function([A,B],out)
    func_list = []
    for ta in (0, 1):
        for tb in (0, 1):
            for b in (1, 0):
                func_list.append((gemm(ta,tb, b), "fused_gemm"))
    return PartitionGraphByExpr(func_list)
def fused_conv_nn_bias():
    """Fuse the series of convolution ops and bias add ops"""
    return PartitionGraphInOrder(
        op_attrs = [('nn.bias_add', None), None],
        include = [('nn.conv1d', None),
                   ('nn.conv2d', None),
                   ('nn.conv3d', None),
                   ('nn.conv1d_transpose', None),
                   ('nn.conv2d_transpose', None)],
        func_name = ''
    )

def fused_unsqueeze():
    """Fuse expand_dims ops to unsqueeze op."""
    return PartitionGraphInUnorder(
        op_attrs = [('expand_dims',None)],
        func_name = 'fused_unsqueeze',
    )
def fused_nonzero():
    """Fuse expand_dims ops to nonzero op."""
    return PartitionGraphInOrder(
        op_attrs = [('transpose',None), ('argwhere',None)],
        func_name = 'fused_nonzero',
    )
def fused_onehot():
    """Fuse expand_dims ops to unsqueeze op."""
    def onehot():
        indices = new_var('indices',shape = (3,))
        values = new_var('values')
        off_value, on_value = _op.take(values,_op.const(0)), _op.take(values, _op.const(1))
        out = _op.one_hot(indices, on_value, off_value, depth = 1, axis=-1, dtype=None) # set as default.
        return tvm.relay.Function([indices, values],out)
    return PartitionGraphByExpr([(onehot(),'fused_onehot')])
def fused_matmul():
    def dense():
        a = new_var('a',shape = (3,2))
        b = new_var('b',shape = (2,3))
        b_t = _op.transpose(b, axes=[1, 0])
        out =  _op.nn.dense(a,b_t)
        return tvm.relay.Function([a, b],out)
    def batch_matmul():
        a = new_var('a',shape = (2,3,2))
        b = new_var('b',shape = (2,2,3))
        b_t = _op.transpose(b, axes=[0, 2, 1])
        out =  _op.nn.batch_matmul(a,b_t)
        return tvm.relay.Function([a, b],out)
    return PartitionGraphByExpr([(dense(),'fused_matmul'),(batch_matmul(),'fused_matmul')])
def fuse_special_ops(model,device_type):
    "Fuse the Relay ops to fit ONNX model"
    return Sequential([fused_gemm(), fused_unsqueeze(), fused_onehot()])(model)
def fuse_ops(model):
    "Fuse the Relay ops to fit ONNX model"
    return Sequential([fused_gemm(), fused_conv_nn_bias(), fused_unsqueeze(), fused_onehot(), fused_matmul(), fused_nonzero()])(model)

def to_onnx_check_non_suppoted_op(tar_ops):
    """Check if operators in list are not supported for Relay to ONNX conversion.
    Parameters
    ----------
    tar_ops : list of str
        The target operator names.
    Returns
    -------
    no_ops : list of str
        The operator names which is not supported by TVM.
    """
    no_ops = []
    exception = ['nn.bias_add']
    ops = _get_impl_map(1).keys()
    for op in tar_ops:
        if op not in ops and op not in exception: # nn.bias_add is exception.
            no_ops.append(op)
    return no_ops

def to_onnx(model, params, graph_name="", opset = None, doc_string=None):
    """Convert a Relay Function into ONNX model.
    Parameters
    ----------
    model : tvm.IRModule or _op.Function
        The relay module for compilation
    params : dict of str to tvm.nd.NDArray
        The parameter dict to be used by relay
    graph_name : str
        The name of the model's graph.
    doc_string :str
        The description of the model's graph
    Returns
    -------
    onnx_graph : protobuf object
        The returned ONNX ModelProto
    """
    if opset is None:
        try:
            opset = onnx.defs.onnx_opset_version() # get the supported opset version
        except AttributeError:
            opset = 11
    if isinstance(model, tvm.IRModule):
        pass
    elif isinstance(model, _op.Function):
        model = IRModule.from_expr(model)
    else:
        raise TypeError(
                'The paramter "model" must be tvm.IRModule or tvm.relay.Function.')
    # Fuse op
    model = fuse_ops(model)
    func = model['main']
    # Traverse the Relay function and record the nodes.
    onnx_model = ONNXGenerator(params, opset, graph_name, doc_string).to_onnx(func)
    return onnx_model
