# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
from paddle import _C_ops
from paddle.fluid import core
from paddle.fluid import framework
from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid.regularizer import L2DecayRegularizer
from collections import defaultdict


class HybridMomentum(paddle.optimizer.Momentum):
    def __init__(self,
                 learning_rate=0.001,
                 momentum=0.9,
                 parameters=None,
                 use_nesterov=False,
                 weight_decay=None,
                 grad_clip=None,
                 rescale_grad=1.0,
                 name=None):

        parameters = list(parameters)
        if len(parameters) == 0:
            raise ValueError("optimizer got an empty parameter list")
        if not isinstance(parameters[0], dict):
            parameters = [{'params': parameters}]

        multi_precision = False
        use_multi_tensor = True
        super(HybridMomentum, self).__init__(
            learning_rate, momentum, parameters, use_nesterov, weight_decay,
            grad_clip, multi_precision, rescale_grad, use_multi_tensor, name)

        self._param_dict_list = defaultdict(lambda: defaultdict(list))
        self._velocity_dict_list = defaultdict(lambda: defaultdict(list))
        self._regularization_method_dict_list = defaultdict(
            lambda: defaultdict(list))
        self._regularization_coeff_dict_list = defaultdict(
            lambda: defaultdict(list))

        # learning_rate must be float32, regardless of whether the precsion is used.
        self._dtype = 'float32'

    def _get_accumulator(self, name, param):
        if self._name is not None:
            name = self._name + "_" + name
        if name not in self._accumulators:
            raise Exception("Accumulator {} does not exist".format(name))
        if param.name not in self._accumulators[name]:
            if getattr(param, 'has_sparse_grad', None) and param.stop_gradient:
                self._add_accumulator(self._velocity_acc_str, param)
            else:
                raise Exception(
                    "Accumulator {} does not exist for parameter {}".format(
                        name, param.name))
        return self._accumulators[name][param.name]

    def clear_grad(self):
        for param_group in self._param_groups:
            for p in param_group['params']:
                if not p.stop_gradient:
                    p.clear_gradient()

                if getattr(p, 'has_sparse_grad', None):
                    p.clear_gradient(set_to_zero=False)
                    delattr(p, 'index')
                    delattr(p, 'axis')

    def _create_accumulators(self, block, parameters):
        assert isinstance(block, framework.Block)

        if isinstance(parameters, dict):
            parameters = self._update_param_group(parameters)

        for p in parameters:
            self._add_accumulator(self._velocity_acc_str, p)

    def _multi_tensor_init(self, target_block, parameters_and_grads):

        if parameters_and_grads['param_group_index'] in self._param_dict_list:
            return

        group_id = parameters_and_grads['param_group_index']

        parameters = [
            p[0] for p in parameters_and_grads['params']
            if not p[0].stop_gradient \
            or getattr(p[0], 'has_sparse_grad', None)
        ]
        self._update_param_group(parameters_and_grads)
        self._create_accumulators(target_block, parameters)
        for param in parameters:
            velocity_acc = self._get_accumulator(self._velocity_acc_str, param)
            regularization_method = self._regularization_method
            regularization_coeff = self._regularization_coeff
            if hasattr(param, 'regularizer'):
                # we skip param's l2decay before, so fuse it with momentum here.
                if isinstance(param.regularizer, L2DecayRegularizer):
                    regularization_method = "l2_decay"
                    regularization_coeff = param.regularizer._regularization_coeff
                elif param.regularizer is not None:
                    regularization_method = ""
                    regularization_coeff = 0.0

            if param.type != core.VarDesc.VarType.LOD_TENSOR:
                raise ValueError(
                    "Multi tensor momentum only support LOD_TENSOR.")

            dict_key = ''
            if getattr(param, 'has_sparse_grad', None):
                dict_key += 'SPARSE_'

            if param.dtype == paddle.float32:
                dict_key += 'FP32_'
            elif param.dtype == paddle.float16:
                dict_key += 'FP16_'
            else:
                raise ValueError(
                    "Multi tensor momentum only support fp32 and fp16 parameters."
                )
            dict_key += 'LODTensor'

            self._param_dict_list[group_id][dict_key].append(param)
            self._velocity_dict_list[group_id][dict_key].append(velocity_acc)
            self._regularization_method_dict_list[group_id][dict_key].append(
                regularization_method)
            self._regularization_coeff_dict_list[group_id][dict_key].append(
                regularization_coeff)

    def _append_optimize_multi_tensor_op(self, target_block,
                                         parameters_and_grads):
        """ 
        For Multi Tensor, append optimize merged_operator to block.
        """
        assert isinstance(target_block, framework.Block)

        grad_dict = defaultdict(list)
        lr_dict = defaultdict(list)

        for param_and_grad in parameters_and_grads['params']:
            if param_and_grad[1] is None:
                continue
            if param_and_grad[0].stop_gradient is False or \
                    getattr(param_and_grad[0], 'has_sparse_grad', None):
                param_grad_dict = dict()
                param_grad_dict['params'] = param_and_grad
                param_grad_dict.update({
                    k: v
                    for k, v in parameters_and_grads.items() if k != 'params'
                })
                param_and_grad = self._update_param_group(param_grad_dict)

                if param_and_grad[1].type != core.VarDesc.VarType.LOD_TENSOR:
                    raise ValueError(
                        "Multi tensor momentum only support LOD_TENSOR.")

                dict_key = ''
                if getattr(param_and_grad[0], 'has_sparse_grad', None):
                    dict_key += 'SPARSE_'

                if param_and_grad[0].dtype == paddle.float32:
                    dict_key += 'FP32_'
                elif param_and_grad[0].dtype == paddle.float16:
                    dict_key += 'FP16_'
                else:
                    raise ValueError(
                        "Multi tensor momentum only support fp32 and fp16 parameters."
                    )
                dict_key += 'LODTensor'

                grad_dict[dict_key].append(param_and_grad[1])
                lr = self._create_param_lr(param_and_grad)
                lr_dict[dict_key].append(lr)

        group_id = parameters_and_grads['param_group_index']

        self._param_dict_list[group_id]
        for key in self._param_dict_list[group_id]:
            if len(self._param_dict_list[group_id][key]) == 0:
                continue

            if 'SPARSE_' in key:
                assert len(self._param_dict_list[group_id][key]) == len(
                    grad_dict[key])
                for idx in range(len(self._param_dict_list[group_id][key])):
                    param_and_grad = [
                        self._param_dict_list[group_id][key][idx],
                        grad_dict[key][idx]
                    ]
                    velocity_acc = self._velocity_dict_list[group_id][key][idx]
                    regularization_method = self._regularization_method_dict_list[
                        group_id][key][idx]
                    regularization_coeff = self._regularization_coeff_dict_list[
                        group_id][key][idx]
                    lr = lr_dict[key][idx]
                    index = getattr(param_and_grad[0], 'index', None)
                    axis = getattr(param_and_grad[0], 'axis', None)
                    _, _ = paddle._C_ops.sparse_momentum(
                        param_and_grad[0], param_and_grad[1], velocity_acc,
                        index, lr, param_and_grad[0], velocity_acc, 'mu',
                        self._momentum, 'use_nesterov', self._use_nesterov,
                        'regularization_method', regularization_method,
                        'regularization_coeff', regularization_coeff, 'axis',
                        axis)
            else:
                _, _, _ = _C_ops.merged_momentum(
                    self._param_dict_list[group_id][key], grad_dict[key],
                    self._velocity_dict_list[group_id][key], lr_dict[key],
                    None, self._param_dict_list[group_id][key],
                    self._velocity_dict_list[group_id][key], None, 'mu',
                    self._momentum, 'use_nesterov', self._use_nesterov,
                    'regularization_method',
                    self._regularization_method_dict_list[group_id][key],
                    'regularization_coeff',
                    self._regularization_coeff_dict_list[group_id][key],
                    'multi_precision', False)

        return None

    def _create_optimization_pass(self, parameters_and_grads):
        global_block = framework.default_main_program().global_block()
        target_block = global_block
        current_block = framework.default_main_program().current_block()
        if current_block.idx != global_block.idx:
            assert current_block.backward_block_idx != -1, \
                "current block is not global_block, but it doesn't have backward block."
            target_block = framework.default_main_program().blocks[
                current_block.backward_block_idx]

        start = len(target_block.ops)
        self.helper = LayerHelper(self.__class__.__name__)

        self._create_global_learning_rate()

        self._multi_tensor_init(target_block, parameters_and_grads)

        self._append_optimize_multi_tensor_op(target_block,
                                              parameters_and_grads)

        # Get custom finish ops for subclasses
        # FIXME: Need to fix this once we figure out how to handle dependencies
        self._finish_update(target_block, parameters_and_grads)

        end = len(target_block.ops)
        return target_block._slice_ops(start, end)

    @paddle.no_grad()
    def step(self):
        for idx, param_group in enumerate(self._param_groups):
            params_grads = defaultdict(lambda: list())
            params_grads['param_group_index'] = 'param_group_%d' % idx
            for param in param_group['params']:
                if param._grad_ivar() is not None:
                    grad_var = param._grad_ivar()
                    params_grads['params'].append((param, grad_var))

            params_grads.update(
                {k: v
                 for k, v in param_group.items() if k != 'params'})

            with framework.program_guard(framework.default_main_program(),
                                         framework.default_startup_program()):
                grad_clip = params_grads['grad_clip']
                if grad_clip is not None:
                    params_grads['params'] = grad_clip(params_grads['params'])

                optimize_ops = self._create_optimization_pass(params_grads)
