# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

import math
from six.moves import reduce

import paddle
from paddle.utils import unique_name
from paddle.distributed.fleet.utils import class_center_sample

__all__ = ["BaseModel"]


class BaseModel(object):
    """
    Base class for custom models.
    The sub-class must implement the build_network method,
    which constructs the custom model. And we will add the
    distributed fc layer for you automatically.
    """

    def __init__(self):
        super(BaseModel, self).__init__()

    def build_network(self, input, is_train=True):
        """
        Construct the custom model, and we will add the distributed fc layer
        at the end of your model automatically.
        """
        raise NotImplementedError(
            "You must implement this method in your subclass.")

    def get_output(self,
                   input,
                   num_classes,
                   num_ranks=1,
                   rank_id=0,
                   model_parallel=False,
                   is_train=True,
                   param_attr=None,
                   bias_attr=None,
                   margin1=1.0,
                   margin2=0.5,
                   margin3=0.0,
                   scale=64.0,
                   sample_ratio=1.0):
        """
        Add the distributed fc layer for the custom model.

        Params:
            input: input for the model
            label: label for the input
            num_classes: number of classes for the classifier
            num_ranks: number of trainers, i.e., GPUs
            rank_id: id for the current trainer, from 0 to num_ranks - 1
            model_parallel: whether use model parallel to training
            is_train: build the network for training or not
            param_attr: param_attr for the weight parameter of fc
            bias_attr: bias_attr for the weight parameter for fc
            margin1: the margin parameter for margin loss, cos(m1 * theta + m2) - m3
            margin2: the margin parameter for margin loss
            margin3: the margin parameter for margin loss
            scale: the scale parameter for margin loss
        """
        emb = self.build_network(input, is_train)
        label = input.label
        prob = None
        loss = None
        if model_parallel:
            loss = BaseModel._distributed_margin_softmax_classify(
                x=emb,
                label=label,
                nclasses=num_classes,
                nranks=num_ranks,
                rank_id=rank_id,
                margin1=margin1,
                margin2=margin2,
                margin3=margin3,
                logit_scale=scale,
                param_attr=param_attr,
                sample_ratio=sample_ratio)
        else:
            loss, prob = BaseModel._margin_softmax(
                emb, label, num_classes, param_attr, margin1, margin2, margin3,
                scale, sample_ratio)

        return emb, loss, prob

    @staticmethod
    def _distributed_margin_softmax_classify(x,
                                             label,
                                             nclasses,
                                             nranks,
                                             rank_id,
                                             margin1=1.0,
                                             margin2=0.5,
                                             margin3=0.0,
                                             logit_scale=64.0,
                                             param_attr=None,
                                             sample_ratio=1.0,
                                             name=None):
        if name is None:
            name = 'dist@margin_softmax@rank@%05d' % rank_id

        shard_dim = (nclasses + nranks - 1) // nranks
        if nclasses % nranks != 0:
            if rank_id == nranks - 1:
                shard_dim = nclasses % shard_dim

        in_dim = reduce(lambda a, b: a * b, x.shape[1:], 1)

        if param_attr is None:
            stddev = math.sqrt(2.0 / (in_dim + nclasses))
            param_attr = paddle.ParamAttr(
                initializer=paddle.nn.initializer.Normal(std=stddev))
        weight_shape = [in_dim, shard_dim]
        weight = paddle.static.create_parameter(
            shape=weight_shape,
            dtype=x.dtype,
            name=name,
            attr=param_attr,
            is_bias=False)

        # avoid allreducing gradients for distributed parameters
        weight.is_distributed = True
        # avoid broadcasting distributed parameters in startup program
        paddle.static.default_startup_program().global_block().vars[
            weight.name].is_distributed = True

        # normalize x
        x_l2 = paddle.sqrt(paddle.sum(paddle.square(x), axis=1, keepdim=True))
        norm_x = paddle.divide(x, x_l2)

        norm_x_list = []
        paddle.distributed.all_gather(norm_x_list, norm_x)
        norm_x_all = paddle.concat(norm_x_list, axis=0)

        label_list = []
        paddle.distributed.all_gather(label_list, label)
        label_all = paddle.concat(label_list, axis=0)
        label_all.stop_gradient = True

        if sample_ratio < 1.0:
            # partial fc sample process
            num_sample = int(shard_dim * sample_ratio)
            label_all, sampled_class_index = class_center_sample(
                label_all, shard_dim, num_sample, nranks, rank_id)
            sampled_class_index.stop_gradient = True
            weight = paddle.gather(weight, sampled_class_index, axis=1)

        # normalize weight
        weight_l2 = paddle.sqrt(
            paddle.sum(paddle.square(weight), axis=0, keepdim=True))
        norm_weight = paddle.divide(weight, weight_l2)

        shard_logit = paddle.matmul(norm_x_all, norm_weight)

        global_loss, shard_prob = paddle.distributed.collective._c_margin_softmax_with_cross_entropy(
            shard_logit,
            label_all,
            margin1=margin1,
            margin2=margin2,
            margin3=margin3,
            scale=logit_scale,
            return_softmax=True)

        avg_loss = paddle.mean(global_loss)

        avg_loss._set_info('shard_logit', shard_logit)
        avg_loss._set_info('shard_prob', shard_prob)
        avg_loss._set_info('label_all', label_all)
        avg_loss._set_info('shard_dim', shard_dim)

        return avg_loss

    @staticmethod
    def _margin_softmax(input, label, out_dim, param_attr, margin1, margin2,
                        margin3, scale, sample_ratio):
        input_norm = paddle.sqrt(
            paddle.sum(paddle.square(input), axis=1, keepdim=True))
        input = paddle.divide(input, input_norm)

        if param_attr is None:
            param_attr = paddle.ParamAttr(
                initializer=paddle.nn.initializer.XavierNormal(fan_in=0.0))
        weight = paddle.static.create_parameter(
            shape=[input.shape[1], out_dim],
            dtype='float32',
            name=unique_name.generate('final_fc_w'),
            attr=param_attr)

        if sample_ratio < 1.0:
            # partial fc sample process
            num_sample = int(out_dim * sample_ratio)
            label, sampled_class_index = class_center_sample(label, out_dim,
                                                             num_sample, 1, 0)
            sampled_class_index.stop_gradient = True
            weight = paddle.gather(weight, sampled_class_index, axis=1)
            out_dim = paddle.shape(sampled_class_index)

        weight_norm = paddle.sqrt(
            paddle.sum(paddle.square(weight), axis=0, keepdim=True))
        weight = paddle.divide(weight, weight_norm)
        cos = paddle.matmul(input, weight)

        theta = paddle.acos(cos)
        if margin1 != 1.0:
            theta = margin1 * theta
        if margin2 != 0.0:
            theta = theta + margin2
        margin_cos = paddle.cos(theta)
        if margin3 != 0.0:
            margin_cos = margin_cos - margin3

        one_hot = paddle.nn.functional.one_hot(label, num_classes=out_dim)
        diff = paddle.multiply(paddle.subtract(margin_cos, cos), one_hot)
        target_cos = paddle.add(cos, diff)
        logit = paddle.scale(target_cos, scale=scale)

        loss, prob = paddle.nn.functional.softmax_with_cross_entropy(
            logits=logit,
            label=paddle.reshape(label, (-1, 1)),
            return_softmax=True)
        avg_loss = paddle.mean(x=loss)

        one_hot.stop_gradient = True

        return avg_loss, prob
