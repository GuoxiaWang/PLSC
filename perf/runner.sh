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

model=${1:-"r50"}
batch_size_per_device=${2:-128}
gpus=${3:-0}
node_num=${4:-1}
dtype=${5:-"fp32"}
test_num=${6:-1}
a=`expr ${#gpus} + 1`
gpu_num_per_node=`expr ${a} / 2`

log_dir=./logs/paddle-plsc/arcface/bz${batch_size_per_device}/${node_num}n${gpu_num_per_node}g
mkdir -p $log_dir
log_file=$log_dir/${model}_b${batch_size_per_device}_${dtype}_${test_num}.log

if [ $model = "r100" ]; then
    sed -i "s/ins.set_model_name\S*/ins.set_model_name\(\"ResNet100\"\)/" train.py
else
    sed -i "s/ins.set_model_name\S*/ins.set_model_name\(\"ResNet50\"\)/" train.py
fi

if [ $dtype = "fp32" ]; then
    sed -i "s/ins.set_mixed_precision\S*/ins.set_mixed_precision\(False\)/" train.py
else
    sed -i "s/ins.set_mixed_precision\S*/ins.set_mixed_precision\(True\)/" train.py
fi

sed -i "s/ins.set_train_batch_size\S*/ins.set_train_batch_size\(${batch_size_per_device}\)/" train.py

if [ ${gpu_num_per_node} -eq 1 ] ; then
    sed -i "s/ins.set_model_parallel\S*/ins.set_model_parallel\(False\)/" train.py
    python3 train.py  2>&1 | tee $log_file
else
    sed -i "s/ins.set_model_parallel\S*/ins.set_model_parallel\(True\)/" train.py
    python3 -m paddle.distributed.launch   \
        --gpus=${gpus}   train.py  2>&1 | tee $log_file
fi
