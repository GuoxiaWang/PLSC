#!/usr/bin/bash

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

shell_folder=$(dirname $(readlink -f "$0"))
model=${1:-r50}
batch_size_per_device=${2:-128}
dtype=${3:-'fp32'}
test_num=${4:-5}


i=1
while [ $i -le ${test_num} ]
do
    bash $shell_folder/runner.sh ${model} ${batch_size_per_device}  0  1    $dtype  ${i}
    echo " >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Finished Test Case ${i}!<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< "
    let i++
    sleep 20s
done


i=1
while [ $i -le ${test_num} ]
do
    bash $shell_folder/runner.sh ${model} ${batch_size_per_device}  0,1,2,3  1   $dtype  ${i}
    echo " >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Finished Test Case ${i}!<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< "
    let i++
    sleep 20s
done


i=1
while [ $i -le ${test_num} ]
do
    bash $shell_folder/runner.sh ${model} ${batch_size_per_device}  0,1,2,3,4,5,6,7  1  $dtype  ${i}
    echo " >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Finished Test Case ${i}!<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< "
    let i++
    sleep 20s
done
