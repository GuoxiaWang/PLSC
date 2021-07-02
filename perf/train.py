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

import sys
from pathlib import Path
file = Path(__file__).resolve()
sys.path.append(str(file.parents[1]))
from plsc import Entry

if __name__ == "__main__":
    ins = Entry()
    #ins.set_dataset_dir('/path/to/your/data/folder/')
    ins.set_model_name("ResNet100")
    ins.set_dataset_dir('/plsc/MS1M_v2/')
    ins.set_step_boundaries([100000, 140000, 160000])
    ins.set_loss_type('arcface')
    ins.set_model_parallel(True)
    ins.set_sample_ratio(0.1)
    ins.set_mixed_precision(False)
    ins.set_stop_step(150)
    ins.set_train_epochs(17)
    ins.set_test_period(11373)
    ins.set_train_batch_size(128)
    ins.set_log_period(1)
    ins.set_calc_acc(False)
    ins.set_model_save_dir('./saved_model')
    ins.train()
