# Copyright (2024) Tsinghua University, Bytedance Ltd. and/or its affiliates
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

from .salmonn import *

def load_model(config):
    model_type = config.get("model_type", "salmonn")
    if model_type == "salmonn":
        return SALMONN.from_config(config)
    elif model_type == "cot_salmonn":
        return CoTSALMONN.from_config(config)
    elif model_type == "reverse_salmonn":
        return ReverseSALMONN.from_config(config)
    elif model_type == "mutor":
        return MutorSALMONN.from_config(config)
    elif model_type == "mutor_bert_summary":
        return MutorBERTSummarySALMONN.from_config(config)
    elif model_type == "mutor_bert_conclusion":
        return MutorBERTConclusionSALMONN.from_config(config)
    elif model_type == "mutor_bert_multi_conclusion":
        return MutorBERTMultiConclusionSALMONN.from_config(config)
    elif model_type == "caption_attention_salmonn":
        return CaptionAttentionSALMONN.from_config(config)
    elif model_type == "mutor_bert_triplet_loss":
        return MutorBERTTripletLossSALMONN.from_config(config)
    else:
        raise ValueError(f"Invalid model type: {model_type}")