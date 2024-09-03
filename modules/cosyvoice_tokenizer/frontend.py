# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from functools import partial
import onnxruntime
import torch
import numpy as np
import whisper
import torchaudio.compliance.kaldi as kaldi

class CosyVoiceFrontEnd:

    def __init__(self, speech_tokenizer_model: str, device: str = 'cuda', device_id: int = 0):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        option = onnxruntime.SessionOptions()
        option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        option.intra_op_num_threads = 1
        self.speech_tokenizer_session = onnxruntime.InferenceSession(speech_tokenizer_model, sess_options=option, providers=["CUDAExecutionProvider"if device == "cuda" else "CPUExecutionProvider"])
        if device == 'cuda':
            self.speech_tokenizer_session.set_providers(['CUDAExecutionProvider'], [ {'device_id': device_id}])

    def extract_speech_token(self, speech):
        feat = whisper.log_mel_spectrogram(speech, n_mels=128)
        speech_token = self.speech_tokenizer_session.run(None, {self.speech_tokenizer_session.get_inputs()[0].name: feat.detach().cpu().numpy(),
                                                                self.speech_tokenizer_session.get_inputs()[1].name: np.array([feat.shape[2]], dtype=np.int32)})[0].flatten().tolist()
        speech_token = torch.tensor([speech_token], dtype=torch.int32).to(self.device)
        speech_token_len = torch.tensor([speech_token.shape[1]], dtype=torch.int32).to(self.device)
        return speech_token, speech_token_len

    def _extract_spk_embedding(self, speech):
        feat = kaldi.fbank(speech,
                           num_mel_bins=80,
                           dither=0,
                           sample_frequency=16000)
        feat = feat - feat.mean(dim=0, keepdim=True)
        embedding = self.campplus_session.run(None, {self.campplus_session.get_inputs()[0].name: feat.unsqueeze(dim=0).cpu().numpy()})[0].flatten().tolist()
        embedding = torch.tensor([embedding]).to(self.device)
        return embedding

    def _extract_speech_feat(self, speech):
        speech_feat = self.feat_extractor(speech).squeeze(dim=0).transpose(0, 1).to(self.device)
        speech_feat = speech_feat.unsqueeze(dim=0)
        speech_feat_len = torch.tensor([speech_feat.shape[1]], dtype=torch.int32).to(self.device)
        return speech_feat, speech_feat_len