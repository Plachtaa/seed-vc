import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'  # 添加这行
import sys
from dotenv import load_dotenv
import shutil
from funasr import AutoModel
import soundfile as sf
load_dotenv()

os.environ["OMP_NUM_THREADS"] = "4"
if sys.platform == "darwin":
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

now_dir = os.getcwd()
sys.path.append(now_dir)
import multiprocessing
import warnings
import yaml

warnings.simplefilter("ignore")

from tqdm import tqdm
from modules.commons import *
import librosa
import torchaudio
import torchaudio.compliance.kaldi as kaldi

from hf_utils import load_custom_model_from_hf

import os
import sys
import torch
from modules.commons import str2bool
from pathlib import Path
# Load model and configuration
device = None

flag_vc = False

prompt_condition, mel2, style2 = None, None, None
reference_wav_name = ""

prompt_len = 3  # in seconds
ce_dit_difference = 2.0  # 2 seconds
fp16 = False
@torch.no_grad()  # 推理时不需要计算梯度
def custom_infer(model_set,
                 reference_wav,        # 参考音频数据
                 new_reference_wav_name,  # 参考音频文件名
                 input_wav_res,        # 输入的音频数据(16kHz采样率)
                 block_frame_16k,      # 16kHz采样率下的帧长
                 skip_head,            # 要跳过的头部帧数
                 skip_tail,            # 要跳过的尾部帧数
                 return_length,        # 要返回的音频长度
                 diffusion_steps,      # 扩散模型的推理步数
                 inference_cfg_rate,   # 推理时的CFG rate
                 max_prompt_length,    # 最大提示音频长度(秒)
                 cd_difference=2.0,    # 内容编码器和DiT的时间差异
                 pitch_shift=0,        # 添加音高调节参数
                 ):
    # 使用全局变量来缓存计算结果，避免重复计算
    global prompt_condition, mel2, style2
    global reference_wav_name
    global prompt_len
    global ce_dit_difference

    # 解包模型集合
    (
        model,              # 主模型
        semantic_fn,        # 语义特征提取函数
        vocoder_fn,         # 声码器
        campplus_model,     # 说话人风格编码器
        to_mel,            # 梅尔频谱图转换函数
        mel_fn_args,       # 梅尔频谱图参数
    ) = model_set

    sr = mel_fn_args["sampling_rate"]        # 采样率
    hop_length = mel_fn_args["hop_size"]     # 帧移

    # 更新时间差异设置
    if ce_dit_difference != cd_difference:
        ce_dit_difference = cd_difference
        print(f"Setting ce_dit_difference to {cd_difference} seconds.")

    # 只有在必要时才重新计算参考音频的特征
    if prompt_condition is None or reference_wav_name != new_reference_wav_name or prompt_len != max_prompt_length:
        prompt_len = max_prompt_length
        print(f"Setting max prompt length to {max_prompt_length} seconds.")
        # 截取指定长度的参考音频
        reference_wav = reference_wav[:int(sr * prompt_len)]
        reference_wav_tensor = torch.from_numpy(reference_wav).to(device)

        # 将参考音频重采样到16kHz
        ori_waves_16k = torchaudio.functional.resample(reference_wav_tensor, sr, 16000)
        # 提取语义特征
        S_ori = semantic_fn(ori_waves_16k.unsqueeze(0))
        # 计算声学特征
        feat2 = torchaudio.compliance.kaldi.fbank(
            ori_waves_16k.unsqueeze(0), num_mel_bins=80, dither=0, sample_frequency=16000
        )
        feat2 = feat2 - feat2.mean(dim=0, keepdim=True)  # 均值归一化
        # 提取说话人风格特征
        style2 = campplus_model(feat2.unsqueeze(0))

        # 计算梅尔频谱图
        mel2 = to_mel(reference_wav_tensor.unsqueeze(0))
        target2_lengths = torch.LongTensor([mel2.size(2)]).to(mel2.device)
        # 通过长度调节器处理特征
        prompt_condition = model.length_regulator(
            S_ori, ylens=target2_lengths, n_quantizers=3, f0=None
        )[0]

        reference_wav_name = new_reference_wav_name

    # 在处理输入音频之前进行音高调节
    if pitch_shift != 0:
        # 使用librosa进行音高调节
        input_wav_res_np = input_wav_res.cpu().numpy()
        input_wav_res_shifted = librosa.effects.pitch_shift(
            y=input_wav_res_np,
            sr=16000,  # 固定使用16kHz采样率
            n_steps=pitch_shift,
            bins_per_octave=12
        )
        input_wav_res = torch.from_numpy(input_wav_res_shifted).to(device)
    
    # 处理输入音频
    converted_waves_16k = input_wav_res
    
    # 计时开始
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start_event.record()
    
    # 提取输入音频的语义特征
    S_alt = semantic_fn(converted_waves_16k.unsqueeze(0))
    
    # 计时结束
    end_event.record()
    torch.cuda.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event)
    print(f"Time taken for semantic_fn: {elapsed_time_ms}ms")

    # 处理时间差异
    ce_dit_frame_difference = int(ce_dit_difference * 50)
    S_alt = S_alt[:, ce_dit_frame_difference:]  # 裁剪特征
    # 计算目标长度
    target_lengths = torch.LongTensor([(skip_head + return_length + skip_tail - ce_dit_frame_difference) / 50 * sr // hop_length]).to(S_alt.device)
    print(f"target_lengths: {target_lengths}")
    
    # 通过长度调节器处理特征
    cond = model.length_regulator(
        S_alt, ylens=target_lengths , n_quantizers=3, f0=None
    )[0]
    # 拼接提示特征和输入特征
    cat_condition = torch.cat([prompt_condition, cond], dim=1)

    # 使用自动混合精度进行推理
    with torch.autocast(device_type=device.type, dtype=torch.float16 if fp16 else torch.float32):
        # 通过扩散模型生成目标梅尔频谱图
        vc_target = model.cfm.inference(
            cat_condition,
            torch.LongTensor([cat_condition.size(1)]).to(mel2.device),
            mel2,
            style2,
            None,
            n_timesteps=diffusion_steps,
            inference_cfg_rate=inference_cfg_rate,
        )
        # 只保留非提示部分
        vc_target = vc_target[:, :, mel2.size(-1) :]
        print(f"vc_target.shape: {vc_target.shape}")
        # 通过声码器生成波形
        vc_wave = vocoder_fn(vc_target).squeeze()

    # 计算输出长度和尾部长度
    output_len = return_length * sr // 50
    tail_len = skip_tail * sr // 50
    # 裁剪并返回最终音频
    output = vc_wave[-output_len - tail_len: -tail_len]
    return output

def load_models(args):
    global fp16
    fp16 = args.fp16
    ckpt_root = Path(__file__).parent / "checkpoints"
    print(f"Using fp16: {fp16}")
    if args.checkpoint_path is None or args.checkpoint_path == "":
        dit_checkpoint_path = ckpt_root / "seed_vc" / "real_time_model" / "DiT_uvit_tat_xlsr_ema.pth"
        dit_config_path = ckpt_root / "seed_vc" / "real_time_model" / "config_dit_mel_seed_uvit_xlsr_tiny.yml"
        if not dit_checkpoint_path.exists() or not dit_config_path.exists():
            dit_checkpoint_path, dit_config_path = load_custom_model_from_hf("Plachta/Seed-VC",
                                                                             "DiT_uvit_tat_xlsr_ema.pth",
                                                                             "config_dit_mel_seed_uvit_xlsr_tiny.yml")
    else:
        dit_checkpoint_path = args.checkpoint_path
        dit_config_path = args.config_path
    config = yaml.safe_load(open(dit_config_path, "r"))
    model_params = recursive_munch(config["model_params"])
    model_params.dit_type = 'DiT'
    model = build_model(model_params, stage="DiT")
    hop_length = config["preprocess_params"]["spect_params"]["hop_length"]
    sr = config["preprocess_params"]["sr"]

    # Load checkpoints
    model, _, _, _ = load_checkpoint(
        model,
        None,
        dit_checkpoint_path,
        load_only_params=True,
        ignore_modules=[],
        is_distributed=False,
    )
    for key in model:
        model[key].eval()
        model[key].to(device)
    model.cfm.estimator.setup_caches(max_batch_size=1, max_seq_length=8192)

    # Load additional modules
    from modules.campplus.DTDNN import CAMPPlus
    campplus_ckpt_path = ckpt_root / "campplus" / "campplus_cn_common.bin"
    if not campplus_ckpt_path.exists():
        campplus_ckpt_path = load_custom_model_from_hf(
            "funasr/campplus", "campplus_cn_common.bin", config_filename=None
        )
    campplus_model = CAMPPlus(feat_dim=80, embedding_size=192)
    campplus_model.load_state_dict(torch.load(campplus_ckpt_path, map_location="cpu"))
    campplus_model.eval()
    campplus_model.to(device)

    vocoder_type = model_params.vocoder.type

    if vocoder_type == 'bigvgan':
        from modules.bigvgan import bigvgan
        bigvgan_name = model_params.vocoder.name
        bigvgan_model = bigvgan.BigVGAN.from_pretrained(bigvgan_name, use_cuda_kernel=False)
        # remove weight norm in the model and set to eval mode
        bigvgan_model.remove_weight_norm()
        bigvgan_model = bigvgan_model.eval().to(device)
        vocoder_fn = bigvgan_model
    elif vocoder_type == 'hifigan':
        from modules.hifigan.generator import HiFTGenerator
        from modules.hifigan.f0_predictor import ConvRNNF0Predictor
        hift_config = yaml.safe_load(open('configs/hifigan.yml', 'r'))
        hift_gen = HiFTGenerator(**hift_config['hift'], f0_predictor=ConvRNNF0Predictor(**hift_config['f0_predictor']))
        hift_gen.load_state_dict(torch.load(str(ckpt_root / "cosy_hifigan" / "hift.pt"), map_location='cpu'))
        hift_gen.eval()
        hift_gen.to(device)
        vocoder_fn = hift_gen
    elif vocoder_type == "vocos":
        vocos_config = yaml.safe_load(open(model_params.vocoder.vocos.config, 'r'))
        vocos_path = model_params.vocoder.vocos.path
        vocos_model_params = recursive_munch(vocos_config['model_params'])
        vocos = build_model(vocos_model_params, stage='mel_vocos')
        vocos_checkpoint_path = vocos_path
        vocos, _, _, _ = load_checkpoint(vocos, None, vocos_checkpoint_path,
                                         load_only_params=True, ignore_modules=[], is_distributed=False)
        _ = [vocos[key].eval().to(device) for key in vocos]
        _ = [vocos[key].to(device) for key in vocos]
        total_params = sum(sum(p.numel() for p in vocos[key].parameters() if p.requires_grad) for key in vocos.keys())
        print(f"Vocoder model total parameters: {total_params / 1_000_000:.2f}M")
        vocoder_fn = vocos.decoder
    else:
        raise ValueError(f"Unknown vocoder type: {vocoder_type}")

    speech_tokenizer_type = model_params.speech_tokenizer.type
    if speech_tokenizer_type == 'whisper':
        # whisper
        from transformers import AutoFeatureExtractor, WhisperModel
        whisper_name = model_params.speech_tokenizer.name
        whisper_model = WhisperModel.from_pretrained(whisper_name, torch_dtype=torch.float16).to(device)
        del whisper_model.decoder
        whisper_feature_extractor = AutoFeatureExtractor.from_pretrained(whisper_name)

        def semantic_fn(waves_16k):
            ori_inputs = whisper_feature_extractor([waves_16k.squeeze(0).cpu().numpy()],
                                                   return_tensors="pt",
                                                   return_attention_mask=True)
            ori_input_features = whisper_model._mask_input_features(
                ori_inputs.input_features, attention_mask=ori_inputs.attention_mask).to(device)
            with torch.no_grad():
                ori_outputs = whisper_model.encoder(
                    ori_input_features.to(whisper_model.encoder.dtype),
                    head_mask=None,
                    output_attentions=False,
                    output_hidden_states=False,
                    return_dict=True,
                )
            S_ori = ori_outputs.last_hidden_state.to(torch.float32)
            S_ori = S_ori[:, :waves_16k.size(-1) // 320 + 1]
            return S_ori
    elif speech_tokenizer_type == 'cnhubert':
        from transformers import (
            Wav2Vec2FeatureExtractor,
            HubertModel,
        )
        hubert_model_name = config['model_params']['speech_tokenizer']['name']
        hubert_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(hubert_model_name)
        hubert_model = HubertModel.from_pretrained(hubert_model_name)
        hubert_model = hubert_model.to(device)
        hubert_model = hubert_model.eval()
        hubert_model = hubert_model.half()

        def semantic_fn(waves_16k):
            ori_waves_16k_input_list = [
                waves_16k[bib].cpu().numpy()
                for bib in range(len(waves_16k))
            ]
            ori_inputs = hubert_feature_extractor(ori_waves_16k_input_list,
                                                  return_tensors="pt",
                                                  return_attention_mask=True,
                                                  padding=True,
                                                  sampling_rate=16000).to(device)
            with torch.no_grad():
                ori_outputs = hubert_model(
                    ori_inputs.input_values.half(),
                )
            S_ori = ori_outputs.last_hidden_state.float()
            return S_ori
    elif speech_tokenizer_type == 'xlsr':
        from transformers import (
            Wav2Vec2FeatureExtractor,
            Wav2Vec2Model,
        )
        model_name = config['model_params']['speech_tokenizer']['name']
        output_layer = config['model_params']['speech_tokenizer']['output_layer']
        wav2vec_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        wav2vec_model = Wav2Vec2Model.from_pretrained(model_name)
        wav2vec_model.encoder.layers = wav2vec_model.encoder.layers[:output_layer]
        wav2vec_model = wav2vec_model.to(device)
        wav2vec_model = wav2vec_model.eval()
        wav2vec_model = wav2vec_model.half()

        def semantic_fn(waves_16k):
            ori_waves_16k_input_list = [
                waves_16k[bib].cpu().numpy()
                for bib in range(len(waves_16k))
            ]
            ori_inputs = wav2vec_feature_extractor(ori_waves_16k_input_list,
                                                   return_tensors="pt",
                                                   return_attention_mask=True,
                                                   padding=True,
                                                   sampling_rate=16000).to(device)
            with torch.no_grad():
                ori_outputs = wav2vec_model(
                    ori_inputs.input_values.half(),
                )
            S_ori = ori_outputs.last_hidden_state.float()
            return S_ori
    else:
        raise ValueError(f"Unknown speech tokenizer type: {speech_tokenizer_type}")
    # Generate mel spectrograms
    mel_fn_args = {
        "n_fft": config['preprocess_params']['spect_params']['n_fft'],
        "win_size": config['preprocess_params']['spect_params']['win_length'],
        "hop_size": config['preprocess_params']['spect_params']['hop_length'],
        "num_mels": config['preprocess_params']['spect_params']['n_mels'],
        "sampling_rate": sr,
        "fmin": config['preprocess_params']['spect_params'].get('fmin', 0),
        "fmax": None if config['preprocess_params']['spect_params'].get('fmax', "None") == "None" else 8000,
        "center": False
    }
    from modules.audio import mel_spectrogram

    to_mel = lambda x: mel_spectrogram(x, **mel_fn_args)

    return (
        model,
        semantic_fn,
        vocoder_fn,
        campplus_model,
        to_mel,
        mel_fn_args,
    )

def printt(strr, *args):
    if len(args) == 0:
        print(strr)
    else:
        print(strr % args)

class Config:
    def __init__(self):
        self.device = device


if __name__ == "__main__":
    import json
    import multiprocessing
    import re
    import threading
    import time
    import traceback
    from multiprocessing import Queue, cpu_count
    import argparse

    import librosa
    import numpy as np
    import FreeSimpleGUI as sg
    import sounddevice as sd
    import torch
    import torch.nn.functional as F
    import torchaudio.transforms as tat


    current_dir = os.getcwd()
    n_cpu = cpu_count()
    class GUIConfig:
        def __init__(self) -> None:
            self.reference_audio_path: str = ""
            # self.index_path: str = ""
            self.diffusion_steps: int = 10
            self.sr_type: str = "sr_model"
            self.block_time: float = 0.25  # s
            self.threhold: int = -60
            self.crossfade_time: float = 0.05
            self.extra_time_ce: float = 2.5
            self.extra_time: float = 0.5
            self.extra_time_right: float = 2.0
            self.I_noise_reduce: bool = False
            self.O_noise_reduce: bool = False
            self.inference_cfg_rate: float = 0.7
            self.sg_hostapi: str = ""
            self.wasapi_exclusive: bool = False
            self.sg_input_device: str = ""
            self.sg_output_device: str = ""
            self.pitch_shift: int = 0  # 添加音高调节参数，0表示不调节


    class GUI:
        def __init__(self, args) -> None:
            self.gui_config = GUIConfig()
            self.config = Config()
            self.function = "vc"
            self.delay_time = 0
            self.hostapis = None
            self.input_devices = None
            self.output_devices = None
            self.input_devices_indices = None
            self.output_devices_indices = None
            self.stream = None
            self.model_set = load_models(args)
            self.vad_model = AutoModel(model="checkpoints/speech_fsmn_vad_zh-cn-16k-common-pytorch", model_revision="v2.0.4")
            self.update_devices()
            self.launcher()

        def load(self):
            try:
                os.makedirs("configs/inuse", exist_ok=True)
                if not os.path.exists("configs/inuse/config.json"):
                    shutil.copy("configs/config.json", "configs/inuse/config.json")
                with open("configs/inuse/config.json", "r") as j:
                    data = json.load(j)
                    data["sr_model"] = data["sr_type"] == "sr_model"
                    data["sr_device"] = data["sr_type"] == "sr_device"
                    if data["sg_hostapi"] in self.hostapis:
                        self.update_devices(hostapi_name=data["sg_hostapi"])
                        if (
                            data["sg_input_device"] not in self.input_devices
                            or data["sg_output_device"] not in self.output_devices
                        ):
                            self.update_devices()
                            data["sg_hostapi"] = self.hostapis[0]
                            data["sg_input_device"] = self.input_devices[
                                self.input_devices_indices.index(sd.default.device[0])
                            ]
                            data["sg_output_device"] = self.output_devices[
                                self.output_devices_indices.index(sd.default.device[1])
                            ]
                    else:
                        data["sg_hostapi"] = self.hostapis[0]
                        data["sg_input_device"] = self.input_devices[
                            self.input_devices_indices.index(sd.default.device[0])
                        ]
                        data["sg_output_device"] = self.output_devices[
                            self.output_devices_indices.index(sd.default.device[1])
                        ]
            except:
                with open("configs/inuse/config.json", "w") as j:
                    data = {
                        "sg_hostapi": self.hostapis[0],
                        "sg_wasapi_exclusive": False,
                        "sg_input_device": self.input_devices[
                            self.input_devices_indices.index(sd.default.device[0])
                        ],
                        "sg_output_device": self.output_devices[
                            self.output_devices_indices.index(sd.default.device[1])
                        ],
                        "sr_type": "sr_model",
                        "block_time": 0.3,
                        "crossfade_length": 0.04,
                        "extra_time_ce": 2.5,
                        "extra_time": 0.5,
                        "extra_time_right": 0.02,
                        "diffusion_steps": 10,
                        "inference_cfg_rate": 0.7,
                        "max_prompt_length": 3.0,
                    }
                    data["sr_model"] = data["sr_type"] == "sr_model"
                    data["sr_device"] = data["sr_type"] == "sr_device"
            return data

        def launcher(self):
            self.config = Config()
            data = self.load()
            sg.theme("LightBlue3")
            layout = [
                [
                    sg.Frame(
                        title="Load reference audio",
                        layout=[
                            [
                                sg.Input(
                                    default_text=data.get("reference_audio_path", ""),
                                    key="reference_audio_path",
                                ),
                                sg.FileBrowse(
                                    "choose an audio file",
                                    initial_folder=os.path.join(
                                        os.getcwd(), "examples/reference"
                                    ),
                                    file_types=((". wav"), (". mp3"), (". flac"), (". m4a"), (". ogg"), (". opus")),
                                ),
                            ],
                        ],
                    )
                ],
                [
                    sg.Frame(
                        layout=[
                            [
                                sg.Text("Device type"),
                                sg.Combo(
                                    self.hostapis,
                                    key="sg_hostapi",
                                    default_value=data.get("sg_hostapi", ""),
                                    enable_events=True,
                                    size=(20, 1),
                                ),
                                sg.Checkbox(
                                    "WASAPI Exclusive Device",
                                    key="sg_wasapi_exclusive",
                                    default=data.get("sg_wasapi_exclusive", False),
                                    enable_events=True,
                                ),
                            ],
                            [
                                sg.Text("Input Device"),
                                sg.Combo(
                                    self.input_devices,
                                    key="sg_input_device",
                                    default_value=data.get("sg_input_device", ""),
                                    enable_events=True,
                                    size=(45, 1),
                                ),
                            ],
                            [
                                sg.Text("Output Device"),
                                sg.Combo(
                                    self.output_devices,
                                    key="sg_output_device",
                                    default_value=data.get("sg_output_device", ""),
                                    enable_events=True,
                                    size=(45, 1),
                                ),
                            ],
                            [
                                sg.Button("Reload devices", key="reload_devices"),
                                sg.Radio(
                                    "Use model sampling rate",
                                    "sr_type",
                                    key="sr_model",
                                    default=data.get("sr_model", True),
                                    enable_events=True,
                                ),
                                sg.Radio(
                                    "Use device sampling rate",
                                    "sr_type",
                                    key="sr_device",
                                    default=data.get("sr_device", False),
                                    enable_events=True,
                                ),
                                sg.Text("Sampling rate:"),
                                sg.Text("", key="sr_stream"),
                            ],
                        ],
                        title="Sound Device",
                    )
                ],
                [
                    sg.Frame(
                        layout=[
                            [
                                sg.Text("Activation threshold"),
                                sg.Slider(
                                    range=(-60, 0),
                                    key="threhold",
                                    resolution=1,
                                    orientation="h",
                                    default_value=data.get("threhold", -40),
                                    enable_events=True,
                                ),
                            ],
                            [
                                sg.Text("Diffusion steps"),
                                sg.Slider(
                                    range=(1, 30),
                                    key="diffusion_steps",
                                    resolution=1,
                                    orientation="h",
                                    default_value=data.get("diffusion_steps", 10),
                                    enable_events=True,
                                ),
                            ],
                            [
                                sg.Text("Inference cfg rate"),
                                sg.Slider(
                                    range=(0.0, 1.0),
                                    key="inference_cfg_rate",
                                    resolution=0.1,
                                    orientation="h",
                                    default_value=data.get("inference_cfg_rate", 0.7),
                                    enable_events=True,
                                ),
                            ],
                            [
                                sg.Text("Max prompt length (s)"),
                                sg.Slider(
                                    range=(1.0, 20.0),
                                    key="max_prompt_length",
                                    resolution=0.5,
                                    orientation="h",
                                    default_value=data.get("max_prompt_length", 3.0),
                                    enable_events=True,
                                ),
                            ],
                            [
                                sg.Text("Pitch shift (semitones)"),
                                sg.Slider(
                                    range=(-12, 12),
                                    key="pitch_shift",
                                    resolution=1,
                                    orientation="h",
                                    default_value=data.get("pitch_shift", 0),
                                    enable_events=True,
                                ),
                            ],
                        ],
                        title="Regular settings",
                    ),
                    sg.Frame(
                        layout=[
                            [
                                sg.Text("Block time"),
                                sg.Slider(
                                    range=(0.04, 3.0),
                                    key="block_time",
                                    resolution=0.02,
                                    orientation="h",
                                    default_value=data.get("block_time", 1.0),
                                    enable_events=True,
                                ),
                            ],
                            [
                                sg.Text("Crossfade length"),
                                sg.Slider(
                                    range=(0.02, 0.5),
                                    key="crossfade_length",
                                    resolution=0.02,
                                    orientation="h",
                                    default_value=data.get("crossfade_length", 0.1),
                                    enable_events=True,
                                ),
                            ],
                            [
                                sg.Text("Extra content encoder context time (left)"),
                                sg.Slider(
                                    range=(0.5, 10.0),
                                    key="extra_time_ce",
                                    resolution=0.1,
                                    orientation="h",
                                    default_value=data.get("extra_time_ce", 5.0),
                                    enable_events=True,
                                ),
                            ],
                            [
                                sg.Text("Extra DiT context time (left)"),
                                sg.Slider(
                                    range=(0.5, 10.0),
                                    key="extra_time",
                                    resolution=0.1,
                                    orientation="h",
                                    default_value=data.get("extra_time", 5.0),
                                    enable_events=True,
                                ),
                            ],
                            [
                                sg.Text("Extra context time (right)"),
                                sg.Slider(
                                    range=(0.02, 10.0),
                                    key="extra_time_right",
                                    resolution=0.02,
                                    orientation="h",
                                    default_value=data.get("extra_time_right", 2.0),
                                    enable_events=True,
                                ),
                            ],
                        ],
                        title="Performance settings",
                    ),
                ],
                [
                    sg.Button("Start Voice Conversion", key="start_vc"),
                    sg.Button("Stop Voice Conversion", key="stop_vc"),
                    sg.Radio(
                        "Input listening",
                        "function",
                        key="im",
                        default=False,
                        enable_events=True,
                    ),
                    sg.Radio(
                        "Voice Conversion",
                        "function",
                        key="vc",
                        default=True,
                        enable_events=True,
                    ),
                    sg.Text("Algorithm delay (ms):"),
                    sg.Text("0", key="delay_time"),
                    sg.Text("Inference time (ms):"),
                    sg.Text("0", key="infer_time"),
                ],
            ]
            self.window = sg.Window("Seed-VC - GUI", layout=layout, finalize=True)
            self.event_handler()

        def event_handler(self):
            global flag_vc
            while True:
                event, values = self.window.read()
                if event == sg.WINDOW_CLOSED:
                    self.stop_stream()
                    exit()
                if event == "reload_devices" or event == "sg_hostapi":
                    self.gui_config.sg_hostapi = values["sg_hostapi"]
                    self.update_devices(hostapi_name=values["sg_hostapi"])
                    if self.gui_config.sg_hostapi not in self.hostapis:
                        self.gui_config.sg_hostapi = self.hostapis[0]
                    self.window["sg_hostapi"].Update(values=self.hostapis)
                    self.window["sg_hostapi"].Update(value=self.gui_config.sg_hostapi)
                    if (
                        self.gui_config.sg_input_device not in self.input_devices
                        and len(self.input_devices) > 0
                    ):
                        self.gui_config.sg_input_device = self.input_devices[0]
                    self.window["sg_input_device"].Update(values=self.input_devices)
                    self.window["sg_input_device"].Update(
                        value=self.gui_config.sg_input_device
                    )
                    if self.gui_config.sg_output_device not in self.output_devices:
                        self.gui_config.sg_output_device = self.output_devices[0]
                    self.window["sg_output_device"].Update(values=self.output_devices)
                    self.window["sg_output_device"].Update(
                        value=self.gui_config.sg_output_device
                    )
                if event == "start_vc" and not flag_vc:
                    if self.set_values(values) == True:
                        printt("cuda_is_available: %s", torch.cuda.is_available())
                        self.start_vc()
                        settings = {
                            "reference_audio_path": values["reference_audio_path"],
                            # "index_path": values["index_path"],
                            "sg_hostapi": values["sg_hostapi"],
                            "sg_wasapi_exclusive": values["sg_wasapi_exclusive"],
                            "sg_input_device": values["sg_input_device"],
                            "sg_output_device": values["sg_output_device"],
                            "sr_type": ["sr_model", "sr_device"][
                                [
                                    values["sr_model"],
                                    values["sr_device"],
                                ].index(True)
                            ],
                            "threhold": values["threhold"],
                            "diffusion_steps": values["diffusion_steps"],
                            "inference_cfg_rate": values["inference_cfg_rate"],
                            "max_prompt_length": values["max_prompt_length"],
                            "block_time": values["block_time"],
                            "crossfade_length": values["crossfade_length"],
                            "extra_time_ce": values["extra_time_ce"],
                            "extra_time": values["extra_time"],
                            "extra_time_right": values["extra_time_right"],
                            "pitch_shift": values["pitch_shift"],
                        }
                        with open("configs/inuse/config.json", "w") as j:
                            json.dump(settings, j)
                        if self.stream is not None:
                            self.delay_time = (
                                self.stream.latency[-1]
                                + values["block_time"]
                                + values["crossfade_length"]
                                + values["extra_time_right"]
                                + 0.01
                            )
                        self.window["sr_stream"].update(self.gui_config.samplerate)
                        self.window["delay_time"].update(
                            int(np.round(self.delay_time * 1000))
                        )
                # Parameter hot update
                if event == "threhold":
                    self.gui_config.threhold = values["threhold"]
                elif event == "diffusion_steps":
                    self.gui_config.diffusion_steps = values["diffusion_steps"]
                elif event == "inference_cfg_rate":
                    self.gui_config.inference_cfg_rate = values["inference_cfg_rate"]
                elif event in ["vc", "im"]:
                    self.function = event
                elif event == "stop_vc" or event != "start_vc":
                    # Other parameters do not support hot update
                    self.stop_stream()
                elif event == "pitch_shift":
                    self.gui_config.pitch_shift = int(values["pitch_shift"])

        def set_values(self, values):
            if len(values["reference_audio_path"].strip()) == 0:
                sg.popup("Choose an audio file")
                return False
            pattern = re.compile("[^\x00-\x7F]+")
            if pattern.findall(values["reference_audio_path"]):
                sg.popup("audio file path contains non-ascii characters")
                return False
            self.set_devices(values["sg_input_device"], values["sg_output_device"])
            self.gui_config.sg_hostapi = values["sg_hostapi"]
            self.gui_config.sg_wasapi_exclusive = values["sg_wasapi_exclusive"]
            self.gui_config.sg_input_device = values["sg_input_device"]
            self.gui_config.sg_output_device = values["sg_output_device"]
            self.gui_config.reference_audio_path = values["reference_audio_path"]
            self.gui_config.sr_type = ["sr_model", "sr_device"][
                [
                    values["sr_model"],
                    values["sr_device"],
                ].index(True)
            ]
            # self.gui_config.threhold = values["threhold"]
            self.gui_config.diffusion_steps = values["diffusion_steps"]
            self.gui_config.inference_cfg_rate = values["inference_cfg_rate"]
            self.gui_config.max_prompt_length = values["max_prompt_length"]
            self.gui_config.block_time = values["block_time"]
            self.gui_config.crossfade_time = values["crossfade_length"]
            self.gui_config.extra_time_ce = values["extra_time_ce"]
            self.gui_config.extra_time = values["extra_time"]
            self.gui_config.extra_time_right = values["extra_time_right"]
            self.gui_config.pitch_shift = values["pitch_shift"]
            return True

        def start_vc(self):
            # 清空 CUDA 缓存
            torch.cuda.empty_cache()
            
            # 加载参考音频
            self.reference_wav, _ = librosa.load(
                self.gui_config.reference_audio_path, sr=self.model_set[-1]["sampling_rate"]
            )
            
            # 设置采样率 - 使用模型采样率或设备采样率
            self.gui_config.samplerate = (
                self.model_set[-1]["sampling_rate"]
                if self.gui_config.sr_type == "sr_model"
                else self.get_device_samplerate()
            )
            
            # 获取音频通道数
            self.gui_config.channels = self.get_device_channels()
            
            # 计算基本帧长度 - 采样率/50作为基本单位
            self.zc = self.gui_config.samplerate // 50  # 例如44100 // 50 = 882
            
            # 计算音频块的帧数
            self.block_frame = (
                int(
                    np.round(
                        self.gui_config.block_time
                        * self.gui_config.samplerate
                        / self.zc
                    )
                )
                * self.zc
            )
            
            # 计算16k采样率下的块帧数 16000 / 50 = 320
            self.block_frame_16k = 320 * self.block_frame // self.zc
            
            # 计算交叉淡入淡出的帧数
            self.crossfade_frame = (
                int(
                    np.round(
                        self.gui_config.crossfade_time
                        * self.gui_config.samplerate
                        / self.zc
                    )
                )
                * self.zc
            )
            
            # 设置SOLA算法的缓冲区大小
            self.sola_buffer_frame = min(self.crossfade_frame, 4 * self.zc)
            self.sola_search_frame = self.zc
            
            # 计算内容编码器额外上下文的帧数
            self.extra_frame = (
                int(
                    np.round(
                        self.gui_config.extra_time_ce
                        * self.gui_config.samplerate
                        / self.zc
                    )
                )
                * self.zc
            )
            
            # 计算右侧额外上下文的帧数
            self.extra_frame_right = (
                    int(
                        np.round(
                            self.gui_config.extra_time_right
                            * self.gui_config.samplerate
                            / self.zc
                        )
                    )
                    * self.zc
            )
            
            # 初始化输入音频缓冲区 2 * 44100 + 0.08 * 44100 + 0.01 * 44100 + 0.25 * 44100
            self.input_wav: torch.Tensor = torch.zeros(
                self.extra_frame
                + self.crossfade_frame
                + self.sola_search_frame
                + self.block_frame
                + self.extra_frame_right,
                device=self.config.device,
                dtype=torch.float32,
            )
            # 缓冲区大小
            print(f"缓存区大小：{self.input_wav.shape[0]/self.gui_config.samplerate}秒")
            
            # 初始化去噪后的输入音频缓冲区
            self.input_wav_denoise: torch.Tensor = self.input_wav.clone()
            
            # 初始化重采样后的输入音频缓冲区(16kHz)
            self.input_wav_res: torch.Tensor = torch.zeros(
                320 * self.input_wav.shape[0] // self.zc,
                device=self.config.device,
                dtype=torch.float32,
            )
            
            # 初始化RMS计算用的缓冲区
            self.rms_buffer: np.ndarray = np.zeros(4 * self.zc, dtype="float32")
            
            # 初始化SOLA算法的缓冲区
            self.sola_buffer: torch.Tensor = torch.zeros(
                self.sola_buffer_frame, device=self.config.device, dtype=torch.float32
            )
            
            # 初始化降噪缓冲区
            self.nr_buffer: torch.Tensor = self.sola_buffer.clone()
            
            # 初始化输出缓冲区
            self.output_buffer: torch.Tensor = self.input_wav.clone()
            
            # 设置跳过的头尾帧数
            self.skip_head = self.extra_frame // self.zc
            self.skip_tail = self.extra_frame_right // self.zc
            
            # 计算返回长度
            self.return_length = (
                self.block_frame + self.sola_buffer_frame + self.sola_search_frame
            ) // self.zc
            
            # 创建淡入淡出窗口
            self.fade_in_window: torch.Tensor = (
                torch.sin(
                    0.5
                    * np.pi
                    * torch.linspace(
                        0.0,
                        1.0,
                        steps=self.sola_buffer_frame,
                        device=self.config.device,
                        dtype=torch.float32,
                    )
                )
                ** 2
            )
            self.fade_out_window: torch.Tensor = 1 - self.fade_in_window
            
            # 初始化重采样器(原始采样率到16kHz)
            self.resampler = tat.Resample(
                orig_freq=self.gui_config.samplerate,
                new_freq=16000,
                dtype=torch.float32,
            ).to(self.config.device)
            
            # 如果需要，初始化第二个重采样器(模型采样率到设备采样率)
            print(f"model_set[-1]['sampling_rate']: {self.model_set[-1]['sampling_rate']}, gui_config.samplerate: {self.gui_config.samplerate}")
            if self.model_set[-1]["sampling_rate"] != self.gui_config.samplerate:
                self.resampler2 = tat.Resample(
                    orig_freq=self.model_set[-1]["sampling_rate"],
                    new_freq=self.gui_config.samplerate,
                    dtype=torch.float32,
                ).to(self.config.device)
            else:
                self.resampler2 = None
            
            # 初始化VAD(语音活动检测)相关变量
            self.vad_cache = {}
            self.vad_chunk_size = min(500, 1000 * self.gui_config.block_time)
            self.vad_speech_detected = False
            self.set_speech_detected_false_at_end_flag = False
            
            # 启动音频流
            self.start_stream()

        def start_stream(self):
            global flag_vc
            if not flag_vc:
                flag_vc = True
                if (
                    "WASAPI" in self.gui_config.sg_hostapi
                    and self.gui_config.sg_wasapi_exclusive
                ):
                    extra_settings = sd.WasapiSettings(exclusive=True)
                else:
                    extra_settings = None
                '''
                sounddevice 从麦克风读取音频数据，放入 indata
                audio_callback 处理这些数据（进行语音转换）
                处理后的音频数据被写入 outdata
                sounddevice 将 outdata 中的数据发送到声卡播放
                '''
                self.stream = sd.Stream(
                    callback=self.audio_callback,
                    blocksize=self.block_frame,
                    samplerate=self.gui_config.samplerate,
                    channels=self.gui_config.channels,
                    dtype="float32",
                    extra_settings=extra_settings,
                )
                self.stream.start()

        def stop_stream(self):
            global flag_vc
            if flag_vc:
                flag_vc = False
                if self.stream is not None:
                    self.stream.abort()
                    self.stream.close()
                    self.stream = None

        def audio_callback(
            self, indata: np.ndarray, outdata: np.ndarray, frames, times, status
        ):
            """
            音频块回调函数，处理实时音频流
            参数:
                indata: 输入音频数据，形状为 (frames, channels)
                outdata: 输出音频缓冲区，需要填充处理后的音频
                frames: 当前音频块的帧数
                times: 音频时间戳信息
                status: 音频流状态信息
            Audio block callback function
            硬件层 (声卡) 
            ↓
            PortAudio (底层音频库)
            ↓
            sounddevice (Python包装器)
            ↓
            生成 frames, times, status
            ↓
            传递给 audio_callback
            """
            global flag_vc
            print(indata.shape)
            start_time = time.perf_counter()  # 记录处理开始时间
            indata = librosa.to_mono(indata.T)  # 将多通道音频转换为单声道
            # 语音活动检测(VAD)
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start_event.record()
            
            # 将音频重采样到16kHz用于VAD
            indata_16k = librosa.resample(indata, orig_sr=self.gui_config.samplerate, target_sr=16000)
            
            # 使用VAD模型检测语音
            res = self.vad_model.generate(input=indata_16k, cache=self.vad_cache, is_final=False, chunk_size=self.vad_chunk_size)
            res_value = res[0]["value"]
            print(res_value)
            if len(res_value) % 2 == 1 and not self.vad_speech_detected:
                self.vad_speech_detected = True
            elif len(res_value) % 2 == 1 and self.vad_speech_detected:
                self.set_speech_detected_false_at_end_flag = True
            
            end_event.record()
            torch.cuda.synchronize()  # Wait for the events to be recorded!
            elapsed_time_ms = start_event.elapsed_time(end_event)
            print(f"Time taken for VAD: {elapsed_time_ms}ms")

            # 音量阈值处理
            if self.gui_config.threhold > -60:
                # 添加历史缓冲以计算RMS
                indata = np.append(self.rms_buffer, indata)
                # 计算音频的RMS能量
                rms = librosa.feature.rms(
                    y=indata, frame_length=4 * self.zc, hop_length=self.zc
                )[:, 2:]
                # 更新RMS缓冲区
                self.rms_buffer[:] = indata[-4 * self.zc :]
                indata = indata[2 * self.zc - self.zc // 2 :]
                # 将RMS转换为分贝并与阈值比较
                db_threhold = (
                    librosa.amplitude_to_db(rms, ref=1.0)[0] < self.gui_config.threhold
                )
                # 将低于阈值的部分置零
                for i in range(db_threhold.shape[0]):
                    if db_threhold[i]:
                        indata[i * self.zc : (i + 1) * self.zc] = 0
                indata = indata[self.zc // 2 :]
            # 将self.block_frame和 indata.shape[0] 打印出来
            print(f"self.block_frame: {self.block_frame}, indata.shape[0]: {indata.shape}")
            print(f"self.input_wav.shape: {self.input_wav.shape}")
            
            # 更新输入音频缓冲区
            self.input_wav[: -self.block_frame] = self.input_wav[
                self.block_frame :
            ].clone()
            self.input_wav[-indata.shape[0] :] = torch.from_numpy(indata).to(
                self.config.device
            )
            # 保存self.input_wav
            # sf.write("self.input_wav.wav", self.input_wav.cpu().numpy(), self.gui_config.samplerate)
            
            # 更新重采样后的输入缓冲区
            # self.input_wav_res[: -self.block_frame_16k] = self.input_wav_res[
            #     self.block_frame_16k :
            # ].clone()
            # self.input_wav_res[-320 * (indata.shape[0] // self.zc + 1) :] = (
            #     self.resampler(self.input_wav[-indata.shape[0] - 2 * self.zc :])[
            #         320:
            #     ]
            # 替换为使用 librosa 的重采样：
            self.input_wav_res = torch.from_numpy(
                librosa.resample(
                    self.input_wav.cpu().numpy(),
                    orig_sr=self.gui_config.samplerate,
                    target_sr=16000
                )
            ).to(self.config.device)
            print(f"preprocess time: {time.perf_counter() - start_time:.2f}")
            # infer
            if self.function == "vc": 
                if self.gui_config.extra_time_ce - self.gui_config.extra_time < 0:
                    raise ValueError("Content encoder extra context must be greater than DiT extra context!")
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize()
                start_event.record()
                infer_wav = custom_infer(
                    self.model_set,
                    self.reference_wav,
                    self.gui_config.reference_audio_path,
                    self.input_wav_res,
                    self.block_frame_16k,
                    self.skip_head,
                    self.skip_tail,
                    self.return_length,                          
                    int(self.gui_config.diffusion_steps),
                    self.gui_config.inference_cfg_rate,
                    self.gui_config.max_prompt_length,
                    self.gui_config.extra_time_ce - self.gui_config.extra_time,
                    self.gui_config.pitch_shift,
                )
                if self.resampler2 is not None:
                    infer_wav = self.resampler2(infer_wav)
                end_event.record()
                torch.cuda.synchronize()  # Wait for the events to be recorded!
                elapsed_time_ms = start_event.elapsed_time(end_event)
                print(f"Time taken for VC: {elapsed_time_ms}ms")
                if not self.vad_speech_detected:
                    infer_wav = torch.zeros_like(self.input_wav[self.extra_frame :])
            elif self.gui_config.I_noise_reduce:
                infer_wav = self.input_wav_denoise[self.extra_frame :].clone()
            else:
                infer_wav = self.input_wav[self.extra_frame :].clone()
            
            # SOLA algorithm from https://github.com/yxlllc/DDSP-SVC
            conv_input = infer_wav[
                None, None, : self.sola_buffer_frame + self.sola_search_frame
            ]

            cor_nom = F.conv1d(conv_input.float(), self.sola_buffer[None, None, :])
            cor_den = torch.sqrt(
                F.conv1d(
                    (conv_input**2).float(),
                    torch.ones(1, 1, self.sola_buffer_frame, device=self.config.device),
                )
                + 1e-8
            )
            if sys.platform == "darwin":
                _, sola_offset = torch.max(cor_nom[0, 0] / cor_den[0, 0])
                sola_offset = sola_offset.item()
            else:

                sola_offset = torch.argmax(cor_nom[0, 0] / cor_den[0, 0])

            print(f"sola_offset = {int(sola_offset)}")

            # post_process_start = time.perf_counter()
            infer_wav = infer_wav[sola_offset:]
            infer_wav[: self.sola_buffer_frame] *= self.fade_in_window
            infer_wav[: self.sola_buffer_frame] += (
                self.sola_buffer * self.fade_out_window
            )
            self.sola_buffer[:] = infer_wav[
                self.block_frame : self.block_frame + self.sola_buffer_frame
            ]
            outdata[:] = (
                infer_wav[: self.block_frame]
                .repeat(self.gui_config.channels, 1)
                .t()
                .cpu()
                .numpy()
            )

            total_time = time.perf_counter() - start_time
            if flag_vc:
                self.window["infer_time"].update(int(total_time * 1000))

            if self.set_speech_detected_false_at_end_flag:
                self.vad_speech_detected = False
                self.set_speech_detected_false_at_end_flag = False

            print(f"Infer time: {total_time:.2f}")

        def update_devices(self, hostapi_name=None):
            """Get input and output devices."""
            global flag_vc
            flag_vc = False
            sd._terminate()
            sd._initialize()
            devices = sd.query_devices()
            hostapis = sd.query_hostapis()
            for hostapi in hostapis:
                for device_idx in hostapi["devices"]:
                    devices[device_idx]["hostapi_name"] = hostapi["name"]
            self.hostapis = [hostapi["name"] for hostapi in hostapis]
            if hostapi_name not in self.hostapis:
                hostapi_name = self.hostapis[0]
            self.input_devices = [
                d["name"]
                for d in devices
                if d["max_input_channels"] > 0 and d["hostapi_name"] == hostapi_name
            ]
            self.output_devices = [
                d["name"]
                for d in devices
                if d["max_output_channels"] > 0 and d["hostapi_name"] == hostapi_name
            ]
            self.input_devices_indices = [
                d["index"] if "index" in d else d["name"]
                for d in devices
                if d["max_input_channels"] > 0 and d["hostapi_name"] == hostapi_name
            ]
            self.output_devices_indices = [
                d["index"] if "index" in d else d["name"]
                for d in devices
                if d["max_output_channels"] > 0 and d["hostapi_name"] == hostapi_name
            ]

        def set_devices(self, input_device, output_device):
            """set input and output devices."""
            sd.default.device[0] = self.input_devices_indices[
                self.input_devices.index(input_device)
            ]
            sd.default.device[1] = self.output_devices_indices[
                self.output_devices.index(output_device)
            ]
            printt("Input device: %s:%s", str(sd.default.device[0]), input_device)
            printt("Output device: %s:%s", str(sd.default.device[1]), output_device)

        def get_device_samplerate(self):
            return int(
                sd.query_devices(device=sd.default.device[0])["default_samplerate"]
            )

        def get_device_channels(self):
            max_input_channels = sd.query_devices(device=sd.default.device[0])[
                "max_input_channels"
            ]
            max_output_channels = sd.query_devices(device=sd.default.device[1])[
                "max_output_channels"
            ]
            return min(max_input_channels, max_output_channels, 2)


    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-path", type=str, default='runs/real_time_bachong/ft_model.pth', help="Path to the model checkpoint")
    parser.add_argument("--config-path", type=str, default='runs/real_time_bachong/config_dit_mel_seed_uvit_xlsr_tiny.yml', help="Path to the vocoder checkpoint")
    parser.add_argument("--fp16", type=str2bool, nargs="?", const=True, help="Whether to use fp16", default=True)
    parser.add_argument("--gpu", type=int, help="Which GPU id to use", default=0)
    args = parser.parse_args()
    cuda_target = f"cuda:{args.gpu}" if args.gpu else "cuda" 
    device = torch.device(cuda_target if torch.cuda.is_available() else "cpu")
    gui = GUI(args)