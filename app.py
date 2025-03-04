import gradio as gr
import torch
import torchaudio
import librosa
from modules.commons import build_model, load_checkpoint, recursive_munch
import yaml
from hf_utils import load_custom_model_from_hf
import numpy as np
from pydub import AudioSegment

# Load model and configuration

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

dit_checkpoint_path, dit_config_path = load_custom_model_from_hf("Plachta/Seed-VC",
                                                "DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth",
                                                "config_dit_mel_seed_uvit_whisper_small_wavenet.yml")
config = yaml.safe_load(open(dit_config_path, 'r'))
model_params = recursive_munch(config['model_params'])
model = build_model(model_params, stage='DiT')
hop_length = config['preprocess_params']['spect_params']['hop_length']
sr = config['preprocess_params']['sr']

# Load checkpoints
model, _, _, _ = load_checkpoint(model, None, dit_checkpoint_path,
                                 load_only_params=True, ignore_modules=[], is_distributed=False)
for key in model:
    model[key].eval()
    model[key].to(device)
model.cfm.estimator.setup_caches(max_batch_size=1, max_seq_length=8192)

# Load additional modules
from modules.campplus.DTDNN import CAMPPlus

campplus_ckpt_path = load_custom_model_from_hf("funasr/campplus", "campplus_cn_common.bin", config_filename=None)
campplus_model = CAMPPlus(feat_dim=80, embedding_size=192)
campplus_model.load_state_dict(torch.load(campplus_ckpt_path, map_location="cpu"))
campplus_model.eval()
campplus_model.to(device)

from modules.bigvgan import bigvgan

bigvgan_model = bigvgan.BigVGAN.from_pretrained('nvidia/bigvgan_v2_22khz_80band_256x', use_cuda_kernel=False)

# remove weight norm in the model and set to eval mode
bigvgan_model.remove_weight_norm()
bigvgan_model = bigvgan_model.eval().to(device)

# whisper
from transformers import AutoFeatureExtractor, WhisperModel

whisper_name = model_params.speech_tokenizer.whisper_name if hasattr(model_params.speech_tokenizer,
                                                                     'whisper_name') else "openai/whisper-small"
whisper_model = WhisperModel.from_pretrained(whisper_name, torch_dtype=torch.float16).to(device)
del whisper_model.decoder
whisper_feature_extractor = AutoFeatureExtractor.from_pretrained(whisper_name)

# Generate mel spectrograms
mel_fn_args = {
    "n_fft": config['preprocess_params']['spect_params']['n_fft'],
    "win_size": config['preprocess_params']['spect_params']['win_length'],
    "hop_size": config['preprocess_params']['spect_params']['hop_length'],
    "num_mels": config['preprocess_params']['spect_params']['n_mels'],
    "sampling_rate": sr,
    "fmin": 0,
    "fmax": None,
    "center": False
}
from modules.audio import mel_spectrogram

to_mel = lambda x: mel_spectrogram(x, **mel_fn_args)

# f0 conditioned model
dit_checkpoint_path, dit_config_path = load_custom_model_from_hf("Plachta/Seed-VC",
                                                "DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema.pth",
                                                "config_dit_mel_seed_uvit_whisper_base_f0_44k.yml")

config = yaml.safe_load(open(dit_config_path, 'r'))
model_params = recursive_munch(config['model_params'])
model_f0 = build_model(model_params, stage='DiT')
hop_length = config['preprocess_params']['spect_params']['hop_length']
sr = config['preprocess_params']['sr']

# Load checkpoints
model_f0, _, _, _ = load_checkpoint(model_f0, None, dit_checkpoint_path,
                                 load_only_params=True, ignore_modules=[], is_distributed=False)
for key in model_f0:
    model_f0[key].eval()
    model_f0[key].to(device)
model_f0.cfm.estimator.setup_caches(max_batch_size=1, max_seq_length=8192)

# f0 extractor
from modules.rmvpe import RMVPE

model_path = load_custom_model_from_hf("lj1995/VoiceConversionWebUI", "rmvpe.pt", None)
rmvpe = RMVPE(model_path, is_half=False, device=device)

mel_fn_args_f0 = {
    "n_fft": config['preprocess_params']['spect_params']['n_fft'],
    "win_size": config['preprocess_params']['spect_params']['win_length'],
    "hop_size": config['preprocess_params']['spect_params']['hop_length'],
    "num_mels": config['preprocess_params']['spect_params']['n_mels'],
    "sampling_rate": sr,
    "fmin": 0,
    "fmax": None,
    "center": False
}
to_mel_f0 = lambda x: mel_spectrogram(x, **mel_fn_args_f0)
bigvgan_44k_model = bigvgan.BigVGAN.from_pretrained('nvidia/bigvgan_v2_44khz_128band_512x', use_cuda_kernel=False)

# remove weight norm in the model and set to eval mode
bigvgan_44k_model.remove_weight_norm()
bigvgan_44k_model = bigvgan_44k_model.eval().to(device)

def adjust_f0_semitones(f0_sequence, n_semitones):
    factor = 2 ** (n_semitones / 12)
    return f0_sequence * factor

def crossfade(chunk1, chunk2, overlap):
    fade_out = np.cos(np.linspace(0, np.pi / 2, overlap)) ** 2
    fade_in = np.cos(np.linspace(np.pi / 2, 0, overlap)) ** 2
    if len(chunk2) < overlap:
        chunk2[:overlap] = chunk2[:overlap] * fade_in[:len(chunk2)] + (chunk1[-overlap:] * fade_out)[:len(chunk2)]
    else:
        chunk2[:overlap] = chunk2[:overlap] * fade_in + chunk1[-overlap:] * fade_out
    return chunk2

# streaming and chunk processing related params
overlap_frame_len = 16
bitrate = "320k"

@torch.no_grad()
@torch.inference_mode()
def voice_conversion(source, target, diffusion_steps, length_adjust, inference_cfg_rate, f0_condition, auto_f0_adjust, pitch_shift):
    inference_module = model if not f0_condition else model_f0
    mel_fn = to_mel if not f0_condition else to_mel_f0
    bigvgan_fn = bigvgan_model if not f0_condition else bigvgan_44k_model
    sr = 22050 if not f0_condition else 44100
    hop_length = 256 if not f0_condition else 512
    max_context_window = sr // hop_length * 30
    overlap_wave_len = overlap_frame_len * hop_length
    # Load audio
    source_audio = librosa.load(source, sr=sr)[0]
    ref_audio = librosa.load(target, sr=sr)[0]

    # Process audio
    source_audio = torch.tensor(source_audio).unsqueeze(0).float().to(device)
    ref_audio = torch.tensor(ref_audio[:sr * 25]).unsqueeze(0).float().to(device)

    # Resample
    ref_waves_16k = torchaudio.functional.resample(ref_audio, sr, 16000)
    converted_waves_16k = torchaudio.functional.resample(source_audio, sr, 16000)
    # if source audio less than 30 seconds, whisper can handle in one forward
    if converted_waves_16k.size(-1) <= 16000 * 30:
        alt_inputs = whisper_feature_extractor([converted_waves_16k.squeeze(0).cpu().numpy()],
                                               return_tensors="pt",
                                               return_attention_mask=True,
                                               sampling_rate=16000)
        alt_input_features = whisper_model._mask_input_features(
            alt_inputs.input_features, attention_mask=alt_inputs.attention_mask).to(device)
        alt_outputs = whisper_model.encoder(
            alt_input_features.to(whisper_model.encoder.dtype),
            head_mask=None,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        S_alt = alt_outputs.last_hidden_state.to(torch.float32)
        S_alt = S_alt[:, :converted_waves_16k.size(-1) // 320 + 1]
    else:
        overlapping_time = 5  # 5 seconds
        S_alt_list = []
        buffer = None
        traversed_time = 0
        while traversed_time < converted_waves_16k.size(-1):
            if buffer is None:  # first chunk
                chunk = converted_waves_16k[:, traversed_time:traversed_time + 16000 * 30]
            else:
                chunk = torch.cat([buffer, converted_waves_16k[:, traversed_time:traversed_time + 16000 * (30 - overlapping_time)]], dim=-1)
            alt_inputs = whisper_feature_extractor([chunk.squeeze(0).cpu().numpy()],
                                                   return_tensors="pt",
                                                   return_attention_mask=True,
                                                   sampling_rate=16000)
            alt_input_features = whisper_model._mask_input_features(
                alt_inputs.input_features, attention_mask=alt_inputs.attention_mask).to(device)
            alt_outputs = whisper_model.encoder(
                alt_input_features.to(whisper_model.encoder.dtype),
                head_mask=None,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
            )
            S_alt = alt_outputs.last_hidden_state.to(torch.float32)
            S_alt = S_alt[:, :chunk.size(-1) // 320 + 1]
            if traversed_time == 0:
                S_alt_list.append(S_alt)
            else:
                S_alt_list.append(S_alt[:, 50 * overlapping_time:])
            buffer = chunk[:, -16000 * overlapping_time:]
            traversed_time += 30 * 16000 if traversed_time == 0 else chunk.size(-1) - 16000 * overlapping_time
        S_alt = torch.cat(S_alt_list, dim=1)

    ori_waves_16k = torchaudio.functional.resample(ref_audio, sr, 16000)
    ori_inputs = whisper_feature_extractor([ori_waves_16k.squeeze(0).cpu().numpy()],
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
    S_ori = S_ori[:, :ori_waves_16k.size(-1) // 320 + 1]

    mel = mel_fn(source_audio.to(device).float())
    mel2 = mel_fn(ref_audio.to(device).float())

    target_lengths = torch.LongTensor([int(mel.size(2) * length_adjust)]).to(mel.device)
    target2_lengths = torch.LongTensor([mel2.size(2)]).to(mel2.device)

    feat2 = torchaudio.compliance.kaldi.fbank(ref_waves_16k,
                                              num_mel_bins=80,
                                              dither=0,
                                              sample_frequency=16000)
    feat2 = feat2 - feat2.mean(dim=0, keepdim=True)
    style2 = campplus_model(feat2.unsqueeze(0))

    if f0_condition:
        F0_ori = rmvpe.infer_from_audio(ref_waves_16k[0], thred=0.03)
        F0_alt = rmvpe.infer_from_audio(converted_waves_16k[0], thred=0.03)

        if device == "mps":
            F0_ori = torch.from_numpy(F0_ori).float().to(device)[None]
            F0_alt = torch.from_numpy(F0_alt).float().to(device)[None]
        else:
            F0_ori = torch.from_numpy(F0_ori).to(device)[None]
            F0_alt = torch.from_numpy(F0_alt).to(device)[None]

        voiced_F0_ori = F0_ori[F0_ori > 1]
        voiced_F0_alt = F0_alt[F0_alt > 1]

        log_f0_alt = torch.log(F0_alt + 1e-5)
        voiced_log_f0_ori = torch.log(voiced_F0_ori + 1e-5)
        voiced_log_f0_alt = torch.log(voiced_F0_alt + 1e-5)
        median_log_f0_ori = torch.median(voiced_log_f0_ori)
        median_log_f0_alt = torch.median(voiced_log_f0_alt)

        # shift alt log f0 level to ori log f0 level
        shifted_log_f0_alt = log_f0_alt.clone()
        if auto_f0_adjust:
            shifted_log_f0_alt[F0_alt > 1] = log_f0_alt[F0_alt > 1] - median_log_f0_alt + median_log_f0_ori
        shifted_f0_alt = torch.exp(shifted_log_f0_alt)
        if pitch_shift != 0:
            shifted_f0_alt[F0_alt > 1] = adjust_f0_semitones(shifted_f0_alt[F0_alt > 1], pitch_shift)
    else:
        F0_ori = None
        F0_alt = None
        shifted_f0_alt = None

    # Length regulation
    cond, _, codes, commitment_loss, codebook_loss = inference_module.length_regulator(S_alt, ylens=target_lengths, n_quantizers=3, f0=shifted_f0_alt)
    prompt_condition, _, codes, commitment_loss, codebook_loss = inference_module.length_regulator(S_ori, ylens=target2_lengths, n_quantizers=3, f0=F0_ori)

    max_source_window = max_context_window - mel2.size(2)
    # split source condition (cond) into chunks
    processed_frames = 0
    generated_wave_chunks = []
    # generate chunk by chunk and stream the output
    while processed_frames < cond.size(1):
        chunk_cond = cond[:, processed_frames:processed_frames + max_source_window]
        is_last_chunk = processed_frames + max_source_window >= cond.size(1)
        cat_condition = torch.cat([prompt_condition, chunk_cond], dim=1)
        with torch.autocast(device_type=device.type, dtype=torch.float16):
            # Voice Conversion
            vc_target = inference_module.cfm.inference(cat_condition,
                                                       torch.LongTensor([cat_condition.size(1)]).to(mel2.device),
                                                       mel2, style2, None, diffusion_steps,
                                                       inference_cfg_rate=inference_cfg_rate)
            vc_target = vc_target[:, :, mel2.size(-1):]
        vc_wave = bigvgan_fn(vc_target.float())[0]
        if processed_frames == 0:
            if is_last_chunk:
                output_wave = vc_wave[0].cpu().numpy()
                generated_wave_chunks.append(output_wave)
                output_wave = (output_wave * 32768.0).astype(np.int16)
                mp3_bytes = AudioSegment(
                    output_wave.tobytes(), frame_rate=sr,
                    sample_width=output_wave.dtype.itemsize, channels=1
                ).export(format="mp3", bitrate=bitrate).read()
                yield mp3_bytes, (sr, np.concatenate(generated_wave_chunks))
                break
            output_wave = vc_wave[0, :-overlap_wave_len].cpu().numpy()
            generated_wave_chunks.append(output_wave)
            previous_chunk = vc_wave[0, -overlap_wave_len:]
            processed_frames += vc_target.size(2) - overlap_frame_len
            output_wave = (output_wave * 32768.0).astype(np.int16)
            mp3_bytes = AudioSegment(
                output_wave.tobytes(), frame_rate=sr,
                sample_width=output_wave.dtype.itemsize, channels=1
            ).export(format="mp3", bitrate=bitrate).read()
            yield mp3_bytes, None
        elif is_last_chunk:
            output_wave = crossfade(previous_chunk.cpu().numpy(), vc_wave[0].cpu().numpy(), overlap_wave_len)
            generated_wave_chunks.append(output_wave)
            processed_frames += vc_target.size(2) - overlap_frame_len
            output_wave = (output_wave * 32768.0).astype(np.int16)
            mp3_bytes = AudioSegment(
                output_wave.tobytes(), frame_rate=sr,
                sample_width=output_wave.dtype.itemsize, channels=1
            ).export(format="mp3", bitrate=bitrate).read()
            yield mp3_bytes, (sr, np.concatenate(generated_wave_chunks))
            break
        else:
            output_wave = crossfade(previous_chunk.cpu().numpy(), vc_wave[0, :-overlap_wave_len].cpu().numpy(), overlap_wave_len)
            generated_wave_chunks.append(output_wave)
            previous_chunk = vc_wave[0, -overlap_wave_len:]
            processed_frames += vc_target.size(2) - overlap_frame_len
            output_wave = (output_wave * 32768.0).astype(np.int16)
            mp3_bytes = AudioSegment(
                output_wave.tobytes(), frame_rate=sr,
                sample_width=output_wave.dtype.itemsize, channels=1
            ).export(format="mp3", bitrate=bitrate).read()
            yield mp3_bytes, None


if __name__ == "__main__":
    description = ("Zero-shot voice conversion with in-context learning. For local deployment please check [GitHub repository](https://github.com/Plachtaa/seed-vc) "
                   "for details and updates.<br>Note that any reference audio will be forcefully clipped to 25s if beyond this length.<br> "
                   "If total duration of source and reference audio exceeds 30s, source audio will be processed in chunks.<br> "
                   "无需训练的 zero-shot 语音/歌声转换模型，若需本地部署查看[GitHub页面](https://github.com/Plachtaa/seed-vc)<br>"
                   "请注意，参考音频若超过 25 秒，则会被自动裁剪至此长度。<br>若源音频和参考音频的总时长超过 30 秒，源音频将被分段处理。")
    inputs = [
        gr.Audio(type="filepath", label="Source Audio / 源音频"),
        gr.Audio(type="filepath", label="Reference Audio / 参考音频"),
        gr.Slider(minimum=1, maximum=200, value=10, step=1, label="Diffusion Steps / 扩散步数", info="10 by default, 50~100 for best quality / 默认为 10，50~100 为最佳质量"),
        gr.Slider(minimum=0.5, maximum=2.0, step=0.1, value=1.0, label="Length Adjust / 长度调整", info="<1.0 for speed-up speech, >1.0 for slow-down speech / <1.0 加速语速，>1.0 减慢语速"),
        gr.Slider(minimum=0.0, maximum=1.0, step=0.1, value=0.7, label="Inference CFG Rate", info="has subtle influence / 有微小影响"),
        gr.Checkbox(label="Use F0 conditioned model / 启用F0输入", value=False, info="Must set to true for singing voice conversion / 歌声转换时必须勾选"),
        gr.Checkbox(label="Auto F0 adjust / 自动F0调整", value=True,
                    info="Roughly adjust F0 to match target voice. Only works when F0 conditioned model is used. / 粗略调整 F0 以匹配目标音色，仅在勾选 '启用F0输入' 时生效"),
        gr.Slider(label='Pitch shift / 音调变换', minimum=-24, maximum=24, step=1, value=0, info="Pitch shift in semitones, only works when F0 conditioned model is used / 半音数的音高变换，仅在勾选 '启用F0输入' 时生效"),
    ]

    examples = [["examples/source/yae_0.wav", "examples/reference/dingzhen_0.wav", 25, 1.0, 0.7, False, True, 0],
                ["examples/source/jay_0.wav", "examples/reference/azuma_0.wav", 25, 1.0, 0.7, True, True, 0],
                ["examples/source/Wiz Khalifa,Charlie Puth - See You Again [vocals]_[cut_28sec].wav",
                 "examples/reference/teio_0.wav", 100, 1.0, 0.7, True, False, 0],
                ["examples/source/TECHNOPOLIS - 2085 [vocals]_[cut_14sec].wav",
                 "examples/reference/trump_0.wav", 50, 1.0, 0.7, True, False, -12],
                ]

    outputs = [gr.Audio(label="Stream Output Audio / 流式输出", streaming=True, format='mp3'),
               gr.Audio(label="Full Output Audio / 完整输出", streaming=False, format='wav')]

    gr.Interface(fn=voice_conversion,
                 description=description,
                 inputs=inputs,
                 outputs=outputs,
                 title="Seed Voice Conversion",
                 examples=examples,
                 cache_examples=False,
                 ).launch()