import gradio as gr
import torch
import torchaudio
import librosa
from modules.commons import build_model, load_checkpoint, recursive_munch
import yaml
from hf_utils import load_custom_model_from_hf

# Load model and configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dit_checkpoint_path, dit_config_path = load_custom_model_from_hf("Plachta/Seed-VC",
                                                "DiT_step_298000_seed_uvit_facodec_small_wavenet_pruned.pth",
                                                "config_dit_mel_seed_facodec_small_wavenet.yml")

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

from modules.hifigan.generator import HiFTGenerator
from modules.hifigan.f0_predictor import ConvRNNF0Predictor

hift_checkpoint_path, hift_config_path = load_custom_model_from_hf("Plachta/Seed-VC",
                                                "hift.pt",
                                                "hifigan.yml")
hift_config = yaml.safe_load(open(hift_config_path, 'r'))
hift_gen = HiFTGenerator(**hift_config['hift'], f0_predictor=ConvRNNF0Predictor(**hift_config['f0_predictor']))
hift_gen.load_state_dict(torch.load(hift_checkpoint_path, map_location='cpu'))
hift_gen.eval()
hift_gen.to(device)

speech_tokenizer_type = config['model_params']['speech_tokenizer'].get('type', 'cosyvoice')
if speech_tokenizer_type == 'cosyvoice':
    from modules.cosyvoice_tokenizer.frontend import CosyVoiceFrontEnd
    speech_tokenizer_path = load_custom_model_from_hf("Plachta/Seed-VC", "speech_tokenizer_v1.onnx", None)
    cosyvoice_frontend = CosyVoiceFrontEnd(speech_tokenizer_model=speech_tokenizer_path,
                                           device='cuda', device_id=0)
elif speech_tokenizer_type == 'facodec':
    ckpt_path, config_path = load_custom_model_from_hf("Plachta/FAcodec", 'pytorch_model.bin', 'config.yml')

    codec_config = yaml.safe_load(open(config_path))
    codec_model_params = recursive_munch(codec_config['model_params'])
    codec_encoder = build_model(codec_model_params, stage="codec")

    ckpt_params = torch.load(ckpt_path, map_location="cpu")

    for key in codec_encoder:
        codec_encoder[key].load_state_dict(ckpt_params[key], strict=False)
    _ = [codec_encoder[key].eval() for key in codec_encoder]
    _ = [codec_encoder[key].to(device) for key in codec_encoder]
# Generate mel spectrograms
mel_fn_args = {
    "n_fft": config['preprocess_params']['spect_params']['n_fft'],
    "win_size": config['preprocess_params']['spect_params']['win_length'],
    "hop_size": config['preprocess_params']['spect_params']['hop_length'],
    "num_mels": config['preprocess_params']['spect_params']['n_mels'],
    "sampling_rate": sr,
    "fmin": 0,
    "fmax": 8000,
    "center": False
}
from modules.audio import mel_spectrogram

to_mel = lambda x: mel_spectrogram(x, **mel_fn_args)

# f0 conditioned model
dit_checkpoint_path, dit_config_path = load_custom_model_from_hf("Plachta/Seed-VC",
                                                "DiT_step_440000_seed_v2_uvit_facodec_small_wavenet_f0_pruned.pth",
                                                "config_dit_mel_seed_facodec_small_wavenet_f0.yml")

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

def adjust_f0_semitones(f0_sequence, n_semitones):
    factor = 2 ** (n_semitones / 12)
    return f0_sequence * factor

@torch.no_grad()
@torch.inference_mode()
def voice_conversion(source, target, diffusion_steps, length_adjust, inference_cfg_rate, n_quantizers, f0_condition, auto_f0_adjust, pitch_shift, concat_prompt):
    inference_module = model if not f0_condition else model_f0
    # Load audio
    source_audio = librosa.load(source, sr=sr)[0]
    ref_audio = librosa.load(target, sr=sr)[0]

    # Process audio
    source_audio = torch.tensor(source_audio[:sr * 30]).unsqueeze(0).float().to(device)
    ref_audio = torch.tensor(ref_audio[:sr * 30]).unsqueeze(0).float().to(device)

    # Resample
    source_waves_16k = torchaudio.functional.resample(source_audio, sr, 16000)
    ref_waves_16k = torchaudio.functional.resample(ref_audio, sr, 16000)

    # Extract features
    if speech_tokenizer_type == 'cosyvoice':
        S_alt = cosyvoice_frontend.extract_speech_token(source_waves_16k)[0]
        S_ori = cosyvoice_frontend.extract_speech_token(ref_waves_16k)[0]
    elif speech_tokenizer_type == 'facodec':
        converted_waves_24k = torchaudio.functional.resample(source_audio, sr, 24000)
        wave_lengths_24k = torch.LongTensor([converted_waves_24k.size(1)]).to(converted_waves_24k.device)
        waves_input = converted_waves_24k.unsqueeze(1)
        z = codec_encoder.encoder(waves_input)
        (
            quantized,
            codes
        ) = codec_encoder.quantizer(
            z,
            waves_input,
        )
        S_alt = torch.cat([codes[1], codes[0]], dim=1)

        # S_ori should be extracted in the same way
        waves_24k = torchaudio.functional.resample(ref_audio, sr, 24000)
        waves_input = waves_24k.unsqueeze(1)
        z = codec_encoder.encoder(waves_input)
        (
            quantized,
            codes
        ) = codec_encoder.quantizer(
            z,
            waves_input,
        )
        S_ori = torch.cat([codes[1], codes[0]], dim=1)

    mel = to_mel(source_audio.to(device).float())
    mel2 = to_mel(ref_audio.to(device).float())

    target_lengths = torch.LongTensor([int(mel.size(2) * length_adjust)]).to(mel.device)
    target2_lengths = torch.LongTensor([mel2.size(2)]).to(mel2.device)

    feat2 = torchaudio.compliance.kaldi.fbank(ref_waves_16k,
                                              num_mel_bins=80,
                                              dither=0,
                                              sample_frequency=16000)
    feat2 = feat2 - feat2.mean(dim=0, keepdim=True)
    style2 = campplus_model(feat2.unsqueeze(0))

    if f0_condition:
        waves_16k = torchaudio.functional.resample(waves_24k, sr, 16000)
        converted_waves_16k = torchaudio.functional.resample(converted_waves_24k, sr, 16000)
        F0_ori = rmvpe.infer_from_audio(waves_16k[0], thred=0.03)
        F0_alt = rmvpe.infer_from_audio(converted_waves_16k[0], thred=0.03)

        F0_ori = torch.from_numpy(F0_ori).to(device)[None]
        F0_alt = torch.from_numpy(F0_alt).to(device)[None]

        voiced_F0_ori = F0_ori[F0_ori > 1]
        voiced_F0_alt = F0_alt[F0_alt > 1]

        log_f0_alt = torch.log(F0_alt + 1e-5)
        voiced_log_f0_ori = torch.log(voiced_F0_ori + 1e-5)
        voiced_log_f0_alt = torch.log(voiced_F0_alt + 1e-5)
        median_log_f0_ori = torch.median(voiced_log_f0_ori)
        median_log_f0_alt = torch.median(voiced_log_f0_alt)
        # mean_log_f0_ori = torch.mean(voiced_log_f0_ori)
        # mean_log_f0_alt = torch.mean(voiced_log_f0_alt)

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
    cond = inference_module.length_regulator(S_alt, ylens=target_lengths, n_quantizers=int(n_quantizers), f0=shifted_f0_alt)[0]
    prompt_condition = inference_module.length_regulator(S_ori, ylens=target2_lengths, n_quantizers=int(n_quantizers), f0=F0_ori)[0]
    if concat_prompt:
        cat_condition = torch.cat([prompt_condition, cond], dim=1)
    else:
        cat_condition = cond
        mel2 = mel2[:, :, mel2.size(-1):]

    # Voice Conversion
    vc_target = inference_module.cfm.inference(cat_condition, torch.LongTensor([cat_condition.size(1)]).to(mel2.device),
                                    mel2, style2, None, diffusion_steps, inference_cfg_rate=inference_cfg_rate)
    if concat_prompt:
        vc_target = vc_target[:, :, mel2.size(-1):]

    # Convert to waveform
    if f0_condition and not auto_f0_adjust and pitch_shift == 0:
        f04vocoder = torch.nn.functional.interpolate(F0_ori.unsqueeze(1), size=vc_target.size(-1),
                                                     mode='nearest').squeeze(1)
    else:
        f04vocoder = None
    vc_wave = hift_gen.inference(vc_target, f0=f04vocoder)

    return sr, vc_wave.squeeze(0).cpu().numpy()


if __name__ == "__main__":
    description = "Zero-shot voice conversion with in-context learning. Check out our [GitHub repository](https://github.com/Plachtaa/seed-vc) for details and updates."
    inputs = [
        gr.Audio(type="filepath", label="Source Audio"),
        gr.Audio(type="filepath", label="Reference Audio"),
        gr.Slider(minimum=1, maximum=200, value=10, step=1, label="Diffusion Steps", info="10 by default, 50~100 for best quality"),
        gr.Slider(minimum=0.5, maximum=2.0, step=0.1, value=1.0, label="Length Adjust", info="<1.0 for speed-up speech, >1.0 for slow-down speech"),
        gr.Slider(minimum=0.0, maximum=1.0, step=0.1, value=0.7, label="Inference CFG Rate", info="has subtle influence"),
        gr.Slider(minimum=1, maximum=3, step=1, value=3, label="N Quantizers", info="the less quantizer used, the less prosody of source audio is preserved"),
        gr.Checkbox(label="Use F0 conditioned model", value=False, info="Must set to true for singing voice conversion"),
        gr.Checkbox(label="Auto F0 adjust", value=True,
                    info="Roughly adjust F0 to match target voice. Only works when F0 conditioned model is used."),
        gr.Slider(label='Pitch shift', minimum=-24, maximum=24, step=1, value=0, info='Pitch shift in semitones, only works when F0 conditioned model is used'),
        gr.Checkbox(label="Concat Prompt", value=True,
                    info="Concat original speech as prompt"),
    ]

    examples = [["examples/source/yae_0.wav", "examples/reference/dingzhen_0.wav", 25, 1.0, 0.7, 1, False, True, 0, True],
                ["examples/source/Wiz Khalifa,Charlie Puth - See You Again [vocals]_[cut_28sec].wav",
                 "examples/reference/teio_0.wav", 100, 1.0, 0.7, 3, True, True, 0, True],]

    outputs = gr.Audio(label="Output Audio")

    gr.Interface(fn=voice_conversion,
                 description=description,
                 inputs=inputs,
                 outputs=outputs,
                 title="Seed Voice Conversion",
                 examples=examples,
                 cache_examples=False,
                 ).launch()