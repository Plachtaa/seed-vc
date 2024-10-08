import shutil
import warnings
import argparse
import torch
import os
import os.path as osp
import yaml

warnings.simplefilter('ignore')

# load packages
import random

from modules.commons import *
import time

import torchaudio
import librosa
import torchaudio.compliance.kaldi as kaldi

from hf_utils import load_custom_model_from_hf


# Load model and configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_models(args):
    if not args.f0_condition:
        dit_checkpoint_path, dit_config_path = load_custom_model_from_hf("Plachta/Seed-VC",
                                                                         "DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth",
                                                                         "config_dit_mel_seed_uvit_whisper_small_wavenet.yml")
        f0_extractor = None
    else:
        dit_checkpoint_path, dit_config_path = load_custom_model_from_hf("Plachta/Seed-VC",
                                                                         "DiT_seed_v2_uvit_facodec_small_wavenet_f0_bigvgan_pruned.pth",
                                                                         "config_dit_mel_seed_facodec_small_wavenet_f0.yml")
        # f0 extractor
        from modules.rmvpe import RMVPE

        model_path = load_custom_model_from_hf("lj1995/VoiceConversionWebUI", "rmvpe.pt", None)
        f0_extractor = RMVPE(model_path, is_half=False, device=device)

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

    if model_params.speech_tokenizer.type == "facodec":
        ckpt_path, config_path = load_custom_model_from_hf("Plachta/FAcodec", 'pytorch_model.bin', 'config.yml')

        codec_config = yaml.safe_load(open(config_path))
        codec_model_params = recursive_munch(codec_config['model_params'])
        codec_encoder = build_model(codec_model_params, stage="codec")

        ckpt_params = torch.load(ckpt_path, map_location="cpu")

        for key in codec_encoder:
            codec_encoder[key].load_state_dict(ckpt_params[key], strict=False)
        _ = [codec_encoder[key].eval() for key in codec_encoder]
        _ = [codec_encoder[key].to(device) for key in codec_encoder]
        speechtokenizer_set = ('facodec', codec_encoder, None)
    elif model_params.speech_tokenizer.type == "whisper":
        from transformers import AutoFeatureExtractor, WhisperModel
        whisper_name = model_params.speech_tokenizer.whisper_name if hasattr(model_params.speech_tokenizer, 'whisper_name') else "whisper-large-v3"
        whisper_model = WhisperModel.from_pretrained(whisper_name, torch_dtype=torch.float16).to(device)
        del whisper_model.decoder
        whisper_feature_extractor = AutoFeatureExtractor.from_pretrained(whisper_name)
        speechtokenizer_set = ('whisper', whisper_model, whisper_feature_extractor)
    else:
        raise ValueError(f"Unsupported speech tokenizer type: {model_params.speech_tokenizer.type}")


    # Generate mel spectrograms
    mel_fn_args = {
        "n_fft": config['preprocess_params']['spect_params']['n_fft'],
        "win_size": config['preprocess_params']['spect_params']['win_length'],
        "hop_size": config['preprocess_params']['spect_params']['hop_length'],
        "num_mels": config['preprocess_params']['spect_params']['n_mels'],
        "sampling_rate": sr,
        "fmin": config['preprocess_params'].get('fmin', 0),
        "fmax": None if config['preprocess_params'].get('fmax', "None") == "None" else 8000,
        "center": False
    }
    from modules.audio import mel_spectrogram

    to_mel = lambda x: mel_spectrogram(x, **mel_fn_args)

    return model, speechtokenizer_set, f0_extractor, bigvgan_model, campplus_model, to_mel, mel_fn_args

def adjust_f0_semitones(f0_sequence, n_semitones):
    factor = 2 ** (n_semitones / 12)
    return f0_sequence * factor

@torch.no_grad()
def main(args):
    model, speechtokenizer_set, f0_extractor, bigvgan_model, campplus_model, to_mel, mel_fn_args = load_models(args)
    sr = mel_fn_args['sampling_rate']
    f0_condition = args.f0_condition
    auto_f0_adjust = args.auto_f0_adjust
    pitch_shift = args.semi_tone_shift

    source = args.source
    target_name = args.target
    diffusion_steps = args.diffusion_steps
    length_adjust = args.length_adjust
    inference_cfg_rate = args.inference_cfg_rate
    source_audio = librosa.load(source, sr=sr)[0]
    ref_audio = librosa.load(target_name, sr=sr)[0]

    source_audio = source_audio[:sr * 30]
    source_audio = torch.tensor(source_audio).unsqueeze(0).float().to(device)

    ref_audio = ref_audio[:(sr * 30 - source_audio.size(-1))]
    ref_audio = torch.tensor(ref_audio).unsqueeze(0).float().to(device)

    source_waves_16k = torchaudio.functional.resample(source_audio, sr, 16000)
    ref_waves_16k = torchaudio.functional.resample(ref_audio, sr, 16000)

    converted_waves_24k = torchaudio.functional.resample(source_audio, sr, 24000)
    wave_lengths_24k = torch.LongTensor([converted_waves_24k.size(1)]).to(converted_waves_24k.device)
    waves_input = converted_waves_24k.unsqueeze(1)
    if speechtokenizer_set[0] == 'facodec':
        codec_encoder = speechtokenizer_set[1]
        z = codec_encoder.encoder(waves_input)
        (quantized, codes) = codec_encoder.quantizer(z, waves_input)
        S_alt = torch.cat([codes[1], codes[0]], dim=1)

        # S_ori should be extracted in the same way
        waves_24k = torchaudio.functional.resample(ref_audio, sr, 24000)
        waves_input = waves_24k.unsqueeze(1)
        z = codec_encoder.encoder(waves_input)
        (quantized, codes) = codec_encoder.quantizer(z, waves_input)
        S_ori = torch.cat([codes[1], codes[0]], dim=1)
    elif speechtokenizer_set[0] == 'whisper':
        whisper_model = speechtokenizer_set[1]
        whisper_feature_extractor = speechtokenizer_set[2]
        converted_waves_16k = torchaudio.functional.resample(source_audio, sr, 16000)
        alt_inputs = whisper_feature_extractor([converted_waves_16k.squeeze(0).cpu().numpy()],
                                               return_tensors="pt",
                                               return_attention_mask=True,)
        alt_input_features = whisper_model._mask_input_features(
            alt_inputs.input_features, attention_mask=alt_inputs.attention_mask).to(device)
        with torch.no_grad():
            alt_outputs = whisper_model.encoder(
                alt_input_features.to(whisper_model.encoder.dtype),
                head_mask=None,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
            )
        S_alt = alt_outputs.last_hidden_state.to(torch.float32)
        S_alt = S_alt[:, :converted_waves_16k.size(-1)//320 + 1]

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
    else:
        raise ValueError(f"Unsupported speech tokenizer type: {speechtokenizer_set[0]}")

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
        F0_ori = f0_extractor.infer_from_audio(waves_16k[0], thred=0.03)
        F0_alt = f0_extractor.infer_from_audio(converted_waves_16k[0], thred=0.03)

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
    cond, _, codes, commitment_loss, codebook_loss = model.length_regulator(S_alt, ylens=target_lengths, n_quantizers=3, f0=shifted_f0_alt)
    prompt_condition, _, prompt_codes, commitment_loss, codebook_loss = model.length_regulator(S_ori, ylens=target2_lengths, n_quantizers=3, f0=F0_ori)
    cat_condition = torch.cat([prompt_condition, cond], dim=1)

    time_vc_start = time.time()
    vc_target = model.cfm.inference(
            cat_condition,
            torch.LongTensor([cat_condition.size(1)]).to(mel2.device),
            mel2, style2, None, diffusion_steps,
            inference_cfg_rate=inference_cfg_rate)
    vc_target = vc_target[:, :, mel2.size(-1):]

    # Convert to waveform
    # if f0_condition:
    vc_wave = bigvgan_model(vc_target).squeeze(1)  # wav_gen is FloatTensor with shape [B(1), 1, T_time] and values in [-1, 1]

    time_vc_end = time.time()
    print(f"RTF: {(time_vc_end - time_vc_start) / vc_wave.size(-1) * sr}")

    source_name = source.split("/")[-1].split(".")[0]
    target_name = target_name.split("/")[-1].split(".")[0]
    os.makedirs(args.output, exist_ok=True)
    torchaudio.save(os.path.join(args.output, f"vc_{source_name}_{target_name}_{length_adjust}_{diffusion_steps}_{inference_cfg_rate}.wav"), vc_wave.cpu(), sr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="./examples/source/source_s1.wav")
    parser.add_argument("--target", type=str, default="./examples/reference/s1p1.wav")
    parser.add_argument("--output", type=str, default="./reconstructed")
    parser.add_argument("--diffusion-steps", type=int, default=30)
    parser.add_argument("--length-adjust", type=float, default=1.0)
    parser.add_argument("--inference-cfg-rate", type=float, default=0.7)
    parser.add_argument("--f0-condition", type=bool, default=False)
    parser.add_argument("--auto-f0-adjust", type=bool, default=False)
    parser.add_argument("--semi-tone-shift", type=int, default=0)
    args = parser.parse_args()
    main(args)
