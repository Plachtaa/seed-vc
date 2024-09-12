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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = yaml.safe_load(open("configs/config_dit_mel_seed.yml"))
model_params = recursive_munch(config['model_params'])
model = build_model(model_params, stage='DiT')
hop_length = config['preprocess_params']['spect_params']['hop_length']
sr = config['preprocess_params']['sr']

model, _, _, _ = load_checkpoint(model, None, "checkpoints/DiT_step_315000_seed_v2_online_pruned.pth",
                                                       load_only_params=True,
                                                       ignore_modules=[], is_distributed=False)
_ = [model[key].eval() for key in model]
_ = [model[key].to(device) for key in model]

content_type = model_params.DiT.content_type

diffusion_target = config['model_params']['DiT']['target']
model.cfm.estimator.setup_caches(max_batch_size=1, max_seq_length=8192)

spect_params = config['preprocess_params']['spect_params']
mel_fn_args = {
    "n_fft": spect_params['n_fft'],
    "win_size": spect_params['win_length'],
    "hop_size": spect_params['hop_length'],
    "num_mels": spect_params['n_mels'],
    "sampling_rate": sr,
    "fmin": 0,
    "fmax": 8000,
    "center": False
}
from modules.audio import mel_spectrogram
to_mel = lambda x: mel_spectrogram(x, **mel_fn_args)

# style encoder
from modules.campplus.DTDNN import CAMPPlus
campplus_model = CAMPPlus(feat_dim=80, embedding_size=192)
campplus_model.load_state_dict(torch.load(config['model_params']['style_encoder']['campplus_path']))
campplus_model.eval()
campplus_model.to(device)

# hifi-gan
from modules.hifigan.generator import HiFTGenerator
from modules.hifigan.f0_predictor import ConvRNNF0Predictor

hift_config = yaml.safe_load(open('configs/hifigan.yml', 'r'))
hift_gen = HiFTGenerator(**hift_config['hift'], f0_predictor=ConvRNNF0Predictor(**hift_config['f0_predictor']))
hift_gen.load_state_dict(torch.load(hift_config['pretrained_model_path'], map_location='cpu'))
hift_gen.eval()
hift_gen.to(device)

# speech tokenizer
from modules.cosyvoice_tokenizer.frontend import CosyVoiceFrontEnd

cosyvoice_frontend = CosyVoiceFrontEnd(speech_tokenizer_model=config['model_params']['speech_tokenizer']['path'],
                                       device='cuda', device_id=0)

@torch.no_grad()
def main(args):
    source = args.source
    target_name = args.target
    diffusion_steps = args.diffusion_steps
    length_adjust = args.length_adjust
    inference_cfg_rate = args.inference_cfg_rate
    source_audio = librosa.load(source, sr=sr)[0]
    ref_audio = librosa.load(target_name, sr=sr)[0]
    # decoded_wav = encodec_model.decoder(encodec_latent)
    # torchaudio.save("test.wav", decoded_wav.cpu().squeeze(0), 24000)
    # crop only the first 30 seconds
    source_audio = source_audio[:sr * 30]
    source_audio = torch.tensor(source_audio).unsqueeze(0).float().to(device)

    ref_audio = ref_audio[:(sr * 30 - source_audio.size(-1))]
    ref_audio = torch.tensor(ref_audio).unsqueeze(0).float().to(device)

    source_waves_16k = torchaudio.functional.resample(source_audio, sr, 16000)
    ref_waves_16k = torchaudio.functional.resample(ref_audio, sr, 16000)

    S_alt = [
        cosyvoice_frontend.extract_speech_token(source_waves_16k, )
    ]
    S_alt_lens = torch.LongTensor([s[1] for s in S_alt]).to(device)
    S_alt = torch.cat([torch.nn.functional.pad(s[0], (0, max(S_alt_lens) - s[0].size(1))) for s in S_alt], dim=0)

    S_ori = [
        cosyvoice_frontend.extract_speech_token(ref_waves_16k, )
    ]
    S_ori_lens = torch.LongTensor([s[1] for s in S_ori]).to(device)
    S_ori = torch.cat([torch.nn.functional.pad(s[0], (0, max(S_ori_lens) - s[0].size(1))) for s in S_ori], dim=0)

    mel = to_mel(source_audio.to(device).float())
    mel2 = to_mel(ref_audio.to(device).float())

    if diffusion_target == "mel":
        target = mel
        target2 = mel2

        target_lengths = torch.LongTensor([int(target.size(2) * length_adjust)]).to(target.device)
        target2_lengths = torch.LongTensor([target2.size(2)]).to(target2.device)
    else:
        raise NotImplementedError

    feat = kaldi.fbank(source_waves_16k,
                       num_mel_bins=80,
                       dither=0,
                       sample_frequency=16000)
    feat = feat - feat.mean(dim=0, keepdim=True)
    style1 = campplus_model(feat.unsqueeze(0))

    feat2 = kaldi.fbank(ref_waves_16k,
                        num_mel_bins=80,
                        dither=0,
                        sample_frequency=16000)
    feat2 = feat2 - feat2.mean(dim=0, keepdim=True)
    style2 = campplus_model(feat2.unsqueeze(0))

    cond = model.length_regulator(S_alt, ylens=target_lengths)[0]
    prompt_condition = model.length_regulator(S_ori, ylens=target2_lengths)[0]
    cat_condition = torch.cat([prompt_condition, cond], dim=1)
    prompt_target = target2

    time_vc_start = time.time()
    vc_target = model.cfm.inference(cat_condition, torch.LongTensor([cat_condition.size(1)]).to(prompt_target.device), prompt_target, style2, None, diffusion_steps, inference_cfg_rate=inference_cfg_rate)
    vc_target = vc_target[:, :, prompt_target.size(-1):]
    if diffusion_target == "mel":
        vc_wave = hift_gen.inference(vc_target)
    else:
        raise NotImplementedError
    time_vc_end = time.time()
    print(f"RTF: {(time_vc_end - time_vc_start) / vc_wave.size(-1) * sr}")

    source_name = source.split("/")[-1].split(".")[0]
    target_name = target_name.split("/")[-1].split(".")[0]
    torchaudio.save(os.path.join(args.output, f"vc_{source_name}_{target_name}_{length_adjust}_{diffusion_steps}_{inference_cfg_rate}.wav"), vc_wave.cpu(), sr)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="./test_waves/s4p2.wav")
    parser.add_argument("--target", type=str, default="./test_waves/cafe_0.wav")
    parser.add_argument("--output", type=str, default="./reconstructed")
    parser.add_argument("--diffusion-steps", type=int, default=100)
    parser.add_argument("--length-adjust", type=float, default=1.0)
    parser.add_argument("--inference-cfg-rate", type=float, default=0.7)
    args = parser.parse_args()
    main(args)
