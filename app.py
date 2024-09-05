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
                                                "DiT_step_315000_seed_v2_online_pruned.pth",
                                                "config_dit_mel_seed.yml")

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

campplus_model = CAMPPlus(feat_dim=80, embedding_size=192)
campplus_model.load_state_dict(torch.load(config['model_params']['style_encoder']['campplus_path']))
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

from modules.cosyvoice_tokenizer.frontend import CosyVoiceFrontEnd

speech_tokenizer_path = load_custom_model_from_hf("Plachta/Seed-VC", "speech_tokenizer_v1.onnx", None)

cosyvoice_frontend = CosyVoiceFrontEnd(speech_tokenizer_model=speech_tokenizer_path,
                                       device='cuda', device_id=0)
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

@torch.no_grad()
@torch.inference_mode()
def voice_conversion(source, target, diffusion_steps, length_adjust, inference_cfg_rate):
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
    S_alt = cosyvoice_frontend.extract_speech_token(source_waves_16k)[0]
    S_ori = cosyvoice_frontend.extract_speech_token(ref_waves_16k)[0]

    mel = to_mel(source_audio.to(device).float())
    mel2 = to_mel(ref_audio.to(device).float())

    target_lengths = torch.LongTensor([int(mel.size(2) * length_adjust)]).to(mel.device)
    target2_lengths = torch.LongTensor([mel2.size(2)]).to(mel2.device)

    # Style encoding
    feat = torchaudio.compliance.kaldi.fbank(source_waves_16k,
                                             num_mel_bins=80,
                                             dither=0,
                                             sample_frequency=16000)
    feat = feat - feat.mean(dim=0, keepdim=True)
    style1 = campplus_model(feat.unsqueeze(0))

    feat2 = torchaudio.compliance.kaldi.fbank(ref_waves_16k,
                                              num_mel_bins=80,
                                              dither=0,
                                              sample_frequency=16000)
    feat2 = feat2 - feat2.mean(dim=0, keepdim=True)
    style2 = campplus_model(feat2.unsqueeze(0))

    # Length regulation
    cond = model.length_regulator(S_alt, ylens=target_lengths)[0]
    prompt_condition = model.length_regulator(S_ori, ylens=target2_lengths)[0]
    cat_condition = torch.cat([prompt_condition, cond], dim=1)

    # Voice Conversion
    vc_target = model.cfm.inference(cat_condition, torch.LongTensor([cat_condition.size(1)]).to(mel2.device),
                                    mel2, style2, None, diffusion_steps, inference_cfg_rate=inference_cfg_rate)
    vc_target = vc_target[:, :, mel2.size(-1):]

    # Convert to waveform
    vc_wave = hift_gen.inference(vc_target)

    return (sr, vc_wave.squeeze(0).cpu().numpy())


if __name__ == "__main__":
    description = "Zero-shot voice conversion with in-context learning. Check out our [GitHub repository](https://github.com/Plachtaa/seed-vc) for details and updates."
    inputs = [
        gr.Audio(source="upload", type="filepath", label="Source Audio"),
        gr.Audio(source="upload", type="filepath", label="Reference Audio"),
        gr.Slider(minimum=1, maximum=200, value=100, step=1, label="Diffusion Steps"),
        gr.Slider(minimum=0.5, maximum=2.0, step=0.1, value=1.0, label="Length Adjust"),
        gr.Slider(minimum=0.0, maximum=1.0, step=0.1, value=0.7, label="Inference CFG Rate"),
    ]

    outputs = gr.Audio(label="Output Audio")

    gr.Interface(fn=voice_conversion, description=description, inputs=inputs, outputs=outputs, title="Seed Voice Conversion").launch()