# Seed-VC  
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Demo-blue)](https://huggingface.co/spaces/Plachta/Seed-VC)  

*English | [简体中文](README-CN.md)*  
Currently released model supports *zero-shot voice conversion* 🔊 and *zero-shot singing voice conversion* 🎙. Without any training, it is able to clone a voice given a reference speech of 1~30 seconds.  

To find a list of demos and comparisons with previous voice conversion models, please visit our [demo page](https://plachtaa.github.io/seed-vc/)🌐  

We are keeping on improving the model quality and adding more features.

## Installation📥
Suggested python 3.10 on Windows or Linux.
```bash
pip install -r requirements.txt
```

## Usage🛠️
Checkpoints of the latest model release will be downloaded automatically when first run inference.  

Command line inference:
```bash
python inference.py --source <source-wav> \
--target <referene-wav>
--output <output-dir>
--diffusion-steps 25 # recommended 50~100 for singingvoice conversion
--length-adjust 1.0
--inference-cfg-rate 0.7
--n-quantizers 3
--f0-condition False # set to True for singing voice conversion
--auto-f0-condition False # set to True to auto adjust source pitch to target pitch level, normally not used in singing voice conversion
--semi-tone-shift 0 # pitch shift in semitones for singing voice conversion
```
where:
- `source` is the path to the speech file to convert to reference voice
- `target` is the path to the speech file as voice reference
- `output` is the path to the output directory
- `diffusion-steps` is the number of diffusion steps to use, default is 25, use 50-100 for best quality, use 4-10 for fastest inference
- `length-adjust` is the length adjustment factor, default is 1.0, set <1.0 for speed-up speech, >1.0 for slow-down speech
- `inference-cfg-rate` has subtle difference in the output, default is 0.7
- `n-quantizers` is the number of quantizers from FAcodec to use, default is 3, the less quantizer used, the less prosody of source audio is preserved  
- `f0-condition` is the flag to condition the pitch of the output to the pitch of the source audio, default is False, set to True for singing voice conversion  
- `auto-f0-condition` is the flag to auto adjust source pitch to target pitch level, default is False, normally not used in singing voice conversion
- `semi-tone-shift` is the pitch shift in semitones for singing voice conversion, default is 0  

Gradio web interface:
```bash
python app.py
```
Then open the browser and go to `http://localhost:7860/` to use the web interface.
## TODO📝
- [x] Release code
- [x] Release v0.1 pretrained model: [![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-SeedVC-blue)](https://huggingface.co/Plachta/Seed-VC)
- [x] Huggingface space demo: [![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Space-blue)](https://huggingface.co/spaces/Plachta/Seed-VC)
- [x] HTML demo page (maybe with comparisons to other VC models): [Demo](https://plachtaa.github.io/seed-vc/)
- [ ] Streaming inference
- [x] Singing voice conversion
- [ ] Noise resiliency for source & reference audio
    - [x] This is enabled for the f0 conditioned model but not sure whether it works well...
- [ ] Potential architecture improvements
    - [x] U-ViT style skip connections
    - [x] Changed input to [FAcodec](https://github.com/Plachtaa/FAcodec) tokens
- [ ] Code for training on custom data
- [x] Changed to BigVGAN from NVIDIA for singing voice decoding
- [ ] 44k Hz model for singing voice conversion of better quality
- [ ] More to be added

## CHANGELOGS🗒️
- 2024-09-22:
    - Updated singing voice conversion model to use BigVGAN from NVIDIA, providing large improvement to high-pitched singing voices
    - Support chunking and  streaming output for long audio files in Web UI
- 2024-09-18:
    - Updated f0 conditioned model for singing voice conversion
- 2024-09-14:
    - Updated v0.2 pretrained model, with smaller size and less diffusion steps to achieve same quality, and additional ability to control prosody preservation
    - Added command line inference script
    - Added installation and usage instructions
