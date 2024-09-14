# Seed-VC
Zero-shot voice conversion trained according to the scheme described in SEED-TTS.  
The VC quality is surprisingly good in terms of both audio quality and timbre similarity. We decide to continue along this pathway see where it can achieve.  

## Installation
```bash
pip install -r requirements.txt
```

## Usage
Checkpoints of the latest model release will be downloaded automatically when first run inference.  

Command line inference:
```bash
python inference.py --source <source-wav> \
--target <referene-wav>
--output <output-dir>
--diffusion-steps 10
--length-adjust 1.0
--inference-cfg-rate 0.7
--n-quantizers 3
```
where:
- `source` is the path to the speech file to convert to reference voice
- `target` is the path to the speech file as voice reference
- `output` is the path to the output directory
- `diffusion-steps` is the number of diffusion steps to use, default is 10, use 50~100 for best quality
- `length-adjust` is the length adjustment factor, default is 1.0, set <1.0 for speed-up speech, >1.0 for slow-down speech
- `inference-cfg-rate` has subtle difference in the output, default is 0.7
- `n-quantizers` is the number of quantizers from FAcodec to use, default is 3, the less quantizer used, the less prosody of source audio is preserved  

Gradio web interface:
```bash
python app.py
```
Then open the browser and go to `http://localhost:7860/` to use the web interface.
## TODO
- [x] Release code
- [x] Release v0.1 pretrained model: [![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-SeedVC-blue)](https://huggingface.co/Plachta/Seed-VC)
- [x] Huggingface space demo: [![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Space-blue)](https://huggingface.co/spaces/Plachta/Seed-VC)
- [x] HTML demo page (maybe with comparisons to other VC models): [Demo](https://plachtaa.github.io/seed-vc/)
- [ ] Code for training on custom data
- [ ] Streaming inference
- [ ] Singing voice conversion
- [ ] Noise resiliency for source & reference audio
- [ ] Potential architecture improvements
    - [x] U-ViT style skip connections
    - [x] Changed input to [FAcodec](https://github.com/Plachtaa/FAcodec) tokens
- [ ] More to be added

## CHANGELOGS
- 2024-09-14:
    - Updated v0.2 pretrained model, with smaller size and less diffusion steps to achieve same quality, and additional ability to control prosody preservation
    - Added command line inference script
    - Added installation and usage instructions