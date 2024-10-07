# Seed-VC  
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Demo-blue)](https://huggingface.co/spaces/Plachta/Seed-VC)  

*English | [简体中文](README-CN.md)*  
Currently released model supports *zero-shot voice conversion* 🔊 and *zero-shot singing voice conversion* 🎙. Without any training, it is able to clone a voice given a reference speech of 1~30 seconds.  

To find a list of demos and comparisons with previous voice conversion models, please visit our [demo page](https://plachtaa.github.io/seed-vc/)🌐  

We are keeping on improving the model quality and adding more features.

## Evaluation📊
We have performed a series of objective evaluations on our Seed-VC's voice conversion capabilities. 
For ease for reproduction, source audios are 100 random utterances from LibriTTS-test-clean, and reference audios are 12 randomly picked in-the-wild voices with unique characteristics. <br>  

Source audios can be found under `./examples/libritts-test-clean` <br>
Reference audios can be found under `./examples/reference` <br>

We evaluate the conversion results in terms of speaker embedding cosine similarity (SECS), word error rate (WER) and character error rate (CER) and compared
our results with two strong open sourced baselines, namely [OpenVoice](https://github.com/myshell-ai/OpenVoice) and [CosyVoice](https://github.com/FunAudioLLM/CosyVoice).  
Results in the table below shows that our Seed-VC model significantly outperforms the baseline models in both intelligibility and speaker similarity.<br>

| Models\Metrics | SECS↑      | WER↓       | CER↓       |
|----------------|------------|------------|------------|
| OpenVoice      | 0.7547     | 0.1546     | 0.0473     |
| CosyVoice      | 0.8440     | 0.1898     | 0.0729     |
| Seed-VC(Ours)  | **0.8676** | **0.1199** | **0.0292** |  

*ASR result computed by facebook/hubert-large-ls960-ft model*   
*Speaker embedding computed by resemblyzer model* <br>  

You can reproduce the evaluation by running `eval.py` script.  
```bash
python eval.py --source ./examples/libritts-test-clean \
--reference ./examples/reference
--output ./examples/eval/converted/
--diffusion-steps 25
--length-adjust 1.0
--inference-cfg-rate 0.7
--xvector-extractor "resemblyzer"
--baseline ""  # fill in openvoice or cosyvoice to compute baseline result
--max-samples 100  # max source utterances to go through
```
Before that, make sure you have openvoice and cosyvoice repo correctly installed on `../OpenVoice/` and `../CosyVoice/` if you would like to run baseline evaluation.

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
    - [x] Source audio is noise resilience
- [ ] Potential architecture improvements
    - [x] U-ViT style skip connections
    - [x] Changed input to OpenAI Whisper
- [ ] Code for training on custom data
- [x] Changed to BigVGAN from NVIDIA for singing voice decoding
- [ ] Whisper version model for singing voice conversion
- [ ] More to be added

## CHANGELOGS🗒️
- 2024-09-26:
    - Added objective evaluation results
    - Changed speech content encoder to OpenAI Whisper
- 2024-09-22:
    - Updated singing voice conversion model to use BigVGAN from NVIDIA, providing large improvement to high-pitched singing voices
    - Support chunking and  streaming output for long audio files in Web UI
- 2024-09-18:
    - Updated f0 conditioned model for singing voice conversion
- 2024-09-14:
    - Updated v0.2 pretrained model, with smaller size and less diffusion steps to achieve same quality, and additional ability to control prosody preservation
    - Added command line inference script
    - Added installation and usage instructions
