# Seed-VC
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Demo-blue)](https://huggingface.co/spaces/Plachta/Seed-VC)  

*[English](README.md) | 简体中文*    

目前发布的模型支持零样本语音转换和零样本歌声转换。无需任何训练，只需提供1~30秒的参考语音即可克隆声音。  

要查看演示列表和与之前语音转换模型的比较，请访问我们的 [演示页面](https://plachtaa.github.io/seed-vc/)🌐  

我们将继续改进模型质量并添加更多功能。

## 评估📊

我们对 Seed-VC 的语音转换能力进行了系列客观评估。  
为了便于复现，源音频是来自 LibriTTS-test-clean 的 100 个随机语句，参考音频是 12 个随机挑选的具有独特特征的自然声音。<br>  

源音频位于 `./examples/libritts-test-clean` <br>
参考音频位于 `./examples/reference` <br>

我们从说话人嵌入余弦相似度（SECS）、词错误率（WER）和字符错误率（CER）三个方面评估了转换结果，并将我们的结果与两个强大的开源基线模型，即 [OpenVoice](https://github.com/myshell-ai/OpenVoice) 和 [CosyVoice](https://github.com/FunAudioLLM/CosyVoice)，进行了比较。  
下表的结果显示，我们的 Seed-VC 模型在发音清晰度和说话人相似度上均显著优于基线模型。<br>

| 模型\指标         | SECS↑      | WER↓       | CER↓       | SIG↑     | BAK↑     | OVRL↑    |
|---------------|------------|------------|------------|----------|----------|----------|
| Ground Truth  | 1.0000     | 0.0802     | 0.0157     | ~        | ~        | ~        |
| OpenVoice     | 0.7547     | 0.1546     | 0.0473     | **3.56** | **4.02** | **3.27** |
| CosyVoice     | 0.8440     | 0.1898     | 0.0729     | 3.51     | **4.02** | 3.21     |
| Seed-VC（Ours） | **0.8676** | **0.1199** | **0.0292** | 3.42     | 3.97     | 3.11     |

我们也与非zero-shot的声线转换模型在特定角色上进行了比较（基于可以找到的公开模型）:

| Characters | Models\Metrics | SECS↑      | WER↓      | CER↓     | SIG↑     | BAK↑     | OVRL↑    |
|------------|----------------|------------|-----------|----------|----------|----------|----------|
| ~          | Ground Truth   | 1.0000     | 6.43      | 1.00     | ~        | ~        | ~        |
| 东海帝王       | So-VITS-4.0    | 0.8637     | 21.46     | 9.63     | 3.06     | 3.66     | 2.68     |
|            | Seed-VC(Ours)  | **0.8899** | **15.32** | **4.66** | **3.12** | **3.71** | **2.72** |
| 明前奶绿       | So-VITS-4.0    | 0.6850     | 48.43     | 32.50    | 3.34     | 3.51     | 2.82     |
|            | Seed-VC(Ours)  | **0.8072** | **7.26**  | **1.32** | **3.48** | **4.07** | **3.20** |
| 待兼诗歌剧      | So-VITS-4.0    | 0.8594     | 16.25     | 8.64     | **3.25** | 3.71     | 2.84     |
|            | Seed-VC(Ours)  | **0.8768** | **12.62** | **5.86** | 3.18     | **3.83** | **2.85** |

结果显示，即便我们的模型没有在特定说话人上进行微调或训练，在音色相似度和咬字清晰度上也全面优于在特定说话人数据集上专门训练的SoVITS模型。 
但是该项测试结果高度依赖于SoVITS模型质量。如果您认为此对比不公平或不够准确，欢迎提issue或PR。  
(东海帝王模型来自 [zomehwh/sovits-tannhauser](https://huggingface.co/spaces/zomehwh/sovits-tannhauser))   
(待兼诗歌剧模型来自 [zomehwh/sovits-tannhauser](https://huggingface.co/spaces/zomehwh/sovits-tannhauser))  
(明前奶绿模型来自 [sparanoid/milky-green-sovits-4](https://huggingface.co/spaces/sparanoid/milky-green-sovits-4))  

*ASR 结果由 [facebook/hubert-large-ls960-ft](https://huggingface.co/facebook/hubert-large-ls960-ft) 模型计算*  
*说话人嵌入由 [resemblyzer](https://github.com/resemble-ai/Resemblyzer) 模型计算* <br>

你可以通过运行 `eval.py` 脚本来复现评估。  
```bash
python eval.py 
--source ./examples/libritts-test-clean
--target ./examples/reference
--output ./examples/eval/converted
--diffusion-steps 25
--length-adjust 1.0
--inference-cfg-rate 0.7
--xvector-extractor "resemblyzer"
--baseline ""  # 填入 openvoice 或 cosyvoice 来计算基线结果
--max-samples 100  # 要处理的最大源语句数
```
在此之前，如果你想运行基线评估，请确保已在 `../OpenVoice/` 和 `../CosyVoice/` 目录下正确安装了 openvoice 和 cosyvoice 仓库。
## 安装 📥
建议在 Windows 或 Linux 上使用 Python 3.10：
```bash
pip install -r requirements.txt
```

## 使用方法🛠️
首次运行推理时，将自动下载最新模型的检查点。  

命令行推理：
```bash
python inference.py --source <源语音文件路径>
--target <参考语音文件路径>
--output <输出目录>
--diffusion-steps 25 # 建议歌声转换时使用50~100
--length-adjust 1.0
--inference-cfg-rate 0.7
--f0-condition False # 歌声转换时设置为 True
--auto-f0-adjust False # 设置为 True 可自动调整源音高到目标音高，歌声转换中通常不使用
--semi-tone-shift 0 # 歌声转换的半音移调
```
其中:
- `source` 待转换为参考声音的源语音文件路径
- `target` 声音参考的语音文件路径
- `output` 输出目录的路径
- `diffusion-steps` 使用的扩散步数，默认25，最佳质量建议使用50-100，最快推理使用4-10
- `length-adjust` 长度调整系数，默认1.0，<1.0加速语音，>1.0减慢语音
- `inference-cfg-rate` 对输出有细微影响，默认0.7
- `f0-condition` 是否根据源音频的音高调整输出音高，默认 False，歌声转换时设置为 True  
- `auto-f0-adjust` 是否自动将源音高调整到目标音高水平，默认 False，歌声转换中通常不使用
- `semi-tone-shift` 歌声转换中的半音移调，默认0  

Gradio 网页界面:
```bash
python app.py
```
然后在浏览器中打开 `http://localhost:7860/` 使用网页界面。
## TODO📝
- [x] 发布代码
- [x] 发布 v0.1 预训练模型： [![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-SeedVC-blue)](https://huggingface.co/Plachta/Seed-VC)
- [x] Hugging Face Space 演示： [![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Space-blue)](https://huggingface.co/spaces/Plachta/Seed-VC)
- [x] HTML 演示页面（可能包含与其他 VC 模型的比较）： [Demo](https://plachtaa.github.io/seed-vc/)
- [ ] 流式推理
- [x] 歌声转换
- [x] 提高源音频抗噪性
- [ ] 潜在的架构改进
    - [x] 类似U-ViT 的skip connection
    - [x] 将输入更改为 OpenAI Whisper
- [ ] 自定义数据训练代码
- [x] 歌声解码器更改为 NVIDIA 的 BigVGAN
- [ ] 44k Hz 歌声转换模型
- [ ] 更多待添加

## 更新日志 🗒️
- 2024-09-26:
    - 更新了 v0.3 预训练模型，将语音内容编码器更改为 OpenAI Whisper
    - 添加了 v0.3 预训练模型的客观指标评估结果
- 2024-09-22:
    - 将歌声转换模型的解码器更改为 BigVGAN，解决了大部分高音部分无法正确转换的问题
    - 在Web UI中支持对长输入音频的分段处理以及流式输出
- 2024-09-18:
    - 更新了用于歌声转换的模型
- 2024-09-14:
    - 更新了 v0.2 预训练模型，具有更小的尺寸和更少的扩散步骤即可达到相同质量，且增加了控制韵律保留的能力
    - 添加了命令行推理脚本
    - 添加了安装和使用说明
