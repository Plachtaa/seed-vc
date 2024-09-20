# Seed-VC
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Demo-blue)](https://huggingface.co/spaces/Plachta/Seed-VC)  

*[English](README.md) | 简体中文*    

一个受 SEED-TTS 启发的新型零样本声音转换方案。    

目前发布的模型支持零样本语音转换和零样本歌声转换。无需任何训练，只需提供1~30秒的参考语音即可克隆声音。  

要查看演示列表和与之前语音转换模型的比较，请访问我们的 [演示页面](https://plachtaa.github.io/seed-vc/)🌐  

我们将继续改进模型质量并添加更多功能。

## 安装 📥
建议在 Windows 或 Linux 上使用 Python 3.10：
```bash
pip install -r requirements.txt
```

## 使用方法🛠️
首次运行推理时，将自动下载最新模型的检查点。  

命令行推理：
```bash
python inference.py --source <源语音文件路径> \
--target <参考语音文件路径> \
--output <输出目录> \
--diffusion-steps 25 \ # 建议歌声转换时使用50~100
--length-adjust 1.0 \
--inference-cfg-rate 0.7 \
--n-quantizers 3 \
--f0-condition False \ # 歌声转换时设置为 True
--auto-f0-condition False \ # 设置为 True 可自动调整源音高到目标音高，歌声转换中通常不使用
--semi-tone-shift 0 # 歌声转换的半音移调
```
其中:
- `source` 待转换为参考声音的源语音文件路径
- `target` 声音参考的语音文件路径
- `output` 输出目录的路径
- `diffusion-steps` 使用的扩散步数，默认25，最佳质量建议使用50-100，最快推理使用4-10
- `length-adjust` 长度调整系数，默认1.0，<1.0加速语音，>1.0减慢语音
- `inference-cfg-rate` 对输出有细微影响，默认0.7
- `n-quantizers` 用的 FAcodec 码本数量，默认3，使用的码本越少，保留的源音频韵律越少  
- `f0-condition` 是否根据源音频的音高调整输出音高，默认 False，歌声转换时设置为 True  
- `auto-f0-condition` 是否自动将源音高调整到目标音高水平，默认 False，歌声转换中通常不使用
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
- [ ] 提高源音频和参考音频的抗噪性
    - [x] 这已在 f0 条件模型中启用，但不确定效果如何...
- [ ] 潜在的架构改进
    - [x] 类似U-ViT 的skip connection
    - [x] 将输入更改为 [FAcodec](https://github.com/Plachtaa/FAcodec) tokens
- [ ] 自定义数据训练代码
- [ ] 重新训练 NSF Hifi-GAN 用于歌声解码
- [ ] 更多待添加

## 更新日志 🗒️
- 2024-09-18:
    - 更新了用于歌声转换的模型
- 2024-09-14:
    - 更新了 v0.2 预训练模型，具有更小的尺寸和更少的扩散步骤即可达到相同质量，且增加了控制韵律保留的能力
    - 添加了命令行推理脚本
    - 添加了安装和使用说明
