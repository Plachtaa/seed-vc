# Seed-VC  
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Demo-blue)](https://huggingface.co/spaces/Plachta/Seed-VC)  

*[English](README.md) | [簡体中文](README-CN.md) | 日本語*  
現在リリースされているモデルは、*ゼロショット音声変換* 🔊 と *ゼロショット歌声変換* 🎙 に対応しています。トレーニングなしで、1〜30秒のリファレンス音声を与えるだけで声をクローンすることができます。  

以前の音声変換モデルとの比較やデモのリストを見つけるには、[デモページ](https://plachtaa.github.io/seed-vc/)🌐 をご覧ください。  

私たちはモデルの品質を向上させ、さらに多くの機能を追加し続けています。

## 評価📊
Seed-VCの音声変換能力に関する一連の客観的評価を行いました。  
再現性を高めるために、ソース音声はLibriTTS-test-cleanからランダムに選ばれた100の発話、リファレンス音声は独自の特徴を持つ12のランダムに選ばれた自然音声です。<br>  

ソース音声は `./examples/libritts-test-clean` にあります <br>
リファレンス音声は `./examples/reference` にあります <br>

変換結果を話者埋め込みコサイン類似度（SECS）、単語誤り率（WER）、文字誤り率（CER）の観点から評価し、  
私たちの結果を2つの強力なオープンソースベースラインモデル、[OpenVoice](https://github.com/myshell-ai/OpenVoice) と [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) と比較しました。  
以下の表の結果は、私たちのSeed-VCモデルが知覚の明瞭さと話者の類似性の両方でベースラインモデルを大幅に上回っていることを示しています。<br>

| モデル\指標         | SECS↑      | WER↓       | CER↓       | SIG↑     | BAK↑     | OVRL↑    |
|---------------|------------|------------|------------|----------|----------|----------|
| Ground Truth  | 1.0000     | 0.0802     | 0.0157     | ~        | ~        | ~        |
| OpenVoice     | 0.7547     | 0.1546     | 0.0473     | **3.56** | **4.02** | **3.27** |
| CosyVoice     | 0.8440     | 0.1898     | 0.0729     | 3.51     | **4.02** | 3.21     |
| Seed-VC（Ours） | **0.8676** | **0.1199** | **0.0292** | 3.42     | 3.97     | 3.11     |

また、特定の話者に対して非ゼロショット音声変換モデルと比較しました（公開モデルの可用性に基づく）:

| Characters | Models\Metrics | SECS↑      | WER↓      | CER↓     | SIG↑     | BAK↑     | OVRL↑    |
|------------|----------------|------------|-----------|----------|----------|----------|----------|
| ~          | Ground Truth   | 1.0000     | 6.43      | 1.00     | ~        | ~        | ~        |
| 東海帝王       | So-VITS-4.0    | 0.8637     | 21.46     | 9.63     | 3.06     | 3.66     | 2.68     |
|            | Seed-VC(Ours)  | **0.8899** | **15.32** | **4.66** | **3.12** | **3.71** | **2.72** |
| 明前奶绿       | So-VITS-4.0    | 0.6850     | 48.43     | 32.50    | 3.34     | 3.51     | 2.82     |
|            | Seed-VC(Ours)  | **0.8072** | **7.26**  | **1.32** | **3.48** | **4.07** | **3.20** |
| 待兼诗歌剧      | So-VITS-4.0    | 0.8594     | 16.25     | 8.64     | **3.25** | 3.71     | 2.84     |
|            | Seed-VC(Ours)  | **0.8768** | **12.62** | **5.86** | 3.18     | **3.83** | **2.85** |

結果は、特定の話者に対して微調整やトレーニングを行っていないにもかかわらず、Seed-VCが非ゼロショットモデルよりもはるかに優れた結果を達成できることを示しています。 
ただし、これはSoVITSモデルの品質によって大きく異なる場合があります。この比較が不公平または不正確であると感じた場合は、PRまたはIssueを歓迎します。  
(東海帝王モデルは [zomehwh/sovits-tannhauser](https://huggingface.co/spaces/zomehwh/sovits-tannhauser) から取得)   
(待兼诗歌剧モデルは [zomehwh/sovits-tannhauser](https://huggingface.co/spaces/zomehwh/sovits-tannhauser) から取得)  
(明前奶绿モデルは [sparanoid/milky-green-sovits-4](https://huggingface.co/spaces/sparanoid/milky-green-sovits-4) から取得)  

*ASR結果は [facebook/hubert-large-ls960-ft](https://huggingface.co/facebook/hubert-large-ls960-ft) モデルによって計算されました*  
*話者埋め込みは [resemblyzer](https://github.com/resemble-ai/Resemblyzer) モデルによって計算されました* <br>

`eval.py` スクリプトを実行することで評価を再現できます。  
```bash
python eval.py 
--source ./examples/libritts-test-clean
--target ./examples/reference
--output ./examples/eval/converted
--diffusion-steps 25
--length-adjust 1.0
--inference-cfg-rate 0.7
--xvector-extractor "resemblyzer"
--baseline ""  # ベースライン結果を計算するには openvoice または cosyvoice を入力
--max-samples 100  # 処理する最大ソース発話数
```
その前に、ベースライン評価を実行する場合は、`../OpenVoice/` と `../CosyVoice/` にopenvoiceとcosyvoiceのリポジトリが正しくインストールされていることを確認してください。

## インストール📥
WindowsまたはLinuxでのPython 3.10を推奨します。
```bash
pip install -r requirements.txt
```

## 使用方法🛠️
初めて推論を実行する際には、最新のモデルリリースのチェックポイントが自動的にダウンロードされます。  

コマンドライン推論:
```bash
python inference.py --source <source-wav>
--target <referene-wav>
--output <output-dir>
--diffusion-steps 25 # 歌声変換には50〜100を推奨
--length-adjust 1.0
--inference-cfg-rate 0.7
--f0-condition False # 歌声変換にはTrueを設定
--auto-f0-adjust False # ソースのピッチをターゲットのピッチレベルに自動調整するにはTrueを設定、通常は歌声変換には使用しません
--semi-tone-shift 0 # 歌声変換の半音シフト
```
ここで:
- `source` はリファレンス音声に変換するための音声ファイルのパスです
- `target` はリファレンスとして使用する音声ファイルのパスです
- `output` は出力ディレクトリのパスです
- `diffusion-steps` は使用する拡散ステップ数です。デフォルトは25で、最高品質には50〜100を使用し、最速の推論には4〜10を使用します
- `length-adjust` は長さ調整係数です。デフォルトは1.0で、<1.0は音声を加速し、>1.0は音声を遅くします
- `inference-cfg-rate` は出力に微妙な違いがあります。デフォルトは0.7です
- `f0-condition` は出力のピッチをソース音声のピッチに条件付けるフラグです。デフォルトはFalseで、歌声変換にはTrueを設定します  
- `auto-f0-adjust` はソースのピッチをターゲットのピッチレベルに自動調整するフラグです。デフォルトはFalseで、通常は歌声変換には使用しません
- `semi-tone-shift` は歌声変換の半音シフトです。デフォルトは0です  

Gradioウェブインターフェース:
```bash
python app.py
```
その後、ブラウザを開いて `http://localhost:7860/` にアクセスしてウェブインターフェースを使用します。
## TODO📝
- [x] コードのリリース
- [x] v0.1の事前トレーニング済みモデルのリリース: [![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-SeedVC-blue)](https://huggingface.co/Plachta/Seed-VC)
- [x] Huggingfaceスペースデモ: [![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Space-blue)](https://huggingface.co/spaces/Plachta/Seed-VC)
- [x] HTMLデモページ（他のVCモデルとの比較を含む場合があります）: [Demo](https://plachtaa.github.io/seed-vc/)
- [ ] ストリーミング推論（現在の実装では品質低下を防ぐために1〜2秒の遅延が必要で、受け入れがたいほど高いです...😥）
- [x] 歌声変換
- [ ] ソースおよびリファレンス音声のノイズ耐性
    - [x] ソース音声はノイズ耐性があります
- [ ] 潜在的なアーキテクチャの改善
    - [x] U-ViTスタイルのスキップ接続
    - [x] 入力をOpenAI Whisperに変更
- [ ] カスタムデータでのトレーニング用コード
- [x] 歌声デコードにNVIDIAのBigVGANを使用
- [x] 歌声変換用のWhisperバージョンモデル
- [ ] 歌声変換の客観的評価とRVC/SoVITSとの比較
- [ ] 追加予定の項目

## 変更履歴🗒️
- 2024-10-24:
    - OpenAI Whisperを音声内容入力として使用した44kHzの歌声変換モデルを更新
- 2024-10-07:
    - v0.3の事前トレーニング済みモデルを更新し、音声内容エンコーダをOpenAI Whisperに変更
    - v0.3の事前トレーニング済みモデルの客観的評価結果を追加
- 2024-09-22:
    - 歌声変換モデルをNVIDIAのBigVGANに更新し、高音部分の変換が大幅に改善
    - Web UIで長い音声ファイルのチャンク処理とストリーミング出力をサポート
- 2024-09-18:
    - 歌声変換用のf0条件付きモデルを更新
- 2024-09-14:
    - v0.2の事前トレーニング済みモデルを更新し、同じ品質を達成するために拡散ステップが少なく、サイズが小さくなり、韻律保持の制御能力が追加
    - コマンドライン推論スクリプトを追加
    - インストールと使用方法の説明を追加
