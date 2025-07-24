# _OmniCharacter_: Towards Immersive Role-Playing Agents with Seamless Speech-Language Personality Interaction (ACL 2025 main)

[Haonan Zhang](https://zchoi.github.io/)\*, [Run Luo](https://scholar.google.com/citations?user=phg8yxoAAAAJ&hl=zh-CN&oi=ao)\*, Xiong Liu\*, Yuchuan Wu, Ting-En Lin, Pengpeng Zeng, Qiang Qu, Feiteng Fang, Min Yang, Lianli Gao, Jingkuan Song<sup>‡</sup>, Fei Huang, Yongbin Li<sup>‡</sup> (\* Equal contribution ‡ Corresponding author)

This is the official code implementation of the paper "**OmniCharacter: Towards Immersive Role-Playing Agents with Seamless Speech-Language Personality Interaction**".

We are continuously refactoring our code, be patient and wait for the latest updates!

## 🔥 Updates

- [ ] Release the pre-trained weight and datasets.
- [x] Release the training and evaluation code.
- [x] We release the [paper](https://arxiv.org/abs/2505.20277) for OmniCharacter!

## ⚙️ Installation

1.  Clone the repo

```
git clone --recursive https://github.com/AlibabaResearch/DAMO-ConvAI/tree/main/OmniCharacter
cd OmniCharacter
```

2. Create Conda env:
```
conda create -n omnicharacter python=3.10 -y
conda activate omnicharacter
pip install --upgrade pip  # enable PEP 660 support
pip install -e ".[train]"
pip install -r requirements.txt

# Install Flash Attention 2 for training (https://github.com/Dao-AILab/flash-attention)
#   =>> If you run into difficulty, try `pip cache remove flash_attn` first
pip install packaging ninja
ninja --version; echo $?  # Verify Ninja --> should return exit code "0"
pip install "flash-attn" --no-build-isolation
```

## 🚀 Train
1. Download the dataset.
First, download the OmniCharacter training and test sets from our [HuggingFace🤗](https://huggingface.co/datasets/Tongyi-ConvAI/OmniCharacter) repository.
After downloading, place the dataset in a folder named data/ under the project root:

```
mkdir -p data
# Put the downloaded files into the data/ folder
```

2. Prepare checkpoints and Speech Modules

We finetune OmniCharacter based on the [OpenOmni](https://arxiv.org/abs/2501.04561) pre-trained weights.
You can download the OpenOmni checkpoints from [HuggingFace🤗](https://huggingface.co/Tongyi-ConvAI/OpenOmni/tree/main) and place them in a checkpoints/ directory, which will be also downloaded with the OpenOmni:
```
mkdir -p checkpoints
# Put the OpenOmni weights into checkpoints/
```
In addition, make sure the following modules are also placed under the checkpoints/ directory:

- speech_projector: The pre-trained speech encoder used to extract speech features from reference audio.

- speech_generator: The pre-trained speech decoder model used for generating speech tokens.

Your directory structure should look like this:
```
OmniCharacter/
├── checkpoints/
│   └── openomni/
│       ├── pretrained/
│       │   ├── speech_projector/
│       │   └── speech_generator/
│       └── qwen/
├── data/
│   ├── omnicharacter_10k_train.json
│   ├── omnicharacter_test.json
│   └── audio_data/
```

3. You can train the model with the following command:

**Stage-1**: focuses on aligning speech features (user query) and text (role profile, dialogue contexts, etc.) in the shared personality space. Use the provided shell script to launch training:
```
bash omnicharacter_stage1_qwen2.5.sh
```
This will save outputs to a designated directory ```results/```.

**Stage-2**: further finetunes the speech generator

Once Stage 1 completes, locate the checkpoint (e.g., results/stage1/checkpoint-xxx/) and pass it to Stage 2 as ```--model_name_or_path```:
```
bash omnicharacter_stage2_qwen2.5.sh
```

## 🍃 Inference
After downloading the weights and configuring the paths properly. A speech tokenizer are needed for speech discretization and reconstruction, _i.e._, [GLM-4-Voice](https://github.com/THUDM/GLM-4-Voice)

Fast inference:
```
python inference.py
```

## 📖 Citation
If this project contributes to your research, we kindly ask you to cite the following paper:
```
@article{zhang2025omnicharacter,
  title={OmniCharacter: Towards Immersive Role-Playing Agents with Seamless Speech-Language Personality Interaction},
  author={Zhang, Haonan and Luo, Run and Liu, Xiong and Wu, Yuchuan and Lin, Ting-En and Zeng, Pengpeng and Qu, Qiang and Fang, Feiteng and Yang, Min and Gao, Lianli and others},
  journal={ACL 2025},
  year={2025}
}
```
```
@article{luo2025openomni,
  title={OpenOmni: Large Language Models Pivot Zero-shot Omnimodal Alignment across Language with Real-time Self-Aware Emotional Speech Synthesis},
  author={Luo, Run and Lin, Ting-En and Zhang, Haonan and Wu, Yuchuan and Liu, Xiong and Yang, Min and Li, Yongbin and Chen, Longze and Li, Jiaming and Zhang, Lei and others},
  journal={arXiv preprint arXiv:2501.04561},
  year={2025}
}
```
```
@article{luo2024mmevol,
  title={Mmevol: Empowering multimodal large language models with evol-instruct},
  author={Luo, Run and Zhang, Haonan and Chen, Longze and Lin, Ting-En and Liu, Xiong and Wu, Yuchuan and Yang, Min and Wang, Minzheng and Zeng, Pengpeng and Gao, Lianli and others},
  journal={ACL 2025},
  year={2024}
}
```
## 📧 Contact
If you have any questions or need assistance, feel free to reach out via the contact information below.

- Haonan Zhang — zchiowal@gmail.com

- Run Luo — r.luo@siat.ac.cn


## Acknowledgement

- [**OpenOmni**](https://huggingface.co/AlibabaResearch/OpenOmni): The backbone multimodal foundation model powering our speech-language finetuning. We are truly excited to build on top of this open effort!
  
- [**LLaVA**](https://github.com/haotian-liu/LLaVA) and [**LLaVA-Omni**](https://github.com/ictnlp/LLaMA-Omni): The foundational codebases our work builds upon. We sincerely appreciate their pioneering contributions to the community!

- [**CosVoice**](https://github.com/FunAudioLLM/CosyVoice): An excellent open-source speech tokenizer enabling discretization and reconstruction with a 6k vocabulary—essential for expressive speech representation.

- [**GLM4Voice**](https://github.com/THUDM/GLM-4-Voice): Another impressive speech tokenizer supporting high-fidelity reconstruction with a 16k vocabulary. Huge thanks for making this resource available!
