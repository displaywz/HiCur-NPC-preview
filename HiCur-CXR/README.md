# HiCur-CXR: Transferring HiCur-NPC's HFFCL Method to Chest X-ray Tasks

This repository explores the migration of the HFFCL method from HiCur-NPC to chest X-ray (CXR) related tasks, specifically focusing on the **MIMIC-CXR-VQA** and **CXLSeg** datasets for **CXR Visual Question Answering (VQA)** and **Lung Cancer Segmentation**, respectively.

## Implementation Steps

### 1. Download MIMIC-CXR-JPG Dataset
- **Dataset Link**: [MIMIC-CXR-JPG 2.1.0](https://physionet.org/content/mimic-cxr-jpg/2.1.0/)
- **Usage**: We use the pre-divided `train` set for training. Notably, we do not use the corresponding text descriptions, only the pure CXR images. We use 20% of the total data for **Fast Continuous Pre-training (Fast-CPT)** of **HCMAE-ViT** and **HCMAE-ConvNeXt**.

### 2. Download MIMIC-CXR-VQA Dataset
- **Dataset Link**: [MIMIC-CXR-VQA](https://github.com/baeseongsu/mimic-cxr-vqa)
- **Usage**: We use the **CLOSED** portion of the dataset, where the answers are deterministic and unique. We use the pre-divided `train` set for training, experimenting with both 20% and 100% of the data. Note that the train/test split in this dataset is consistent with MIMIC-CXR-JPG, and any potential overlaps have already been excluded in the original dataset (this is not a special operation of this repository).

### 3. Download CXLSeg Dataset
- **Dataset Link**: [Chest X-ray Segmentation 1.0.0](https://physionet.org/content/chest-x-ray-segmentation/1.0.0/)
- **Usage**: We use the original data split format provided by the repository to train our **HiCur-CXR** model.

### 4. Comparative Experiments

#### 4.1 VQA Comparison
We compare **HiCur-CXR** with several open-source and commercial models on the VQA task. Specifically, we compare:
- **CXR-Specific Models**: XrayGPT, LLM-CXR
- **Open-Source Medical Models**: LLaVA-Med, Qwen-2-VL-7B
- **Commercial Models**: GPT-4o (with official SFT), Gemini-1.5-pro

For each model that can be fine-tuned, we perform full-parameter fine-tuning using the methods provided by their respective repositories. The fine-tuning links are as follows:
1. **Xray-GPT**: [Xray-GPT GitHub](https://github.com/mbzuai-oryx/XrayGPT)
2. **LLM-CXR**: [LLM-CXR GitHub](https://github.com/hyn2028/llm-cxr)
3. **LLaVA-Med**: [LLaVA-Med GitHub](https://github.com/microsoft/LLaVA-Med)
4. **Qwen-2-VL (LLaMA-Factory Fine-tuning)**: [LLaMA-Factory GitHub](https://github.com/hiyouga/LLaMA-Factory)
5. **GPT-4o (Fine-tuning)**: [GPT-4o Fine-tuning](https://openai.com/index/gpt-4o-fine-tuning/)
6. **Gemini-1.5-pro (No Fine-tuning)**: [Gemini-1.5-pro](https://deepmind.google/technologies/gemini/pro/)

#### 4.2 Lung Cancer Segmentation Comparison
We compare **HiCur-CXR** with open-source models on the lung cancer segmentation task. Specifically, we compare:
- **MedSAM**: We perform detailed fine-tuning following the methods provided in its repository.
- **SA-UNet**: The previous state-of-the-art model on this dataset.

The fine-tuning links are as follows:
1. **SA-UNet**: [SA-UNet GitHub](https://github.com/clguo/SA-UNet)
2. **MedSAM**: [MedSAM GitHub](https://github.com/bowang-lab/MedSAM)

## Model Weights
The weights for all models used in this study will be publicly available and continuously updated at:  
**[CASMI Model Hub](http://www.radiomics.net.cn/post/143)**

## Conclusion
This repository provides a comprehensive exploration of transferring the HFFCL method to CXR tasks, with detailed comparisons against state-of-the-art models in both VQA and lung cancer segmentation. The results demonstrate the effectiveness of **HiCur-CXR** in these domains.

For further details, please refer to the respective dataset and model repositories linked above.