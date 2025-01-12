# Implementation of Motion Mamba

## Overview
This repository provides an implementation of the **Motion Mamba** paper. The work draws significant inspiration from the key ideas and codebases of several related projects, as detailed below.

---

## Inspired by
- [Mamba](https://github.com/state-spaces/mamba): A state-space model-based architecture, licensed under Apache License 2.0.
- [MLD (Motion Latent Diffusion)](https://github.com/ChenFengYe/motion-latent-diffusion): A diffusion-based approach for motion generation, licensed under MIT License.
- [DiM (Diffusion Mamba)](https://github.com/tyshiwo1/DiM-DiffusionMamba?tab=readme-ov-file): A diffusion-based approach for image generation. Note: This repository does not provide an explicit license.
- [ViM (Vision Mamba)](https://github.com/hustvl/Vim): A Vision Transformer-inspired variant of Mamba, licensed under Apache License 2.0.
- [Motion Mamba](https://github.com/steve-zeyu-zhang/MotionMamba/): The official repository for the Motion Mamba paper. Note: This repository does not provide an explicit license.

---

## Key Contributions
1. **Mamba Block Construction**:
   - Combines the **Mamba Block design approach from the DiM project** with **code implementation techniques from the ViM project**.
   - Ensures compliance with the Apache License 2.0 as required by Mamba and ViM.

2. **Training Pipeline**:
   - The training process integrates methodologies introduced in the **MLD project**, which is licensed under MIT License.

3. **Code Integration**:
   - All code is developed with respect to the licenses and intellectual property of the referenced repositories.

---

## License

This project is licensed under:
- **Apache License 2.0**: Applies to portions of the code derived from [Mamba](https://github.com/state-spaces/mamba) and [ViM](https://github.com/hustvl/Vim).
- **MIT License**: Applies to portions of the code derived from [MLD](https://github.com/ChenFengYe/motion-latent-diffusion).

### Legal Notes:
1. Portions of this project are inspired by [DiM (Diffusion Mamba)](https://github.com/tyshiwo1/DiM-DiffusionMamba) and [Motion Mamba](https://github.com/steve-zeyu-zhang/MotionMamba). However, neither repository provides an explicit license. As such, no direct reuse of their code has been made unless explicitly permitted by the respective authors.
2. If you plan to use or redistribute this code, ensure compliance with the licenses of the referenced projects.

Refer to the `LICENSE` file for full license details.

---

## Acknowledgments

This project builds upon the ideas and implementations from the following repositories:
- [Mamba](https://github.com/state-spaces/mamba): Apache License 2.0.
- [ViM (Vision Mamba)](https://github.com/hustvl/Vim): Derived from Mamba, Apache License 2.0.
- [MLD (Motion Latent Diffusion)](https://github.com/ChenFengYe/motion-latent-diffusion): MIT License.
- [DiM (Diffusion Mamba)](https://github.com/tyshiwo1/DiM-DiffusionMamba): No explicit license provided.
- [Motion Mamba](https://github.com/steve-zeyu-zhang/MotionMamba): Official repository for the Motion Mamba paper, no explicit license provided.

We sincerely thank the authors of these projects for their groundbreaking work and inspiration.

---

## Requirements

This implementation heavily relies on the requirements of the DiM and MLD projects.

1. **Install DiM Dependencies**: Follow the installation steps provided in the [DiM repository](https://github.com/tyshiwo1/DiM-DiffusionMamba).

2. **Install Additional Dependencies**: Use the provided `requirements.txt` file in this repository to install additional dependencies required for the [MLD](https://github.com/ChenFengYe/motion-latent-diffusion)-based training pipeline.

   - Additionally, `causal-conv1d` is pinned to version **1.1.1** to address specific issues in Mamba's implementation. For more details, refer to [ViM Issue#41](https://github.com/hustvl/Vim/issues/41).

3. **Download Required Resources**: Run the following scripts to download necessary files and dependencies for [MLD (Motion Latent Diffusion)](https://github.com/ChenFengYe/motion-latent-diffusion):
   - For **MLD dependencies**:
     ```bash
     bash prepare/download_smpl_model.sh
     bash prepare/prepare_clip.sh
     ```
   - For **Text-to-Motion Evaluation**:
     ```bash
     bash prepare/download_t2m_evaluators.sh
     ```

---


## System Requirements

- **Operating System**: Linux
- **Hardware**: NVIDIA GPU
- **Framework**: PyTorch 2.1.1

## Train your Model
This process is largely based on the [MLD](https://github.com/ChenFengYe/motion-latent-diffusion) project.  
<details>
  <summary><b>Training guidance</b></summary>  

### 1. Prepare datasets
Please refer to [HumanML3D](https://github.com/EricGuo5513/HumanML3D) for text-to-motion dataset setup. We will provide instructions for other datasets soon.

### 2.1. Ready to train VAE model

Please first check the parameters in `configs/config_vae_humanml3d.yaml`, e.g. `NAME`,`DEBUG`.

Then, run the following command:

```
python -m train --cfg configs/config_vae_humanml3d.yaml --cfg_assets configs/assets.yaml --batch_size 64 --nodebug
```
**※ Caution ※**  
If you want to follow the Motion Mamba paper directly, ensure your **VAE latent shape** is `[2, 256]`, not `[1, 256]`.  

Please update the **VAE latent shape** in `configs/config_vae_humanml3d.yaml`, specifically the `model.latent_dim`.

### 2.2. Ready to train Motion Mamba Model

Please update the parameters in `configs/config_motionmamba_1_humanml3d.yaml`, e.g. `NAME`,`DEBUG`,`PRETRAINED_VAE` (change to your `latest ckpt model path` in previous step)

Then, run the following command:

```
python -m train --cfg configs/config_motionmamba_1_humanml3d.yaml --cfg_assets configs/assets.yaml --batch_size 64 --nodebug
```

If you are using MLD's VAE, ensure your config's `latent_dim` is `[1, 256]`.  
If you want to fully follow the Motion Mamba paper, you need to train your VAE from scratch.


### 3. Evaluate the model

Please first put the tained model checkpoint path to `TEST.CHECKPOINT` in `configs/config_mld_humanml3d.yaml`.

Then, run the following command:

```
python -m test --cfg configs/config_motionmamba_1_humanml3d.yaml --cfg_assets configs/assets.yaml
```

</details>

<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>

# Motion Mamba Paper
<div align="center"><h1> Motion Mamba: Efficient and Long Sequence Motion Generation<br>
<sub><sup><a href="https://eccv2024.ecva.net/">ECCV 2024</a></sup></sub>
</h1>

[Zeyu Zhang](https://steve-zeyu-zhang.github.io)<sup>\*</sup>, [Akide Liu](https://www.linkedin.com/in/akideliu/)<sup>\*</sup>, [Ian Reid](https://mbzuai.ac.ae/study/faculty/ian-reid/), [Richard Hartley](http://users.cecs.anu.edu.au/~hartley/), [Bohan Zhuang](https://bohanzhuang.github.io/), [Hao Tang](https://ha0tang.github.io/)<sup>✉</sup>

<sup>*</sup>Equal contribution
<sup>✉</sup>Corresponding author: bjdxtanghao@gmail.com

[![Website](https://img.shields.io/badge/Website-Demo-fedcba?style=flat-square)](https://steve-zeyu-zhang.github.io/MotionMamba/) [![arXiv](https://img.shields.io/badge/arXiv-2403.07487-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2403.07487) [![Papers With Code](https://img.shields.io/badge/Papers%20With%20Code-555555.svg?style=flat-square&logo=data:image/svg%2bxml;base64,PHN2ZyB4bWxuczp4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB2aWV3Qm94PSIwIDAgNTEyIDUxMiIgd2lkdGg9IjUxMiIgIGhlaWdodD0iNTEyIiA+PHBhdGggZD0iTTg4IDEyOGg0OHYyNTZIODh6bTE0NCAwaDQ4djI1NmgtNDh6bS03MiAxNmg0OHYyMjRoLTQ4em0xNDQgMGg0OHYyMjRoLTQ4em03Mi0xNmg0OHYyNTZoLTQ4eiIgc3Ryb2tlPSIjMjFDQkNFIiBmaWxsPSIjMjFDQkNFIj48L3BhdGg+PHBhdGggZD0iTTEwNCAxMDRWNTZIMTZ2NDAwaDg4di00OEg2NFYxMDR6bTMwNC00OHY0OGg0MHYzMDRoLTQwdjQ4aDg4VjU2eiIgc3Ryb2tlPSIjMjFDQkNFIiBmaWxsPSIjMjFDQkNFIj48L3BhdGg+PC9zdmc+)](https://paperswithcode.com/paper/motion-mamba-efficient-and-long-sequence) [![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-555555?style=flat-square)](https://huggingface.co/papers/2403.07487) [![BibTeX](https://img.shields.io/badge/BibTeX-Citation-eeeeee?style=flat-square)](https://steve-zeyu-zhang.github.io/MotionMamba/static/scholar.html)


</div>

_Human motion generation stands as a significant pursuit
in generative computer vision, while achieving long-sequence and efficient motion generation remains challenging. Recent advancements in
state space models (SSMs), notably Mamba, have showcased considerable promise in long sequence modeling with an efficient hardware-aware
design, which appears to be a promising direction to build motion generation model upon it. Nevertheless, adapting SSMs to motion generation faces hurdles since the lack of a specialized design architecture to
model motion sequence. To address these challenges, we propose **Motion
Mamba**, a simple and efficient approach that presents the pioneering
motion generation model utilized SSMs. Specifically, we design a **Hierarchical Temporal Mamba** (**HTM**) block to process temporal data by
ensembling varying numbers of isolated SSM modules across a symmetric U-Net architecture aimed at preserving motion consistency between
frames. We also design a **Bidirectional Spatial Mamba** (**BSM**) block to bidirectionally process latent poses, to enhance accurate motion generation within a temporal frame. Our proposed method achieves up to **50%** FID improvement and up to **4** times faster on the HumanML3D and
KIT-ML datasets compared to the previous best diffusion-based method,
which demonstrates strong capabilities of high-quality long sequence motion modeling and real-time human motion generation._

<div align="center">
<img src="static/images/main.svg" style="width: 100%;">
<img src="static/images/block.svg" style="width: 80%;">
</div>

## News

<b>(07/22/2024)</b> &#127881; Our paper was invited for a talk at <a href="https://www.mihoyo.com/"><b>miHoYo</b></a>. You can find our slides <a href="https://steve-zeyu-zhang.github.io/MotionMamba/static/pdfs/Motion_Mamba_Slides_miHoYo.pdf"><b>here</b></a>!

<b>(07/05/2024)</b> &#127881; Our paper has been highlighted twice by <a href="https://wx.zsxq.com/dweb2/index/topic_detail/5122458815888184"><b>CVer</b></a>!

<b>(07/02/2024)</b> &#127881; Our paper has been accepted to <a href="https://eccv2024.ecva.net/"><b>ECCV 2024</b></a>!

<b>(03/15/2024)</b> &#127881; Our paper has been highlighted by <a href="https://twitter.com/Marktechpost/status/1768770427680424176"><b>MarkTechPost</b></a>!

<b>(03/13/2024)</b> &#127881; Our paper has been featured in <a href="https://twitter.com/_akhaliq/status/1767750847239262532"><b>Daily Papers</b></a>!

<b>(03/13/2024)</b> &#127881; Our paper has been highlighted by <a href="https://wx.zsxq.com/dweb2/index/topic_detail/1522541851241522"><b>CVer</b></a>!

## Citation

```
@inproceedings{zhang2025motion,
  title={Motion Mamba: Efficient and Long Sequence Motion Generation},
  author={Zhang, Zeyu and Liu, Akide and Reid, Ian and Hartley, Richard and Zhuang, Bohan and Tang, Hao},
  booktitle={European Conference on Computer Vision},
  pages={265--282},
  year={2025},
  organization={Springer}
}
```

## Acknowledgements

- [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://github.com/state-spaces/mamba)

- [Motion Latent Diffusion: Executing your Commands via Motion Diffusion in Latent Space](https://github.com/ChenFengYe/motion-latent-diffusion)
