# Generative Data Augmentation for Skeleton Action Recognition

# Code is coming in 2 days. #

**Accepted at 20th IEEE International Conference on Automatic Face and Gesture Recognition (IEEE FG’26) as Oral Presentation.**

[![Paper](https://img.shields.io/badge/Paper-IEEE%20FG%202026-blue)]([https://your-paper-link-here](https://arxiv.org/abs/2604.14933))
[![Project Page](https://img.shields.io/badge/Page-ProjectPage-red)]([https://your-paper-link-here](https://andrewjohngilbert.github.io/SkelActRec/))

---

## 📄 Abstract

With only a small set of labelled skeleton sequences, the model generates diverse and high-fidelity samples. When combined with a reduced amount of real data for training, these synthetic samples enable our skeleton action recognisers to achieve performance close to the state of the art on HumanAct12 and Refined NTU RGB+D.

---



## 🔧 Installation

Python 3.8+ is required.

```bash
pip install torch torchvision
pip install numpy tqdm timm mmcv
```

---

## 📂 Datasets

Download the datasets and place them under `data/`.

| Dataset | Original Dataset | Our 263 Features|
|---|---|---|
| HumanAct12 | [Download]() | |
| Refined NTU RGB+D | [Download]() | |

Expected structure:

```
data/
├── humanact12/
│   └── NTU60_XSub.npz
└── vibe/
```

---

## 🚀 Training

```bash
# HumanAct12-22
python3 main.py --config configs/humanact1222.py

# NTU RGB+D Vibe
python3 main.py --config configs/nturgbvibe.py
```


---

## 📁 Project Structure

```
├── configs/           # experiment config files
├── dataloaders/       # dataset loaders
├── models/            # MDM model and diffusion
│   ├── MDM.py
│   ├── diffusion.py
│   └── sampler.py
├── utils/
│   └── utils.py
└── main.py
```

---

## 📜 Citation

If you find this work useful, please cite:

```bibtex
@misc{dong2026generativedataaugmentationskeleton,
      title={Generative Data Augmentation for Skeleton Action Recognition}, 
      author={Xu Dong and Wanqing Li and Anthony Adeyemi-Ejeye and Andrew Gilbert},
      year={2026},
      eprint={2604.14933},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2604.14933}, 
}
```

---

## 🙏 Acknowledgements

This codebase builds upon [MDM](https://github.com/GuyTevet/motion-diffusion-model) and [guided-diffusion](https://github.com/openai/guided-diffusion).
