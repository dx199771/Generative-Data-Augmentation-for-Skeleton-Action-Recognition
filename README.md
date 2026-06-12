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

Python 3.8+ and mmcv 1.7.1 are required.

```bash
pip install torch torchvision
pip install mmcv==1.7.1
# pip install numpy tqdm timm
pip install -r requirements.txt
```

---

## 📂 Datasets

Download the datasets and place them under `data/`.

| Dataset | Original Dataset | Our 263 Features|
|---|---|---|
| HumanAct12 | [Download]() | [Download](https://drive.google.com/file/d/1wWcDZ4Assjx4Mlh4L0P8c2632QZTKupu/view?usp=sharing) |
| Refined NTU RGB+D | [Download]() | [Download](https://drive.google.com/file/d/1xdQdNKa8voIUwKDqIALdXufOuIyF__N-/view?usp=sharing) |

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
# HumanAct12
python3 main.py --config configs/humanact1222.py

# NTU RGB+D Vibe
python3 main.py --config configs/nturgbvibe.py
```

## Sampling

# clone MDM first to use its recover_from_ric function
```bash
git clone https://github.com/GuyTevet/motion-diffusion-model.git
```

```bash
# HumanAct12
python3 sample.py --config configs/cfg_humanact.py 

# NTU RGB+D Vibe
python3 sample.py --config configs/cfg_nturgbvibe.py 

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

### Evaluation framework
To evaluate our framework, the generated synthetic data is trained and tested across the three standard SOTA backbones provided by [PYSKL](https://github.com/kennymckormick/pyskl), and [BlockGCN](https://github.com/zhouyuxuanyx/blockgcn) implemented as an independent baseline.


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
