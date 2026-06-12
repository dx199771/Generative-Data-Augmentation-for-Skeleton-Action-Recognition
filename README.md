# Generative Data Augmentation for Skeleton Action Recognition


**Accepted at 20th IEEE International Conference on Automatic Face and Gesture Recognition (IEEE FG'26) as Oral Presentation.**

[![Paper](https://img.shields.io/badge/Paper-IEEE%20FG%202026-blue)](https://arxiv.org/abs/2604.14933)
[![Project Page](https://img.shields.io/badge/Page-ProjectPage-red)](https://andrewjohngilbert.github.io/SkelActRec/)

---

## 📄 Abstract

With only a small set of labelled skeleton sequences, the model generates diverse and high-fidelity samples. When combined with a reduced amount of real data for training, these synthetic samples enable our skeleton action recognisers to achieve performance close to the state of the art on HumanAct12 and Refined NTU RGB+D.

---

## 🔧 Installation

Python 3.8+ and mmcv 1.7.1 are required.

```bash
pip install torch torchvision
pip install mmcv==1.7.1
pip install -r requirements.txt
```

---

## 📂 Datasets

Download the datasets and place them under `data/`.

| Dataset | Original Dataset | Our 263 Features |
|---|---|---|
| HumanAct12 | [Download](https://ericguo5513.github.io/action-to-motion/#data) | [Download](https://drive.google.com/file/d/1wWcDZ4Assjx4Mlh4L0P8c2632QZTKupu/view?usp=sharing) |
| Refined NTU RGB+D | [Download](https://ericguo5513.github.io/action-to-motion/#data) | [Download](https://drive.google.com/file/d/1xdQdNKa8voIUwKDqIALdXufOuIyF__N-/view?usp=sharing) |

Expected structure:

```
data/
├── humanact12/
│   ├── new_joint_vecs_humanact/
│   ├── test_mean_std/
│   ├── train_mean_std/
│   ├── 48_test.pkl
│   ├── 48_train.pkl
│   └── humanact12_old_new_map.json
└── ntuvibe/
    ├── new_joint_vecs_vibe/
    ├── vibe_48_new_test/
    └── vibe_48_new_train/
```

---

## 🚀 Training

```bash
# HumanAct12
python3 main.py --config configs/cfg_humanact.py

# NTU RGB+D Vibe
python3 main.py --config configs/cfg_ntuvibe.py
```

---

## 🎲 Sampling

First clone MDM to use its `recover_from_ric` function:

```bash
git clone https://github.com/GuyTevet/motion-diffusion-model.git
```

Then run sampling:

```bash
# HumanAct12
python3 sample.py --config configs/cfg_humanact.py

# NTU RGB+D Vibe
python3 sample.py --config configs/cfg_ntuvibe.py
```

---

## 📁 Project Structure

```
├── configs/
│   ├── cfg_humanact.py
│   └── cfg_ntuvibe.py
├── data/
│   ├── humanact12/
│   └── ntuvibe/
├── dataloaders/
│   ├── humanact12.py
│   └── nturgbvibe.py
├── models/
│   ├── __init__.py
│   ├── cgn.py
│   ├── diffusion.py
│   ├── losses.py
│   ├── nn.py
│   └── sampler.py
├── .gitignore
├── main.py
├── README.md
├── requirements.txt
├── sample.py
└── utils.py
```

---

## 📊 Evaluation

The generated synthetic data is trained and tested across three standard SOTA backbones provided by [PySkl](https://github.com/kennymckormick/pyskl), and [BlockGCN](https://github.com/zhouyuxuanyx/blockgcn) implemented as an independent baseline.

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
