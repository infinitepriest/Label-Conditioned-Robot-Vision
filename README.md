# Label-Conditioned Robot Vision
**Semantic-Conditioned Generative Perception for Robotics**

---

## Overview

**Label-Conditioned Robot Vision** is a robotics-oriented perception project that studies **semantic controllability in visual generation** using **Conditional Generative Adversarial Networks (cGANs)**.

The project explores how symbolic labels (high-level semantic commands) can condition visual outputs across increasing levels of perceptual complexity, ranging from simple digits to fine-grained natural scenes. The work is framed from a **Robotics** perspective, focusing on **perception modeling, semantic grounding, and controllable world representation**, rather than purely photorealistic image synthesis.

---

## Project Motivation (Robotics Perspective)

In robotics, perception systems must operate under:
- semantic uncertainty,
- environmental clutter,
- limited or imbalanced data.

This project investigates whether **explicit semantic conditioning** can guide visual perception modules to:
- maintain class-level consistency,
- scale to large semantic spaces,
- expose failure modes relevant to real robotic sensing.

The goal is **not artistic image generation**, but understanding **how controllable perception behaves under complexity**.

---

## Datasets Used

| Dataset | Purpose | Classes | Resolution |
|-------|--------|---------|------------|
| **MNIST** | Semantic controllability baseline | 10 | 28×28 (grayscale) |
| **Oxford-102 Flowers** | Natural scene perception | 102 | 64×64 (RGB) |
| **CUB-200 Birds** | Fine-grained perception stress test | 200 | 96×96 (RGB) |

---

## **Mathematical Modeling: Label-Conditioned Perceptual Generation**

This project formulates label-conditioned robot vision as a **conditional generative modeling problem**, where the goal is to synthesize visual observations that are both *realistic* and *semantically consistent* with a given symbolic label.

### Problem Formulation

Let:
- $z \sim \mathcal{N}(0, I)$ be a latent noise vector,
- $y \in \{0,1\}^K$ be a one-hot encoded semantic label over $K$ classes,
- $x \in \mathbb{R}^{C \times H \times W}$ be a visual observation (image).

The generator learns a mapping:

$G(z, y) \rightarrow \hat{x}$

where $\hat{x}$ is a synthetic image conditioned on the semantic prior $y$.

The discriminator evaluates:

$D(x, y) \rightarrow [0,1]$

representing the probability that image $x$ is real **and** consistent with label $y$.

### Conditional Adversarial Objective

Training follows a **conditional GAN (cGAN)** objective:

min_G max_D →
E_{x,y ~ p_data}[log D(x,y)]
+ E_{z ~ p(z), y ~ p(y)}[log(1 − D(G(z,y), y))]

- The **Discriminator** maximizes its ability to distinguish real from generated samples under semantic conditioning.
- The **Generator** minimizes this objective by producing samples that are indistinguishable from real observations *given the same label*.

### Semantic Conditioning Mechanism

Semantic labels are injected explicitly into both networks:

- **Generator**:
  - Labels are concatenated with latent vectors at the input level.
  - For convolutional architectures, labels are spatially expanded and concatenated with feature maps at multiple scales.

- **Discriminator**:
  - Labels are similarly concatenated with input images or intermediate feature maps.
  - This enforces **semantic consistency verification**, not just visual realism.

This multi-scale conditioning mirrors robotics perception systems where **semantic priors influence interpretation at multiple abstraction levels**.

### Dataset-Specific Modeling Choices

| Dataset | Modeling Rationale |
|------|------------------|
| MNIST | Fully connected cGAN sufficient due to low visual entropy |
| Oxford-102 | Convolutional cGAN captures color and texture statistics |
| CUB-200 | Higher-resolution convolutional cGAN to retain fine-grained cues |

The latent dimension $z = 100$ is held constant across datasets to isolate the effect of **semantic complexity and environmental variability**.

### Robotics Interpretation

This formulation treats perception as a **probabilistic mapping from semantic intent to plausible sensory observations**.

Rather than predicting a single deterministic image, the model learns a **distribution over possible observations** conditioned on a task-relevant semantic label. This aligns with how real robotic sensors operate under uncertainty, clutter, and incomplete information.

The mathematical framework thus provides a principled basis for:
- controllable perception,
- semantic grounding,
- and diagnostic analysis of perception failure modes.

---

## Methodology

### 1) Label-Conditioned Perception
Each model learns a mapping:
> **(noise + semantic label) → visual observation**

Labels are one-hot encoded and injected into both the **Generator** and **Discriminator**, enforcing semantic alignment.

### 2) Architecture Progression
- **MNIST**: Fully Connected cGAN (baseline reproduction)
- **Oxford-102 / CUB-200**: Deep Convolutional cGANs with transposed convolutions and spatial label conditioning

### 3) Training Strategy
- Binary Cross-Entropy loss
- Adam optimizer (β = 0.5, 0.999)
- Fixed latent dimension (z = 100)
- Periodic qualitative sampling and quantitative evaluation

---

## Evaluation Metrics

The models are evaluated using **distribution-level metrics**:
- **FID (Fréchet Inception Distance)**: realism & distribution alignment
- **Inception Score (IS)**: diversity & class awareness

Detailed results are available in **Evaluation (FID&IS).pdf**.

### Summary of Results

| Dataset | FID ↓ | Inception Score ↑ |
|-------|-------|------------------|
| MNIST | 6.24 | 1.29 ± 0.01 |
| Oxford-102 | 3.41 | 2.72 ± 0.21 |
| CUB-200 | 4.67 | 3.84 ± 0.26 |

---

## Output Interpretation (See the main notebook)

- **MNIST** validates semantic control.
- **Oxford-102** captures dominant colors and coarse structure.
- **CUB-200** preserves class-level color priors but struggles with background clutter, pose variability, and fine details.

These behaviors mirror **real robotic sensor observations** in cluttered environments.

---

## Repository Structure

```
Label-Conditioned-Robot-Vision/
├── data/                                  # Would be created automatically as the jupyter notebook is run
├── outputs/                               # Would be created automatically as the jupyter notebook is run
├── Label-Conditioned-Robot-Vision.ipynb   # The main jupyter notebook (standalone; end-to-end)
├── Evaluation (FID&IS).pdf                # for Authors' execution
├── requirements.txt
├── LICENSE
└── README.md                              # This file
```

---

## How to Run

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Open **Label-Conditioned-Robot-Vision.ipynb** and run cells in order.

---

## What We Achieved

- Reproduced CGAN baseline
- Scaled to 100+ and 200-class datasets
- Demonstrated semantic controllability
- Identified robotics-relevant perception limits

---

## Future Potential Extensions
- Semantic embeddings
- Attention or diffusion models
- Foreground–background disentanglement
- Sim-to-real augmentation

---

## Contributions

### Snavya Sai Munti Mudugu Badri Prasad  
https://github.com/snavya0309  
- CGAN baseline reproduction
- MNIST experiments

### Sushma Niresh  
https://github.com/Sushmaniresh  
- Dataset pipelines
- Architecture design
- Label conditioning

### Ayushman Mishra  
https://github.com/aymisxx  
- Architecture design
- Training and evaluation
- Robotics perspectivization
- Repo management and reproducibility
- Convolutional cGAN extensions

---

## References
1. Mirza & Osindero (2014). Conditional GANs  
2. Reed et al. (2016). Text-to-Image GANs  
3. Jennewein et al. (2023). Sol Supercomputer at ASU

---