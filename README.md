# Edge-to-Face Realistic Face Generation

A PyTorch-based repository containing deep learning models for generating realistic human faces from sketches or edge maps. This project compares standard conditional GAN architectures like **Pix2Pix** against advanced encoder-decoder networks enhanced with **Attention Gates** and **Residual Blocks** (Deep Attention U-Net variants).

---

## 🚀 Key Features

* **Multiple Architectures**: Baseline Pix2Pix models compared against Deep Attention U-Nets (standard, small residual, normal residual, and big residual variants).
* **Attention-Guided Synthesis**: Integrates channel and spatial soft-attention gates on the U-Net skip connections to prioritize facial features (eyes, nose, mouth) while suppressing background noise.
* **Residual Connections**: Enhances information flow and prevents vanishing gradients, enabling deeper generator networks.
* **Perceptual & Structural Evaluation**: Includes pipeline scripts to compute Perceptual Similarity (**LPIPS**) and Structural Similarity (**SSIM**).
* **Hand-drawn Sketch Support**: Testing tools configured for evaluation on hand-drawn sketch datasets (e.g., CUHK Face Sketch Database and AR Face Database).

---

## 📂 Project Structure

```
edge-to-face-realistic-generation/
├── src/
│   ├── PerceptualSimilarity/        # Dependency submodule for computing LPIPS metrics
│   ├── compute_lpips.py             # Evaluates LPIPS distance between real and generated faces
│   ├── ssim.py                      # Computes Structural Similarity Index Measure (SSIM)
│   ├── create_data.py               # Preprocessing (Canny edge map creation, downscaling, horizontal concatenation)
│   ├── dataset.py                   # PyTorch dataset loader using Torchvision transforms
│   ├── pix2pix.py                   # Standard Pix2Pix generator/discriminator & Albumentations dataset loader
│   ├── network.py                   # General model definitions (UNet with/without Attention)
│   ├── deep_att.py                  # Module for GeneratorDeepWithAttention & PatchGAN Discriminator
│   ├── deep_att_res.py              # Generator model utilizing standard Residual Blocks and Attention Gates
│   ├── deep_att_small_res.py        # Generator model utilizing smaller Residual Blocks and Attention Gates
│   ├── deep_att_big_res.py          # Generator model utilizing deeper/larger Residual Blocks and Attention Gates
│   ├── expo.py                      # Side-by-side comparison visualization exporter (Pix2Pix vs. Att vs. ResAtt)
│   ├── generate.py                  # Batch image generation for Deep Attention variants
│   ├── generate_pix2pix.py          # Batch image generation for Pix2Pix
│   ├── test.py                      # Batch validation visual tester (generates edge + face grid)
│   ├── test_one.py                  # Evaluates SSIM and visual quality on a single input image/sketch
│   ├── test_pix2pix.py              # Batch validation visual tester for Pix2Pix
│   ├── test_sketch.py               # Specific inference pipelines for hand-drawn datasets (CUHK & AR)
│   └── mean_and_std.py              # Basic dataset diagnostic script
└── README.md                        # Project documentation (this file)
```

---

## 🛠️ Installation & Dependencies

To set up the environment, ensure you have python 3.8+ installed, then install the required packages:

```bash
pip install torch torchvision numpy opencv-python albumentations Pillow scikit-image prettytable pytorch-msssim scipy tqdm
```

Make sure the LPIPS dependency `PerceptualSimilarity` is checked out if evaluating LPIPS:
```bash
git submodule update --init --recursive
```

---

## 🤖 Models & Architectures

The project contains several generator models matching different experiments:

| Model | Script / Module | Description |
| :--- | :--- | :--- |
| **Pix2Pix Generator** | `src/pix2pix.py` | Baseline conditional GAN U-Net. Expects 3-channel input/output. |
| **GeneratorDeepWithAttention** | `src/network.py`, `src/deep_att.py` | Deeper U-Net with Attention Gates on skip connections. Expects 1-channel grayscale edge map input. |
| **Generator (Residual + Attention)** | `src/deep_att_res.py` | Replaces standard U-Net convolutions with robust Residual Blocks combined with Attention Gates. |
| **Generator (Small Residual + Attention)**| `src/deep_att_small_res.py`| A lightweight version of the Residual-Attention generator to minimize parameter count. |
| **Generator (Big Residual + Attention)**  | `src/deep_att_big_res.py`  | Deepened residual representation for capturing high-frequency spatial details. |

### Attention Gate Block
The soft-attention gates (`AttentionGate` module) perform channel and spatial attention on skip-connection activations $x$ using the gating signal $g$ from the deeper upsampling layer:

$$\theta_i = W_g^T g + W_x^T x$$
$$\alpha = \sigma(\psi^T(\text{ReLU}(\theta_i)))$$
$$\text{Output} = \alpha \cdot x$$

---

## 📊 Dataset & Preprocessing

### Horizontal Concatenation Format
Like standard Pix2Pix, the dataset files are formatted as horizontally concatenated $256 \times 512$ images. The left $256 \times 256$ region holds the edge map input, and the right $256 \times 256$ region holds the target face image.

### Preprocessing Pipeline (`src/create_data.py`)
1. **Edge Map Generation**: Extracts edge maps using an adaptive Canny threshold based on mean intensity:
   * Lower Threshold: $l = 0.66 \times \text{mean}$
   * Upper Threshold: $u = 1.33 \times \text{mean}$
2. **Concatenation**: Pairs the edge maps side-by-side with original photos.
3. **Splitting**: Assigns 8% of the images to validation (`val/`) and 80% to train (`train/`) (e.g. 80-20 split by checking `n % 5 == 0`).

To run preprocessing:
```bash
python src/create_data.py
```

---

## ⚡ Execution & Evaluation

### 1. Generating Outputs
To run batch generation on the validation dataset using your trained check-pointed model weight `.pth.tar`:
* For Deep Attention models:
  ```bash
  python src/generate.py
  ```
* For Pix2Pix models:
  ```bash
  python src/generate_pix2pix.py
  ```

### 2. Computing Metrics
To run standard metrics:
* **LPIPS Distance** (Perceptual loss score):
  ```bash
  python src/compute_lpips.py
  ```
* **Structural Similarity Index (SSIM)**:
  ```bash
  python src/ssim.py
  ```

### 3. Visual Grid Comparisons
To export visual comparisons showcasing generated faces from Pix2Pix, Attention U-Net, and Residual Attention U-Net:
```bash
python src/expo.py
```

---

## 🎓 Acknowledgments
This repository is created as part of a license graduation project ("Licență") focusing on deep learning methods for face sketch synthesis and image-to-image translations.
