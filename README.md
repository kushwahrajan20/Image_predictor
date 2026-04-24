# 🤖 AI vs Real Image Detection

A deep learning framework for distinguishing between real photographs and AI-generated images using a dual-branch CNN architecture that combines spatial and frequency-domain analysis.

---

## 📄 Overview

With the rapid advancement of generative models (GANs, Diffusion Models), it has become increasingly difficult to distinguish real images from AI-generated ones. This project proposes a **Dual-Stream CNN-FFT model** that captures both spatial artifacts and frequency-domain inconsistencies to classify images as either **real** or **AI-generated**.

The model achieves **78.78% accuracy** on unseen test data, outperforming the baseline CNN model (73.09%).

---

## 🏗️ Architecture

The proposed model is a **dual-branch deep neural network**:

### Branch 1 — Spatial Feature Extraction
Processes raw RGB images (`224×224×3`) through:
- Stacked Convolutional blocks with Batch Normalization & ReLU
- **Residual Blocks** for gradient flow and deeper feature learning
- **Squeeze-and-Excitation (SE) Attention Blocks** for channel-wise recalibration
- Max Pooling for spatial downsampling
- Global Average Pooling → Dropout for regularization

### Branch 2 — Frequency Domain (FFT)
Captures generative artifacts invisible to the human eye:
- Custom **FFT Layer**: converts images to grayscale → 2D FFT → log magnitude → normalize
- Convolutional + Residual Blocks with SE Attention
- Global Average Pooling → Dropout

### Fusion & Classification
Both branches are concatenated and passed through:
```
Dense(256) → BN → Dropout(0.5)
Dense(128) → Dropout(0.4)
Dense(64)  → Dropout(0.3)
Sigmoid Output → {AI Image, Real Image}
```

### Architecture Diagrams

| Full Dual-Stream Architecture | Sub-modules (Residual & SE Blocks) |
|---|---|
| ![Architecture](image1.png) | ![Sub-modules](image2.png) |

---

## 📊 Results

### Dual-Stream CNN-FFT Model (Proposed)

| Metric | Value |
|---|---|
| Test Accuracy | **78.78%** |
| Precision | 77.6% |
| Recall | 80.8% |
| F1-Score | 79.1% |
| Train Accuracy | 81.69% |
| Val Accuracy | 80.88% |

### Baseline CNN Model

| Metric | Value |
|---|---|
| Test Accuracy | 73.09% |
| Precision | 70.82% |
| Recall | 78.51% |

### Training Curves

| Baseline Model | Dual-Stream FFT Model |
|---|---|
| ![Baseline](baseline_performance.jpg) | ![Dual Stream](dualStream_performance.jpg) |

### Confusion Matrix (Dual-Stream Model)

![Confusion Matrix](confusion_matrix.png)

| | Predicted: Real | Predicted: AI |
|---|---|---|
| **True: Real** | 1862 ✅ | 442 ❌ |
| **True: AI** | 536 ❌ | 1768 ✅ |

---

## 📁 Dataset

The model is trained on **COCO_AI**, a dataset designed for real vs. AI image classification.

- **Real images**: sourced from the [COCO dataset](https://cocodataset.org/)
- **AI-generated images**: created using Stable Diffusion variants, DALL-E, and MidJourney
- **Format**: Parquet files with images encoded as byte streams
- **Preprocessing**: Decoded → Resized to `224×224` → Normalized to `[0, 1]`
- **Sampling**: Balanced — one real image and one randomly selected AI image per record
- **Pipeline**: TensorFlow `tf.data` API with generator-based lazy loading

---

## ⚙️ Training Configuration

| Parameter | Value |
|---|---|
| Input Size | `224 × 224 × 3` |
| Batch Size | 32 |
| Epochs | 10 |
| Optimizer | Adam |
| Learning Rate | `1e-4` |
| Loss Function | Binary Cross-Entropy |
| Regularization | Dropout, Batch Normalization, L2 weight decay |

---

## 🛠️ Setup & Usage

### Prerequisites
```bash
pip install tensorflow numpy pandas pyarrow matplotlib scikit-learn
```

### Clone the Repository
```bash
git clone https://github.com/<your-username>/ai-vs-real-image-detection.git
cd ai-vs-real-image-detection
```

### Training
```bash
python train.py
```

### Evaluation
```bash
python evaluate.py --model_path saved_model/ --test_data path/to/test/
```

### Inference
```bash
python predict.py --image path/to/image.jpg
```

---

## 📂 Project Structure

```
ai-vs-real-image-detection/
│
├── models/
│   ├── spatial_branch.py       # Spatial CNN branch
│   ├── fft_branch.py           # Frequency domain branch
│   ├── dual_stream_model.py    # Full dual-stream model
│   └── blocks.py               # Residual & SE Attention blocks
│
├── data/
│   ├── dataset_loader.py       # tf.data pipeline & Parquet loader
│   └── preprocessing.py        # Image preprocessing utilities
│
├── train.py                    # Training script
├── evaluate.py                 # Evaluation script
├── predict.py                  # Single image inference
│
├── results/
│   ├── confusion_matrix.png
│   ├── baseline_performance.jpg
│   └── dualStream_performance.jpg
│
└── README.md
```

---

## 🔬 Key Findings

- The dual-stream model improves overall accuracy by **~5.7%** over the baseline CNN.
- Combining **spatial** and **frequency-domain** features reduces overfitting, with only a ~0.81% gap between training and validation accuracy.
- The model shows stronger performance on AI-generated images (recall: 80.8%) compared to real images, reflecting a slight bias toward the synthetic class.

---

## 🚀 Future Work

- Incorporate data augmentation to improve generalization on real images
- Experiment with pretrained backbones (EfficientNet, ViT) for the spatial branch
- Extend to multi-class classification (identifying specific generative models)
- Optimize for real-time deployment on edge devices

---

## 👥 Authors

| Name | Institute | Email |
|---|---|---|
| Rajan Kushwaha | Dept. of Electronics Engineering, SVNIT Surat | u23ec051@eced.svnit.ac.in |
| Dhruvil Mali | Dept. of Electronics Engineering, SVNIT Surat | u23ec062@eced.svnit.ac.in |
| Dayshaun Kakadiya | Dept. of Electronics Engineering, SVNIT Surat | u23ec064@eced.svnit.ac.in |

---

## 📚 References

1. I. J. Goodfellow et al., "Generative Adversarial Networks," NeurIPS, 2014.
2. J. Hu et al., "Squeeze-and-Excitation Networks," CVPR, 2018.
3. K. He et al., "Deep Residual Learning for Image Recognition," CVPR, 2016.
4. T.-Y. Lin et al., "Microsoft COCO: Common Objects in Context," ECCV, 2014.
5. D. P. Kingma and J. Ba, "Adam: A Method for Stochastic Optimization," ICLR, 2015.

---

## 📃 License

This project is released for academic and research purposes. Please cite our work if you use this codebase.
