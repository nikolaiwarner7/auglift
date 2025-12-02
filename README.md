# **AugLift: Depth-Augmented 3D Human Pose Lifting**

AugLift is a research fork built on **MMPose (OpenMMLab)** and **DepthAnything V2**, designed to improve 3D lifting by adding **ordinal / metric depth cues**, **confidence channels**, and **depth-augmented PoseFormer / TCN backbones**.

This repo includes:

* custom dataloaders
* custom codecs
* modified PoseFormer + regression heads
* depth-estimation pipelines
* XYCD merge utilities
* cross-dataset training scripts
* evaluation utilities

---

## **1. Environment Setup**

AugLift requires **Python 3.10**, **PyTorch**, **MMPose**, **DepthAnything V2**, and our custom modules.

### **STEP 1: Create the Base MMPose Environment**

First, create a clean baseline MMPose environment:

```bash
conda create -n auglift python=3.10 -y
conda activate auglift
```

Install PyTorch (select the CUDA version used on your cluster):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Install MMPose dependencies:

```bash
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
mim install "mmpose>=1.2.0"
```

---

### **STEP 2: Update Environment with AugLift Dependencies**

After creating the base environment, update it with our custom dependencies:

```bash
conda env update -f openmmlab_env.yml --prune
```

Or, if updating an existing environment:

```bash
conda activate auglift
conda env update -f openmmlab_env.yml --prune
```

---

## **2. External Libraries**

### **MMPose (Apache-2.0 License)**

[https://github.com/open-mmlab/mmpose](https://github.com/open-mmlab/mmpose)

Used for:

* PoseFormer, TCN, and regression heads
* dataset pipelines
* training infrastructure
* codec templates

### **DepthAnything V2**

[https://github.com/DepthAnything/Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2)

Used for:

* monocular metric depth estimation
* depth-augmented XYCD features
* depth caches used during pose lifting

Clone it next to the repo:

```bash
git clone https://github.com/DepthAnything/Depth-Anything-V2.git
cd Depth-Anything-V2
pip install -r requirements.txt
```

---

## **3. Install This Repository**

```bash
git clone https://github.com/nikolaiwarner7/auglift.git
cd auglift
pip install -e .
```

This installs:

* custom datasets
* custom codecs
* custom backbones + heads
* merge utilities
* training scripts

so they can be imported from anywhere on the system.

---

## **4. Dataset Setup**

Place datasets under the repo root with these expected names:

```
3dpw_data/
h36m_data/
fit3d_data/
data/                # for 3DHP
```

### **Dataset notes**

| Dataset   | Expected Structure                                            |
| --------- | ------------------------------------------------------------- |
| **3DPW**  | dataset.json, videos inside `3dpw_data/`                      |
| **H36M**  | MMPose-standard directory structure                           |
| **3DHP**  | `.mat` annotations + images inside `data/`                    |
| **Fit3D** | custom loaders expect the folder layout used in merge scripts |

Your dataloader modules include:

* `mmpose/mmpose/datasets/datasets/base/base_mocap_dataset.py`
* `mmpose/mmpose/datasets/datasets/body3d/mpi_3dhp_inf_dataset.py`

---

## **5. Preprocessing AugLift Data**

See [docs/PREPROCESSING.md](docs/PREPROCESSING.md) for detailed instructions on:

* Running MMPose preprocessing on raw images and annotations
* Depth estimation and 2D pose detection pipelines
* XYCD feature merging
* Feature map caching (AugLift V2)
* SLURM distribution scripts

---

## **6. Training AugLift Models**

See [docs/TRAINING.md](docs/TRAINING.md) for detailed instructions on:

* Training configuration
* Cross-dataset training
* Model architectures and variants
* Evaluation protocols

---

## **7. Reproducibility Checklist**

- [ ] Clone AugLift
- [ ] Install dependencies (PyTorch + MMPose + DepthAnything V2)
- [ ] Run `pip install -e .`
- [ ] Download datasets (3DPW, H36M, 3DHP, Fit3D)
- [ ] Run depth estimation scripts
- [ ] Run XYCD merge scripts
- [ ] Train with provided configs
- [ ] Evaluate on matched test splits

---

## **8. Repository Structure**

```
auglift/
├── mmpose/                          # MMPose fork with custom modules
│   ├── mmpose/
│   │   ├── codecs/                  # Custom codecs
│   │   ├── models/
│   │   │   ├── backbones/           # PoseFormer, TCN
│   │   │   └── heads/               # Regression heads
│   │   └── datasets/                # Custom dataloaders
│   ├── configs/                     # Training configs
│   └── tools/                       # Training scripts
├── coarse_depth_experiments/        # Depth estimation pipelines
│   └── Depth-Anything-OfficialV2/
│       └── Depth-Anything-V2/
│           └── metric_depth/        # Depth scripts
├── 2_3_merge_xycd_npz_*.py         # XYCD merge utilities
├── docs/                            # Documentation
│   ├── PREPROCESSING.md             # Data preprocessing guide
│   ├── TRAINING.md                  # Training guide
│   └── CHANGES.md                   # Changelist vs MMPose
├── setup.py                         # Package setup
├── pyproject.toml                   # Build config
└── README.md                        # This file
```

---

## **9. Changes vs MMPose Base**

AugLift extends MMPose with several custom components. See [docs/CHANGES.md](docs/CHANGES.md) for the full changelist.

**Key modifications:**

### **Preprocessing:**
* Custom dataset classes for 3DPW, 3DHP, and Fit3D (based on H36M class with custom naming)
* Modified 3D evaluator to align coordinate systems across all 4 datasets
* Custom dataloaders: 
  * `mmpose/mmpose/datasets/datasets/base/base_mocap_dataset.py`
  * `mmpose/mmpose/datasets/datasets/body3d/mpi_3dhp_inf_dataset.py`

### **Training:**
* Depth-augmented PoseFormer and TCN backbones
* Custom regression heads for XYCD input
* Modified codecs:
  * `mmpose/mmpose/codecs/image_pose_lifting.py`
  * `mmpose/mmpose/codecs/poseformer_label.py`
* Cross-dataset training configurations
* Root-relative keypoint handling in codecs

---

## **License**

This project builds upon:

* **MMPose** (Apache-2.0): [https://github.com/open-mmlab/mmpose](https://github.com/open-mmlab/mmpose)
* **DepthAnything V2**: [https://github.com/DepthAnything/Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2)

Please cite the original works when using this repository.

---

## **Citation**

If you use AugLift in your research, please cite:

```bibtex
@misc{auglift2024,
  title={AugLift: Depth-Augmented 3D Human Pose Lifting},
  author={AugLift Team},
  year={2024},
  howpublished={\url{https://github.com/nikolaiwarner7/auglift}}
}
```

---

## **Contact**

For questions or issues, please open an issue on the GitHub repository.
