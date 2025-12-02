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
3dpw_data/          # 11 GB
h36m_data/          # 1004 GB (1 TB)
fit3d_data/         # 138 GB
data/               # 977 GB (for 3DHP)
```

### **Dataset Structure & Sizes**

| Dataset   | Size   | Expected Structure                                            |
| --------- | ------ | ------------------------------------------------------------- |
| **3DPW**  | 11 GB  | `sequenceFiles/` (train/test), `processed_mmpose_shards/`     |
| **H36M**  | 1 TB   | `images/images/` (extracted from tar.gz archives)             |
| **3DHP**  | 977 GB | `test/` folders (TS1-TS6), each with `annot_data.mat` + `imageSequence/` |
| **Fit3D** | 138 GB | Custom folder layout used in merge scripts                    |

### **Detailed Directory Structure**

#### **3DPW (`3dpw_data/`):**
```
3dpw_data/
├── sequenceFiles/
│   ├── train/
│   └── test/
└── processed_mmpose_shards/
```

#### **H36M (`h36m_data/`):**
```
h36m_data/
└── images/
    ├── images.tar.gzaa          # Split archives
    ├── images.tar.gzab
    ├── ...
    └── images/                  # Extracted images
        ├── s_01_act_12_subact_02_ca_01/
        │   └── s_01_act_12_subact_02_ca_01_*.jpg
        ├── s_08_act_11_subact_02_ca_01/
        │   └── s_08_act_11_subact_02_ca_01_*.jpg
        └── ...
```

**Note:** Extract all `images.tar.gz*` archives before running preprocessing.

#### **3DHP (`data/`):**
```
data/
└── test/
    ├── TS1/
    │   ├── annot_data.mat
    │   └── imageSequence/
    │       └── img_*.jpg
    ├── TS2/
    ├── TS3/
    ├── TS4/
    ├── TS5/
    └── TS6/
```

#### **Fit3D (`fit3d_data/`):**
```
fit3d_data/
└── (custom folder layout - see merge scripts)
```

### **Dataloader Modules**

Your custom dataloader modules include:

* `mmpose/mmpose/datasets/datasets/base/base_mocap_dataset.py`
* `mmpose/mmpose/datasets/datasets/body3d/mpi_3dhp_inf_dataset.py`
* `mmpose/mmpose/datasets/datasets/body3d/pw3d_dataset.py`
* `mmpose/mmpose/datasets/datasets/body3d/fit3d_dataset.py`

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

## **7. Post-Training Analysis**

See [docs/ANALYSIS.md](docs/ANALYSIS.md) for detailed instructions on:

* Quantitative metrics comparison (MPJPE, PCK3D, ordinal depth)
* Occlusion analysis (frame-level, joint-level, depth-based)
* Domain similarity analysis (OOD generalization)
* 2D detection quality robustness
* Angle/orientation error analysis
* Qualitative visualizations for paper figures

---

## **8. Reproducibility Checklist**

- [ ] Clone AugLift
- [ ] Install dependencies (PyTorch + MMPose + DepthAnything V2)
- [ ] Run `pip install -e .`
- [ ] Download datasets (3DPW, H36M, 3DHP, Fit3D)
- [ ] Run depth estimation scripts
- [ ] Run XYCD merge scripts
- [ ] Train with provided configs
- [ ] Evaluate on matched test splits
- [ ] Run post-training analysis (metrics, occlusion, visualizations)

---

## **9. Repository Structure**

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
├── poseformer_metrics_simple.py    # Post-training analysis script
├── docs/                            # Documentation
│   ├── PREPROCESSING.md             # Data preprocessing guide
│   ├── TRAINING.md                  # Training guide
│   ├── ANALYSIS.md                  # Post-training analysis guide
│   └── CHANGES.md                   # Changelist vs MMPose
├── setup.py                         # Package setup
├── pyproject.toml                   # Build config
└── README.md                        # This file
```

---

## **10. Changes vs MMPose Base**

AugLift extends MMPose with several custom components. See [docs/CHANGES.md](docs/CHANGES.md) for the full changelist.

**Key modifications:**

### **Preprocessing:**
* Custom dataset classes for 3DPW, 3DHP, and Fit3D (based on H36M class with custom naming)
* Modified 3D evaluator to align coordinate systems across all 4 datasets
* Custom dataloaders: 
  * `mmpose/mmpose/datasets/datasets/base/base_mocap_dataset.py` - **Key changes to load C, D, feature maps**
  * `mmpose/mmpose/datasets/datasets/body3d/mpi_3dhp_inf_dataset.py`
  * `mmpose/mmpose/datasets/datasets/body3d/pw3d_dataset.py`
  * `mmpose/mmpose/datasets/datasets/body3d/fit3d_dataset.py`

### **Training:**
* **4 architectures:** PoseFormer (many-to-one transformer), MotionBERT (many-to-many transformer), TCN (temporal convolution), SimpleBaseline (single-frame)
* Input layer sizes modified for richer input formulations (specified in pyconfigs)
* Custom regression heads for XYCD input (all architectures)
* Modified codecs:
  * `mmpose/mmpose/codecs/image_pose_lifting.py` - Input normalization for C, D
  * `mmpose/mmpose/codecs/poseformer_label.py` - Temporal XYCD sequences
  * `mmpose/mmpose/codecs/motionbert_label.py` - Camera-invariant codec
* Cross-dataset training configurations (Section 4.1)
* Root-relative keypoint handling in codecs
* Representation study (XY, XYC, XYD, XYCD, AugLift V2, Image-Features) - Section 5

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
