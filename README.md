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

### **1.1. Create a Conda environment**

```bash
conda create -n auglift python=3.10 -y
conda activate auglift
```

---

### **1.2. Install PyTorch**

Select the CUDA version used on your cluster:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

### **1.3. Install MMEngine, MMCV, and MMPose**

```bash
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
mim install "mmpose>=1.2.0"
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

## **5. Depth Estimation Pipelines**

Depth scripts are located in:

```
coarse_depth_experiments/Depth-Anything-OfficialV2/Depth-Anything-V2/metric_depth/
```

Example command:

```bash
python metric_depth/1_30_estimate_depth_proto_loop_metric_h36m_v3_sampling_patch_statistics.py
```

Each dataset has its own:

* prototype depth estimator
* loop scripts (`launch_depth_metric_loop_*.sh`)
* outer-loop sampling scripts

These scripts output cached **metric depth NPZ files** used during XYCD merging.

---

## **6. XYCD Merge Utilities**

Scripts like:

```
2_3_merge_xycd_npz_h36m.py
2_3_merge_xycd_npz_fit3d.py
2_3_merge_xycd_npz_3dpw.py
2_3_merge_xycd_npz_3dhp.py
```

combine:

* XY keypoints
* C confidence
* D depth
* dataset metadata

Usage example:

```bash
python 2_3_merge_xycd_npz_h36m.py
```

Fit3D split helper:

```bash
python 2_25_split_fit3d_train_test.py
```

---

## **7. Custom Codecs, Backbones, & Heads**

### **Codecs**

* `mmpose/mmpose/codecs/image_pose_lifting.py`
* `mmpose/mmpose/codecs/poseformer_label.py` (modified)

### **Model Definitions**

* `mmpose/mmpose/models/backbones/poseformer.py`
* `mmpose/mmpose/models/heads/regression_heads/poseformer_regression_head.py`

### **Configs**

Including:

```
mmpose/configs/body_3d_keypoint/image_pose_lift/h36m/image-pose-lift_tcn_8xb64-200e_h36m_oct25_casp.py
mmpose/configs/body_3d_keypoint/poseformer/h36m/poseformer_h36m_config_img_depth_baselines.py
```

Training scripts:

```
mmpose/tools/launch_train_cross_datasets_may_25_poseformer_test_outer.sh
mmpose/tools/launch_train_cross_datasets_may_25_pf.sh
```

---

## **8. Training AugLift**

Using Bash launcher scripts:

```bash
bash mmpose/tools/launch_train_cross_datasets_may_25_poseformer_test_outer.sh
```

Or directly with MMEngine:

```bash
python tools/train.py path/to/config.py
```

---

## **9. Reproducibility Checklist**

- [ ] Clone AugLift
- [ ] Install dependencies (PyTorch + MMPose + DepthAnything V2)
- [ ] Run `pip install -e .`
- [ ] Download datasets (3DPW, H36M, 3DHP, Fit3D)
- [ ] Run depth estimation scripts
- [ ] Run XYCD merge scripts
- [ ] Train with provided configs
- [ ] Evaluate on matched test splits

---

## **10. Repository Structure**

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
├── setup.py                         # Package setup
├── pyproject.toml                   # Build config
└── README.md                        # This file
```

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
