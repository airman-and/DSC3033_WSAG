# Group 1 Report: Selective Contrastive Learning for Weakly Supervised Affordance Grounding

**Paper title:** *Selective Contrastive Learning for Weakly Supervised Affordance Grounding* (ICCV 2025)

## 1. Git Clone Status

The repository is cloned at `/root/workspace/andycho/CV/SelectiveCL`. It is on branch `main` with latest local commit hash `8a50d38`. It has both the original SelectiveCL remote (`https://github.com/hynnsk/SelectiveCL.git`) and our project remote (`git@github.com:airman-and/DSC3033_WSAG.git`). The working directory also contains our analysis notes, extracted figures, notebooks, checkpoints, logs, and generated run outputs.

## 2. Code Execution Status

We were able to run the code on an NVIDIA Quadro RTX 8000 GPU server using the pinned `selectivecl` conda environment: Python 3.7.16, PyTorch 1.9.0, CUDA Toolkit 11.1.1, torchvision 0.10.0, OpenAI CLIP, `fast-pytorch-kmeans`, `pycocotools`, OpenCV, and Matplotlib. The script `run_selectivecl_all_gpu3.sh` targets server GPU index 3, an NVIDIA Quadro RTX 8000, and exposes it as logical GPU 0. AGD20K is located at `/root/workspace/andycho/CV/AGD20K`.

The full AGD20K run completed on May 8, 2026. Both 15-epoch training jobs finished for `Seen` and `Unseen`, and official checkpoints were tested:

| Split / output | KLD | SIM | NSS |
| --- | ---: | ---: | ---: |
| Seen official | 1.142 | 0.415 | 1.303 |
| Seen refined ego-ego | 1.124 | 0.433 | 1.280 |
| Unseen official | 1.287 | 0.378 | 1.377 |
| Unseen refined ego-ego | 1.243 | 0.405 | 1.368 |

A short smoke-training demo also ran for two training steps and two evaluation steps. Main issues were environment compatibility and dataset availability: the code requires an older Python/CUDA/PyTorch stack, long API-shell background jobs needed a persistent terminal session, and HICO-IIF has not been evaluated because `/DATA/HICO-IIF` is absent.

## 3. Task Division Plan

- **Christian Zantua:** repository/environment setup, dependency checks, checkpoint management, reproducible scripts, and GPU/run-log tracking.
- **Hwang Hee Seok:** paper and source-code understanding, especially mapping the method to `models/locate.py`, `loss/loss.py`, and the data pipeline.
- **Cho Hyun Young:** evaluation, KLD/SIM/NSS collection, qualitative visualizations from `test.py`, and final report/demo notebook preparation.

## 4. Paper Understanding

SelectiveCL studies weakly supervised affordance grounding: localizing object parts that support an action, such as grasping, cutting, or pouring, without dense masks during training. It learns from image-level affordance labels and paired egocentric/exocentric images.

The method first uses CLIP/ClearCLIP-style patch-text similarity to find action-associated object regions. It then uses DINO ViT-S/16 patch descriptors and self-attention to reason about parts. If an exocentric part cue is reliable, selective prototypical contrastive learning uses it as part-level supervision; otherwise, the model falls back to object-level cues to avoid noisy pseudo-parts. Selective pixel contrastive learning further separates egocentric patch features into positive and negative groups using exocentric object affinity.

In code, the main training logic is in `models/locate.py::Net.forward`; inference is simpler and uses only the egocentric image through `Net.test_forward`. Dense GT heatmaps are not used in the training objective. GT masks are used only for epoch-end validation and final testing with KLD, SIM, and NSS.

## 5. Training and Evaluation Plans

**Hardware:** local NVIDIA Quadro RTX 8000 GPU server with the pinned `selectivecl` environment. Colab/Colab Pro is a fallback; the notebooks support AGD20K mounted from Google Drive.

**Models and datasets:** evaluate official SelectiveCL checkpoints and our reproduced checkpoints on AGD20K-Seen and AGD20K-Unseen. HICO-IIF is future work if the dataset and test path are prepared.

**Metrics:** reproduce KLD, SIM, and NSS for raw ego predictions and CLIP-refined outputs (`refined_CLIP_ego_ego`, `refined_CLIP_ego_mean`). KLD is lower-is-better; SIM and NSS are higher-is-better. Training loss/classification accuracy logs and qualitative heatmap overlays will be used as sanity checks.
