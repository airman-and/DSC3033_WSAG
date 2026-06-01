# Group 0 Report: Selective Contrastive Learning for Weakly Supervised Affordance Grounding

**Paper title:** *Selective Contrastive Learning for Weakly Supervised Affordance Grounding* (ICCV 2025)

## 1. Git Clone Status

The repository is cloned and available at `/root/workspace/andycho/CV/SelectiveCL`. It is on branch `main`, with latest local commit hash `8a50d38`. The repository has two remotes configured: the original SelectiveCL repository (`https://github.com/hynnsk/SelectiveCL.git`) and our working project remote (`git@github.com:airman-and/DSC3033_WSAG.git`). The working tree currently also contains project artifacts such as paper analysis notes, extracted figures, notebooks, logs, checkpoints, and generated run outputs.

## 2. Code Execution Status

We were able to run the code on the GPU server using the pinned `selectivecl` conda environment. The environment uses Python 3.7.16, PyTorch 1.9.0, CUDA Toolkit 11.1.1, torchvision 0.10.0, OpenAI CLIP, `fast-pytorch-kmeans`, `pycocotools`, OpenCV, and Matplotlib. The main execution script is `run_selectivecl_all_gpu3.sh`, which exposes physical GPU 3 as logical GPU 0 and uses the AGD20K dataset at `/root/workspace/andycho/CV/AGD20K`.

The full AGD20K run completed on May 8, 2026. Both 15-epoch training runs finished: `train.py --divide Seen` and `train.py --divide Unseen`. Official AGD20K checkpoints were also downloaded and tested. The official Seen checkpoint produced `KLD/SIM/NSS = 1.142/0.415/1.303`; the CLIP-refined ego-ego output produced `1.124/0.433/1.280`. The official Unseen checkpoint produced `1.287/0.378/1.377`; the CLIP-refined ego-ego output produced `1.243/0.405/1.368`. A short smoke-training demo also ran successfully for two training steps and two evaluation steps.

The main technical issues were environment and data related. The original code depends on an older Python/CUDA/PyTorch stack, so the pinned conda environment is important. Long background jobs did not stay alive reliably through the API shell, so the full run was launched through a persistent terminal session. HICO-IIF was not evaluated yet because the checkpoint is available but the `/DATA/HICO-IIF` dataset is absent, and the current `test.py` path is mainly set up for AGD20K Seen/Unseen.

## 3. Task Division Plan

RX will handle repository and environment maintenance, including dependency checks, checkpoint management, and reproducible training scripts. RX will also keep track of GPU usage and run logs.

BSY will focus on paper and source-code understanding. This includes mapping the paper's method to `models/locate.py`, `loss/loss.py`, and the data pipeline, and preparing explanations of CLIP object discovery, DINO features, and selective contrastive losses.

SHJ will focus on evaluation and presentation artifacts. This includes running official checkpoint tests, collecting KLD/SIM/NSS metrics, preparing qualitative visualizations from `test.py`, and maintaining the final report/demo notebooks.

## 4. Paper Understanding

SelectiveCL addresses weakly supervised affordance grounding. The goal is to localize image regions that support a given action, such as the part of an object that can be grasped, cut, or poured, without using dense ground-truth masks during training. Instead of learning only from pixel annotations, the method learns from image-level affordance labels and paired egocentric/exocentric images.

The core idea is selective supervision. The model first uses CLIP/ClearCLIP-style patch-text similarity to discover action-associated object regions. It then uses DINO ViT-S/16 patch descriptors and self-attention to reason about spatial parts. If an exocentric part cue is reliable, the model learns part-level affordance information through selective prototypical contrastive learning. If the part cue is unreliable, it falls back to object-level cues so the model does not overfit to noisy pseudo-parts. A second selective pixel contrastive loss uses exocentric object affinity to split egocentric patch features into positive and negative groups. In code, the main train-time logic is concentrated in `models/locate.py::Net.forward`, while the inference path is simpler and uses only the egocentric image through `Net.test_forward`.

Importantly, the training objective does not use dense GT heatmaps. GT masks are used for epoch-end validation and final testing only. Evaluation compares predicted affordance heatmaps against GT masks with KLD, SIM, and NSS.

## 5. Training and Evaluation Plans

**Hardware.** The primary plan is to use the local GPU server with the pinned `selectivecl` conda environment and CUDA GPU access. The tested full-run setup used physical GPU 3. Colab or Colab Pro is a fallback option; the included notebooks are written to support Colab with AGD20K mounted from Google Drive.

**Models and datasets.** We will evaluate both official SelectiveCL checkpoints and our reproduced training checkpoints on AGD20K-Seen and AGD20K-Unseen. HICO-IIF is optional future work if the dataset is prepared and the test path is adapted.

**Metrics.** We will reproduce KLD, SIM, and NSS for raw ego predictions and CLIP-refined outputs (`refined_CLIP_ego_ego` and `refined_CLIP_ego_mean`). KLD is lower-is-better, while SIM and NSS are higher-is-better. We will also keep training loss/classification accuracy logs as sanity checks and include qualitative heatmap/overlay examples for interpretation.
