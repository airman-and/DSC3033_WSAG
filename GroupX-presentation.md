# SelectiveCL Presentation Draft

## Slide 1: Title
Selective Contrastive Learning for Weakly Supervised Affordance Grounding

- ICCV 2025
- WonJun Moon, Hyun Seok Seong, Jae-Pil Heo
- DSC3032 Deep Learning 1 Final Project
- Group X

Speaker note:
This presentation explains the paper, reproduces the AGD20K-Seen setting, and demonstrates inference with trained SelectiveCL checkpoints.

## Slide 2: Problem
Weakly supervised affordance grounding asks a model to localize object regions that support an action.

- Input: an egocentric object-focused image and an action label.
- Weak supervision: training uses image-level action labels and third-person exocentric examples.
- Output: an affordance heatmap over the target object.

Speaker note:
The task is not object detection. The model must identify the part that is usable for an action, such as the handle of a cup for holding or the seat of a bicycle for riding.

## Slide 3: Motivation
Previous WSAG methods often rely heavily on action classification.

- Classification can focus on class-specific shortcuts.
- Distinguishable regions are not always affordable regions.
- Part discovery is unreliable when target parts are small, occluded, or visually ambiguous.

Speaker note:
SelectiveCL addresses this by providing contrastive cues even when reliable part-level cues are missing.

## Slide 4: Main Idea
SelectiveCL learns from both object-level and part-level cues.

- CLIP provides object affinity maps for action-associated objects.
- DINO features support part discovery and CAM prediction.
- Reliable parts guide part-level learning.
- Unreliable parts fall back to object-level learning against background.

Speaker note:
The key design is selectivity. The method does not force part supervision when the part candidate is unreliable.

## Slide 5: Model Pipeline
Pipeline overview:

- DINO ViT-S/16 extracts dense image features.
- CLIP ViT-B/16 with ClearCLIP-style dense inference creates object affinity maps.
- A shared classifier produces CAMs for egocentric and exocentric views.
- Projection heads feed prototypical and pixel contrastive losses.
- Inference multiplies CAM output with the CLIP object affinity map for calibration.

Speaker note:
The implementation uses `models/locate.py` for the network and `loss/loss.py` for the contrastive objectives.

## Slide 6: Selective Prototypical Contrastive Learning
The prototypical loss compares positive and negative prototypes.

- If a reliable part prototype exists, the model pulls egocentric features toward affordance-relevant part prototypes.
- If no reliable part exists, the model pulls features toward object prototypes.
- Background and other action prototypes act as negatives.

Speaker note:
This reduces attention to unaffordable object regions and background context.

## Slide 7: Pixel Contrastive Learning
The pixel contrastive loss improves fine-grained localization.

- Positive pixels are selected from egocentric object affinity maps.
- Negative pixels are the remaining object/background regions.
- Pixel features are contrasted inside each egocentric image.

Speaker note:
This loss directly encourages pixels with the same affordance relevance to become similar.

## Slide 8: Dataset and Metrics
Main reproduction target: AGD20K-Seen.

- Train exocentric images: 20,061
- Train egocentric images: 6,929
- Test egocentric images: 1,710
- Affordance classes: 36
- Object classes: 50

Metrics:

- KLD: lower is better.
- SIM: higher is better.
- NSS: higher is better.

## Slide 9: Hyperparameter Comparison
Paper setting and our reproduction setting are aligned.

| Item | Paper | Reproduction |
|---|---:|---:|
| Backbone | DINO ViT-S/16 + CLIP ViT-B/16 | DINO ViT-S/16 + CLIP ViT-B/16 |
| Dataset | AGD20K-Seen | AGD20K-Seen |
| Epochs | 15 | 15 |
| Batch size | 8 | 8 |
| Learning rate | 1e-3 | 1e-3 |
| Weight decay | 5e-4 | 5e-4 |
| Exocentric images E | 3 | 3 |
| alpha, gamma | 0.6 | 0.6 |
| Temperature tau | 0.5 | 0.5 |

## Slide 10: Metric Comparison
AGD20K-Seen results.

| Source | Prediction | KLD lower | SIM higher | NSS higher |
|---|---|---:|---:|---:|
| Paper Table 1 | Ours | 1.124 | 0.433 | 1.280 |
| Official checkpoint re-evaluation | refined ego-ego | 1.124 | 0.433 | 1.280 |
| Local full training best | refined ego-ego, epoch 4 | 1.136 | 0.427 | 1.278 |

Speaker note:
The official checkpoint reproduces the paper metric exactly for AGD20K-Seen. The local full training run is close to the paper result.

## Slide 11: Demo Plan
The live demo uses `GroupX-demo.ipynb`.

- Verify environment, data paths, and checkpoint paths.
- Run limited visual inference on AGD20K-Seen.
- Display original image, ground-truth mask, predicted heatmap, and overlay.
- Show the minimal training cell and explain why full training is not run live.

Speaker note:
The live part is kept under three minutes by using a pretrained checkpoint and saving only five visual examples.

## Slide 12: Code Walkthrough
Important code files:

- `data/datatrain.py`: pairs exocentric and egocentric images by action and object.
- `data/datatest.py`: loads test images and dense GT masks.
- `models/locate.py`: implements SelectiveCL forward pass and inference.
- `loss/loss.py`: implements prototypical and pixel contrastive losses.
- `test.py`: computes KLD, SIM, NSS and saves visualizations.

## Slide 13: Reproduction Notes
Implementation details:

- The server environment uses Python 3.7.16, Torch 1.9.0, and CUDA.
- Full training took about 5 hours 9 minutes for Seen and 3 hours 15 minutes for Unseen on the selected GPU.
- HICO-IIF is excluded because the local dataset is not available and the current test pipeline is AGD20K-oriented.

## Slide 14: Conclusion
SelectiveCL improves weakly supervised affordance grounding by combining:

- CLIP-based object discovery.
- DINO-based dense visual features.
- Selective prototypical contrastive learning.
- Pixel-level contrastive learning.
- CAM calibration using object affinity maps.

Speaker note:
The reproduction confirms the main AGD20K-Seen result and provides a short visual demo for qualitative understanding.

## Slide 15: References
- Moon, Seong, and Heo. Selective Contrastive Learning for Weakly Supervised Affordance Grounding. ICCV 2025.
- Official paper: https://openaccess.thecvf.com/content/ICCV2025/html/Moon_Selective_Contrastive_Learning_for_Weakly_Supervised_Affordance_Grounding_ICCV_2025_paper.html
- Supplemental material: https://openaccess.thecvf.com/content/ICCV2025/supplemental/Moon_Selective_Contrastive_Learning_ICCV_2025_supplemental.pdf
- Official code: https://github.com/hynnsk/SelectiveCL
