# DSC3033_WSAG

## Selective Contrastive Learning for Weakly Supervised Affordance Grounding (ICCV 2025)
WonJun Moon*</sup>, Hyun Seok Seong*</sup>, Jae-Pil Heo</sup> (*: equal contribution)

[[Arxiv](https://arxiv.org/abs/2508.07877)]

## Abstract
> Facilitating an entity’s interaction with objects requires accurately identifying parts that afford specific actions. Weakly
supervised affordance grounding (WSAG) seeks to imitate
human learning from third-person demonstrations, where
humans intuitively grasp functional parts without needing
pixel-level annotations. To achieve this, grounding is typically learned using a shared classifier across images from
different perspectives, along with distillation strategies incorporating part discovery process. However, since affordancerelevant parts are not always easily distinguishable, models
primarily rely on classification, often focusing on common
class-specific patterns that are unrelated to affordance. To
address this limitation, we move beyond isolated part-level
learning by introducing selective prototypical and pixel contrastive objectives that adaptively learn affordance-relevant
cues at both the part and object levels, depending on the
granularity of the available information. Initially, we find the
action-associated objects in both egocentric (object-focused)
and exocentric (third-person example) images by leveraging
CLIP. Then, by cross-referencing the discovered objects of
complementary views, we excavate the precise part-level affordance clues in each perspective. By consistently learning
to distinguish affordance-relevant regions from affordanceirrelevant background context, our approach effectively shifts
activation from irrelevant areas toward meaningful affordance cues. Experimental results demonstrate the effectiveness of our method.
----------

## Requirements
The current workspace uses the `selectivecl` conda environment. Recreate it
from `environment.yml` instead of committing or copying a local environment
directory.

```bash
conda env create -f environment.yml
conda activate selectivecl
```

If you already have a `selectivecl` environment, update it with:

```bash
conda env update -n selectivecl -f environment.yml --prune
conda activate selectivecl
```

`environment.yml` is exported from the working conda environment and pins the
conda and pip dependencies used here, including Python 3.7.16, CUDA Toolkit
11.1.1, PyTorch 1.9.0, torchvision 0.10.0, OpenAI CLIP at a fixed Git commit,
`fast-pytorch-kmeans`, `pycocotools`, OpenCV, matplotlib, and notebook support.

## Dataset
We follow the dataset setup from the original [LOCATE](https://github.com/Reagan1311/LOCATE) repository.

You should modify the 'data_root' according to your dataset path.

## Training

- AGD20K-Seen
> python train.py --divide Seen
> 
- AGD20K-Unseen
> python train.py --divide Unseen


## Test
- AGD20K-Seen
> python test.py --model_file [checkpoint.pth] --divide Seen
- AGD20K-Unseen
> python test.py --model_file [checkpoint.pth] --divide Unseen




## Checkpoints
Dataset | Model file
 -- | --
AGD20K-Seen | [checkpoint](https://drive.google.com/file/d/1cYC2PBEjhLntySyP51R46J7i8f1Cf1NT/view?usp=sharing)
AGD20K-Unseen | [checkpoint](https://drive.google.com/file/d/1YojVtXtl4gCiqDRDOpHn59vdIPSIIgdt/view?usp=sharing)
HICO-IIF | [checkpoint](https://drive.google.com/file/d/1fOIarlqETEpY7JrqUWjgzvHtwCzRfeGb/view?usp=sharing)


## Group1 Notebooks and Colab Requirements

This repository includes two project submission notebooks:

- `Group1-full-training.ipynb`: full-training reproduction summary and optional 15-epoch rerun command.
- `Group1-demo.ipynb`: live demo notebook with a short smoke-training example, checkpoint inference, and five visualized predictions.

Both notebooks are written in dual-mode style. They detect whether they are
running on this GPU server or on Google Colab.

### Required Colab Setup

1. Enable a GPU runtime in Colab.

   Use `Runtime > Change runtime type > GPU`. CPU execution is not recommended
   because SelectiveCL uses DINO, CLIP, and dense test-set inference.

2. Prepare AGD20K in Google Drive.

   The notebooks expect this default path in Colab:

   ```python
   DATA_ROOT = Path('/content/drive/MyDrive/AGD20K')
   ```

   The expected directory structure is:

   ```text
   AGD20K/
     Seen/trainset/exocentric
     Seen/trainset/egocentric
     Seen/testset/egocentric
     Seen/testset/GT
     Unseen/trainset/exocentric
     Unseen/trainset/egocentric
     Unseen/testset/egocentric
     Unseen/testset/GT
   ```

3. Mount Google Drive when prompted.

   In Colab, the notebooks call:

   ```python
   drive.mount('/content/drive')
   ```

4. Ensure checkpoint access.

   The notebooks can download the official checkpoints with `gdown`. If Google
   Drive download quota or access blocks the automatic download, place the files
   manually under:

   ```text
   /content/SelectiveCL/checkpoints/agd20k_seen.pth
   /content/SelectiveCL/checkpoints/agd20k_unseen.pth
   ```

   `Group1-demo.ipynb` only needs `agd20k_seen.pth`.

### Recommended Colab Execution Order

1. Open `Group1-demo.ipynb` first.
2. Select a GPU runtime.
3. Run the setup and checkpoint cells.
4. Keep `RUN_MINIMAL_TRAINING = True` to run the two-step smoke training
   example during the live demo. Set it to `False` only to skip execution and
   review the saved smoke log.
5. Run limited visual inference and inspect the five output panels.
6. Open `Group1-full-training.ipynb` to review the full-training setup,
   training-log scores, and KLD/SIM/NSS reproduction tables.

The full-training notebook is configured with `RUN_FULL_TRAINING = False` by
default so evaluators can review the saved reference logs and metric tables
without starting a multi-hour job. Set it to `True` only when you explicitly
want to rerun the 15-epoch AGD20K training job.

### Version Notes

The original working environment is pinned in `environment.yml` and uses Python
3.7.16, CUDA Toolkit 11.1.1, PyTorch 1.9.0, and torchvision 0.10.0. Colab often
uses newer Python and PyTorch versions, so dependency installation may require
minor adjustment if Colab changes its base image.



## Licence
Our codes are released under [MIT](https://opensource.org/licenses/MIT) license.
