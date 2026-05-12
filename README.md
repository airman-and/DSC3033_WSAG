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


## Licence
Our codes are released under [MIT](https://opensource.org/licenses/MIT) license.
