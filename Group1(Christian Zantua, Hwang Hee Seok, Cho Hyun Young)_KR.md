# Group 1 보고서: Selective Contrastive Learning for Weakly Supervised Affordance Grounding

**논문 제목:** *Selective Contrastive Learning for Weakly Supervised Affordance Grounding* (ICCV 2025)

## 1. Git Clone 상태

저장소는 `/root/workspace/andycho/CV/SelectiveCL` 경로에 clone되어 있다. 현재 branch는 `main`이며, 최신 로컬 commit hash는 `8a50d38`이다. remote는 원본 SelectiveCL 저장소(`https://github.com/hynnsk/SelectiveCL.git`)와 우리 작업용 프로젝트 remote(`git@github.com:airman-and/DSC3033_WSAG.git`)가 함께 설정되어 있다. 현재 working tree에는 논문 분석 노트, 추출한 figure, notebook, log, checkpoint, 실행 결과물 등 프로젝트 산출물도 포함되어 있다.

## 2. 코드 실행 상태

고정된 `selectivecl` conda 환경을 사용해 NVIDIA Quadro RTX 8000 GPU 서버에서 코드를 실행할 수 있었다. 이 환경은 Python 3.7.16, PyTorch 1.9.0, CUDA Toolkit 11.1.1, torchvision 0.10.0, OpenAI CLIP, `fast-pytorch-kmeans`, `pycocotools`, OpenCV, Matplotlib을 사용한다. 주요 실행 스크립트는 `run_selectivecl_all_gpu3.sh`이며, 서버 GPU index 3번인 NVIDIA Quadro RTX 8000을 logical GPU 0번으로 노출하고 AGD20K dataset은 `/root/workspace/andycho/CV/AGD20K` 경로를 사용한다.

AGD20K 전체 실행은 2026년 5월 8일에 완료되었다. `train.py --divide Seen`과 `train.py --divide Unseen`의 15-epoch 학습이 모두 끝났고, 공식 AGD20K checkpoint도 다운로드 후 test까지 수행했다. 공식 Seen checkpoint의 결과는 `KLD/SIM/NSS = 1.142/0.415/1.303`이었고, CLIP-refined ego-ego output은 `1.124/0.433/1.280`이었다. 공식 Unseen checkpoint의 결과는 `1.287/0.378/1.377`이었고, CLIP-refined ego-ego output은 `1.243/0.405/1.368`이었다. 또한 짧은 smoke-training demo도 2 training steps와 2 evaluation steps까지 정상 실행되었다.

주요 기술적 이슈는 환경과 데이터 준비였다. 원본 코드는 오래된 Python/CUDA/PyTorch stack에 의존하므로, 고정된 conda 환경을 사용하는 것이 중요했다. API shell에서는 긴 background job이 안정적으로 유지되지 않아 full run은 persistent terminal session을 통해 실행했다. HICO-IIF의 경우 checkpoint는 준비되어 있지만 `/DATA/HICO-IIF` dataset이 없고, 현재 `test.py` 경로가 주로 AGD20K Seen/Unseen에 맞춰져 있어 아직 평가하지 않았다.

## 3. 팀원별 역할 분담 계획

Christian Zantua는 저장소와 실행 환경 관리를 담당한다. dependency 확인, checkpoint 관리, 재현 가능한 training script 정리, GPU 사용 상태와 run log 관리가 주요 역할이다.

Hwang Hee Seok은 논문과 source code 이해를 담당한다. 논문 방법론을 `models/locate.py`, `loss/loss.py`, data pipeline과 연결해 분석하고, CLIP object discovery, DINO feature, selective contrastive loss 구조를 설명하는 역할을 맡는다.

Cho Hyun Young은 평가와 발표 산출물을 담당한다. 공식 checkpoint test 실행, KLD/SIM/NSS metric 정리, `test.py` 기반 qualitative visualization 생성, 최종 report와 demo notebook 관리를 맡는다.

## 4. 논문 이해

SelectiveCL은 weakly supervised affordance grounding 문제를 다룬다. 목표는 grasp, cut, pour와 같은 action에 대응하는 object region이나 part를 localization하는 것이다. 이때 training 단계에서는 dense ground-truth mask를 사용하지 않고, image-level affordance label과 egocentric/exocentric image pair를 활용해 학습한다.

핵심 아이디어는 selective supervision이다. 모델은 먼저 CLIP/ClearCLIP 방식의 patch-text similarity를 사용해 action과 관련된 object region을 찾는다. 이후 DINO ViT-S/16의 patch descriptor와 self-attention을 사용해 spatial part 정보를 추론한다. exocentric part cue가 신뢰 가능하면 selective prototypical contrastive learning을 통해 part-level affordance 정보를 학습하고, part cue가 불안정하면 object-level cue로 fallback하여 noisy pseudo-part에 과적합되는 것을 줄인다. 두 번째 loss인 selective pixel contrastive loss는 exocentric object affinity를 이용해 egocentric patch feature를 positive group과 negative group으로 나눈다. 코드 기준으로 training-time 핵심 로직은 `models/locate.py::Net.forward`에 집중되어 있고, inference는 더 단순하게 `Net.test_forward`에서 egocentric image만 사용한다.

중요한 점은 training objective가 dense GT heatmap을 사용하지 않는다는 것이다. GT mask는 epoch-end validation과 final test metric 계산에만 사용된다. Evaluation에서는 predicted affordance heatmap과 GT mask를 비교해 KLD, SIM, NSS를 계산한다.

## 5. 학습 및 평가 계획

**Hardware.** 기본 계획은 고정된 `selectivecl` conda 환경과 CUDA GPU를 사용할 수 있는 로컬 NVIDIA Quadro RTX 8000 GPU 서버를 사용하는 것이다. 이미 검증한 full-run setup은 NVIDIA Quadro RTX 8000 GPU(서버 GPU index 3)를 사용했다. Colab 또는 Colab Pro는 fallback option으로 사용할 수 있으며, 포함된 notebook들은 Google Drive에 AGD20K를 mount하는 Colab 실행 방식도 지원하도록 작성되어 있다.

**Models and datasets.** 공식 SelectiveCL checkpoint와 우리가 재현 학습한 checkpoint를 모두 AGD20K-Seen, AGD20K-Unseen에서 평가할 계획이다. HICO-IIF는 dataset 준비와 test path 수정이 끝난 뒤 optional future work로 진행할 수 있다.

**Metrics.** raw ego prediction과 CLIP-refined output(`refined_CLIP_ego_ego`, `refined_CLIP_ego_mean`)에 대해 KLD, SIM, NSS를 재현할 계획이다. KLD는 낮을수록 좋고, SIM과 NSS는 높을수록 좋다. 추가로 training loss와 classification accuracy log를 sanity check로 유지하고, 해석을 위해 qualitative heatmap/overlay 예시도 포함할 예정이다.
