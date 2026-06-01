# SelectiveCL Source Code Analysis Report

작성일: 2026-05-14 KST

## 1. 분석 대상과 결론

이 저장소는 `Selective Contrastive Learning for Weakly Supervised Affordance Grounding` 구현체로, 입력 이미지에서 특정 affordance action에 대응하는 물체 부위를 약지도 방식으로 localization하는 모델이다. 전체 구조는 `train.py`와 `test.py`가 실행 엔트리포인트이고, 핵심 모델은 `models/locate.py`의 `Net`이다.

핵심 아이디어는 다음과 같다.

- CLIP ViT-B/16의 patch-token image feature와 text feature를 사용해 action/object 관련 coarse affinity map을 만든다.
- DINO ViT-S/16의 key descriptor와 self-attention을 사용해 egocentric/exocentric view 사이에서 part-level cue를 찾는다.
- 학습 가능한 projection/classifier head가 DINO descriptor 위에서 affordance class map을 예측한다.
- 손실은 egocentric/exocentric classification CE, pixel-level contrastive loss, prototype-level contrastive loss를 합산한다.
- inference에서는 egocentric 이미지만 사용하고, classifier map과 CLIP affinity map을 곱한 refined map을 평가한다.

## 2. 최상위 실행 흐름

### Training

`train.py`는 argparse로 dataset split, image size, optimizer, contrastive threshold를 받고, `Seen`, `Unseen`, `HICO`에 따라 class 수와 데이터 경로를 구성한다. 관련 위치는 `train.py:22-55`, `train.py:67-93`이다.

학습 루프는 다음 흐름이다.

1. `TrainData`에서 batch를 로드한다. 반환값은 `exocentric_images`, `egocentric_image`, `aff_label`, `aff_name`이다.
2. `models.locate.Net.forward(exo, ego, aff_label, epoch)`를 호출한다.
3. 모델이 반환하는 4개 loss를 단순 합산한다.
4. SGD optimizer로 update한다.
5. 매 epoch 끝에서 `TestData`로 전체 또는 제한된 validation/test set을 평가한다.
6. `refined_CLIP_ego_ego` 기준 KLD가 개선될 때 checkpoint를 저장한다.

주요 코드 위치:

- 데이터로더 생성: `train.py:126-148`
- 모델/optimizer 생성: `train.py:150-154`
- loss 합산과 update: `train.py:184-203`
- epoch 평가: `train.py:236-278`
- best checkpoint 저장: `train.py:299-309`

### Testing

`test.py`는 checkpoint를 로드한 뒤 `model.test_forward(image, label)`로 세 종류의 prediction map을 얻는다.

- `ego_pred`: learned classifier의 raw affordance map
- `refined_CLIP_ego_ego`: CLIP affinity map과 classifier map의 곱
- `refined_CLIP_ego_mean`: CLIP affinity map과 local-smoothed classifier map의 곱

평가지표는 KLD, SIM, NSS이며, 옵션으로 overlay/heatmap 이미지를 저장할 수 있다. 관련 위치는 `test.py:121-193`, 시각화 유틸은 `test.py:69-118`이다.

## 3. 데이터 파이프라인

### TrainData

`data/datatrain.py`는 exocentric root를 기준으로 이미지 목록을 만든다. 각 sample은 하나의 exocentric image path에서 `aff_name`과 `object`를 추출하고, 같은 affordance-object directory의 egocentric 이미지 하나를 random sampling한다. 동시에 같은 exocentric directory에서 총 3장의 exocentric image를 구성한다.

반환 tensor shape는 코드 기준으로 다음과 같다.

- exocentric images: `[num_exo=3, 3, 224, 224]`
- egocentric image: `[3, 224, 224]`
- affordance label: scalar class index

관련 위치:

- split별 affordance list: `data/datatrain.py:19-46`
- transform: `data/datatrain.py:48-54`
- exocentric image index 구성: `data/datatrain.py:56-66`
- egocentric/exocentric sampling: `data/datatrain.py:69-108`

### TestData

`data/datatest.py`는 egocentric test image와 GT mask가 같이 존재하는 sample만 평가 대상으로 추가한다. 평가 시 image, label, mask path를 반환한다.

관련 위치:

- transform: `data/datatest.py:46-50`
- image/mask pair filtering: `data/datatest.py:52-64`
- sample 반환: `data/datatest.py:83-94`

## 4. 모델 전체 구조

핵심 클래스는 `models/locate.py:70`의 `Net`이다.

### 4.1 Frozen feature extractors

모델은 두 개의 pretrained vision backbone을 사용한다.

- DINO ViT-S/16: `self.vit_model`, descriptor 차원 384, patch size 16
- OpenCLIP ViT-B/16: `self.net`, CLIP image/text embedding과 patch affinity map 생성

관련 위치:

- DINO 생성과 weight load: `models/locate.py:81-88`
- CLIP 생성: `models/locate.py:89`
- CLIP `encode_image` 반환 경로: `models/open_clip/model.py:265-267`
- ClearCLIP-style visual token 추출: `models/open_clip/transformer.py:500-568`
- DINO key/attention 추출: `models/dino/vision_transformer.py:257-272`

현재 `forward` 내부에서는 CLIP과 DINO feature extraction이 `torch.no_grad()` 안에서 실행되므로 backbone 자체는 사실상 frozen feature provider로 동작한다. 학습되는 파라미터는 주로 projection/head 계열이다.

### 4.2 Trainable heads

`Net.__init__`에서 학습 가능한 모듈은 다음과 같다.

- `aff_proj`: DINO key descriptor를 384-dim으로 다시 projection하는 MLP
- `aff_ego_proj`: egocentric feature용 2-layer Conv-BN-ReLU block
- `aff_exo_proj`: exocentric feature용 2-layer Conv-BN-ReLU block
- `aff_classifier`: 1x1 conv classifier, output channel은 affordance class 수
- `K_contrast_projection`: prototype contrastive용 1x1 conv projection
- `pixel_contrast_projection`: pixel contrastive용 1x1 conv projection

관련 위치는 `models/locate.py:107-136`이다.

### 4.3 Text prompt 구성

모델은 affordance label을 text prompt로 바꿔 CLIP text feature를 만든다.

- egocentric/object-focused prompt: `an item to {affordance} with`
- exocentric/person-action prompt: `a person {affordance} an item`

관련 위치는 `models/locate.py:202-246`이다. 이 prompt는 CLIP affinity map의 semantic query 역할을 한다.

### 4.4 CLIP affinity map

CLIP image patch features와 text features를 normalize한 뒤 dot product로 similarity map을 만든다.

- egocentric map: `ego patch feature x affordance text`
- exocentric object map: `exo patch feature x exo text`
- exocentric affordance map: `exo patch feature x affordance text`
- 최종 exocentric prior: local mean object map과 affordance map의 곱

관련 위치는 `models/locate.py:167-200`이다. `forward`에서는 이 값들이 `CLIP_ego_similarity`, `CLIP_exo_similarity`로 사용된다.

### 4.5 DINO descriptor 기반 part mining

모델은 DINO의 last key와 attention을 가져와 14x14 patch map으로 reshape한다. egocentric attention은 part 후보 검증용 hard mask로 쓰인다.

관련 위치:

- DINO key/attention 획득: `models/locate.py:265-269`
- descriptor/projection reshape: `models/locate.py:271-279`
- egocentric self-attention hard mask 생성: `models/locate.py:281-286`

part mining 절차는 batch sample별로 수행된다.

1. exocentric CLIP/classifier 기반 mask에서 threshold `gamma1`보다 높은 descriptor만 모은다.
2. descriptor 수가 부족하면 CLIP similarity로 fallback한다.
3. KMeans로 `cluster_num=3`개의 part prototype을 만든다.
4. 각 prototype을 exocentric/egocentric descriptor map에 투영해 similarity map을 만든다.
5. egocentric DINO self-attention hard mask와 가장 잘 겹치는 cluster를 선택한다.
6. overlap score가 `alpha`보다 낮으면 fallback한다.
7. 선택된 cluster similarity와 centroid distance softmax를 평균해 part-level pseudo map을 만든다.

관련 위치는 `models/locate.py:331-417`이다.

### 4.6 Classifier prediction과 CE loss

Projection된 DINO feature는 egocentric/exocentric branch를 거쳐 `aff_classifier`로 class map이 된다. 이후 global average pooling으로 class logits를 만들고 CE loss를 계산한다.

관련 위치:

- projection/classifier forward: `models/locate.py:294-307`
- CE loss: `models/locate.py:309-314`

### 4.7 Contrastive losses

#### PixelContrastiveLoss

`PixelContrastiveLoss`는 egocentric pixel feature 간 contrast를 수행한다. exocentric CLIP similarity의 최대값을 threshold 기준처럼 사용해 egocentric foreground 후보를 만들고, 같은 foreground/background 상태의 pixel을 positive로 둔다.

관련 위치:

- 호출부: `models/locate.py:426-433`
- 구현부: `loss/loss.py:13-87`

#### ContrastiveLoss

`ContrastiveLoss`는 egocentric anchor와 positive/negative prototype을 contrast한다. prototype은 similarity map으로 weighted average한 feature다.

구성:

- ego positive prototype
- ego negative prototype
- exo positive prototype
- exo negative prototype

같은 affordance label의 positive prototype을 당기고, 배경/다른 class는 ignore/negative mask로 처리한다.

관련 위치:

- 호출부: `models/locate.py:435-453`
- 구현부: `loss/loss.py:90-165`

## 5. Inference 구조

`test_forward`는 training forward보다 훨씬 단순하다.

1. affordance label로 CLIP text feature를 만든다.
2. egocentric image에서 CLIP patch similarity map을 만든다.
3. egocentric image에서 DINO key descriptor를 추출한다.
4. `aff_proj -> aff_ego_proj -> aff_classifier`로 class map을 예측한다.
5. label에 해당하는 class map을 선택한다.
6. raw map, CLIP 곱 map, local mean 곱 map을 반환한다.

관련 위치는 `models/locate.py:458-485`이다.

중요한 점은 inference에서는 exocentric branch와 KMeans part mining이 사용되지 않는다는 것이다. exocentric view는 training pseudo supervision을 정교하게 만들기 위한 장치에 가깝다.

## 6. 현재 코드의 강점

- CLIP semantic prior와 DINO structural descriptor를 역할별로 분리해 사용한다.
- exocentric demonstration과 egocentric object-focused image를 같은 affordance-object pair로 묶어 weak supervision을 만든다.
- training-time part mining과 inference-time simple forward가 분리되어 inference 비용이 상대적으로 낮다.
- `test.py`에는 overlay/heatmap 저장 옵션이 있어 qualitative debugging이 가능하다.
- `run_selectivecl_all_gpu3.sh`는 checkpoint download, train, test를 일괄 실행하도록 정리되어 있다.

## 7. 주요 리스크와 개선 제안

### 7.1 설정과 class metadata 중복 제거

현재 affordance list가 `train.py`, `test.py`, `data/datatrain.py`, `data/datatest.py`, `models/locate.py`에 반복 정의되어 있다. split별 class list와 object list를 하나의 config/module로 분리하는 것이 좋다.

기대 효과:

- split 추가 시 변경 지점 감소
- `test.py`의 HICO 미지원 같은 분기 누락 방지
- checkpoint metadata와 class order mismatch 위험 감소

### 7.2 device 처리 정리

모델 내부에서 `.cuda()`를 직접 호출하는 부분이 있다. 예를 들어 `aff_proj`, `aff_classifier`, loss 객체가 생성 시점에 CUDA로 고정된다. 이 방식은 CPU fallback, multi-GPU, DDP, unit test에 불리하다.

개선 방향:

- module 내부 `.cuda()` 제거
- 외부에서 `model.to(device)`만 호출
- tensor 생성 시 `device=input.device` 사용

### 7.3 hard-coded image size 제거

`_reshape_transform`은 `224`를 직접 사용한다. 현재는 crop size가 기본값 224라 동작하지만, argparse에는 `--crop_size`가 있으므로 다른 크기를 넣으면 shape가 깨질 수 있다.

개선 방향:

- feature token 수에서 `height = width = int(sqrt(num_tokens))`를 계산
- 혹은 backbone patch grid 정보를 명시적으로 전달

관련 위치: `models/locate.py:487-492`

### 7.4 KMeans 루프 병목 개선

part mining은 batch sample마다 Python loop와 KMeans를 실행한다. batch size가 커지거나 epoch가 늘면 큰 병목이 된다.

개선 방향:

- KMeans input sampling 수 제한
- prototype update를 EMA memory bank로 대체
- batch vectorization 가능한 부분 분리
- fallback 비율과 KMeans 소요시간을 log에 기록

관련 위치: `models/locate.py:341-417`

### 7.5 numerical stability 보강

몇몇 normalize/log 계산에서 denominator가 0에 가까울 때 불안정할 수 있다.

예시:

- `normalize`는 `max-min`에 epsilon이 없다.
- `PixelContrastiveLoss`에서 foreground positive가 비면 `mean()`이 NaN이 될 수 있다.
- `cal_nss`는 `std == 0`일 때 불안정하다.

개선 방향:

- 모든 min-max normalize에 epsilon 추가
- contrastive mask empty case를 명시적으로 skip 또는 zero loss 처리
- metric 계산에 안정화 epsilon 적용

관련 위치: `models/locate.py:494-498`, `loss/loss.py:71-82`, `utils/evaluation.py:23-37`

### 7.6 학습 loss weighting 도입

현재 loss는 네 항을 모두 동일 가중치로 단순 합산한다. CE, pixel contrast, prototype contrast의 scale이 서로 다르면 특정 loss가 학습을 지배할 수 있다.

개선 방향:

- `lambda_ce_ego`, `lambda_ce_exo`, `lambda_pixel`, `lambda_proto` argparse 추가
- epoch별 warmup/schedule 적용
- loss scale 평균을 log로 남겨 ablation 가능하게 구성

관련 위치: `train.py:192-199`

### 7.7 experiment/reproducibility 강화

현재 `train.py`는 실행 시 `models/locate.py`와 `train.py`만 저장한다. 실제 재현에는 class metadata, git commit, environment, args json, dataset path snapshot이 더 중요하다.

개선 방향:

- `args.json` 저장
- `git rev-parse HEAD`, `git diff --stat` 저장
- `environment.yml` 또는 package freeze 저장
- best checkpoint와 마지막 checkpoint 모두 저장

### 7.8 evaluation split 지원 일관화

`train.py`는 HICO 분기를 갖고 있지만 `test.py`는 Seen/Unseen 중심으로 class 수를 정한다. HICO checkpoint가 제공되는 구조라면 standalone test에서도 HICO를 지원해야 한다.

개선 방향:

- split metadata 공통화
- HICO data root override를 test에도 추가
- HICO mask/image 경로 convention 검증

관련 위치: `train.py:79-88`, `test.py:40-57`

### 7.9 prompt engineering과 prompt ensemble

현재 prompt는 한 가지 template에 의존한다. CLIP prior 품질이 localization pseudo label 품질을 크게 좌우하므로 prompt ensemble을 실험할 가치가 있다.

개선 방향:

- affordance별 template ensemble
- object-aware prompt 추가
- text feature cache 적용
- prompt별 map uncertainty를 pseudo label confidence로 사용

관련 위치: `models/locate.py:202-246`

### 7.10 모듈 분리와 테스트 가능성 개선

`Net.forward`가 feature extraction, prompt 생성, CLIP map 생성, KMeans mining, loss 계산을 모두 담당한다. 기능 단위로 분리하면 ablation과 unit test가 쉬워진다.

권장 분리:

- `PromptEncoder`
- `ClipAffinityBuilder`
- `DinoDescriptorExtractor`
- `PartMiner`
- `AffordanceHead`
- `SelectiveCLLoss`

## 8. 우선순위 높은 개선 로드맵

1. 공통 split metadata/config를 만들고 중복 class list를 제거한다.
2. `.cuda()` 직접 호출과 hard-coded 224를 제거해 device/image-size robustness를 확보한다.
3. normalize/contrastive empty mask의 numerical stability를 보강한다.
4. loss weighting과 logging을 추가해 ablation 가능한 학습 루프를 만든다.
5. KMeans part mining의 fallback ratio, prototype quality, runtime을 log로 측정한다.
6. HICO test path와 class metadata를 정리해 checkpoint 평가 흐름을 완성한다.
7. prompt ensemble과 text feature cache를 도입해 CLIP prior 품질과 속도를 개선한다.

## 9. 빠른 구조 요약

```text
train.py
  -> TrainData: exo 3장 + ego 1장 + affordance label
  -> Net.forward
       -> CLIP text/image features
       -> CLIP ego/exo affinity maps
       -> DINO key descriptors + ego self-attention
       -> affordance classifier maps
       -> exo descriptor KMeans part mining
       -> pixel contrastive loss
       -> prototype contrastive loss
       -> CE losses
  -> epoch-end TestData evaluation
  -> best refined map checkpoint 저장

test.py
  -> TestData: ego image + label + GT mask
  -> Net.test_forward
       -> CLIP ego affinity
       -> DINO descriptor classifier map
       -> raw/refined maps
  -> KLD/SIM/NSS 계산
```
