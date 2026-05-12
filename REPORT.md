# SelectiveCL 실행 리포트

작성일: 2026-05-08 KST

## 요약

- `/root/workspace/andycho/CV/SelectiveCL/run_selectivecl_all_gpu3.sh` 스크립트를 작성했다.
- 스크립트는 `selectivecl` conda 환경을 활성화하고, 물리 GPU 3번만 보이도록 `CUDA_VISIBLE_DEVICES=3`을 설정한다.
- GPU 3번만 노출되면 Python 내부에서는 해당 GPU가 logical GPU 0번이 되므로, `train.py`와 `test.py`에는 `--gpu 0`을 넘긴다.
- README의 Google Drive checkpoint 3개를 모두 다운로드했다.
- AGD20K Seen/Unseen의 full training 및 official checkpoint test를 순차 실행하도록 구성했다.
- HICO-IIF checkpoint는 다운로드했지만, `/DATA/HICO-IIF` 데이터가 없고 현재 `test.py`가 AGD20K Seen/Unseen 중심으로 작성되어 있어 HICO 실행은 스킵한다.

## 체크포인트

다운로드 위치:

```bash
/root/workspace/andycho/CV/SelectiveCL/checkpoints
```

다운로드된 파일:

```text
agd20k_seen.pth    712621477 bytes
agd20k_unseen.pth  712587557 bytes
hico_iif.pth       712541349 bytes
```

스크립트는 이미 파일이 존재하고 크기가 0보다 크면 재다운로드하지 않는다.

## 실행 스크립트 사용법

기본 실행:

```bash
cd /root/workspace/andycho/CV/SelectiveCL
./run_selectivecl_all_gpu3.sh
```

foreground 실행:

```bash
./run_selectivecl_all_gpu3.sh --foreground
```

checkpoint 다운로드만 실행:

```bash
./run_selectivecl_all_gpu3.sh --foreground --download-only
```

스크립트 내부 실행 순서:

```text
1. checkpoint 확인 및 다운로드
2. train.py --divide Seen
3. train.py --divide Unseen
4. test.py --model_file checkpoints/agd20k_seen.pth --divide Seen
5. test.py --model_file checkpoints/agd20k_unseen.pth --divide Unseen
```

## 현재 실행 상태

이 작업 환경에서는 일반적인 `nohup ... &` 백그라운드 프로세스가 API 명령 종료와 함께 바로 종료되었다. 그래서 실제 장기 실행은 `tmux` 세션으로 띄웠다.

현재 tmux 세션:

```bash
selectivecl_full_gpu3_20260508_023621
```

접속:

```bash
tmux attach -t selectivecl_full_gpu3_20260508_023621
```

tmux에서 빠져나오기:

```text
Ctrl-b 누른 뒤 d
```

현재 run directory:

```bash
/root/workspace/andycho/CV/SelectiveCL/full_runs/20260508_023621
```

현재 상태:

```text
train_seen 단계 시작됨
CLIP pretrained ViT-B-16 로드 완료
GPU 3번에서 Python 프로세스 실행 확인됨
```

## 주요 로그 확인 명령

전체 진행 로그:

```bash
tail -f /root/workspace/andycho/CV/SelectiveCL/full_runs/20260508_023621/orchestrator.log
```

Seen training 로그:

```bash
tail -f /root/workspace/andycho/CV/SelectiveCL/full_runs/20260508_023621/train_seen.log
```

Unseen training 로그:

```bash
tail -f /root/workspace/andycho/CV/SelectiveCL/full_runs/20260508_023621/train_unseen.log
```

Seen official checkpoint test 로그:

```bash
tail -f /root/workspace/andycho/CV/SelectiveCL/full_runs/20260508_023621/test_seen_official.log
```

Unseen official checkpoint test 로그:

```bash
tail -f /root/workspace/andycho/CV/SelectiveCL/full_runs/20260508_023621/test_unseen_official.log
```

GPU 상태:

```bash
nvidia-smi
```

tmux 세션 확인:

```bash
tmux list-sessions
```

## 중단 방법

실행 중인 tmux 세션을 종료하려면:

```bash
tmux kill-session -t selectivecl_full_gpu3_20260508_023621
```

주의: 이 명령은 현재 진행 중인 training/test를 중단한다.

## 완료 판단 기준

`orchestrator.log`에 아래 메시지가 나오면 전체 파이프라인이 끝난 것이다.

```text
SelectiveCL full run completed
```

각 단계별 완료 기준:

- `train_seen.log`: epoch metric 로그와 checkpoint 저장 여부 확인
- `train_unseen.log`: epoch metric 로그와 checkpoint 저장 여부 확인
- `test_seen_official.log`: `KLD, SIM, NSS`, `reeKLD`, `remKLD` 출력 확인
- `test_unseen_official.log`: `KLD, SIM, NSS`, `reeKLD`, `remKLD` 출력 확인

## 주의사항

- 전체 학습은 smoke test가 아니라 full training이므로 오래 걸린다.
- 같은 GPU 3번에서 중복 실행하면 VRAM과 속도 문제가 생길 수 있으므로, 새로 실행하기 전에 기존 tmux 세션 상태를 먼저 확인해야 한다.
- HICO-IIF는 checkpoint만 받아둔 상태다. HICO까지 실제 test를 돌리려면 `/DATA/HICO-IIF` 데이터 준비와 `test.py`의 HICO 분기 보완이 필요하다.
- `run_selectivecl_all_gpu3.sh`의 기본 백그라운드 방식은 일반 터미널에서는 동작할 수 있지만, 이 Codex/API 실행 환경에서는 장기 프로세스 유지를 위해 tmux 실행을 사용했다.
