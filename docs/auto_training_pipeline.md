# JAVIS 자동 재학습 파이프라인

## 개요

JAVIS AI 어시스턴트의 자동 재학습 파이프라인 구현 문서입니다.
Modal.com GPU를 사용하여 스케줄 기반으로 모델을 자동 학습하고 배포합니다.

## 아키텍처

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  Scheduler  │────>│   Pipeline   │────>│  Deployer   │
│ (APScheduler)│     │ (Orchestrator)│     │ (Version Mgr)│
└─────────────┘     └──────────────┘     └─────────────┘
       │                   │                    │
       v                   v                    v
  cron 스케줄         Modal.com GPU         모델 버전관리
  조건 체크           원격 학습             자동 롤백
```

## 생성된 파일

### 핵심 모듈 (`javis/training/`)

| 파일 | 설명 | 라인 수 |
|------|------|---------|
| `scheduler.py` | APScheduler 기반 백그라운드 스케줄러 | ~150 |
| `remote.py` | Modal.com GPU 학습 함수 | ~280 |
| `pipeline.py` | 전체 파이프라인 오케스트레이션 | ~200 |
| `version_manager.py` | 모델 버전 관리 및 롤백 | ~150 |
| `notifications.py` | Discord 알림 | ~100 |
| `manage.py` | CLI 인터페이스 (기존 파일 확장) | - |

### 학습 스크립트

| 파일 | 설명 |
|------|------|
| `run_training.py` | Modal 직접 실행 스크립트 (진행률 표시) |
| `run_modal.py` | Windows 인코딩 문제 해결 래퍼 |
| `test_modal_train.py` | Modal GPU 연결 테스트 |
| `debug_train.py` | 로컬 디버깅용 스크립트 |

## 설정

### `configs/config.yaml` 추가 항목

```yaml
training:
  schedule:
    enabled: true
    cron: "0 0 * * 0"  # 매주 일요일 자정
    timezone: "Asia/Seoul"
  provider: "modal"  # modal 또는 local
  data:
    min_conversations: 50
    exclude_bad_feedback: true
  deployment:
    auto_deploy: true
    keep_versions: 5
  notifications:
    discord_webhook: null
```

### 환경 변수 (`.env`)

```bash
MODAL_TOKEN_ID=ak-xxxxx
MODAL_TOKEN_SECRET=as-xxxxx
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/xxx  # 선택사항
```

## CLI 사용법

```bash
# 학습 상태 확인
python -m javis.training.manage status

# 수동 학습 실행
python -m javis.training.manage train
python -m javis.training.manage train --force  # 조건 무시하고 강제 실행

# 스케줄러 제어
python -m javis.training.manage scheduler start
python -m javis.training.manage scheduler stop
python -m javis.training.manage scheduler status

# 버전 관리
python -m javis.training.manage versions        # 버전 목록
python -m javis.training.manage rollback v20251230  # 롤백

# 데이터 내보내기
python -m javis.training.manage export
```

## 학습 파이프라인 흐름

1. **조건 체크**: 최소 대화 수, 마지막 학습 이후 경과 시간
2. **데이터 준비**: 대화 데이터 → JSONL 포맷 변환
3. **원격 학습**: Modal.com A10G GPU에서 QLoRA 학습
4. **어댑터 저장**: `models/v{날짜}/adapter/` 에 저장
5. **버전 등록**: 활성 버전으로 설정
6. **알림 발송**: Discord 웹훅 (설정된 경우)

## 학습 설정

### 기본 학습 파라미터

| 파라미터 | 값 | 설명 |
|----------|-----|------|
| base_model | Qwen/Qwen2.5-7B-Instruct | 베이스 모델 |
| epochs | 3 | 에포크 수 |
| batch_size | 2 | 배치 크기 |
| learning_rate | 2e-4 | 학습률 |
| lora_r | 64 | LoRA rank |
| lora_alpha | 16 | LoRA alpha |
| max_length | 2048 | 최대 시퀀스 길이 |
| gradient_accumulation_steps | 4 | 그래디언트 누적 |

### QLoRA 설정

- 4-bit 양자화 (NF4)
- Double quantization 활성화
- bfloat16 연산
- Paged AdamW 8-bit 옵티마이저

## 해결된 이슈

### 1. Windows 인코딩 문제 (cp949)
- **문제**: Modal 출력이 cp949로 인코딩되어 Unicode 에러 발생
- **해결**: `run_modal.py` 래퍼로 UTF-8 인코딩 강제 + 로그 파일 출력

### 2. NumPy 2.x 호환성
- **문제**: NumPy 2.x와 PyTorch 2.1.0 호환 안됨
- **해결**: `numpy<2.0` 버전 제약 추가

### 3. TRL 버전 호환성
- **문제**: `SFTConfig` import 에러 (trl 0.7.x)
- **해결**: `trl>=0.12.0` 으로 업그레이드

### 4. SFTConfig 파라미터 변경
- **문제**: `max_seq_length` 파라미터가 `max_length`로 변경됨
- **해결**: 파라미터명 수정 (`remote.py`, `run_training.py`)

## Modal.com 의존성

```python
training_image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "numpy<2.0",
    "torch>=2.1.0,<2.5.0",
    "transformers>=4.40.0",
    "datasets>=2.21.0",
    "peft>=0.10.0",
    "trl>=0.12.0",
    "bitsandbytes>=0.42.0",
    "accelerate>=0.34.0",
    "scipy",
    "sentencepiece",
)
```

## 테스트 결과

### 2024-12-30 학습 테스트

```
GPU: NVIDIA A10G
모델: Qwen/Qwen2.5-7B-Instruct
학습 파라미터: 161,480,704 / 4,514,452,992 (3.6%)
학습 시간: 76.5초
Loss: 3.729
어댑터 크기: 308MB
저장 위치: models/v20251230/adapter
```

## 비용 예상

| 항목 | 비용 |
|------|------|
| Modal A10G | $1.10/hr |
| 주 1회 학습 (1-2시간) | $4-8/월 |

## 향후 개선사항

- [ ] 학습 데이터 품질 필터링 강화
- [ ] 모델 평가 메트릭 추가
- [ ] A/B 테스트 지원
- [ ] 분산 학습 지원
