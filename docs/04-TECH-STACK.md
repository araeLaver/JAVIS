# JAVIS 기술 스택 및 인프라

## 기술 스택 개요

```
┌─────────────────────────────────────────────────────────────┐
│                        기술 스택                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  [언어 & 런타임]                                             │
│  └── Python 3.11+                                           │
│                                                             │
│  [AI/ML]                                                    │
│  ├── 베이스 모델: Qwen2.5-7B-Instruct                        │
│  ├── 파인튜닝: Unsloth / Axolotl                            │
│  ├── 서빙: vLLM                                             │
│  └── 임베딩: sentence-transformers                          │
│                                                             │
│  [인프라]                                                    │
│  ├── 모델 서빙: RunPod Serverless                           │
│  ├── 파인튜닝: RunPod GPU Pods                              │
│  └── 스토리지: 로컬 + 클라우드 백업                          │
│                                                             │
│  [데이터베이스]                                               │
│  ├── 메타데이터: SQLite (로컬)                               │
│  └── 벡터 DB: Chroma (로컬)                                 │
│                                                             │
│  [핵심 라이브러리]                                           │
│  ├── httpx: HTTP 클라이언트                                 │
│  ├── pydantic: 데이터 검증                                  │
│  ├── rich: CLI UI                                          │
│  └── typer: CLI 프레임워크                                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 상세 기술 스택

### 1. 언어 및 런타임

| 항목 | 선택 | 이유 |
|------|------|------|
| 언어 | Python 3.11+ | ML 생태계, 라이브러리 풍부 |
| 패키지 관리 | uv 또는 pip | 빠른 의존성 관리 |
| 가상환경 | venv | 단순, 표준 |

### 2. AI/ML 스택

#### 베이스 모델

```yaml
model:
  name: Qwen2.5-7B-Instruct
  provider: Alibaba/Qwen
  license: Apache 2.0
  parameters: 7B
  context_length: 128K
  languages: [en, zh, ko, ja, ...]  # 다국어 지원

huggingface:
  repo: Qwen/Qwen2.5-7B-Instruct
  size: ~15GB
```

#### 파인튜닝 도구

```yaml
# 옵션 1: Unsloth (추천 - 빠름)
unsloth:
  speed: 2-5x faster than standard
  memory: 40-60% less VRAM
  methods: [LoRA, QLoRA]

# 옵션 2: Axolotl (유연함)
axolotl:
  flexibility: high
  methods: [LoRA, QLoRA, full finetune]
  config: YAML based

# 옵션 3: LLaMA-Factory (GUI 있음)
llama_factory:
  ui: WebUI available
  methods: [LoRA, QLoRA, DPO, RLHF]
```

#### 모델 서빙

```yaml
# vLLM - 고성능 추론 엔진
vllm:
  features:
    - PagedAttention (메모리 효율)
    - Continuous batching
    - OpenAI 호환 API
  throughput: 10-24x faster than HF
```

### 3. 인프라

#### RunPod 구성

```yaml
# Serverless (추론용)
serverless:
  type: Serverless Endpoint
  base_image: runpod/pytorch:2.1.0-py3.10-cuda11.8.0
  model: Qwen2.5-7B-Instruct + LoRA
  gpu: A10G / RTX 4090 (자동 할당)
  scaling:
    min_workers: 0
    max_workers: 3
  pricing: ~$0.00025/초

# GPU Pod (학습용)
training_pod:
  type: GPU Pod
  gpu: A100 40GB (권장) 또는 RTX 4090
  hourly_rate: $1.5-2.5/시간
  usage: 필요할 때만 실행
```

#### 로컬 환경

```yaml
development:
  os: Windows / macOS / Linux
  python: 3.11+
  storage:
    - configs/
    - data/
    - logs/
  database:
    - sqlite (메타데이터)
    - chroma (벡터)
```

### 4. 데이터베이스

```yaml
# SQLite - 메타데이터, 대화 기록
sqlite:
  use_case:
    - 대화 히스토리
    - 설정 저장
    - 피드백 기록
  location: data/javis.db

# Chroma - 벡터 데이터베이스
chroma:
  use_case:
    - 문서 임베딩 저장
    - 의미 검색 (RAG)
    - 장기 기억
  location: data/vectors/
  embedding_model: sentence-transformers/all-MiniLM-L6-v2
```

### 5. 핵심 Python 패키지

```toml
# pyproject.toml

[project]
name = "javis"
version = "0.1.0"
requires-python = ">=3.11"

dependencies = [
    # HTTP & API
    "httpx>=0.25.0",              # 비동기 HTTP 클라이언트
    "websockets>=12.0",           # WebSocket (스트리밍)

    # CLI
    "typer>=0.9.0",               # CLI 프레임워크
    "rich>=13.0.0",               # 터미널 UI

    # 데이터 처리
    "pydantic>=2.5.0",            # 데이터 검증
    "pyyaml>=6.0",                # YAML 설정
    "python-dotenv>=1.0.0",       # 환경변수

    # 데이터베이스
    "sqlalchemy>=2.0.0",          # ORM
    "chromadb>=0.4.0",            # 벡터 DB

    # AI/ML (로컬 임베딩용)
    "sentence-transformers>=2.2.0",

    # 유틸리티
    "aiofiles>=23.0.0",           # 비동기 파일 처리
    "tenacity>=8.2.0",            # 재시도 로직
]

[project.optional-dependencies]
training = [
    "unsloth>=2024.1",            # 파인튜닝
    "transformers>=4.36.0",
    "datasets>=2.16.0",
    "accelerate>=0.25.0",
    "bitsandbytes>=0.42.0",
    "peft>=0.7.0",
]

dev = [
    "pytest>=7.4.0",
    "ruff>=0.1.0",
    "mypy>=1.7.0",
]
```

---

## 프로젝트 구조

```
12.JAVIS/
│
├── docs/                          # 문서
│   ├── 01-PROJECT-OVERVIEW.md
│   ├── 02-ROADMAP.md
│   ├── 03-MODEL-VERSIONING.md
│   └── 04-TECH-STACK.md
│
├── javis/                         # 메인 소스코드
│   ├── __init__.py
│   ├── __main__.py               # 진입점
│   │
│   ├── core/                     # 코어 엔진
│   │   ├── __init__.py
│   │   ├── engine.py            # 메인 오케스트레이션
│   │   ├── conversation.py      # 대화 관리
│   │   ├── context.py           # 컨텍스트 구성
│   │   └── router.py            # 의도 라우팅
│   │
│   ├── models/                   # 모델 연동
│   │   ├── __init__.py
│   │   ├── client.py            # RunPod API 클라이언트
│   │   ├── schemas.py           # 요청/응답 스키마
│   │   └── prompts.py           # 프롬프트 템플릿
│   │
│   ├── tools/                    # 도구 시스템
│   │   ├── __init__.py
│   │   ├── base.py              # 도구 베이스 클래스
│   │   ├── registry.py          # 도구 등록
│   │   ├── file_tools.py        # 파일 작업
│   │   ├── shell_tools.py       # 쉘 명령
│   │   ├── web_tools.py         # 웹 검색
│   │   └── system_tools.py      # 시스템 도구
│   │
│   ├── memory/                   # 메모리 시스템
│   │   ├── __init__.py
│   │   ├── short_term.py        # 단기 기억
│   │   ├── long_term.py         # 장기 기억 (벡터)
│   │   └── storage.py           # 저장소
│   │
│   ├── rag/                      # RAG 시스템
│   │   ├── __init__.py
│   │   ├── loader.py            # 문서 로더
│   │   ├── chunker.py           # 청크 분할
│   │   ├── embedder.py          # 임베딩
│   │   └── retriever.py         # 검색
│   │
│   ├── training/                 # 학습 파이프라인
│   │   ├── __init__.py
│   │   ├── data_collector.py    # 피드백 수집
│   │   ├── data_formatter.py    # 데이터 포맷
│   │   ├── trainer.py           # 파인튜닝 실행
│   │   └── evaluator.py         # 모델 평가
│   │
│   ├── interfaces/               # 사용자 인터페이스
│   │   ├── __init__.py
│   │   ├── cli.py               # CLI 인터페이스
│   │   └── api.py               # REST API (추후)
│   │
│   └── utils/                    # 유틸리티
│       ├── __init__.py
│       ├── config.py            # 설정 관리
│       ├── logging.py           # 로깅
│       └── helpers.py           # 헬퍼 함수
│
├── data/                         # 데이터 디렉토리
│   ├── training/                # 학습 데이터
│   │   └── v0.1/
│   ├── feedback/                # 피드백 데이터
│   ├── vectors/                 # 벡터 DB
│   └── documents/               # RAG용 문서
│
├── models/                       # 모델 관련
│   ├── adapters/                # LoRA 어댑터
│   │   └── javis-v0.1/
│   └── configs/                 # 학습 설정
│
├── configs/                      # 설정 파일
│   ├── config.yaml              # 메인 설정
│   └── prompts.yaml             # 프롬프트 설정
│
├── scripts/                      # 스크립트
│   ├── train.py                 # 학습 스크립트
│   ├── deploy.py                # 배포 스크립트
│   └── evaluate.py              # 평가 스크립트
│
├── tests/                        # 테스트
│   └── ...
│
├── .env.example                  # 환경변수 예시
├── .gitignore
├── pyproject.toml               # 프로젝트 설정
└── README.md
```

---

## 인프라 아키텍처

```
┌─────────────────────────────────────────────────────────────────┐
│                      전체 인프라 구성                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   [로컬 환경 - 당신의 PC]                                        │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                                                         │   │
│   │   ┌───────────────┐    ┌───────────────┐               │   │
│   │   │  JAVIS CLI    │    │  데이터 관리   │               │   │
│   │   │  (Python)     │    │  (SQLite,     │               │   │
│   │   │               │    │   Chroma)     │               │   │
│   │   └───────┬───────┘    └───────────────┘               │   │
│   │           │                                             │   │
│   └───────────┼─────────────────────────────────────────────┘   │
│               │                                                 │
│               │ HTTPS API                                       │
│               │                                                 │
│   ┌───────────┼─────────────────────────────────────────────┐   │
│   │           ▼                                             │   │
│   │   [RunPod - 클라우드]                                    │   │
│   │                                                         │   │
│   │   ┌─────────────────────────────────────────────────┐   │   │
│   │   │  Serverless Endpoint (추론)                      │   │   │
│   │   │  ┌─────────────────────────────────────────┐    │   │   │
│   │   │  │  vLLM Server                            │    │   │   │
│   │   │  │  ├── Qwen2.5-7B-Instruct (베이스)       │    │   │   │
│   │   │  │  └── javis-lora-v0.x (어댑터)           │    │   │   │
│   │   │  └─────────────────────────────────────────┘    │   │   │
│   │   │                                                 │   │   │
│   │   │  Auto Scaling: 0 ~ N workers                    │   │   │
│   │   │  과금: 사용한 시간만                              │   │   │
│   │   └─────────────────────────────────────────────────┘   │   │
│   │                                                         │   │
│   │   ┌─────────────────────────────────────────────────┐   │   │
│   │   │  GPU Pod (학습용, 필요시만)                       │   │   │
│   │   │  ├── A100 40GB                                  │   │   │
│   │   │  ├── 파인튜닝 실행                               │   │   │
│   │   │  └── 완료 후 종료                                │   │   │
│   │   └─────────────────────────────────────────────────┘   │   │
│   │                                                         │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│   [Hugging Face - 모델 저장소]                                   │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  • 베이스 모델 다운로드                                   │   │
│   │  • LoRA 어댑터 업로드/다운로드 (선택)                      │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 환경 설정

### 환경변수 (.env)

```bash
# .env.example

# RunPod
RUNPOD_API_KEY=your_runpod_api_key
RUNPOD_ENDPOINT_ID=your_endpoint_id

# Hugging Face (모델 다운로드용)
HF_TOKEN=your_huggingface_token

# 앱 설정
JAVIS_ENV=development  # development / production
JAVIS_LOG_LEVEL=INFO
JAVIS_DATA_DIR=./data
```

### 메인 설정 (config.yaml)

```yaml
# configs/config.yaml

app:
  name: JAVIS
  version: "0.1.0"

model:
  provider: runpod
  endpoint_id: ${RUNPOD_ENDPOINT_ID}

  generation:
    max_tokens: 2048
    temperature: 0.7
    top_p: 0.9

conversation:
  system_prompt: |
    너는 JAVIS, 개발자를 위한 개인 AI 비서다.
    간결하고 정확하게 답변하며, 필요할 때 도구를 사용한다.

  max_history: 20

tools:
  enabled: true
  available:
    - file_tools
    - shell_tools
    - web_tools

memory:
  short_term:
    max_messages: 50
  long_term:
    enabled: true
    db_path: ./data/vectors

logging:
  level: INFO
  file: ./logs/javis.log
```

---

## 비용 상세

### 월간 비용 예상

| 항목 | 사용량 | 단가 | 월 비용 |
|------|--------|------|---------|
| **서빙 (Serverless)** | | | |
| - 일 50회 요청 | 1,500회/월 | ~$0.01/회 | $15-30 |
| - 일 100회 요청 | 3,000회/월 | ~$0.01/회 | $30-60 |
| **파인튜닝** | | | |
| - 월 2회 학습 | 4시간 | $2/시간 | $8-16 |
| **스토리지** | | | |
| - 로컬 | - | - | $0 |
| - 클라우드 백업 (선택) | 10GB | - | $1-5 |

### 총 예상 비용

| 사용 수준 | 월 비용 |
|-----------|---------|
| 라이트 (일 30회) | $25-40 |
| 보통 (일 50회) | $40-70 |
| 헤비 (일 100회) | $70-130 |

---

## 보안 고려사항

```yaml
security:
  api_keys:
    - 환경변수로 관리 (.env)
    - Git에 절대 커밋 금지

  data:
    - 민감한 대화는 로컬에만 저장
    - 학습 데이터 암호화 (선택)

  network:
    - HTTPS only
    - API 키 인증
```

---

## 다음 단계

1. 개발 환경 세팅
2. RunPod 계정 생성 및 설정
3. 프로젝트 기본 구조 생성
4. 첫 번째 MVP 구현 시작
