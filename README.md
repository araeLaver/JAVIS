# JAVIS

**Personal AI Assistant with Custom Fine-tuned Model**

나만의 AI 비서 시스템. 오픈소스 모델을 기반으로 파인튜닝하여 완전한 통제권을 가진 개인 AI를 구축합니다.

## Features

- **나만의 모델**: 오픈소스 LLM + 커스텀 파인튜닝
- **완전한 통제권**: 모델, 데이터, 인프라 모두 소유
- **지속적 발전**: 버전업을 통한 꾸준한 모델 향상
- **클라우드 기반**: RunPod Serverless로 비용 효율적 운영

## Quick Start

### 1. 환경 설정

```bash
# 저장소 클론 (또는 현재 디렉토리 사용)
cd 12.JAVIS

# 가상환경 생성
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 의존성 설치
pip install -e .
```

### 2. 환경변수 설정

```bash
# .env 파일 생성
cp .env.example .env

# .env 파일 편집하여 API 키 입력
# RUNPOD_API_KEY=your_key
# RUNPOD_ENDPOINT_ID=your_endpoint
```

### 3. 실행

```bash
# CLI 실행
python -m javis

# 또는
javis chat
```

## Project Structure

```
12.JAVIS/
├── docs/                    # 프로젝트 문서
├── javis/                   # 메인 소스코드
│   ├── core/               # 코어 엔진
│   ├── models/             # 모델 클라이언트
│   ├── tools/              # 도구 시스템
│   ├── memory/             # 메모리 시스템
│   ├── rag/                # RAG 시스템
│   ├── training/           # 학습 파이프라인
│   └── interfaces/         # CLI/API
├── data/                    # 데이터
│   ├── training/           # 학습 데이터
│   └── vectors/            # 벡터 DB
├── models/                  # 모델 어댑터
├── configs/                 # 설정 파일
└── scripts/                 # 유틸리티 스크립트
```

## Documentation

- [01-PROJECT-OVERVIEW.md](docs/01-PROJECT-OVERVIEW.md) - 프로젝트 개요
- [02-ROADMAP.md](docs/02-ROADMAP.md) - 개발 로드맵
- [03-MODEL-VERSIONING.md](docs/03-MODEL-VERSIONING.md) - 모델 버전업 전략
- [04-TECH-STACK.md](docs/04-TECH-STACK.md) - 기술 스택 및 인프라

## Roadmap

- [x] Phase 0: 환경 구축
- [ ] Phase 1: MVP (CLI 대화)
- [ ] Phase 2: 첫 파인튜닝
- [ ] Phase 3: 도구 시스템
- [ ] Phase 4: 메모리 & RAG
- [ ] Phase 5: 지속적 학습 파이프라인

## Tech Stack

- **Language**: Python 3.11+
- **Base Model**: Qwen2.5-7B-Instruct
- **Fine-tuning**: QLoRA (Unsloth)
- **Serving**: RunPod Serverless (vLLM)
- **Vector DB**: Chroma

## License

MIT License
