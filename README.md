# JAVIS

**Personal AI Assistant with Custom Fine-tuned Model**

나만의 AI 비서. Groq API로 빠른 응답, Google Colab으로 무료 파인튜닝, 지속적인 모델 버전업.

## 현재 상태

- [x] Groq API 연동 (Llama 3.3 70B)
- [x] 웹 UI (음성 입력/출력)
- [x] 대화 로깅 (파인튜닝 데이터 수집)
- [x] 피드백 시스템 (좋아요/싫어요)
- [x] Google Colab 파인튜닝 (Qwen2.5-7B + QLoRA)
- [x] 모델 버전 관리 (v1.0 완료)
- [ ] Koyeb 클라우드 배포 (진행중)

## Quick Start

### 1. 설치

```bash
git clone https://github.com/araeLaver/JAVIS.git
cd JAVIS

python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Mac/Linux

pip install -r requirements.txt
```

### 2. 환경변수

```bash
# .env 파일 생성
copy .env.example .env

# .env 수정
GROQ_API_KEY=gsk_xxxxxxxxxxxxxx
```

### 3. 실행

```bash
# 서버 시작
uvicorn javis.interfaces.api:app --reload

# 브라우저에서 http://localhost:8000 접속
```

## 주요 기능

### 웹 인터페이스
- 텍스트 채팅
- 음성 입력 (마이크 버튼)
- 음성 출력 (TTS)
- 피드백 (좋아요/싫어요)

### 파인튜닝 워크플로우

```bash
# 1. 대화 통계 확인
python -m javis.training.manage stats

# 2. 학습 데이터 내보내기
python -m javis.training.manage export

# 3. Google Colab에서 파인튜닝
#    notebooks/finetune_colab.ipynb 사용

# 4. 어댑터를 models/버전/adapter/에 압축 해제
```

### 모델 버전 관리

```bash
# 버전 목록
python -m javis.training.manage list

# 출력:
# Version    Created      Dataset Size    Status
# v1.0       2025-12-19   2               ready
```

## 프로젝트 구조

```
JAVIS/
├── javis/
│   ├── interfaces/
│   │   └── api.py          # FastAPI 서버
│   ├── models/
│   │   ├── groq_client.py  # Groq API 클라이언트
│   │   └── local_client.py # 로컬 모델 클라이언트
│   ├── training/
│   │   └── manage.py       # 버전/데이터 관리 CLI
│   └── utils/
│       ├── config.py       # 설정
│       └── conversation_logger.py  # 대화 로깅
├── notebooks/
│   └── finetune_colab.ipynb  # Colab 파인튜닝 노트북
├── data/
│   ├── conversations/      # 대화 로그 (JSON)
│   └── training/exported/  # 학습 데이터 (JSONL)
├── models/                 # 파인튜닝된 어댑터 (로컬 전용)
├── static/
│   └── index.html          # 웹 UI
└── configs/
    └── config.yaml         # 설정 파일
```

## 기술 스택

| 구분 | 기술 |
|------|------|
| API | Groq (Llama 3.3 70B) - 무료, 빠름 |
| 파인튜닝 | QLoRA + Qwen2.5-7B-Instruct |
| 학습 환경 | Google Colab (무료 T4 GPU) |
| 백엔드 | FastAPI + Uvicorn |
| 프론트엔드 | HTML + Vanilla JS |
| 음성 | Web Speech API |
| 배포 | Koyeb (무료) |

## 파인튜닝 가이드

### 1. 데이터 수집
- 웹 UI에서 대화
- 좋은 응답에 좋아요 클릭
- 데이터가 data/conversations/에 자동 저장

### 2. 데이터 내보내기
```bash
python -m javis.training.manage export
# data/training/exported/conversations_YYYYMMDD_HHMMSS.jsonl 생성
```

### 3. Google Colab 학습
1. notebooks/finetune_colab.ipynb를 Colab에 업로드
2. 런타임 - 런타임 유형 변경 - T4 GPU
3. 셀 순서대로 실행 (1-2시간)
4. javis-adapter.zip 다운로드

### 4. 어댑터 적용
```bash
# models/v1.0/adapter/ 에 압축 해제
python -m javis.training.manage list
```

## 클라우드 배포 (Koyeb)

1. https://koyeb.com 가입 (GitHub 계정)
2. Create App - GitHub - araeLaver/JAVIS 선택
3. 설정:
   - Builder: Buildpack
   - Build command: (비워두기)
   - Run command: (비워두기 - Procfile 사용)
   - Port: 8000
4. Environment Variables:
   - GROQ_API_KEY = gsk_xxxxxx
5. Deploy

## API 엔드포인트

| Method | Path | 설명 |
|--------|------|------|
| GET | / | 웹 UI |
| POST | /chat | 채팅 |
| POST | /feedback | 피드백 저장 |
| GET | /conversations | 대화 목록 |
| POST | /export | 학습 데이터 내보내기 |
| GET | /health | 상태 확인 |

## 버전 히스토리

| 버전 | 날짜 | 데이터 | 설명 |
|------|------|--------|------|
| v1.0 | 2025-12-19 | 2 대화 | 첫 파인튜닝 |

## License

MIT License
