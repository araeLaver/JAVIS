# JAVIS 아키텍처 문서

## 개요

JAVIS는 개인 AI 비서 시스템으로, 지속적인 모델 버전업을 통해 사용자 맞춤형으로 발전하는 것이 목표입니다.

## 시스템 구조

```
┌─────────────────────────────────────────────────────────────────┐
│                         JAVIS 시스템                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   [클라이언트]                                                   │
│   - 웹 브라우저 (localhost:8000)                                 │
│   - CLI (추후)                                                   │
│   - 음성 인터페이스 (Web Speech API)                             │
│                                                                 │
│         │                                                       │
│         ▼                                                       │
│                                                                 │
│   [FastAPI 서버] ─────────────────────────────────────────────  │
│   - POST /api/chat          : 대화                              │
│   - POST /api/feedback      : 응답 평가                          │
│   - POST /api/clear         : 세션 종료 및 저장                   │
│   - POST /api/upload        : 파일 업로드                        │
│   - POST /api/export-training : 학습 데이터 내보내기              │
│                                                                 │
│         │                                                       │
│         ▼                                                       │
│                                                                 │
│   [Model Client] ◄────────► [LLM Provider]                      │
│   - 현재: Groq API (무료, 임시)                                  │
│   - 목표: RunPod + 자체 파인튜닝 모델                             │
│                                                                 │
│         │                                                       │
│         ▼                                                       │
│                                                                 │
│   [Conversation Logger]                                         │
│   - 모든 대화 자동 저장                                          │
│   - 피드백 수집 (good/bad)                                       │
│   - JSONL 형식 내보내기                                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 디렉토리 구조

```
C:\Develop\workspace\12.JAVIS\
├── .env                          # 환경 변수 (API 키)
├── configs/
│   └── config.yaml               # 앱 설정
├── data/
│   ├── conversations/            # 대화 기록 (JSON)
│   │   └── YYYY-MM/
│   │       └── {session_id}_{timestamp}.json
│   ├── training/
│   │   └── exported/             # 파인튜닝용 데이터 (JSONL)
│   └── uploads/                  # 업로드된 파일
├── docs/                         # 문서
├── javis/
│   ├── data/
│   │   └── conversation_logger.py
│   ├── interfaces/
│   │   └── api.py
│   ├── models/
│   │   └── client.py
│   └── utils/
│       └── config.py
├── models/                       # 파인튜닝된 모델 저장 (추후)
│   └── v1/
│       └── adapter/
└── static/
    └── index.html
```

## 핵심 컴포넌트

### 1. API 서버 (`javis/interfaces/api.py`)
- FastAPI 기반 REST API
- 세션 관리 (in-memory, 추후 Redis)
- CORS 지원

### 2. 모델 클라이언트 (`javis/models/client.py`)
- OpenAI 호환 API 형식
- 현재 Groq API 사용
- Provider 교체 가능한 구조

### 3. 대화 로거 (`javis/data/conversation_logger.py`)
- 실시간 대화 기록
- 피드백 수집
- 학습 데이터 내보내기

### 4. 웹 UI (`static/index.html`)
- 채팅 인터페이스
- 음성 입력/출력 (STT/TTS)
- 파일 업로드
- 피드백 버튼

## 데이터 형식

### 대화 기록 (JSON)
```json
{
  "session_id": "session_abc123",
  "started_at": "2025-12-18T18:51:57",
  "turns": [
    {
      "role": "user",
      "content": "안녕",
      "timestamp": "2025-12-18T18:51:57"
    },
    {
      "role": "assistant",
      "content": "안녕하세요!",
      "timestamp": "2025-12-18T18:51:58"
    }
  ],
  "feedback": "good",
  "tags": ["greeting"]
}
```

### 학습 데이터 (JSONL)
```json
{"messages": [{"role": "user", "content": "안녕"}, {"role": "assistant", "content": "안녕하세요!"}]}
```

## 기술 스택

| 구분 | 기술 |
|------|------|
| 백엔드 | Python 3.13, FastAPI, Pydantic |
| HTTP 클라이언트 | httpx (async) |
| 설정 관리 | python-dotenv, PyYAML |
| 현재 LLM | Groq API (llama-3.1-8b-instant) |
| 목표 LLM | Qwen2.5-7B-Instruct + QLoRA |
| 프론트엔드 | Vanilla JS, Web Speech API |
| 배포 (예정) | RunPod Serverless |
