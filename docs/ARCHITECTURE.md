# JAVIS 아키텍처 문서

## 목차

1. [프로젝트 개요](#1-프로젝트-개요)
2. [시스템 아키텍처](#2-시스템-아키텍처)
3. [디렉토리 구조](#3-디렉토리-구조)
4. [핵심 모듈 상세](#4-핵심-모듈-상세)
5. [데이터 흐름](#5-데이터-흐름)
6. [학습 파이프라인](#6-학습-파이프라인)
7. [기술 스택](#7-기술-스택)
8. [설정 및 환경변수](#8-설정-및-환경변수)
9. [API 명세](#9-api-명세)
10. [확장 가이드](#10-확장-가이드)

---

## 1. 프로젝트 개요

### 1.1 비전

**"타사 API에 의존하지 않는, 완전히 내가 통제하는 개인 AI 비서 시스템"**

### 1.2 핵심 원칙

```
├── 완전한 통제권: 모델, 데이터, 인프라 모두 내 소유
├── 지속적 발전: 버전업을 통한 꾸준한 모델 향상
├── 독립성: 외부 서비스 의존 최소화
└── 확장성: 작게 시작해서 점진적으로 확장
```

### 1.3 주요 기능

| Phase | 기능 | 상태 |
|-------|------|------|
| 1-2 | 기본 채팅, Groq API, 웹 UI, 대화 로깅 | ✅ 완료 |
| 3 | 도구 호출 시스템 (Tool Calling) | ✅ 완료 |
| 4 | 장기 메모리 & RAG 시스템 | ✅ 완료 |
| 5 | 음성 I/O, 자동 학습, DPO | ✅ 완료 |
| 6 | 워크플로우 자동화 | ✅ 완료 |
| 7 | 외부 서비스 연동 (Calendar, Notion, Slack) | ✅ 완료 |

---

## 2. 시스템 아키텍처

### 2.1 전체 시스템 구조

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              JAVIS System                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         사용자 인터페이스                             │   │
│  │   ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐            │   │
│  │   │ Web UI  │   │   CLI   │   │  Voice  │   │   API   │            │   │
│  │   └────┬────┘   └────┬────┘   └────┬────┘   └────┬────┘            │   │
│  └────────┼─────────────┼─────────────┼─────────────┼──────────────────┘   │
│           │             │             │             │                       │
│           └─────────────┴──────┬──────┴─────────────┘                       │
│                                ▼                                            │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      FastAPI Server (api.py)                         │   │
│  │   /chat  /feedback  /upload  /voice  /conversations  /health        │   │
│  └────────────────────────────────┬────────────────────────────────────┘   │
│                                   ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                       JAVIS Core Engine                              │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐               │   │
│  │  │ ChatEngine   │  │ ToolExecutor │  │ContextManager│               │   │
│  │  │              │  │              │  │  Registry    │               │   │
│  │  │ - chat()     │  │ - execute()  │  │              │               │   │
│  │  │ - tools      │  │ - parallel() │  │ - Memory     │               │   │
│  │  │ - context    │  │ - timeout    │  │ - RAG        │               │   │
│  │  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘               │   │
│  └─────────┼─────────────────┼─────────────────┼───────────────────────┘   │
│            │                 │                 │                            │
│            ▼                 ▼                 ▼                            │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐               │
│  │   Tool System   │ │  Memory System  │ │    RAG System   │               │
│  │  ┌───────────┐  │ │  ┌───────────┐  │ │  ┌───────────┐  │               │
│  │  │ Registry  │  │ │  │ Manager   │  │ │  │ Manager   │  │               │
│  │  │ Web Tools │  │ │  │ Store     │  │ │  │ Store     │  │               │
│  │  │ File Tools│  │ │  │ Embeddings│  │ │  │ Chunker   │  │               │
│  │  │ System    │  │ │  │ Summarizer│  │ │  │ Loaders   │  │               │
│  │  │ Calendar  │  │ │  └───────────┘  │ │  └───────────┘  │               │
│  │  │ Notion    │  │ │        │        │ │        │        │               │
│  │  │ Slack     │  │ │        ▼        │ │        ▼        │               │
│  │  └───────────┘  │ │   ChromaDB      │ │   ChromaDB      │               │
│  └─────────────────┘ └─────────────────┘ └─────────────────┘               │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         Model Layer                                  │   │
│  │  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐            │   │
│  │  │  Groq Client  │  │ Local Client  │  │ Modal Client  │            │   │
│  │  │ (Llama 3.3)   │  │ (Fine-tuned)  │  │  (Training)   │            │   │
│  │  └───────────────┘  └───────────────┘  └───────────────┘            │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      Training Pipeline                               │   │
│  │   데이터 수집 → 품질 평가 → 파인튜닝 → 검증 → 버전 관리 → 배포        │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 레이어 구조

```
┌─────────────────────────────────────┐
│     Presentation Layer              │  ← Web UI, CLI, Voice
├─────────────────────────────────────┤
│     Interface Layer                 │  ← FastAPI endpoints
├─────────────────────────────────────┤
│     Core Layer                      │  ← ChatEngine, ToolExecutor
├─────────────────────────────────────┤
│     Feature Layer                   │  ← Tools, Memory, RAG, Voice
├─────────────────────────────────────┤
│     Integration Layer               │  ← External APIs (Calendar, Notion, Slack)
├─────────────────────────────────────┤
│     Model Layer                     │  ← LLM Clients (Groq, Local, Modal)
├─────────────────────────────────────┤
│     Infrastructure Layer            │  ← Config, Logging, Storage
└─────────────────────────────────────┘
```

---

## 3. 디렉토리 구조

```
JAVIS/
├── javis/                              # 메인 애플리케이션 패키지
│   ├── __init__.py
│   ├── __main__.py                     # CLI 진입점
│   │
│   ├── core/                           # 핵심 엔진
│   │   ├── __init__.py
│   │   └── engine.py                   # ChatEngine, ContextManagerRegistry
│   │
│   ├── interfaces/                     # 인터페이스 계층
│   │   ├── __init__.py
│   │   ├── api.py                      # FastAPI 서버
│   │   └── cli.py                      # CLI 인터페이스
│   │
│   ├── models/                         # 모델 클라이언트
│   │   ├── __init__.py
│   │   ├── client.py                   # 공통 인터페이스 (Message, ChatResponse)
│   │   ├── groq_client.py              # Groq API 클라이언트
│   │   ├── local_client.py             # 로컬 모델 클라이언트
│   │   ├── modal_client.py             # Modal 클라이언트
│   │   └── vision_client.py            # 비전 모델 클라이언트
│   │
│   ├── tools/                          # 도구 시스템 (Phase 3)
│   │   ├── __init__.py
│   │   ├── base.py                     # BaseTool, ToolResult
│   │   ├── registry.py                 # ToolRegistry (싱글톤)
│   │   ├── executor.py                 # ToolExecutor (병렬 실행)
│   │   ├── web_tools.py                # 웹 검색, URL 가져오기
│   │   ├── file_tools.py               # 파일 읽기/쓰기
│   │   ├── system_tools.py             # 시스템 시간, 명령 실행
│   │   ├── calendar_tools.py           # Google Calendar 도구
│   │   ├── notion_tools.py             # Notion 도구
│   │   └── slack_tools.py              # Slack 도구
│   │
│   ├── memory/                         # 장기 메모리 시스템 (Phase 4)
│   │   ├── __init__.py
│   │   ├── manager.py                  # MemoryManager
│   │   ├── store.py                    # VectorStore (ChromaDB)
│   │   ├── embeddings.py               # EmbeddingService
│   │   ├── summarizer.py               # ConversationSummarizer
│   │   └── models.py                   # MemoryEntry, SearchResult
│   │
│   ├── rag/                            # RAG 시스템 (Phase 4)
│   │   ├── __init__.py
│   │   ├── manager.py                  # RAGManager
│   │   ├── store.py                    # DocumentStore
│   │   ├── chunker.py                  # TextChunker, CodeChunker
│   │   ├── models.py                   # Document, DocumentChunk
│   │   └── loaders/                    # 문서 로더
│   │       ├── __init__.py
│   │       ├── file_loader.py          # 파일 로더
│   │       ├── code_loader.py          # 코드 로더
│   │       └── web_loader.py           # 웹 로더
│   │
│   ├── voice/                          # 음성 시스템 (Phase 5)
│   │   ├── __init__.py
│   │   ├── session.py                  # 음성 세션 관리
│   │   ├── stt/                        # Speech-to-Text
│   │   │   ├── __init__.py
│   │   │   └── groq_whisper.py         # Groq Whisper STT
│   │   └── tts/                        # Text-to-Speech
│   │       ├── __init__.py
│   │       └── edge_tts_provider.py    # Edge TTS
│   │
│   ├── training/                       # 학습 파이프라인 (Phase 5)
│   │   ├── __init__.py
│   │   ├── manage.py                   # CLI 관리 도구
│   │   ├── pipeline.py                 # TrainingPipeline
│   │   ├── scheduler.py                # TrainingScheduler (APScheduler)
│   │   ├── remote.py                   # RemoteTrainer (Modal)
│   │   ├── version_manager.py          # VersionManager
│   │   ├── data_quality.py             # DataQualityScorer
│   │   ├── validation.py               # ModelValidator
│   │   ├── ab_testing.py               # ABTestManager
│   │   ├── feedback_store.py           # FeedbackStore
│   │   ├── notifications.py            # NotificationService (Discord)
│   │   ├── dpo_trainer.py              # DPO 학습기
│   │   ├── preference_data.py          # 선호도 데이터 생성
│   │   └── metrics.py                  # 성능 메트릭
│   │
│   ├── workflows/                      # 워크플로우 자동화 (Phase 6)
│   │   ├── __init__.py
│   │   ├── models.py                   # Workflow 모델
│   │   ├── executor.py                 # WorkflowExecutor
│   │   ├── scheduler.py                # WorkflowScheduler
│   │   └── triggers/                   # 트리거
│   │       └── __init__.py
│   │
│   ├── integrations/                   # 외부 서비스 연동 (Phase 7)
│   │   ├── __init__.py
│   │   ├── google_calendar.py          # Google Calendar API
│   │   ├── notion_client.py            # Notion API
│   │   └── slack_client.py             # Slack API
│   │
│   ├── data/                           # 데이터 처리
│   │   ├── __init__.py
│   │   └── conversation_logger.py      # ConversationLogger
│   │
│   └── utils/                          # 유틸리티
│       ├── __init__.py
│       ├── config.py                   # Config (Pydantic)
│       ├── constants.py                # 상수 정의
│       └── logging.py                  # 로깅 설정
│
├── tests/                              # 테스트
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_memory.py
│   ├── test_rag.py
│   ├── test_tools.py
│   ├── test_training.py
│   └── test_file_tools.py
│
├── static/                             # 웹 UI
│   ├── index.html                      # 메인 페이지
│   ├── styles.css                      # 스타일
│   └── app.js                          # JavaScript
│
├── configs/                            # 설정 파일
│   ├── config.yaml                     # 메인 설정
│   ├── validation_prompts.json         # 모델 검증 프롬프트
│   └── workflows/                      # 워크플로우 정의
│
├── data/                               # 데이터 저장소
│   ├── conversations/                  # 대화 로그 (JSON)
│   │   └── YYYY-MM/
│   ├── training/
│   │   └── exported/                   # 학습 데이터 (JSONL)
│   ├── vectors/
│   │   ├── memory/                     # 메모리 벡터 DB
│   │   └── documents/                  # RAG 문서 벡터 DB
│   └── uploads/                        # 업로드 파일
│
├── models/                             # 파인튜닝 모델
│   ├── .active_version                 # 활성 버전 포인터
│   ├── v1.0/
│   │   ├── adapter/                    # LoRA 어댑터
│   │   └── metadata.json               # 버전 메타데이터
│   └── v20251230/
│       ├── adapter/
│       └── metadata.json
│
├── notebooks/                          # Jupyter 노트북
│   └── finetune_colab.ipynb            # Colab 학습 노트북
│
├── docs/                               # 문서
│   ├── 01-PROJECT-OVERVIEW.md
│   ├── 02-ROADMAP.md
│   ├── 03-MODEL-VERSIONING.md
│   ├── 04-TECH-STACK.md
│   ├── 05-DEPLOYMENT.md
│   ├── ARCHITECTURE.md                 # 이 문서
│   ├── USER_GUIDE.md
│   └── VERSIONING.md
│
├── .github/
│   └── workflows/
│       └── ci.yml                      # CI/CD 파이프라인
│
├── pyproject.toml                      # 프로젝트 설정 (PEP 517)
├── requirements.txt                    # 의존성
├── requirements-training.txt           # 학습용 의존성
├── Dockerfile                          # 컨테이너 빌드
├── Procfile                            # Heroku/Koyeb 배포
├── .env.example                        # 환경변수 예시
└── README.md                           # 프로젝트 소개
```

---

## 4. 핵심 모듈 상세

### 4.1 ChatEngine (`javis/core/engine.py`)

채팅 요청을 처리하는 핵심 엔진입니다.

```python
class ChatEngine:
    """
    도구 호출을 지원하는 채팅 엔진

    Features:
    - Multi-turn 도구 호출 (자동 결과 주입)
    - 장기 메모리 통합 (과거 대화 컨텍스트)
    - RAG 문서 검색 (지식 증강)
    - 설정 가능한 타임아웃 및 반복 제한
    """
```

**주요 메서드:**

| 메서드 | 설명 |
|--------|------|
| `chat_with_tools()` | 도구 호출 지원 채팅 (메인) |
| `chat_simple()` | 도구 없이 단순 채팅 |
| `_enhance_with_context()` | Memory/RAG 컨텍스트 강화 |
| `store_conversation_to_memory()` | 대화 메모리 저장 |

**데이터 흐름:**

```
User Message
      ↓
┌─────────────────────────────────────┐
│  1. Tool Schema 준비                │
│     registry.get_openai_tools()     │
├─────────────────────────────────────┤
│  2. Context Enhancement             │
│     ├─ MemoryManager.search()       │
│     └─ RAGManager.search()          │
├─────────────────────────────────────┤
│  3. Tool Loop (max_iterations)      │
│     ┌────────────────────────┐      │
│     │ Groq API Call          │      │
│     │      ↓                 │      │
│     │ tool_calls?            │      │
│     │   ├─ Yes → 도구 실행   │      │
│     │   └─ No  → 응답 반환   │      │
│     └────────────────────────┘      │
└─────────────────────────────────────┘
      ↓
ChatResponse
```

### 4.2 ToolExecutor (`javis/tools/executor.py`)

도구를 병렬로 실행하는 실행기입니다.

```python
class ToolExecutor:
    """
    도구 실행기 - 타임아웃 및 병렬 실행 지원

    Features:
    - asyncio.wait_for()로 개별 타임아웃
    - Semaphore로 동시 실행 수 제한
    - 에러 격리 (개별 실패가 전체에 영향 없음)
    """
```

**병렬 실행 패턴:**

```
tool_calls = [call_1, call_2, call_3, call_4, call_5, call_6]
                │       │       │       │       │       │
                ▼       ▼       ▼       ▼       ▼       │
         ┌─────────────────────────────────────────┐   │
         │     Semaphore (max_parallel=5)         │   │
         │  ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐        │   │
         │  │ 1 │ │ 2 │ │ 3 │ │ 4 │ │ 5 │  실행중 │   │
         │  └───┘ └───┘ └───┘ └───┘ └───┘        │   │
         └─────────────────────────────────────────┘   │
                                                       ▼
                                               대기 중: 6
```

### 4.3 MemoryManager (`javis/memory/manager.py`)

장기 대화 기억을 관리합니다.

```python
class MemoryManager:
    """
    장기 메모리 관리자

    Features:
    - ChromaDB 기반 벡터 저장
    - sentence-transformers 임베딩
    - 대화 요약 및 저장
    - 관련 컨텍스트 검색
    """
```

**메모리 구조:**

```
Memory Entry
├── session_id: str         # 세션 ID
├── summary: str            # 대화 요약
├── embedding: vector       # 벡터 임베딩
├── turns: list             # 원본 대화
├── metadata: dict          # 메타데이터
└── created_at: datetime    # 생성 시간
```

### 4.4 RAGManager (`javis/rag/manager.py`)

문서 검색 및 지식 증강을 담당합니다.

```python
class RAGManager:
    """
    RAG (Retrieval-Augmented Generation) 관리자

    Features:
    - 다양한 문서 로더 (File, Code, Web)
    - 청킹 전략 (Text, Code)
    - 벡터 검색 및 재순위화
    """
```

**문서 처리 파이프라인:**

```
Document
    ↓
┌─────────────┐
│   Loader    │  ← FileLoader, CodeLoader, WebLoader
└─────────────┘
    ↓
┌─────────────┐
│   Chunker   │  ← TextChunker (500자), CodeChunker (함수 단위)
└─────────────┘
    ↓
┌─────────────┐
│  Embedding  │  ← sentence-transformers
└─────────────┘
    ↓
┌─────────────┐
│  ChromaDB   │  ← 벡터 저장
└─────────────┘
```

### 4.5 TrainingPipeline (`javis/training/pipeline.py`)

자동 학습 파이프라인을 관리합니다.

```python
class TrainingPipeline:
    """
    End-to-end 학습 파이프라인

    Phases:
    1. 조건 검사 (데이터 충분성)
    2. 데이터 내보내기 (품질 필터링)
    3. 원격 학습 (Modal GPU)
    4. 모델 검증
    5. 버전 생성
    6. 배포 또는 A/B 테스트
    7. 알림 발송
    """
```

---

## 5. 데이터 흐름

### 5.1 채팅 요청 흐름

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           채팅 요청 흐름                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  User Input (Web/API)                                                   │
│       │                                                                 │
│       ▼                                                                 │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  FastAPI /chat endpoint                                          │   │
│  │  - 세션 관리                                                      │   │
│  │  - 시스템 프롬프트 주입                                           │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│       │                                                                 │
│       ▼                                                                 │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  ChatEngine.chat_with_tools()                                    │   │
│  │  ┌─────────────────────────────────────────────────────────┐    │   │
│  │  │  Context Enhancement                                     │    │   │
│  │  │  ├─ MemoryManager.get_relevant_context()                │    │   │
│  │  │  └─ RAGManager.get_context_for_query()                  │    │   │
│  │  └─────────────────────────────────────────────────────────┘    │   │
│  │       │                                                          │   │
│  │       ▼                                                          │   │
│  │  ┌─────────────────────────────────────────────────────────┐    │   │
│  │  │  Groq API Call (with tools)                              │    │   │
│  │  └─────────────────────────────────────────────────────────┘    │   │
│  │       │                                                          │   │
│  │       ▼                                                          │   │
│  │  ┌─────────────────────────────────────────────────────────┐    │   │
│  │  │  Tool Calls? ──Yes──► ToolExecutor.execute_parallel()   │    │   │
│  │  │       │                      │                           │    │   │
│  │  │      No                      ▼                           │    │   │
│  │  │       │              Tool Results                        │    │   │
│  │  │       │                      │                           │    │   │
│  │  │       ▼                      ▼                           │    │   │
│  │  │  Final Response ◄─────── Loop Back                       │    │   │
│  │  └─────────────────────────────────────────────────────────┘    │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│       │                                                                 │
│       ▼                                                                 │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  ConversationLogger.log_turn()                                   │   │
│  │  - 대화 기록 저장                                                 │   │
│  │  - 학습 데이터 축적                                               │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│       │                                                                 │
│       ▼                                                                 │
│  Response to User                                                       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 5.2 학습 데이터 흐름

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          학습 데이터 흐름                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  대화 로그 (data/conversations/)                                        │
│       │                                                                 │
│       ▼                                                                 │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  DataQualityScorer                                               │   │
│  │  - 피드백 점수 (40%)                                              │   │
│  │  - 길이 점수 (20%)                                                │   │
│  │  - 일관성 점수 (20%)                                              │   │
│  │  - 최신성 점수 (20%)                                              │   │
│  │  - 최소 품질 점수: 3.5                                            │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│       │                                                                 │
│       ▼                                                                 │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Export to JSONL                                                 │   │
│  │  - 나쁜 피드백 제외                                               │   │
│  │  - 90일 이상 오래된 데이터 제외                                   │   │
│  │  - 품질 점수 미달 제외                                            │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│       │                                                                 │
│       ▼                                                                 │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Modal GPU Training                                              │   │
│  │  - Qwen2.5-7B-Instruct                                           │   │
│  │  - QLoRA (r=64, alpha=16)                                        │   │
│  │  - SFT 또는 DPO 학습                                              │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│       │                                                                 │
│       ▼                                                                 │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  ModelValidator                                                  │   │
│  │  - 파일 무결성 검사                                               │   │
│  │  - 추론 테스트 (선택)                                             │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│       │                                                                 │
│       ▼                                                                 │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  VersionManager                                                  │   │
│  │  - 버전 생성 (v20250115_120000)                                   │   │
│  │  - 메타데이터 저장                                                │   │
│  │  - 활성 버전 전환                                                 │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│       │                                                                 │
│       ▼                                                                 │
│  Discord 알림                                                           │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 6. 학습 파이프라인

### 6.1 자동 학습 스케줄

```yaml
# config.yaml
training:
  schedule:
    enabled: true
    cron: "0 0 * * 0"      # 매주 일요일 자정
    timezone: "Asia/Seoul"
```

### 6.2 학습 조건

| 조건 | 기본값 | 설명 |
|------|--------|------|
| `min_conversations` | 2 | 최소 대화 수 |
| `min_good_feedback` | 0 | 최소 좋은 피드백 수 |
| `exclude_bad_feedback` | true | 나쁜 피드백 제외 |
| `max_age_days` | 90 | 데이터 유효 기간 |

### 6.3 학습 설정

| 설정 | 기본값 | 설명 |
|------|--------|------|
| `base_model` | Qwen/Qwen2.5-7B-Instruct | 베이스 모델 |
| `epochs` | 1 | 에폭 수 |
| `batch_size` | 2 | 배치 크기 (A10G용) |
| `learning_rate` | 0.0002 | 학습률 |
| `lora_r` | 64 | LoRA rank |
| `lora_alpha` | 16 | LoRA alpha |
| `max_seq_length` | 1024 | 최대 시퀀스 길이 |

### 6.4 DPO 학습

```python
# Direct Preference Optimization
training_config = {
    "method": "dpo",
    "beta": 0.1,                    # DPO 강도
    "loss_type": "sigmoid",         # 손실 함수 유형
    "max_prompt_length": 512,       # 프롬프트 최대 길이
    "max_response_length": 512,     # 응답 최대 길이
}
```

### 6.5 버전 관리

```
models/
├── .active_version              # "v20251230"
├── v1.0/
│   ├── adapter/
│   │   ├── adapter_config.json
│   │   ├── adapter_model.safetensors
│   │   └── README.md
│   └── metadata.json
│       {
│         "version": "v1.0",
│         "created_at": "2025-12-19T...",
│         "base_model": "Qwen/Qwen2.5-7B-Instruct",
│         "dataset_size": 100,
│         "status": "ready"
│       }
└── v20251230/
    ├── adapter/
    └── metadata.json
```

**버전 상태:**
- `ready`: 사용 가능
- `active`: 현재 활성
- `failed`: 검증 실패

---

## 7. 기술 스택

### 7.1 백엔드

| 기술 | 버전 | 용도 |
|------|------|------|
| Python | 3.11+ | 런타임 |
| FastAPI | 0.104+ | 웹 프레임워크 |
| Uvicorn | 0.24+ | ASGI 서버 |
| Pydantic | 2.5+ | 데이터 검증 |
| httpx | 0.25+ | 비동기 HTTP 클라이언트 |

### 7.2 LLM & AI

| 기술 | 용도 |
|------|------|
| Groq API | 추론 (Llama 3.3 70B) |
| Qwen2.5-7B-Instruct | 파인튜닝 베이스 모델 |
| QLoRA | 효율적 파인튜닝 |
| Modal | 서버리스 GPU 학습 |
| sentence-transformers | 임베딩 |

### 7.3 데이터 저장

| 기술 | 용도 |
|------|------|
| ChromaDB | 벡터 데이터베이스 |
| JSON/JSONL | 대화 로그, 학습 데이터 |
| YAML | 설정 파일 |

### 7.4 음성

| 기술 | 용도 |
|------|------|
| Groq Whisper | Speech-to-Text |
| Edge TTS | Text-to-Speech |

### 7.5 외부 연동

| 서비스 | 용도 |
|--------|------|
| Google Calendar API | 일정 관리 |
| Notion API | 데이터베이스 연동 |
| Slack API | 메시징 |
| Discord Webhook | 알림 |

### 7.6 개발 도구

| 도구 | 용도 |
|------|------|
| Ruff | 린팅 |
| MyPy | 타입 체크 |
| Pytest | 테스트 |
| GitHub Actions | CI/CD |

---

## 8. 설정 및 환경변수

### 8.1 환경변수 (.env)

```bash
# LLM API
GROQ_API_KEY=gsk_xxxxxxxxxxxxxx

# 학습 인프라
HUGGINGFACE_TOKEN=hf_xxxxxxxxxxxxxx
MODAL_TOKEN_ID=ak-xxxxxxxxxxxxxx
MODAL_TOKEN_SECRET=as-xxxxxxxxxxxxxx

# 알림
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...

# 외부 서비스
GOOGLE_CREDENTIALS_PATH=./credentials.json
NOTION_API_KEY=secret_xxxxxxxxxxxxxx
SLACK_BOT_TOKEN=xoxb-xxxxxxxxxxxxxx

# 설정
LOG_LEVEL=INFO
CORS_ORIGINS=http://localhost:3000,http://localhost:8000
```

### 8.2 설정 파일 (config.yaml)

```yaml
app:
  name: JAVIS
  version: "0.1.0"
  description: "Personal AI Assistant"

model:
  provider: runpod
  base_model: "Qwen/Qwen2.5-7B-Instruct"
  generation:
    max_tokens: 2048
    temperature: 0.7
    top_p: 0.9

conversation:
  system_prompt: |
    너는 JAVIS, 개발자를 위한 개인 AI 비서다.
  max_history: 20
  max_context_tokens: 8000

tools:
  enabled: true
  available:
    - file_tools
    - web_tools
    - system_tools
    - calendar_tools
    - notion_tools
    - slack_tools

memory:
  long_term:
    enabled: true
    db_path: "./data/vectors/memory"
    collection_name: "javis_memory"
    embedding_model: "sentence-transformers/all-MiniLM-L6-v2"

rag:
  enabled: true
  db_path: "./data/vectors/documents"
  chunk_size: 500
  top_k: 5

training:
  schedule:
    enabled: true
    cron: "0 0 * * 0"
    timezone: "Asia/Seoul"
  provider: "modal"
  deployment:
    auto_deploy: true
    validation_required: true
    keep_versions: 5

voice:
  enabled: true
  language: "ko-KR"
  stt:
    provider: "groq_whisper"
  tts:
    provider: "edge_tts"
    edge_tts:
      voice: "ko-KR-SunHiNeural"
```

---

## 9. API 명세

### 9.1 채팅 API

#### POST /chat

```json
// Request
{
  "message": "안녕하세요",
  "session_id": "optional-session-id"
}

// Response
{
  "response": "안녕하세요! 무엇을 도와드릴까요?",
  "session_id": "session_abc123"
}
```

#### POST /feedback

```json
// Request
{
  "session_id": "session_abc123",
  "feedback": "good"  // "good" | "bad"
}

// Response
{
  "status": "success"
}
```

### 9.2 파일 API

#### POST /upload

```
Content-Type: multipart/form-data
file: <binary>

// Response
{
  "filename": "document.pdf",
  "size": 12345,
  "status": "uploaded"
}
```

### 9.3 음성 API

#### POST /voice/transcribe

```
Content-Type: multipart/form-data
audio: <binary>

// Response
{
  "text": "음성 인식 결과"
}
```

#### POST /voice/synthesize

```json
// Request
{
  "text": "읽을 텍스트"
}

// Response (audio/mpeg)
<binary>
```

### 9.4 관리 API

#### GET /health

```json
{
  "status": "healthy",
  "version": "0.1.0"
}
```

#### GET /conversations

```json
{
  "conversations": [
    {
      "session_id": "session_abc123",
      "started_at": "2025-01-15T10:00:00",
      "turns": 5,
      "feedback": "good"
    }
  ]
}
```

---

## 10. 확장 가이드

### 10.1 새 도구 추가

```python
# javis/tools/my_tools.py
from javis.tools.base import BaseTool, ToolResult

class MyCustomTool(BaseTool):
    name = "my_custom_tool"
    description = "도구 설명"
    parameters = {
        "type": "object",
        "properties": {
            "param1": {"type": "string", "description": "파라미터 설명"}
        },
        "required": ["param1"]
    }

    async def execute(self, param1: str) -> ToolResult:
        # 도구 로직 구현
        result = do_something(param1)
        return ToolResult(success=True, output=result)

# 레지스트리에 등록
from javis.tools.registry import get_registry
registry = get_registry()
registry.register(MyCustomTool())
```

### 10.2 새 문서 로더 추가

```python
# javis/rag/loaders/my_loader.py
from javis.rag.models import Document

class MyLoader:
    """새 문서 형식 로더"""

    def load(self, path: str) -> list[Document]:
        # 문서 로드 로직
        content = read_my_format(path)
        return [Document(content=content, metadata={"source": path})]

# 레지스트리에 등록
from javis.rag.loaders import LoaderRegistry
LoaderRegistry.register(".myext", MyLoader())
```

### 10.3 새 외부 서비스 연동

```python
# javis/integrations/my_service.py
class MyServiceClient:
    def __init__(self, api_key: str):
        self.api_key = api_key

    async def get_data(self) -> dict:
        # API 호출 로직
        pass

# 도구로 연결
# javis/tools/my_service_tools.py
class MyServiceTool(BaseTool):
    name = "my_service_action"
    # ...
```

### 10.4 커스텀 학습 전략

```python
# javis/training/my_trainer.py
from javis.training.remote import RemoteTrainer

class MyTrainer(RemoteTrainer):
    def train(self, data_path, config):
        # 커스텀 학습 로직
        pass
```

---

## 문서 이력

| 버전 | 날짜 | 변경 내용 |
|------|------|-----------|
| 1.0 | 2025-01-16 | 초기 작성 - 전체 아키텍처 문서화 |

---

*이 문서는 JAVIS 프로젝트의 전체 아키텍처와 설계를 설명합니다. 최신 코드와 동기화되도록 주기적으로 업데이트됩니다.*
