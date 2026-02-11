# JAVIS 로컬 무료 실행 TODO

## 핵심: Groq API 무료 tier로 추론, 나머지는 모두 로컬/비활성화

### 1. [x] Python 가상환경 생성 & 의존성 설치
- Python 3.12.10 설치 완료 (winget)
- `.venv` 생성 후 `requirements.txt` 설치 완료
```bash
cd C:\down\08.JAVIS\JAVIS
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. [x] `.env` 파일 생성
- `.env.example` 기반 최소 설정 완료
```env
JAVIS_STORAGE_MODE=local
JAVIS_ENV=development
GROQ_API_KEY=<Groq에서 무료 발급: https://console.groq.com/keys>
```

### 3. [x] `configs/config.yaml` 수정 — 유료 서비스 비활성화
- `model.provider: groq` (RunPod → Groq 변경)
- `training.schedule.enabled: false` (Modal GPU 학습 비활성화)
- `memory.long_term.enabled: false` (ChromaDB 의존성 줄이기)
- `rag.enabled: false`
- `voice.enabled: false`
- 외부 연동 모두 `enabled: false` (이미 기본값)

### 4. [x] `javis/interfaces/cli.py` — RunPod 체크 → Groq 체크로 변경
- chat 커맨드: RunPod 자격증명 체크 → Groq API 키 체크로 변경
- config 커맨드: RunPod 상태 표시 → Groq API Key 상태 표시로 변경

### 5. [x] 서버 실행 & 검증
```bash
uvicorn javis.interfaces.api:app --reload --host 127.0.0.1 --port 8000
```
- 서버 정상 기동 확인 완료
- `http://localhost:8000` → 웹 UI 응답 확인
- `http://localhost:8000/health` → `{"status":"healthy"}` 확인
- ⚠️ 채팅 기능은 `.env`에 실제 Groq API 키 입력 후 사용 가능

## 수정 대상 파일
| 파일 | 작업 |
|------|------|
| `.env` (신규) | `.env.example` 기반 최소 설정 |
| `configs/config.yaml` | 유료 기능 비활성화 |
| `javis/interfaces/cli.py` | RunPod 체크 → Groq 체크 |
