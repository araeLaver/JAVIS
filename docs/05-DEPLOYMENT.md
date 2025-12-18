# JAVIS 배포 가이드

## 전체 아키텍처 (어디서든 접속 가능)

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   [사용자]                                                       │
│   ├── PC 브라우저                                               │
│   ├── 모바일 브라우저                                            │
│   └── 어디서든 접속 가능                                         │
│           │                                                     │
│           │ HTTPS                                               │
│           ▼                                                     │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  Railway/Koyeb (JAVIS 서버)                             │   │
│   │  https://javis-xxxxx.railway.app                        │   │
│   │  ┌─────────────────────────────────────────────────┐    │   │
│   │  │  FastAPI Server                                 │    │   │
│   │  │  ├── 웹 인터페이스 제공                          │    │   │
│   │  │  ├── API 엔드포인트                              │    │   │
│   │  │  └── 세션 관리                                  │    │   │
│   │  └─────────────────────────────────────────────────┘    │   │
│   └─────────────────────────┬───────────────────────────────┘   │
│                             │                                   │
│                             │ API 호출                          │
│                             ▼                                   │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  RunPod Serverless (AI 모델)                            │   │
│   │  ┌─────────────────────────────────────────────────┐    │   │
│   │  │  Qwen2.5-7B + Custom LoRA                       │    │   │
│   │  └─────────────────────────────────────────────────┘    │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 방법 1: Railway 배포 (추천)

### 1단계: Railway 가입

1. https://railway.app 접속
2. GitHub 계정으로 로그인
3. 무료 $5/월 크레딧 제공

### 2단계: GitHub 저장소 생성

```bash
# 로컬에서 Git 초기화
cd C:\Develop\workspace\12.JAVIS
git init
git add .
git commit -m "Initial commit"

# GitHub에 저장소 생성 후 푸시
git remote add origin https://github.com/YOUR_USERNAME/javis.git
git branch -M main
git push -u origin main
```

### 3단계: Railway에서 배포

```
1. Railway 대시보드 → "New Project"
2. "Deploy from GitHub repo" 선택
3. javis 저장소 선택
4. 자동으로 Dockerfile 감지 → 빌드 시작
```

### 4단계: 환경변수 설정

```
Railway 대시보드 → 프로젝트 → Variables 탭

추가할 변수:
┌────────────────────────────────────────────┐
│ RUNPOD_API_KEY      = rp_xxxxxxxxxxxxx    │
│ RUNPOD_ENDPOINT_ID  = your-endpoint-id    │
└────────────────────────────────────────────┘
```

### 5단계: 도메인 설정

```
Railway 대시보드 → Settings → Domains

기본 제공: javis-xxxxx.railway.app
커스텀 도메인: javis.yourdomain.com (선택)
```

### 완료!

```
접속 URL: https://javis-xxxxx.railway.app

어디서든:
- PC 브라우저
- 모바일 브라우저
- 태블릿
```

---

## 방법 2: Koyeb 배포

### 1단계: Koyeb 가입

1. https://koyeb.com 접속
2. GitHub 계정으로 로그인
3. 무료 크레딧 제공

### 2단계: 앱 생성

```
1. Koyeb 대시보드 → "Create App"
2. "GitHub" 선택
3. javis 저장소 선택
4. 설정:
   - Builder: Dockerfile
   - Port: 8000
   - Region: 가까운 지역 선택
```

### 3단계: 환경변수 설정

```
Environment variables 섹션에서:

RUNPOD_API_KEY=rp_xxxxxxxxxxxxx
RUNPOD_ENDPOINT_ID=your-endpoint-id
```

### 4단계: 배포

```
"Deploy" 클릭 → 빌드 및 배포 진행
완료 후 URL 제공: https://javis-xxxxx.koyeb.app
```

---

## 방법 3: 로컬에서 웹 서버 실행 (테스트용)

```bash
# 가상환경 활성화
.venv\Scripts\activate

# 웹 서버 실행
uvicorn javis.interfaces.api:app --reload --port 8000

# 브라우저에서 접속
# http://localhost:8000
```

---

## 비용 비교

| 서비스 | 무료 티어 | 유료 |
|--------|----------|------|
| Railway | $5/월 크레딧 | $5~/월 |
| Koyeb | 제한적 무료 | $5~/월 |
| Render | 무료 (느림) | $7~/월 |
| RunPod (AI) | 없음 | $30-80/월 |

**총 예상 비용**: $35-90/월

---

## 보안 설정

### 1. 환경변수 보호

```bash
# .env 파일은 절대 Git에 커밋하지 않음
# .gitignore에 이미 포함됨

# 클라우드 서비스의 환경변수 기능 사용
Railway → Variables
Koyeb → Environment variables
```

### 2. API 인증 추가 (선택)

간단한 API 키 인증을 추가하려면:

```python
# javis/interfaces/api.py에 추가

from fastapi import Header, HTTPException

async def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != os.getenv("JAVIS_API_KEY"):
        raise HTTPException(status_code=401, detail="Invalid API key")
    return x_api_key

# 엔드포인트에 의존성 추가
@app.post("/api/chat", dependencies=[Depends(verify_api_key)])
```

### 3. CORS 제한 (프로덕션)

```python
# 특정 도메인만 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://javis.yourdomain.com"],
    ...
)
```

---

## 모니터링

### Railway 로그 확인

```
Railway 대시보드 → 프로젝트 → Deployments → View Logs
```

### 헬스체크

```bash
curl https://javis-xxxxx.railway.app/health
# {"status": "healthy"}
```

---

## 문제 해결

### 배포 실패 시

```
1. Railway 로그 확인
2. Dockerfile 빌드 에러 확인
3. 환경변수 설정 확인
```

### 응답이 느릴 때

```
원인: RunPod Serverless 콜드 스타트 (첫 요청 시)
해결:
- Min Workers를 1로 설정 (비용 증가)
- 또는 첫 요청 후 정상 속도
```

### API 오류 시

```
1. RunPod 엔드포인트 상태 확인
2. API 키 유효성 확인
3. 모델 배포 상태 확인
```

---

## 다음 단계

1. [ ] GitHub 저장소 생성
2. [ ] Railway/Koyeb 가입
3. [ ] 환경변수 설정
4. [ ] 배포 실행
5. [ ] 모바일에서 접속 테스트
