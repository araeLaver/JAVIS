# JAVIS 클라우드 배포 가이드

이 가이드는 JAVIS를 클라우드에 배포하는 방법을 설명합니다.

---

## 개요

| 서비스 | 용도 | 비용 |
|--------|------|------|
| Supabase | 데이터베이스 (PostgreSQL) | 무료 |
| HuggingFace Hub | 모델 저장소 | 무료 |
| Koyeb | API 서버 호스팅 | 무료 |
| Modal.com | GPU 학습 | ~$5-10/월 |

---

## 1단계: Supabase 설정

### 1.1 프로젝트 생성

1. [Supabase](https://supabase.com) 접속 → 회원가입/로그인
2. "New Project" 클릭
3. 프로젝트 이름: `javis` (원하는 이름)
4. 데이터베이스 비밀번호 설정 (저장해두세요!)
5. Region: `Northeast Asia (Seoul)` 선택 → "Create new project"

### 1.2 테이블 생성

1. 프로젝트 대시보드 → 좌측 메뉴 "SQL Editor" 클릭
2. "New Query" 클릭
3. `scripts/setup_supabase.sql` 파일 내용 전체를 복사하여 붙여넣기
4. "Run" 버튼 클릭
5. 성공 메시지 확인

### 1.3 API 키 복사

1. 좌측 메뉴 → "Settings" → "API"
2. 다음 값들을 복사해서 `.env` 파일에 저장:

```env
SUPABASE_URL=https://xxxxx.supabase.co  # Project URL
SUPABASE_SERVICE_ROLE_KEY=eyJ...        # service_role (secret)
```

⚠️ **주의**: `service_role` 키는 절대 공개하지 마세요!

---

## 2단계: HuggingFace Hub 설정

### 2.1 토큰 생성

1. [HuggingFace](https://huggingface.co) 접속 → 로그인
2. 우측 상단 프로필 → "Settings"
3. 좌측 메뉴 → "Access Tokens"
4. "New token" 클릭
   - Name: `javis`
   - Type: `Write` (쓰기 권한 필요)
5. "Generate a token" → 토큰 복사

### 2.2 저장소 생성

1. 상단 메뉴 → "New" → "Model"
2. Model name: `javis-adapters`
3. **Private** 선택 (중요!)
4. "Create model"

### 2.3 환경변수 설정

```env
HF_TOKEN=hf_xxxxx
HF_REPO_ID=your-username/javis-adapters
```

---

## 3단계: 로컬 테스트

배포 전에 로컬에서 클라우드 모드가 잘 동작하는지 확인합니다.

### 3.1 환경변수 설정

`.env` 파일을 생성하고 다음 내용을 입력:

```env
# 클라우드 모드 활성화
JAVIS_STORAGE_MODE=cloud

# Supabase
SUPABASE_URL=https://xxxxx.supabase.co
SUPABASE_SERVICE_ROLE_KEY=eyJ...

# HuggingFace
HF_TOKEN=hf_xxxxx
HF_REPO_ID=your-username/javis-adapters

# 기타
JAVIS_ENV=development
JAVIS_LOG_LEVEL=INFO
```

### 3.2 기존 데이터 마이그레이션 (선택사항)

기존에 로컬에 저장된 데이터가 있다면 클라우드로 마이그레이션:

```bash
# 먼저 dry-run으로 확인
python scripts/migrate_to_cloud.py --dry-run

# 실제 마이그레이션
python scripts/migrate_to_cloud.py
```

### 3.3 서버 실행 테스트

```bash
python -m uvicorn javis.api.main:app --reload
```

API가 정상 작동하면 배포 준비 완료!

---

## 4단계: Koyeb 배포 (API 서버)

### 4.1 Dockerfile 확인

프로젝트 루트에 `Dockerfile`이 있는지 확인:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "javis.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 4.2 GitHub 연동

1. 프로젝트를 GitHub에 푸시
2. `.env` 파일은 `.gitignore`에 추가되어 있는지 확인!

### 4.3 Koyeb 설정

1. [Koyeb](https://www.koyeb.com) 접속 → 회원가입/로그인
2. "Create App" 클릭
3. "GitHub" 선택 → 저장소 연결
4. 설정:
   - Name: `javis`
   - Builder: `Dockerfile`
   - Region: `Frankfurt` 또는 가까운 곳
   - Instance: `nano` (무료)

### 4.4 환경변수 설정

Koyeb 대시보드 → App → Settings → Environment variables:

```
JAVIS_STORAGE_MODE=cloud
SUPABASE_URL=https://xxxxx.supabase.co
SUPABASE_SERVICE_ROLE_KEY=eyJ...
HF_TOKEN=hf_xxxxx
HF_REPO_ID=your-username/javis-adapters
JAVIS_ENV=production
```

### 4.5 배포

"Deploy" 클릭 → 배포 완료 후 URL 확인:
`https://javis-your-username.koyeb.app`

---

## 5단계: Modal.com 설정 (GPU 학습)

### 5.1 계정 설정

1. [Modal](https://modal.com) 접속 → 회원가입
2. 터미널에서:

```bash
pip install modal
modal setup  # 브라우저에서 인증
```

### 5.2 환경변수 설정

Modal 대시보드 → Settings → Secrets에서 환경변수 추가:

```bash
modal secret create javis-secrets \
  SUPABASE_URL=https://xxxxx.supabase.co \
  SUPABASE_SERVICE_ROLE_KEY=eyJ... \
  HF_TOKEN=hf_xxxxx \
  HF_REPO_ID=your-username/javis-adapters
```

### 5.3 학습 실행

```bash
# 수동 학습
modal run javis.training.remote:train

# 또는 스케줄러 배포 (주간 자동 학습)
modal deploy javis.training.scheduler
```

---

## 관리 및 모니터링

### Supabase 대시보드

- **Table Editor**: 데이터 확인/수정
- **SQL Editor**: 쿼리 실행
- **Logs**: 에러 로그 확인
- **Usage**: 사용량 모니터링 (무료 한도 확인)

### Koyeb 대시보드

- **Logs**: 실시간 로그 확인
- **Metrics**: CPU/메모리 사용량
- **Domains**: 커스텀 도메인 설정

### HuggingFace Hub

- 모델 버전 히스토리 확인
- 어댑터 다운로드/관리

---

## 문제 해결

### 연결 오류

```
Error: Could not connect to Supabase
```
→ `SUPABASE_URL`과 `SUPABASE_SERVICE_ROLE_KEY` 확인

### 권한 오류

```
Error: HuggingFace upload failed
```
→ `HF_TOKEN`이 Write 권한인지 확인

### 메모리 부족

Koyeb 무료 티어는 256MB RAM. 부족하면:
1. 유료 티어로 업그레이드
2. 또는 Railway/Render 등 다른 서비스 사용

---

## 비용 요약

| 서비스 | 무료 한도 | 예상 비용 |
|--------|----------|----------|
| Supabase | 500MB DB, 1GB 저장소 | $0 |
| HuggingFace | 무제한 (Private) | $0 |
| Koyeb | nano 인스턴스 | $0 |
| Modal | $30 크레딧/월 | ~$5-10 |
| **합계** | | **~$5-10/월** |

---

## 다음 단계

1. ✅ Supabase 설정
2. ✅ HuggingFace Hub 설정
3. ✅ 로컬 테스트
4. ✅ Koyeb 배포
5. ✅ Modal 설정

배포 완료 후:
- API 엔드포인트: `https://javis-xxx.koyeb.app`
- 대시보드: `https://javis-xxx.koyeb.app/dashboard` (있다면)
