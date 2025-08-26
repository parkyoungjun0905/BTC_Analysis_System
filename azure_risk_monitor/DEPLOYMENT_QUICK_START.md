# 🚀 19개 지표 시스템 Azure 배포 - 빠른 시작

## ⚡ 즉시 배포 명령어

### 1. Azure 리소스 생성
```bash
# 로그인
az login

# 리소스 그룹 생성
az group create --name btc-monitor-rg --location koreacentral

# 스토리지 계정 생성
az storage account create \
  --name btcmonitor19storage \
  --resource-group btc-monitor-rg \
  --location koreacentral \
  --sku Standard_LRS

# Function App 생성 (19개 지표 버전)
az functionapp create \
  --name btc-risk-monitor-19 \
  --resource-group btc-monitor-rg \
  --storage-account btcmonitor19storage \
  --consumption-plan-location koreacentral \
  --runtime python \
  --runtime-version 3.11 \
  --functions-version 4 \
  --os-type Linux
```

### 2. 환경변수 설정 (필수!)
```bash
# API 키들 설정
az functionapp config appsettings set \
  --name btc-risk-monitor-19 \
  --resource-group btc-monitor-rg \
  --settings \
    "CRYPTOQUANT_API_KEY=여기에-실제-키-입력" \
    "CLAUDE_API_KEY=여기에-실제-키-입력" \
    "TELEGRAM_BOT_TOKEN=여기에-봇-토큰-입력" \
    "TELEGRAM_CHAT_ID=여기에-채팅ID-입력"
```

### 3. 코드 배포
```bash
# 현재 디렉터리에서 배포
func azure functionapp publish btc-risk-monitor-19

# 실시간 로그 확인
func azure functionapp logstream btc-risk-monitor-19
```

---

## 🔑 필수 API 키 가져오기

### CryptoQuant API 키
1. [CryptoQuant 대시보드](https://cryptoquant.com/dashboard) 접속
2. API Keys 메뉴
3. 새 API 키 생성
4. 키 복사하여 위 명령어에 입력

### Claude API 키
1. [Anthropic Console](https://console.anthropic.com/) 접속
2. API Keys 생성
3. 키 복사

### 텔레그램 설정
1. @BotFather에게 `/newbot` 명령
2. 봇 이름 설정 → 토큰 받기
3. 봇을 그룹/채널에 추가
4. 채팅 ID 확인: `https://api.telegram.org/bot<TOKEN>/getUpdates`

---

## ✅ 배포 완료 후 테스트

### 상태 확인
```bash
# 헬스 체크
curl https://btc-risk-monitor-19.azurewebsites.net/api/health

# 수동 실행 테스트
curl https://btc-risk-monitor-19.azurewebsites.net/api/monitor
```

### 예상 응답 (헬스 체크)
```json
{
  "status": "healthy",
  "system": "19-Indicator Enhanced System",
  "environment": {
    "cryptoquant_api": "✅",
    "claude_api": "✅", 
    "telegram_bot": "✅",
    "telegram_chat": "✅"
  },
  "indicators": {
    "free_basic": 8,
    "free_advanced": 8,
    "cryptoquant": 3,
    "total": 19
  }
}
```

---

## ⚙️ 설정 확인

### 환경변수 상태 확인
```bash
az functionapp config appsettings list \
  --name btc-risk-monitor-19 \
  --resource-group btc-monitor-rg \
  --query "[].{name:name, value:value}" \
  --output table
```

### 실행 스케줄 확인
- **현재**: 30분마다 실행
- **수정하려면**: `function_app.py`의 `schedule="0 */30 * * * *"` 변경

---

## 🚨 트러블슈팅

### 1. 배포 실패 시
```bash
# Python 버전 확인 (3.11 필수)
python3 --version

# 의존성 재설치  
pip install -r requirements.txt --force-reinstall

# 다시 배포
func azure functionapp publish btc-risk-monitor-19 --force
```

### 2. API 키 오류 시
```bash
# 환경변수 다시 설정
az functionapp config appsettings set \
  --name btc-risk-monitor-19 \
  --resource-group btc-monitor-rg \
  --settings "CRYPTOQUANT_API_KEY=새로운-키"
```

### 3. 텔레그램 알림 안 올 때
- 봇이 그룹에 정확히 추가되었는지 확인
- 채팅 ID가 음수인지 확인 (그룹의 경우)
- 봇에 메시지 전송 권한이 있는지 확인

---

## 📊 모니터링 대시보드

### Azure Portal에서 확인
1. [Azure Portal](https://portal.azure.com) → Function Apps
2. `btc-risk-monitor-19` 클릭
3. 모니터링 → 로그 스트림

### 예상 로그 출력
```
2024-12-20 10:30:00 - INFO - 🚀 BTC 리스크 모니터링 시작
2024-12-20 10:30:02 - INFO - ✅ 19개 지표 수집 완료 (1.8초)
2024-12-20 10:30:03 - INFO - 예측: BULLISH 87%
2024-12-20 10:30:04 - INFO - 📨 알림 발송 완료
```

---

## 💰 비용 예상

### 월 예상 비용: **3-7만원**
- 실행 횟수: 1,440회/월 (30분마다)
- 평균 실행 시간: 3-5초
- 메모리 사용: 256MB
- 네트워크: API 호출 포함

### 비용 절약 팁
- 1시간마다로 변경: `schedule="0 0 * * * *"`
- 정확도 낮은 시간대 스킵 추가
- 불필요한 로그 줄이기

---

## 🎯 성공 확인 체크리스트

- [ ] Azure 리소스 생성 완료
- [ ] 모든 환경변수 설정됨 
- [ ] 코드 배포 성공
- [ ] 헬스 체크 통과 (모든 ✅)
- [ ] 첫 번째 알림 수신 확인
- [ ] 로그에서 19개 지표 수집 확인

**모든 체크리스트 완료 시 = 배포 성공!** 🎉

---

**즉시 배포하려면 위 명령어들을 순서대로 실행하세요!**