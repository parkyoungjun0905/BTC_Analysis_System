# ☁️ Azure BTC 모니터링 시스템 배포 가이드

## 🎯 배포 개요
**24시간 실시간 BTC 모니터링** + **위험 감지시 텔레그램 알림** 시스템을 Azure Function으로 배포

## 📋 필요한 준비사항

### 1️⃣ 텔레그램 봇 설정
```bash
# 1. @BotFather에게 메시지 보내기
/start
/newbot

# 2. 봇 이름 설정
BTC Monitor Bot

# 3. 봇 사용자명 설정  
btc_monitor_alert_bot

# 4. API 토큰 받기 (예: 123456789:ABCDEF...)
# 5. 채팅방 ID 확인
# - 봇과 대화 시작
# - https://api.telegram.org/bot[TOKEN]/getUpdates 접속하여 chat.id 확인
```

### 2️⃣ Azure 계정 및 도구 설치
```bash
# Azure CLI 설치 (Mac)
brew install azure-cli

# Azure Functions Core Tools 설치
npm install -g azure-functions-core-tools@4

# Azure 로그인
az login
```

## 🚀 배포 단계별 가이드

### 1단계: Azure Function 프로젝트 초기화
```bash
cd /Users/parkyoungjun/Desktop/BTC_Analysis_System

# Functions 프로젝트 생성
func init azure-btc-monitor --python

cd azure-btc-monitor

# HTTP 트리거 함수 생성
func new --name btc-monitor --template "HTTP trigger"
```

### 2단계: 코드 배치
```bash
# 메인 코드 복사
cp ../azure_모니터링_시스템.py btc-monitor/__init__.py

# requirements.txt 수정
cat > requirements.txt << 'EOF'
azure-functions
aiohttp>=3.8.0
pandas>=1.5.0
numpy>=1.21.0
requests>=2.28.0
EOF
```

### 3단계: Azure 리소스 생성
```bash
# 리소스 그룹 생성
az group create --name btc-monitor-rg --location "East US"

# Storage Account 생성
az storage account create \
  --name btcmonitorstorage \
  --location "East US" \
  --resource-group btc-monitor-rg \
  --sku Standard_LRS

# Function App 생성
az functionapp create \
  --resource-group btc-monitor-rg \
  --consumption-plan-location "East US" \
  --runtime python \
  --runtime-version 3.9 \
  --functions-version 4 \
  --name btc-monitor-function \
  --storage-account btcmonitorstorage
```

### 4단계: 환경 변수 설정
```bash
# 텔레그램 봇 토큰 설정
az functionapp config appsettings set \
  --name btc-monitor-function \
  --resource-group btc-monitor-rg \
  --settings "TELEGRAM_BOT_TOKEN=여기에_봇_토큰_입력"

# 텔레그램 채팅 ID 설정  
az functionapp config appsettings set \
  --name btc-monitor-function \
  --resource-group btc-monitor-rg \
  --settings "TELEGRAM_CHAT_ID=여기에_채팅_ID_입력"
```

### 5단계: 함수 배포
```bash
# Azure에 배포
func azure functionapp publish btc-monitor-function
```

### 6단계: 타이머 트리거 설정 (자동 실행용)
Azure Portal에서 다음 설정:

1. **Azure Portal** → **Function Apps** → **btc-monitor-function** 접속
2. **Functions** → **+ Create** 클릭
3. **Timer trigger** 선택
4. **Schedule (CRON)**: `0 */5 * * * *` (5분마다 실행)
5. **Function name**: `btc-monitor-timer`

### 7단계: 테스트 및 확인
```bash
# 수동 실행 테스트
func start

# Azure에서 실행 로그 확인
func azure functionapp logstream btc-monitor-function

# 텔레그램 알림 테스트
curl -X POST "https://btc-monitor-function.azurewebsites.net/api/btc-monitor"
```

## 📱 텔레그램 알림 예시

### 🔴 높은 위험도 알림
```
🚨 BTC 모니터링 알림

🔴 BTC 급락 감지: -4.5%
📊 현재 가격: $108,500
💡 예측 모델에 450$ 이상 영향 예상

🔴 고래 움직임 감지: 0.120 변화  
📊 고래 비율: 0.567
💡 대형 거래 예상, 예측 정확도에 큰 영향

⏰ 2025-08-25 16:30:15
```

### 🟡 보통 위험도 알림
```
🚨 BTC 모니터링 알림

🟡 거래량 급증: 2.3x
📊 현재: 156,780 BTC
💡 큰 시장 움직임 예상, 예측 변동성 증가

🟡 MACD 골든크로스 신호
📊 MACD: 0.0034 (변화: +0.0012)
💡 기술적 분석 신호 변화, 예측 재검토 권장

⏰ 2025-08-25 16:35:20
```

## 🛡️ 보안 및 유지보수

### 환경 변수 보안
- **절대** 코드에 API 키를 하드코딩하지 마세요
- Azure Key Vault 사용 권장 (프로덕션 환경)

### 비용 관리
- **Consumption Plan** 사용으로 사용한 만큼만 과금
- 월 100만 실행까지 무료
- 5분마다 실행시 월 비용: $1-2 예상

### 모니터링
```bash
# 실행 로그 실시간 확인
az webapp log tail --name btc-monitor-function --resource-group btc-monitor-rg

# 메트릭 확인
az monitor metrics list \
  --resource btc-monitor-function \
  --resource-group btc-monitor-rg \
  --metric-names Requests
```

## 🔧 문제 해결

### 일반적인 문제
1. **타이머가 작동 안함**: CRON 표현식 확인
2. **텔레그램 알림 안옴**: 봇 토큰과 채팅 ID 재확인
3. **배포 실패**: Python 버전 및 requirements.txt 확인

### 디버깅
```bash
# 로컬에서 테스트
python azure_모니터링_시스템.py

# Azure 로그 확인
func azure functionapp logstream btc-monitor-function --browser
```

## ✅ 완료 후 확인사항
- [ ] Azure Function 정상 배포
- [ ] 타이머 트리거 5분마다 실행 
- [ ] 텔레그램 봇 응답 확인
- [ ] 테스트 알림 수신 확인
- [ ] 로그 모니터링 설정

## 💰 예상 비용
- **Azure Functions**: 월 $1-3
- **Storage Account**: 월 $0.1-0.5  
- **총 예상 비용**: 월 $2-5

## 📞 추가 지원
문제 발생시 Azure 문서 참조:
- https://docs.microsoft.com/en-us/azure/azure-functions/
- https://docs.microsoft.com/en-us/azure/azure-functions/functions-create-first-function-python