#!/bin/bash

# 🔧 BTC Analysis System 권한 문제 해결사
echo "🔧 권한 문제 완전 해결 중..."

PROJECT_PATH="/Users/parkyoungjun/Desktop/BTC_Analysis_System"
cd "$PROJECT_PATH" || exit 1

echo "📋 현재 권한 상태 확인..."
ls -la *.sh

echo ""
echo "🔨 모든 스크립트 실행 권한 강제 설정..."

# 모든 .sh 파일에 실행 권한 부여
chmod +x *.sh
chmod +x *.py 2>/dev/null || true

echo "✅ 실행 권한 설정 완료"

echo ""
echo "🔐 macOS 보안 설정 확인 및 해제..."

# macOS Gatekeeper 설정 확인
echo "📱 Gatekeeper 상태:"
spctl --status

# 스크립트들을 신뢰할 수 있는 목록에 추가
echo "🛡️ 스크립트들을 신뢰 목록에 추가 중..."
for script in *.sh *.py; do
    if [[ -f "$script" ]]; then
        # 확장 속성 제거 (macOS 다운로드 표시 제거)
        xattr -d com.apple.quarantine "$script" 2>/dev/null || true
        echo "✅ $script 신뢰 설정 완료"
    fi
done

echo ""
echo "🔧 추가 권한 설정..."

# 소유자 권한 확실히 설정
sudo chown -R $(whoami):staff . 2>/dev/null || chown -R $(whoami):staff .

echo ""
echo "🎯 권한 문제 해결 방법들:"
echo ""

echo "1️⃣ 즉시 해결 방법:"
echo "   sudo 없이 실행: ./스크립트명.sh"
echo "   또는: bash 스크립트명.sh"
echo ""

echo "2️⃣ 완전 해결 방법:"
echo "   시스템 환경설정 > 보안 및 개인정보보호"
echo "   > 개인정보보호 > 전체 디스크 접근 권한"
echo "   > 터미널 앱 추가 후 허용"
echo ""

echo "3️⃣ 터미널별 해결 방법:"
echo "   VS Code 터미널: code . 후 내장 터미널 사용"
echo "   일반 터미널: 위치 이동 후 실행"
echo ""

echo "4️⃣ 자동 승인 실행:"
echo "   yes | ./스크립트명.sh"
echo "   DEBIAN_FRONTEND=noninteractive ./스크립트명.sh"

echo ""
echo "🚨 그래도 안 되면:"
echo "   1. 시스템 재시작"
echo "   2. sudo ./fix_permissions.sh 실행"
echo "   3. 보안 설정에서 터미널 권한 허용"

echo ""
echo "✅ 권한 문제 해결 완료!"