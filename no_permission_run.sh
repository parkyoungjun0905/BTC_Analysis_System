#!/bin/bash

# 🚀 권한 질문 없이 실행하는 래퍼 스크립트들

echo "🚀 권한 질문 없는 실행 모드"
echo ""

# 환경 변수 설정으로 자동 승인
export DEBIAN_FRONTEND=noninteractive
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_NO_INPUT=1

# 사용자에게 선택 메뉴 제공
echo "실행할 작업을 선택하세요:"
echo ""
echo "1) 🔧 초기 설정 (자동 승인)"
echo "2) 📊 분석 실행 (자동 승인)"  
echo "3) ⚡ 빠른 복구 (자동 승인)"
echo "4) 🛠️ 권한 문제 해결"
echo "5) 📋 상태 확인"
echo ""

read -p "선택하세요 (1-5): " choice

case $choice in
    1)
        echo "🔧 초기 설정 시작 (자동 승인 모드)..."
        yes | bash setup_auto.sh
        ;;
    2)
        echo "📊 분석 실행 시작 (자동 승인 모드)..."
        yes | bash start_analysis_auto.sh
        ;;
    3)
        echo "⚡ 빠른 복구 시작 (자동 승인 모드)..."
        yes | bash claude_quick_recovery.sh
        ;;
    4)
        echo "🛠️ 권한 문제 해결 시작..."
        bash fix_permissions.sh
        ;;
    5)
        echo "📋 현재 상태 확인..."
        echo ""
        echo "📁 파일 권한:"
        ls -la *.sh | grep -E "\.(sh)$"
        echo ""
        echo "🔍 프로세스:"
        ps aux | grep -i claude | grep -v grep | head -3
        echo ""
        echo "📊 최근 분석 결과:"
        ls -la historical_data/ | tail -3
        ;;
    *)
        echo "❌ 잘못된 선택입니다."
        exit 1
        ;;
esac

echo ""
echo "✅ 작업 완료!"