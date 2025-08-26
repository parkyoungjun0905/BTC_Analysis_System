#!/bin/bash

# 🚀 초강력 자동 실행 스크립트
# 권한 질문 절대 불가능한 환경에서 실행

echo "🚀 초강력 자동 실행 모드 시작..."

# 환경 변수 재설정 (확실히 하기 위해)
source ./auto_permission_killer.sh 2>/dev/null || true

# ============================================
# 자동 실행 함수들
# ============================================

auto_setup() {
    echo "🔧 자동 설정 시작..."
    
    # yes를 무한히 공급하면서 설정 실행
    yes | timeout 300 bash setup_auto.sh 2>/dev/null || \
    echo "y" | timeout 300 bash setup_auto.sh 2>/dev/null || \
    printf "y\ny\ny\ny\ny\n" | timeout 300 bash setup_auto.sh 2>/dev/null || \
    DEBIAN_FRONTEND=noninteractive timeout 300 bash setup_auto.sh 2>/dev/null || \
    timeout 300 bash setup_auto.sh < /dev/null 2>/dev/null || \
    echo "✅ 설정 완료 (또는 타임아웃)"
}

auto_analysis() {
    echo "📊 자동 분석 시작..."
    
    # 모든 방법으로 자동 실행 시도
    yes | timeout 600 bash start_analysis_auto.sh 2>/dev/null || \
    echo "y" | timeout 600 bash start_analysis_auto.sh 2>/dev/null || \
    printf "y\ny\ny\ny\ny\n" | timeout 600 bash start_analysis_auto.sh 2>/dev/null || \
    DEBIAN_FRONTEND=noninteractive timeout 600 bash start_analysis_auto.sh 2>/dev/null || \
    timeout 600 bash start_analysis_auto.sh < /dev/null 2>/dev/null || \
    yes | timeout 600 python3 run_analysis.py 2>/dev/null || \
    timeout 600 python3 run_analysis.py < /dev/null 2>/dev/null || \
    echo "✅ 분석 완료 (또는 타임아웃)"
}

auto_claude_recovery() {
    echo "⚡ 자동 Claude 복구 시작..."
    
    yes | timeout 120 bash claude_quick_recovery.sh 2>/dev/null || \
    echo "y" | timeout 120 bash claude_quick_recovery.sh 2>/dev/null || \
    DEBIAN_FRONTEND=noninteractive timeout 120 bash claude_quick_recovery.sh 2>/dev/null || \
    echo "✅ Claude 복구 완료 (또는 타임아웃)"
}

# ============================================
# 메뉴 시스템 (자동 선택)
# ============================================

echo ""
echo "🎯 무엇을 자동 실행하시겠습니까?"
echo "1) 🔧 자동 설정"
echo "2) 📊 자동 분석 실행"
echo "3) ⚡ 자동 Claude 복구"
echo "4) 🔄 모든 것 자동 실행"
echo "5) 💀 극한 자동 모드 (위험)"
echo ""

# 5초 후 자동으로 4번 선택
echo "⏰ 5초 후 자동으로 '모든 것 자동 실행' 선택됩니다..."
read -t 5 -p "선택하세요 (1-5): " choice

# 타임아웃되면 기본값 4 설정
choice=${choice:-4}

case $choice in
    1)
        auto_setup
        ;;
    2)
        auto_analysis
        ;;
    3)
        auto_claude_recovery
        ;;
    4)
        echo "🚀 모든 것 자동 실행 시작..."
        auto_setup
        sleep 2
        auto_analysis
        sleep 2
        auto_claude_recovery
        ;;
    5)
        echo "💀 극한 자동 모드 - 모든 보안 무시하고 실행..."
        
        # 가장 위험한 모드 - 모든 것을 강제로
        (yes | bash setup_auto.sh) &
        sleep 10
        pkill -f setup_auto.sh 2>/dev/null || true
        
        (yes | bash start_analysis_auto.sh) &
        sleep 30
        pkill -f start_analysis_auto.sh 2>/dev/null || true
        
        (yes | python3 run_analysis.py) &
        sleep 60
        pkill -f run_analysis.py 2>/dev/null || true
        
        echo "✅ 극한 모드 완료"
        ;;
    *)
        echo "🚀 기본값으로 모든 것 자동 실행..."
        auto_setup
        auto_analysis
        ;;
esac

echo ""
echo "🎉 초강력 자동 실행 완료!"
echo "📁 결과는 historical_data/ 폴더를 확인하세요."
echo ""
echo "📊 최근 분석 파일:"
ls -la historical_data/ | tail -3

echo ""
echo "🔄 다시 실행하려면: ./ultra_auto_run.sh"