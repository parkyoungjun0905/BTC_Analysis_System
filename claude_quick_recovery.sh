#!/bin/bash

# ⚡ Claude Code 빠른 복구 스크립트
# 컴퓨터 꺼진 후 원클릭 복구

echo "⚡ Claude Code 빠른 복구 시작..."

PROJECT_PATH="/Users/parkyoungjun/Desktop/BTC_Analysis_System"

cd "$PROJECT_PATH" || exit 1

echo "🔍 복구 상태 확인 중..."

# 1단계: VS Code 복구
echo "📝 VS Code 복구 중..."
if ! pgrep -f "Visual Studio Code" > /dev/null; then
    code . &
    echo "✅ VS Code 시작됨"
else
    echo "✅ VS Code 이미 실행 중"
fi

# 2단계: 터미널 환경 준비
echo "💻 터미널 환경 준비 중..."
if ! command -v tmux &> /dev/null; then
    echo "⚠️ tmux 설치 필요 - 기본 터미널 모드로 진행"
    TMUX_MODE=false
else
    TMUX_MODE=true
fi

# 3단계: 프로젝트 상태 확인
echo "📊 프로젝트 상태 확인 중..."
if [[ -f "enhanced_data_collector.py" ]]; then
    echo "✅ 메인 분석 엔진 확인됨"
else
    echo "❌ 메인 분석 엔진 누락!"
fi

if [[ -d "historical_data" ]]; then
    LATEST_FILE=$(ls -t historical_data/*.json 2>/dev/null | head -1)
    if [[ -n "$LATEST_FILE" ]]; then
        echo "✅ 최신 분석 데이터: $(basename "$LATEST_FILE")"
    fi
fi

# 4단계: Claude Code 환경 준비
echo "🤖 Claude Code 환경 준비 중..."

if [[ "$TMUX_MODE" == true ]]; then
    echo "🚀 tmux 세션으로 Claude Code 시작..."
    
    # 기존 세션 정리
    tmux kill-session -t claude_work 2>/dev/null || true
    
    # 새 세션 생성
    tmux new-session -d -s claude_work -c "$PROJECT_PATH"
    tmux send-keys -t claude_work "echo '🎯 BTC Analysis System 복구 완료'" Enter
    tmux send-keys -t claude_work "echo '💡 사용법: ./start_analysis_auto.sh'" Enter
    tmux send-keys -t claude_work "ls -la" Enter
    
    # Claude 창 추가
    tmux new-window -t claude_work -n claude -c "$PROJECT_PATH"
    tmux send-keys -t claude_work:claude "echo '🤖 Claude Code 준비 완료'" Enter
    tmux send-keys -t claude_work:claude "claude" Enter
    
    echo "✅ tmux 세션 'claude_work' 생성 완료"
    echo "🔗 연결 명령어: tmux attach-session -t claude_work"
else
    echo "💻 기본 터미널 모드로 Claude Code 시작..."
    echo "🤖 Claude Code 실행 준비 완료"
fi

# 5단계: 복구 완료 안내
echo ""
echo "🎉 Claude Code 복구 완료!"
echo ""
echo "📋 다음 단계:"
echo "1. VS Code가 자동으로 열렸습니다"
echo "2. 터미널에서 다음 중 선택:"

if [[ "$TMUX_MODE" == true ]]; then
    echo "   📱 tmux 모드: tmux attach-session -t claude_work"
    echo "   💻 일반 모드: claude"
else
    echo "   💻 일반 모드: claude"
fi

echo ""
echo "🎯 BTC 분석 실행:"
echo "   ./start_analysis_auto.sh"
echo ""
echo "💡 팁: 앞으로는 다음 명령어로 빠른 복구 가능"
echo "   ./claude_quick_recovery.sh"

# 6단계: 자동 연결 옵션
echo ""
read -p "🤖 지금 Claude Code를 자동으로 시작하시겠습니까? (y/n): " AUTO_START

if [[ "$AUTO_START" == "y" || "$AUTO_START" == "Y" ]]; then
    if [[ "$TMUX_MODE" == true ]]; then
        echo "🚀 tmux 세션으로 연결 중..."
        tmux attach-session -t claude_work
    else
        echo "🚀 Claude Code 시작 중..."
        claude
    fi
fi