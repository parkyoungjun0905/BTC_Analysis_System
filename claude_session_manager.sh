#!/bin/bash

# 🤖 Claude Code 세션 관리 스크립트
# 컴퓨터 꺼짐/재시작 후 Claude Code 작업 복구 도구

SESSION_NAME="claude_btc_work"
PROJECT_PATH="/Users/parkyoungjun/Desktop/BTC_Analysis_System"
CLAUDE_HISTORY_FILE="$HOME/.claude_session_history"

echo "🤖 Claude Code 세션 관리자"

# 함수: 새 세션 시작
start_new_session() {
    echo "🚀 새 Claude Code 세션 시작..."
    
    # tmux 세션 생성
    tmux new-session -d -s "$SESSION_NAME" -c "$PROJECT_PATH"
    
    # 기본 창들 설정
    tmux rename-window -t "$SESSION_NAME:0" "main"
    tmux send-keys -t "$SESSION_NAME:main" "cd $PROJECT_PATH" Enter
    tmux send-keys -t "$SESSION_NAME:main" "echo '🎯 BTC Analysis System 작업 환경'" Enter
    tmux send-keys -t "$SESSION_NAME:main" "ls -la" Enter
    
    # Claude Code 창 생성
    tmux new-window -t "$SESSION_NAME" -n "claude" -c "$PROJECT_PATH"
    tmux send-keys -t "$SESSION_NAME:claude" "claude" Enter
    
    # 분석 실행 창 생성
    tmux new-window -t "$SESSION_NAME" -n "analysis" -c "$PROJECT_PATH"
    tmux send-keys -t "$SESSION_NAME:analysis" "echo '📊 분석 실행 준비 완료'" Enter
    
    # 세션 정보 저장
    echo "$(date): 새 세션 시작 - $SESSION_NAME" >> "$CLAUDE_HISTORY_FILE"
    
    echo "✅ 세션 '$SESSION_NAME' 생성 완료!"
    echo "📋 창 구성:"
    echo "  - main: 메인 작업 창"
    echo "  - claude: Claude Code 실행 창"
    echo "  - analysis: 분석 스크립트 실행 창"
}

# 함수: 기존 세션 복구
restore_session() {
    echo "🔄 기존 Claude Code 세션 복구..."
    
    if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
        echo "✅ 세션 '$SESSION_NAME' 발견!"
        echo "📋 현재 창들:"
        tmux list-windows -t "$SESSION_NAME"
        
        echo ""
        echo "🎯 세션에 연결 중..."
        tmux attach-session -t "$SESSION_NAME"
    else
        echo "❌ 기존 세션을 찾을 수 없습니다."
        echo "🆕 새 세션을 생성하시겠습니까? (y/n)"
        read -r response
        if [[ "$response" == "y" || "$response" == "Y" ]]; then
            start_new_session
            tmux attach-session -t "$SESSION_NAME"
        else
            echo "🔚 종료합니다."
            exit 0
        fi
    fi
}

# 함수: VS Code와 연동
open_vscode() {
    echo "📝 VS Code에서 프로젝트 열기..."
    code "$PROJECT_PATH"
    echo "✅ VS Code 열기 완료!"
}

# 함수: 세션 상태 확인
check_status() {
    echo "📊 현재 세션 상태:"
    echo ""
    
    if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
        echo "✅ tmux 세션 '$SESSION_NAME' 실행 중"
        tmux list-windows -t "$SESSION_NAME"
    else
        echo "❌ tmux 세션 없음"
    fi
    
    echo ""
    echo "🔍 Claude 프로세스:"
    ps aux | grep -i claude | grep -v grep | head -3
    
    echo ""
    echo "📁 프로젝트 경로: $PROJECT_PATH"
    echo "📈 최근 분석 파일:"
    ls -la "$PROJECT_PATH/historical_data/" | tail -3
}

# 함수: 자동 복구 스크립트
auto_recovery() {
    echo "🔧 자동 복구 모드..."
    
    # VS Code 실행 확인
    if ! pgrep -f "Visual Studio Code" > /dev/null; then
        echo "🚀 VS Code 시작 중..."
        open_vscode
        sleep 3
    fi
    
    # tmux 세션 복구/생성
    if ! tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
        start_new_session
    fi
    
    # Claude Code 세션 연결
    echo "🤖 Claude Code 세션 연결 준비 완료!"
    tmux attach-session -t "$SESSION_NAME"
}

# 함수: 도움말
show_help() {
    echo "🤖 Claude Code 세션 관리자 사용법:"
    echo ""
    echo "사용법: $0 [옵션]"
    echo ""
    echo "옵션:"
    echo "  start     - 새 세션 시작"
    echo "  restore   - 기존 세션 복구"
    echo "  status    - 현재 상태 확인"
    echo "  vscode    - VS Code 열기"
    echo "  auto      - 자동 복구 (추천)"
    echo "  help      - 이 도움말 표시"
    echo ""
    echo "🎯 추천 사용법:"
    echo "  컴퓨터 재시작 후: ./claude_session_manager.sh auto"
    echo "  작업 중단 후 복구: ./claude_session_manager.sh restore"
}

# 메인 로직
case "${1:-auto}" in
    "start")
        start_new_session
        tmux attach-session -t "$SESSION_NAME"
        ;;
    "restore")
        restore_session
        ;;
    "status")
        check_status
        ;;
    "vscode")
        open_vscode
        ;;
    "auto")
        auto_recovery
        ;;
    "help"|"-h"|"--help")
        show_help
        ;;
    *)
        echo "❌ 알 수 없는 옵션: $1"
        show_help
        exit 1
        ;;
esac