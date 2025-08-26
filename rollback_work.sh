#!/bin/bash
# 🌱 초급: 작업 되돌리기 (복사해서 사용)
# 문제가 생겼을 때 이전 상태로 되돌리는 스크립트예요

echo "🔄 작업을 되돌립니다!"
echo ""
echo "어떻게 되돌릴까요?"
echo "1. 최근 작업으로 되돌리기 (추천)"
echo "2. 특정 작업 번호로 되돌리기"
echo "3. 오늘 시작 시점으로 되돌리기"
echo ""
echo "번호를 선택하세요 (1-3):"
read choice

case $choice in
    1)
        echo "🔄 최근 작업으로 되돌리는 중..."
        python3 auto_git_logger.py rollback
        ;;
    2)
        echo "작업 번호를 입력하세요:"
        read work_number
        if [[ "$work_number" =~ ^[0-9]+$ ]]; then
            echo "🔄 작업 #$work_number으로 되돌리는 중..."
            python3 auto_git_logger.py rollback $work_number
        else
            echo "❌ 올바른 숫자를 입력해주세요!"
            exit 1
        fi
        ;;
    3)
        echo "🔄 오늘 시작 시점으로 되돌리는 중..."
        python3 auto_git_logger.py rollback today
        ;;
    *)
        echo "❌ 1, 2, 3 중에서 선택해주세요!"
        exit 1
        ;;
esac

echo ""
echo "💡 현재 상황을 확인하려면:"
echo "   ./check_status.sh"
echo ""
echo "🚀 새로운 작업을 시작하려면:"
echo "   ./start_work.sh"