#!/bin/bash
# 🌱 초급: 작업 완료하기 (복사해서 사용)
# 이 스크립트는 작업을 완료할 때 사용해요

echo "✅ 작업을 완료합니다!"
echo "완료한 내용을 입력해주세요 (예: 로그인 기능 구현 완료):"
read completion_description

# 완료 설명이 비어있는지 확인
if [ -z "$completion_description" ]; then
    echo "❌ 완료 내용을 입력해주세요!"
    exit 1
fi

# Git 자동화 로그 시스템 실행
python3 auto_git_logger.py complete "$completion_description"

echo ""
echo "🎉 축하합니다! 작업이 완료되었습니다"
echo ""
echo "💡 다음 작업을 시작하려면:"
echo "   ./start_work.sh"
echo ""
echo "📊 전체 작업 현황 보기:"
echo "   ./check_status.sh"