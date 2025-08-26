#!/bin/bash
# 🌱 초급: 작업 시작하기 (복사해서 사용)
# 이 스크립트는 새로운 작업을 시작할 때 사용해요

echo "🚀 작업을 시작합니다!"
echo "작업 내용을 입력해주세요 (예: 로그인 기능 만들기):"
read work_description

# 작업 설명이 비어있는지 확인
if [ -z "$work_description" ]; then
    echo "❌ 작업 내용을 입력해주세요!"
    exit 1
fi

# Git 자동화 로그 시스템 실행
python3 auto_git_logger.py start "$work_description"

echo ""
echo "💡 작업을 완료했을 때는 다음 명령어를 사용하세요:"
echo "   ./complete_work.sh"
echo ""
echo "🔧 문제가 생기면 되돌리기:"
echo "   ./rollback_work.sh"