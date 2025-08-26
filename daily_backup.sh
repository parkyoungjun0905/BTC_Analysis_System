#!/bin/bash
# 🌱 초급: 하루 종료 백업 (복사해서 사용)
# 하루 작업을 마칠 때 전체 백업을 만드는 스크립트예요

echo "💾 하루 종료 백업을 시작합니다..."
echo ""

# Git 자동화 로그 시스템으로 백업
python3 auto_git_logger.py backup

echo ""
echo "🎉 백업이 완료되었습니다!"
echo ""
echo "💡 백업으로 되돌리고 싶을 때는:"
echo "   git branch -a  (백업 브랜치 목록 보기)"
echo "   git checkout backup-날짜  (특정 백업으로 이동)"
echo ""
echo "😴 수고하셨습니다! 좋은 하루 되세요!"