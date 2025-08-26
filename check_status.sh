#!/bin/bash
# 🌱 초급: 작업 현황 확인하기 (복사해서 사용)
# 현재까지의 모든 작업 상황을 보여주는 스크립트예요

echo "📊 작업 현황을 확인합니다..."
echo ""

# Git 자동화 로그 시스템으로 상태 확인
python3 auto_git_logger.py status

echo ""
echo "📝 상세한 로그를 보려면:"
echo "   cat coding_log.md"
echo ""
echo "🔧 다른 명령어:"
echo "   ./start_work.sh     - 새 작업 시작"
echo "   ./complete_work.sh  - 작업 완료"
echo "   ./rollback_work.sh  - 작업 되돌리기"
echo "   ./daily_backup.sh   - 하루 종료 백업"