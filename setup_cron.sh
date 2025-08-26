#!/bin/bash

echo "⏰ CryptoQuant CSV 자동 다운로드 cron 설정..."

# 현재 디렉토리 확인
SCRIPT_DIR="/Users/parkyoungjun/Desktop/BTC_Analysis_System"
DOWNLOAD_SCRIPT="$SCRIPT_DIR/cryptoquant_downloader.py"

# Python 경로 확인
PYTHON_PATH=$(which python3)

echo "📁 스크립트 위치: $DOWNLOAD_SCRIPT"
echo "🐍 Python 경로: $PYTHON_PATH"

# cron 작업 생성
CRON_JOB="0 9 * * * cd $SCRIPT_DIR && $PYTHON_PATH cryptoquant_downloader.py >> logs/cron_download.log 2>&1"

# 기존 cron 작업 백업
crontab -l > cron_backup.txt 2>/dev/null || echo "# 기존 cron 작업 없음" > cron_backup.txt

# 새 cron 작업 추가 (중복 제거)
(crontab -l 2>/dev/null | grep -v "cryptoquant_downloader.py"; echo "$CRON_JOB") | crontab -

echo ""
echo "✅ cron 작업 설정 완료!"
echo "📅 스케줄: 매일 오전 9시"
echo "📁 로그: logs/cron_download.log"
echo ""
echo "🔧 확인 명령어:"
echo "crontab -l  # 설정된 cron 작업 확인"
echo "tail -f $SCRIPT_DIR/logs/cron_download.log  # 로그 실시간 확인"
echo ""
echo "🧪 테스트 실행:"
echo "cd $SCRIPT_DIR && python3 cryptoquant_downloader.py"