#!/bin/bash

echo "🚀 BTC 종합 분석 시스템 설정 시작 (자동 승인 모드)..."

# Python 버전 확인
python_version=$(python3 --version 2>&1 | cut -d" " -f2)
echo "🐍 Python 버전: $python_version"

# pip 업그레이드 (자동 승인)
echo "📦 pip 업그레이드 중..."
yes | python3 -m pip install --upgrade pip 2>/dev/null

# 필수 패키지 설치 (자동 승인)
echo "📚 필수 패키지 설치 중..."
yes | pip3 install -r requirements.txt 2>/dev/null

echo ""
echo "✅ 설정 완료! (모든 질문 자동 승인됨)"
echo ""
echo "🎯 사용법:"
echo "1. ./start_analysis_auto.sh 실행 (자동 승인)"
echo "2. 또는 yes | python3 run_analysis.py 실행"
echo ""
echo "📁 생성된 파일은 historical_data/ 폴더에 저장됩니다."