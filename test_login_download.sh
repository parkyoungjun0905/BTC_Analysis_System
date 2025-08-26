#!/bin/bash

# 🧪 로그인 다운로드 테스트 스크립트
echo "🧪 CryptoQuant 로그인 다운로드 테스트..."

PROJECT_PATH="/Users/parkyoungjun/Desktop/BTC_Analysis_System"
cd "$PROJECT_PATH" || exit 1

echo "📋 테스트 시나리오:"
echo "1. 첫 실행 - 다운로드 진행"
echo "2. 재실행 - 중복 방지로 스킵"
echo ""

echo "🚀 첫 번째 실행 테스트..."
python3 cryptoquant_downloader.py

echo ""
echo "⏰ 5초 대기 후 재실행 테스트..."
sleep 5

echo "🔄 두 번째 실행 테스트 (중복 방지 확인)..."
python3 cryptoquant_downloader.py

echo ""
echo "📁 다운로드 결과 확인:"
ls -la cryptoquant_csv_data/ | head -10

echo ""
echo "📊 다운로드 요약:"
if [[ -f "cryptoquant_csv_data/download_summary.json" ]]; then
    cat cryptoquant_csv_data/download_summary.json
else
    echo "❌ 다운로드 요약 파일 없음"
fi

echo ""
echo "✅ 테스트 완료!"