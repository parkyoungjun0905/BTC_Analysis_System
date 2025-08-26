#!/bin/bash

echo "🚀 BTC 종합 분석 시작..."
echo "📊 500+ 지표 수집 중..."
echo ""

# Python 스크립트 실행
python3 run_analysis.py

echo ""
echo "📁 결과 확인:"
echo "historical_data/ 폴더의 최신 파일을 확인하세요."

echo ""
echo "🎯 다음 단계:"
echo "1. 생성된 JSON 파일을 열어서 내용 복사"
echo "2. Claude에게 질문과 함께 전달"
echo ""
echo "예시: '이 데이터로 지지선 분석해줘' + JSON 데이터"