#!/usr/bin/env python3
"""
BTC 종합 분석 시스템 실행기
사용자가 원할 때 클릭만으로 전체 데이터 수집 및 분석 수행
"""

import asyncio
import os
import sys
from datetime import datetime

# 메인 수집기 import
from enhanced_data_collector import EnhancedBTCDataCollector

def print_banner():
    """시작 배너 출력"""
    print("=" * 80)
    print("🚀 BTC 종합 분석 시스템 v2.0")
    print("📊 500+ 지표 수집 + 시계열 분석")
    print("=" * 80)
    print(f"⏰ 실행 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("")

def print_system_info():
    """시스템 정보 출력"""
    print("📋 시스템 구성:")
    print("✅ 기존 analyzer.py 431개 지표")
    print("✅ 고급 온체인 데이터")
    print("✅ 거시경제 지표 (DXY, S&P, VIX, 금, 국채 등)")
    print("✅ 암호화폐 뉴스 데이터")
    print("✅ CryptoQuant CSV 통합")
    print("✅ 시계열 추세 분석")
    print("✅ 시장 체제 변화 감지")
    print("")

def check_requirements():
    """필수 요구사항 확인"""
    print("🔧 시스템 요구사항 확인 중...")
    
    checks = {
        "pandas": False,
        "numpy": False,
        "aiohttp": False,
        "yfinance": False,
        "feedparser": False
    }
    
    for package in checks:
        try:
            __import__(package)
            checks[package] = True
            print(f"✅ {package}: 설치됨")
        except ImportError:
            print(f"❌ {package}: 미설치")
    
    missing = [k for k, v in checks.items() if not v]
    
    if missing:
        print(f"\n⚠️ 미설치 패키지: {', '.join(missing)}")
        print("설치 명령어:")
        for package in missing:
            print(f"pip install {package}")
        print("")
    
    # 핵심 패키지는 필수
    if not all([checks['pandas'], checks['numpy'], checks['aiohttp']]):
        print("❌ 핵심 패키지 미설치로 실행 불가")
        return False
    
    return True

async def main():
    """메인 실행 함수"""
    print_banner()
    print_system_info()
    
    # 요구사항 확인
    if not check_requirements():
        print("🛑 요구사항 확인 실패. 패키지 설치 후 다시 실행하세요.")
        return
    
    try:
        # 데이터 수집 시작
        print("🎯 데이터 수집 시작...")
        print("예상 소요 시간: 2-3분")
        print("(네트워크 상태에 따라 달라질 수 있습니다)")
        print("")
        
        # 수집기 초기화
        collector = EnhancedBTCDataCollector()
        
        # 전체 데이터 수집 실행
        result_file = await collector.collect_all_data()
        
        if result_file:
            print("")
            print("🎉 데이터 수집 성공!")
            print(f"📁 저장된 파일: {result_file}")
            print("")
            print("📋 다음 단계:")
            print("1. 생성된 JSON 파일을 텍스트 에디터로 열기")
            print("2. 전체 내용을 복사")
            print("3. Claude에게 다음과 같이 질문:")
            print("")
            print("   예시:")
            print("   '이 BTC 데이터를 분석해서 다음 질문에 답해줘:'")
            print("   '지금 비트코인 지지선이 어떻게 되지?'")
            print("   [JSON 데이터 붙여넣기]")
            print("")
            print("🔍 시계열 분석도 포함되어 더 정확한 분석이 가능합니다!")
            
        else:
            print("❌ 데이터 수집 실패")
            print("네트워크 연결이나 API 상태를 확인해주세요.")
        
    except KeyboardInterrupt:
        print("\n⏹️ 사용자에 의해 중단됨")
    except Exception as e:
        print(f"\n❌ 예상치 못한 오류: {e}")
    
    print("")
    print("=" * 80)

if __name__ == "__main__":
    # asyncio 이벤트 루프 실행
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 종료됨")
    
    # 사용자가 결과를 볼 수 있도록 대기
    input("\n⏎ Enter 키를 눌러 종료...")