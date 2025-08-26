#!/usr/bin/env python3
"""
전체 시스템 통합 테스트
로컬에서 실행하여 모든 컴포넌트 동작 확인
"""

import asyncio
import logging
from datetime import datetime
import sys

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

async def test_individual_components():
    """각 컴포넌트별 개별 테스트"""
    print("=" * 60)
    print("🧪 개별 컴포넌트 테스트")
    print("=" * 60)
    
    results = {}
    
    # 1. 데이터 수집기 테스트
    print("\n1️⃣ 데이터 수집기 테스트...")
    try:
        from data_collector import test_data_collection
        data, risk_indicators = await test_data_collection()
        results["data_collector"] = True
        print("   ✅ 데이터 수집기 정상")
    except Exception as e:
        print(f"   ❌ 데이터 수집기 오류: {e}")
        results["data_collector"] = False
    
    # 2. 위험 분석기 테스트  
    print("\n2️⃣ 위험 분석기 테스트...")
    try:
        from risk_analyzer import test_risk_analyzer
        risk_analysis = test_risk_analyzer()
        results["risk_analyzer"] = True
        print("   ✅ 위험 분석기 정상")
    except Exception as e:
        print(f"   ❌ 위험 분석기 오류: {e}")
        results["risk_analyzer"] = False
    
    # 3. 텔레그램 알리미 테스트
    print("\n3️⃣ 텔레그램 알리미 테스트...")
    try:
        from telegram_notifier import test_telegram_notifier
        telegram_success = await test_telegram_notifier()
        results["telegram_notifier"] = telegram_success
        if telegram_success:
            print("   ✅ 텔레그램 알리미 정상")
        else:
            print("   ❌ 텔레그램 알리미 실패")
    except Exception as e:
        print(f"   ❌ 텔레그램 알리미 오류: {e}")
        results["telegram_notifier"] = False
    
    return results

async def test_integrated_system():
    """통합 시스템 테스트"""
    print("\n" + "=" * 60)
    print("🚀 통합 시스템 테스트")
    print("=" * 60)
    
    try:
        from main_monitor import run_local_test
        await run_local_test()
        return True
    except Exception as e:
        print(f"❌ 통합 시스템 테스트 실패: {e}")
        return False

def print_test_summary(component_results, integration_result):
    """테스트 결과 요약 출력"""
    print("\n" + "=" * 60)
    print("📊 테스트 결과 요약")
    print("=" * 60)
    
    print("\n🔧 개별 컴포넌트:")
    for component, success in component_results.items():
        status = "✅ 통과" if success else "❌ 실패"
        component_names = {
            "data_collector": "데이터 수집기",
            "risk_analyzer": "위험 분석기", 
            "telegram_notifier": "텔레그램 알리미"
        }
        name = component_names.get(component, component)
        print(f"   {name}: {status}")
    
    print(f"\n🚀 통합 시스템: {'✅ 통과' if integration_result else '❌ 실패'}")
    
    # 전체 성공률 계산
    total_tests = len(component_results) + 1
    passed_tests = sum(component_results.values()) + (1 if integration_result else 0)
    success_rate = (passed_tests / total_tests) * 100
    
    print(f"\n📈 전체 성공률: {success_rate:.1f}% ({passed_tests}/{total_tests})")
    
    if success_rate == 100:
        print("\n🎉 모든 테스트 통과! Azure 배포 준비 완료")
        return True
    elif success_rate >= 75:
        print(f"\n⚠️ 일부 실패 있음. 문제 해결 후 재테스트 권장")
        return False
    else:
        print(f"\n❌ 심각한 문제 발견. 시스템 점검 필요")
        return False

def print_deployment_guide():
    """배포 가이드 출력"""
    print("\n" + "=" * 60)
    print("🌩️ Azure 배포 가이드")
    print("=" * 60)
    
    guide = """
📋 다음 단계로 Azure에 배포하세요:

1️⃣ Azure Function App 생성:
   az functionapp create --resource-group btc-monitor-rg --consumption-plan-location koreacentral --runtime python --runtime-version 3.9 --functions-version 4 --name btc-risk-monitor --storage-account btcmonitorstorage

2️⃣ 환경 변수 설정:
   az functionapp config appsettings set --name btc-risk-monitor --resource-group btc-monitor-rg --settings TG_BOT_TOKEN="{}" TG_CHAT_ID="{}"

3️⃣ 코드 배포:
   func azure functionapp publish btc-risk-monitor

4️⃣ 타이머 함수 설정:
   - function.json에서 schedule 설정: "0 */1 * * * *" (1분마다)

5️⃣ 모니터링 설정:
   - Application Insights 활성화
   - 알림 규칙 설정

💰 예상 비용: 월 3-5만원
⏱️ 예상 배포 시간: 30-60분
""".format("8333838666:AAECXOavuX4aaUI9bFttGTqzNY2vb_tqGJI", "5373223115")

    print(guide)

async def main():
    """메인 테스트 함수"""
    print("🚀 Azure BTC 위험 감지 시스템 - 전체 테스트")
    print(f"📅 테스트 시작: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # 개별 컴포넌트 테스트
        component_results = await test_individual_components()
        
        # 통합 시스템 테스트 (개별 테스트가 모두 통과한 경우만)
        if all(component_results.values()):
            integration_result = await test_integrated_system()
        else:
            print("\n⚠️ 개별 컴포넌트 실패로 통합 테스트 스킵")
            integration_result = False
        
        # 결과 요약
        all_passed = print_test_summary(component_results, integration_result)
        
        # 배포 가이드 (모든 테스트 통과 시)
        if all_passed:
            print_deployment_guide()
            
    except KeyboardInterrupt:
        print("\n\n⚠️ 테스트가 사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"\n\n❌ 테스트 중 예상치 못한 오류 발생: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n📅 테스트 완료: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    # 전체 테스트 실행
    asyncio.run(main())