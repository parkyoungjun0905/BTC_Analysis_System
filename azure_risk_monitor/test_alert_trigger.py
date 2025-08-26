#!/usr/bin/env python3
"""
공포지수 알림 트리거 테스트
"""

import asyncio
from custom_alert_system import CustomAlertSystem
import aiohttp

async def test_fear_greed_alert():
    """공포지수 알림 테스트"""
    
    user_id = "5373223115"
    alert_system = CustomAlertSystem()
    
    print("🔍 공포지수 알림 트리거 테스트...")
    
    # 테스트용 지표 데이터 (실제 구조로 설정)
    test_indicators = {
        "enhanced_19_system": {
            "detailed_analysis": {
                "fear_greed": {
                    "current_value": 30,  # 조건 만족 (< 50)
                    "signal": "EXTREME_FEAR"
                },
                "price_momentum": {
                    "rsi_14": 45
                },
                "funding_rate": {
                    "current_value": 0.01
                },
                "whale_activity": {
                    "current_value": 60
                }
            }
        }
    }
    
    fear_greed_value = test_indicators['enhanced_19_system']['detailed_analysis']['fear_greed']['current_value']
    print(f"📊 테스트 데이터: 공포지수 = {fear_greed_value}")
    
    # 맞춤 알림 조건 확인
    triggered_alerts = await alert_system.check_custom_alerts(test_indicators, user_id)
    
    print(f"🚨 트리거된 알림: {len(triggered_alerts)}개")
    
    if triggered_alerts:
        for alert in triggered_alerts:
            print(f"  🔔 알림: {alert['message']}")
            print(f"  📈 조건: {alert['indicator']} {alert['operator']} {alert['threshold']}")
            print(f"  💹 현재값: {alert.get('current_value', 'N/A')}")
            
            # 텔레그램으로 테스트 알림 발송
            await send_test_alert(alert)
    else:
        print("❌ 조건에 맞는 알림이 없습니다.")
        
        # 현재 설정된 알림 상태 확인
        alerts = alert_system.get_user_alerts(user_id)
        for alert in alerts:
            print(f"  📋 알림 ID {alert['id']}: {alert['indicator']} {alert['operator']} {alert['threshold']}")
            print(f"      활성: {alert['is_active']}, 발송됨: {alert['is_triggered']}")

async def send_test_alert(alert):
    """테스트 알림 발송"""
    
    bot_token = "8333838666:AAECXOavuX4aaUI9bFttGTqzNY2vb_tqGJI"
    chat_id = "5373223115"
    
    message = f"""🚨 **맞춤 알림 발생!** (테스트)

🎯 **조건 달성**:
• 지표: {alert['indicator']}
• 조건: {alert['operator']} {alert['threshold']}
• 현재값: {alert.get('current_value', 'N/A')}

💬 **메시지**: {alert['message']}

⏰ 시간: {alert.get('triggered_at', 'N/A')}

✅ **1회성 알림** - 다시 발송되지 않습니다."""
    
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    
    try:
        async with aiohttp.ClientSession() as session:
            data = {
                "chat_id": chat_id,
                "text": message,
                "parse_mode": "HTML"
            }
            
            async with session.post(url, json=data) as response:
                if response.status == 200:
                    print("✅ 테스트 알림 발송 성공")
                else:
                    error_text = await response.text()
                    print(f"❌ 알림 발송 실패: {response.status}")
                    print(f"오류: {error_text}")
                    
    except Exception as e:
        print(f"❌ 알림 발송 중 오류: {e}")

if __name__ == "__main__":
    asyncio.run(test_fear_greed_alert())