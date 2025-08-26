#!/usr/bin/env python3
"""
맞춤 알림 시스템 테스트
"""

import asyncio
from telegram_command_handler import TelegramCommandHandler
from custom_alert_system import CustomAlertSystem
import os

async def test_command_processing():
    """텔레그램 명령어 처리 테스트"""
    
    bot_token = "8333838666:AAECXOavuX4aaUI9bFttGTqzNY2vb_tqGJI"
    chat_id = "5373223115"
    
    handler = TelegramCommandHandler(bot_token, chat_id)
    
    print("🔍 텔레그램 명령어 처리 테스트...")
    
    # 1. 업데이트 확인
    updates = await handler.process_telegram_updates()
    print(f"📨 처리된 업데이트: {len(updates)}개")
    
    for update in updates:
        print(f"  - {update}")
    
    # 2. 명령어 처리 및 응답
    responses = await handler.process_and_respond()
    print(f"📤 발송된 응답: {responses}개")
    
    # 3. 현재 설정된 알림 확인
    alert_system = CustomAlertSystem()
    user_alerts = alert_system.get_user_alerts("5373223115")  # 실제 user_id
    print(f"🔔 설정된 알림: {len(user_alerts)}개")
    
    for alert in user_alerts:
        print(f"  - {alert['indicator_kr']} {alert['operator_kr']} {alert['threshold']}: {alert['message']}")

async def send_command_guide():
    """올바른 명령어 형식 안내"""
    
    bot_token = "8333838666:AAECXOavuX4aaUI9bFttGTqzNY2vb_tqGJI"
    chat_id = "5373223115"
    
    message = """🤖 **맞춤 알림 명령어 안내**

❌ **잘못된 예시**:
"공포지수가 50이하로 떨어지면 알람줘"

✅ **올바른 명령어**:
`/set_alert fear_greed < 50 "공포지수 하락 알림"`

📋 **지원되는 지표들**:
• `RSI` - RSI 지표
• `funding_rate` - 펀딩비
• `fear_greed` - 공포탐욕지수  
• `whale_activity` - 고래 활동
• `social_volume` - 소셜 볼륨
• `exchange_flows` - 거래소 유입

🎯 **명령어 형식**:
`/set_alert [지표] [조건] [값] "[메시지]"`

💡 **다른 예시들**:
• `/set_alert RSI > 70 "RSI 과매수"`
• `/set_alert funding_rate < -0.01 "펀딩비 마이너스"`
• `/set_alert whale_activity > 80 "대량거래"`

다시 시도해보세요! 👆"""
    
    import aiohttp
    
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    
    try:
        async with aiohttp.ClientSession() as session:
            data = {
                "chat_id": chat_id,
                "text": message,
                "parse_mode": "Markdown"
            }
            
            async with session.post(url, json=data) as response:
                if response.status == 200:
                    print("✅ 명령어 안내 발송 성공")
                    return True
                else:
                    error_text = await response.text()
                    print(f"❌ 발송 실패: {response.status} - {error_text}")
                    return False
                    
    except Exception as e:
        print(f"❌ 오류: {e}")
        return False

async def main():
    print("1️⃣ 텔레그램 명령어 처리 테스트")
    await test_command_processing()
    
    print("\n2️⃣ 올바른 명령어 형식 안내 발송")
    await send_command_guide()

if __name__ == "__main__":
    asyncio.run(main())