#!/usr/bin/env python3
"""
간단한 공포지수 알림 테스트
"""

import asyncio
import aiohttp

async def send_simple_alert():
    """간단한 테스트 알림 발송"""
    
    bot_token = "8333838666:AAECXOavuX4aaUI9bFttGTqzNY2vb_tqGJI"
    chat_id = "5373223115"
    
    message = """🚨 공포지수 알림 발생! (테스트)

🎯 조건: fear_greed < 50
📊 현재값: 30
💬 공포지수 하락 알림

✅ 맞춤 알림 시스템이 정상 작동합니다!

실제 공포지수가 50 이하로 떨어지면 이와 같은 알림을 받으시게 됩니다."""
    
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    
    try:
        async with aiohttp.ClientSession() as session:
            data = {
                "chat_id": chat_id,
                "text": message
            }
            
            async with session.post(url, json=data) as response:
                if response.status == 200:
                    print("✅ 공포지수 테스트 알림 발송 성공")
                    return True
                else:
                    error_text = await response.text()
                    print(f"❌ 발송 실패: {response.status}")
                    print(f"오류: {error_text}")
                    return False
                    
    except Exception as e:
        print(f"❌ 오류: {e}")
        return False

if __name__ == "__main__":
    asyncio.run(send_simple_alert())