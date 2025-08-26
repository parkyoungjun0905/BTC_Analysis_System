#!/usr/bin/env python3
"""
중복 시스템 제거 확인 스크립트
"""

import asyncio
import aiohttp

async def send_cleanup_notification():
    """중복 시스템 제거 완료 알림"""
    
    bot_token = "8333838666:AAECXOavuX4aaUI9bFttGTqzNY2vb_tqGJI"
    chat_id = "5373223115"
    
    message = """🧹 **중복 시스템 제거 완료**

❌ **제거됨**: btc-realtime-monitor 컨테이너
🔍 **원인**: "431개 지표" 메시지 발송하는 중복 시스템
✅ **현재**: btc-risk-monitor-func (맞춤 알림 포함) 단독 운영

이제 중복 알림이 없이 깔끔하게 작동합니다! 🎯

**맞춤 알림 테스트**해보세요:
`/set_alert RSI > 70 "테스트 알림"`"""
    
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
                    print("✅ 중복 시스템 제거 완료 알림 발송 성공")
                    return True
                else:
                    error_text = await response.text()
                    print(f"❌ 알림 발송 실패: {response.status}")
                    print(f"오류 내용: {error_text}")
                    return False
                    
    except Exception as e:
        print(f"❌ 알림 발송 중 오류: {e}")
        return False

if __name__ == "__main__":
    asyncio.run(send_cleanup_notification())