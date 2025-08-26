#!/usr/bin/env python3
"""
공포지수 알림 설정 성공 메시지 발송
"""

import asyncio
import aiohttp

async def send_success_message():
    """설정 성공 메시지 발송"""
    
    bot_token = "8333838666:AAECXOavuX4aaUI9bFttGTqzNY2vb_tqGJI"
    chat_id = "5373223115"
    
    message = """✅ **공포지수 알림 설정 완료!**

🎯 **설정된 알림**:
• 지표: 공포탐욕지수 (Fear & Greed Index)
• 조건: 50 미만으로 하락시
• 메시지: "공포지수 하락 알림"
• 알림 ID: #1

🔔 **작동 방식**:
• 5분마다 조건 확인
• 조건 달성시 **1회만** 알림 발송
• 중복 알림 방지

📊 **현재 상태**: 
• 활성화됨 ✅
• 대기 중... (조건 확인 중)

🎉 이제 공포지수가 50 이하로 떨어지면 알림을 받으실 수 있습니다!

💡 **추가 명령어**:
• `/list_alerts` - 설정된 알림 목록
• `/help_alerts` - 도움말
• `/set_alert RSI > 70 "RSI 과매수"` - 추가 알림 설정"""
    
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
                    print("✅ 공포지수 알림 설정 완료 메시지 발송 성공")
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
    asyncio.run(send_success_message())