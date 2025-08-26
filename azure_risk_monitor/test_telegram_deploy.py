#!/usr/bin/env python3
"""
맞춤 알림 시스템 배포 완료 테스트 스크립트
"""

import asyncio
import aiohttp

async def send_deployment_notification():
    """배포 완료 알림 발송"""
    
    bot_token = "8333838666:AAECXOavuX4aaUI9bFttGTqzNY2vb_tqGJI"
    chat_id = "5373223115"
    
    message = """🚀 **맞춤 알림 시스템 배포 완료!**

📋 **사용 가능한 명령어**:
• `/set_alert RSI > 70 "RSI 과매수 경고"` - 알림 설정
• `/list_alerts` - 설정된 알림 목록 보기
• `/remove_alert [ID]` - 알림 삭제  
• `/clear_all_alerts` - 모든 알림 삭제
• `/help_alerts` - 자세한 도움말

💡 **사용 예시**:
• `/set_alert funding_rate < -0.01 "펀딩비 마이너스"`
• `/set_alert fear_greed < 20 "극공포 상태"`
• `/set_alert whale_activity > 80 "대량거래 감지"`

✅ **1회성 알림**: 조건 달성시 1회만 발송
⚡ **실시간 감시**: 5분마다 조건 확인 중

시작해보세요! 👆"""
    
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
                    print("✅ 맞춤 알림 시스템 배포 완료 알림 발송 성공")
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
    asyncio.run(send_deployment_notification())