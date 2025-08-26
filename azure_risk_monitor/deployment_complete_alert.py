#!/usr/bin/env python3
"""
배포 완료 알림
"""

import asyncio
import aiohttp

async def send_deployment_complete():
    """배포 완료 알림 발송"""
    
    bot_token = "8333838666:AAECXOavuX4aaUI9bFttGTqzNY2vb_tqGJI"
    chat_id = "5373223115"
    
    message = """🚀 **자연어 시스템 배포 완료!**

## ✅ **실제 배포됨**
• **Azure Functions**: btc-risk-monitor-func 업데이트 완료
• **자연어 처리**: 실제 적용됨  
• **100+ 지표**: 실시간 감시 활성화

## 🧠 **지금부터 실제 작동!**

**테스트해보세요**:
• "달러지수가 105 넘으면 알려줘"
• "VIX가 30 초과하면 공포지수 경고"  
• "금가격이 2000 넘으면 알림"
• "나스닥이 하락하면 감지"

## ⚡ **실시간 동시 작동**
1️⃣ **AI 예측 시스템** (5분마다 자동)
2️⃣ **맞춤 알림 시스템** (자연어 명령 처리)

## 🎯 **이제 정말로 완성!**
• 자연어로 편하게 명령하세요
• 90개 지표 모두 실시간 감시
• 중복 없는 1회성 알림

바로 테스트해보세요! 👆"""
    
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
                    print("✅ 배포 완료 알림 발송 성공")
                    return True
                else:
                    error_text = await response.text()
                    print(f"❌ 발송 실패: {response.status}")
                    return False
                    
    except Exception as e:
        print(f"❌ 오류: {e}")
        return False

if __name__ == "__main__":
    asyncio.run(send_deployment_complete())