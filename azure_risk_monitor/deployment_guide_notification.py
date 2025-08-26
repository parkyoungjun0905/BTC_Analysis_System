#!/usr/bin/env python3
"""
배포 안전 가이드 알림 발송
"""

import asyncio
import aiohttp

async def send_deployment_guide():
    """배포 안전 가이드 알림"""
    
    bot_token = "8333838666:AAECXOavuX4aaUI9bFttGTqzNY2vb_tqGJI"
    chat_id = "5373223115"
    
    message = """🛡️ **클로드 코드 배포 안전 시스템 완성!**

## ⚠️ **문제 해결됨**
**기존 문제**: "수정했다 = 배포됐다" 착각
**해결책**: 자동 검증 + 알림 시스템

## 🚀 **새로운 안전 배포 방법**

### 기존 (위험한) 방식 ❌
```
1. 코드 수정
2. zip + az deploy
3. "배포됐겠지?" (확인 안함)  
4. 사용자는 모름 → 위험!
```

### 새로운 (안전한) 방식 ✅
```
1. 코드 수정
2. python3 safe_deploy.py  
3. 자동 검증 (30초 대기)
4. 텔레그램 알림 (성공/실패)
5. 사용자 확인 완료!
```

## 🔧 **사용법**
앞으로는 이렇게 배포하세요:
```bash
# 안전한 배포 (한 번에 처리)
python3 safe_deploy.py
```

## 📋 **자동 검증 항목**
• ☁️ Azure Functions 헬스체크
• 🤖 텔레그램 봇 응답
• ⚡ 핵심 기능 테스트
• 📱 사용자 알림 발송

## 🎯 **이제 걱정 없이 개발하세요!**
• 배포 상태를 항상 정확히 알 수 있음
• 실패시 자동으로 알림 받음
• 롤백 필요시 즉시 대응 가능

**클로드 코드의 "배포 불일치" 문제가 완전히 해결됐습니다!** 🎉"""
    
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
                    print("✅ 배포 안전 가이드 알림 발송 성공")
                    return True
                else:
                    error_text = await response.text()
                    print(f"❌ 발송 실패: {response.status}")
                    return False
                    
    except Exception as e:
        print(f"❌ 오류: {e}")
        return False

if __name__ == "__main__":
    asyncio.run(send_deployment_guide())