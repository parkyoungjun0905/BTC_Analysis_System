#!/usr/bin/env python3
"""
자연어 명령 + 100+ 지표 시스템 배포 알림
"""

import asyncio
import aiohttp
from enhanced_natural_language_alert import EnhancedNaturalLanguageAlert

async def deploy_enhanced_system():
    """향상된 시스템 배포 알림"""
    
    bot_token = "8333838666:AAECXOavuX4aaUI9bFttGTqzNY2vb_tqGJI"
    chat_id = "5373223115"
    
    # 시스템 인스턴스 생성
    enhanced_system = EnhancedNaturalLanguageAlert()
    
    # 지표 통계
    total_indicators = len(enhanced_system.extended_indicators)
    categories = enhanced_system.get_all_supported_indicators()
    
    message = f"""🚀 **시스템 대폭 업그레이드!**

## 🧠 자연어 명령 지원
✅ **기존**: `/set_alert fear_greed < 50 "알림"`
✅ **신규**: "공포지수가 50 이하로 떨어지면 알려줘"

## 📊 지표 수 대폭 확장
• **기존**: 19개 → **신규**: {total_indicators}개 ({total_indicators-19}개 추가)

**카테고리별 지표 수**:
• 기본 가격: {len(categories.get('기본 가격', []))}개
• 기술적 지표: {len(categories.get('기술적 지표', []))}개  
• 거래량: {len(categories.get('거래량', []))}개
• 온체인: {len(categories.get('온체인', []))}개
• 파생상품: {len(categories.get('파생상품', []))}개
• 감정지표: {len(categories.get('감정지표', []))}개
• 거시경제: {len(categories.get('거시경제', []))}개

## 💬 자연어 명령 예시
• "RSI가 70 넘으면 과매수 경고"
• "펀딩비가 마이너스로 가면 알림" 
• "고래활동이 80 초과하면 감지"
• "거래량이 2배 오르면"
• "달러지수가 105 넘으면"

## ⚡ 시스템 특징
• **두 방식 모두 지원**: 자연어 + 정확한 명령어
• **실시간 감지**: 기존 위험감지 + 맞춤알림 동시작동
• **1회성 알림**: 중복 방지
• **100+ 지표**: 온체인/기술/감정/거시경제 모두 포함

지금 바로 테스트해보세요! 👆"""
    
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
                    print("✅ 향상된 시스템 배포 알림 발송 성공")
                    
                    # 지표 가이드도 발송
                    guide = enhanced_system.format_indicator_guide()
                    await asyncio.sleep(2)  # 간격
                    
                    data2 = {
                        "chat_id": chat_id,
                        "text": guide,
                        "parse_mode": "Markdown"
                    }
                    
                    async with session.post(url, json=data2) as response2:
                        if response2.status == 200:
                            print("✅ 지표 가이드 발송 성공")
                        else:
                            print(f"❌ 가이드 발송 실패: {response2.status}")
                    
                    return True
                else:
                    error_text = await response.text()
                    print(f"❌ 발송 실패: {response.status}")
                    print(f"오류: {error_text}")
                    return False
                    
    except Exception as e:
        print(f"❌ 오류: {e}")
        return False

# 자연어 명령 테스트
async def test_natural_commands():
    """사용자 요청 명령들 테스트"""
    
    enhanced_system = EnhancedNaturalLanguageAlert()
    
    # 사용자가 원하는 자연어 명령들
    user_requests = [
        "공포지수가 50이하로 떨어지면 알람줘",  # 원래 요청
        "RSI가 70 넘으면 과매수 경고",
        "펀딩비가 마이너스가 되면 알림",
        "고래들이 대량매매 시작하면 감지",
        "거래량이 급증하면 알려줘",
        "달러지수가 강세면 알림"
    ]
    
    print("🧪 사용자 요청 명령어 테스트")
    
    for cmd in user_requests:
        result = enhanced_system.parse_natural_command(cmd)
        print(f"\n📝 '{cmd}'")
        
        if result and result.get("valid"):
            print(f"✅ 파싱 성공:")
            print(f"   지표: {result['indicator']}")
            print(f"   조건: {result['operator']} {result['threshold']}")
            print(f"   메시지: {result['message']}")
        else:
            print(f"❌ 파싱 실패: {result.get('error', '알 수 없는 오류')}")

async def main():
    print("1️⃣ 자연어 명령 테스트")
    await test_natural_commands()
    
    print("\n" + "="*50)
    print("2️⃣ 향상된 시스템 배포 알림")
    await deploy_enhanced_system()

if __name__ == "__main__":
    asyncio.run(main())