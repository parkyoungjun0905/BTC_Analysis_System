#!/usr/bin/env python3
"""
최종 시스템 현황 정리 및 사용자 안내
"""

import asyncio
import aiohttp
import sqlite3
from custom_alert_system import CustomAlertSystem
from enhanced_natural_language_alert import EnhancedNaturalLanguageAlert

async def send_final_status():
    """최종 시스템 현황 발송"""
    
    bot_token = "8333838666:AAECXOavuX4aaUI9bFttGTqzNY2vb_tqGJI"
    chat_id = "5373223115"
    user_id = "5373223115"
    
    # 시스템 정보 수집
    alert_system = CustomAlertSystem()
    enhanced_system = EnhancedNaturalLanguageAlert()
    
    # 현재 설정된 알림
    current_alerts = alert_system.get_user_alerts(user_id)
    
    # 지표 통계
    total_indicators = len(enhanced_system.extended_indicators)
    categories = enhanced_system.get_all_supported_indicators()
    
    message = f"""🎯 **BTC 분석 시스템 - 최종 완성!**

## 🔄 **동시 작동 시스템** (2개)

### 1️⃣ **실시간 위험 감지** (AI 예측)
• **19개 지표** 종합 분석
• **Claude AI + 시계열** 앙상블 예측
• **자동 위험 알림** (높은 신뢰도시)
• **5분마다** 실행 (중요시간)

### 2️⃣ **맞춤 알림 시스템** (사용자 요청)
• **자연어 명령** 지원 ✨
• **{total_indicators}개 지표** 감시 가능
• **1회성 알림** (중복 방지)
• **실시간 조건 체크**

## 🧠 **자연어 명령 예시**

✅ **성공한 명령들**:
• "공포지수가 50 이하로 떨어지면 알려줘"
• "RSI가 80 넘으면 과매수 경고해줘" 
• "펀딩비가 0.02 초과하면 알림"
• "거래량이 100 넘으면 급증알림"

📋 **정확한 명령어도 가능**:
• `/set_alert fear_greed < 30 "극공포 감지"`
• `/list_alerts` - 알림 목록
• `/help_alerts` - 도움말

## 📊 **현재 상태**

**설정된 맞춤 알림**: {len(current_alerts)}개"""
    
    if current_alerts:
        message += "\n"
        for alert in current_alerts[:3]:  # 최대 3개만 표시
            status = "✅ 대기중" if alert['is_active'] and not alert['is_triggered'] else "🔕 발송완료"
            message += f"• {alert['indicator']} {alert['operator']} {alert['threshold']} - {status}\n"
        
        if len(current_alerts) > 3:
            message += f"• ... 외 {len(current_alerts) - 3}개\n"
    
    message += f"""
## 🎯 **지표 카테고리** ({total_indicators}개)

• **기본 가격**: {len(categories.get('기본 가격', []))}개
• **기술적 지표**: {len(categories.get('기술적 지표', []))}개
• **거래량**: {len(categories.get('거래량', []))}개  
• **온체인**: {len(categories.get('온체인', []))}개
• **파생상품**: {len(categories.get('파생상품', []))}개
• **감정지표**: {len(categories.get('감정지표', []))}개
• **거시경제**: {len(categories.get('거시경제', []))}개

## ⚡ **시스템 특징**

✅ **이중 보호**: AI 예측 + 맞춤 감시
✅ **자연어 지원**: 편리한 명령 입력
✅ **100+ 지표**: 모든 시장 영역 커버
✅ **1회성 알림**: 스팸 방지
✅ **24/7 감시**: Azure Functions 자동 실행

🚀 **이제 원하는 조건을 자연스럽게 말해보세요!**
예: "달러지수가 105 넘으면 알려줘" """
    
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
                    print("✅ 최종 시스템 현황 발송 성공")
                    return True
                else:
                    error_text = await response.text()
                    print(f"❌ 발송 실패: {response.status}")
                    print(f"오류: {error_text}")
                    return False
                    
    except Exception as e:
        print(f"❌ 오류: {e}")
        return False

def show_local_status():
    """로컬 시스템 상태"""
    
    print("🎯 BTC 분석 시스템 - 최종 완성 현황")
    print("=" * 50)
    
    # 알림 DB 상태
    try:
        conn = sqlite3.connect('custom_alerts.db')
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM custom_alerts')
        total_alerts = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM custom_alerts WHERE is_active = 1')
        active_alerts = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM custom_alerts WHERE is_triggered = 1')
        triggered_alerts = cursor.fetchone()[0]
        
        conn.close()
        
        print(f"📊 알림 데이터베이스:")
        print(f"   • 전체 알림: {total_alerts}개")
        print(f"   • 활성 알림: {active_alerts}개")
        print(f"   • 발송완료: {triggered_alerts}개")
        
    except Exception as e:
        print(f"❌ DB 오류: {e}")
    
    # 지표 시스템 상태
    enhanced_system = EnhancedNaturalLanguageAlert()
    total_indicators = len(enhanced_system.extended_indicators)
    
    print(f"\n📈 지표 시스템:")
    print(f"   • 지원 지표: {total_indicators}개")
    print(f"   • 자연어 처리: ✅ 활성화")
    print(f"   • 카테고리: 7개 분야")
    
    print(f"\n🔄 동시 작동 시스템:")
    print(f"   1️⃣ AI 예측 시스템 (19개 지표)")
    print(f"   2️⃣ 맞춤 알림 시스템 ({total_indicators}개 지표)")
    
    print(f"\n✅ 완성된 기능:")
    print(f"   • 자연어 명령 처리")
    print(f"   • 100+ 지표 감시")
    print(f"   • 1회성 알림")
    print(f"   • Azure Functions 통합")
    print(f"   • 텔레그램 인터페이스")

async def main():
    print("1️⃣ 로컬 시스템 현황")
    show_local_status()
    
    print("\n" + "="*50)
    print("2️⃣ 텔레그램 최종 현황 발송")
    await send_final_status()

if __name__ == "__main__":
    asyncio.run(main())