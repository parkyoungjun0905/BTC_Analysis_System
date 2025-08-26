#!/usr/bin/env python3
"""
수동 텔레그램 명령어 처리기 (Azure Functions 문제 해결용)
"""

import asyncio
import aiohttp
from telegram_command_handler import TelegramCommandHandler
from custom_alert_system import CustomAlertSystem

async def process_manual_command():
    """수동으로 명령어 처리"""
    
    bot_token = "8333838666:AAECXOavuX4aaUI9bFttGTqzNY2vb_tqGJI"
    chat_id = "5373223115"
    user_id = "5373223115"  # 실제 사용자 ID
    
    # 수동으로 명령어 파싱 및 처리
    command = "/set_alert fear_greed < 50 \"공포지수 하락 알림\""
    
    alert_system = CustomAlertSystem()
    
    print(f"🔄 수동 명령어 처리: {command}")
    
    # 명령어 파싱
    parsed = alert_system.parse_alert_command(command)
    
    if parsed and "error" not in parsed:
        print(f"✅ 파싱 성공: {parsed}")
        
        # 알림 추가
        result = alert_system.add_custom_alert(user_id, parsed)
        print(f"📝 알림 추가 결과: {result}")
        
        if result["success"]:
            # 성공 메시지 전송
            message = f"""✅ **맞춤 알림 설정 완료**

🎯 **설정된 알림**:
• 지표: {parsed['indicator_kr']}
• 조건: {parsed['operator_kr']} {parsed['threshold']}  
• 메시지: {parsed['message']}

🔔 조건 달성시 1회 알림을 보내드립니다!

📋 다른 명령어:
• `/list_alerts` - 알림 목록
• `/help_alerts` - 도움말"""
            
            success = await send_telegram_message(bot_token, chat_id, message)
            if success:
                print("✅ 성공 메시지 발송됨")
            else:
                print("❌ 성공 메시지 발송 실패")
                
    else:
        print(f"❌ 파싱 실패: {parsed}")
        
        # 오류 안내 메시지
        error_message = """❌ **명령어 형식 오류**

올바른 형식:
`/set_alert fear_greed < 50 "공포지수 하락 알림"`

지원되는 지표:
• `fear_greed` - 공포탐욕지수
• `RSI` - RSI 지표  
• `funding_rate` - 펀딩비
• `whale_activity` - 고래 활동

다시 시도해주세요! 👆"""
        
        await send_telegram_message(bot_token, chat_id, error_message)

async def send_telegram_message(bot_token: str, chat_id: str, message: str) -> bool:
    """텔레그램 메시지 발송"""
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    
    try:
        async with aiohttp.ClientSession() as session:
            data = {
                "chat_id": chat_id,
                "text": message,
                "parse_mode": "Markdown"
            }
            
            async with session.post(url, json=data) as response:
                return response.status == 200
                
    except Exception as e:
        print(f"메시지 발송 오류: {e}")
        return False

async def show_current_alerts():
    """현재 설정된 알림 표시"""
    
    user_id = "5373223115"
    alert_system = CustomAlertSystem()
    
    alerts = alert_system.get_user_alerts(user_id)
    
    print(f"\n📋 현재 설정된 알림: {len(alerts)}개")
    
    for alert in alerts:
        print(f"  🔔 ID {alert['id']}: {alert['indicator_kr']} {alert['operator_kr']} {alert['threshold']}")
        print(f"     메시지: {alert['message']}")
        print(f"     활성: {alert['is_active']}, 발송됨: {alert['is_triggered']}")

async def main():
    print("1️⃣ 수동 명령어 처리")
    await process_manual_command()
    
    print("\n2️⃣ 현재 알림 상태")
    await show_current_alerts()

if __name__ == "__main__":
    asyncio.run(main())