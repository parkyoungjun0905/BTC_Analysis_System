#!/usr/bin/env python3
"""
자연어 명령 통합 테스트
"""

import asyncio
from telegram_command_handler import TelegramCommandHandler

async def test_natural_integration():
    """자연어 통합 테스트"""
    
    bot_token = "8333838666:AAECXOavuX4aaUI9bFttGTqzNY2vb_tqGJI"
    chat_id = "5373223115"
    user_id = "5373223115"
    
    handler = TelegramCommandHandler(bot_token, chat_id)
    
    print("🧠 자연어 명령 통합 테스트")
    
    # 사용자 원본 요청
    natural_commands = [
        "RSI가 80 넘으면 과매수 경고해줘",
        "펀딩비가 0.02 초과하면 알림",
        "거래량이 100 넘으면 급증알림"
    ]
    
    for cmd in natural_commands:
        print(f"\n📝 테스트: '{cmd}'")
        
        # 자연어 처리
        result = await handler._handle_natural_language(user_id, cmd)
        
        print(f"📊 결과: {result['type']}")
        print(f"💬 메시지: {result['message'][:200]}..." if len(result['message']) > 200 else result['message'])
        
        # 성공시 텔레그램 발송
        if result['type'] == 'success':
            success = await handler.send_telegram_message(result['message'])
            print(f"📤 텔레그램 발송: {'✅ 성공' if success else '❌ 실패'}")
        
        await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(test_natural_integration())