#!/usr/bin/env python3
"""
텔레그램 봇 명령어 처리기
사용자 맞춤 알림 설정을 위한 텔레그램 인터페이스
"""

import asyncio
import aiohttp
import json
import os
from typing import Dict, List, Optional
import logging
from custom_alert_system import CustomAlertSystem
from enhanced_natural_language_alert import EnhancedNaturalLanguageAlert

class TelegramCommandHandler:
    """텔레그램 봇 명령어 처리기"""
    
    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.alert_system = CustomAlertSystem()
        self.enhanced_system = EnhancedNaturalLanguageAlert()  # 자연어 처리 추가
        self.logger = logging.getLogger(__name__)
        
        # 마지막 처리된 메시지 ID (중복 처리 방지)
        self.last_update_id = 0
    
    async def process_telegram_updates(self) -> List[Dict]:
        """텔레그램 업데이트 처리"""
        try:
            # 새 메시지 가져오기
            url = f"https://api.telegram.org/bot{self.bot_token}/getUpdates"
            params = {
                "offset": self.last_update_id + 1,
                "limit": 10,
                "timeout": 30
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=35)) as response:
                    if response.status != 200:
                        self.logger.error(f"텔레그램 API 오류: {response.status}")
                        return []
                    
                    data = await response.json()
                    
                    if not data.get("ok"):
                        self.logger.error(f"텔레그램 API 응답 오류: {data}")
                        return []
                    
                    updates = data.get("result", [])
                    processed_commands = []
                    
                    for update in updates:
                        # 업데이트 ID 갱신
                        if update["update_id"] > self.last_update_id:
                            self.last_update_id = update["update_id"]
                        
                        # 메시지 처리
                        if "message" in update:
                            command_result = await self._process_message(update["message"])
                            if command_result:
                                processed_commands.append(command_result)
                    
                    return processed_commands
                    
        except Exception as e:
            self.logger.error(f"텔레그램 업데이트 처리 오류: {e}")
            return []
    
    async def _process_message(self, message: Dict) -> Optional[Dict]:
        """개별 메시지 처리"""
        try:
            # 사용자 정보
            user_id = str(message.get("from", {}).get("id", "unknown"))
            chat_id = str(message.get("chat", {}).get("id", ""))
            text = message.get("text", "").strip()
            
            # 설정된 채팅방인지 확인
            if chat_id != self.chat_id:
                return None
            
            # 명령어 또는 자연어 처리
            if text.startswith("/"):
                # 정확한 명령어 처리
                pass
            elif any(keyword in text for keyword in ["알림", "알람", "감지", "경고", "알려"]):
                # 자연어 명령 처리
                return await self._handle_natural_language(user_id, text)
            else:
                return None
            
            self.logger.info(f"처리할 명령어: {text}")
            
            # 명령어별 처리
            if text.startswith("/set_alert"):
                return await self._handle_set_alert(user_id, text)
            elif text == "/list_alerts":
                return await self._handle_list_alerts(user_id)
            elif text.startswith("/remove_alert"):
                return await self._handle_remove_alert(user_id, text)
            elif text == "/help_alerts":
                return await self._handle_help_alerts()
            elif text.startswith("/clear_all"):
                return await self._handle_clear_all_alerts(user_id)
            else:
                return None  # 다른 명령어는 무시
                
        except Exception as e:
            self.logger.error(f"메시지 처리 오류: {e}")
            return None
    
    async def _handle_set_alert(self, user_id: str, command: str) -> Dict:
        """알림 설정 명령어 처리"""
        try:
            # 명령어 파싱
            parsed = self.alert_system.parse_alert_command(command)
            
            if not parsed:
                return {
                    "type": "error",
                    "message": "❌ 명령어 형식이 잘못되었습니다.\n\n"
                              "📝 **올바른 형식**:\n"
                              "`/set_alert RSI > 70 \"RSI 과매수\"`\n"
                              "`/set_alert funding_rate < -0.01 \"펀딩비 마이너스\"`\n\n"
                              "💡 `/help_alerts`로 자세한 도움말을 확인하세요."
                }
            
            if "error" in parsed:
                return {
                    "type": "error", 
                    "message": f"❌ {parsed['error']}"
                }
            
            # 알림 추가
            result = self.alert_system.add_custom_alert(user_id, parsed)
            
            return {
                "type": "success" if result["success"] else "error",
                "message": result["message"]
            }
            
        except Exception as e:
            self.logger.error(f"알림 설정 오류: {e}")
            return {
                "type": "error",
                "message": f"❌ 알림 설정 중 오류가 발생했습니다: {str(e)}"
            }
    
    async def _handle_list_alerts(self, user_id: str) -> Dict:
        """알림 목록 명령어 처리"""
        try:
            alerts = self.alert_system.get_user_alerts(user_id)
            
            if not alerts:
                return {
                    "type": "info",
                    "message": "📋 설정된 맞춤 알림이 없습니다.\n\n"
                              "💡 `/set_alert RSI > 70 \"과매수 경고\"`로 알림을 설정해보세요!"
                }
            
            message = "📋 **설정된 맞춤 알림 목록**\n\n"
            
            active_count = 0
            triggered_count = 0
            
            for i, alert in enumerate(alerts):
                status_emoji = "✅" if alert["is_active"] and not alert["is_triggered"] else "🔕"
                if alert["is_triggered"]:
                    status_emoji = "✅🔔"
                    triggered_count += 1
                elif alert["is_active"]:
                    active_count += 1
                
                message += f"{status_emoji} **#{alert['id']}** {alert['indicator_kr']} "
                message += f"{alert['operator_kr']} {alert['threshold']}\n"
                message += f"   💬 {alert['message']}\n"
                
                if alert['is_triggered'] and alert['triggered_at']:
                    triggered_time = alert['triggered_at'][:19].replace('T', ' ')
                    message += f"   🔔 발송됨: {triggered_time}\n"
                
                message += "\n"
            
            # 요약 정보
            message += f"📊 **요약**: 전체 {len(alerts)}개 (활성 {active_count}개, 발송완료 {triggered_count}개)\n\n"
            message += "🗑️ 알림 삭제: `/remove_alert [ID]`\n"
            message += "🧹 전체 삭제: `/clear_all_alerts`"
            
            return {
                "type": "info",
                "message": message
            }
            
        except Exception as e:
            self.logger.error(f"알림 목록 조회 오류: {e}")
            return {
                "type": "error",
                "message": f"❌ 알림 목록 조회 실패: {str(e)}"
            }
    
    async def _handle_remove_alert(self, user_id: str, command: str) -> Dict:
        """알림 삭제 명령어 처리"""
        try:
            # ID 추출
            parts = command.split()
            if len(parts) != 2:
                return {
                    "type": "error",
                    "message": "❌ 사용법: `/remove_alert [알림ID]`\n예시: `/remove_alert 3`"
                }
            
            try:
                alert_id = int(parts[1])
            except ValueError:
                return {
                    "type": "error", 
                    "message": "❌ 알림 ID는 숫자여야 합니다.\n예시: `/remove_alert 3`"
                }
            
            # 알림 삭제
            result = self.alert_system.remove_alert(user_id, alert_id)
            
            return {
                "type": "success" if result["success"] else "error",
                "message": result["message"]
            }
            
        except Exception as e:
            self.logger.error(f"알림 삭제 오류: {e}")
            return {
                "type": "error",
                "message": f"❌ 알림 삭제 실패: {str(e)}"
            }
    
    async def _handle_clear_all_alerts(self, user_id: str) -> Dict:
        """모든 알림 삭제"""
        try:
            import sqlite3
            
            conn = sqlite3.connect(self.alert_system.db_path)
            cursor = conn.cursor()
            
            # 사용자 알림 개수 확인
            cursor.execute('SELECT COUNT(*) FROM custom_alerts WHERE user_id = ?', (user_id,))
            count = cursor.fetchone()[0]
            
            if count == 0:
                conn.close()
                return {
                    "type": "info",
                    "message": "📋 삭제할 알림이 없습니다."
                }
            
            # 모든 알림 삭제
            cursor.execute('DELETE FROM custom_alerts WHERE user_id = ?', (user_id,))
            conn.commit()
            conn.close()
            
            return {
                "type": "success",
                "message": f"🧹 모든 맞춤 알림 삭제 완료! (삭제된 알림: {count}개)"
            }
            
        except Exception as e:
            self.logger.error(f"전체 알림 삭제 오류: {e}")
            return {
                "type": "error", 
                "message": f"❌ 전체 삭제 실패: {str(e)}"
            }
    
    async def _handle_natural_language(self, user_id: str, text: str) -> Dict:
        """자연어 명령 처리"""
        try:
            self.logger.info(f"자연어 처리: {text}")
            
            # 자연어 파싱
            parsed = self.enhanced_system.parse_natural_command(text)
            
            if not parsed or "error" in parsed:
                return {
                    "type": "error",
                    "message": f"❌ {parsed.get('error', '자연어 처리 실패')}\n\n" \
                              "💡 **예시**:\n" \
                              "• '공포지수가 50 이하로 떨어지면 알려줘'\n" \
                              "• 'RSI가 70 넘으면 과매수 경고'\n" \
                              "• '고래활동이 80 초과하면 감지'\n\n" \
                              "📋 정확한 명령어: `/set_alert fear_greed < 50 \"알림\"`"
                }
            
            # 알림 추가
            result = self.enhanced_system.add_custom_alert(user_id, parsed)
            
            if result["success"]:
                success_message = f"✅ **자연어 명령 처리 완료!**\n\n" \
                                 f"📝 **원본**: \"*{text}*\"\n" \
                                 f"🔄 **파싱**: `{parsed['indicator']} {parsed['operator']} {parsed['threshold']}`\n" \
                                 f"💬 **메시지**: {parsed['message']}\n\n" \
                                 f"🎯 이제 조건 달성시 1회 알림을 받으실 수 있습니다!"
                
                return {
                    "type": "success",
                    "message": success_message
                }
            else:
                return {
                    "type": "error",
                    "message": f"❌ {result['message']}"
                }
            
        except Exception as e:
            self.logger.error(f"자연어 처리 오류: {e}")
            return {
                "type": "error",
                "message": f"❌ 자연어 처리 중 오류: {str(e)}"
            }
    
    async def _handle_help_alerts(self) -> Dict:
        """도움말 명령어 처리"""
        return {
            "type": "info",
            "message": self.alert_system.format_help_message()
        }
    
    async def send_telegram_message(self, message: str, parse_mode: str = "Markdown") -> bool:
        """텔레그램 메시지 발송"""
        try:
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            
            data = {
                "chat_id": self.chat_id,
                "text": message,
                "parse_mode": parse_mode
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=data, timeout=aiohttp.ClientTimeout(total=30)) as response:
                    if response.status == 200:
                        return True
                    else:
                        error_text = await response.text()
                        self.logger.error(f"텔레그램 메시지 발송 실패: {response.status} - {error_text}")
                        return False
                        
        except Exception as e:
            self.logger.error(f"텔레그램 메시지 발송 오류: {e}")
            return False
    
    async def process_and_respond(self) -> int:
        """명령어 처리 및 응답 (배치 처리)"""
        try:
            commands = await self.process_telegram_updates()
            
            responses_sent = 0
            for command in commands:
                if command and command.get("message"):
                    success = await self.send_telegram_message(command["message"])
                    if success:
                        responses_sent += 1
                        
                    # 메시지 간 간격
                    await asyncio.sleep(0.5)
            
            return responses_sent
            
        except Exception as e:
            self.logger.error(f"명령어 처리 및 응답 오류: {e}")
            return 0

# 테스트 함수
async def test_telegram_handler():
    """텔레그램 명령어 핸들러 테스트"""
    print("🤖 텔레그램 명령어 핸들러 테스트...")
    
    # 환경변수에서 토큰 가져오기
    bot_token = os.environ.get('TELEGRAM_BOT_TOKEN')
    chat_id = os.environ.get('TELEGRAM_CHAT_ID')
    
    if not bot_token or not chat_id:
        print("❌ 텔레그램 토큰 또는 채팅 ID가 설정되지 않았습니다.")
        return
    
    handler = TelegramCommandHandler(bot_token, chat_id)
    
    # 도움말 메시지 발송
    help_message = handler.alert_system.format_help_message()
    success = await handler.send_telegram_message(
        f"🧪 **맞춤 알림 시스템 테스트**\n\n{help_message}"
    )
    
    if success:
        print("✅ 도움말 메시지 발송 성공")
    else:
        print("❌ 메시지 발송 실패")

if __name__ == "__main__":
    asyncio.run(test_telegram_handler())