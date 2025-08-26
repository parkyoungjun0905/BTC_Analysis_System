#!/usr/bin/env python3
"""
텔레그램 알림 시스템
위험도별 맞춤형 메시지 생성 및 발송
"""

import asyncio
import aiohttp
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging
from config import TELEGRAM_CONFIG, NOTIFICATION_CONFIG
from custom_watchlist import CustomWatchlistManager

class TelegramNotifier:
    def __init__(self):
        self.bot_token = TELEGRAM_CONFIG["BOT_TOKEN"]
        self.chat_id = TELEGRAM_CONFIG["CHAT_ID"]
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}"
        self.logger = logging.getLogger(__name__)
        
        # 알림 쿨다운 관리
        self.last_alerts = {}
        self.alert_counts = {"CRITICAL": [], "WARNING": [], "INFO": []}
        
        # 커스텀 감시 관리자
        self.watchlist_manager = CustomWatchlistManager()
        self.last_update_id = 0  # 텔레그램 메시지 수신용
        
    async def send_risk_alert(self, risk_analysis: Dict, current_data: Dict) -> bool:
        """위험 분석 결과를 기반으로 알림 발송"""
        try:
            risk_level = risk_analysis.get("risk_level", "INFO")
            risk_score = risk_analysis.get("composite_risk_score", 0)
            
            # 쿨다운 체크
            if not self.should_send_alert(risk_level):
                self.logger.info(f"쿨다운으로 인한 {risk_level} 알림 스킵")
                return False
                
            # 시간당 알림 한도 체크  
            if not self.check_hourly_limit(risk_level):
                self.logger.warning(f"시간당 한도 초과로 {risk_level} 알림 스킵")
                return False
                
            # 메시지 생성
            message = self.generate_alert_message(risk_analysis, current_data)
            
            # 발송
            success = await self.send_message(message)
            
            if success:
                self.update_alert_tracking(risk_level)
                self.logger.info(f"{risk_level} 알림 발송 성공")
            else:
                self.logger.error(f"{risk_level} 알림 발송 실패")
                
            return success
            
        except Exception as e:
            self.logger.error(f"알림 발송 오류: {e}")
            return False

    def should_send_alert(self, risk_level: str) -> bool:
        """쿨다운 체크"""
        cooldown_minutes = NOTIFICATION_CONFIG["cooldown_minutes"].get(risk_level, 60)
        last_alert_time = self.last_alerts.get(risk_level)
        
        if last_alert_time is None:
            return True
            
        time_since_last = datetime.utcnow() - last_alert_time
        return time_since_last.total_seconds() >= cooldown_minutes * 60

    def check_hourly_limit(self, risk_level: str) -> bool:
        """시간당 알림 한도 체크"""
        max_alerts = NOTIFICATION_CONFIG["max_alerts_per_hour"].get(risk_level, 5)
        current_time = datetime.utcnow()
        one_hour_ago = current_time - timedelta(hours=1)
        
        # 지난 1시간 내 알림 개수 계산
        recent_alerts = [
            alert_time for alert_time in self.alert_counts[risk_level]
            if alert_time > one_hour_ago
        ]
        
        return len(recent_alerts) < max_alerts

    def update_alert_tracking(self, risk_level: str):
        """알림 발송 기록 업데이트"""
        current_time = datetime.utcnow()
        self.last_alerts[risk_level] = current_time
        self.alert_counts[risk_level].append(current_time)
        
        # 오래된 기록 정리 (24시간 이상)
        cutoff_time = current_time - timedelta(hours=24)
        self.alert_counts[risk_level] = [
            alert_time for alert_time in self.alert_counts[risk_level]
            if alert_time > cutoff_time
        ]

    def generate_alert_message(self, risk_analysis: Dict, current_data: Dict) -> str:
        """위험도별 맞춤형 알림 메시지 생성"""
        try:
            risk_level = risk_analysis.get("risk_level", "INFO")
            risk_score = risk_analysis.get("composite_risk_score", 0)
            confidence = risk_analysis.get("confidence", 0)
            timestamp = datetime.utcnow().strftime("%H:%M:%S")
            
            # 메시지 헤더
            headers = {
                "CRITICAL": "🚨 비트코인 긴급 위험 신호",
                "WARNING": "⚠️ 비트코인 주의 신호 감지", 
                "INFO": "📊 비트코인 참고 정보",
                "LOW": "✅ 비트코인 안정 상태"
            }
            
            message = f"{headers.get(risk_level, '📊 비트코인 알림')}\n\n"
            
            # 기본 정보
            if "price_data" in current_data:
                price_data = current_data["price_data"]
                current_price = price_data.get("current_price", 0)
                change_24h = price_data.get("change_24h", 0)
                volume_24h = price_data.get("volume_24h", 0)
                
                message += f"💰 현재가: ${current_price:,.0f}\n"
                message += f"📈 24시간 변동: {change_24h:+.2f}%\n"
                message += f"📊 거래량: ${volume_24h/1e9:.1f}B\n\n"
                
            # 위험 분석 결과
            message += f"🎯 위험도 분석:\n"
            message += f"├─ 종합 점수: {risk_score:.1f}/1.0\n"
            message += f"├─ 위험 레벨: {risk_level}\n"
            message += f"└─ 신뢰도: {confidence:.0%}\n\n"
            
            # 컴포넌트별 상세 분석 (CRITICAL, WARNING만)
            if risk_level in ["CRITICAL", "WARNING"] and "components" in risk_analysis:
                message += f"🔍 상세 분석:\n"
                components = risk_analysis["components"]
                
                for comp_name, comp_data in components.items():
                    score = comp_data.get("composite_score", 0)
                    if score > 0.3:  # 중요한 컴포넌트만 표시
                        comp_display_names = {
                            "sudden_change": "급변 감지",
                            "pattern_match": "패턴 매칭",
                            "anomaly": "이상 감지", 
                            "trend_change": "추세 변화",
                            "correlation": "상관관계"
                        }
                        display_name = comp_display_names.get(comp_name, comp_name)
                        message += f"├─ {display_name}: {score:.2f}\n"
                        
                message += "\n"
                
            # 거시경제 상황 (있는 경우)
            if "macro_data" in current_data and risk_level in ["CRITICAL", "WARNING"]:
                macro_data = current_data["macro_data"]
                message += f"🌍 거시경제:\n"
                
                if "vix" in macro_data:
                    vix_current = macro_data["vix"]["current"]
                    vix_change = macro_data["vix"]["change"]
                    message += f"├─ VIX: {vix_current:.1f} ({vix_change:+.1f})\n"
                    
                if "dxy" in macro_data:
                    dxy_current = macro_data["dxy"]["current"] 
                    dxy_change = macro_data["dxy"]["change"]
                    message += f"└─ DXY: {dxy_current:.2f} ({dxy_change:+.2f}%)\n"
                    
                message += "\n"
                
            # 센티먼트 (있는 경우)
            if "sentiment_data" in current_data and "fear_greed" in current_data["sentiment_data"]:
                fg_data = current_data["sentiment_data"]["fear_greed"]
                fg_index = fg_data["current_index"]
                fg_classification = fg_data["classification"]
                message += f"🌡️ 공포탐욕지수: {fg_index} ({fg_classification})\n\n"
                
            # 권장사항
            if "recommendations" in risk_analysis and risk_analysis["recommendations"]:
                message += f"💡 권장사항:\n"
                for i, rec in enumerate(risk_analysis["recommendations"][:3], 1):  # 최대 3개
                    message += f"{i}. {rec}\n"
                message += "\n"
                
            # 다음 체크 시간
            if "next_check_in" in risk_analysis:
                try:
                    next_check = datetime.fromisoformat(risk_analysis["next_check_in"].replace('Z', '+00:00'))
                    korea_time = next_check + timedelta(hours=9)  # UTC -> KST
                    message += f"⏰ 다음 점검: {korea_time.strftime('%H:%M')}\n"
                except:
                    pass
                    
            # 푸터
            message += f"📅 {timestamp} | 자동 분석"
            
            # 메시지 길이 제한
            max_length = TELEGRAM_CONFIG["MAX_MESSAGE_LENGTH"]
            if len(message) > max_length:
                message = message[:max_length-50] + "\n\n... (메시지가 너무 길어 생략됨)"
                
            return message
            
        except Exception as e:
            self.logger.error(f"메시지 생성 실패: {e}")
            return f"🤖 분석 완료 ({timestamp})\n위험도: {risk_score:.2f}/1.0\n레벨: {risk_level}"

    async def send_message(self, message: str) -> bool:
        """텔레그램 메시지 발송"""
        try:
            url = f"{self.base_url}/sendMessage"
            
            payload = {
                "chat_id": self.chat_id,
                "text": message,
                "parse_mode": "HTML",
                "disable_web_page_preview": True
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as response:
                    if response.status == 200:
                        return True
                    else:
                        error_text = await response.text()
                        self.logger.error(f"텔레그램 API 오류 {response.status}: {error_text}")
                        return False
                        
        except Exception as e:
            self.logger.error(f"메시지 발송 실패: {e}")
            return False

    async def send_test_message(self) -> bool:
        """테스트 메시지 발송"""
        test_message = (
            "🧪 Azure BTC 위험 감지 시스템 테스트\n\n"
            "✅ 연결 상태: 정상\n"
            f"📅 테스트 시간: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC\n"
            "🤖 시스템이 정상적으로 작동 중입니다."
        )
        
        return await self.send_message(test_message)

    async def send_system_start_notification(self) -> bool:
        """시스템 시작 알림"""
        message = (
            "🚀 Azure BTC 위험 감지 시스템 시작\n\n"
            "✅ 24시간 모니터링 활성화\n"
            "📊 1,827개 지표 실시간 분석\n"
            "🎯 위험 감지 시 즉시 알림\n\n"
            f"📅 시작 시간: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC\n"
            "🔔 알림 설정: 활성화됨"
        )
        
        return await self.send_message(message)

    async def send_error_notification(self, error_message: str) -> bool:
        """시스템 오류 알림"""
        message = (
            "❌ 시스템 오류 발생\n\n"
            f"🔍 오류 내용: {error_message}\n"
            f"📅 발생 시간: {datetime.utcnow().strftime('%H:%M:%S')}\n\n"
            "🔄 자동 복구 시도 중...\n"
            "⚠️ 지속되면 수동 점검 필요"
        )
        
        return await self.send_message(message)

    def generate_summary_report(self, daily_stats: Dict) -> str:
        """일일 요약 보고서 생성"""
        try:
            message = "📈 일일 BTC 모니터링 요약\n\n"
            
            # 기본 통계
            if "alerts_sent" in daily_stats:
                alerts = daily_stats["alerts_sent"]
                message += f"🚨 발송 알림:\n"
                message += f"├─ 긴급: {alerts.get('CRITICAL', 0)}건\n"
                message += f"├─ 경고: {alerts.get('WARNING', 0)}건\n"
                message += f"└─ 정보: {alerts.get('INFO', 0)}건\n\n"
                
            # 최고/최저 위험도
            if "risk_stats" in daily_stats:
                risk_stats = daily_stats["risk_stats"]
                message += f"📊 위험도 통계:\n"
                message += f"├─ 최고: {risk_stats.get('max_risk', 0):.2f}\n"
                message += f"├─ 평균: {risk_stats.get('avg_risk', 0):.2f}\n"
                message += f"└─ 최저: {risk_stats.get('min_risk', 0):.2f}\n\n"
                
            # 시스템 상태
            if "system_stats" in daily_stats:
                sys_stats = daily_stats["system_stats"]
                message += f"⚙️ 시스템 상태:\n"
                message += f"├─ 가동율: {sys_stats.get('uptime', 100):.1f}%\n"
                message += f"├─ API 성공률: {sys_stats.get('api_success_rate', 100):.1f}%\n"
                message += f"└─ 평균 응답시간: {sys_stats.get('avg_response_time', 0):.1f}초\n\n"
                
            message += f"📅 {datetime.utcnow().strftime('%Y-%m-%d')} 요약 완료"
            
            return message
            
        except Exception as e:
            self.logger.error(f"요약 보고서 생성 실패: {e}")
            return f"📈 일일 요약 (오류로 인한 간단 버전)\n📅 {datetime.utcnow().strftime('%Y-%m-%d')}"

    async def check_incoming_messages(self) -> List[str]:
        """텔레그램에서 새 메시지 확인"""
        try:
            url = f"{self.base_url}/getUpdates"
            params = {
                "offset": self.last_update_id + 1,
                "limit": 10,
                "timeout": 5
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get("ok") and data.get("result"):
                            processed_messages = []
                            
                            for update in data["result"]:
                                self.last_update_id = update["update_id"]
                                
                                if "message" in update:
                                    message = update["message"]
                                    # 우리 채팅방에서 온 메시지만 처리
                                    if str(message["chat"]["id"]) == str(self.chat_id):
                                        text = message.get("text", "")
                                        if text.strip():
                                            processed_messages.append(text.strip())
                            
                            return processed_messages
                    
            return []
            
        except Exception as e:
            self.logger.error(f"메시지 수신 오류: {e}")
            return []
    
    async def process_user_command(self, message: str) -> str:
        """사용자 명령 처리"""
        try:
            msg = message.strip()
            
            # 기본 명령어들
            if msg.lower() in ["/상태", "/status"]:
                return self._get_system_status()
            elif msg.lower() in ["/목록", "/list"]:
                return self.watchlist_manager.get_active_watchlists()
            elif msg.lower() in ["/도움말", "/help"]:
                return self._get_help_message()
            elif msg.startswith("/삭제 "):
                watchlist_id = msg.split(" ", 1)[1]
                return self.watchlist_manager.remove_watchlist(watchlist_id)
            
            # 감시 조건 추가 명령
            else:
                # 자연어로 된 감시 요청 파싱
                condition = self.watchlist_manager.parse_command(msg)
                if condition:
                    return self.watchlist_manager.add_watchlist(condition)
                else:
                    return ("❌ 명령을 이해할 수 없습니다.\n\n"
                           "💡 사용 예시:\n"
                           "• RSI 30 이하\n"
                           "• 5% 급락 10분\n"
                           "• 60000달러 돌파\n"
                           "• /도움말 - 전체 명령어 보기")
                           
        except Exception as e:
            self.logger.error(f"명령 처리 오류: {e}")
            return "❌ 명령 처리 중 오류가 발생했습니다."
    
    def _get_system_status(self) -> str:
        """시스템 상태 메시지"""
        active_count = len(self.watchlist_manager.active_watchlists)
        return (f"🤖 시스템 상태: 정상 운영 중\n"
                f"📊 기본 위험 감지: 활성화\n"
                f"🎯 개인 요청 조건: {active_count}개\n"
                f"📅 현재 시간: {datetime.utcnow().strftime('%H:%M:%S')} UTC\n\n"
                f"💡 새로운 조건 추가: 메시지로 요청하세요")
    
    def _get_help_message(self) -> str:
        """도움말 메시지"""
        return """🤖 BTC 위험 감지 시스템 도움말

📊 기본 기능:
• 24시간 자동 위험 감지 및 알림
• 1분마다 시장 상황 분석

🎯 개인 요청 기능:
• 원하는 조건을 메시지로 보내면 1회 알림
• 조건 달성 시 자동으로 삭제됨

💬 사용 예시:
• "RSI 30 이하" - RSI가 30 이하 되면 알림
• "5% 급락 10분" - 10분 내 5% 급락 시 알림  
• "60000달러 돌파" - 6만달러 돌파 시 알림
• "거래량 3배 증가" - 거래량 급증 시 알림

📋 관리 명령어:
• /상태 - 시스템 상태 확인
• /목록 - 현재 설정된 조건들 보기
• /삭제 W001 - 특정 조건 삭제
• /도움말 - 이 메시지 보기

⚠️ 주의: 모든 조건은 1회만 알림 후 자동 삭제됩니다."""

    async def check_custom_alerts(self, current_data: Dict) -> List[str]:
        """개인 요청 조건들 체크하고 알림 메시지 반환"""
        try:
            # 만료된 조건들 정리
            self.watchlist_manager.clear_expired_watchlists()
            
            # 조건 체크
            triggered_alerts = await self.watchlist_manager.check_conditions(current_data)
            
            # 알림 메시지 생성
            alert_messages = []
            for alert in triggered_alerts:
                alert_messages.append(alert['message'])
                self.logger.info(f"개인 요청 알림 발송: {alert['id']}")
            
            return alert_messages
            
        except Exception as e:
            self.logger.error(f"개인 요청 체크 오류: {e}")
            return []

# 테스트 함수
async def test_telegram_notifier():
    """텔레그램 알리미 테스트"""
    print("📱 텔레그램 알리미 테스트 시작...")
    
    notifier = TelegramNotifier()
    
    # 테스트 메시지 발송
    print("  테스트 메시지 발송 중...")
    test_success = await notifier.send_test_message()
    print(f"  테스트 메시지: {'✅ 성공' if test_success else '❌ 실패'}")
    
    # 가짜 위험 분석 데이터로 알림 테스트
    fake_risk_analysis = {
        "composite_risk_score": 0.75,
        "risk_level": "WARNING",
        "confidence": 0.82,
        "components": {
            "sudden_change": {"composite_score": 0.6},
            "pattern_match": {"composite_score": 0.8}, 
            "anomaly": {"composite_score": 0.4}
        },
        "recommendations": [
            "포지션 관리 점검 권장",
            "시장 변화 주의 깊게 모니터링",
            "1시간 후 재평가"
        ],
        "next_check_in": (datetime.utcnow() + timedelta(minutes=30)).isoformat()
    }
    
    fake_current_data = {
        "price_data": {
            "current_price": 58500,
            "change_24h": -6.8,
            "volume_24h": 28500000000
        },
        "macro_data": {
            "vix": {"current": 26.3, "change": 3.2},
            "dxy": {"current": 102.8, "change": 0.6}
        },
        "sentiment_data": {
            "fear_greed": {"current_index": 35, "classification": "Fear"}
        }
    }
    
    print("  위험 알림 메시지 발송 중...")
    alert_success = await notifier.send_risk_alert(fake_risk_analysis, fake_current_data)
    print(f"  위험 알림: {'✅ 성공' if alert_success else '❌ 실패'}")
    
    return test_success and alert_success

if __name__ == "__main__":
    # 직접 실행 시 테스트
    asyncio.run(test_telegram_notifier())