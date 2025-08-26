#!/usr/bin/env python3
"""
사용자 맞춤형 알림 시스템
텔레그램 명령어로 개별 지표 조건 설정 및 1회성 알림
"""

import sqlite3
import json
import re
import asyncio
import aiohttp
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import logging

class CustomAlertSystem:
    """사용자 맞춤형 알림 관리 시스템"""
    
    def __init__(self, db_path: str = "custom_alerts.db"):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self.init_database()
        
        # 지원 지표 목록
        self.supported_indicators = {
            # 온체인 지표
            "mempool_pressure": "멤풀 압력",
            "funding_rate": "펀딩비", 
            "orderbook_imbalance": "호가창 불균형",
            "options_put_call": "옵션 PUT/CALL",
            "stablecoin_flows": "스테이블코인 유출입",
            "exchange_flows": "거래소 유출입",
            "whale_activity": "고래 활동",
            "miner_flows": "마이너 유출입",
            
            # 기술적 지표
            "rsi": "RSI",
            "macd": "MACD",
            "bollinger_upper": "볼린저밴드 상단",
            "bollinger_lower": "볼린저밴드 하단",
            "sma_20": "20일 이동평균",
            "ema_12": "12일 지수이동평균",
            "atr": "ATR (변동성)",
            "volume_sma": "거래량 이평",
            
            # 감정 지표  
            "fear_greed": "공포탐욕지수",
            "social_volume": "소셜 볼륨",
            "news_sentiment": "뉴스 감정",
            
            # 기타
            "btc_price": "BTC 가격",
            "volume_24h": "24시간 거래량"
        }
        
        # 조건 연산자
        self.operators = {
            ">": "초과",
            "<": "미만", 
            ">=": "이상",
            "<=": "이하",
            "=": "같음",
            "!=": "다름"
        }
    
    def init_database(self):
        """알림 규칙 데이터베이스 초기화"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS custom_alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                indicator_name TEXT NOT NULL,
                operator TEXT NOT NULL,
                threshold_value REAL NOT NULL,
                alert_message TEXT NOT NULL,
                is_active BOOLEAN DEFAULT TRUE,
                created_at TEXT NOT NULL,
                triggered_at TEXT,
                is_triggered BOOLEAN DEFAULT FALSE,
                
                -- 추가 설정
                priority TEXT DEFAULT 'MEDIUM',
                repeat_after_hours INTEGER DEFAULT 0,  -- 0이면 1회성
                last_value REAL,
                trigger_count INTEGER DEFAULT 0
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def parse_alert_command(self, message: str) -> Optional[Dict]:
        """텔레그램 알림 명령어 파싱"""
        try:
            # 명령어 패턴: /set_alert RSI > 70 "RSI 과매수"
            # 또는: /set_alert funding_rate < -0.01 펀딩비마이너스
            
            # 기본 패턴 매칭
            pattern = r'/set_alert\s+(\w+)\s*([><=!]+)\s*([\d\.-]+)\s*["\']?([^"\']*)["\']?'
            match = re.match(pattern, message.strip())
            
            if not match:
                return None
            
            indicator = match.group(1).lower()
            operator = match.group(2)
            threshold = float(match.group(3))
            alert_msg = match.group(4).strip()
            
            # 지표명 검증
            if indicator not in self.supported_indicators:
                return {
                    "error": f"지원하지 않는 지표입니다.\n사용 가능: {', '.join(list(self.supported_indicators.keys())[:10])}..."
                }
            
            # 연산자 검증
            if operator not in self.operators:
                return {
                    "error": f"지원하지 않는 연산자입니다.\n사용 가능: {', '.join(self.operators.keys())}"
                }
            
            # 기본 메시지 생성
            if not alert_msg:
                indicator_kr = self.supported_indicators[indicator]
                operator_kr = self.operators[operator]
                alert_msg = f"{indicator_kr}이(가) {threshold} {operator_kr}"
            
            return {
                "indicator": indicator,
                "operator": operator, 
                "threshold": threshold,
                "message": alert_msg,
                "valid": True
            }
            
        except Exception as e:
            self.logger.error(f"명령어 파싱 오류: {e}")
            return {"error": f"명령어 형식이 잘못되었습니다.\n예시: /set_alert RSI > 70 \"RSI 과매수\""}
    
    def add_custom_alert(self, user_id: str, parsed_command: Dict) -> Dict:
        """사용자 맞춤 알림 추가"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 중복 체크
            cursor.execute('''
                SELECT COUNT(*) FROM custom_alerts 
                WHERE user_id = ? AND indicator_name = ? AND operator = ? 
                AND threshold_value = ? AND is_active = TRUE
            ''', (user_id, parsed_command["indicator"], parsed_command["operator"], parsed_command["threshold"]))
            
            if cursor.fetchone()[0] > 0:
                conn.close()
                return {"success": False, "message": "동일한 알림 조건이 이미 존재합니다."}
            
            # 새 알림 추가
            cursor.execute('''
                INSERT INTO custom_alerts 
                (user_id, indicator_name, operator, threshold_value, alert_message, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                user_id,
                parsed_command["indicator"],
                parsed_command["operator"], 
                parsed_command["threshold"],
                parsed_command["message"],
                datetime.now().isoformat()
            ))
            
            alert_id = cursor.lastrowid
            conn.commit()
            conn.close()
            
            indicator_kr = self.supported_indicators[parsed_command["indicator"]]
            operator_kr = self.operators[parsed_command["operator"]]
            
            return {
                "success": True,
                "alert_id": alert_id,
                "message": f"✅ 알림 설정 완료!\n\n"
                          f"🎯 **조건**: {indicator_kr} {operator_kr} {parsed_command['threshold']}\n"
                          f"📱 **메시지**: {parsed_command['message']}\n"
                          f"🔢 **ID**: {alert_id}\n\n"
                          f"조건 만족시 1회 알림을 보내드립니다."
            }
            
        except Exception as e:
            self.logger.error(f"알림 추가 오류: {e}")
            return {"success": False, "message": f"알림 설정 실패: {str(e)}"}
    
    def get_user_alerts(self, user_id: str) -> List[Dict]:
        """사용자 알림 목록 조회"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT id, indicator_name, operator, threshold_value, alert_message, 
                       is_active, is_triggered, created_at, triggered_at
                FROM custom_alerts 
                WHERE user_id = ? 
                ORDER BY created_at DESC
            ''', (user_id,))
            
            alerts = []
            for row in cursor.fetchall():
                indicator_kr = self.supported_indicators.get(row[1], row[1])
                operator_kr = self.operators.get(row[2], row[2])
                
                alerts.append({
                    "id": row[0],
                    "indicator": row[1],
                    "indicator_kr": indicator_kr,
                    "operator": row[2],
                    "operator_kr": operator_kr,
                    "threshold": row[3],
                    "message": row[4],
                    "is_active": row[5],
                    "is_triggered": row[6],
                    "created_at": row[7],
                    "triggered_at": row[8]
                })
            
            conn.close()
            return alerts
            
        except Exception as e:
            self.logger.error(f"알림 목록 조회 오류: {e}")
            return []
    
    def remove_alert(self, user_id: str, alert_id: int) -> Dict:
        """알림 삭제"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 사용자 소유 확인
            cursor.execute('''
                SELECT alert_message FROM custom_alerts 
                WHERE id = ? AND user_id = ?
            ''', (alert_id, user_id))
            
            result = cursor.fetchone()
            if not result:
                conn.close()
                return {"success": False, "message": "해당 알림을 찾을 수 없거나 권한이 없습니다."}
            
            # 알림 삭제
            cursor.execute('''
                DELETE FROM custom_alerts WHERE id = ? AND user_id = ?
            ''', (alert_id, user_id))
            
            conn.commit()
            conn.close()
            
            return {
                "success": True,
                "message": f"✅ 알림 #{alert_id} 삭제 완료\n📝 {result[0]}"
            }
            
        except Exception as e:
            self.logger.error(f"알림 삭제 오류: {e}")
            return {"success": False, "message": f"알림 삭제 실패: {str(e)}"}
    
    async def check_custom_alerts(self, current_indicators: Dict, user_id: str) -> List[Dict]:
        """사용자 맞춤 알림 조건 체크"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 활성화된 미발송 알림들 조회
            cursor.execute('''
                SELECT id, indicator_name, operator, threshold_value, alert_message
                FROM custom_alerts 
                WHERE user_id = ? AND is_active = TRUE AND is_triggered = FALSE
            ''', (user_id,))
            
            active_alerts = cursor.fetchall()
            triggered_alerts = []
            
            for alert in active_alerts:
                alert_id, indicator, operator, threshold, message = alert
                
                # 현재 지표값 추출
                current_value = self._extract_indicator_value(current_indicators, indicator)
                
                if current_value is None:
                    continue
                
                # 조건 체크
                condition_met = self._evaluate_condition(current_value, operator, threshold)
                
                if condition_met:
                    # 알림 트리거 상태 업데이트
                    cursor.execute('''
                        UPDATE custom_alerts 
                        SET is_triggered = TRUE, triggered_at = ?, last_value = ?, trigger_count = trigger_count + 1
                        WHERE id = ?
                    ''', (datetime.now().isoformat(), current_value, alert_id))
                    
                    # 트리거된 알림 정보
                    triggered_alerts.append({
                        "id": alert_id,
                        "indicator": indicator,
                        "indicator_kr": self.supported_indicators.get(indicator, indicator),
                        "operator": operator,
                        "threshold": threshold,
                        "current_value": current_value,
                        "message": message
                    })
            
            conn.commit()
            conn.close()
            
            return triggered_alerts
            
        except Exception as e:
            self.logger.error(f"맞춤 알림 체크 오류: {e}")
            return []
    
    def _extract_indicator_value(self, indicators: Dict, indicator_name: str) -> Optional[float]:
        """지표 데이터에서 특정 값 추출"""
        try:
            # 19개 지표 시스템에서 값 추출
            enhanced_19 = indicators.get("enhanced_19_system", {})
            detailed_analysis = enhanced_19.get("detailed_analysis", {})
            
            # 직접 매칭
            if indicator_name in detailed_analysis:
                data = detailed_analysis[indicator_name]
                if isinstance(data, dict):
                    return data.get("current_value", data.get("value"))
                return float(data) if data is not None else None
            
            # 특별 처리
            if indicator_name == "btc_price":
                return indicators.get("metadata", {}).get("current_price")
            elif indicator_name == "fear_greed":
                return detailed_analysis.get("fear_greed", {}).get("current_value")
            elif indicator_name == "rsi":
                return detailed_analysis.get("price_momentum", {}).get("rsi_14")
            elif indicator_name == "funding_rate":
                return detailed_analysis.get("funding_rate", {}).get("current_value")
            
            # 추가 free 지표에서 검색
            additional_free = indicators.get("additional_free", {})
            if indicator_name in additional_free:
                data = additional_free[indicator_name]
                if isinstance(data, dict):
                    return data.get("current_value", data.get("value"))
                return float(data) if data is not None else None
            
            return None
            
        except Exception as e:
            self.logger.error(f"지표값 추출 오류 ({indicator_name}): {e}")
            return None
    
    def _evaluate_condition(self, current_value: float, operator: str, threshold: float) -> bool:
        """조건 평가"""
        try:
            if operator == ">":
                return current_value > threshold
            elif operator == "<":
                return current_value < threshold
            elif operator == ">=":
                return current_value >= threshold
            elif operator == "<=":
                return current_value <= threshold
            elif operator == "=":
                return abs(current_value - threshold) < 0.0001  # 부동소수점 비교
            elif operator == "!=":
                return abs(current_value - threshold) >= 0.0001
            
            return False
            
        except Exception:
            return False
    
    def format_triggered_alert(self, alert_data: Dict) -> str:
        """트리거된 알림 메시지 포맷"""
        try:
            indicator_kr = alert_data["indicator_kr"]
            current_value = alert_data["current_value"]
            threshold = alert_data["threshold"]
            operator_kr = self.operators.get(alert_data["operator"], alert_data["operator"])
            
            message = f"🚨 **맞춤 알림 발생!**\n\n"
            message += f"📊 **{indicator_kr}**: {current_value:.4f}\n"
            message += f"🎯 **조건**: {operator_kr} {threshold}\n"
            message += f"💬 **메시지**: {alert_data['message']}\n\n"
            message += f"🔢 **알림ID**: #{alert_data['id']}\n"
            message += f"⏰ **시간**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            message += f"✅ 이 알림은 1회 발송되며 자동으로 비활성화됩니다."
            
            return message
            
        except Exception as e:
            self.logger.error(f"알림 메시지 포맷 오류: {e}")
            return f"🚨 알림 #{alert_data.get('id', '?')}: {alert_data.get('message', '조건 만족')}"
    
    def format_help_message(self) -> str:
        """도움말 메시지"""
        return """🔧 **맞춤 알림 시스템 사용법**

📝 **알림 설정**
`/set_alert RSI > 70 "RSI 과매수 경고"`
`/set_alert funding_rate < -0.01 "펀딩비 마이너스"`
`/set_alert btc_price >= 50000 "5만달러 돌파"`

📋 **알림 관리**  
`/list_alerts` - 설정된 알림 목록
`/remove_alert 3` - 알림 #3 삭제
`/help_alerts` - 이 도움말

🎯 **지원 지표 (일부)**
• btc_price - BTC 가격
• rsi - RSI 지표  
• funding_rate - 펀딩비
• fear_greed - 공포탐욕지수
• mempool_pressure - 멤풀 압력
• whale_activity - 고래 활동

⚡ **연산자**
`>` `<` `>=` `<=` `=` `!=`

💡 **특징**
• 조건 만족시 **1회만** 알림
• 자동으로 비활성화
• 실시간 모니터링"""

# 테스트 함수
async def test_custom_alert_system():
    """맞춤 알림 시스템 테스트"""
    print("🧪 맞춤 알림 시스템 테스트...")
    
    system = CustomAlertSystem()
    user_id = "test_user"
    
    # 명령어 파싱 테스트
    test_commands = [
        "/set_alert RSI > 70 RSI과매수경고",
        "/set_alert funding_rate < -0.01 펀딩비마이너스", 
        "/set_alert btc_price >= 50000 \"5만달러 돌파\""
    ]
    
    for cmd in test_commands:
        parsed = system.parse_alert_command(cmd)
        print(f"명령어: {cmd}")
        print(f"파싱 결과: {parsed}")
        
        if parsed and parsed.get("valid"):
            result = system.add_custom_alert(user_id, parsed)
            print(f"추가 결과: {result}")
        print("-" * 50)
    
    # 알림 목록 조회
    alerts = system.get_user_alerts(user_id)
    print(f"설정된 알림: {len(alerts)}개")
    
    for alert in alerts:
        print(f"  - {alert['indicator_kr']} {alert['operator_kr']} {alert['threshold']}")

if __name__ == "__main__":
    asyncio.run(test_custom_alert_system())