"""
향상된 텔레그램 알림 시스템
초보자 친화적 설명 + 정확도 표시 + 상세 분석
"""

import asyncio
import aiohttp
import logging
from typing import Dict, Optional, List
from datetime import datetime
import os

from beginner_friendly_explainer import BeginnerFriendlyExplainer, AdvancedMetricsExplainer

logger = logging.getLogger(__name__)

class EnhancedTelegramNotifier:
    """향상된 텔레그램 알림 시스템"""
    
    def __init__(self):
        self.bot_token = os.environ.get('TELEGRAM_BOT_TOKEN', '')
        self.chat_id = os.environ.get('TELEGRAM_CHAT_ID', '')
        self.explainer = BeginnerFriendlyExplainer()
        self.advanced_explainer = AdvancedMetricsExplainer()
        self.logger = logger
        
        # 알림 우선순위별 이모지
        self.priority_emojis = {
            "CRITICAL": "🚨🚨🚨",
            "HIGH": "⚠️⚠️",
            "MEDIUM": "📊",
            "LOW": "📌"
        }
    
    async def send_prediction_alert(
        self, 
        prediction: Dict, 
        indicators: Dict,
        time_series_analysis: Dict,
        system_performance: Dict
    ) -> bool:
        """종합 예측 알림 발송"""
        try:
            # 우선순위 결정
            priority = self._determine_priority(prediction)
            
            # 메시지 구성
            message = self._build_comprehensive_message(
                prediction, 
                indicators, 
                time_series_analysis,
                system_performance,
                priority
            )
            
            # 텔레그램 발송
            success = await self._send_telegram_message(message)
            
            if success:
                self.logger.info(f"✅ {priority} 우선순위 알림 발송 성공")
            
            return success
            
        except Exception as e:
            self.logger.error(f"알림 발송 실패: {e}")
            return False
    
    def _determine_priority(self, prediction: Dict) -> str:
        """알림 우선순위 결정"""
        confidence = prediction.get("confidence", "LOW")
        probability = prediction.get("probability", 50)
        
        if confidence == "VERY_HIGH" and probability > 90:
            return "CRITICAL"
        elif confidence == "HIGH" and probability > 80:
            return "HIGH"
        elif confidence == "MEDIUM" and probability > 70:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _build_comprehensive_message(
        self,
        prediction: Dict,
        indicators: Dict,
        time_series: Dict,
        performance: Dict,
        priority: str
    ) -> str:
        """종합적인 알림 메시지 구성"""
        
        # 헤더
        header = self._build_header(priority, prediction)
        
        # 예측 요약
        prediction_summary = self._build_prediction_summary(prediction)
        
        # 핵심 근거 (초보자 친화적)
        key_reasons = self._build_key_reasons(indicators, time_series)
        
        # 기술적 상세 (선택적)
        technical_details = self._build_technical_details(indicators)
        
        # 시스템 성과
        system_stats = self._build_system_performance(performance)
        
        # 행동 가이드
        action_guide = self._build_action_guide(prediction, priority)
        
        # 리스크 경고
        risk_warning = self._build_risk_warning(prediction)
        
        # 메시지 조합
        message = f"""
{header}

{prediction_summary}

{key_reasons}

{technical_details}

{action_guide}

{risk_warning}

{system_stats}

⏰ 발송 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        return message
    
    def _build_header(self, priority: str, prediction: Dict) -> str:
        """메시지 헤더 구성"""
        emoji = self.priority_emojis[priority]
        direction = prediction.get("direction", "NEUTRAL")
        
        if priority == "CRITICAL":
            return f"{emoji} **긴급 알림** {emoji}\n💥 강력한 {direction} 신호 포착!"
        elif priority == "HIGH":
            return f"{emoji} **중요 알림** {emoji}\n📈 {direction} 신호 감지"
        else:
            return f"{emoji} **일반 알림**\n📊 시장 분석 업데이트"
    
    def _build_prediction_summary(self, prediction: Dict) -> str:
        """예측 요약 (초보자 친화적)"""
        direction = prediction.get("direction", "NEUTRAL")
        probability = prediction.get("probability", 50)
        target_price = prediction.get("target_price", 0)
        timeframe = prediction.get("timeframe", "24시간")
        confidence = prediction.get("confidence", "LOW")
        
        # 방향별 설명
        direction_explain = {
            "BULLISH": "📈 **상승** 예상",
            "BEARISH": "📉 **하락** 예상",
            "NEUTRAL": "➡️ **횡보** 예상"
        }
        
        # 신뢰도별 설명
        confidence_explain = {
            "VERY_HIGH": "매우 높은 신뢰도",
            "HIGH": "높은 신뢰도",
            "MEDIUM": "보통 신뢰도",
            "LOW": "낮은 신뢰도"
        }
        
        return f"""
━━━━━━━━━━━━━━━━━━━━━━━━━
📊 **예측 결과**
━━━━━━━━━━━━━━━━━━━━━━━━━

• 방향: {direction_explain.get(direction, direction)}
• 확률: **{probability}%**
• 목표가: **${target_price:,.0f}**
• 예상 시간: {timeframe} 이내
• 신뢰도: {confidence_explain.get(confidence, confidence)}
"""
    
    def _build_key_reasons(self, indicators: Dict, time_series: Dict) -> str:
        """핵심 근거 설명 (초보자 친화적)"""
        reasons = []
        
        # 가장 강한 신호 3개 추출
        composite = indicators.get("composite_analysis", {})
        key_signals = indicators.get("high_confidence_signals", [])[:3]
        
        for signal in key_signals:
            indicator_name = signal.get("indicator", "")
            strength = signal.get("strength", 0)
            
            # 초보자 친화적 설명 추가
            if "funding" in indicator_name.lower():
                reasons.append(f"• 펀딩비 {strength:.1%} - 선물 시장 과열/냉각")
            elif "whale" in indicator_name.lower():
                reasons.append(f"• 고래 활동 {strength:.1%} - 큰손들이 움직임")
            elif "exchange" in indicator_name.lower():
                reasons.append(f"• 거래소 플로우 {strength:.1%} - 매도/매수 압력")
            elif "fear" in indicator_name.lower():
                reasons.append(f"• 공포탐욕 {strength:.1%} - 시장 심리")
            else:
                reasons.append(f"• {indicator_name} {strength:.1%}")
        
        # 시계열 패턴
        if time_series and time_series.get("pattern_found"):
            pattern_confidence = time_series.get("confidence", 0)
            reasons.append(f"• 과거 유사 패턴 발견 ({pattern_confidence:.0f}% 일치)")
        
        return f"""
━━━━━━━━━━━━━━━━━━━━━━━━━
💡 **왜 이런 예측이 나왔나요?**
━━━━━━━━━━━━━━━━━━━━━━━━━

{chr(10).join(reasons) if reasons else "• 종합 지표 분석 중"}

💭 쉽게 말하면:
{self._generate_simple_explanation(indicators)}
"""
    
    def _generate_simple_explanation(self, indicators: Dict) -> str:
        """매우 쉬운 설명 생성"""
        composite = indicators.get("composite_analysis", {})
        signal = composite.get("overall_signal", "NEUTRAL")
        
        if "BULLISH" in signal:
            return """
지금 여러 신호들이 가격 상승을 가리키고 있어요.
마치 여러 사람이 동시에 "오를 것 같아!"라고 말하는 것과 비슷해요.
"""
        elif "BEARISH" in signal:
            return """
지금 여러 신호들이 가격 하락을 가리키고 있어요.
시장에 매도 압력이 증가하고 있다는 뜻이에요.
"""
        else:
            return """
지금은 명확한 방향이 없어요.
좀 더 지켜봐야 할 시점입니다.
"""
    
    def _build_technical_details(self, indicators: Dict) -> str:
        """기술적 상세 정보 (접을 수 있게)"""
        composite = indicators.get("composite_analysis", {})
        
        return f"""
━━━━━━━━━━━━━━━━━━━━━━━━━
🔬 **상세 분석** (고급)
━━━━━━━━━━━━━━━━━━━━━━━━━

• 종합 신호: {composite.get('overall_signal', 'N/A')}
• 신호 품질: {composite.get('signal_quality', 0):.1f}%
• 강세 지표: {composite.get('bullish_strength', 0):.2f}
• 약세 지표: {composite.get('bearish_strength', 0):.2f}
• 분석된 지표: {composite.get('indicators_analyzed', 0)}개
"""
    
    def _build_action_guide(self, prediction: Dict, priority: str) -> str:
        """행동 가이드"""
        direction = prediction.get("direction", "NEUTRAL")
        confidence = prediction.get("confidence", "LOW")
        
        if priority == "CRITICAL":
            if direction == "BULLISH":
                action = """
✅ 강한 상승 신호
• 보유자: 홀딩 유지
• 미보유: 분할 매수 고려
• 주의: 과도한 레버리지 금지
"""
            elif direction == "BEARISH":
                action = """
⚠️ 강한 하락 신호
• 보유자: 일부 익절/손절 고려
• 미보유: 관망 권장
• 주의: 저점 매수는 확인 후
"""
            else:
                action = "• 포지션 유지, 추가 신호 대기"
        else:
            action = """
• 현재 전략 유지
• 리스크 관리 우선
• 성급한 결정 금지
"""
        
        return f"""
━━━━━━━━━━━━━━━━━━━━━━━━━
🎯 **권장 행동**
━━━━━━━━━━━━━━━━━━━━━━━━━

{action}
"""
    
    def _build_risk_warning(self, prediction: Dict) -> str:
        """리스크 경고"""
        confidence = prediction.get("confidence", "LOW")
        
        if confidence == "VERY_HIGH":
            risk_level = "⚠️ 중간 리스크"
        elif confidence == "HIGH":
            risk_level = "⚠️⚠️ 높은 리스크"
        else:
            risk_level = "⚠️⚠️⚠️ 매우 높은 리스크"
        
        return f"""
━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️ **리스크 경고**
━━━━━━━━━━━━━━━━━━━━━━━━━

• 리스크 수준: {risk_level}
• 이것은 AI 예측입니다
• 100% 정확하지 않습니다
• 투자 손실 가능성 있습니다
• 여유 자금만 사용하세요
"""
    
    def _build_system_performance(self, performance: Dict) -> str:
        """시스템 성과 표시"""
        overall_accuracy = performance.get("overall_accuracy", 0)
        recent_7d = performance.get("7d_accuracy", 0)
        confidence_stats = performance.get("confidence_breakdown", {})
        
        return f"""
━━━━━━━━━━━━━━━━━━━━━━━━━
📈 **시스템 성과 (투명성)**
━━━━━━━━━━━━━━━━━━━━━━━━━

• 전체 정확도: {overall_accuracy:.1f}%
• 최근 7일: {recent_7d:.1f}%
• HIGH 신뢰도 정확도: {confidence_stats.get('HIGH', 0):.1f}%
• 거짓 신호율: {performance.get('false_positive_rate', 0):.1f}%
"""
    
    async def _send_telegram_message(self, message: str) -> bool:
        """텔레그램 메시지 발송"""
        if not self.bot_token or not self.chat_id:
            self.logger.warning("텔레그램 설정 없음")
            return False
        
        try:
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            
            # 메시지가 너무 길면 분할
            if len(message) > 4000:
                messages = self._split_message(message)
                for msg in messages:
                    await self._send_single_message(url, msg)
            else:
                await self._send_single_message(url, message)
            
            return True
            
        except Exception as e:
            self.logger.error(f"텔레그램 발송 오류: {e}")
            return False
    
    async def _send_single_message(self, url: str, message: str) -> bool:
        """단일 메시지 발송"""
        payload = {
            "chat_id": self.chat_id,
            "text": message,
            "parse_mode": "Markdown",
            "disable_web_page_preview": True
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                return response.status == 200
    
    def _split_message(self, message: str, max_length: int = 4000) -> List[str]:
        """긴 메시지 분할"""
        messages = []
        current = ""
        
        for line in message.split('\n'):
            if len(current) + len(line) + 1 > max_length:
                messages.append(current)
                current = line
            else:
                current += '\n' + line if current else line
        
        if current:
            messages.append(current)
        
        return messages
    
    async def send_error_alert(self, error_type: str, error_message: str):
        """시스템 오류 알림"""
        message = f"""
🔴 **시스템 오류 알림**

• 오류 유형: {error_type}
• 메시지: {error_message}
• 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

시스템이 자동으로 복구를 시도합니다.
"""
        await self._send_telegram_message(message)
    
    async def send_daily_summary(self, stats: Dict):
        """일일 요약 보고서"""
        message = f"""
📊 **일일 요약 보고서**

━━━━━━━━━━━━━━━━━━━━━━━━━

📈 **예측 성과**
• 총 예측: {stats.get('total_predictions', 0)}건
• 정확한 예측: {stats.get('correct_predictions', 0)}건
• 정확도: {stats.get('accuracy', 0):.1f}%

💰 **가격 변동**
• 시작가: ${stats.get('open_price', 0):,.0f}
• 종가: ${stats.get('close_price', 0):,.0f}
• 변동률: {stats.get('price_change', 0):.2f}%

🎯 **알림 통계**
• 발송된 알림: {stats.get('alerts_sent', 0)}건
• 중요 알림: {stats.get('high_priority_alerts', 0)}건

━━━━━━━━━━━━━━━━━━━━━━━━━

내일도 24시간 모니터링을 계속합니다.
"""
        await self._send_telegram_message(message)

async def test_enhanced_notifier():
    """향상된 알림 시스템 테스트"""
    print("🧪 향상된 텔레그램 알림 테스트")
    print("="*50)
    
    notifier = EnhancedTelegramNotifier()
    
    # 테스트 데이터
    test_prediction = {
        "direction": "BULLISH",
        "probability": 92,
        "confidence": "VERY_HIGH",
        "target_price": 70000,
        "timeframe": "6시간"
    }
    
    test_indicators = {
        "composite_analysis": {
            "overall_signal": "STRONG_BULLISH",
            "signal_quality": 85.5,
            "bullish_strength": 0.82,
            "bearish_strength": 0.18,
            "indicators_analyzed": 19
        },
        "high_confidence_signals": [
            {"indicator": "funding_rate", "strength": 0.9},
            {"indicator": "whale_activity", "strength": 0.85},
            {"indicator": "exchange_outflow", "strength": 0.8}
        ]
    }
    
    test_time_series = {
        "pattern_found": True,
        "confidence": 78,
        "similar_patterns_count": 5
    }
    
    test_performance = {
        "overall_accuracy": 78.5,
        "7d_accuracy": 82.3,
        "confidence_breakdown": {
            "HIGH": 85.2,
            "MEDIUM": 72.1,
            "LOW": 54.3
        },
        "false_positive_rate": 12.5
    }
    
    # 메시지 생성 테스트
    priority = notifier._determine_priority(test_prediction)
    print(f"우선순위: {priority}")
    
    message = notifier._build_comprehensive_message(
        test_prediction,
        test_indicators,
        test_time_series,
        test_performance,
        priority
    )
    
    print("\n생성된 메시지:")
    print(message)
    
    # 실제 발송 테스트 (환경변수 설정 시)
    if notifier.bot_token and notifier.chat_id:
        success = await notifier.send_prediction_alert(
            test_prediction,
            test_indicators,
            test_time_series,
            test_performance
        )
        print(f"\n발송 결과: {'성공' if success else '실패'}")

if __name__ == "__main__":
    asyncio.run(test_enhanced_notifier())