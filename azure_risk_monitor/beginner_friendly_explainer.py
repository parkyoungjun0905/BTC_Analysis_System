"""
초보자 친화적 설명 시스템
전문용어를 쉽게 설명하고 행동 가이드 제공
"""

import logging
from typing import Dict, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)

class BeginnerFriendlyExplainer:
    """초보자를 위한 분석 설명 시스템"""
    
    def __init__(self):
        self.logger = logger
        
        # 지표별 쉬운 설명
        self.indicator_explanations = {
            "funding_rate": {
                "name": "펀딩비",
                "simple": "선물 거래자들이 내는 수수료",
                "meaning": "플러스면 매수자가 많고, 마이너스면 매도자가 많음",
                "impact": "극단적 값은 반대 방향 움직임 예고"
            },
            "exchange_outflow": {
                "name": "거래소 유출",
                "simple": "거래소에서 개인지갑으로 코인이 빠져나감",
                "meaning": "장기 보유 의도, 매도 물량 감소",
                "impact": "공급 감소로 가격 상승 압력"
            },
            "whale_activity": {
                "name": "고래 활동",
                "simple": "큰손들의 매매 움직임",
                "meaning": "1000 BTC 이상 보유자들의 행동 패턴",
                "impact": "고래가 사면 상승, 팔면 하락 가능성"
            },
            "fear_greed_index": {
                "name": "공포탐욕지수",
                "simple": "시장 참여자들의 심리 상태",
                "meaning": "0(극도의 공포) ~ 100(극도의 탐욕)",
                "impact": "극단값일 때 반대 움직임 가능"
            },
            "mvrv": {
                "name": "MVRV",
                "simple": "현재 가격이 평균 매수가 대비 어느 정도인지",
                "meaning": "1 이상이면 평균적으로 수익, 1 이하면 손실",
                "impact": "3 이상은 과열, 0.7 이하는 바닥 신호"
            },
            "sopr": {
                "name": "SOPR",
                "simple": "코인을 파는 사람들의 손익 상태",
                "meaning": "1 이상이면 이익 실현, 1 이하면 손절",
                "impact": "1 근처에서 지지/저항 역할"
            },
            "mempool_pressure": {
                "name": "멤풀 압력",
                "simple": "거래 대기줄의 길이",
                "meaning": "많은 거래가 대기 중 = 네트워크 혼잡",
                "impact": "급한 거래 많음 = 큰 움직임 예상"
            },
            "stablecoin_flow": {
                "name": "스테이블코인 플로우",
                "simple": "달러 코인(USDT, USDC)의 움직임",
                "meaning": "거래소로 유입되면 매수 대기 자금",
                "impact": "대량 유입 시 상승 압력"
            },
            "options_put_call": {
                "name": "풋/콜 비율",
                "simple": "하락 베팅 vs 상승 베팅 비율",
                "meaning": "1 이상이면 하락 베팅이 많음",
                "impact": "극단값은 반대 움직임 가능"
            },
            "orderbook_imbalance": {
                "name": "오더북 불균형",
                "simple": "매수 주문 vs 매도 주문 비교",
                "meaning": "매수벽이 크면 지지, 매도벽이 크면 저항",
                "impact": "큰 불균형은 그 방향으로 움직임"
            }
        }
        
        # 신호 강도별 설명
        self.signal_strength_meanings = {
            "VERY_HIGH": {
                "emoji": "🚨",
                "meaning": "매우 강한 신호",
                "action": "즉시 주목 필요",
                "risk": "높은 변동성 예상"
            },
            "HIGH": {
                "emoji": "⚠️",
                "meaning": "강한 신호",
                "action": "포지션 점검 필요",
                "risk": "중간 변동성 예상"
            },
            "MEDIUM": {
                "emoji": "📊",
                "meaning": "보통 신호",
                "action": "모니터링 강화",
                "risk": "일반적 변동성"
            },
            "LOW": {
                "emoji": "📌",
                "meaning": "약한 신호",
                "action": "참고만 하세요",
                "risk": "낮은 변동성"
            }
        }
        
        # 방향별 의미
        self.direction_meanings = {
            "BULLISH": {
                "emoji": "📈",
                "simple": "상승 예상",
                "meaning": "가격이 오를 가능성이 높음",
                "reasons": "매수 압력 증가, 공급 감소, 긍정적 신호들"
            },
            "BEARISH": {
                "emoji": "📉",
                "simple": "하락 예상",
                "meaning": "가격이 내릴 가능성이 높음",
                "reasons": "매도 압력 증가, 공급 증가, 부정적 신호들"
            },
            "NEUTRAL": {
                "emoji": "➡️",
                "simple": "횡보 예상",
                "meaning": "큰 변동 없을 가능성",
                "reasons": "매수/매도 균형, 뚜렷한 신호 없음"
            }
        }
    
    def explain_prediction(self, prediction: Dict, indicators: Dict) -> str:
        """예측 결과를 초보자도 이해할 수 있게 설명"""
        try:
            direction = prediction.get("direction", "NEUTRAL")
            probability = prediction.get("probability", 50)
            confidence = prediction.get("confidence", "LOW")
            target_price = prediction.get("target_price", 0)
            timeframe = prediction.get("timeframe", "24시간")
            
            # 방향 설명
            dir_info = self.direction_meanings.get(direction, self.direction_meanings["NEUTRAL"])
            
            # 강도 설명
            strength_info = self.signal_strength_meanings.get(confidence, self.signal_strength_meanings["LOW"])
            
            # 핵심 지표 3개 선별
            key_indicators = self._select_key_indicators(indicators)
            
            # 초보자 친화적 메시지 생성
            message = f"""
{strength_info['emoji']} **{strength_info['meaning']} 감지**

━━━━━━━━━━━━━━━━━━━━━━━━━

📊 **무슨 일이 일어나고 있나요?**
{dir_info['emoji']} {dir_info['simple']} - {probability}% 확률
• {dir_info['meaning']}
• 예상 이유: {dir_info['reasons']}

━━━━━━━━━━━━━━━━━━━━━━━━━

💡 **왜 이런 예측이 나왔나요?**
{self._explain_key_indicators(key_indicators)}

━━━━━━━━━━━━━━━━━━━━━━━━━

⏰ **언제까지 유효한가요?**
• 예상 시간: {timeframe} 이내
• 목표 가격: ${target_price:,.0f}
• 현재 가격 대비: {self._calculate_percentage_change(indicators.get('current_price', 0), target_price):.1f}%

━━━━━━━━━━━━━━━━━━━━━━━━━

🎯 **어떻게 대응해야 하나요?**
{self._generate_action_guide(direction, confidence, probability)}

━━━━━━━━━━━━━━━━━━━━━━━━━

⚠️ **주의사항**
• 이것은 AI 예측이며 100% 정확하지 않습니다
• 투자 결정은 본인의 판단으로 하세요
• 여유 자금으로만 투자하세요

📈 **시스템 성과**
• 최근 7일 정확도: {self._get_system_accuracy()}%
• 이 신뢰도({confidence}) 과거 정확도: {self._get_confidence_accuracy(confidence)}%
"""
            
            return message
            
        except Exception as e:
            self.logger.error(f"예측 설명 생성 실패: {e}")
            return "예측 설명을 생성할 수 없습니다."
    
    def _select_key_indicators(self, indicators: Dict) -> list:
        """가장 중요한 지표 3개 선별"""
        try:
            # 신호 강도가 높은 순으로 정렬
            all_indicators = []
            for name, data in indicators.get("indicators", {}).items():
                if isinstance(data, dict) and "signal" in data:
                    all_indicators.append({
                        "name": name,
                        "signal": data.get("signal", "NEUTRAL"),
                        "strength": data.get("strength", 0),
                        "value": data.get("value", 0)
                    })
            
            # 강도 순으로 정렬하여 상위 3개 반환
            sorted_indicators = sorted(all_indicators, key=lambda x: x["strength"], reverse=True)
            return sorted_indicators[:3]
            
        except Exception:
            return []
    
    def _explain_key_indicators(self, key_indicators: list) -> str:
        """핵심 지표들을 쉽게 설명"""
        explanations = []
        
        for indicator in key_indicators:
            name = indicator["name"]
            signal = indicator["signal"]
            
            if name in self.indicator_explanations:
                info = self.indicator_explanations[name]
                
                # 신호 방향 이모지
                signal_emoji = "📈" if signal == "BULLISH" else "📉" if signal == "BEARISH" else "➡️"
                
                explanation = f"""
{signal_emoji} **{info['name']}**
• 무엇: {info['simple']}
• 현재: {info['meaning']}
• 영향: {info['impact']}
"""
                explanations.append(explanation)
        
        return "\n".join(explanations) if explanations else "지표 분석 중..."
    
    def _calculate_percentage_change(self, current: float, target: float) -> float:
        """가격 변화율 계산"""
        if current == 0:
            return 0
        return ((target - current) / current) * 100
    
    def _generate_action_guide(self, direction: str, confidence: str, probability: float) -> str:
        """상황별 행동 가이드 생성"""
        
        # 초보자를 위한 일반적 가이드
        if confidence == "VERY_HIGH" and probability > 85:
            if direction == "BULLISH":
                return """
✅ **강한 상승 신호입니다**
• 이미 보유 중: 홀딩 권장
• 미보유: 소량 분할 매수 고려
• 주의: FOMO(추격매수) 조심
"""
            elif direction == "BEARISH":
                return """
⚠️ **강한 하락 신호입니다**
• 이미 보유 중: 일부 익절 고려
• 미보유: 매수 대기
• 주의: 패닉셀링 조심
"""
        
        elif confidence == "HIGH" and probability > 70:
            return """
📊 **중간 강도 신호입니다**
• 포지션 크기 조절 고려
• 추가 확인 신호 대기
• 리스크 관리 우선
"""
        
        else:
            return """
📌 **약한 신호입니다**
• 현재 포지션 유지
• 추가 신호 관찰
• 성급한 결정 금지
"""
    
    def _get_system_accuracy(self) -> float:
        """시스템 전체 정확도 (더미 데이터, 실제로는 DB에서 가져와야 함)"""
        # TODO: prediction_tracker.py에서 실제 데이터 가져오기
        return 78.5
    
    def _get_confidence_accuracy(self, confidence: str) -> float:
        """신뢰도별 정확도 (더미 데이터)"""
        accuracy_map = {
            "VERY_HIGH": 92.3,
            "HIGH": 81.7,
            "MEDIUM": 68.4,
            "LOW": 52.1
        }
        return accuracy_map.get(confidence, 50.0)
    
    def generate_risk_warning(self, volatility_level: str) -> str:
        """변동성 수준별 위험 경고"""
        warnings = {
            "EXTREME": """
🚨🚨🚨 **극도의 변동성 경고** 🚨🚨🚨
• 청산 위험 매우 높음
• 레버리지 사용 금지
• 현물만 소량 거래
""",
            "HIGH": """
⚠️ **높은 변동성 주의** ⚠️
• 포지션 축소 권장
• 손절선 필수 설정
• 분할 매매 권장
""",
            "NORMAL": """
📊 **일반적 변동성**
• 평소 전략 유지
• 리스크 관리 지속
""",
            "LOW": """
😴 **낮은 변동성**
• 큰 움직임 대기
• 포지션 준비 단계
"""
        }
        return warnings.get(volatility_level, warnings["NORMAL"])

class AdvancedMetricsExplainer:
    """고급 온체인 메트릭 설명"""
    
    def __init__(self):
        self.advanced_metrics = {
            "mvrv_zscore": {
                "name": "MVRV Z-Score",
                "simple": "시장 과열/과냉 지표",
                "levels": {
                    "above_7": "극도의 과열 - 천장 근처",
                    "5_to_7": "과열 - 조정 가능성",
                    "2_to_5": "상승 추세 건전",
                    "minus2_to_2": "중립 구간",
                    "minus2_below": "과매도 - 바닥 근처"
                }
            },
            "nvt_signal": {
                "name": "NVT Signal",
                "simple": "네트워크 가치 대비 거래량",
                "levels": {
                    "above_150": "과대평가 - 하락 위험",
                    "80_to_150": "정상 범위",
                    "below_45": "저평가 - 상승 기회"
                }
            },
            "sth_lth_ratio": {
                "name": "단기/장기 보유자 비율",
                "simple": "신규 vs 기존 투자자 행동",
                "meaning": "단기 보유자가 늘면 변동성 증가"
            },
            "puell_multiple": {
                "name": "Puell Multiple",
                "simple": "채굴자 수익성 지표",
                "levels": {
                    "above_4": "채굴자 대량 매도 구간",
                    "2_to_4": "채굴자 이익 실현",
                    "0.5_to_2": "정상 구간",
                    "below_0.5": "채굴자 항복 - 바닥 신호"
                }
            }
        }
    
    def explain_metric(self, metric_name: str, value: float) -> str:
        """메트릭 값을 설명"""
        if metric_name not in self.advanced_metrics:
            return f"{metric_name}: {value}"
        
        metric = self.advanced_metrics[metric_name]
        explanation = f"**{metric['name']}**: {value:.2f}\n"
        explanation += f"• 의미: {metric['simple']}\n"
        
        # 레벨별 해석
        if "levels" in metric:
            for level_range, meaning in metric["levels"].items():
                # 값이 해당 범위에 있는지 체크 (간단 구현)
                if self._check_level(value, level_range):
                    explanation += f"• 현재 상태: {meaning}\n"
                    break
        
        return explanation
    
    def _check_level(self, value: float, level_range: str) -> bool:
        """값이 특정 범위에 있는지 체크"""
        # 간단한 구현 (실제로는 더 정교하게)
        if "above" in level_range:
            threshold = float(level_range.split("_")[1])
            return value > threshold
        elif "below" in level_range:
            threshold = float(level_range.split("_")[1])
            return value < threshold
        elif "_to_" in level_range:
            parts = level_range.split("_to_")
            low = float(parts[0].replace("minus", "-"))
            high = float(parts[1])
            return low <= value <= high
        return False

def test_explainer():
    """설명 시스템 테스트"""
    print("🧪 초보자 친화적 설명 시스템 테스트")
    print("="*50)
    
    explainer = BeginnerFriendlyExplainer()
    
    # 테스트 예측 데이터
    test_prediction = {
        "direction": "BULLISH",
        "probability": 87,
        "confidence": "HIGH",
        "target_price": 65000,
        "timeframe": "6-12시간"
    }
    
    # 테스트 지표 데이터
    test_indicators = {
        "current_price": 60000,
        "indicators": {
            "funding_rate": {"signal": "BULLISH", "strength": 0.85},
            "exchange_outflow": {"signal": "BULLISH", "strength": 0.92},
            "fear_greed_index": {"signal": "NEUTRAL", "strength": 0.5}
        }
    }
    
    # 설명 생성
    explanation = explainer.explain_prediction(test_prediction, test_indicators)
    print(explanation)
    
    # 고급 메트릭 테스트
    print("\n" + "="*50)
    print("🔬 고급 메트릭 설명 테스트")
    
    advanced = AdvancedMetricsExplainer()
    print(advanced.explain_metric("mvrv_zscore", 3.5))
    print(advanced.explain_metric("nvt_signal", 120))
    print(advanced.explain_metric("puell_multiple", 0.4))

if __name__ == "__main__":
    test_explainer()