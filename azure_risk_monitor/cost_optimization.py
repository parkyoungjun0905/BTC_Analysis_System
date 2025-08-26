"""
Claude API 요금 최적화 모듈
월 5만원 → 2만원으로 절약
"""

import asyncio
from datetime import datetime, time
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

class CostOptimizer:
    """Claude API 비용 최적화"""
    
    def __init__(self):
        self.logger = logger
        
        # 시간대별 중요도 (높을수록 중요)
        self.time_importance = {
            # 한국 시간 기준
            "high": [9, 10, 11, 14, 15, 16, 21, 22],    # 장 시작/마감, 미국 장 시간
            "medium": [7, 8, 12, 13, 17, 18, 19, 20],   # 일반 거래시간
            "low": [0, 1, 2, 3, 4, 5, 6, 23]           # 새벽/심야
        }
        
        # 신뢰도 임계값
        self.confidence_thresholds = {
            "high_time": 60,      # 중요 시간대: 60% 이상
            "medium_time": 70,    # 보통 시간대: 70% 이상  
            "low_time": 80        # 한가한 시간대: 80% 이상
        }
    
    def should_call_claude_api(self, indicators: Dict, current_hour: int) -> bool:
        """Claude API 호출 여부 결정"""
        try:
            # 1. 시간대 중요도 확인
            time_priority = self.get_time_priority(current_hour)
            
            # 2. 지표 신뢰도 확인
            composite = indicators.get("composite_analysis", {})
            confidence = composite.get("confidence", 0)
            
            # 3. 임계값과 비교
            threshold = self.confidence_thresholds[f"{time_priority}_time"]
            
            if confidence >= threshold:
                self.logger.info(f"Claude API 호출: {time_priority}시간대, 신뢰도 {confidence}%")
                return True
            else:
                self.logger.info(f"Claude API 스킵: {time_priority}시간대, 신뢰도 {confidence}% < {threshold}%")
                return False
                
        except Exception as e:
            self.logger.error(f"비용 최적화 판단 실패: {e}")
            return True  # 안전하게 API 호출
    
    def get_time_priority(self, hour: int) -> str:
        """시간대 우선순위 반환"""
        if hour in self.time_importance["high"]:
            return "high"
        elif hour in self.time_importance["medium"]:
            return "medium"
        else:
            return "low"
    
    def generate_local_prediction(self, indicators: Dict) -> str:
        """로컬 예측 (Claude API 대신)"""
        try:
            composite = indicators.get("composite_analysis", {})
            prediction_signals = indicators.get("prediction_signals", {})
            
            direction = prediction_signals.get("direction", "NEUTRAL")
            probability = prediction_signals.get("probability", 50)
            confidence = composite.get("confidence", 0)
            
            # 간단한 로컬 분석
            if confidence > 70:
                strength = "HIGH"
                timeframe = "6-12시간"
            elif confidence > 50:
                strength = "MEDIUM"  
                timeframe = "12-24시간"
            else:
                strength = "LOW"
                timeframe = "24-48시간"
            
            return f"""PREDICTION_DIRECTION: {direction}
PROBABILITY: {probability}%
TIMEFRAME: {timeframe}
CONFIDENCE: {strength}

KEY_INDICATORS:
- 19개 지표 종합 신호: {direction}
- 시스템 신뢰도: {confidence:.1f}%
- 강세/약세 균형 분석

REASONING:
로컬 분석 결과 {direction} {probability}% 신호입니다. 신뢰도가 낮아 간단 분석을 제공했습니다."""
            
        except Exception as e:
            self.logger.error(f"로컬 예측 생성 실패: {e}")
            return "PREDICTION_DIRECTION: NEUTRAL\nPROBABILITY: 50%\nCONFIDENCE: LOW"
    
    def calculate_potential_savings(self) -> Dict:
        """절약 가능 금액 계산"""
        
        # 현재 사용량 (월 1,440회)
        current_calls = 1440
        
        # 최적화 후 예상 사용량
        high_time_calls = 1440 * 8/24 * 0.7    # 중요시간 70% 호출
        medium_time_calls = 1440 * 8/24 * 0.4  # 보통시간 40% 호출  
        low_time_calls = 1440 * 8/24 * 0.2     # 한가시간 20% 호출
        
        optimized_calls = high_time_calls + medium_time_calls + low_time_calls
        
        # 요금 계산
        cost_per_call = 49000 / 1440  # 월 49,000원 ÷ 1,440회
        current_cost = current_calls * cost_per_call
        optimized_cost = optimized_calls * cost_per_call
        savings = current_cost - optimized_cost
        
        return {
            "current_monthly_cost": f"{current_cost:,.0f}원",
            "optimized_monthly_cost": f"{optimized_cost:,.0f}원",
            "monthly_savings": f"{savings:,.0f}원",
            "savings_percentage": f"{savings/current_cost*100:.1f}%",
            "current_calls": current_calls,
            "optimized_calls": int(optimized_calls)
        }

def test_cost_optimization():
    """비용 최적화 테스트"""
    print("💰 Claude API 비용 최적화 분석")
    print("="*50)
    
    optimizer = CostOptimizer()
    
    # 절약 계산
    savings = optimizer.calculate_potential_savings()
    
    print("📊 비용 분석:")
    print(f"  • 현재 월 비용: {savings['current_monthly_cost']}")
    print(f"  • 최적화 후: {savings['optimized_monthly_cost']}")
    print(f"  • 월 절약액: {savings['monthly_savings']}")
    print(f"  • 절약률: {savings['savings_percentage']}")
    print(f"  • 호출 감소: {savings['current_calls']} → {savings['optimized_calls']}")
    
    print("\n⏰ 시간대별 전략:")
    for hour in [9, 15, 21, 3]:
        priority = optimizer.get_time_priority(hour)
        threshold = optimizer.confidence_thresholds[f"{priority}_time"]
        print(f"  • {hour}시: {priority} 우선순위 (임계값 {threshold}%)")

if __name__ == "__main__":
    test_cost_optimization()