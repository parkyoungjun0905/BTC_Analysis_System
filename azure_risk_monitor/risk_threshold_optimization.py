#!/usr/bin/env python3
"""
리스크 알림 임계값 최적화 시스템
실제로 유용한 위험 알림을 위한 다차원 분석
"""

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum

class MarketContext(Enum):
    """시장 컨텍스트 정의"""
    BULL_EUPHORIA = "강세_과열"      # 버블 위험
    BULL_HEALTHY = "건강한_상승"      # 정상 상승
    SIDEWAYS_ACCUMULATION = "횡보_누적"  # 스마트머니 매집
    SIDEWAYS_DISTRIBUTION = "횡보_분산"  # 스마트머니 매도
    BEAR_PANIC = "공포_하락"         # 패닉 매도
    BEAR_CAPITULATION = "항복_국면"   # 바닥 신호
    RECOVERY = "회복_초기"           # 반등 시작

@dataclass
class RiskThreshold:
    """리스크 임계값 정의"""
    indicator: str
    context: MarketContext
    threshold: float
    confidence_required: float  # 알림 발송 필요 신뢰도
    time_persistence: int  # 지속 시간 (분)
    priority: int  # 1-5 (1이 가장 높음)
    action_type: str  # "IMMEDIATE", "WATCH", "PREPARE"

class AdaptiveRiskThresholdSystem:
    """적응형 리스크 임계값 시스템"""
    
    def __init__(self):
        # 🎯 핵심 인사이트: 같은 5% 변동도 상황에 따라 의미가 다름
        self.volatility_contexts = {
            # 평상시 변동성 (연간 기준)
            "normal_btc_volatility": 0.60,  # 60% 연간
            "normal_daily_volatility": 0.04,  # 4% 일간
            "normal_hourly_volatility": 0.015,  # 1.5% 시간당
            
            # 위험 수준별 변동성 배수
            "low_risk_multiplier": 1.5,     # 평상시의 1.5배
            "medium_risk_multiplier": 2.0,   # 평상시의 2배
            "high_risk_multiplier": 3.0,     # 평상시의 3배
            "extreme_risk_multiplier": 5.0   # 평상시의 5배
        }
        
        # 🔍 다차원 위험 매트릭스
        self.risk_matrix = self._build_risk_matrix()
        
    def _build_risk_matrix(self) -> Dict[str, List[RiskThreshold]]:
        """시장 상황별 리스크 매트릭스 구축"""
        
        matrix = {}
        
        # 1️⃣ 강세 과열 (버블) 상황 - 하락 리스크 극대화
        matrix[MarketContext.BULL_EUPHORIA] = [
            # 극도의 과매수 + 다이버전스
            RiskThreshold(
                indicator="rsi_divergence",
                context=MarketContext.BULL_EUPHORIA,
                threshold=85,  # RSI 85 이상에서 다이버전스
                confidence_required=0.85,
                time_persistence=30,  # 30분 지속
                priority=1,
                action_type="IMMEDIATE"
            ),
            # 극단적 펀딩비
            RiskThreshold(
                indicator="funding_rate",
                context=MarketContext.BULL_EUPHORIA,
                threshold=0.1,  # 0.1% (연 109% 수준)
                confidence_required=0.80,
                time_persistence=60,  # 1시간 지속
                priority=1,
                action_type="IMMEDIATE"
            ),
            # 거래소 대량 유입
            RiskThreshold(
                indicator="exchange_inflows",
                context=MarketContext.BULL_EUPHORIA,
                threshold=2.0,  # 평균의 2배
                confidence_required=0.75,
                time_persistence=15,
                priority=2,
                action_type="WATCH"
            ),
            # 소셜 극도 탐욕
            RiskThreshold(
                indicator="fear_greed",
                context=MarketContext.BULL_EUPHORIA,
                threshold=90,  # 극도의 탐욕
                confidence_required=0.70,
                time_persistence=120,  # 2시간
                priority=3,
                action_type="PREPARE"
            )
        ]
        
        # 2️⃣ 건강한 상승 - 조정 리스크 감지
        matrix[MarketContext.BULL_HEALTHY] = [
            RiskThreshold(
                indicator="volume_exhaustion",
                context=MarketContext.BULL_HEALTHY,
                threshold=-30,  # 거래량 30% 감소
                confidence_required=0.70,
                time_persistence=60,
                priority=2,
                action_type="WATCH"
            ),
            RiskThreshold(
                indicator="whale_distribution",
                context=MarketContext.BULL_HEALTHY,
                threshold=1.5,  # 고래 매도 평균의 1.5배
                confidence_required=0.75,
                time_persistence=30,
                priority=2,
                action_type="WATCH"
            )
        ]
        
        # 3️⃣ 횡보 누적 - 브레이크아웃 감지
        matrix[MarketContext.SIDEWAYS_ACCUMULATION] = [
            RiskThreshold(
                indicator="volatility_compression",
                context=MarketContext.SIDEWAYS_ACCUMULATION,
                threshold=0.5,  # 변동성 50% 압축
                confidence_required=0.80,
                time_persistence=240,  # 4시간
                priority=2,
                action_type="PREPARE"
            ),
            RiskThreshold(
                indicator="smart_money_accumulation",
                context=MarketContext.SIDEWAYS_ACCUMULATION,
                threshold=1.8,  # 스마트머니 매집 평균의 1.8배
                confidence_required=0.75,
                time_persistence=360,  # 6시간
                priority=3,
                action_type="PREPARE"
            )
        ]
        
        # 4️⃣ 공포 하락 - 과매도 반등 감지
        matrix[MarketContext.BEAR_PANIC] = [
            RiskThreshold(
                indicator="rsi_oversold_extreme",
                context=MarketContext.BEAR_PANIC,
                threshold=20,  # RSI 20 이하
                confidence_required=0.70,
                time_persistence=15,
                priority=1,
                action_type="IMMEDIATE"
            ),
            RiskThreshold(
                indicator="funding_rate_negative",
                context=MarketContext.BEAR_PANIC,
                threshold=-0.05,  # -0.05% 
                confidence_required=0.75,
                time_persistence=30,
                priority=2,
                action_type="WATCH"
            ),
            RiskThreshold(
                indicator="capitulation_volume",
                context=MarketContext.BEAR_PANIC,
                threshold=3.0,  # 평균 거래량의 3배
                confidence_required=0.80,
                time_persistence=5,  # 5분 스파이크
                priority=1,
                action_type="IMMEDIATE"
            )
        ]
        
        # 5️⃣ 항복 국면 - 바닥 신호
        matrix[MarketContext.BEAR_CAPITULATION] = [
            RiskThreshold(
                indicator="long_liquidations",
                context=MarketContext.BEAR_CAPITULATION,
                threshold=100000000,  # 1억 달러 청산
                confidence_required=0.85,
                time_persistence=5,
                priority=1,
                action_type="IMMEDIATE"
            ),
            RiskThreshold(
                indicator="miner_capitulation",
                context=MarketContext.BEAR_CAPITULATION,
                threshold=0.7,  # 해시레이트 30% 하락
                confidence_required=0.80,
                time_persistence=1440,  # 24시간
                priority=2,
                action_type="WATCH"
            )
        ]
        
        return matrix
    
    def calculate_dynamic_threshold(self, 
                                   base_volatility: float,
                                   market_context: MarketContext,
                                   indicator_type: str) -> Dict[str, float]:
        """동적 임계값 계산"""
        
        # 시장 상황별 가중치
        context_weights = {
            MarketContext.BULL_EUPHORIA: 0.5,      # 더 민감하게
            MarketContext.BULL_HEALTHY: 1.0,       # 정상
            MarketContext.SIDEWAYS_ACCUMULATION: 1.2,  # 덜 민감하게
            MarketContext.SIDEWAYS_DISTRIBUTION: 0.8,
            MarketContext.BEAR_PANIC: 0.6,         # 민감하게
            MarketContext.BEAR_CAPITULATION: 0.4,  # 매우 민감하게
            MarketContext.RECOVERY: 0.9
        }
        
        # 지표 타입별 민감도
        indicator_sensitivities = {
            "price": 1.0,
            "volume": 1.2,
            "sentiment": 0.8,
            "onchain": 0.9,
            "derivatives": 1.1,
            "technical": 1.0
        }
        
        context_weight = context_weights.get(market_context, 1.0)
        indicator_sensitivity = indicator_sensitivities.get(indicator_type, 1.0)
        
        # 🎯 핵심 공식: 적응형 임계값
        adaptive_threshold = base_volatility * context_weight * indicator_sensitivity
        
        return {
            "immediate_action": adaptive_threshold * 2.0,   # 즉시 행동
            "high_alert": adaptive_threshold * 1.5,         # 높은 경고
            "medium_alert": adaptive_threshold * 1.2,       # 중간 경고
            "low_alert": adaptive_threshold * 1.0,          # 낮은 경고
            "monitoring": adaptive_threshold * 0.8          # 모니터링
        }
    
    def assess_real_risk(self, current_data: Dict) -> Dict[str, any]:
        """실제 리스크 평가"""
        
        # 1. 현재 시장 컨텍스트 파악
        context = self._identify_market_context(current_data)
        
        # 2. 해당 컨텍스트의 리스크 체크
        applicable_thresholds = self.risk_matrix.get(context, [])
        
        triggered_risks = []
        for threshold in applicable_thresholds:
            if self._check_threshold_condition(current_data, threshold):
                triggered_risks.append({
                    "indicator": threshold.indicator,
                    "priority": threshold.priority,
                    "action": threshold.action_type,
                    "confidence": self._calculate_confidence(current_data, threshold)
                })
        
        # 3. 종합 리스크 스코어
        if not triggered_risks:
            overall_risk = "LOW"
            recommended_action = "MONITOR"
        else:
            max_priority = min(r["priority"] for r in triggered_risks)
            if max_priority == 1:
                overall_risk = "CRITICAL"
                recommended_action = "IMMEDIATE_ACTION"
            elif max_priority == 2:
                overall_risk = "HIGH"
                recommended_action = "PREPARE_ACTION"
            else:
                overall_risk = "MEDIUM"
                recommended_action = "INCREASE_MONITORING"
        
        return {
            "market_context": context.value,
            "overall_risk": overall_risk,
            "recommended_action": recommended_action,
            "triggered_risks": triggered_risks,
            "confidence": np.mean([r["confidence"] for r in triggered_risks]) if triggered_risks else 0
        }
    
    def _identify_market_context(self, data: Dict) -> MarketContext:
        """시장 컨텍스트 식별"""
        
        # 복합 지표 기반 컨텍스트 판단
        rsi = data.get("rsi", 50)
        fear_greed = data.get("fear_greed", 50)
        trend = data.get("trend_strength", 0)
        volume = data.get("volume_ratio", 1.0)
        funding = data.get("funding_rate", 0)
        
        # 규칙 기반 분류
        if rsi > 80 and fear_greed > 85 and funding > 0.05:
            return MarketContext.BULL_EUPHORIA
        elif rsi > 60 and trend > 0 and volume > 0.8:
            return MarketContext.BULL_HEALTHY
        elif 40 <= rsi <= 60 and abs(trend) < 0.2:
            if volume < 0.7:
                return MarketContext.SIDEWAYS_ACCUMULATION
            else:
                return MarketContext.SIDEWAYS_DISTRIBUTION
        elif rsi < 30 and fear_greed < 20:
            if volume > 2.0:
                return MarketContext.BEAR_CAPITULATION
            else:
                return MarketContext.BEAR_PANIC
        else:
            return MarketContext.RECOVERY
    
    def _check_threshold_condition(self, data: Dict, threshold: RiskThreshold) -> bool:
        """임계값 조건 체크"""
        indicator_value = data.get(threshold.indicator, 0)
        
        # 방향성에 따른 비교
        if threshold.threshold >= 0:
            return indicator_value >= threshold.threshold
        else:
            return indicator_value <= threshold.threshold
    
    def _calculate_confidence(self, data: Dict, threshold: RiskThreshold) -> float:
        """신뢰도 계산"""
        # 여러 요소를 종합한 신뢰도
        base_confidence = 0.5
        
        # 지표 강도
        indicator_strength = min(abs(data.get(threshold.indicator, 0) / threshold.threshold), 2.0) / 2.0
        base_confidence += indicator_strength * 0.3
        
        # 지속성 (시간)
        # 실제로는 시계열 데이터 필요
        persistence_factor = 0.8  # 임시값
        base_confidence += persistence_factor * 0.2
        
        return min(base_confidence, 1.0)

class PracticalAlertOptimizer:
    """실용적 알림 최적화"""
    
    def __init__(self):
        self.alert_fatigue_threshold = 5  # 하루 최대 알림 수
        self.minimum_interval = 30  # 최소 알림 간격 (분)
        
    def optimize_alert_frequency(self, risk_level: str, user_profile: str) -> Dict:
        """사용자별 알림 빈도 최적화"""
        
        profiles = {
            "aggressive": {  # 공격적 트레이더
                "CRITICAL": {"max_daily": 10, "min_interval": 5},
                "HIGH": {"max_daily": 8, "min_interval": 15},
                "MEDIUM": {"max_daily": 5, "min_interval": 30},
                "LOW": {"max_daily": 2, "min_interval": 60}
            },
            "moderate": {  # 중도적 투자자
                "CRITICAL": {"max_daily": 5, "min_interval": 15},
                "HIGH": {"max_daily": 3, "min_interval": 30},
                "MEDIUM": {"max_daily": 2, "min_interval": 60},
                "LOW": {"max_daily": 1, "min_interval": 120}
            },
            "conservative": {  # 보수적 홀더
                "CRITICAL": {"max_daily": 3, "min_interval": 30},
                "HIGH": {"max_daily": 2, "min_interval": 60},
                "MEDIUM": {"max_daily": 1, "min_interval": 120},
                "LOW": {"max_daily": 0, "min_interval": 0}
            }
        }
        
        return profiles.get(user_profile, profiles["moderate"]).get(risk_level)

# 🎯 핵심 인사이트 정리
def get_optimal_thresholds_summary() -> Dict:
    """최적 임계값 요약"""
    
    return {
        "핵심_원칙": {
            "1_컨텍스트_우선": "같은 5% 변동도 강세장과 약세장에서 의미가 다름",
            "2_다차원_분석": "단일 지표가 아닌 복합 조건으로 판단",
            "3_시간_지속성": "순간 스파이크보다 지속되는 신호가 중요",
            "4_신뢰도_가중": "모든 신호가 같은 가치를 갖지 않음",
            "5_피로도_관리": "너무 많은 알림은 오히려 무시하게 됨"
        },
        
        "실용적_임계값": {
            "가격_변동": {
                "1시간": {
                    "일반": "3%",
                    "높은변동성": "5%",
                    "극단상황": "7%"
                },
                "4시간": {
                    "일반": "5%",
                    "높은변동성": "8%",
                    "극단상황": "12%"
                },
                "일간": {
                    "일반": "7%",
                    "높은변동성": "12%",
                    "극단상황": "20%"
                }
            },
            
            "기술적_지표": {
                "RSI": {
                    "과매수_경고": 75,
                    "과매수_위험": 85,
                    "과매도_경고": 25,
                    "과매도_기회": 15
                },
                "MACD": {
                    "강한_다이버전스": "3일_이상_지속",
                    "약한_다이버전스": "1일_지속"
                }
            },
            
            "온체인": {
                "거래소_유입": {
                    "경고": "평균의_1.5배",
                    "위험": "평균의_2배",
                    "극단": "평균의_3배"
                },
                "고래_활동": {
                    "주목": "1000_BTC",
                    "경고": "2000_BTC",
                    "위험": "5000_BTC"
                }
            },
            
            "파생상품": {
                "펀딩비": {
                    "과열": "0.05%",
                    "극단": "0.1%",
                    "역과열": "-0.05%"
                },
                "청산": {
                    "주목": "5천만_달러",
                    "경고": "1억_달러",
                    "극단": "3억_달러"
                }
            }
        },
        
        "알림_우선순위": {
            "P1_즉시": [
                "가격_7%_급변동_5분내",
                "1억달러_이상_청산",
                "RSI_90_또는_10",
                "펀딩비_0.1%_초과"
            ],
            "P2_긴급": [
                "가격_5%_변동_15분내",
                "5천만달러_청산",
                "거래소_대량_유입",
                "RSI_다이버전스_확인"
            ],
            "P3_주의": [
                "가격_3%_변동_1시간내",
                "고래_이동_감지",
                "거래량_이상_패턴",
                "기술적_패턴_완성"
            ]
        },
        
        "사용자_맞춤": {
            "단기_트레이더": {
                "초점": "1-4시간_변동성",
                "민감도": "높음",
                "알림빈도": "최대_10회/일"
            },
            "스윙_트레이더": {
                "초점": "일간_패턴",
                "민감도": "중간",
                "알림빈도": "최대_5회/일"
            },
            "장기_홀더": {
                "초점": "주요_전환점",
                "민감도": "낮음",
                "알림빈도": "최대_2회/일"
            }
        }
    }

if __name__ == "__main__":
    # 시스템 초기화
    risk_system = AdaptiveRiskThresholdSystem()
    optimizer = PracticalAlertOptimizer()
    
    # 테스트 데이터
    test_data = {
        "rsi": 82,
        "fear_greed": 88,
        "trend_strength": 0.8,
        "volume_ratio": 1.5,
        "funding_rate": 0.08,
        "whale_activity": 1500,
        "exchange_inflows": 2.2
    }
    
    # 리스크 평가
    risk_assessment = risk_system.assess_real_risk(test_data)
    
    print("🎯 리스크 평가 결과")
    print(f"시장 상황: {risk_assessment['market_context']}")
    print(f"전체 리스크: {risk_assessment['overall_risk']}")
    print(f"권장 조치: {risk_assessment['recommended_action']}")
    print(f"신뢰도: {risk_assessment['confidence']:.1%}")
    
    # 최적 임계값 요약
    summary = get_optimal_thresholds_summary()
    print("\n📊 최적 임계값 요약")
    print(json.dumps(summary, ensure_ascii=False, indent=2))