#!/usr/bin/env python3
"""
시계열 기반 위험 분석 엔진
과거 패턴 매칭과 실시간 이상 감지를 통한 고정확도 위험 예측
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class TimeSeriesRiskAnalyzer:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.scaler = StandardScaler()
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        
        # 과거 위험 패턴들 (실제로는 데이터베이스에서 로드)
        self.historical_patterns = self.load_historical_patterns()
        
    def load_historical_patterns(self) -> Dict:
        """과거 위험 패턴들 로드 (하드코딩으로 시작, 나중에 DB 연동)"""
        return {
            "flash_crash_2022": {
                "price_drop_5min": -0.15,
                "volume_spike": 8.5,
                "funding_rate": 0.008,
                "fear_greed": 12,
                "vix_level": 32
            },
            "luna_collapse_2022": {
                "price_drop_1hour": -0.35,
                "correlation_break": 0.6,
                "volume_anomaly": 12.0,
                "social_sentiment": -0.8
            },
            "covid_crash_2020": {
                "price_drop_1day": -0.50,
                "macro_correlation": 0.9,
                "vix_spike": 45,
                "liquidation_cascade": 500000000
            },
            "china_ban_2021": {
                "price_decline_7day": -0.40,
                "hash_rate_drop": -0.35,
                "regulatory_news": 1.0,
                "asian_premium": -0.05
            }
        }

    def analyze_timeseries_risk(self, current_data: Dict, historical_data: List[Dict]) -> Dict:
        """시계열 데이터 기반 종합 위험 분석"""
        try:
            # 1. 급변 감지 (Sudden Change Detection)
            sudden_change_risk = self.detect_sudden_changes(current_data, historical_data)
            
            # 2. 패턴 매칭 (Historical Pattern Matching)
            pattern_match_risk = self.match_historical_patterns(current_data, historical_data)
            
            # 3. 이상 감지 (Anomaly Detection)
            anomaly_risk = self.detect_anomalies(current_data, historical_data)
            
            # 4. 추세 변화 감지 (Trend Change Detection)
            trend_change_risk = self.detect_trend_changes(historical_data)
            
            # 5. 상관관계 파괴 감지 (Correlation Breakdown)
            correlation_risk = self.detect_correlation_breakdown(current_data, historical_data)
            
            # 6. 종합 위험도 계산
            composite_risk = self.calculate_composite_risk({
                "sudden_change": sudden_change_risk,
                "pattern_match": pattern_match_risk,
                "anomaly": anomaly_risk,
                "trend_change": trend_change_risk,
                "correlation": correlation_risk
            })
            
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "composite_risk_score": composite_risk["total_score"],
                "risk_level": composite_risk["risk_level"],
                "confidence": composite_risk["confidence"],
                "components": {
                    "sudden_change": sudden_change_risk,
                    "pattern_match": pattern_match_risk,
                    "anomaly": anomaly_risk,
                    "trend_change": trend_change_risk,
                    "correlation": correlation_risk
                },
                "recommendations": self.generate_recommendations(composite_risk),
                "next_check_in": self.calculate_next_check_time(composite_risk["risk_level"])
            }
            
        except Exception as e:
            self.logger.error(f"위험 분석 실패: {e}")
            return {
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat(),
                "composite_risk_score": 0.5,  # 기본값
                "risk_level": "WARNING"
            }

    def detect_sudden_changes(self, current_data: Dict, historical_data: List[Dict]) -> Dict:
        """급변 감지 알고리즘"""
        try:
            sudden_change_indicators = {
                "price_velocity": 0,
                "volume_spike": 0,
                "funding_rate_jump": 0,
                "macro_shock": 0,
                "composite_score": 0
            }
            
            if not historical_data or len(historical_data) < 10:
                return sudden_change_indicators
                
            # 가격 급변 감지
            if "price_data" in current_data:
                current_price = current_data["price_data"].get("current_price", 0)
                
                # 최근 5분, 1시간, 24시간 변화율 계산
                recent_prices = []
                for i, hist_data in enumerate(historical_data[-10:]):
                    if "price_data" in hist_data:
                        recent_prices.append(hist_data["price_data"].get("current_price", current_price))
                        
                if len(recent_prices) >= 5:
                    # 5분간 변화율 (최근 5개 데이터 포인트)
                    price_5min_change = (current_price - recent_prices[-5]) / recent_prices[-5]
                    sudden_change_indicators["price_velocity"] = min(abs(price_5min_change) / 0.05, 1.0)
                    
            # 거래량 급증 감지
            if "price_data" in current_data and "volume_24h" in current_data["price_data"]:
                current_volume = current_data["price_data"]["volume_24h"]
                
                # 과거 평균 거래량 계산
                historical_volumes = []
                for hist_data in historical_data[-30:]:  # 최근 30개 데이터
                    if "price_data" in hist_data and "volume_24h" in hist_data["price_data"]:
                        historical_volumes.append(hist_data["price_data"]["volume_24h"])
                        
                if historical_volumes:
                    avg_volume = np.mean(historical_volumes)
                    volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
                    sudden_change_indicators["volume_spike"] = min((volume_ratio - 1) / 4, 1.0)
                    
            # VIX 급등 감지
            if "macro_data" in current_data and "vix" in current_data["macro_data"]:
                current_vix = current_data["macro_data"]["vix"]["current"]
                vix_change = current_data["macro_data"]["vix"]["change"]
                
                if abs(vix_change) > 3:  # VIX 3포인트 이상 변화
                    sudden_change_indicators["macro_shock"] = min(abs(vix_change) / 10, 1.0)
                    
            # 종합 급변 점수
            weights = {
                "price_velocity": 0.4,
                "volume_spike": 0.25,
                "funding_rate_jump": 0.2,
                "macro_shock": 0.15
            }
            
            sudden_change_indicators["composite_score"] = sum(
                sudden_change_indicators[key] * weight 
                for key, weight in weights.items()
            )
            
            return sudden_change_indicators
            
        except Exception as e:
            self.logger.error(f"급변 감지 실패: {e}")
            return {"composite_score": 0, "error": str(e)}

    def match_historical_patterns(self, current_data: Dict, historical_data: List[Dict]) -> Dict:
        """과거 위험 패턴과의 유사도 분석"""
        try:
            pattern_matches = {}
            
            # 현재 상황의 특징 벡터 생성
            current_features = self.extract_features(current_data, historical_data)
            
            # 각 과거 패턴과 비교
            for pattern_name, pattern_features in self.historical_patterns.items():
                similarity = self.calculate_pattern_similarity(current_features, pattern_features)
                pattern_matches[pattern_name] = {
                    "similarity": similarity,
                    "risk_level": self.get_pattern_risk_level(pattern_name),
                    "trigger_probability": similarity * self.get_pattern_severity(pattern_name)
                }
                
            # 가장 유사한 패턴 찾기
            best_match = max(pattern_matches.items(), key=lambda x: x[1]["similarity"])
            
            return {
                "best_match": {
                    "pattern": best_match[0],
                    "similarity": best_match[1]["similarity"],
                    "risk_level": best_match[1]["risk_level"]
                },
                "all_matches": pattern_matches,
                "composite_score": best_match[1]["trigger_probability"]
            }
            
        except Exception as e:
            self.logger.error(f"패턴 매칭 실패: {e}")
            return {"composite_score": 0, "error": str(e)}

    def detect_anomalies(self, current_data: Dict, historical_data: List[Dict]) -> Dict:
        """머신러닝 기반 이상 감지"""
        try:
            if len(historical_data) < 50:  # 최소 데이터 요구량
                return {"composite_score": 0, "note": "insufficient_data"}
                
            # 특징 벡터 생성
            feature_vectors = []
            for hist_data in historical_data:
                features = self.extract_numerical_features(hist_data)
                if features:
                    feature_vectors.append(features)
                    
            if len(feature_vectors) < 20:
                return {"composite_score": 0, "note": "insufficient_features"}
                
            # 정규화
            feature_matrix = np.array(feature_vectors)
            feature_matrix_scaled = self.scaler.fit_transform(feature_matrix)
            
            # Isolation Forest로 이상 감지 모델 훈련
            self.isolation_forest.fit(feature_matrix_scaled)
            
            # 현재 데이터의 이상도 계산
            current_features = self.extract_numerical_features(current_data)
            if current_features:
                current_scaled = self.scaler.transform([current_features])
                anomaly_score = self.isolation_forest.decision_function(current_scaled)[0]
                is_anomaly = self.isolation_forest.predict(current_scaled)[0] == -1
                
                # 점수 정규화 (0-1 범위)
                normalized_score = max(0, min(1, (0.5 - anomaly_score) / 1.0))
                
                return {
                    "is_anomaly": is_anomaly,
                    "anomaly_score": float(anomaly_score),
                    "composite_score": normalized_score,
                    "confidence": 0.7 if len(feature_vectors) > 100 else 0.5
                }
            else:
                return {"composite_score": 0, "note": "no_current_features"}
                
        except Exception as e:
            self.logger.error(f"이상 감지 실패: {e}")
            return {"composite_score": 0, "error": str(e)}

    def detect_trend_changes(self, historical_data: List[Dict]) -> Dict:
        """추세 변화점 감지"""
        try:
            if len(historical_data) < 20:
                return {"composite_score": 0, "note": "insufficient_data"}
                
            # 가격 시계열 추출
            prices = []
            timestamps = []
            
            for hist_data in historical_data:
                if "price_data" in hist_data and "current_price" in hist_data["price_data"]:
                    prices.append(hist_data["price_data"]["current_price"])
                    timestamps.append(hist_data.get("timestamp", datetime.utcnow().isoformat()))
                    
            if len(prices) < 20:
                return {"composite_score": 0, "note": "insufficient_price_data"}
                
            prices = np.array(prices)
            
            # 이동평균 기반 추세 분석
            short_ma = np.mean(prices[-5:])   # 단기 (5개)
            medium_ma = np.mean(prices[-10:]) # 중기 (10개)
            long_ma = np.mean(prices[-20:])   # 장기 (20개)
            
            # 추세 강도 계산
            trend_strength = 0
            if long_ma > 0:
                short_vs_long = (short_ma - long_ma) / long_ma
                medium_vs_long = (medium_ma - long_ma) / long_ma
                
                # 추세 변화 감지 (단기가 장기와 크게 벗어나는 경우)
                trend_strength = abs(short_vs_long) + abs(medium_vs_long)
                
            # 변화율 가속도 (2차 미분)
            if len(prices) >= 10:
                returns = np.diff(np.log(prices))
                acceleration = np.diff(returns)
                recent_acceleration = np.mean(np.abs(acceleration[-5:]))
                trend_strength += recent_acceleration * 100
                
            return {
                "trend_strength": float(trend_strength),
                "short_ma": float(short_ma),
                "medium_ma": float(medium_ma), 
                "long_ma": float(long_ma),
                "composite_score": min(trend_strength / 0.1, 1.0)  # 0.1 = 10% 변화를 1.0 점수로
            }
            
        except Exception as e:
            self.logger.error(f"추세 변화 감지 실패: {e}")
            return {"composite_score": 0, "error": str(e)}

    def detect_correlation_breakdown(self, current_data: Dict, historical_data: List[Dict]) -> Dict:
        """상관관계 파괴 감지"""
        try:
            if len(historical_data) < 30:
                return {"composite_score": 0, "note": "insufficient_data"}
                
            # BTC vs VIX, BTC vs DXY 상관관계 분석
            btc_prices = []
            vix_values = []
            dxy_values = []
            
            for hist_data in historical_data:
                if "price_data" in hist_data and "current_price" in hist_data["price_data"]:
                    btc_prices.append(hist_data["price_data"]["current_price"])
                    
                if "macro_data" in hist_data:
                    if "vix" in hist_data["macro_data"]:
                        vix_values.append(hist_data["macro_data"]["vix"]["current"])
                    if "dxy" in hist_data["macro_data"]:
                        dxy_values.append(hist_data["macro_data"]["dxy"]["current"])
                        
            correlation_breakdown = 0
            
            # BTC-VIX 상관관계 (보통은 음의 상관관계)
            if len(btc_prices) == len(vix_values) and len(btc_prices) >= 20:
                btc_returns = np.diff(np.log(btc_prices))
                vix_returns = np.diff(vix_values)
                
                # 최근 상관관계 vs 과거 상관관계
                recent_corr = np.corrcoef(btc_returns[-10:], vix_returns[-10:])[0,1]
                historical_corr = np.corrcoef(btc_returns[:-10], vix_returns[:-10])[0,1]
                
                if not (np.isnan(recent_corr) or np.isnan(historical_corr)):
                    corr_change = abs(recent_corr - historical_corr)
                    correlation_breakdown += corr_change
                    
            return {
                "correlation_breakdown_score": float(correlation_breakdown),
                "composite_score": min(correlation_breakdown / 0.5, 1.0)  # 0.5 상관관계 변화를 1.0으로
            }
            
        except Exception as e:
            self.logger.error(f"상관관계 분석 실패: {e}")
            return {"composite_score": 0, "error": str(e)}

    def calculate_composite_risk(self, risk_components: Dict) -> Dict:
        """종합 위험도 계산"""
        try:
            # 가중치 설정 (합계 = 1.0)
            weights = {
                "sudden_change": 0.30,    # 급변이 가장 중요
                "pattern_match": 0.25,    # 과거 패턴 매칭
                "anomaly": 0.20,         # 이상 감지
                "trend_change": 0.15,     # 추세 변화
                "correlation": 0.10       # 상관관계 파괴
            }
            
            # 각 컴포넌트 점수 추출
            scores = {}
            for component, weight in weights.items():
                component_data = risk_components.get(component, {})
                scores[component] = component_data.get("composite_score", 0) * weight
                
            # 종합 점수 계산
            total_score = sum(scores.values())
            
            # 위험 레벨 결정
            if total_score >= 0.8:
                risk_level = "CRITICAL"
                confidence = 0.9
            elif total_score >= 0.6:
                risk_level = "WARNING"  
                confidence = 0.8
            elif total_score >= 0.4:
                risk_level = "INFO"
                confidence = 0.7
            else:
                risk_level = "LOW"
                confidence = 0.6
                
            return {
                "total_score": float(total_score),
                "risk_level": risk_level,
                "confidence": confidence,
                "component_scores": scores,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"종합 위험도 계산 실패: {e}")
            return {
                "total_score": 0.5,
                "risk_level": "WARNING",
                "confidence": 0.5,
                "error": str(e)
            }

    def extract_features(self, data: Dict, historical_data: List[Dict]) -> Dict:
        """데이터에서 특징 추출"""
        features = {}
        
        try:
            # 가격 관련 특징
            if "price_data" in data:
                features["current_price"] = data["price_data"].get("current_price", 0)
                features["volume_24h"] = data["price_data"].get("volume_24h", 0)
                features["change_24h"] = data["price_data"].get("change_24h", 0)
                
            # 거시경제 특징
            if "macro_data" in data:
                if "vix" in data["macro_data"]:
                    features["vix_level"] = data["macro_data"]["vix"]["current"]
                    features["vix_change"] = data["macro_data"]["vix"]["change"]
                if "dxy" in data["macro_data"]:
                    features["dxy_level"] = data["macro_data"]["dxy"]["current"]
                    features["dxy_change"] = data["macro_data"]["dxy"]["change"]
                    
            # 센티먼트 특징
            if "sentiment_data" in data and "fear_greed" in data["sentiment_data"]:
                features["fear_greed_index"] = data["sentiment_data"]["fear_greed"]["current_index"]
                
            return features
            
        except Exception as e:
            self.logger.error(f"특징 추출 실패: {e}")
            return {}

    def extract_numerical_features(self, data: Dict) -> Optional[List[float]]:
        """수치형 특징만 추출 (ML용)"""
        try:
            features = []
            
            # 가격 데이터
            if "price_data" in data:
                price_data = data["price_data"]
                features.extend([
                    price_data.get("current_price", 0),
                    price_data.get("volume_24h", 0),
                    price_data.get("change_24h", 0),
                    price_data.get("market_cap", 0)
                ])
                
            # 거시경제 데이터
            if "macro_data" in data:
                macro_data = data["macro_data"]
                if "vix" in macro_data:
                    features.extend([
                        macro_data["vix"].get("current", 20),
                        macro_data["vix"].get("change", 0)
                    ])
                else:
                    features.extend([20, 0])  # 기본값
                    
                if "dxy" in macro_data:
                    features.extend([
                        macro_data["dxy"].get("current", 100),
                        macro_data["dxy"].get("change", 0)
                    ])
                else:
                    features.extend([100, 0])  # 기본값
            else:
                features.extend([20, 0, 100, 0])  # 기본값들
                
            # 센티먼트 데이터
            if "sentiment_data" in data and "fear_greed" in data["sentiment_data"]:
                features.append(data["sentiment_data"]["fear_greed"]["current_index"])
            else:
                features.append(50)  # 중립값
                
            return features if len(features) > 0 else None
            
        except Exception as e:
            self.logger.error(f"수치형 특징 추출 실패: {e}")
            return None

    def calculate_pattern_similarity(self, current_features: Dict, pattern_features: Dict) -> float:
        """패턴 유사도 계산"""
        try:
            if not current_features or not pattern_features:
                return 0.0
                
            # 공통 특징만 비교
            common_features = set(current_features.keys()) & set(pattern_features.keys())
            
            if not common_features:
                return 0.0
                
            similarities = []
            for feature in common_features:
                current_val = current_features[feature]
                pattern_val = pattern_features[feature]
                
                # 정규화된 차이 계산
                if pattern_val != 0:
                    diff = abs(current_val - pattern_val) / abs(pattern_val)
                    similarity = max(0, 1 - diff)
                    similarities.append(similarity)
                    
            return np.mean(similarities) if similarities else 0.0
            
        except Exception as e:
            self.logger.error(f"유사도 계산 실패: {e}")
            return 0.0

    def get_pattern_risk_level(self, pattern_name: str) -> str:
        """패턴별 위험 레벨 반환"""
        risk_levels = {
            "flash_crash_2022": "CRITICAL",
            "luna_collapse_2022": "CRITICAL",
            "covid_crash_2020": "CRITICAL",
            "china_ban_2021": "WARNING"
        }
        return risk_levels.get(pattern_name, "INFO")

    def get_pattern_severity(self, pattern_name: str) -> float:
        """패턴별 심각도 반환"""
        severities = {
            "flash_crash_2022": 0.9,
            "luna_collapse_2022": 0.95,
            "covid_crash_2020": 1.0,
            "china_ban_2021": 0.8
        }
        return severities.get(pattern_name, 0.5)

    def generate_recommendations(self, composite_risk: Dict) -> List[str]:
        """위험도에 따른 권장사항 생성"""
        recommendations = []
        risk_level = composite_risk["risk_level"]
        total_score = composite_risk["total_score"]
        
        if risk_level == "CRITICAL":
            recommendations.extend([
                "즉시 레버리지 포지션 점검 필요",
                "손절가 상향 조정 권장",
                "포지션 크기 축소 고려",
                "15분 내 재분석 예정"
            ])
        elif risk_level == "WARNING":
            recommendations.extend([
                "포지션 관리 점검 권장",
                "시장 변화 주의 깊게 모니터링",
                "1시간 후 재평가"
            ])
        elif risk_level == "INFO":
            recommendations.extend([
                "일반적인 시장 모니터링 지속",
                "4시간 후 정기 점검"
            ])
            
        return recommendations

    def calculate_next_check_time(self, risk_level: str) -> str:
        """다음 체크 시점 계산"""
        intervals = {
            "CRITICAL": 5,   # 5분
            "WARNING": 30,   # 30분
            "INFO": 120,     # 2시간
            "LOW": 240       # 4시간
        }
        
        minutes = intervals.get(risk_level, 60)
        next_time = datetime.utcnow() + timedelta(minutes=minutes)
        return next_time.isoformat()

# 테스트 함수
def test_risk_analyzer():
    """위험 분석기 테스트"""
    print("🧠 위험 분석기 테스트 시작...")
    
    analyzer = TimeSeriesRiskAnalyzer()
    
    # 테스트 데이터 생성
    current_data = {
        "price_data": {
            "current_price": 60000,
            "volume_24h": 30000000000,
            "change_24h": -8.5
        },
        "macro_data": {
            "vix": {"current": 28, "change": 5.2},
            "dxy": {"current": 103, "change": 0.8}
        },
        "sentiment_data": {
            "fear_greed": {"current_index": 25}
        }
    }
    
    # 가짜 히스토리컬 데이터 생성
    historical_data = []
    base_price = 65000
    for i in range(100):
        hist_point = {
            "timestamp": (datetime.utcnow() - timedelta(minutes=i)).isoformat(),
            "price_data": {
                "current_price": base_price + np.random.normal(0, 1000),
                "volume_24h": 25000000000 + np.random.normal(0, 5000000000),
                "change_24h": np.random.normal(0, 3)
            },
            "macro_data": {
                "vix": {"current": 22 + np.random.normal(0, 2), "change": np.random.normal(0, 1)},
                "dxy": {"current": 102 + np.random.normal(0, 1), "change": np.random.normal(0, 0.5)}
            }
        }
        historical_data.append(hist_point)
    
    # 분석 실행
    risk_analysis = analyzer.analyze_timeseries_risk(current_data, historical_data)
    
    print("✅ 위험 분석 결과:")
    print(f"  종합 위험도: {risk_analysis['composite_risk_score']:.3f}")
    print(f"  위험 레벨: {risk_analysis['risk_level']}")
    print(f"  신뢰도: {risk_analysis['confidence']:.3f}")
    print(f"  다음 체크: {risk_analysis['next_check_in']}")
    
    print("\n  컴포넌트별 점수:")
    for component, data in risk_analysis['components'].items():
        score = data.get('composite_score', 0)
        print(f"    {component}: {score:.3f}")
        
    print("\n  권장사항:")
    for i, rec in enumerate(risk_analysis['recommendations'], 1):
        print(f"    {i}. {rec}")
    
    return risk_analysis

if __name__ == "__main__":
    test_risk_analyzer()