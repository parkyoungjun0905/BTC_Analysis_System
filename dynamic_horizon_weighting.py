#!/usr/bin/env python3
"""
🎯 Dynamic Horizon Weighting System
동적 시간대 가중치 시스템 - 시장 변동성과 성능 기반 실시간 가중치 최적화

주요 기능:
1. Market Regime Detection - 시장 체제 감지 및 분류
2. Volatility-Based Weighting - 변동성 기반 동적 가중치
3. Performance Tracking - 시간대별 성능 추적 및 학습
4. Adaptive Optimization - 실시간 가중치 최적화
5. Risk-Return Balance - 위험-수익 균형 최적화
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import deque, defaultdict
import json
import logging
from scipy import stats
from scipy.optimize import minimize
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import warnings

warnings.filterwarnings('ignore')

@dataclass
class MarketRegime:
    """시장 체제 정의"""
    name: str
    volatility_range: Tuple[float, float]
    trend_strength_range: Tuple[float, float]
    volume_factor: float
    typical_duration: int  # 시간 단위
    characteristics: Dict[str, float] = field(default_factory=dict)

@dataclass
class HorizonPerformance:
    """시간대별 성능 추적"""
    horizon: int
    accuracy_history: deque = field(default_factory=lambda: deque(maxlen=100))
    mae_history: deque = field(default_factory=lambda: deque(maxlen=100))
    directional_accuracy: deque = field(default_factory=lambda: deque(maxlen=100))
    volatility_conditions: deque = field(default_factory=lambda: deque(maxlen=100))
    last_updated: datetime = field(default_factory=datetime.now)
    
    @property
    def recent_performance(self) -> Dict[str, float]:
        """최근 성능 요약"""
        if not self.accuracy_history:
            return {'accuracy': 0.0, 'mae': float('inf'), 'direction': 0.5}
        
        return {
            'accuracy': float(np.mean(list(self.accuracy_history)[-10:])),
            'mae': float(np.mean(list(self.mae_history)[-10:])),
            'direction': float(np.mean(list(self.directional_accuracy)[-10:]))
        }

class MarketRegimeDetector:
    """시장 체제 감지기"""
    
    def __init__(self):
        self.regimes = self._define_market_regimes()
        self.regime_classifier = None
        self.feature_scaler = StandardScaler()
        self.regime_history = deque(maxlen=168)  # 1주일 히스토리
        
    def _define_market_regimes(self) -> Dict[str, MarketRegime]:
        """시장 체제 정의"""
        return {
            'low_volatility_bull': MarketRegime(
                name='Low Volatility Bull',
                volatility_range=(0.0, 0.03),
                trend_strength_range=(0.5, 1.0),
                volume_factor=0.8,
                typical_duration=72,
                characteristics={'risk_preference': 'low', 'trend_following': 'strong'}
            ),
            'high_volatility_bull': MarketRegime(
                name='High Volatility Bull',
                volatility_range=(0.03, 0.10),
                trend_strength_range=(0.3, 0.8),
                volume_factor=1.2,
                typical_duration=48,
                characteristics={'risk_preference': 'high', 'trend_following': 'medium'}
            ),
            'low_volatility_bear': MarketRegime(
                name='Low Volatility Bear',
                volatility_range=(0.0, 0.03),
                trend_strength_range=(-1.0, -0.3),
                volume_factor=0.9,
                typical_duration=96,
                characteristics={'risk_preference': 'low', 'trend_following': 'strong'}
            ),
            'high_volatility_bear': MarketRegime(
                name='High Volatility Bear',
                volatility_range=(0.03, 0.15),
                trend_strength_range=(-1.0, -0.2),
                volume_factor=1.5,
                typical_duration=24,
                characteristics={'risk_preference': 'extreme', 'trend_following': 'weak'}
            ),
            'sideways_low_vol': MarketRegime(
                name='Sideways Low Volatility',
                volatility_range=(0.0, 0.02),
                trend_strength_range=(-0.3, 0.3),
                volume_factor=0.7,
                typical_duration=120,
                characteristics={'risk_preference': 'low', 'trend_following': 'none'}
            ),
            'sideways_high_vol': MarketRegime(
                name='Sideways High Volatility',
                volatility_range=(0.02, 0.08),
                trend_strength_range=(-0.4, 0.4),
                volume_factor=1.1,
                typical_duration=36,
                characteristics={'risk_preference': 'medium', 'trend_following': 'weak'}
            ),
            'extreme_volatility': MarketRegime(
                name='Extreme Volatility',
                volatility_range=(0.10, 1.0),
                trend_strength_range=(-1.0, 1.0),
                volume_factor=2.0,
                typical_duration=12,
                characteristics={'risk_preference': 'extreme', 'trend_following': 'chaotic'}
            )
        }
    
    def extract_market_features(self, price_data: np.ndarray, volume_data: np.ndarray = None) -> np.ndarray:
        """시장 특성 추출"""
        if len(price_data) < 24:
            return np.zeros(10)
        
        # 가격 기반 특성
        returns = np.diff(price_data) / price_data[:-1]
        
        # 1. 변동성 (다양한 시간창)
        vol_1h = np.std(returns[-1:]) if len(returns) >= 1 else 0
        vol_4h = np.std(returns[-4:]) if len(returns) >= 4 else 0
        vol_24h = np.std(returns[-24:]) if len(returns) >= 24 else 0
        
        # 2. 트렌드 강도
        if len(price_data) >= 24:
            trend_slope = np.polyfit(range(24), price_data[-24:], 1)[0]
            trend_strength = abs(trend_slope) / np.mean(price_data[-24:])
            trend_direction = np.sign(trend_slope)
        else:
            trend_strength = trend_direction = 0
        
        # 3. 가격 모멘텀
        momentum_4h = (price_data[-1] - price_data[-5]) / price_data[-5] if len(price_data) >= 5 else 0
        momentum_24h = (price_data[-1] - price_data[-25]) / price_data[-25] if len(price_data) >= 25 else 0
        
        # 4. 변동성의 변동성
        if len(returns) >= 24:
            rolling_vol = [np.std(returns[i:i+6]) for i in range(len(returns)-5) if i+6 <= len(returns)]
            vol_of_vol = np.std(rolling_vol) if len(rolling_vol) > 1 else 0
        else:
            vol_of_vol = 0
        
        # 5. 볼륨 특성 (사용 가능한 경우)
        if volume_data is not None and len(volume_data) >= 24:
            volume_trend = np.polyfit(range(24), volume_data[-24:], 1)[0]
            volume_volatility = np.std(volume_data[-24:]) / np.mean(volume_data[-24:])
        else:
            volume_trend = volume_volatility = 0
        
        features = np.array([
            vol_1h, vol_4h, vol_24h,
            trend_strength, trend_direction,
            momentum_4h, momentum_24h,
            vol_of_vol,
            volume_trend, volume_volatility
        ])
        
        return features
    
    def detect_regime(self, price_data: np.ndarray, volume_data: np.ndarray = None) -> str:
        """현재 시장 체제 감지"""
        features = self.extract_market_features(price_data, volume_data)
        
        # 간단한 규칙 기반 분류 (ML 분류기가 없는 경우)
        if self.regime_classifier is None:
            return self._rule_based_regime_detection(features)
        
        # ML 기반 분류
        features_scaled = self.feature_scaler.transform(features.reshape(1, -1))
        regime_name = self.regime_classifier.predict(features_scaled)[0]
        
        # 히스토리에 추가
        self.regime_history.append({
            'regime': regime_name,
            'timestamp': datetime.now(),
            'features': features.tolist()
        })
        
        return regime_name
    
    def _rule_based_regime_detection(self, features: np.ndarray) -> str:
        """규칙 기반 체제 감지"""
        vol_1h, vol_4h, vol_24h, trend_strength, trend_direction, momentum_4h, momentum_24h, vol_of_vol, volume_trend, volume_volatility = features
        
        # 극단 변동성 체크
        if vol_24h > 0.10:
            return 'extreme_volatility'
        
        # 트렌드 방향 판단
        is_bullish = trend_direction > 0 and momentum_24h > 0.02
        is_bearish = trend_direction < 0 and momentum_24h < -0.02
        is_sideways = abs(momentum_24h) < 0.02 and trend_strength < 0.3
        
        # 변동성 레벨
        is_high_vol = vol_24h > 0.03
        
        # 체제 분류
        if is_bullish:
            return 'high_volatility_bull' if is_high_vol else 'low_volatility_bull'
        elif is_bearish:
            return 'high_volatility_bear' if is_high_vol else 'low_volatility_bear'
        elif is_sideways:
            return 'sideways_high_vol' if is_high_vol else 'sideways_low_vol'
        else:
            return 'sideways_low_vol'  # 기본값
    
    def get_regime_stability(self) -> float:
        """체제 안정성 측정 (0-1)"""
        if len(self.regime_history) < 10:
            return 0.5
        
        recent_regimes = [r['regime'] for r in list(self.regime_history)[-10:]]
        most_common = max(set(recent_regimes), key=recent_regimes.count)
        stability = recent_regimes.count(most_common) / len(recent_regimes)
        
        return stability
    
    def train_regime_classifier(self, historical_data: List[Dict]):
        """과거 데이터로 체제 분류기 훈련"""
        if len(historical_data) < 100:
            logging.warning("체제 분류기 훈련을 위한 데이터가 부족합니다")
            return
        
        features = []
        labels = []
        
        for data_point in historical_data:
            if 'features' in data_point and 'regime' in data_point:
                features.append(data_point['features'])
                labels.append(data_point['regime'])
        
        if len(features) > 50:
            X = np.array(features)
            y = np.array(labels)
            
            # 특성 정규화
            X_scaled = self.feature_scaler.fit_transform(X)
            
            # 랜덤 포레스트 분류기 훈련
            self.regime_classifier = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            self.regime_classifier.fit(X_scaled, y)
            
            logging.info(f"체제 분류기 훈련 완료: {len(features)}개 샘플")

class VolatilityAnalyzer:
    """변동성 분석기"""
    
    def __init__(self):
        self.volatility_history = deque(maxlen=168)  # 1주일
        self.garch_params = None
        
    def calculate_realized_volatility(self, returns: np.ndarray, window: int = 24) -> float:
        """실현 변동성 계산"""
        if len(returns) < window:
            return np.std(returns) if len(returns) > 1 else 0.0
        
        # 연율화된 변동성 (24시간 기준)
        realized_vol = np.sqrt(24) * np.std(returns[-window:])
        return float(realized_vol)
    
    def calculate_garch_volatility(self, returns: np.ndarray) -> float:
        """GARCH 모델 기반 변동성 예측 (단순화 버전)"""
        if len(returns) < 50:
            return self.calculate_realized_volatility(returns)
        
        # 단순화된 GARCH(1,1) 추정
        returns_sq = returns ** 2
        
        # 초기 파라미터 (일반적인 값들)
        omega = 0.000001  # 상수항
        alpha = 0.1       # ARCH 계수
        beta = 0.85       # GARCH 계수
        
        # 조건부 분산 계산
        conditional_variance = np.zeros(len(returns))
        conditional_variance[0] = np.var(returns)
        
        for t in range(1, len(returns)):
            conditional_variance[t] = (omega + 
                                     alpha * returns_sq[t-1] + 
                                     beta * conditional_variance[t-1])
        
        # 다음 기간 변동성 예측
        next_vol = np.sqrt(omega + alpha * returns_sq[-1] + beta * conditional_variance[-1])
        
        return float(next_vol)
    
    def get_volatility_regime(self, current_vol: float) -> str:
        """변동성 체제 분류"""
        if current_vol < 0.02:
            return 'very_low'
        elif current_vol < 0.05:
            return 'low'
        elif current_vol < 0.10:
            return 'medium'
        elif current_vol < 0.20:
            return 'high'
        else:
            return 'extreme'
    
    def predict_volatility_persistence(self, returns: np.ndarray) -> float:
        """변동성 지속성 예측 (0-1)"""
        if len(returns) < 20:
            return 0.5
        
        # 변동성 자기상관 계산
        volatilities = [np.std(returns[i:i+5]) for i in range(len(returns)-4)]
        
        if len(volatilities) < 2:
            return 0.5
        
        # 1차 자기상관
        autocorr = np.corrcoef(volatilities[:-1], volatilities[1:])[0, 1]
        
        # NaN 처리
        if np.isnan(autocorr):
            return 0.5
        
        # 0-1 범위로 정규화
        persistence = (autocorr + 1) / 2
        
        return float(max(0, min(1, persistence)))

class PerformanceTracker:
    """성능 추적기"""
    
    def __init__(self, horizons: List[int]):
        self.horizons = horizons
        self.performance_data = {h: HorizonPerformance(h) for h in horizons}
        self.regime_performance = defaultdict(lambda: defaultdict(list))
        
    def update_performance(self, horizon: int, actual: float, predicted: float, 
                          market_regime: str, volatility: float):
        """성능 업데이트"""
        if horizon not in self.performance_data:
            return
        
        perf = self.performance_data[horizon]
        
        # 정확도 계산 (MAPE 기반)
        accuracy = max(0, 1 - abs(actual - predicted) / abs(actual)) if actual != 0 else 0
        mae = abs(actual - predicted)
        
        # 방향 정확도
        actual_direction = 1 if actual > 0 else 0
        pred_direction = 1 if predicted > 0 else 0
        direction_accuracy = 1 if actual_direction == pred_direction else 0
        
        # 히스토리 업데이트
        perf.accuracy_history.append(accuracy)
        perf.mae_history.append(mae)
        perf.directional_accuracy.append(direction_accuracy)
        perf.volatility_conditions.append(volatility)
        perf.last_updated = datetime.now()
        
        # 체제별 성능 기록
        self.regime_performance[market_regime][horizon].append({
            'accuracy': accuracy,
            'mae': mae,
            'direction': direction_accuracy,
            'timestamp': datetime.now()
        })
    
    def get_horizon_ranking(self, market_regime: str = None, lookback: int = 20) -> List[Tuple[int, float]]:
        """시간대별 성능 순위"""
        scores = []
        
        for horizon in self.horizons:
            perf = self.performance_data[horizon]
            
            if market_regime and market_regime in self.regime_performance:
                # 특정 체제에서의 성능
                regime_data = self.regime_performance[market_regime][horizon][-lookback:]
                if regime_data:
                    score = np.mean([d['accuracy'] for d in regime_data])
                else:
                    score = 0.5  # 기본값
            else:
                # 전반적 성능
                recent_performance = perf.recent_performance
                score = (recent_performance['accuracy'] + recent_performance['direction']) / 2
            
            scores.append((horizon, score))
        
        # 성능 순으로 정렬
        scores.sort(key=lambda x: x[1], reverse=True)
        
        return scores
    
    def calculate_performance_stability(self, horizon: int) -> float:
        """성능 안정성 계산"""
        if horizon not in self.performance_data:
            return 0.0
        
        perf = self.performance_data[horizon]
        
        if len(perf.accuracy_history) < 5:
            return 0.0
        
        accuracies = list(perf.accuracy_history)[-20:]  # 최근 20개
        stability = 1 - np.std(accuracies)  # 변동성이 낮을수록 안정적
        
        return float(max(0, stability))

class WeightOptimizer:
    """가중치 최적화기"""
    
    def __init__(self, horizons: List[int]):
        self.horizons = horizons
        self.optimization_history = []
        
    def optimize_weights(self, performance_scores: Dict[int, float], 
                        market_regime: str, 
                        volatility_level: float,
                        risk_tolerance: float = 0.5) -> Dict[int, float]:
        """다목적 최적화로 가중치 계산"""
        
        def objective_function(weights):
            """목적 함수: 성능 극대화 + 위험 최소화"""
            weights_dict = dict(zip(self.horizons, weights))
            
            # 성능 점수
            performance_score = sum(weights_dict[h] * performance_scores.get(h, 0.5) 
                                  for h in self.horizons)
            
            # 위험 점수 (분산 최소화)
            risk_score = np.var(list(weights_dict.values()))
            
            # 다양성 점수 (균등 분산 선호)
            diversity_penalty = sum(abs(w - 1/len(self.horizons)) for w in weights)
            
            # 체제별 조정
            regime_adjustment = self._get_regime_adjustment(weights_dict, market_regime, volatility_level)
            
            # 복합 점수
            total_score = (performance_score - 
                          risk_tolerance * risk_score - 
                          0.1 * diversity_penalty +
                          regime_adjustment)
            
            return -total_score  # 최대화를 위해 음수 반환
        
        # 제약 조건: 가중치 합 = 1, 각 가중치 >= 0
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
        ]
        bounds = [(0, 1) for _ in self.horizons]
        
        # 초기 가중치 (균등)
        x0 = np.array([1.0 / len(self.horizons)] * len(self.horizons))
        
        # 최적화 실행
        result = minimize(
            objective_function,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if result.success:
            optimized_weights = dict(zip(self.horizons, result.x))
        else:
            # 최적화 실패시 성능 기반 가중치
            optimized_weights = self._fallback_weighting(performance_scores)
        
        # 최적화 기록
        self.optimization_history.append({
            'timestamp': datetime.now(),
            'market_regime': market_regime,
            'volatility_level': volatility_level,
            'weights': optimized_weights,
            'optimization_success': result.success if hasattr(result, 'success') else False
        })
        
        return optimized_weights
    
    def _get_regime_adjustment(self, weights: Dict[int, float], 
                             market_regime: str, volatility_level: float) -> float:
        """체제별 가중치 조정"""
        adjustment = 0.0
        
        # 체제별 선호 시간대
        regime_preferences = {
            'low_volatility_bull': {168: 0.3, 72: 0.3, 24: 0.2, 4: 0.1, 1: 0.1},
            'high_volatility_bull': {24: 0.3, 4: 0.3, 1: 0.2, 72: 0.1, 168: 0.1},
            'low_volatility_bear': {168: 0.4, 72: 0.3, 24: 0.2, 4: 0.05, 1: 0.05},
            'high_volatility_bear': {1: 0.4, 4: 0.3, 24: 0.2, 72: 0.07, 168: 0.03},
            'sideways_low_vol': {72: 0.3, 24: 0.3, 168: 0.2, 4: 0.1, 1: 0.1},
            'sideways_high_vol': {4: 0.3, 24: 0.3, 1: 0.2, 72: 0.1, 168: 0.1},
            'extreme_volatility': {1: 0.5, 4: 0.3, 24: 0.15, 72: 0.03, 168: 0.02}
        }
        
        if market_regime in regime_preferences:
            preferred_weights = regime_preferences[market_regime]
            
            # 선호도와 실제 가중치 간 일치도 계산
            for horizon in self.horizons:
                preferred = preferred_weights.get(horizon, 0.2)
                actual = weights.get(horizon, 0.2)
                alignment = 1 - abs(preferred - actual)
                adjustment += alignment * 0.1  # 조정 강도
        
        return adjustment
    
    def _fallback_weighting(self, performance_scores: Dict[int, float]) -> Dict[int, float]:
        """대안 가중치 계산"""
        total_score = sum(performance_scores.values())
        
        if total_score <= 0:
            # 균등 가중치
            return {h: 1.0 / len(self.horizons) for h in self.horizons}
        
        # 성능 비례 가중치
        weights = {h: score / total_score for h, score in performance_scores.items()}
        
        return weights

class DynamicHorizonWeightingSystem:
    """동적 시간대 가중치 시스템"""
    
    def __init__(self, horizons: List[int] = [1, 4, 24, 72, 168]):
        self.horizons = horizons
        
        # 구성 요소 초기화
        self.regime_detector = MarketRegimeDetector()
        self.volatility_analyzer = VolatilityAnalyzer()
        self.performance_tracker = PerformanceTracker(horizons)
        self.weight_optimizer = WeightOptimizer(horizons)
        
        # 상태 추적
        self.current_weights = {h: 1.0 / len(horizons) for h in horizons}
        self.weight_history = []
        self.system_performance = []
        
        # 설정
        self.rebalance_frequency = 6  # 6시간마다 재조정
        self.last_rebalance = datetime.now()
        
        self.setup_logging()
    
    def setup_logging(self):
        """로깅 설정"""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def update_market_data(self, price_data: np.ndarray, volume_data: np.ndarray = None):
        """시장 데이터 업데이트"""
        if len(price_data) < 24:
            self.logger.warning("가격 데이터가 부족합니다 (24시간 미만)")
            return
        
        # 현재 시장 체제 감지
        current_regime = self.regime_detector.detect_regime(price_data, volume_data)
        
        # 변동성 분석
        returns = np.diff(price_data) / price_data[:-1]
        current_volatility = self.volatility_analyzer.calculate_realized_volatility(returns)
        volatility_regime = self.volatility_analyzer.get_volatility_regime(current_volatility)
        
        # 변동성 지속성 예측
        vol_persistence = self.volatility_analyzer.predict_volatility_persistence(returns)
        
        self.logger.info(f"시장 체제: {current_regime}, 변동성: {current_volatility:.4f} ({volatility_regime})")
        
        return {
            'market_regime': current_regime,
            'volatility': current_volatility,
            'volatility_regime': volatility_regime,
            'volatility_persistence': vol_persistence
        }
    
    def update_performance(self, horizon: int, actual: float, predicted: float, 
                          market_regime: str, volatility: float):
        """성능 업데이트"""
        self.performance_tracker.update_performance(
            horizon, actual, predicted, market_regime, volatility
        )
        
        # 시스템 전체 성능 기록
        accuracy = max(0, 1 - abs(actual - predicted) / abs(actual)) if actual != 0 else 0
        
        self.system_performance.append({
            'timestamp': datetime.now(),
            'horizon': horizon,
            'accuracy': accuracy,
            'market_regime': market_regime,
            'volatility': volatility
        })
        
        # 성능 기록 제한 (메모리 관리)
        if len(self.system_performance) > 1000:
            self.system_performance = self.system_performance[-800:]
    
    def rebalance_weights(self, market_data: Dict, risk_tolerance: float = 0.5) -> Dict[int, float]:
        """가중치 재조정"""
        current_time = datetime.now()
        time_since_rebalance = (current_time - self.last_rebalance).total_seconds() / 3600
        
        # 재조정 주기 확인 또는 체제 변화시 강제 재조정
        regime_changed = self._detect_regime_change()
        
        if time_since_rebalance < self.rebalance_frequency and not regime_changed:
            return self.current_weights
        
        self.logger.info("🔄 가중치 재조정 시작")
        
        # 현재 성능 점수 계산
        market_regime = market_data.get('market_regime', 'sideways_low_vol')
        performance_ranking = self.performance_tracker.get_horizon_ranking(market_regime)
        
        performance_scores = dict(performance_ranking)
        
        # 가중치 최적화
        optimized_weights = self.weight_optimizer.optimize_weights(
            performance_scores=performance_scores,
            market_regime=market_regime,
            volatility_level=market_data.get('volatility', 0.05),
            risk_tolerance=risk_tolerance
        )
        
        # 가중치 변화량 제한 (안정성 확보)
        dampened_weights = self._dampen_weight_changes(optimized_weights, damping_factor=0.3)
        
        # 가중치 업데이트
        self.current_weights = dampened_weights
        self.last_rebalance = current_time
        
        # 히스토리 저장
        self.weight_history.append({
            'timestamp': current_time,
            'weights': dampened_weights.copy(),
            'market_regime': market_regime,
            'performance_scores': performance_scores,
            'volatility': market_data.get('volatility', 0.0)
        })
        
        self.logger.info(f"✅ 가중치 재조정 완료: {dampened_weights}")
        
        return dampened_weights
    
    def _detect_regime_change(self) -> bool:
        """체제 변화 감지"""
        if len(self.regime_detector.regime_history) < 5:
            return False
        
        recent_regimes = [r['regime'] for r in list(self.regime_detector.regime_history)[-5:]]
        
        # 최근 5개 중 3개 이상이 다른 체제면 변화로 감지
        most_common = max(set(recent_regimes), key=recent_regimes.count)
        change_ratio = 1 - recent_regimes.count(most_common) / len(recent_regimes)
        
        return change_ratio > 0.4
    
    def _dampen_weight_changes(self, new_weights: Dict[int, float], 
                              damping_factor: float = 0.2) -> Dict[int, float]:
        """가중치 변화량 완화"""
        dampened_weights = {}
        
        for horizon in self.horizons:
            current_weight = self.current_weights.get(horizon, 1.0 / len(self.horizons))
            new_weight = new_weights.get(horizon, 1.0 / len(self.horizons))
            
            # 지수 이동 평균 방식으로 완화
            dampened_weight = current_weight * (1 - damping_factor) + new_weight * damping_factor
            dampened_weights[horizon] = dampened_weight
        
        # 정규화
        total_weight = sum(dampened_weights.values())
        if total_weight > 0:
            dampened_weights = {h: w / total_weight for h, w in dampened_weights.items()}
        
        return dampened_weights
    
    def get_current_strategy(self) -> Dict:
        """현재 전략 상태 반환"""
        recent_regime = 'unknown'
        if self.regime_detector.regime_history:
            recent_regime = self.regime_detector.regime_history[-1]['regime']
        
        # 성능 순위
        performance_ranking = self.performance_tracker.get_horizon_ranking()
        
        # 체제 안정성
        regime_stability = self.regime_detector.get_regime_stability()
        
        return {
            'current_weights': self.current_weights,
            'market_regime': recent_regime,
            'regime_stability': regime_stability,
            'performance_ranking': performance_ranking,
            'last_rebalance': self.last_rebalance.isoformat(),
            'system_status': {
                'total_updates': len(self.system_performance),
                'avg_recent_accuracy': self._calculate_recent_system_accuracy(),
                'weight_diversity': self._calculate_weight_diversity(),
                'adaptation_rate': len(self.weight_history)
            }
        }
    
    def _calculate_recent_system_accuracy(self) -> float:
        """최근 시스템 정확도 계산"""
        if len(self.system_performance) < 10:
            return 0.0
        
        recent_performance = self.system_performance[-50:]
        accuracies = [p['accuracy'] for p in recent_performance]
        
        return float(np.mean(accuracies))
    
    def _calculate_weight_diversity(self) -> float:
        """가중치 다양성 계산 (엔트로피 기반)"""
        weights = list(self.current_weights.values())
        
        # 샤논 엔트로피
        entropy = -sum(w * np.log(w + 1e-10) for w in weights)
        max_entropy = np.log(len(weights))
        
        diversity = entropy / max_entropy if max_entropy > 0 else 0
        
        return float(diversity)
    
    def save_system_state(self, filepath: str):
        """시스템 상태 저장"""
        state_data = {
            'horizons': self.horizons,
            'current_weights': self.current_weights,
            'weight_history': [
                {**entry, 'timestamp': entry['timestamp'].isoformat()}
                for entry in self.weight_history
            ],
            'system_performance': [
                {**entry, 'timestamp': entry['timestamp'].isoformat()}
                for entry in self.system_performance
            ],
            'regime_history': [
                {**entry, 'timestamp': entry['timestamp'].isoformat()}
                for entry in self.regime_detector.regime_history
            ],
            'last_rebalance': self.last_rebalance.isoformat(),
            'system_summary': {
                'total_rebalances': len(self.weight_history),
                'total_performance_updates': len(self.system_performance),
                'current_strategy': self.get_current_strategy()
            }
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(state_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"시스템 상태 저장 완료: {filepath}")

def main():
    """메인 테스트 함수"""
    # 테스트 데이터 생성
    np.random.seed(42)
    
    # 합성 BTC 가격 데이터 (다양한 체제 포함)
    time_points = 500
    
    # 다양한 시장 체제 시뮬레이션
    regimes = [
        ('low_volatility_bull', 100, 0.02, 0.001),      # 낮은 변동성 상승
        ('high_volatility_bull', 80, 0.08, 0.003),      # 높은 변동성 상승
        ('sideways_low_vol', 120, 0.015, 0.0),          # 횡보 낮은 변동성
        ('high_volatility_bear', 100, 0.12, -0.002),    # 높은 변동성 하락
        ('extreme_volatility', 100, 0.20, 0.001)        # 극단 변동성
    ]
    
    price_data = [50000.0]  # 시작 가격
    volume_data = []
    regime_labels = []
    
    for regime_name, duration, volatility, drift in regimes:
        for i in range(duration):
            # 가격 변동
            return_rate = np.random.normal(drift, volatility)
            new_price = price_data[-1] * (1 + return_rate)
            price_data.append(new_price)
            
            # 볼륨 (체제에 따라 다름)
            base_volume = 1000000
            if 'high_volatility' in regime_name or 'extreme' in regime_name:
                volume = base_volume * np.random.lognormal(0, 0.5)
            else:
                volume = base_volume * np.random.lognormal(0, 0.2)
            volume_data.append(volume)
            
            regime_labels.append(regime_name)
    
    price_data = np.array(price_data[1:])  # 첫 번째 시작가격 제외
    volume_data = np.array(volume_data)
    
    # 동적 가중치 시스템 초기화
    weighting_system = DynamicHorizonWeightingSystem()
    
    print("🎯 Dynamic Horizon Weighting System Test")
    print("="*60)
    print(f"📊 테스트 데이터: {len(price_data)} 시간, 5개 시장 체제")
    
    # 시스템 테스트 시뮬레이션
    for i in range(100, len(price_data), 24):  # 24시간마다 분석
        # 시장 데이터 업데이트
        current_data = price_data[max(0, i-168):i+1]  # 최근 1주일 데이터
        current_volume = volume_data[max(0, i-168):i+1]
        
        if len(current_data) < 24:
            continue
            
        market_data = weighting_system.update_market_data(current_data, current_volume)
        
        # 가상의 예측 성능 시뮬레이션
        current_regime = market_data['market_regime']
        current_volatility = market_data['volatility']
        
        for horizon in weighting_system.horizons:
            if i + horizon < len(price_data):
                # 실제 미래 가격
                actual_price = price_data[i + horizon]
                current_price = price_data[i]
                actual_return = (actual_price - current_price) / current_price
                
                # 체제별 예측 정확도 시뮬레이션 (현실적 편향 적용)
                regime_bias = {
                    'low_volatility_bull': {1: 0.85, 4: 0.80, 24: 0.75, 72: 0.70, 168: 0.65},
                    'high_volatility_bull': {1: 0.75, 4: 0.70, 24: 0.65, 72: 0.60, 168: 0.55},
                    'sideways_low_vol': {1: 0.60, 4: 0.65, 24: 0.70, 72: 0.75, 168: 0.70},
                    'high_volatility_bear': {1: 0.80, 4: 0.75, 24: 0.65, 72: 0.60, 168: 0.50},
                    'extreme_volatility': {1: 0.70, 4: 0.60, 24: 0.50, 72: 0.45, 168: 0.40}
                }
                
                base_accuracy = regime_bias.get(current_regime, {}).get(horizon, 0.6)
                noise_factor = np.random.normal(0, 0.1)  # 노이즈 추가
                
                predicted_return = actual_return * base_accuracy + noise_factor * current_volatility
                predicted_price = current_price * (1 + predicted_return)
                
                # 성능 업데이트
                weighting_system.update_performance(
                    horizon, actual_price, predicted_price, current_regime, current_volatility
                )
        
        # 가중치 재조정
        new_weights = weighting_system.rebalance_weights(market_data, risk_tolerance=0.4)
        
        # 진행률 출력
        if i % 96 == 0:  # 4일마다
            progress = (i - 100) / (len(price_data) - 100) * 100
            print(f"📈 진행률: {progress:.1f}% - 체제: {current_regime}, 변동성: {current_volatility:.4f}")
    
    # 최종 전략 상태
    final_strategy = weighting_system.get_current_strategy()
    
    print(f"\n🎯 최종 동적 가중치 전략:")
    print(f"="*50)
    
    print(f"📊 현재 가중치:")
    for horizon, weight in final_strategy['current_weights'].items():
        print(f"  {horizon}h: {weight:.3f}")
    
    print(f"\n🏛️ 시장 체제: {final_strategy['market_regime']}")
    print(f"🔒 체제 안정성: {final_strategy['regime_stability']:.3f}")
    
    print(f"\n🏆 성능 순위:")
    for i, (horizon, score) in enumerate(final_strategy['performance_ranking'][:3], 1):
        print(f"  {i}위: {horizon}h (점수: {score:.3f})")
    
    system_status = final_strategy['system_status']
    print(f"\n📈 시스템 상태:")
    print(f"  총 업데이트: {system_status['total_updates']}")
    print(f"  최근 정확도: {system_status['avg_recent_accuracy']:.3f}")
    print(f"  가중치 다양성: {system_status['weight_diversity']:.3f}")
    print(f"  적응 횟수: {system_status['adaptation_rate']}")
    
    # 가중치 변화 분석
    if len(weighting_system.weight_history) > 1:
        print(f"\n📊 가중치 진화 분석:")
        initial_weights = weighting_system.weight_history[0]['weights']
        final_weights = weighting_system.weight_history[-1]['weights']
        
        for horizon in weighting_system.horizons:
            change = final_weights[horizon] - initial_weights[horizon]
            direction = "증가" if change > 0 else "감소" if change < 0 else "유지"
            print(f"  {horizon}h: {initial_weights[horizon]:.3f} → {final_weights[horizon]:.3f} ({direction})")
    
    # 결과 저장
    weighting_system.save_system_state('dynamic_horizon_weighting_results.json')
    
    print(f"\n💾 결과 저장 완료: dynamic_horizon_weighting_results.json")
    
    return weighting_system

if __name__ == "__main__":
    main()