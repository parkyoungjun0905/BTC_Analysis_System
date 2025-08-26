#!/usr/bin/env python3
"""
🎯 Integration Strategies System
통합 전략 시스템 - 계층적 예측 통합 및 성능 최적화

주요 기능:
1. Hierarchical Prediction Reconciliation - 계층적 예측 조정
2. Multi-Objective Optimization - 다목적 최적화
3. Performance Attribution Analysis - 성과 기여도 분석
4. Cross-Horizon Validation - 교차 시간대 검증
5. Adaptive Ensemble Methods - 적응형 앙상블 방법
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, deque
import json
import logging
from scipy.optimize import minimize, differential_evolution
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

@dataclass
class PredictionRecord:
    """예측 기록"""
    timestamp: datetime
    horizon: int
    prediction: float
    confidence: float
    uncertainty: float
    method: str
    market_regime: str = ""
    volatility: float = 0.0
    features: List[float] = field(default_factory=list)

@dataclass
class IntegrationResult:
    """통합 결과"""
    integrated_predictions: Dict[int, float]
    confidence_scores: Dict[int, float]
    uncertainty_bounds: Dict[int, Tuple[float, float]]
    attribution_scores: Dict[str, float]
    performance_metrics: Dict[str, float]
    optimization_details: Dict = field(default_factory=dict)

class HierarchicalReconciliation:
    """계층적 예측 조정"""
    
    def __init__(self, horizons: List[int]):
        self.horizons = sorted(horizons)
        self.reconciliation_matrix = None
        self.hierarchy_constraints = self._build_hierarchy_constraints()
        
    def _build_hierarchy_constraints(self) -> Dict[str, List[Tuple[int, int, float]]]:
        """계층 구조 제약 조건 구축"""
        constraints = {
            'temporal_consistency': [],  # 시간적 일관성
            'trend_coherence': [],       # 트렌드 일관성
            'volatility_scaling': []     # 변동성 스케일링
        }
        
        # 시간적 일관성: 단기 예측이 장기 예측의 방향과 일치해야 함
        for i in range(len(self.horizons) - 1):
            short_h = self.horizons[i]
            long_h = self.horizons[i + 1]
            
            # 가중치: 시간 차이에 비례
            weight = 1.0 / (long_h - short_h)
            constraints['temporal_consistency'].append((short_h, long_h, weight))
        
        # 트렌드 일관성: 중장기 트렌드와 단기 예측 간 조화
        for short_h in self.horizons[:2]:  # 단기 (1h, 4h)
            for long_h in self.horizons[-2:]:  # 장기 (72h, 168h)
                weight = 0.5 / (long_h / short_h)
                constraints['trend_coherence'].append((short_h, long_h, weight))
        
        return constraints
    
    def reconcile_predictions(self, raw_predictions: Dict[int, float], 
                            confidence_scores: Dict[int, float],
                            current_price: float) -> Dict[int, float]:
        """예측값 계층적 조정"""
        if not raw_predictions:
            return {}
        
        # 신뢰도 기반 가중치 계산
        total_confidence = sum(confidence_scores.values())
        if total_confidence == 0:
            weights = {h: 1.0/len(raw_predictions) for h in raw_predictions.keys()}
        else:
            weights = {h: conf/total_confidence for h, conf in confidence_scores.items()}
        
        reconciled = raw_predictions.copy()
        
        # 반복적 조정 (5회)
        for iteration in range(5):
            prev_reconciled = reconciled.copy()
            
            # 시간적 일관성 조정
            reconciled = self._apply_temporal_consistency(reconciled, weights, current_price)
            
            # 트렌드 일관성 조정
            reconciled = self._apply_trend_coherence(reconciled, weights, current_price)
            
            # 변동성 스케일링
            reconciled = self._apply_volatility_scaling(reconciled, weights, current_price)
            
            # 수렴 검사
            changes = sum(abs(reconciled[h] - prev_reconciled[h]) for h in reconciled.keys())
            if changes < 1e-6:  # 수렴
                break
        
        return reconciled
    
    def _apply_temporal_consistency(self, predictions: Dict[int, float], 
                                  weights: Dict[int, float], 
                                  current_price: float) -> Dict[int, float]:
        """시간적 일관성 적용"""
        adjusted = predictions.copy()
        
        for short_h, long_h, constraint_weight in self.hierarchy_constraints['temporal_consistency']:
            if short_h in adjusted and long_h in adjusted:
                short_pred = adjusted[short_h]
                long_pred = adjusted[long_h]
                
                # 예상 수익률 계산
                short_return = (short_pred - current_price) / current_price
                long_return = (long_pred - current_price) / current_price
                
                # 연율화된 수익률로 비교 (시간 정규화)
                short_annual = short_return * (8760 / short_h)  # 연간 시간 수
                long_annual = long_return * (8760 / long_h)
                
                # 일관성 조정 (단기가 장기보다 과도하게 다르면 조정)
                if abs(short_annual - long_annual) > 0.5:  # 50% 이상 차이
                    # 신뢰도 가중 평균으로 조정
                    short_weight = weights.get(short_h, 0.5)
                    long_weight = weights.get(long_h, 0.5)
                    
                    total_weight = short_weight + long_weight
                    if total_weight > 0:
                        target_annual = (short_annual * short_weight + long_annual * long_weight) / total_weight
                        
                        # 조정 강도
                        adjustment_strength = constraint_weight * 0.1
                        
                        # 새로운 수익률 계산
                        new_short_return = target_annual * (short_h / 8760)
                        new_long_return = target_annual * (long_h / 8760)
                        
                        # 가격 업데이트 (부분적 조정)
                        adjusted[short_h] = short_pred * (1 - adjustment_strength) + \
                                          (current_price * (1 + new_short_return)) * adjustment_strength
                        adjusted[long_h] = long_pred * (1 - adjustment_strength) + \
                                         (current_price * (1 + new_long_return)) * adjustment_strength
        
        return adjusted
    
    def _apply_trend_coherence(self, predictions: Dict[int, float], 
                             weights: Dict[int, float], 
                             current_price: float) -> Dict[int, float]:
        """트렌드 일관성 적용"""
        adjusted = predictions.copy()
        
        # 전체 트렌드 방향 계산 (신뢰도 가중)
        total_weighted_return = 0.0
        total_weight = 0.0
        
        for horizon, pred in predictions.items():
            return_rate = (pred - current_price) / current_price
            weight = weights.get(horizon, 1.0)
            
            total_weighted_return += return_rate * weight
            total_weight += weight
        
        if total_weight == 0:
            return adjusted
        
        overall_trend = total_weighted_return / total_weight
        
        # 각 예측을 전체 트렌드와 조화시킴
        for short_h, long_h, constraint_weight in self.hierarchy_constraints['trend_coherence']:
            if short_h in adjusted and long_h in adjusted:
                short_pred = adjusted[short_h]
                long_pred = adjusted[long_h]
                
                short_return = (short_pred - current_price) / current_price
                long_return = (long_pred - current_price) / current_price
                
                # 트렌드와의 편차 계산
                short_deviation = short_return - overall_trend * (short_h / 24)  # 일 단위 정규화
                long_deviation = long_return - overall_trend * (long_h / 24)
                
                # 편차가 큰 경우 조정
                if abs(short_deviation) > 0.1 or abs(long_deviation) > 0.1:
                    adjustment_strength = constraint_weight * 0.05
                    
                    # 트렌드로 조정
                    target_short_return = overall_trend * (short_h / 24)
                    target_long_return = overall_trend * (long_h / 24)
                    
                    adjusted[short_h] = short_pred * (1 - adjustment_strength) + \
                                      (current_price * (1 + target_short_return)) * adjustment_strength
                    adjusted[long_h] = long_pred * (1 - adjustment_strength) + \
                                     (current_price * (1 + target_long_return)) * adjustment_strength
        
        return adjusted
    
    def _apply_volatility_scaling(self, predictions: Dict[int, float], 
                                weights: Dict[int, float], 
                                current_price: float) -> Dict[int, float]:
        """변동성 기반 스케일링"""
        adjusted = predictions.copy()
        
        # 시간대별 예상 변동성 (일반적인 BTC 변동성 패턴)
        expected_volatility = {
            1: 0.02,    # 1시간: 2%
            4: 0.04,    # 4시간: 4%
            24: 0.08,   # 1일: 8%
            72: 0.12,   # 3일: 12%
            168: 0.15   # 1주: 15%
        }
        
        for horizon, pred in adjusted.items():
            return_rate = (pred - current_price) / current_price
            expected_vol = expected_volatility.get(horizon, 0.1)
            
            # 예측 수익률이 예상 변동성을 초과하는 경우 조정
            if abs(return_rate) > expected_vol * 2:  # 2배 초과시
                # 신뢰도가 낮으면 더 강하게 조정
                confidence = weights.get(horizon, 0.5)
                adjustment_strength = 0.1 * (1 - confidence)
                
                # 변동성 한계로 조정
                max_return = expected_vol * 1.5 * np.sign(return_rate)
                target_price = current_price * (1 + max_return)
                
                adjusted[horizon] = pred * (1 - adjustment_strength) + target_price * adjustment_strength
        
        return adjusted

class MultiObjectiveOptimizer:
    """다목적 최적화기"""
    
    def __init__(self):
        self.objectives = {
            'accuracy': self._accuracy_objective,
            'consistency': self._consistency_objective,
            'risk': self._risk_objective,
            'stability': self._stability_objective
        }
        self.objective_weights = {
            'accuracy': 0.4,
            'consistency': 0.2,
            'risk': 0.2,
            'stability': 0.2
        }
        
    def optimize_integration(self, prediction_records: List[PredictionRecord],
                           performance_history: Dict[int, List[float]],
                           constraints: Dict = None) -> Dict[int, float]:
        """통합 최적화"""
        
        if not prediction_records:
            return {}
        
        # 시간대별 예측값 그룹화
        horizon_predictions = defaultdict(list)
        for record in prediction_records:
            horizon_predictions[record.horizon].append(record)
        
        # 최적화 변수: 각 시간대별 가중치
        horizons = list(horizon_predictions.keys())
        
        def objective_function(weights):
            """다목적 목적 함수"""
            weights_dict = dict(zip(horizons, weights))
            
            total_score = 0.0
            for obj_name, obj_weight in self.objective_weights.items():
                obj_score = self.objectives[obj_name](
                    weights_dict, horizon_predictions, performance_history
                )
                total_score += obj_weight * obj_score
            
            return -total_score  # 최대화를 위해 음수
        
        # 제약 조건
        constraints_list = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
        bounds = [(0, 1) for _ in horizons]
        
        # 초기값
        x0 = np.array([1.0/len(horizons)] * len(horizons))
        
        # 최적화 실행
        result = minimize(
            objective_function,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints_list
        )
        
        if result.success:
            optimal_weights = dict(zip(horizons, result.x))
        else:
            # 실패시 성과 기반 가중치
            optimal_weights = self._performance_based_weights(performance_history)
        
        return optimal_weights
    
    def _accuracy_objective(self, weights: Dict[int, float], 
                          predictions: Dict[int, List[PredictionRecord]],
                          performance_history: Dict[int, List[float]]) -> float:
        """정확도 목적 함수"""
        total_accuracy = 0.0
        total_weight = 0.0
        
        for horizon, weight in weights.items():
            if horizon in performance_history:
                recent_accuracy = np.mean(performance_history[horizon][-10:]) if performance_history[horizon] else 0.5
                total_accuracy += weight * recent_accuracy
                total_weight += weight
        
        return total_accuracy / total_weight if total_weight > 0 else 0.0
    
    def _consistency_objective(self, weights: Dict[int, float], 
                             predictions: Dict[int, List[PredictionRecord]],
                             performance_history: Dict[int, List[float]]) -> float:
        """일관성 목적 함수"""
        if len(predictions) < 2:
            return 1.0
        
        # 예측값들 간의 상관관계 계산
        horizon_means = {}
        for horizon, records in predictions.items():
            if records:
                horizon_means[horizon] = np.mean([r.prediction for r in records])
        
        if len(horizon_means) < 2:
            return 1.0
        
        correlations = []
        horizon_list = list(horizon_means.keys())
        
        for i in range(len(horizon_list)):
            for j in range(i+1, len(horizon_list)):
                h1, h2 = horizon_list[i], horizon_list[j]
                
                # 시간대별 예측 트렌드 상관관계
                if len(predictions[h1]) > 1 and len(predictions[h2]) > 1:
                    trend1 = [r.prediction for r in predictions[h1][-10:]]
                    trend2 = [r.prediction for r in predictions[h2][-10:]]
                    
                    min_len = min(len(trend1), len(trend2))
                    if min_len > 1:
                        corr, _ = pearsonr(trend1[:min_len], trend2[:min_len])
                        if not np.isnan(corr):
                            correlations.append(abs(corr))
        
        return np.mean(correlations) if correlations else 0.5
    
    def _risk_objective(self, weights: Dict[int, float], 
                       predictions: Dict[int, List[PredictionRecord]],
                       performance_history: Dict[int, List[float]]) -> float:
        """위험 목적 함수 (위험 최소화)"""
        total_risk = 0.0
        
        for horizon, weight in weights.items():
            if horizon in predictions:
                # 예측 불확실성
                uncertainties = [r.uncertainty for r in predictions[horizon] if r.uncertainty > 0]
                avg_uncertainty = np.mean(uncertainties) if uncertainties else 0.1
                
                # 성과 변동성
                if horizon in performance_history and len(performance_history[horizon]) > 5:
                    performance_volatility = np.std(performance_history[horizon][-20:])
                else:
                    performance_volatility = 0.1
                
                horizon_risk = avg_uncertainty + performance_volatility
                total_risk += weight * horizon_risk
        
        return max(0, 1 - total_risk)  # 위험이 낮을수록 높은 점수
    
    def _stability_objective(self, weights: Dict[int, float], 
                           predictions: Dict[int, List[PredictionRecord]],
                           performance_history: Dict[int, List[float]]) -> float:
        """안정성 목적 함수"""
        # 가중치 분산 최소화 (다양성 선호)
        weight_variance = np.var(list(weights.values()))
        
        # 성과 안정성
        stability_scores = []
        for horizon in weights.keys():
            if horizon in performance_history and len(performance_history[horizon]) > 10:
                recent_std = np.std(performance_history[horizon][-20:])
                stability_score = max(0, 1 - recent_std)
                stability_scores.append(stability_score)
        
        performance_stability = np.mean(stability_scores) if stability_scores else 0.5
        
        # 가중치 균형성 (0.5 = 완전 균등)
        weight_balance = 1 - abs(weight_variance - 0.25)  # 적당한 분산 선호
        
        return (performance_stability + weight_balance) / 2
    
    def _performance_based_weights(self, performance_history: Dict[int, List[float]]) -> Dict[int, float]:
        """성과 기반 대체 가중치"""
        if not performance_history:
            return {}
        
        horizon_scores = {}
        for horizon, history in performance_history.items():
            if history:
                recent_performance = np.mean(history[-10:])
                horizon_scores[horizon] = max(0.01, recent_performance)  # 최소 1%
            else:
                horizon_scores[horizon] = 0.5
        
        # 정규화
        total_score = sum(horizon_scores.values())
        weights = {h: score/total_score for h, score in horizon_scores.items()}
        
        return weights

class PerformanceAttributionAnalyzer:
    """성과 기여도 분석기"""
    
    def __init__(self):
        self.attribution_history = []
        
    def analyze_attribution(self, predictions: Dict[int, float],
                          weights: Dict[int, float],
                          actual_outcomes: Dict[int, float],
                          timestamp: datetime) -> Dict[str, float]:
        """성과 기여도 분석"""
        
        attribution = {
            'total_performance': 0.0,
            'horizon_contributions': {},
            'weight_effectiveness': 0.0,
            'integration_benefit': 0.0
        }
        
        if not predictions or not actual_outcomes:
            return attribution
        
        # 개별 시간대 성과
        individual_performances = {}
        for horizon in predictions.keys():
            if horizon in actual_outcomes:
                pred = predictions[horizon]
                actual = actual_outcomes[horizon]
                
                # 정확도 (MAPE 기반)
                accuracy = max(0, 1 - abs(actual - pred) / abs(actual)) if actual != 0 else 0
                individual_performances[horizon] = accuracy
        
        # 가중 평균 성과
        total_weighted_performance = 0.0
        total_weight = 0.0
        
        for horizon, performance in individual_performances.items():
            weight = weights.get(horizon, 0)
            contribution = weight * performance
            
            attribution['horizon_contributions'][horizon] = {
                'individual_performance': performance,
                'weight': weight,
                'contribution': contribution
            }
            
            total_weighted_performance += contribution
            total_weight += weight
        
        attribution['total_performance'] = total_weighted_performance
        
        # 가중치 효과성 (가중 성과 vs 균등 가중 성과)
        if individual_performances:
            equal_weight_performance = np.mean(list(individual_performances.values()))
            attribution['weight_effectiveness'] = total_weighted_performance - equal_weight_performance
        
        # 통합 이익 (최고 개별 성과 vs 통합 성과)
        if individual_performances:
            best_individual = max(individual_performances.values())
            attribution['integration_benefit'] = total_weighted_performance - best_individual
        
        # 히스토리 저장
        self.attribution_history.append({
            'timestamp': timestamp,
            'attribution': attribution.copy()
        })
        
        # 히스토리 제한
        if len(self.attribution_history) > 1000:
            self.attribution_history = self.attribution_history[-800:]
        
        return attribution
    
    def get_attribution_summary(self, lookback_days: int = 7) -> Dict:
        """기여도 요약 분석"""
        cutoff_time = datetime.now() - timedelta(days=lookback_days)
        recent_attributions = [
            a for a in self.attribution_history 
            if a['timestamp'] >= cutoff_time
        ]
        
        if not recent_attributions:
            return {'error': 'insufficient_data'}
        
        # 평균 성과
        avg_performance = np.mean([
            a['attribution']['total_performance'] 
            for a in recent_attributions
        ])
        
        # 시간대별 기여도 평균
        horizon_contributions = defaultdict(list)
        for a in recent_attributions:
            for horizon, contrib in a['attribution']['horizon_contributions'].items():
                horizon_contributions[horizon].append(contrib['contribution'])
        
        avg_contributions = {
            h: np.mean(contribs) 
            for h, contribs in horizon_contributions.items()
        }
        
        # 가중치 효과성 평균
        avg_weight_effectiveness = np.mean([
            a['attribution']['weight_effectiveness'] 
            for a in recent_attributions
        ])
        
        # 통합 이익 평균
        avg_integration_benefit = np.mean([
            a['attribution']['integration_benefit'] 
            for a in recent_attributions
        ])
        
        return {
            'period': f'{lookback_days} days',
            'sample_count': len(recent_attributions),
            'average_performance': avg_performance,
            'horizon_contributions': avg_contributions,
            'weight_effectiveness': avg_weight_effectiveness,
            'integration_benefit': avg_integration_benefit,
            'best_contributing_horizon': max(avg_contributions.items(), key=lambda x: x[1])[0] if avg_contributions else None
        }

class AdaptiveEnsemble:
    """적응형 앙상블"""
    
    def __init__(self, methods: List[str]):
        self.methods = methods
        self.method_weights = {method: 1.0/len(methods) for method in methods}
        self.performance_tracker = defaultdict(lambda: deque(maxlen=100))
        self.meta_model = None
        
    def update_method_performance(self, method: str, accuracy: float):
        """방법별 성과 업데이트"""
        self.performance_tracker[method].append(accuracy)
        
        # 가중치 재계산 (최근 성과 기반)
        self._recompute_weights()
    
    def _recompute_weights(self):
        """가중치 재계산"""
        method_scores = {}
        
        for method in self.methods:
            if method in self.performance_tracker:
                recent_performance = list(self.performance_tracker[method])[-20:]  # 최근 20개
                if recent_performance:
                    # 성과 + 안정성 고려
                    avg_performance = np.mean(recent_performance)
                    stability = 1 - np.std(recent_performance) if len(recent_performance) > 1 else 1
                    method_scores[method] = avg_performance * 0.7 + stability * 0.3
                else:
                    method_scores[method] = 0.5
            else:
                method_scores[method] = 0.5
        
        # 정규화
        total_score = sum(method_scores.values())
        if total_score > 0:
            self.method_weights = {method: score/total_score for method, score in method_scores.items()}
        
    def combine_predictions(self, method_predictions: Dict[str, Dict[int, float]]) -> Dict[int, float]:
        """예측 결합"""
        combined = {}
        
        # 모든 시간대 수집
        all_horizons = set()
        for predictions in method_predictions.values():
            all_horizons.update(predictions.keys())
        
        # 시간대별 결합
        for horizon in all_horizons:
            weighted_pred = 0.0
            total_weight = 0.0
            
            for method, predictions in method_predictions.items():
                if horizon in predictions:
                    weight = self.method_weights.get(method, 0)
                    weighted_pred += predictions[horizon] * weight
                    total_weight += weight
            
            if total_weight > 0:
                combined[horizon] = weighted_pred / total_weight
            else:
                # 대안: 단순 평균
                horizon_preds = [pred[horizon] for pred in method_predictions.values() if horizon in pred]
                combined[horizon] = np.mean(horizon_preds) if horizon_preds else 0.0
        
        return combined
    
    def get_ensemble_confidence(self, method_predictions: Dict[str, Dict[int, float]]) -> Dict[int, float]:
        """앙상블 신뢰도 계산"""
        confidence_scores = {}
        
        for horizon in set().union(*[p.keys() for p in method_predictions.values()]):
            horizon_preds = []
            horizon_weights = []
            
            for method, predictions in method_predictions.items():
                if horizon in predictions:
                    horizon_preds.append(predictions[horizon])
                    horizon_weights.append(self.method_weights.get(method, 0))
            
            if len(horizon_preds) > 1:
                # 가중 분산 계산
                weighted_mean = np.average(horizon_preds, weights=horizon_weights)
                weighted_var = np.average([(p - weighted_mean)**2 for p in horizon_preds], weights=horizon_weights)
                
                # 신뢰도: 분산의 역수
                confidence = 1.0 / (1.0 + weighted_var)
                confidence_scores[horizon] = confidence
            else:
                confidence_scores[horizon] = 0.5  # 기본값
        
        return confidence_scores

class IntegrationStrategiesSystem:
    """통합 전략 시스템"""
    
    def __init__(self, horizons: List[int]):
        self.horizons = horizons
        
        # 구성 요소
        self.reconciliation = HierarchicalReconciliation(horizons)
        self.optimizer = MultiObjectiveOptimizer()
        self.attribution_analyzer = PerformanceAttributionAnalyzer()
        self.adaptive_ensemble = AdaptiveEnsemble(['neural_network', 'random_forest', 'gradient_boosting'])
        
        # 상태 추적
        self.prediction_records = deque(maxlen=1000)
        self.performance_history = defaultdict(list)
        self.integration_history = []
        
        self.setup_logging()
    
    def setup_logging(self):
        """로깅 설정"""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def integrate_predictions(self, 
                            method_predictions: Dict[str, Dict[int, float]],
                            confidence_scores: Dict[str, Dict[int, float]],
                            uncertainty_bounds: Dict[str, Dict[int, Tuple[float, float]]],
                            current_price: float,
                            market_context: Dict = None) -> IntegrationResult:
        """예측 통합"""
        
        self.logger.info(f"🔄 다중 예측 통합 시작 - 방법 수: {len(method_predictions)}")
        
        if not method_predictions:
            return self._create_empty_result()
        
        # 1. 적응형 앙상블로 예측 결합
        ensemble_predictions = self.adaptive_ensemble.combine_predictions(method_predictions)
        ensemble_confidence = self.adaptive_ensemble.get_ensemble_confidence(method_predictions)
        
        # 2. 계층적 조정
        reconciled_predictions = self.reconciliation.reconcile_predictions(
            ensemble_predictions, ensemble_confidence, current_price
        )
        
        # 3. 다목적 최적화
        # 예측 기록 생성
        current_time = datetime.now()
        prediction_records = []
        
        for method, predictions in method_predictions.items():
            for horizon, pred in predictions.items():
                confidence = confidence_scores.get(method, {}).get(horizon, 0.5)
                uncertainty = 0.0
                if method in uncertainty_bounds and horizon in uncertainty_bounds[method]:
                    lower, upper = uncertainty_bounds[method][horizon]
                    uncertainty = (upper - lower) / 2
                
                record = PredictionRecord(
                    timestamp=current_time,
                    horizon=horizon,
                    prediction=pred,
                    confidence=confidence,
                    uncertainty=uncertainty,
                    method=method,
                    market_regime=market_context.get('regime', '') if market_context else '',
                    volatility=market_context.get('volatility', 0.0) if market_context else 0.0
                )
                prediction_records.append(record)
        
        self.prediction_records.extend(prediction_records)
        
        # 최적 가중치 계산
        optimal_weights = self.optimizer.optimize_integration(
            prediction_records, self.performance_history
        )
        
        # 4. 최종 통합 예측
        final_predictions = {}
        final_confidence = {}
        final_uncertainty_bounds = {}
        
        for horizon in self.horizons:
            if horizon in reconciled_predictions:
                # 기본 예측값
                final_predictions[horizon] = reconciled_predictions[horizon]
                
                # 신뢰도 집계
                horizon_confidences = []
                for method in method_predictions.keys():
                    if method in confidence_scores and horizon in confidence_scores[method]:
                        horizon_confidences.append(confidence_scores[method][horizon])
                
                final_confidence[horizon] = np.mean(horizon_confidences) if horizon_confidences else 0.5
                
                # 불확실성 구간 집계
                all_bounds = []
                for method in method_predictions.keys():
                    if method in uncertainty_bounds and horizon in uncertainty_bounds[method]:
                        all_bounds.append(uncertainty_bounds[method][horizon])
                
                if all_bounds:
                    lower_bounds = [b[0] for b in all_bounds]
                    upper_bounds = [b[1] for b in all_bounds]
                    final_uncertainty_bounds[horizon] = (
                        np.mean(lower_bounds),
                        np.mean(upper_bounds)
                    )
                else:
                    pred = final_predictions[horizon]
                    std_error = abs(pred - current_price) * 0.1  # 10% 오차 가정
                    final_uncertainty_bounds[horizon] = (pred - std_error, pred + std_error)
        
        # 5. 성과 기여도 분석 (이전 결과가 있는 경우)
        attribution_scores = {}
        if len(self.integration_history) > 0:
            # 단순화된 기여도 점수
            for method in method_predictions.keys():
                # 최근 성과 기반 기여도
                method_performance = self.performance_history.get(method, [0.5])
                recent_performance = np.mean(method_performance[-10:]) if method_performance else 0.5
                attribution_scores[method] = recent_performance
        
        # 6. 성능 메트릭 계산
        performance_metrics = {
            'integration_confidence': np.mean(list(final_confidence.values())),
            'prediction_diversity': self._calculate_prediction_diversity(method_predictions),
            'ensemble_coherence': self._calculate_ensemble_coherence(reconciled_predictions, current_price),
            'optimization_quality': self._evaluate_optimization_quality(optimal_weights)
        }
        
        # 결과 생성
        result = IntegrationResult(
            integrated_predictions=final_predictions,
            confidence_scores=final_confidence,
            uncertainty_bounds=final_uncertainty_bounds,
            attribution_scores=attribution_scores,
            performance_metrics=performance_metrics,
            optimization_details={
                'optimal_weights': optimal_weights,
                'ensemble_weights': self.adaptive_ensemble.method_weights.copy(),
                'reconciliation_applied': True
            }
        )
        
        # 히스토리 저장
        self.integration_history.append({
            'timestamp': current_time,
            'result': result,
            'market_context': market_context
        })
        
        self.logger.info(f"✅ 예측 통합 완료 - 통합 신뢰도: {performance_metrics['integration_confidence']:.3f}")
        
        return result
    
    def _calculate_prediction_diversity(self, method_predictions: Dict[str, Dict[int, float]]) -> float:
        """예측 다양성 계산"""
        if len(method_predictions) < 2:
            return 0.0
        
        diversity_scores = []
        
        # 모든 시간대에 대해 다양성 계산
        all_horizons = set().union(*[p.keys() for p in method_predictions.values()])
        
        for horizon in all_horizons:
            horizon_preds = []
            for predictions in method_predictions.values():
                if horizon in predictions:
                    horizon_preds.append(predictions[horizon])
            
            if len(horizon_preds) > 1:
                # 변이계수 (CV) 계산
                mean_pred = np.mean(horizon_preds)
                std_pred = np.std(horizon_preds)
                cv = std_pred / abs(mean_pred) if abs(mean_pred) > 0 else 0
                diversity_scores.append(cv)
        
        return np.mean(diversity_scores) if diversity_scores else 0.0
    
    def _calculate_ensemble_coherence(self, predictions: Dict[int, float], current_price: float) -> float:
        """앙상블 일관성 계산"""
        if len(predictions) < 2:
            return 1.0
        
        # 수익률 기반 일관성
        returns = {}
        for horizon, pred in predictions.items():
            returns[horizon] = (pred - current_price) / current_price
        
        return_values = list(returns.values())
        
        # 수익률 간 상관관계
        if len(return_values) > 1:
            correlations = []
            for i in range(len(return_values)):
                for j in range(i+1, len(return_values)):
                    # 시간 가중 상관관계 (시간차가 적을수록 높은 상관관계 기대)
                    horizons_list = list(returns.keys())
                    time_factor = 1.0 / (1.0 + abs(horizons_list[i] - horizons_list[j]) / 24)  # 일 단위
                    correlation = abs(return_values[i] - return_values[j])  # 단순 차이
                    correlations.append((1.0 - correlation) * time_factor)
            
            return np.mean(correlations)
        
        return 1.0
    
    def _evaluate_optimization_quality(self, weights: Dict[int, float]) -> float:
        """최적화 품질 평가"""
        if not weights:
            return 0.0
        
        # 가중치 분포 품질 (너무 집중되지 않고 적당히 분산)
        weight_values = list(weights.values())
        
        # 엔트로피 계산
        entropy = -sum(w * np.log(w + 1e-10) for w in weight_values)
        max_entropy = np.log(len(weight_values))
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        
        # 최적 엔트로피는 완전 균등(1.0)과 완전 집중(0.0)의 중간
        optimal_entropy = 0.7  # 약간의 집중을 선호
        quality = 1.0 - abs(normalized_entropy - optimal_entropy)
        
        return max(0.0, quality)
    
    def update_performance(self, method: str, horizon: int, actual: float, predicted: float):
        """성능 업데이트"""
        # 정확도 계산
        accuracy = max(0, 1 - abs(actual - predicted) / abs(actual)) if actual != 0 else 0
        
        # 방법별 성능 추적
        method_key = f"{method}_{horizon}"
        self.performance_history[method_key].append(accuracy)
        
        # 적응형 앙상블 업데이트
        self.adaptive_ensemble.update_method_performance(method, accuracy)
        
        # 히스토리 제한
        if len(self.performance_history[method_key]) > 200:
            self.performance_history[method_key] = self.performance_history[method_key][-150:]
    
    def _create_empty_result(self) -> IntegrationResult:
        """빈 결과 생성"""
        return IntegrationResult(
            integrated_predictions={},
            confidence_scores={},
            uncertainty_bounds={},
            attribution_scores={},
            performance_metrics={
                'integration_confidence': 0.0,
                'prediction_diversity': 0.0,
                'ensemble_coherence': 0.0,
                'optimization_quality': 0.0
            }
        )
    
    def get_system_summary(self) -> Dict:
        """시스템 요약"""
        recent_performance = []
        for method_key, history in self.performance_history.items():
            if history:
                recent_performance.append(np.mean(history[-20:]))
        
        return {
            'total_integrations': len(self.integration_history),
            'average_recent_performance': np.mean(recent_performance) if recent_performance else 0.0,
            'active_methods': len(self.adaptive_ensemble.methods),
            'method_weights': self.adaptive_ensemble.method_weights.copy(),
            'prediction_record_count': len(self.prediction_records),
            'performance_tracking': {
                method: len(history) for method, history in self.performance_history.items()
            }
        }
    
    def save_system_state(self, filepath: str):
        """시스템 상태 저장"""
        # JSON 직렬화 가능한 데이터로 변환
        serializable_data = {
            'horizons': self.horizons,
            'method_weights': self.adaptive_ensemble.method_weights,
            'performance_history': {
                method: list(history)[-100:]  # 최근 100개만
                for method, history in self.performance_history.items()
            },
            'integration_history': [
                {
                    'timestamp': entry['timestamp'].isoformat(),
                    'performance_metrics': entry['result'].performance_metrics,
                    'market_context': entry.get('market_context', {})
                }
                for entry in self.integration_history[-100:]  # 최근 100개만
            ],
            'system_summary': self.get_system_summary()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(serializable_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"통합 전략 시스템 상태 저장: {filepath}")

def main():
    """메인 테스트 함수"""
    # 테스트 시스템 초기화
    horizons = [1, 4, 24, 72, 168]
    integration_system = IntegrationStrategiesSystem(horizons)
    
    print("🎯 Integration Strategies System Test")
    print("="*60)
    
    # 테스트 시나리오
    current_price = 55000.0
    
    # 다양한 방법의 예측값 시뮬레이션
    np.random.seed(42)
    
    # 방법별 예측 생성
    methods = ['neural_network', 'random_forest', 'gradient_boosting']
    method_predictions = {}
    confidence_scores = {}
    uncertainty_bounds = {}
    
    for method in methods:
        method_predictions[method] = {}
        confidence_scores[method] = {}
        uncertainty_bounds[method] = {}
        
        for horizon in horizons:
            # 방법별 특성을 반영한 예측
            base_return = np.random.normal(0.02, 0.1)  # 기본 2% 상승 경향
            
            # 방법별 편향
            if method == 'neural_network':
                method_bias = np.random.normal(0.001, 0.02)  # 약간 낙관적
            elif method == 'random_forest':
                method_bias = np.random.normal(-0.001, 0.015)  # 약간 보수적
            else:  # gradient_boosting
                method_bias = np.random.normal(0, 0.01)  # 중립적
            
            # 시간대별 특성
            time_factor = 1.0 + (horizon / 168) * 0.1  # 장기일수록 불확실성 증가
            
            predicted_return = (base_return + method_bias) * time_factor
            predicted_price = current_price * (1 + predicted_return)
            
            method_predictions[method][horizon] = predicted_price
            
            # 신뢰도 (시간대가 짧을수록 높음)
            base_confidence = 0.9 - (horizon / 168) * 0.3
            method_confidence = base_confidence + np.random.normal(0, 0.05)
            confidence_scores[method][horizon] = max(0.3, min(0.95, method_confidence))
            
            # 불확실성 구간
            uncertainty = abs(predicted_price - current_price) * (0.05 + horizon / 168 * 0.1)
            lower_bound = predicted_price - uncertainty
            upper_bound = predicted_price + uncertainty
            uncertainty_bounds[method][horizon] = (lower_bound, upper_bound)
    
    # 시장 컨텍스트
    market_context = {
        'regime': 'low_volatility_bull',
        'volatility': 0.04,
        'trend_strength': 0.6
    }
    
    print(f"📊 입력 예측값:")
    for method in methods:
        print(f"  {method}:")
        for horizon in horizons:
            pred = method_predictions[method][horizon]
            conf = confidence_scores[method][horizon]
            print(f"    {horizon}h: ${pred:,.0f} (신뢰도: {conf:.3f})")
        print()
    
    # 통합 실행
    integration_result = integration_system.integrate_predictions(
        method_predictions=method_predictions,
        confidence_scores=confidence_scores,
        uncertainty_bounds=uncertainty_bounds,
        current_price=current_price,
        market_context=market_context
    )
    
    # 결과 출력
    print(f"🔄 통합 결과:")
    print(f"  현재가: ${current_price:,.0f}")
    print()
    
    print(f"📈 통합 예측값:")
    for horizon in horizons:
        if horizon in integration_result.integrated_predictions:
            pred = integration_result.integrated_predictions[horizon]
            conf = integration_result.confidence_scores[horizon]
            lower, upper = integration_result.uncertainty_bounds[horizon]
            
            return_pct = (pred - current_price) / current_price * 100
            print(f"  {horizon}h: ${pred:,.0f} ({return_pct:+.2f}%)")
            print(f"       신뢰도: {conf:.3f}")
            print(f"       구간: [${lower:,.0f}, ${upper:,.0f}]")
    
    print(f"\n⚡ 성능 메트릭:")
    metrics = integration_result.performance_metrics
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.3f}")
    
    print(f"\n🔧 최적화 세부사항:")
    details = integration_result.optimization_details
    print(f"  최적 가중치:")
    for horizon, weight in details.get('optimal_weights', {}).items():
        print(f"    {horizon}h: {weight:.3f}")
    
    print(f"  앙상블 가중치:")
    for method, weight in details.get('ensemble_weights', {}).items():
        print(f"    {method}: {weight:.3f}")
    
    # 성능 업데이트 시뮬레이션
    print(f"\n🎯 성능 업데이트 시뮬레이션:")
    for method in methods:
        for horizon in horizons:
            # 가상의 실제값 (예측값 근처)
            predicted = method_predictions[method][horizon]
            actual = predicted + np.random.normal(0, abs(predicted - current_price) * 0.1)
            
            integration_system.update_performance(method, horizon, actual, predicted)
            
            accuracy = max(0, 1 - abs(actual - predicted) / abs(actual)) if actual != 0 else 0
            print(f"  {method} {horizon}h: 정확도 {accuracy:.3f}")
    
    # 시스템 요약
    summary = integration_system.get_system_summary()
    print(f"\n📊 시스템 요약:")
    print(f"  총 통합 수: {summary['total_integrations']}")
    print(f"  평균 성능: {summary['average_recent_performance']:.3f}")
    print(f"  활성 방법: {summary['active_methods']}")
    print(f"  예측 기록: {summary['prediction_record_count']}")
    
    # 결과 저장
    integration_system.save_system_state('integration_strategies_results.json')
    
    print(f"\n💾 결과 저장 완료: integration_strategies_results.json")
    
    return integration_system

if __name__ == "__main__":
    main()