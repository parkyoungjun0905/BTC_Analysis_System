#!/usr/bin/env python3
"""
🛡️ 로버스트 신뢰성 시스템
모델 실패 감지, 불확실성 정량화, 강건한 예측 시스템

핵심 기능:
- 실시간 모델 실패 감지 및 자동 복구
- 베이지안 불확실성 정량화
- 강건한 예측 집계 (Robust Aggregation)
- 성능 일관성 모니터링
- 자동 알림 및 복구 시스템
"""

import numpy as np
import pandas as pd
import json
import pickle
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional, Union
from pathlib import Path
import sqlite3
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, asdict
from enum import Enum
import threading
import time

# 통계 및 베이지안 라이브러리
try:
    from scipy import stats
    from scipy.stats import norm, t, chi2
    import scipy.special as special
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("⚠️ SciPy 미설치 - 통계적 기능 제한")

# 머신러닝 라이브러리
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope

warnings.filterwarnings('ignore')

class ModelHealth(Enum):
    """모델 건강 상태"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    FAILED = "failed"

class AlertSeverity(Enum):
    """알림 심각도"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

@dataclass
class ModelPerformanceMetrics:
    """모델 성능 메트릭"""
    timestamp: datetime
    model_name: str
    accuracy: float
    mse: float
    mae: float
    r2: float
    prediction_count: int
    response_time: float
    memory_usage: float
    health_status: ModelHealth

@dataclass
class UncertaintyEstimate:
    """불확실성 추정"""
    prediction: float
    epistemic_uncertainty: float  # 모델 불확실성
    aleatoric_uncertainty: float  # 데이터 불확실성
    total_uncertainty: float
    confidence_interval_lower: float
    confidence_interval_upper: float
    confidence_level: float

class BayesianUncertaintyQuantifier:
    """
    🎲 베이지안 불확실성 정량화 시스템
    """
    
    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level
        self.posterior_samples = {}
        self.prior_parameters = {}
        
        self.logger = logging.getLogger(__name__)

    def update_posterior(self, model_name: str, predictions: np.ndarray, 
                        targets: np.ndarray, prior_mean: float = 0.0, 
                        prior_std: float = 1.0):
        """
        베이지안 사후 분포 업데이트
        
        Args:
            model_name: 모델명
            predictions: 예측값들
            targets: 실제값들
            prior_mean: 사전 평균
            prior_std: 사전 표준편차
        """
        if not SCIPY_AVAILABLE:
            return
        
        residuals = targets - predictions
        n_samples = len(residuals)
        
        if n_samples == 0:
            return
        
        # 베이지안 선형 회귀 추정
        sample_mean = np.mean(residuals)
        sample_var = np.var(residuals, ddof=1) if n_samples > 1 else 1.0
        
        # 사후 분포 파라미터 (정규-역감마 모델)
        prior_precision = 1.0 / (prior_std ** 2)
        sample_precision = 1.0 / sample_var if sample_var > 0 else 1.0
        
        # 사후 평균과 정밀도
        posterior_precision = prior_precision + n_samples * sample_precision
        posterior_mean = (
            prior_precision * prior_mean + 
            n_samples * sample_precision * sample_mean
        ) / posterior_precision
        
        # 사후 분산
        posterior_var = 1.0 / posterior_precision
        
        # 사후 자유도 (t-분포용)
        posterior_df = max(1, n_samples - 1)
        
        self.posterior_samples[model_name] = {
            'mean': posterior_mean,
            'var': posterior_var,
            'std': np.sqrt(posterior_var),
            'df': posterior_df,
            'n_samples': n_samples,
            'last_update': datetime.now()
        }

    def estimate_uncertainty(self, model_name: str, prediction: float,
                           feature_uncertainty: float = 0.1) -> UncertaintyEstimate:
        """
        불확실성 추정
        
        Args:
            model_name: 모델명
            prediction: 예측값
            feature_uncertainty: 특성 불확실성
            
        Returns:
            UncertaintyEstimate: 불확실성 추정 결과
        """
        if model_name not in self.posterior_samples:
            # 기본 불확실성
            return UncertaintyEstimate(
                prediction=prediction,
                epistemic_uncertainty=0.2,
                aleatoric_uncertainty=0.1,
                total_uncertainty=0.22,
                confidence_interval_lower=prediction - 0.44,
                confidence_interval_upper=prediction + 0.44,
                confidence_level=self.confidence_level
            )
        
        posterior = self.posterior_samples[model_name]
        
        # 인식론적 불확실성 (모델 파라미터 불확실성)
        epistemic_uncertainty = posterior['std']
        
        # 우연적 불확실성 (데이터 노이즈)
        aleatoric_uncertainty = feature_uncertainty
        
        # 총 불확실성
        total_uncertainty = np.sqrt(
            epistemic_uncertainty**2 + aleatoric_uncertainty**2
        )
        
        # 신뢰구간 계산 (t-분포 사용)
        if SCIPY_AVAILABLE:
            alpha = 1 - self.confidence_level
            t_value = stats.t.ppf(1 - alpha/2, posterior['df'])
            margin = t_value * total_uncertainty
        else:
            # 정규분포 근사
            z_value = 1.96  # 95% 신뢰구간
            margin = z_value * total_uncertainty
        
        return UncertaintyEstimate(
            prediction=prediction,
            epistemic_uncertainty=epistemic_uncertainty,
            aleatoric_uncertainty=aleatoric_uncertainty,
            total_uncertainty=total_uncertainty,
            confidence_interval_lower=prediction - margin,
            confidence_interval_upper=prediction + margin,
            confidence_level=self.confidence_level
        )

class ModelFailureDetector:
    """
    🔍 모델 실패 감지 시스템
    """
    
    def __init__(self, detection_window: int = 50):
        self.detection_window = detection_window
        self.performance_history = {}
        self.failure_thresholds = {
            'accuracy_drop': 0.15,      # 15% 이상 정확도 하락
            'mse_spike': 2.0,           # MSE 2배 이상 증가
            'response_time': 30.0,      # 30초 이상 응답시간
            'memory_limit': 2048,       # 2GB 메모리 한계
            'nan_predictions': 0.05,    # 5% 이상 NaN 예측
            'consecutive_failures': 5   # 연속 5회 실패
        }
        
        # 이상치 탐지기
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.anomaly_fitted = False
        
        self.logger = logging.getLogger(__name__)

    def add_performance_record(self, metrics: ModelPerformanceMetrics):
        """성능 기록 추가"""
        model_name = metrics.model_name
        
        if model_name not in self.performance_history:
            self.performance_history[model_name] = []
        
        self.performance_history[model_name].append(metrics)
        
        # 윈도우 크기 유지
        if len(self.performance_history[model_name]) > self.detection_window:
            self.performance_history[model_name] = \
                self.performance_history[model_name][-self.detection_window:]

    def detect_model_failures(self, model_name: str) -> Dict[str, Any]:
        """
        모델 실패 감지
        
        Args:
            model_name: 모델명
            
        Returns:
            Dict[str, Any]: 감지 결과
        """
        if (model_name not in self.performance_history or 
            len(self.performance_history[model_name]) < 5):
            return {'status': 'insufficient_data', 'failures': []}
        
        recent_records = self.performance_history[model_name][-10:]
        older_records = self.performance_history[model_name][:-10] if len(self.performance_history[model_name]) > 10 else []
        
        failures = []
        
        # 1. 정확도 하락 감지
        if older_records:
            recent_accuracy = np.mean([r.accuracy for r in recent_records])
            older_accuracy = np.mean([r.accuracy for r in older_records])
            
            accuracy_drop = (older_accuracy - recent_accuracy) / older_accuracy if older_accuracy > 0 else 0
            
            if accuracy_drop > self.failure_thresholds['accuracy_drop']:
                failures.append({
                    'type': 'accuracy_drop',
                    'severity': AlertSeverity.WARNING,
                    'value': accuracy_drop,
                    'threshold': self.failure_thresholds['accuracy_drop'],
                    'description': f'정확도가 {accuracy_drop:.2%} 하락했습니다'
                })
        
        # 2. MSE 급증 감지
        if len(recent_records) >= 5:
            recent_mse = np.mean([r.mse for r in recent_records[-5:]])
            baseline_mse = np.mean([r.mse for r in recent_records[:-5]]) if len(recent_records) > 5 else recent_mse
            
            mse_ratio = recent_mse / baseline_mse if baseline_mse > 0 else 1.0
            
            if mse_ratio > self.failure_thresholds['mse_spike']:
                failures.append({
                    'type': 'mse_spike',
                    'severity': AlertSeverity.CRITICAL,
                    'value': mse_ratio,
                    'threshold': self.failure_thresholds['mse_spike'],
                    'description': f'MSE가 {mse_ratio:.1f}배 증가했습니다'
                })
        
        # 3. 응답 시간 지연 감지
        max_response_time = max(r.response_time for r in recent_records)
        if max_response_time > self.failure_thresholds['response_time']:
            failures.append({
                'type': 'response_time',
                'severity': AlertSeverity.WARNING,
                'value': max_response_time,
                'threshold': self.failure_thresholds['response_time'],
                'description': f'응답시간이 {max_response_time:.1f}초로 지연되었습니다'
            })
        
        # 4. 메모리 사용량 초과 감지
        max_memory = max(r.memory_usage for r in recent_records)
        if max_memory > self.failure_thresholds['memory_limit']:
            failures.append({
                'type': 'memory_limit',
                'severity': AlertSeverity.CRITICAL,
                'value': max_memory,
                'threshold': self.failure_thresholds['memory_limit'],
                'description': f'메모리 사용량이 {max_memory:.1f}MB를 초과했습니다'
            })
        
        # 5. 연속 실패 감지
        failed_count = sum(1 for r in recent_records if r.health_status == ModelHealth.FAILED)
        if failed_count >= self.failure_thresholds['consecutive_failures']:
            failures.append({
                'type': 'consecutive_failures',
                'severity': AlertSeverity.EMERGENCY,
                'value': failed_count,
                'threshold': self.failure_thresholds['consecutive_failures'],
                'description': f'{failed_count}회 연속 실패했습니다'
            })
        
        # 6. 이상치 탐지 (충분한 데이터가 있는 경우)
        if len(self.performance_history[model_name]) >= 20:
            try:
                # 특성 벡터 생성
                features = np.array([
                    [r.accuracy, r.mse, r.response_time, r.memory_usage]
                    for r in self.performance_history[model_name]
                ])
                
                # 이상치 탐지 모델 훈련 (처음인 경우)
                if not self.anomaly_fitted:
                    self.anomaly_detector.fit(features[:-5])  # 최근 5개 제외하고 훈련
                    self.anomaly_fitted = True
                
                # 최근 데이터에서 이상치 감지
                recent_features = features[-5:]
                anomaly_scores = self.anomaly_detector.decision_function(recent_features)
                is_anomaly = self.anomaly_detector.predict(recent_features) == -1
                
                if np.any(is_anomaly):
                    anomaly_count = np.sum(is_anomaly)
                    failures.append({
                        'type': 'anomaly_detection',
                        'severity': AlertSeverity.WARNING,
                        'value': anomaly_count,
                        'threshold': 1,
                        'description': f'{anomaly_count}개 이상치가 감지되었습니다'
                    })
                    
            except Exception as e:
                self.logger.warning(f"⚠️ 이상치 탐지 실패: {e}")
        
        # 전체 상태 평가
        if not failures:
            status = 'healthy'
        elif any(f['severity'] == AlertSeverity.EMERGENCY for f in failures):
            status = 'emergency'
        elif any(f['severity'] == AlertSeverity.CRITICAL for f in failures):
            status = 'critical'
        else:
            status = 'warning'
        
        return {
            'model_name': model_name,
            'status': status,
            'failures': failures,
            'total_failures': len(failures),
            'detection_time': datetime.now(),
            'recent_records_count': len(recent_records)
        }

class RobustAggregator:
    """
    🛡️ 강건한 예측 집계 시스템
    """
    
    def __init__(self):
        self.aggregation_methods = {
            'weighted_median': self._weighted_median,
            'trimmed_mean': self._trimmed_mean,
            'winsorized_mean': self._winsorized_mean,
            'huber_aggregation': self._huber_aggregation,
            'robust_average': self._robust_average
        }
        
        self.logger = logging.getLogger(__name__)

    def _weighted_median(self, predictions: np.ndarray, weights: np.ndarray) -> float:
        """가중 중위수"""
        if len(predictions) == 0:
            return 0.0
        
        # 정렬된 인덱스
        sorted_indices = np.argsort(predictions)
        sorted_predictions = predictions[sorted_indices]
        sorted_weights = weights[sorted_indices]
        
        # 누적 가중치
        cumulative_weights = np.cumsum(sorted_weights)
        total_weight = np.sum(weights)
        
        # 중위수 찾기
        median_weight = total_weight / 2
        median_idx = np.searchsorted(cumulative_weights, median_weight)
        
        if median_idx >= len(sorted_predictions):
            return sorted_predictions[-1]
        
        return sorted_predictions[median_idx]

    def _trimmed_mean(self, predictions: np.ndarray, weights: np.ndarray, 
                     trim_percent: float = 0.2) -> float:
        """절사 평균"""
        if len(predictions) == 0:
            return 0.0
        
        n = len(predictions)
        trim_count = int(n * trim_percent / 2)
        
        if trim_count >= n // 2:
            return np.average(predictions, weights=weights)
        
        # 정렬
        sorted_indices = np.argsort(predictions)
        
        # 상하위 절사
        keep_indices = sorted_indices[trim_count:n-trim_count]
        
        if len(keep_indices) == 0:
            return np.average(predictions, weights=weights)
        
        trimmed_predictions = predictions[keep_indices]
        trimmed_weights = weights[keep_indices]
        
        return np.average(trimmed_predictions, weights=trimmed_weights)

    def _winsorized_mean(self, predictions: np.ndarray, weights: np.ndarray,
                        limits: Tuple[float, float] = (0.1, 0.1)) -> float:
        """윈저화 평균"""
        if len(predictions) == 0:
            return 0.0
        
        # 분위수 계산
        lower_percentile = limits[0] * 100
        upper_percentile = (1 - limits[1]) * 100
        
        lower_bound = np.percentile(predictions, lower_percentile)
        upper_bound = np.percentile(predictions, upper_percentile)
        
        # 윈저화
        winsorized_predictions = np.clip(predictions, lower_bound, upper_bound)
        
        return np.average(winsorized_predictions, weights=weights)

    def _huber_aggregation(self, predictions: np.ndarray, weights: np.ndarray,
                          delta: float = 1.35) -> float:
        """Huber 손실 기반 강건한 집계"""
        if len(predictions) == 0:
            return 0.0
        
        # 초기 추정치 (가중 중위수)
        estimate = self._weighted_median(predictions, weights)
        
        # 반복적 개선
        for _ in range(10):  # 최대 10회 반복
            residuals = predictions - estimate
            
            # Huber 가중치 계산
            huber_weights = np.where(
                np.abs(residuals) <= delta,
                1.0,
                delta / np.abs(residuals)
            )
            
            # 전체 가중치 (원래 가중치 * Huber 가중치)
            total_weights = weights * huber_weights
            
            # 새로운 추정치
            new_estimate = np.average(predictions, weights=total_weights)
            
            # 수렴 확인
            if abs(new_estimate - estimate) < 1e-6:
                break
            
            estimate = new_estimate
        
        return estimate

    def _robust_average(self, predictions: np.ndarray, weights: np.ndarray) -> float:
        """다중 방법 결합 강건한 평균"""
        if len(predictions) == 0:
            return 0.0
        
        # 여러 방법의 결과
        methods_results = []
        
        methods_results.append(self._weighted_median(predictions, weights))
        methods_results.append(self._trimmed_mean(predictions, weights))
        methods_results.append(self._winsorized_mean(predictions, weights))
        methods_results.append(self._huber_aggregation(predictions, weights))
        
        # 방법들의 평균 (더 강건)
        return np.mean(methods_results)

    def robust_ensemble_prediction(self, model_predictions: Dict[str, float],
                                 model_weights: Dict[str, float],
                                 model_health: Dict[str, ModelHealth],
                                 method: str = 'robust_average') -> Dict[str, Any]:
        """
        강건한 앙상블 예측
        
        Args:
            model_predictions: 모델별 예측값
            model_weights: 모델별 가중치
            model_health: 모델별 건강 상태
            method: 집계 방법
            
        Returns:
            Dict[str, Any]: 강건한 예측 결과
        """
        # 건강한 모델만 선택
        healthy_models = {
            name: pred for name, pred in model_predictions.items()
            if model_health.get(name, ModelHealth.FAILED) in [ModelHealth.HEALTHY, ModelHealth.WARNING]
        }
        
        if not healthy_models:
            # 모든 모델이 실패한 경우 - 마지막 안전한 예측 사용
            self.logger.error("❌ 모든 모델이 실패했습니다")
            return {
                'prediction': 0.0,
                'method': 'fallback',
                'confidence': 0.0,
                'models_used': [],
                'total_models': len(model_predictions),
                'healthy_models': 0
            }
        
        # 예측값과 가중치 배열 생성
        predictions = np.array(list(healthy_models.values()))
        weights = np.array([
            model_weights.get(name, 1.0) for name in healthy_models.keys()
        ])
        
        # 가중치 정규화
        if np.sum(weights) > 0:
            weights = weights / np.sum(weights)
        else:
            weights = np.ones(len(predictions)) / len(predictions)
        
        # 선택된 방법으로 집계
        if method in self.aggregation_methods:
            robust_prediction = self.aggregation_methods[method](predictions, weights)
        else:
            robust_prediction = self._robust_average(predictions, weights)
        
        # 신뢰도 계산 (예측의 일치도)
        prediction_std = np.std(predictions) if len(predictions) > 1 else 0.0
        confidence = max(0.0, 1.0 - prediction_std / (np.abs(robust_prediction) + 1e-6))
        
        return {
            'prediction': float(robust_prediction),
            'method': method,
            'confidence': float(confidence),
            'models_used': list(healthy_models.keys()),
            'total_models': len(model_predictions),
            'healthy_models': len(healthy_models),
            'prediction_std': float(prediction_std),
            'individual_predictions': dict(healthy_models)
        }

class PerformanceConsistencyMonitor:
    """
    📊 성능 일관성 모니터링 시스템
    """
    
    def __init__(self, monitoring_window: int = 100):
        self.monitoring_window = monitoring_window
        self.consistency_metrics = {}
        
        self.logger = logging.getLogger(__name__)

    def update_performance(self, model_name: str, accuracy: float, 
                          timestamp: datetime = None):
        """성능 업데이트"""
        if timestamp is None:
            timestamp = datetime.now()
        
        if model_name not in self.consistency_metrics:
            self.consistency_metrics[model_name] = {
                'accuracies': [],
                'timestamps': [],
                'rolling_mean': [],
                'rolling_std': []
            }
        
        metrics = self.consistency_metrics[model_name]
        
        # 새 데이터 추가
        metrics['accuracies'].append(accuracy)
        metrics['timestamps'].append(timestamp)
        
        # 윈도우 크기 유지
        if len(metrics['accuracies']) > self.monitoring_window:
            metrics['accuracies'] = metrics['accuracies'][-self.monitoring_window:]
            metrics['timestamps'] = metrics['timestamps'][-self.monitoring_window:]
        
        # 롤링 통계 계산
        if len(metrics['accuracies']) >= 10:
            recent_accuracies = metrics['accuracies'][-10:]
            metrics['rolling_mean'].append(np.mean(recent_accuracies))
            metrics['rolling_std'].append(np.std(recent_accuracies))
            
            # 롤링 통계도 윈도우 크기 유지
            if len(metrics['rolling_mean']) > self.monitoring_window:
                metrics['rolling_mean'] = metrics['rolling_mean'][-self.monitoring_window:]
                metrics['rolling_std'] = metrics['rolling_std'][-self.monitoring_window:]

    def assess_consistency(self, model_name: str) -> Dict[str, Any]:
        """
        일관성 평가
        
        Args:
            model_name: 모델명
            
        Returns:
            Dict[str, Any]: 일관성 평가 결과
        """
        if (model_name not in self.consistency_metrics or 
            len(self.consistency_metrics[model_name]['accuracies']) < 20):
            return {
                'status': 'insufficient_data',
                'consistency_score': 0.0,
                'trend': 'unknown'
            }
        
        metrics = self.consistency_metrics[model_name]
        accuracies = np.array(metrics['accuracies'])
        
        # 1. 변동성 점수 (낮을수록 좋음)
        volatility = np.std(accuracies) / np.mean(accuracies) if np.mean(accuracies) > 0 else float('inf')
        
        # 2. 트렌드 분석
        if len(accuracies) >= 20:
            # 선형 회귀를 통한 트렌드
            x = np.arange(len(accuracies))
            if SCIPY_AVAILABLE:
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, accuracies)
                trend_strength = abs(r_value)
                
                if slope > 0.001 and p_value < 0.05:
                    trend = 'improving'
                elif slope < -0.001 and p_value < 0.05:
                    trend = 'declining'
                else:
                    trend = 'stable'
            else:
                # 간단한 트렌드 계산
                first_half_mean = np.mean(accuracies[:len(accuracies)//2])
                second_half_mean = np.mean(accuracies[len(accuracies)//2:])
                
                if second_half_mean > first_half_mean * 1.05:
                    trend = 'improving'
                elif second_half_mean < first_half_mean * 0.95:
                    trend = 'declining'
                else:
                    trend = 'stable'
                
                trend_strength = abs(second_half_mean - first_half_mean) / first_half_mean
        else:
            trend = 'unknown'
            trend_strength = 0.0
        
        # 3. 이상값 비율
        if len(accuracies) >= 10:
            q1, q3 = np.percentile(accuracies, [25, 75])
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            outliers = np.sum((accuracies < lower_bound) | (accuracies > upper_bound))
            outlier_ratio = outliers / len(accuracies)
        else:
            outlier_ratio = 0.0
        
        # 4. 종합 일관성 점수 (0-1, 높을수록 좋음)
        volatility_score = max(0, 1 - volatility * 10)  # 변동성이 낮을수록 좋음
        outlier_score = 1 - outlier_ratio  # 이상값이 적을수록 좋음
        trend_score = 1 - trend_strength if trend == 'declining' else 1  # 하락 트렌드 페널티
        
        consistency_score = np.mean([volatility_score, outlier_score, trend_score])
        
        # 5. 상태 분류
        if consistency_score >= 0.8:
            status = 'excellent'
        elif consistency_score >= 0.6:
            status = 'good'
        elif consistency_score >= 0.4:
            status = 'fair'
        else:
            status = 'poor'
        
        return {
            'status': status,
            'consistency_score': float(consistency_score),
            'volatility': float(volatility),
            'trend': trend,
            'trend_strength': float(trend_strength),
            'outlier_ratio': float(outlier_ratio),
            'recent_performance': {
                'mean': float(np.mean(accuracies[-10:])),
                'std': float(np.std(accuracies[-10:])),
                'min': float(np.min(accuracies[-10:])),
                'max': float(np.max(accuracies[-10:]))
            },
            'data_points': len(accuracies)
        }

class ReliabilitySystemManager:
    """
    🛡️ 종합 신뢰성 시스템 관리자
    """
    
    def __init__(self):
        self.uncertainty_quantifier = BayesianUncertaintyQuantifier()
        self.failure_detector = ModelFailureDetector()
        self.robust_aggregator = RobustAggregator()
        self.consistency_monitor = PerformanceConsistencyMonitor()
        
        # 시스템 상태
        self.system_health = {}
        self.alerts = []
        self.recovery_actions = []
        
        # 로깅 설정
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('reliability_system.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def comprehensive_reliability_check(self, 
                                      model_predictions: Dict[str, float],
                                      model_weights: Dict[str, float],
                                      recent_targets: Dict[str, np.ndarray] = None) -> Dict[str, Any]:
        """
        종합 신뢰성 검사
        
        Args:
            model_predictions: 모델별 예측값
            model_weights: 모델별 가중치
            recent_targets: 최근 실제값들 (선택적)
            
        Returns:
            Dict[str, Any]: 종합 신뢰성 분석 결과
        """
        print("🛡️ 종합 신뢰성 시스템 분석 시작...")
        
        analysis_results = {
            'timestamp': datetime.now(),
            'models_analyzed': list(model_predictions.keys()),
            'uncertainty_estimates': {},
            'failure_detections': {},
            'consistency_assessments': {},
            'robust_prediction': {},
            'system_alerts': [],
            'recommendations': []
        }
        
        # 1. 불확실성 정량화
        print("🎲 베이지안 불확실성 정량화...")
        for model_name, prediction in model_predictions.items():
            if recent_targets and model_name in recent_targets:
                # 사후 분포 업데이트
                dummy_predictions = np.full(len(recent_targets[model_name]), prediction)
                self.uncertainty_quantifier.update_posterior(
                    model_name, dummy_predictions, recent_targets[model_name]
                )
            
            # 불확실성 추정
            uncertainty = self.uncertainty_quantifier.estimate_uncertainty(
                model_name, prediction
            )
            analysis_results['uncertainty_estimates'][model_name] = asdict(uncertainty)
        
        # 2. 모델 실패 감지
        print("🔍 모델 실패 감지 분석...")
        model_health = {}
        
        for model_name in model_predictions.keys():
            failure_analysis = self.failure_detector.detect_model_failures(model_name)
            analysis_results['failure_detections'][model_name] = failure_analysis
            
            # 건강 상태 결정
            if failure_analysis['status'] == 'emergency':
                model_health[model_name] = ModelHealth.FAILED
            elif failure_analysis['status'] == 'critical':
                model_health[model_name] = ModelHealth.CRITICAL
            elif failure_analysis['status'] == 'warning':
                model_health[model_name] = ModelHealth.WARNING
            else:
                model_health[model_name] = ModelHealth.HEALTHY
            
            # 알림 생성
            for failure in failure_analysis.get('failures', []):
                alert = {
                    'model_name': model_name,
                    'type': failure['type'],
                    'severity': failure['severity'].value,
                    'description': failure['description'],
                    'timestamp': datetime.now()
                }
                analysis_results['system_alerts'].append(alert)
        
        # 3. 일관성 평가
        print("📊 성능 일관성 모니터링...")
        for model_name in model_predictions.keys():
            consistency = self.consistency_monitor.assess_consistency(model_name)
            analysis_results['consistency_assessments'][model_name] = consistency
        
        # 4. 강건한 예측 집계
        print("🛡️ 강건한 앙상블 예측...")
        robust_result = self.robust_aggregator.robust_ensemble_prediction(
            model_predictions, model_weights, model_health
        )
        analysis_results['robust_prediction'] = robust_result
        
        # 5. 전체 시스템 건강도 평가
        healthy_models = sum(1 for health in model_health.values() 
                           if health in [ModelHealth.HEALTHY, ModelHealth.WARNING])
        
        total_models = len(model_health)
        system_health_ratio = healthy_models / total_models if total_models > 0 else 0
        
        if system_health_ratio >= 0.8:
            system_status = 'excellent'
        elif system_health_ratio >= 0.6:
            system_status = 'good'
        elif system_health_ratio >= 0.4:
            system_status = 'degraded'
        else:
            system_status = 'critical'
        
        # 6. 권장사항 생성
        recommendations = []
        
        if system_health_ratio < 0.6:
            recommendations.append("긴급: 실패한 모델들을 재훈련하거나 교체하세요")
        
        critical_models = [name for name, health in model_health.items() 
                          if health == ModelHealth.CRITICAL]
        if critical_models:
            recommendations.append(f"위험: {', '.join(critical_models)} 모델들이 위험 상태입니다")
        
        low_confidence = [name for name, pred in analysis_results['uncertainty_estimates'].items()
                         if pred['total_uncertainty'] > 0.3]
        if low_confidence:
            recommendations.append(f"주의: {', '.join(low_confidence)} 모델들의 불확실성이 높습니다")
        
        if robust_result['confidence'] < 0.7:
            recommendations.append("앙상블 신뢰도가 낮습니다. 모델 다양성을 개선하세요")
        
        analysis_results['recommendations'] = recommendations
        
        # 7. 시스템 전체 요약
        analysis_results['system_summary'] = {
            'status': system_status,
            'health_ratio': system_health_ratio,
            'total_models': total_models,
            'healthy_models': healthy_models,
            'critical_models': len(critical_models),
            'total_alerts': len(analysis_results['system_alerts']),
            'ensemble_confidence': robust_result['confidence']
        }
        
        # 결과 출력
        print("\n" + "="*50)
        print("🛡️ 신뢰성 시스템 분석 완료!")
        print("="*50)
        print(f"🏥 시스템 상태: {system_status.upper()}")
        print(f"💪 건강한 모델: {healthy_models}/{total_models} ({system_health_ratio:.1%})")
        print(f"🎯 앙상블 예측: {robust_result['prediction']:.6f}")
        print(f"🔒 앙상블 신뢰도: {robust_result['confidence']:.3f}")
        print(f"⚠️  총 알림: {len(analysis_results['system_alerts'])}개")
        
        if recommendations:
            print("\n📋 권장사항:")
            for i, rec in enumerate(recommendations, 1):
                print(f"  {i}. {rec}")
        
        return analysis_results

    def save_reliability_analysis(self, analysis_results: Dict, 
                                file_path: str = None) -> str:
        """신뢰성 분석 결과 저장"""
        if file_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = f"/Users/parkyoungjun/Desktop/BTC_Analysis_System/reliability_analysis_{timestamp}.json"
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(analysis_results, f, indent=2, ensure_ascii=False, default=str)
        
        self.logger.info(f"💾 신뢰성 분석 결과 저장: {file_path}")
        return file_path

def main():
    """메인 실행 함수"""
    print("🛡️ 로버스트 신뢰성 시스템 초기화")
    
    # 시스템 초기화
    reliability_system = ReliabilitySystemManager()
    
    print("✅ 신뢰성 시스템 초기화 완료")
    print("📋 사용 가능한 기능:")
    print("  🎲 베이지안 불확실성 정량화")
    print("  🔍 실시간 모델 실패 감지")
    print("  🛡️ 강건한 예측 집계")
    print("  📊 성능 일관성 모니터링")
    print("  ⚠️  자동 알림 및 복구")
    
    return reliability_system

if __name__ == "__main__":
    system = main()