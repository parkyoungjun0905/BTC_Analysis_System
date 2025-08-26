#!/usr/bin/env python3
"""
🎯 Temporal Hierarchy Modeling Engine
시간 계층 모델링 엔진 - 장/중/단기 트렌드 분석 및 노이즈 필터링

주요 기능:
1. Multi-Scale Temporal Analysis - 다중 스케일 시간 분석
2. Noise Filtering - 고급 노이즈 필터링
3. Cross-Horizon Consistency - 시간대 간 일관성 제약
4. Hierarchical Feature Extraction - 계층적 특성 추출
5. Adaptive Time Weighting - 적응형 시간 가중치
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy import signal
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
import json
import matplotlib.pyplot as plt

@dataclass
class TemporalLevel:
    """시간 계층 레벨 정의"""
    name: str
    window_size: int
    sampling_rate: int
    weight: float
    noise_threshold: float

class NoiseFilter:
    """고급 노이즈 필터"""
    
    def __init__(self):
        self.filter_methods = {
            'savgol': self._savitzky_golay_filter,
            'butterworth': self._butterworth_filter,
            'kalman': self._simple_kalman_filter,
            'wavelet': self._wavelet_denoise,
            'median': self._median_filter
        }
    
    def _savitzky_golay_filter(self, data: np.ndarray, window: int = 11, poly_order: int = 3) -> np.ndarray:
        """Savitzky-Golay 필터"""
        if len(data) < window:
            return data
        return signal.savgol_filter(data, window, poly_order)
    
    def _butterworth_filter(self, data: np.ndarray, cutoff: float = 0.1, order: int = 4) -> np.ndarray:
        """Butterworth 저역 통과 필터"""
        if len(data) < 10:
            return data
        
        nyquist = 0.5
        normal_cutoff = cutoff / nyquist
        
        b, a = signal.butter(order, normal_cutoff, btype='low')
        filtered_data = signal.filtfilt(b, a, data)
        
        return filtered_data
    
    def _simple_kalman_filter(self, data: np.ndarray) -> np.ndarray:
        """간단한 칼만 필터"""
        if len(data) == 0:
            return data
        
        # 칼만 필터 매개변수
        Q = 1e-5  # 프로세스 노이즈 분산
        R = 0.1   # 측정 노이즈 분산
        
        # 초기값
        x_hat = data[0]  # 초기 추정값
        P = 1.0          # 초기 오차 공분산
        
        filtered_data = np.zeros_like(data)
        
        for i, measurement in enumerate(data):
            # 예측 단계
            x_hat_minus = x_hat
            P_minus = P + Q
            
            # 업데이트 단계
            K = P_minus / (P_minus + R)  # 칼만 게인
            x_hat = x_hat_minus + K * (measurement - x_hat_minus)
            P = (1 - K) * P_minus
            
            filtered_data[i] = x_hat
        
        return filtered_data
    
    def _wavelet_denoise(self, data: np.ndarray) -> np.ndarray:
        """웨이블릿 디노이즈 (단순 버전)"""
        # 여기서는 간단한 이동평균으로 대체
        window = min(5, len(data) // 4)
        if window <= 1:
            return data
        
        return np.convolve(data, np.ones(window)/window, mode='same')
    
    def _median_filter(self, data: np.ndarray, window: int = 5) -> np.ndarray:
        """미디언 필터"""
        if len(data) < window:
            return data
        return signal.medfilt(data, kernel_size=window)
    
    def apply_adaptive_filter(self, data: np.ndarray, volatility: float) -> np.ndarray:
        """변동성에 따른 적응형 필터링"""
        if volatility > 0.1:  # 고변동성
            # 강한 필터링
            filtered = self._butterworth_filter(data, cutoff=0.05)
            filtered = self._median_filter(filtered, window=7)
        elif volatility > 0.05:  # 중간 변동성
            # 중간 필터링
            filtered = self._savitzky_golay_filter(data)
        else:  # 저변동성
            # 가벼운 필터링
            filtered = self._simple_kalman_filter(data)
        
        return filtered

class TrendAnalyzer:
    """트렌드 분석기"""
    
    def __init__(self):
        self.trend_methods = ['linear', 'polynomial', 'exponential']
    
    def detect_trend_strength(self, data: np.ndarray) -> Dict[str, float]:
        """트렌드 강도 감지"""
        if len(data) < 2:
            return {'strength': 0.0, 'direction': 0, 'consistency': 0.0}
        
        x = np.arange(len(data))
        
        # 선형 트렌드
        linear_coef = np.polyfit(x, data, 1)[0]
        linear_pred = np.polyval(np.polyfit(x, data, 1), x)
        linear_r2 = 1 - np.sum((data - linear_pred)**2) / np.sum((data - np.mean(data))**2)
        
        # 다항식 트렌드 (2차)
        if len(data) >= 3:
            poly_coef = np.polyfit(x, data, 2)
            poly_pred = np.polyval(poly_coef, x)
            poly_r2 = 1 - np.sum((data - poly_pred)**2) / np.sum((data - np.mean(data))**2)
        else:
            poly_r2 = linear_r2
        
        # 트렌드 강도 (R² 기반)
        trend_strength = max(0, max(linear_r2, poly_r2))
        
        # 트렌드 방향
        trend_direction = 1 if linear_coef > 0 else -1 if linear_coef < 0 else 0
        
        # 트렌드 일관성 (변곡점 개수 기반)
        diff_sign_changes = np.sum(np.diff(np.sign(np.diff(data))) != 0)
        consistency = max(0, 1 - diff_sign_changes / max(1, len(data) - 2))
        
        return {
            'strength': float(trend_strength),
            'direction': int(trend_direction),
            'consistency': float(consistency),
            'linear_slope': float(linear_coef),
            'r2_score': float(trend_strength)
        }
    
    def decompose_trend_seasonality(self, data: np.ndarray, period: int = 24) -> Dict[str, np.ndarray]:
        """트렌드-계절성 분해"""
        if len(data) < period * 2:
            return {
                'trend': data.copy(),
                'seasonal': np.zeros_like(data),
                'residual': np.zeros_like(data)
            }
        
        # 단순 이동 평균으로 트렌드 추출
        trend = np.zeros_like(data, dtype=float)
        half_period = period // 2
        
        for i in range(len(data)):
            start = max(0, i - half_period)
            end = min(len(data), i + half_period + 1)
            trend[i] = np.mean(data[start:end])
        
        # 계절성 성분 (트렌드 제거 후 주기성 추출)
        detrended = data - trend
        seasonal = np.zeros_like(data, dtype=float)
        
        for i in range(len(data)):
            seasonal_indices = list(range(i % period, len(data), period))
            if len(seasonal_indices) > 1:
                seasonal[i] = np.mean(detrended[seasonal_indices])
        
        # 잔차
        residual = data - trend - seasonal
        
        return {
            'trend': trend,
            'seasonal': seasonal,
            'residual': residual
        }

class CrossHorizonConsistency:
    """시간대 간 일관성 제약"""
    
    def __init__(self):
        self.consistency_weights = {
            (1, 4): 0.8,    # 1h-4h 높은 일관성
            (4, 24): 0.7,   # 4h-24h 중간 일관성
            (24, 72): 0.6,  # 24h-72h 중간 일관성
            (72, 168): 0.5  # 72h-168h 낮은 일관성
        }
    
    def compute_consistency_score(self, predictions: Dict[int, float]) -> float:
        """일관성 점수 계산"""
        if len(predictions) < 2:
            return 1.0
        
        sorted_horizons = sorted(predictions.keys())
        consistency_scores = []
        
        for i in range(len(sorted_horizons) - 1):
            h1, h2 = sorted_horizons[i], sorted_horizons[i + 1]
            
            # 예측값 차이 정규화
            pred_diff = abs(predictions[h1] - predictions[h2])
            max_pred = max(abs(predictions[h1]), abs(predictions[h2]))
            
            if max_pred > 0:
                normalized_diff = pred_diff / max_pred
                consistency = max(0, 1 - normalized_diff)
            else:
                consistency = 1.0
            
            # 시간대 가중치 적용
            weight = self.consistency_weights.get((h1, h2), 0.5)
            weighted_consistency = consistency * weight
            
            consistency_scores.append(weighted_consistency)
        
        return np.mean(consistency_scores) if consistency_scores else 1.0
    
    def apply_consistency_constraint(self, predictions: Dict[int, float], 
                                   constraint_strength: float = 0.5) -> Dict[int, float]:
        """일관성 제약 적용"""
        if len(predictions) < 2:
            return predictions
        
        sorted_horizons = sorted(predictions.keys())
        adjusted_predictions = predictions.copy()
        
        # 인접한 시간대 간 조정
        for i in range(len(sorted_horizons) - 1):
            h1, h2 = sorted_horizons[i], sorted_horizons[i + 1]
            
            pred1, pred2 = adjusted_predictions[h1], adjusted_predictions[h2]
            
            # 가중 평균으로 조정
            weight = self.consistency_weights.get((h1, h2), 0.5) * constraint_strength
            
            adjusted_pred1 = pred1 * (1 - weight) + pred2 * weight
            adjusted_pred2 = pred2 * (1 - weight) + pred1 * weight
            
            adjusted_predictions[h1] = adjusted_pred1
            adjusted_predictions[h2] = adjusted_pred2
        
        return adjusted_predictions

class HierarchicalFeatureExtractor:
    """계층적 특성 추출기"""
    
    def __init__(self):
        self.scalers = {}
        self.pca_models = {}
    
    def extract_multi_scale_features(self, data: np.ndarray, levels: List[TemporalLevel]) -> Dict[str, np.ndarray]:
        """다중 스케일 특성 추출"""
        features = {}
        
        for level in levels:
            level_features = self._extract_level_features(data, level)
            features[level.name] = level_features
        
        return features
    
    def _extract_level_features(self, data: np.ndarray, level: TemporalLevel) -> np.ndarray:
        """특정 레벨의 특성 추출"""
        if len(data) < level.window_size:
            return np.array([])
        
        # 시간 윈도우별 샘플링
        sampled_data = data[::level.sampling_rate]
        
        if len(sampled_data) < level.window_size:
            return np.array([])
        
        features = []
        
        for i in range(len(sampled_data) - level.window_size + 1):
            window_data = sampled_data[i:i + level.window_size]
            
            # 통계적 특성
            window_features = [
                np.mean(window_data),           # 평균
                np.std(window_data),            # 표준편차
                np.min(window_data),            # 최솟값
                np.max(window_data),            # 최댓값
                np.median(window_data),         # 중앙값
                np.percentile(window_data, 25), # 1사분위수
                np.percentile(window_data, 75), # 3사분위수
            ]
            
            # 트렌드 특성
            if len(window_data) > 1:
                slope = np.polyfit(range(len(window_data)), window_data, 1)[0]
                window_features.append(slope)
            else:
                window_features.append(0.0)
            
            # 변화율 특성
            if len(window_data) > 1:
                returns = np.diff(window_data) / window_data[:-1]
                window_features.extend([
                    np.mean(returns),
                    np.std(returns),
                    np.sum(returns > 0) / len(returns),  # 상승 비율
                ])
            else:
                window_features.extend([0.0, 0.0, 0.5])
            
            features.append(window_features)
        
        return np.array(features)
    
    def combine_hierarchical_features(self, level_features: Dict[str, np.ndarray], 
                                    target_length: int) -> np.ndarray:
        """계층적 특성 결합"""
        combined_features = []
        
        for level_name, features in level_features.items():
            if len(features) == 0:
                continue
            
            # 길이 맞춤 (보간 또는 샘플링)
            if len(features) > target_length:
                # 다운샘플링
                indices = np.linspace(0, len(features) - 1, target_length, dtype=int)
                resampled = features[indices]
            elif len(features) < target_length:
                # 업샘플링 (마지막 값 반복)
                resampled = np.zeros((target_length, features.shape[1]))
                resampled[:len(features)] = features
                resampled[len(features):] = features[-1] if len(features) > 0 else 0
            else:
                resampled = features
            
            # 정규화
            if level_name not in self.scalers:
                self.scalers[level_name] = StandardScaler()
                normalized = self.scalers[level_name].fit_transform(resampled)
            else:
                normalized = self.scalers[level_name].transform(resampled)
            
            combined_features.append(normalized)
        
        if combined_features:
            return np.concatenate(combined_features, axis=1)
        else:
            return np.array([])

class TemporalHierarchyEngine:
    """시간 계층 모델링 엔진"""
    
    def __init__(self):
        # 시간 계층 레벨 정의
        self.temporal_levels = [
            TemporalLevel("short_term", window_size=24, sampling_rate=1, weight=0.2, noise_threshold=0.05),
            TemporalLevel("medium_term", window_size=72, sampling_rate=3, weight=0.3, noise_threshold=0.03),
            TemporalLevel("long_term", window_size=168, sampling_rate=6, weight=0.5, noise_threshold=0.02),
        ]
        
        # 구성 요소 초기화
        self.noise_filter = NoiseFilter()
        self.trend_analyzer = TrendAnalyzer()
        self.consistency_checker = CrossHorizonConsistency()
        self.feature_extractor = HierarchicalFeatureExtractor()
        
        # 상태 저장
        self.analysis_cache = {}
        self.setup_logging()
    
    def setup_logging(self):
        """로깅 설정"""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def analyze_temporal_hierarchy(self, data: np.ndarray) -> Dict:
        """완전한 시간 계층 분석"""
        self.logger.info(f"🕒 시간 계층 분석 시작 - 데이터 길이: {len(data)}")
        
        if len(data) < 24:
            self.logger.warning("데이터가 너무 짧습니다 (24시간 미만)")
            return self._create_empty_analysis()
        
        # 1. 노이즈 필터링
        filtered_data = self._apply_hierarchical_filtering(data)
        
        # 2. 다중 스케일 트렌드 분석
        trend_analysis = self._multi_scale_trend_analysis(filtered_data)
        
        # 3. 계층적 특성 추출
        hierarchical_features = self._extract_hierarchical_features(filtered_data)
        
        # 4. 시간대 간 일관성 분석
        consistency_analysis = self._analyze_cross_horizon_consistency(filtered_data)
        
        # 5. 적응형 가중치 계산
        adaptive_weights = self._compute_adaptive_weights(trend_analysis, consistency_analysis)
        
        # 6. 노이즈 수준 평가
        noise_analysis = self._evaluate_noise_levels(data, filtered_data)
        
        analysis_result = {
            'timestamp': datetime.now().isoformat(),
            'data_length': len(data),
            'filtered_data': filtered_data.tolist(),
            'trend_analysis': trend_analysis,
            'hierarchical_features': hierarchical_features,
            'consistency_analysis': consistency_analysis,
            'adaptive_weights': adaptive_weights,
            'noise_analysis': noise_analysis,
            'temporal_levels': [
                {
                    'name': level.name,
                    'window_size': level.window_size,
                    'sampling_rate': level.sampling_rate,
                    'weight': level.weight
                } for level in self.temporal_levels
            ]
        }
        
        self.logger.info("✅ 시간 계층 분석 완료")
        return analysis_result
    
    def _apply_hierarchical_filtering(self, data: np.ndarray) -> np.ndarray:
        """계층적 노이즈 필터링"""
        # 변동성 계산
        volatility = np.std(data) / np.mean(np.abs(data)) if np.mean(np.abs(data)) > 0 else 0
        
        # 적응형 필터링 적용
        filtered = self.noise_filter.apply_adaptive_filter(data, volatility)
        
        return filtered
    
    def _multi_scale_trend_analysis(self, data: np.ndarray) -> Dict:
        """다중 스케일 트렌드 분석"""
        trend_results = {}
        
        for level in self.temporal_levels:
            if len(data) >= level.window_size:
                # 해당 레벨의 데이터 추출
                level_data = data[-level.window_size::level.sampling_rate]
                
                # 트렌드 분석
                trend_info = self.trend_analyzer.detect_trend_strength(level_data)
                
                # 트렌드-계절성 분해
                decomposition = self.trend_analyzer.decompose_trend_seasonality(
                    level_data, period=min(24, len(level_data) // 4)
                )
                
                trend_results[level.name] = {
                    'trend_strength': trend_info,
                    'decomposition': {
                        'trend_component': decomposition['trend'].tolist()[-10:],  # 최근 10개만
                        'seasonal_component': decomposition['seasonal'].tolist()[-10:],
                        'residual_component': decomposition['residual'].tolist()[-10:],
                    },
                    'volatility': float(np.std(level_data)),
                    'mean_level': float(np.mean(level_data))
                }
        
        return trend_results
    
    def _extract_hierarchical_features(self, data: np.ndarray) -> Dict:
        """계층적 특성 추출"""
        # 다중 스케일 특성 추출
        level_features = self.feature_extractor.extract_multi_scale_features(data, self.temporal_levels)
        
        # 특성 통계
        feature_stats = {}
        for level_name, features in level_features.items():
            if len(features) > 0:
                feature_stats[level_name] = {
                    'feature_count': int(features.shape[1]) if features.ndim > 1 else 1,
                    'sample_count': int(features.shape[0]) if features.ndim > 0 else 0,
                    'mean_values': np.mean(features, axis=0).tolist() if features.ndim > 1 else [float(np.mean(features))],
                    'std_values': np.std(features, axis=0).tolist() if features.ndim > 1 else [float(np.std(features))]
                }
            else:
                feature_stats[level_name] = {
                    'feature_count': 0,
                    'sample_count': 0,
                    'mean_values': [],
                    'std_values': []
                }
        
        return feature_stats
    
    def _analyze_cross_horizon_consistency(self, data: np.ndarray) -> Dict:
        """시간대 간 일관성 분석"""
        if len(data) < 168:  # 최소 1주일 데이터 필요
            return {'consistency_score': 1.0, 'horizon_correlations': {}}
        
        horizons = [1, 4, 24, 72, 168]
        predictions = {}
        
        # 각 시간대의 단순 예측 생성 (트렌드 기반)
        for horizon in horizons:
            if len(data) >= horizon + 24:  # 충분한 데이터가 있는 경우
                recent_data = data[-24:]  # 최근 24시간
                trend_slope = np.polyfit(range(len(recent_data)), recent_data, 1)[0]
                predictions[horizon] = float(data[-1] + trend_slope * horizon)
        
        # 일관성 점수 계산
        consistency_score = self.consistency_checker.compute_consistency_score(predictions)
        
        # 시간대 간 상관관계 분석
        horizon_correlations = {}
        for i, h1 in enumerate(horizons):
            for h2 in horizons[i+1:]:
                if len(data) >= max(h1, h2) + 24:
                    data1 = data[:-max(h1, h2)] if max(h1, h2) < len(data) else data[:1]
                    data2 = data[h2-h1:] if h2 > h1 and h2-h1 < len(data) else data[:1]
                    
                    if len(data1) > 1 and len(data2) > 1 and len(data1) == len(data2):
                        corr, _ = pearsonr(data1, data2)
                        horizon_correlations[f'{h1}h_{h2}h'] = float(corr) if not np.isnan(corr) else 0.0
        
        return {
            'consistency_score': consistency_score,
            'horizon_correlations': horizon_correlations,
            'predictions': predictions
        }
    
    def _compute_adaptive_weights(self, trend_analysis: Dict, consistency_analysis: Dict) -> Dict[str, float]:
        """적응형 가중치 계산"""
        weights = {}
        
        for level in self.temporal_levels:
            base_weight = level.weight
            
            if level.name in trend_analysis:
                trend_strength = trend_analysis[level.name]['trend_strength']['strength']
                consistency_factor = consistency_analysis['consistency_score']
                
                # 트렌드 강도와 일관성에 따라 가중치 조정
                adjusted_weight = base_weight * (0.5 + 0.5 * trend_strength) * (0.8 + 0.2 * consistency_factor)
                weights[level.name] = float(adjusted_weight)
            else:
                weights[level.name] = base_weight
        
        # 정규화
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        
        return weights
    
    def _evaluate_noise_levels(self, original_data: np.ndarray, filtered_data: np.ndarray) -> Dict:
        """노이즈 수준 평가"""
        if len(original_data) != len(filtered_data):
            return {'noise_level': 0.0, 'filter_effectiveness': 0.0}
        
        # 노이즈 추정 (원본 - 필터링된 데이터)
        noise = original_data - filtered_data
        
        # 노이즈 통계
        noise_level = float(np.std(noise))
        signal_level = float(np.std(filtered_data))
        
        # 신호 대 노이즈 비율
        snr = signal_level / noise_level if noise_level > 0 else float('inf')
        
        # 필터 효과성
        original_volatility = np.std(original_data)
        filtered_volatility = np.std(filtered_data)
        filter_effectiveness = (original_volatility - filtered_volatility) / original_volatility if original_volatility > 0 else 0
        
        return {
            'noise_level': noise_level,
            'signal_level': signal_level,
            'snr': float(snr) if snr != float('inf') else 1000.0,
            'filter_effectiveness': float(filter_effectiveness),
            'noise_distribution': {
                'mean': float(np.mean(noise)),
                'std': float(np.std(noise)),
                'skewness': float(self._calculate_skewness(noise)),
                'kurtosis': float(self._calculate_kurtosis(noise))
            }
        }
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """비대칭도 계산"""
        if len(data) < 3:
            return 0.0
        
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        
        if std == 0:
            return 0.0
        
        skewness = np.mean(((data - mean) / std) ** 3)
        return skewness
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """첨도 계산"""
        if len(data) < 4:
            return 0.0
        
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        
        if std == 0:
            return 0.0
        
        kurtosis = np.mean(((data - mean) / std) ** 4) - 3
        return kurtosis
    
    def _create_empty_analysis(self) -> Dict:
        """빈 분석 결과 생성"""
        return {
            'timestamp': datetime.now().isoformat(),
            'data_length': 0,
            'filtered_data': [],
            'trend_analysis': {},
            'hierarchical_features': {},
            'consistency_analysis': {'consistency_score': 0.0, 'horizon_correlations': {}},
            'adaptive_weights': {level.name: level.weight for level in self.temporal_levels},
            'noise_analysis': {'noise_level': 0.0, 'filter_effectiveness': 0.0},
            'temporal_levels': []
        }
    
    def optimize_predictions(self, raw_predictions: Dict[int, float], 
                           hierarchy_analysis: Dict) -> Dict[int, float]:
        """계층 분석 결과로 예측 최적화"""
        if not raw_predictions:
            return raw_predictions
        
        # 적응형 가중치 적용
        adaptive_weights = hierarchy_analysis.get('adaptive_weights', {})
        
        # 일관성 제약 적용
        optimized_predictions = self.consistency_checker.apply_consistency_constraint(
            raw_predictions, constraint_strength=0.3
        )
        
        # 트렌드 분석 결과 반영
        trend_analysis = hierarchy_analysis.get('trend_analysis', {})
        
        for horizon, prediction in optimized_predictions.items():
            horizon_name = self._get_horizon_level_name(horizon)
            
            if horizon_name in trend_analysis:
                trend_info = trend_analysis[horizon_name]['trend_strength']
                
                # 트렌드 강도에 따른 예측 조정
                if trend_info['strength'] > 0.7:  # 강한 트렌드
                    trend_adjustment = trend_info['linear_slope'] * horizon * 0.5
                    optimized_predictions[horizon] = prediction + trend_adjustment
        
        return optimized_predictions
    
    def _get_horizon_level_name(self, horizon_hours: int) -> str:
        """시간대를 레벨명으로 변환"""
        if horizon_hours <= 24:
            return "short_term"
        elif horizon_hours <= 72:
            return "medium_term"
        else:
            return "long_term"

def main():
    """메인 테스트 함수"""
    # 테스트 데이터 생성
    np.random.seed(42)
    time_points = 1000
    
    # 합성 BTC 가격 데이터 (트렌드 + 노이즈)
    trend = np.linspace(50000, 55000, time_points)
    seasonal = 1000 * np.sin(2 * np.pi * np.arange(time_points) / 24)  # 24시간 주기
    noise = np.random.normal(0, 500, time_points)
    synthetic_data = trend + seasonal + noise
    
    # 엔진 초기화 및 분석
    engine = TemporalHierarchyEngine()
    
    print("🕒 Temporal Hierarchy Modeling Engine Test")
    print("="*60)
    
    # 계층 분석 실행
    analysis_result = engine.analyze_temporal_hierarchy(synthetic_data)
    
    # 결과 출력
    print(f"📊 분석 완료:")
    print(f"  데이터 길이: {analysis_result['data_length']}")
    print(f"  일관성 점수: {analysis_result['consistency_analysis']['consistency_score']:.3f}")
    print(f"  노이즈 레벨: {analysis_result['noise_analysis']['noise_level']:.2f}")
    print(f"  필터 효과성: {analysis_result['noise_analysis']['filter_effectiveness']:.3f}")
    
    print(f"\n🎯 시간 계층별 트렌드 분석:")
    for level_name, trend_data in analysis_result['trend_analysis'].items():
        strength = trend_data['trend_strength']['strength']
        direction = "상승" if trend_data['trend_strength']['direction'] > 0 else "하락" if trend_data['trend_strength']['direction'] < 0 else "횡보"
        print(f"  {level_name}: {direction} 트렌드 (강도: {strength:.3f})")
    
    print(f"\n⚖️ 적응형 가중치:")
    for level_name, weight in analysis_result['adaptive_weights'].items():
        print(f"  {level_name}: {weight:.3f}")
    
    # 예측 최적화 테스트
    raw_predictions = {1: 55100, 4: 55200, 24: 55500, 72: 56000, 168: 56500}
    optimized_predictions = engine.optimize_predictions(raw_predictions, analysis_result)
    
    print(f"\n🎯 예측 최적화 결과:")
    for horizon in sorted(raw_predictions.keys()):
        raw = raw_predictions[horizon]
        opt = optimized_predictions[horizon]
        change = opt - raw
        print(f"  {horizon}h: ${raw:,.0f} → ${opt:,.0f} ({change:+.0f})")
    
    # 결과 저장
    with open('temporal_hierarchy_analysis.json', 'w', encoding='utf-8') as f:
        json.dump(analysis_result, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 분석 결과 저장: temporal_hierarchy_analysis.json")
    
    return analysis_result

if __name__ == "__main__":
    main()