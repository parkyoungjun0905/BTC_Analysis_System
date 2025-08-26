#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
데이터 품질 향상 파이프라인
- 이상치 탐지 및 처리
- 결측치 대체 전략
- 노이즈 감소 기법
- 신호 추출 및 강화
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

import os
import sys
import json
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union, Callable
import logging
import pickle
from dataclasses import dataclass
from abc import ABC, abstractmethod

# 과학 계산
from scipy import stats
from scipy.signal import savgol_filter, butter, filtfilt, find_peaks
from scipy.interpolate import interp1d, UnivariateSpline
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.optimize import minimize_scalar
from scipy.fft import fft, fftfreq

# 머신러닝
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA, FastICA
from sklearn.cluster import DBSCAN, IsolationForest
from sklearn.ensemble import IsolationForest, LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import HuberRegressor, RANSACRegressor
from sklearn.impute import KNNImputer

# 통계 및 시각화
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.dates import DateFormatter
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class QualityMetrics:
    """데이터 품질 메트릭"""
    completeness: float  # 완전성 (결측치 비율)
    consistency: float   # 일관성 
    accuracy: float      # 정확성
    validity: float      # 유효성
    uniqueness: float    # 유일성
    timeliness: float    # 적시성
    overall_score: float # 종합 점수

class OutlierDetector(ABC):
    """이상치 탐지 기본 클래스"""
    
    @abstractmethod
    def detect(self, data: np.ndarray) -> np.ndarray:
        """이상치 탐지 (True: 이상치, False: 정상)"""
        pass

class StatisticalOutlierDetector(OutlierDetector):
    """통계적 이상치 탐지"""
    
    def __init__(self, method: str = 'zscore', threshold: float = 3.0):
        """
        Args:
            method: 탐지 방법 ('zscore', 'iqr', 'modified_zscore')
            threshold: 임계값
        """
        self.method = method
        self.threshold = threshold
    
    def detect(self, data: np.ndarray) -> np.ndarray:
        """통계적 방법으로 이상치 탐지"""
        data = np.asarray(data).flatten()
        
        if self.method == 'zscore':
            z_scores = np.abs(stats.zscore(data, nan_policy='omit'))
            return z_scores > self.threshold
            
        elif self.method == 'iqr':
            q1, q3 = np.nanpercentile(data, [25, 75])
            iqr = q3 - q1
            lower_bound = q1 - self.threshold * iqr
            upper_bound = q3 + self.threshold * iqr
            return (data < lower_bound) | (data > upper_bound)
            
        elif self.method == 'modified_zscore':
            median = np.nanmedian(data)
            mad = np.nanmedian(np.abs(data - median))
            modified_z_scores = 0.6745 * (data - median) / mad
            return np.abs(modified_z_scores) > self.threshold
            
        else:
            raise ValueError(f"지원되지 않는 방법: {self.method}")

class IsolationForestDetector(OutlierDetector):
    """Isolation Forest 이상치 탐지"""
    
    def __init__(self, contamination: float = 0.1, random_state: int = 42):
        """
        Args:
            contamination: 이상치 비율
            random_state: 랜덤 시드
        """
        self.contamination = contamination
        self.random_state = random_state
        self.model = IsolationForest(
            contamination=contamination, 
            random_state=random_state
        )
    
    def detect(self, data: np.ndarray) -> np.ndarray:
        """Isolation Forest로 이상치 탐지"""
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        # 결측치 제거
        mask = ~np.isnan(data).any(axis=1)
        clean_data = data[mask]
        
        if len(clean_data) == 0:
            return np.zeros(len(data), dtype=bool)
        
        # 이상치 탐지
        outlier_labels = self.model.fit_predict(clean_data)
        
        # 결과 매핑
        result = np.zeros(len(data), dtype=bool)
        result[mask] = outlier_labels == -1
        
        return result

class TimeSeriesOutlierDetector(OutlierDetector):
    """시계열 특화 이상치 탐지"""
    
    def __init__(self, window_size: int = 24, threshold: float = 3.0):
        """
        Args:
            window_size: 윈도우 크기
            threshold: 임계값
        """
        self.window_size = window_size
        self.threshold = threshold
    
    def detect(self, data: np.ndarray) -> np.ndarray:
        """시계열 패턴 기반 이상치 탐지"""
        data = np.asarray(data).flatten()
        outliers = np.zeros(len(data), dtype=bool)
        
        # 롤링 통계
        rolling_mean = pd.Series(data).rolling(
            window=self.window_size, center=True
        ).mean()
        rolling_std = pd.Series(data).rolling(
            window=self.window_size, center=True
        ).std()
        
        # 이상치 탐지
        for i in range(len(data)):
            if pd.isna(rolling_mean.iloc[i]) or pd.isna(rolling_std.iloc[i]):
                continue
            
            deviation = abs(data[i] - rolling_mean.iloc[i])
            if deviation > self.threshold * rolling_std.iloc[i]:
                outliers[i] = True
        
        return outliers

class MissingValueImputer(ABC):
    """결측치 대체 기본 클래스"""
    
    @abstractmethod
    def impute(self, data: np.ndarray) -> np.ndarray:
        """결측치 대체"""
        pass

class TimeSeriesImputer(MissingValueImputer):
    """시계열 특화 결측치 대체"""
    
    def __init__(self, method: str = 'interpolation', **kwargs):
        """
        Args:
            method: 대체 방법 ('interpolation', 'forward_fill', 'backward_fill', 'seasonal')
            **kwargs: 추가 파라미터
        """
        self.method = method
        self.kwargs = kwargs
    
    def impute(self, data: np.ndarray) -> np.ndarray:
        """시계열 특화 결측치 대체"""
        data = np.asarray(data).copy()
        
        if self.method == 'interpolation':
            # 선형 보간
            mask = ~np.isnan(data)
            if np.sum(mask) < 2:
                return data
            
            indices = np.arange(len(data))
            f = interp1d(indices[mask], data[mask], 
                        kind=self.kwargs.get('kind', 'linear'),
                        bounds_error=False, fill_value='extrapolate')
            data[~mask] = f(indices[~mask])
            
        elif self.method == 'forward_fill':
            # 전진 채우기
            for i in range(1, len(data)):
                if np.isnan(data[i]) and not np.isnan(data[i-1]):
                    data[i] = data[i-1]
                    
        elif self.method == 'backward_fill':
            # 후진 채우기
            for i in range(len(data)-2, -1, -1):
                if np.isnan(data[i]) and not np.isnan(data[i+1]):
                    data[i] = data[i+1]
                    
        elif self.method == 'seasonal':
            # 계절성 기반 대체
            period = self.kwargs.get('period', 24)  # 기본 24시간 주기
            
            for i in range(len(data)):
                if np.isnan(data[i]):
                    # 같은 시간대의 과거 값들 찾기
                    seasonal_indices = np.arange(i % period, len(data), period)
                    seasonal_values = data[seasonal_indices]
                    seasonal_values = seasonal_values[~np.isnan(seasonal_values)]
                    
                    if len(seasonal_values) > 0:
                        data[i] = np.mean(seasonal_values)
        
        return data

class KNNTimeSeriesImputer(MissingValueImputer):
    """KNN 기반 시계열 결측치 대체"""
    
    def __init__(self, n_neighbors: int = 5, window_size: int = 48):
        """
        Args:
            n_neighbors: 이웃 수
            window_size: 윈도우 크기
        """
        self.n_neighbors = n_neighbors
        self.window_size = window_size
    
    def impute(self, data: np.ndarray) -> np.ndarray:
        """KNN으로 시계열 결측치 대체"""
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        # 윈도우 기반 특성 생성
        windowed_data = []
        for i in range(self.window_size, len(data)):
            window = data[i-self.window_size:i].flatten()
            windowed_data.append(window)
        
        windowed_data = np.array(windowed_data)
        
        # KNN 대체
        imputer = KNNImputer(n_neighbors=self.n_neighbors)
        imputed_windowed = imputer.fit_transform(windowed_data)
        
        # 결과 재구성
        result = data.copy()
        for i, window in enumerate(imputed_windowed):
            window = window.reshape(-1, data.shape[1])
            result[i+self.window_size-1] = window[-1]
        
        return result.flatten() if result.shape[1] == 1 else result

class NoiseReducer(ABC):
    """노이즈 감소 기본 클래스"""
    
    @abstractmethod
    def reduce(self, data: np.ndarray) -> np.ndarray:
        """노이즈 감소"""
        pass

class SavitzkyGolayDenoiser(NoiseReducer):
    """Savitzky-Golay 필터 노이즈 감소"""
    
    def __init__(self, window_length: int = 11, polyorder: int = 3):
        """
        Args:
            window_length: 윈도우 길이 (홀수)
            polyorder: 다항식 차수
        """
        self.window_length = window_length if window_length % 2 == 1 else window_length + 1
        self.polyorder = polyorder
    
    def reduce(self, data: np.ndarray) -> np.ndarray:
        """Savitzky-Golay 필터 적용"""
        if len(data) < self.window_length:
            return data
        
        return savgol_filter(data, self.window_length, self.polyorder)

class WaveletDenoiser(NoiseReducer):
    """웨이블릿 기반 노이즈 감소"""
    
    def __init__(self, wavelet: str = 'db4', threshold_mode: str = 'soft'):
        """
        Args:
            wavelet: 웨이블릿 타입
            threshold_mode: 임계값 모드
        """
        self.wavelet = wavelet
        self.threshold_mode = threshold_mode
        
        try:
            import pywt
            self.pywt = pywt
            self.available = True
        except ImportError:
            self.available = False
            logger.warning("PyWavelets 라이브러리가 없습니다. 웨이블릿 노이즈 제거를 건너뜁니다.")
    
    def reduce(self, data: np.ndarray) -> np.ndarray:
        """웨이블릿 노이즈 감소"""
        if not self.available:
            return data
        
        # 웨이블릿 분해
        coeffs = self.pywt.wavedec(data, self.wavelet)
        
        # 임계값 계산 (MAD 기반)
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        threshold = sigma * np.sqrt(2 * np.log(len(data)))
        
        # 임계값 적용
        coeffs_thresh = list(coeffs)
        coeffs_thresh[1:] = [self.pywt.threshold(c, threshold, self.threshold_mode) 
                            for c in coeffs[1:]]
        
        # 웨이블릿 재구성
        return self.pywt.waverec(coeffs_thresh, self.wavelet)

class AdaptiveNoiseReducer(NoiseReducer):
    """적응형 노이즈 감소"""
    
    def __init__(self, methods: List[str] = None):
        """
        Args:
            methods: 사용할 방법들 리스트
        """
        self.methods = methods or ['savgol', 'median', 'gaussian']
        self.reducers = {
            'savgol': SavitzkyGolayDenoiser(),
            'median': lambda data: median_filter(data, size=5),
            'gaussian': lambda data: gaussian_filter1d(data, sigma=1.0)
        }
    
    def reduce(self, data: np.ndarray) -> np.ndarray:
        """적응형 노이즈 감소"""
        results = []
        
        for method in self.methods:
            if method in self.reducers:
                if hasattr(self.reducers[method], 'reduce'):
                    denoised = self.reducers[method].reduce(data)
                else:
                    denoised = self.reducers[method](data)
                results.append(denoised)
        
        if not results:
            return data
        
        # 결과들의 가중 평균
        weights = np.ones(len(results)) / len(results)
        return np.average(results, axis=0, weights=weights)

class SignalExtractor:
    """신호 추출 및 강화"""
    
    def __init__(self):
        """신호 추출기 초기화"""
        pass
    
    def extract_trend(self, data: np.ndarray, method: str = 'hp_filter', **kwargs) -> np.ndarray:
        """트렌드 추출"""
        if method == 'hp_filter':
            return self._hp_filter(data, **kwargs)
        elif method == 'moving_average':
            window = kwargs.get('window', 24)
            return pd.Series(data).rolling(window, center=True).mean().values
        elif method == 'lowess':
            return self._lowess_trend(data, **kwargs)
        else:
            raise ValueError(f"지원되지 않는 방법: {method}")
    
    def _hp_filter(self, data: np.ndarray, lambda_param: float = 1600) -> np.ndarray:
        """Hodrick-Prescott 필터 (단순화된 버전)"""
        from scipy.sparse import diags
        from scipy.sparse.linalg import spsolve
        
        n = len(data)
        if n < 4:
            return data
        
        # 2차 차분 행렬
        d2 = diags([1, -2, 1], [0, 1, 2], shape=(n-2, n)).tocsc()
        
        # HP 필터 행렬
        I = diags([1], [0], shape=(n, n)).tocsc()
        trend = spsolve(I + lambda_param * d2.T @ d2, data)
        
        return trend
    
    def _lowess_trend(self, data: np.ndarray, frac: float = 0.3) -> np.ndarray:
        """LOWESS 트렌드 (단순화된 버전)"""
        n = len(data)
        trend = np.zeros(n)
        
        h = int(np.ceil(frac * n))
        
        for i in range(n):
            # 이웃 인덱스
            start = max(0, i - h // 2)
            end = min(n, i + h // 2 + 1)
            
            # 가중치 (거리 기반)
            indices = np.arange(start, end)
            weights = np.maximum(0, 1 - np.abs(indices - i) / h)
            
            # 가중 회귀
            if np.sum(weights) > 0:
                X = indices.reshape(-1, 1)
                y = data[start:end]
                w = weights
                
                # 가중 최소제곱법
                X_w = X * w.reshape(-1, 1)
                y_w = y * w
                
                try:
                    beta = np.linalg.solve(X_w.T @ X, X_w.T @ y_w)
                    trend[i] = beta @ np.array([i])
                except:
                    trend[i] = np.average(y, weights=w)
            else:
                trend[i] = data[i]
        
        return trend
    
    def extract_cycles(self, data: np.ndarray, min_period: int = 6, max_period: int = 72) -> Dict[str, np.ndarray]:
        """주기적 신호 추출 (FFT 기반)"""
        # FFT
        fft_values = fft(data)
        freqs = fftfreq(len(data))
        
        # 주기 범위 계산
        min_freq = 1 / max_period
        max_freq = 1 / min_period
        
        # 주파수 필터링
        filtered_fft = fft_values.copy()
        mask = (np.abs(freqs) < min_freq) | (np.abs(freqs) > max_freq)
        filtered_fft[mask] = 0
        
        # 역변환
        cycles = np.real(np.fft.ifft(filtered_fft))
        
        # 주요 주기 찾기
        power_spectrum = np.abs(fft_values) ** 2
        periods = 1 / freqs[freqs > 0]
        
        # 유효한 주기 범위 내에서 최대 파워 주기들
        valid_mask = (periods >= min_period) & (periods <= max_period)
        valid_periods = periods[valid_mask]
        valid_powers = power_spectrum[freqs > 0][valid_mask]
        
        # 상위 3개 주기
        top_indices = np.argsort(valid_powers)[-3:]
        dominant_periods = valid_periods[top_indices]
        
        result = {
            'cycles': cycles,
            'dominant_periods': dominant_periods,
            'power_spectrum': power_spectrum,
            'frequencies': freqs
        }
        
        return result
    
    def enhance_signal_to_noise_ratio(self, data: np.ndarray, 
                                    method: str = 'spectral_subtraction') -> np.ndarray:
        """신호 대 잡음 비율 향상"""
        if method == 'spectral_subtraction':
            return self._spectral_subtraction(data)
        elif method == 'wiener_filter':
            return self._wiener_filter(data)
        else:
            raise ValueError(f"지원되지 않는 방법: {method}")
    
    def _spectral_subtraction(self, data: np.ndarray, alpha: float = 2.0) -> np.ndarray:
        """스펙트럴 차감 방법"""
        # FFT
        fft_data = fft(data)
        magnitude = np.abs(fft_data)
        phase = np.angle(fft_data)
        
        # 노이즈 추정 (낮은 주파수 성분 제외한 평균)
        high_freq_mask = np.abs(fftfreq(len(data))) > 0.1
        noise_estimate = np.mean(magnitude[high_freq_mask])
        
        # 스펙트럴 차감
        enhanced_magnitude = magnitude - alpha * noise_estimate
        enhanced_magnitude = np.maximum(enhanced_magnitude, 0.1 * magnitude)
        
        # 재구성
        enhanced_fft = enhanced_magnitude * np.exp(1j * phase)
        enhanced_signal = np.real(np.fft.ifft(enhanced_fft))
        
        return enhanced_signal
    
    def _wiener_filter(self, data: np.ndarray) -> np.ndarray:
        """위너 필터 (단순화된 버전)"""
        # 신호의 자기상관 추정
        signal_power = np.var(data)
        
        # 노이즈 파워 추정 (고주파 성분)
        diff = np.diff(data)
        noise_power = np.var(diff) / 2
        
        # 위너 필터 계수
        wiener_coeff = signal_power / (signal_power + noise_power)
        
        # 필터 적용 (단순 스케일링)
        return data * wiener_coeff

class DataQualityEnhancementPipeline:
    """
    📈 데이터 품질 향상 파이프라인
    
    주요 기능:
    1. 종합적 이상치 탐지 및 처리
    2. 지능형 결측치 대체
    3. 다차원 노이즈 감소
    4. 신호 추출 및 강화
    5. 품질 메트릭 평가
    """
    
    def __init__(self):
        """파이프라인 초기화"""
        self.logger = logging.getLogger(__name__)
        
        # 컴포넌트 초기화
        self.outlier_detectors = {
            'statistical': StatisticalOutlierDetector(),
            'isolation_forest': IsolationForestDetector(),
            'timeseries': TimeSeriesOutlierDetector()
        }
        
        self.imputers = {
            'timeseries': TimeSeriesImputer(),
            'knn': KNNTimeSeriesImputer()
        }
        
        self.denoisers = {
            'savgol': SavitzkyGolayDenoiser(),
            'wavelet': WaveletDenoiser(),
            'adaptive': AdaptiveNoiseReducer()
        }
        
        self.signal_extractor = SignalExtractor()
        
        # 처리 이력
        self.processing_history = []
        
        self.logger.info("📈 데이터 품질 향상 파이프라인 초기화 완료")
    
    def assess_data_quality(self, data: pd.DataFrame) -> QualityMetrics:
        """데이터 품질 평가"""
        metrics = {}
        
        # 완전성 (결측치 비율)
        total_cells = data.size
        missing_cells = data.isnull().sum().sum()
        completeness = 1 - (missing_cells / total_cells) if total_cells > 0 else 0
        
        # 일관성 (데이터 타입 일관성)
        consistency_score = 0
        for col in data.columns:
            if data[col].dtype in ['int64', 'float64']:
                # 숫자형 데이터의 일관성
                finite_ratio = np.isfinite(data[col]).sum() / len(data[col])
                consistency_score += finite_ratio
        consistency_score = consistency_score / len(data.columns) if len(data.columns) > 0 else 0
        
        # 정확성 (이상치 비율로 근사)
        outlier_ratios = []
        for col in data.select_dtypes(include=[np.number]).columns:
            try:
                detector = StatisticalOutlierDetector()
                outliers = detector.detect(data[col].dropna().values)
                outlier_ratio = np.sum(outliers) / len(outliers) if len(outliers) > 0 else 0
                outlier_ratios.append(1 - outlier_ratio)
            except:
                outlier_ratios.append(0.9)  # 기본값
        accuracy = np.mean(outlier_ratios) if outlier_ratios else 0
        
        # 유효성 (유효한 값의 비율)
        validity_scores = []
        for col in data.columns:
            if data[col].dtype in ['int64', 'float64']:
                valid_ratio = (~data[col].isnull() & np.isfinite(data[col])).sum() / len(data[col])
                validity_scores.append(valid_ratio)
        validity = np.mean(validity_scores) if validity_scores else 0
        
        # 유일성 (중복 행 비율)
        uniqueness = 1 - (len(data) - len(data.drop_duplicates())) / len(data) if len(data) > 0 else 0
        
        # 적시성 (시간 간격 일관성, 시계열 데이터인 경우)
        timeliness = 1.0  # 기본값
        if isinstance(data.index, pd.DatetimeIndex):
            time_diffs = data.index.to_series().diff().dropna()
            if len(time_diffs) > 1:
                mode_diff = time_diffs.mode().iloc[0] if len(time_diffs.mode()) > 0 else time_diffs.median()
                consistent_intervals = (time_diffs == mode_diff).sum()
                timeliness = consistent_intervals / len(time_diffs)
        
        # 종합 점수
        overall_score = np.mean([
            completeness * 0.25,
            consistency * 0.20,
            accuracy * 0.25,
            validity * 0.15,
            uniqueness * 0.10,
            timeliness * 0.05
        ])
        
        return QualityMetrics(
            completeness=completeness,
            consistency=consistency,
            accuracy=accuracy,
            validity=validity,
            uniqueness=uniqueness,
            timeliness=timeliness,
            overall_score=overall_score
        )
    
    def detect_outliers(self, data: pd.DataFrame, 
                       methods: List[str] = None) -> Dict[str, pd.DataFrame]:
        """종합적 이상치 탐지"""
        methods = methods or ['statistical', 'isolation_forest', 'timeseries']
        
        results = {}
        
        for method in methods:
            if method not in self.outlier_detectors:
                continue
            
            detector = self.outlier_detectors[method]
            outlier_data = pd.DataFrame(index=data.index)
            
            for col in data.select_dtypes(include=[np.number]).columns:
                try:
                    outliers = detector.detect(data[col].values)
                    outlier_data[col] = outliers
                except Exception as e:
                    self.logger.warning(f"이상치 탐지 오류 ({method}, {col}): {e}")
                    outlier_data[col] = False
            
            results[method] = outlier_data
        
        return results
    
    def handle_outliers(self, data: pd.DataFrame, 
                       outliers: Dict[str, pd.DataFrame],
                       strategy: str = 'ensemble') -> pd.DataFrame:
        """이상치 처리"""
        cleaned_data = data.copy()
        
        if strategy == 'ensemble':
            # 앙상블: 여러 방법에서 공통으로 탐지된 이상치만 처리
            for col in cleaned_data.select_dtypes(include=[np.number]).columns:
                outlier_votes = np.zeros(len(cleaned_data), dtype=int)
                
                for method, outlier_df in outliers.items():
                    if col in outlier_df.columns:
                        outlier_votes += outlier_df[col].astype(int)
                
                # 과반수 투표로 이상치 결정
                threshold = len(outliers) // 2 + 1
                final_outliers = outlier_votes >= threshold
                
                if np.any(final_outliers):
                    # 이상치를 중앙값으로 대체
                    median_value = cleaned_data[col].median()
                    cleaned_data.loc[final_outliers, col] = median_value
                    
                    self.logger.info(f"{col}: {np.sum(final_outliers)}개 이상치 처리")
        
        return cleaned_data
    
    def impute_missing_values(self, data: pd.DataFrame, 
                            method: str = 'adaptive') -> pd.DataFrame:
        """결측치 대체"""
        imputed_data = data.copy()
        
        for col in imputed_data.select_dtypes(include=[np.number]).columns:
            if imputed_data[col].isnull().any():
                missing_count = imputed_data[col].isnull().sum()
                missing_ratio = missing_count / len(imputed_data[col])
                
                try:
                    if method == 'adaptive':
                        # 결측치 비율에 따라 적응적 방법 선택
                        if missing_ratio < 0.05:
                            # 적은 결측치: 선형 보간
                            imputer = TimeSeriesImputer('interpolation')
                        elif missing_ratio < 0.2:
                            # 중간 결측치: 계절성 기반
                            imputer = TimeSeriesImputer('seasonal')
                        else:
                            # 많은 결측치: KNN
                            imputer = KNNTimeSeriesImputer()
                    else:
                        imputer = self.imputers.get(method, TimeSeriesImputer())
                    
                    imputed_values = imputer.impute(imputed_data[col].values)
                    imputed_data[col] = imputed_values
                    
                    self.logger.info(f"{col}: {missing_count}개 결측치 대체")
                    
                except Exception as e:
                    self.logger.warning(f"결측치 대체 오류 ({col}): {e}")
        
        return imputed_data
    
    def reduce_noise(self, data: pd.DataFrame, 
                   method: str = 'adaptive') -> pd.DataFrame:
        """노이즈 감소"""
        denoised_data = data.copy()
        
        if method not in self.denoisers:
            self.logger.warning(f"지원되지 않는 노이즈 제거 방법: {method}")
            return denoised_data
        
        denoiser = self.denoisers[method]
        
        for col in denoised_data.select_dtypes(include=[np.number]).columns:
            try:
                original_data = denoised_data[col].values
                denoised_values = denoiser.reduce(original_data)
                denoised_data[col] = denoised_values
                
                # 노이즈 감소 효과 측정
                noise_reduction = np.std(original_data - denoised_values) / np.std(original_data)
                self.logger.info(f"{col}: 노이즈 {noise_reduction:.3f} 감소")
                
            except Exception as e:
                self.logger.warning(f"노이즈 감소 오류 ({col}): {e}")
        
        return denoised_data
    
    def extract_signals(self, data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """신호 추출"""
        signals = {}
        
        # 트렌드 추출
        trend_data = pd.DataFrame(index=data.index)
        for col in data.select_dtypes(include=[np.number]).columns:
            try:
                trend = self.signal_extractor.extract_trend(data[col].values)
                trend_data[col] = trend
            except Exception as e:
                self.logger.warning(f"트렌드 추출 오류 ({col}): {e}")
                trend_data[col] = data[col]
        
        signals['trend'] = trend_data
        
        # 주기성 추출
        cycles_data = pd.DataFrame(index=data.index)
        for col in data.select_dtypes(include=[np.number]).columns:
            try:
                cycles_result = self.signal_extractor.extract_cycles(data[col].values)
                cycles_data[col] = cycles_result['cycles']
            except Exception as e:
                self.logger.warning(f"주기성 추출 오류 ({col}): {e}")
                cycles_data[col] = 0
        
        signals['cycles'] = cycles_data
        
        # 잔여 성분 (원본 - 트렌드 - 주기성)
        residual_data = data.select_dtypes(include=[np.number]) - trend_data - cycles_data
        signals['residual'] = residual_data
        
        return signals
    
    def enhance_data_quality(self, data: pd.DataFrame, 
                           config: Dict = None) -> Tuple[pd.DataFrame, Dict]:
        """종합적 데이터 품질 향상"""
        self.logger.info("🚀 종합적 데이터 품질 향상 시작...")
        
        config = config or {
            'detect_outliers': True,
            'handle_outliers': True,
            'impute_missing': True,
            'reduce_noise': True,
            'extract_signals': False,
            'enhance_snr': False
        }
        
        enhanced_data = data.copy()
        enhancement_log = {'steps': [], 'metrics': {}}
        
        # 초기 품질 평가
        initial_quality = self.assess_data_quality(enhanced_data)
        enhancement_log['metrics']['initial'] = initial_quality
        
        # 1. 이상치 탐지 및 처리
        if config.get('detect_outliers', True):
            self.logger.info("🔍 이상치 탐지 중...")
            outliers = self.detect_outliers(enhanced_data)
            enhancement_log['steps'].append('outlier_detection')
            
            if config.get('handle_outliers', True):
                self.logger.info("🛠️ 이상치 처리 중...")
                enhanced_data = self.handle_outliers(enhanced_data, outliers)
                enhancement_log['steps'].append('outlier_handling')
        
        # 2. 결측치 대체
        if config.get('impute_missing', True):
            self.logger.info("🔄 결측치 대체 중...")
            enhanced_data = self.impute_missing_values(enhanced_data)
            enhancement_log['steps'].append('missing_value_imputation')
        
        # 3. 노이즈 감소
        if config.get('reduce_noise', True):
            self.logger.info("🔇 노이즈 감소 중...")
            enhanced_data = self.reduce_noise(enhanced_data)
            enhancement_log['steps'].append('noise_reduction')
        
        # 4. 신호 추출 (옵션)
        if config.get('extract_signals', False):
            self.logger.info("📡 신호 추출 중...")
            signals = self.extract_signals(enhanced_data)
            enhancement_log['signals'] = signals
            enhancement_log['steps'].append('signal_extraction')
        
        # 5. SNR 향상 (옵션)
        if config.get('enhance_snr', False):
            self.logger.info("📈 SNR 향상 중...")
            for col in enhanced_data.select_dtypes(include=[np.number]).columns:
                try:
                    enhanced_values = self.signal_extractor.enhance_signal_to_noise_ratio(
                        enhanced_data[col].values
                    )
                    enhanced_data[col] = enhanced_values
                except Exception as e:
                    self.logger.warning(f"SNR 향상 오류 ({col}): {e}")
            enhancement_log['steps'].append('snr_enhancement')
        
        # 최종 품질 평가
        final_quality = self.assess_data_quality(enhanced_data)
        enhancement_log['metrics']['final'] = final_quality
        
        # 개선 효과 계산
        improvement = final_quality.overall_score - initial_quality.overall_score
        enhancement_log['improvement'] = improvement
        
        self.logger.info(f"✅ 데이터 품질 향상 완료! 개선도: {improvement:.3f}")
        
        return enhanced_data, enhancement_log
    
    def generate_quality_report(self, 
                              original_data: pd.DataFrame,
                              enhanced_data: pd.DataFrame,
                              enhancement_log: Dict) -> str:
        """품질 향상 보고서 생성"""
        initial_metrics = enhancement_log['metrics']['initial']
        final_metrics = enhancement_log['metrics']['final']
        
        html_report = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>데이터 품질 향상 보고서</title>
            <meta charset="utf-8">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background: #2c3e50; color: white; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                .metric {{ display: inline-block; margin: 10px; padding: 10px; background: #f8f9fa; border-radius: 3px; }}
                .improvement {{ background: #d4edda; color: #155724; }}
                .degradation {{ background: #f8d7da; color: #721c24; }}
                table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>📈 데이터 품질 향상 보고서</h1>
                <p>생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="section">
                <h2>📊 품질 메트릭 비교</h2>
                <table>
                    <tr><th>메트릭</th><th>처리 전</th><th>처리 후</th><th>개선도</th></tr>
                    <tr>
                        <td>완전성</td>
                        <td>{initial_metrics.completeness:.3f}</td>
                        <td>{final_metrics.completeness:.3f}</td>
                        <td>{final_metrics.completeness - initial_metrics.completeness:.3f}</td>
                    </tr>
                    <tr>
                        <td>일관성</td>
                        <td>{initial_metrics.consistency:.3f}</td>
                        <td>{final_metrics.consistency:.3f}</td>
                        <td>{final_metrics.consistency - initial_metrics.consistency:.3f}</td>
                    </tr>
                    <tr>
                        <td>정확성</td>
                        <td>{initial_metrics.accuracy:.3f}</td>
                        <td>{final_metrics.accuracy:.3f}</td>
                        <td>{final_metrics.accuracy - initial_metrics.accuracy:.3f}</td>
                    </tr>
                    <tr>
                        <td>유효성</td>
                        <td>{initial_metrics.validity:.3f}</td>
                        <td>{final_metrics.validity:.3f}</td>
                        <td>{final_metrics.validity - initial_metrics.validity:.3f}</td>
                    </tr>
                    <tr>
                        <td>종합 점수</td>
                        <td>{initial_metrics.overall_score:.3f}</td>
                        <td>{final_metrics.overall_score:.3f}</td>
                        <td>{enhancement_log['improvement']:.3f}</td>
                    </tr>
                </table>
            </div>
            
            <div class="section">
                <h2>🔄 처리 단계</h2>
                <ul>
                    {''.join(f'<li>{step}</li>' for step in enhancement_log['steps'])}
                </ul>
            </div>
            
            <div class="section">
                <h2>📈 데이터 통계</h2>
                <table>
                    <tr><th>항목</th><th>처리 전</th><th>처리 후</th></tr>
                    <tr>
                        <td>행 수</td>
                        <td>{len(original_data):,}</td>
                        <td>{len(enhanced_data):,}</td>
                    </tr>
                    <tr>
                        <td>열 수</td>
                        <td>{len(original_data.columns)}</td>
                        <td>{len(enhanced_data.columns)}</td>
                    </tr>
                    <tr>
                        <td>결측치 수</td>
                        <td>{original_data.isnull().sum().sum():,}</td>
                        <td>{enhanced_data.isnull().sum().sum():,}</td>
                    </tr>
                </table>
            </div>
        </body>
        </html>
        """
        
        return html_report


def main():
    """메인 실행 함수"""
    print("📈 데이터 품질 향상 파이프라인 시작")
    
    # 파이프라인 초기화
    pipeline = DataQualityEnhancementPipeline()
    
    # 예제 데이터 생성 (품질 문제가 있는 데이터)
    print("\n📊 예제 데이터 생성...")
    np.random.seed(42)
    
    # 시계열 데이터 생성
    n_samples = 1000
    dates = pd.date_range('2024-01-01', periods=n_samples, freq='H')
    
    # 기본 신호 (트렌드 + 주기성 + 노이즈)
    trend = np.linspace(100, 120, n_samples)
    seasonal = 10 * np.sin(2 * np.pi * np.arange(n_samples) / 24)  # 일일 주기
    noise = np.random.normal(0, 2, n_samples)
    signal = trend + seasonal + noise
    
    # 품질 문제 추가
    # 1. 결측치
    missing_indices = np.random.choice(n_samples, size=50, replace=False)
    signal[missing_indices] = np.nan
    
    # 2. 이상치
    outlier_indices = np.random.choice(n_samples, size=20, replace=False)
    signal[outlier_indices] += np.random.normal(0, 20, 20)
    
    # 3. 추가 노이즈
    high_noise_indices = np.random.choice(n_samples, size=100, replace=False)
    signal[high_noise_indices] += np.random.normal(0, 5, 100)
    
    # 데이터프레임 생성
    data = pd.DataFrame({
        'price': signal,
        'volume': np.random.lognormal(10, 0.5, n_samples),
        'feature1': np.cumsum(np.random.randn(n_samples) * 0.1),
        'feature2': signal * 0.5 + np.random.randn(n_samples)
    }, index=dates)
    
    print(f"생성된 데이터: {data.shape}")
    print(f"결측치 수: {data.isnull().sum().sum()}")
    
    # 초기 품질 평가
    print("\n📏 초기 품질 평가...")
    initial_quality = pipeline.assess_data_quality(data)
    print(f"초기 품질 점수: {initial_quality.overall_score:.3f}")
    
    # 품질 향상 실행
    print("\n🚀 품질 향상 실행...")
    enhanced_data, enhancement_log = pipeline.enhance_data_quality(data, {
        'detect_outliers': True,
        'handle_outliers': True,
        'impute_missing': True,
        'reduce_noise': True,
        'extract_signals': False,
        'enhance_snr': True
    })
    
    # 결과 출력
    print(f"\n✅ 품질 향상 완료!")
    print(f"최종 품질 점수: {enhancement_log['metrics']['final'].overall_score:.3f}")
    print(f"개선도: {enhancement_log['improvement']:.3f}")
    print(f"처리된 결측치: {data.isnull().sum().sum() - enhanced_data.isnull().sum().sum()}")
    
    # 보고서 생성
    print("\n📋 품질 보고서 생성...")
    report_html = pipeline.generate_quality_report(data, enhanced_data, enhancement_log)
    
    with open("data_quality_enhancement_report.html", "w", encoding="utf-8") as f:
        f.write(report_html)
    
    print("\n✅ 데이터 품질 향상 파이프라인 완료!")
    print("📋 상세 보고서: data_quality_enhancement_report.html")


if __name__ == "__main__":
    main()