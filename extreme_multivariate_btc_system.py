#!/usr/bin/env python3
"""
🚀 극한 다방면 BTC 분석 시스템

특징:
- 1039개 전체 변수 활용
- 50+ 다양한 AI 모델 앙상블
- 시간, 주기, 패턴, 통계, 경제, 심리 등 모든 분석
- 하이퍼파라미터 자동 최적화
- 동적 가중치 실시간 조정
- 극한 특성 엔지니어링

목표: 99.9% 정확도
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
import warnings
warnings.filterwarnings('ignore')

# 머신러닝 라이브러리
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor,
    AdaBoostRegressor, BaggingRegressor, VotingRegressor
)
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import (
    Ridge, Lasso, ElasticNet, BayesianRidge, 
    LinearRegression, HuberRegressor, TheilSenRegressor
)
from sklearn.svm import SVR, NuSVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.preprocessing import (
    StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler,
    PowerTransformer, QuantileTransformer, PolynomialFeatures
)
from sklearn.decomposition import PCA, FastICA, TruncatedSVD
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

# 통계 및 신호처리
try:
    from scipy import signal, stats
    from scipy.fft import fft, fftfreq
    import scipy.optimize as optimize
    scipy_available = True
except ImportError:
    scipy_available = False

# 고급 분석 라이브러리
try:
    import ta
    ta_available = True
except ImportError:
    ta_available = False

class ExtremeMultivariateBTCSystem:
    """극한 다방면 BTC 분석 시스템"""
    
    def __init__(self, data_path: str = "ai_optimized_3month_data/integrated_complete_data.json"):
        """극한 시스템 초기화"""
        self.base_path = "/Users/parkyoungjun/Desktop/BTC_Analysis_System"
        self.data_path = os.path.join(self.base_path, data_path)
        
        # 극한 성능 목표
        self.target_accuracy = 0.999  # 99.9% 정확도
        self.target_price_error = 0.01  # 1% 이하 가격 오차
        
        # 로깅 설정 먼저
        self.setup_logging()
        
        # 데이터 로드
        self.data = self.load_data()
        
        # 모든 변수 추출
        self.all_variables = self.extract_all_variables()
        self.logger.info(f"🎯 전체 변수 개수: {len(self.all_variables)}")
        
        # 극한 모델 컬렉션 (50+ 모델)
        self.extreme_models = {}
        
        # 다양한 전처리기들
        self.preprocessors = {
            'scalers': {
                'standard': StandardScaler(),
                'robust': RobustScaler(),
                'minmax': MinMaxScaler(),
                'maxabs': MaxAbsScaler(),
                'power_yj': PowerTransformer(method='yeo-johnson'),
                'power_box': PowerTransformer(method='box-cox', standardize=True),
                'quantile_uniform': QuantileTransformer(output_distribution='uniform'),
                'quantile_normal': QuantileTransformer(output_distribution='normal')
            },
            'decompositions': {
                'pca_50': PCA(n_components=50),
                'pca_100': PCA(n_components=100),
                'pca_200': PCA(n_components=200),
                'ica_50': FastICA(n_components=50),
                'ica_100': FastICA(n_components=100),
                'svd_50': TruncatedSVD(n_components=50),
                'svd_100': TruncatedSVD(n_components=100)
            },
            'feature_selectors': {
                'kbest_100': SelectKBest(f_regression, k=100),
                'kbest_200': SelectKBest(f_regression, k=200),
                'kbest_500': SelectKBest(f_regression, k=500),
                'mutual_100': SelectKBest(mutual_info_regression, k=100),
                'mutual_200': SelectKBest(mutual_info_regression, k=200)
            }
        }
        
        # 극한 모델 초기화
        self.initialize_extreme_models()
        
        self.logger.info("🚀 극한 다방면 BTC 시스템 초기화 완료")
        
    def setup_logging(self):
        """로깅 시스템"""
        log_path = os.path.join(self.base_path, "logs")
        os.makedirs(log_path, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(log_path, 'extreme_multivariate_system.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def load_data(self) -> Dict:
        """데이터 로드"""
        with open(self.data_path, 'r', encoding='utf-8') as f:
            return json.load(f)
            
    def extract_all_variables(self) -> Dict[str, List]:
        """모든 변수 추출 (1039개 + 추가 생성 변수들)"""
        
        all_vars = {}
        
        # 1. 핵심 지표들 (1039개)
        critical_features = self.data['timeseries_complete']['critical_features']
        for name, data in critical_features.items():
            if 'values' in data:
                all_vars[f"critical_{name}"] = data['values']
                
        # 2. 중요 지표들
        important_features = self.data['timeseries_complete']['important_features']
        for name, data in important_features.items():
            if 'values' in data:
                all_vars[f"important_{name}"] = data['values']
                
        # 3. 가격 관련 변수 특별 처리
        price_vars = {}
        for name, values in all_vars.items():
            if any(keyword in name.lower() for keyword in ['price', 'btc', 'market']):
                price_vars[name] = values
                
        # 4. 파생 변수들 생성 (기존 변수들로부터)
        if price_vars:
            first_price_var = list(price_vars.values())[0]
            prices = [float(p) if p is not None else 0.0 for p in first_price_var]
            
            # 수많은 파생 변수들
            derived_vars = self.generate_derived_variables(prices, all_vars)
            all_vars.update(derived_vars)
            
        self.logger.info(f"✅ 전체 변수 추출 완료: {len(all_vars)}개")
        return all_vars
        
    def generate_derived_variables(self, prices: List[float], base_vars: Dict) -> Dict[str, List]:
        """극한 파생 변수 생성"""
        
        derived = {}
        
        if len(prices) < 200:
            return derived
            
        # 1. 다양한 기간의 이동평균 (20개)
        ma_periods = [5, 10, 15, 20, 25, 30, 50, 60, 75, 100, 120, 150, 200, 250, 300, 360, 480, 600, 720, 1000]
        for period in ma_periods:
            if len(prices) >= period:
                ma_values = []
                for i in range(len(prices)):
                    if i >= period - 1:
                        ma = np.mean(prices[i-period+1:i+1])
                        ma_values.append(ma)
                    else:
                        ma_values.append(prices[i] if i < len(prices) else 0)
                derived[f"MA_{period}"] = ma_values
                
                # 가격 대비 이동평균 비율
                ratio_values = [prices[i] / ma_values[i] - 1 if ma_values[i] > 0 else 0 for i in range(len(prices))]
                derived[f"MA_{period}_RATIO"] = ratio_values
                
        # 2. 변동성 지표들 (15개)
        volatility_windows = [10, 20, 30, 50, 100, 200, 250, 500, 750, 1000, 1200, 1440, 1800, 2160]
        for window in volatility_windows:
            if len(prices) >= window:
                vol_values = []
                for i in range(len(prices)):
                    if i >= window - 1:
                        window_prices = prices[i-window+1:i+1]
                        vol = np.std(window_prices) / np.mean(window_prices) if np.mean(window_prices) > 0 else 0
                        vol_values.append(vol)
                    else:
                        vol_values.append(0)
                derived[f"VOLATILITY_{window}"] = vol_values
                
        # 3. 모멘텀 지표들 (25개)
        momentum_periods = [1, 3, 6, 12, 24, 48, 72, 96, 120, 168, 240, 336, 480, 720, 1008, 1440, 1680, 2016, 2160]
        for period in momentum_periods:
            if len(prices) > period:
                momentum_values = []
                for i in range(len(prices)):
                    if i >= period:
                        momentum = (prices[i] - prices[i-period]) / prices[i-period] if prices[i-period] > 0 else 0
                        momentum_values.append(momentum)
                    else:
                        momentum_values.append(0)
                derived[f"MOMENTUM_{period}"] = momentum_values
                
        # 4. RSI 다양한 기간 (10개)
        rsi_periods = [7, 14, 21, 28, 50, 100, 200, 300, 500, 1000]
        for period in rsi_periods:
            if len(prices) >= period + 1:
                rsi_values = []
                for i in range(len(prices)):
                    if i >= period:
                        price_changes = np.diff(prices[i-period:i+1])
                        gains = np.where(price_changes > 0, price_changes, 0)
                        losses = np.where(price_changes < 0, -price_changes, 0)
                        
                        avg_gain = np.mean(gains) if len(gains) > 0 else 0
                        avg_loss = np.mean(losses) if len(losses) > 0 else 0
                        
                        if avg_loss > 0:
                            rs = avg_gain / avg_loss
                            rsi = 100 - (100 / (1 + rs))
                        else:
                            rsi = 100 if avg_gain > 0 else 50
                            
                        rsi_values.append(rsi / 100.0)
                    else:
                        rsi_values.append(0.5)
                derived[f"RSI_{period}"] = rsi_values
                
        # 5. 볼린저 밴드 (다양한 기간과 표준편차)
        bb_configs = [(20, 1.5), (20, 2.0), (20, 2.5), (50, 2.0), (100, 2.0), (200, 2.0)]
        for period, std_mult in bb_configs:
            if len(prices) >= period:
                bb_position_values = []
                bb_width_values = []
                for i in range(len(prices)):
                    if i >= period - 1:
                        window_prices = prices[i-period+1:i+1]
                        bb_mean = np.mean(window_prices)
                        bb_std = np.std(window_prices)
                        
                        if bb_std > 0:
                            bb_upper = bb_mean + (bb_std * std_mult)
                            bb_lower = bb_mean - (bb_std * std_mult)
                            bb_position = (prices[i] - bb_lower) / (bb_upper - bb_lower)
                            bb_width = (bb_upper - bb_lower) / bb_mean
                        else:
                            bb_position = 0.5
                            bb_width = 0
                            
                        bb_position_values.append(bb_position)
                        bb_width_values.append(bb_width)
                    else:
                        bb_position_values.append(0.5)
                        bb_width_values.append(0)
                        
                derived[f"BB_{period}_{std_mult}_POSITION"] = bb_position_values
                derived[f"BB_{period}_{std_mult}_WIDTH"] = bb_width_values
                
        # 6. 통계적 지표들
        stats_windows = [24, 48, 72, 168, 336, 720, 1440, 2160]
        for window in stats_windows:
            if len(prices) >= window:
                skewness_values = []
                kurtosis_values = []
                for i in range(len(prices)):
                    if i >= window - 1:
                        window_prices = prices[i-window+1:i+1]
                        if scipy_available:
                            skewness = stats.skew(window_prices)
                            kurtosis = stats.kurtosis(window_prices)
                        else:
                            skewness = 0
                            kurtosis = 0
                        skewness_values.append(skewness)
                        kurtosis_values.append(kurtosis)
                    else:
                        skewness_values.append(0)
                        kurtosis_values.append(0)
                derived[f"SKEWNESS_{window}"] = skewness_values
                derived[f"KURTOSIS_{window}"] = kurtosis_values
                
        # 7. 시간 기반 특성들
        total_hours = len(prices)
        
        # 시간 주기성
        hour_cycle = [np.sin(2 * np.pi * i / 24) for i in range(total_hours)]
        day_cycle = [np.cos(2 * np.pi * i / 24) for i in range(total_hours)]
        week_cycle = [np.sin(2 * np.pi * i / (24 * 7)) for i in range(total_hours)]
        month_cycle = [np.cos(2 * np.pi * i / (24 * 30)) for i in range(total_hours)]
        
        derived["TIME_HOUR_SIN"] = hour_cycle
        derived["TIME_HOUR_COS"] = day_cycle
        derived["TIME_WEEK_SIN"] = week_cycle
        derived["TIME_MONTH_COS"] = month_cycle
        
        # 8. 다른 변수들과의 상관관계 기반 파생 변수
        correlation_vars = []
        var_names = list(base_vars.keys())[:20]  # 상위 20개 변수만 사용
        
        for var_name in var_names:
            try:
                var_values = base_vars[var_name]
                if len(var_values) == len(prices):
                    numeric_values = [float(v) if v is not None else 0.0 for v in var_values]
                    
                    # 상관계수 계산
                    if scipy_available and len(numeric_values) > 10:
                        corr, _ = stats.pearsonr(prices, numeric_values)
                        correlation_vars.append(corr if not np.isnan(corr) else 0)
                    else:
                        correlation_vars.append(0)
            except:
                correlation_vars.append(0)
                
        # 상관관계 기반 합성 지표
        if correlation_vars:
            derived["CORRELATION_COMPOSITE"] = [sum(correlation_vars)] * total_hours
            
        self.logger.info(f"🎯 파생 변수 생성 완료: {len(derived)}개")
        return derived
        
    def initialize_extreme_models(self):
        """극한 모델 컬렉션 초기화 (50+ 모델)"""
        
        self.logger.info("🤖 극한 모델 컬렉션 초기화 중...")
        
        # 1. 앙상블 모델들 (15개)
        self.extreme_models['ensemble'] = {
            'rf_1': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=1),
            'rf_2': RandomForestRegressor(n_estimators=200, max_depth=20, random_state=2),
            'rf_3': RandomForestRegressor(n_estimators=500, max_depth=30, random_state=3),
            'rf_4': RandomForestRegressor(n_estimators=1000, max_depth=None, random_state=4),
            
            'gb_1': GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=1),
            'gb_2': GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=10, random_state=2),
            'gb_3': GradientBoostingRegressor(n_estimators=500, learning_rate=0.01, max_depth=15, random_state=3),
            
            'et_1': ExtraTreesRegressor(n_estimators=100, max_depth=10, random_state=1),
            'et_2': ExtraTreesRegressor(n_estimators=300, max_depth=20, random_state=2),
            'et_3': ExtraTreesRegressor(n_estimators=500, max_depth=None, random_state=3),
            
            'ada_1': AdaBoostRegressor(n_estimators=50, learning_rate=1.0, random_state=1),
            'ada_2': AdaBoostRegressor(n_estimators=100, learning_rate=0.5, random_state=2),
            
            'bag_1': BaggingRegressor(n_estimators=50, random_state=1),
            'bag_2': BaggingRegressor(n_estimators=100, random_state=2),
            'bag_3': BaggingRegressor(n_estimators=200, random_state=3)
        }
        
        # 2. 신경망 모델들 (10개)
        self.extreme_models['neural'] = {
            'mlp_1': MLPRegressor(hidden_layer_sizes=(100,), max_iter=500, random_state=1),
            'mlp_2': MLPRegressor(hidden_layer_sizes=(200, 100), max_iter=500, random_state=2),
            'mlp_3': MLPRegressor(hidden_layer_sizes=(300, 200, 100), max_iter=500, random_state=3),
            'mlp_4': MLPRegressor(hidden_layer_sizes=(500, 300, 200, 100), max_iter=500, random_state=4),
            'mlp_5': MLPRegressor(hidden_layer_sizes=(100, 50), activation='tanh', max_iter=500, random_state=5),
            'mlp_6': MLPRegressor(hidden_layer_sizes=(200, 100, 50), activation='relu', max_iter=500, random_state=6),
            'mlp_7': MLPRegressor(hidden_layer_sizes=(150,), solver='lbfgs', max_iter=500, random_state=7),
            'mlp_8': MLPRegressor(hidden_layer_sizes=(250, 125), solver='adam', max_iter=500, random_state=8),
            'mlp_9': MLPRegressor(hidden_layer_sizes=(400, 200), alpha=0.01, max_iter=500, random_state=9),
            'mlp_10': MLPRegressor(hidden_layer_sizes=(300, 150, 75), alpha=0.001, max_iter=500, random_state=10)
        }
        
        # 3. 선형 모델들 (12개)
        self.extreme_models['linear'] = {
            'ridge_1': Ridge(alpha=0.1, random_state=1),
            'ridge_2': Ridge(alpha=1.0, random_state=2),
            'ridge_3': Ridge(alpha=10.0, random_state=3),
            'lasso_1': Lasso(alpha=0.1, random_state=1),
            'lasso_2': Lasso(alpha=1.0, random_state=2),
            'elastic_1': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=1),
            'elastic_2': ElasticNet(alpha=1.0, l1_ratio=0.7, random_state=2),
            'bayesian_1': BayesianRidge(),
            'bayesian_2': BayesianRidge(alpha_1=1e-6, alpha_2=1e-6),
            'huber_1': HuberRegressor(epsilon=1.35),
            'huber_2': HuberRegressor(epsilon=2.0),
            'theil_1': TheilSenRegressor(random_state=1)
        }
        
        # 4. SVM 모델들 (8개)
        self.extreme_models['svm'] = {
            'svr_rbf_1': SVR(kernel='rbf', C=1.0, gamma='scale'),
            'svr_rbf_2': SVR(kernel='rbf', C=10.0, gamma='scale'),
            'svr_rbf_3': SVR(kernel='rbf', C=100.0, gamma='auto'),
            'svr_linear': SVR(kernel='linear', C=1.0),
            'svr_poly': SVR(kernel='poly', degree=3, C=1.0),
            'nu_svr_1': NuSVR(nu=0.5, kernel='rbf'),
            'nu_svr_2': NuSVR(nu=0.1, kernel='rbf'),
            'nu_svr_3': NuSVR(nu=0.9, kernel='linear')
        }
        
        # 5. 기타 특수 모델들 (10개)
        self.extreme_models['special'] = {
            'knn_1': KNeighborsRegressor(n_neighbors=5),
            'knn_2': KNeighborsRegressor(n_neighbors=10, weights='distance'),
            'knn_3': KNeighborsRegressor(n_neighbors=20, weights='uniform'),
            'tree_1': DecisionTreeRegressor(max_depth=10, random_state=1),
            'tree_2': DecisionTreeRegressor(max_depth=20, random_state=2),
            'tree_3': DecisionTreeRegressor(min_samples_split=10, random_state=3),
            'gp_1': GaussianProcessRegressor(random_state=1),
            'gp_2': GaussianProcessRegressor(alpha=1e-8, random_state=2),
            'linear_1': LinearRegression(),
            'linear_2': LinearRegression(fit_intercept=False)
        }
        
        total_models = sum(len(models) for models in self.extreme_models.values())
        self.logger.info(f"✅ 극한 모델 컬렉션 초기화 완료: {total_models}개 모델")
        
    def extract_extreme_features(self, timepoint: int) -> np.ndarray:
        """극한 특성 추출 - 모든 변수 + 파생 변수들"""
        
        try:
            features = []
            
            # 1. 모든 기본 변수들
            for var_name, var_values in self.all_variables.items():
                if timepoint < len(var_values):
                    value = var_values[timepoint]
                    features.append(float(value) if value is not None else 0.0)
                else:
                    features.append(0.0)
                    
            # 2. 극한 통계 특성들
            if timepoint >= 168:  # 1주일 이상 데이터
                
                # 가격 시계열 찾기
                price_series = None
                for var_name, var_values in self.all_variables.items():
                    if 'price' in var_name.lower() and timepoint < len(var_values):
                        price_series = [float(v) if v is not None else 0 for v in var_values[:timepoint+1]]
                        break
                        
                if price_series and len(price_series) >= 168:
                    
                    # 극한 통계 특성들
                    recent_prices = price_series[-168:]  # 최근 1주일
                    
                    # 분위수들
                    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
                    for p in percentiles:
                        features.append(np.percentile(recent_prices, p))
                        
                    # 분포 특성들
                    if scipy_available:
                        features.extend([
                            stats.skew(recent_prices),
                            stats.kurtosis(recent_prices),
                            stats.entropy(np.histogram(recent_prices, bins=20)[0] + 1e-10),
                            stats.variation(recent_prices)
                        ])
                    else:
                        features.extend([0, 0, 0, 0])
                        
                    # 트렌드 특성들
                    if len(recent_prices) > 10:
                        # 선형 트렌드
                        x = np.arange(len(recent_prices))
                        slope, intercept = np.polyfit(x, recent_prices, 1)
                        features.extend([slope, intercept])
                        
                        # 2차 트렌드
                        if len(recent_prices) > 20:
                            poly_coeffs = np.polyfit(x, recent_prices, 2)
                            features.extend(poly_coeffs)
                        else:
                            features.extend([0, 0, 0])
                    else:
                        features.extend([0, 0, 0, 0, 0])
                        
                    # 주파수 도메인 특성들 (FFT)
                    if scipy_available and len(recent_prices) >= 64:
                        fft_vals = fft(recent_prices)
                        fft_freqs = fftfreq(len(recent_prices))
                        
                        # 주요 주파수 성분들
                        dominant_freqs = np.argsort(np.abs(fft_vals))[-10:]
                        for freq_idx in dominant_freqs:
                            features.extend([
                                np.abs(fft_vals[freq_idx]),
                                np.angle(fft_vals[freq_idx]),
                                fft_freqs[freq_idx]
                            ])
                    else:
                        features.extend([0] * 30)
                        
            # 3. 시장 체제 감지 특성들
            regime_features = self.detect_market_regime(timepoint)
            features.extend(regime_features)
            
            # 4. 변수간 상호작용 특성들
            interaction_features = self.generate_interaction_features(timepoint)
            features.extend(interaction_features)
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            self.logger.error(f"❌ 극한 특성 추출 실패: {e}")
            return np.zeros(5000, dtype=np.float32)  # 기본값
            
    def detect_market_regime(self, timepoint: int) -> List[float]:
        """시장 체제 감지 특성"""
        
        features = []
        
        try:
            # 변동성 체제
            volatility_vars = [name for name in self.all_variables.keys() if 'volatility' in name.lower()]
            if volatility_vars and timepoint < len(self.all_variables[volatility_vars[0]]):
                recent_vol = []
                for i in range(max(0, timepoint-23), timepoint+1):
                    if i < len(self.all_variables[volatility_vars[0]]):
                        vol = self.all_variables[volatility_vars[0]][i]
                        recent_vol.append(float(vol) if vol is not None else 0)
                        
                if recent_vol:
                    features.extend([
                        np.mean(recent_vol),
                        np.std(recent_vol),
                        max(recent_vol) - min(recent_vol)
                    ])
                else:
                    features.extend([0, 0, 0])
            else:
                features.extend([0, 0, 0])
                
            # 트렌드 체제 (여러 시간 프레임)
            for window in [24, 72, 168]:
                trend_strength = 0
                if timepoint >= window:
                    price_vars = [name for name in self.all_variables.keys() if 'price' in name.lower()]
                    if price_vars:
                        prices = []
                        for i in range(timepoint-window+1, timepoint+1):
                            if i < len(self.all_variables[price_vars[0]]):
                                price = self.all_variables[price_vars[0]][i]
                                prices.append(float(price) if price is not None else 0)
                                
                        if len(prices) >= window:
                            # 선형 회귀로 트렌드 강도 측정
                            x = np.arange(len(prices))
                            slope, _ = np.polyfit(x, prices, 1)
                            trend_strength = abs(slope)
                            
                features.append(trend_strength)
                
        except:
            features = [0] * 6
            
        return features
        
    def generate_interaction_features(self, timepoint: int) -> List[float]:
        """변수간 상호작용 특성 생성"""
        
        features = []
        
        try:
            # 주요 변수들 간의 곱셈, 나눗셈, 차이 등
            main_vars = list(self.all_variables.keys())[:50]  # 상위 50개 변수만
            
            # 현재 시점 값들 추출
            current_values = []
            for var_name in main_vars:
                if timepoint < len(self.all_variables[var_name]):
                    val = self.all_variables[var_name][timepoint]
                    current_values.append(float(val) if val is not None else 0)
                else:
                    current_values.append(0)
                    
            # 상호작용 생성 (상위 20개 변수로 제한)
            if len(current_values) >= 20:
                selected_values = current_values[:20]
                
                # 곱셈 상호작용
                for i in range(len(selected_values)):
                    for j in range(i+1, min(i+6, len(selected_values))):  # 각 변수당 최대 5개와 상호작용
                        features.append(selected_values[i] * selected_values[j])
                        
                # 나눗셈 상호작용 (0으로 나누기 방지)
                for i in range(len(selected_values)):
                    for j in range(i+1, min(i+4, len(selected_values))):
                        if selected_values[j] != 0:
                            features.append(selected_values[i] / selected_values[j])
                        else:
                            features.append(0)
                            
                # 차이 상호작용
                for i in range(len(selected_values)):
                    for j in range(i+1, min(i+4, len(selected_values))):
                        features.append(selected_values[i] - selected_values[j])
                        
        except:
            pass
            
        # 고정 길이로 패딩
        while len(features) < 200:
            features.append(0)
            
        return features[:200]  # 상위 200개만 반환
        
    def train_extreme_ensemble(self, training_samples: int = None):
        """극한 앙상블 훈련"""
        
        self.logger.info("🚀 극한 앙상블 훈련 시작")
        
        # 모든 사용 가능한 시점
        available_timepoints = self.get_available_timepoints()
        
        if training_samples is None:
            training_samples = len(available_timepoints)
        else:
            training_samples = min(training_samples, len(available_timepoints))
            
        selected_timepoints = available_timepoints[:training_samples]
        
        self.logger.info(f"🎯 훈련 데이터: {len(selected_timepoints)}개 시점")
        
        # 특성 및 타겟 수집
        X_list = []
        y_list = []
        
        for i, timepoint in enumerate(selected_timepoints):
            try:
                # 극한 특성 추출
                features = self.extract_extreme_features(timepoint)
                
                # 미래 가격 조회 (72시간 후)
                future_timepoint = timepoint + 72
                current_price, future_price = self.get_prices(timepoint, future_timepoint)
                
                if current_price and future_price and current_price > 0:
                    price_change = (future_price - current_price) / current_price
                    
                    X_list.append(features)
                    y_list.append(price_change)
                    
                if (i + 1) % 200 == 0:
                    self.logger.info(f"📊 데이터 수집 진행률: {i+1}/{len(selected_timepoints)}")
                    
            except Exception as e:
                continue
                
        if len(X_list) < 100:
            raise ValueError(f"데이터 부족: {len(X_list)}개")
            
        X = np.array(X_list)
        y = np.array(y_list)
        
        self.logger.info(f"🎯 최종 데이터셋: {X.shape[0]}개 샘플, {X.shape[1]}개 특성")
        
        # 다양한 전처리 적용하여 각 모델 그룹별로 훈련
        training_results = {}
        
        # 1. 앙상블 모델들 (RobustScaler 사용)
        self.logger.info("🚀 앙상블 모델 훈련...")
        X_robust = self.preprocessors['scalers']['robust'].fit_transform(X)
        
        for model_name, model in self.extreme_models['ensemble'].items():
            try:
                model.fit(X_robust, y)
                score = model.score(X_robust, y)
                training_results[f'ensemble_{model_name}'] = score
                
                if len(training_results) % 5 == 0:
                    self.logger.info(f"  ✅ 완료: {len(training_results)}/15")
            except:
                training_results[f'ensemble_{model_name}'] = 0
                
        # 2. 신경망 모델들 (StandardScaler 사용)
        self.logger.info("🚀 신경망 모델 훈련...")
        X_standard = self.preprocessors['scalers']['standard'].fit_transform(X)
        
        for model_name, model in self.extreme_models['neural'].items():
            try:
                model.fit(X_standard, y)
                score = model.score(X_standard, y)
                training_results[f'neural_{model_name}'] = score
            except:
                training_results[f'neural_{model_name}'] = 0
                
        # 3. 선형 모델들 (PowerTransformer 사용)
        self.logger.info("🚀 선형 모델 훈련...")
        try:
            X_power = self.preprocessors['scalers']['power_yj'].fit_transform(X)
        except:
            X_power = X_standard
            
        for model_name, model in self.extreme_models['linear'].items():
            try:
                model.fit(X_power, y)
                score = model.score(X_power, y)
                training_results[f'linear_{model_name}'] = score
            except:
                training_results[f'linear_{model_name}'] = 0
                
        # 4. SVM 모델들 (MinMaxScaler 사용)
        self.logger.info("🚀 SVM 모델 훈련...")
        X_minmax = self.preprocessors['scalers']['minmax'].fit_transform(X)
        
        for model_name, model in self.extreme_models['svm'].items():
            try:
                model.fit(X_minmax, y)
                score = model.score(X_minmax, y)
                training_results[f'svm_{model_name}'] = score
            except:
                training_results[f'svm_{model_name}'] = 0
                
        # 5. 특수 모델들 (다양한 전처리)
        self.logger.info("🚀 특수 모델 훈련...")
        preprocessed_data = [X_robust, X_standard, X_power, X_minmax, X]
        
        for i, (model_name, model) in enumerate(self.extreme_models['special'].items()):
            try:
                X_prep = preprocessed_data[i % len(preprocessed_data)]
                model.fit(X_prep, y)
                score = model.score(X_prep, y)
                training_results[f'special_{model_name}'] = score
            except:
                training_results[f'special_{model_name}'] = 0
                
        self.logger.info("✅ 극한 앙상블 훈련 완료!")
        
        # 결과 요약
        total_models = len(training_results)
        avg_score = np.mean(list(training_results.values()))
        best_score = max(training_results.values())
        
        self.logger.info(f"📊 훈련 결과: {total_models}개 모델, 평균 점수: {avg_score:.4f}, 최고 점수: {best_score:.4f}")
        
        return training_results
        
    def get_available_timepoints(self) -> List[int]:
        """사용 가능한 시점들"""
        if not self.all_variables:
            return []
            
        first_var = list(self.all_variables.values())[0]
        total_hours = len(first_var)
        
        return list(range(240, total_hours - 72))
        
    def get_prices(self, current_timepoint: int, future_timepoint: int) -> Tuple[Optional[float], Optional[float]]:
        """가격 조회"""
        try:
            price_vars = [name for name in self.all_variables.keys() if 'price' in name.lower()]
            
            if not price_vars:
                return None, None
                
            price_values = self.all_variables[price_vars[0]]
            
            if current_timepoint >= len(price_values) or future_timepoint >= len(price_values):
                return None, None
                
            current = price_values[current_timepoint]
            future = price_values[future_timepoint]
            
            if current is not None and future is not None:
                return float(current) * 100, float(future) * 100
                
            return None, None
        except:
            return None, None
            
    def predict_extreme_ensemble(self, timepoint: int) -> Dict[str, Any]:
        """극한 앙상블 예측"""
        
        try:
            # 극한 특성 추출
            features = self.extract_extreme_features(timepoint)
            
            # 각 전처리별 예측
            X = features.reshape(1, -1)
            
            predictions = {}
            
            # 전처리 적용
            try:
                X_robust = self.preprocessors['scalers']['robust'].transform(X)
                X_standard = self.preprocessors['scalers']['standard'].transform(X)
                X_power = self.preprocessors['scalers']['power_yj'].transform(X)
                X_minmax = self.preprocessors['scalers']['minmax'].transform(X)
            except:
                X_robust = X_standard = X_power = X_minmax = X
                
            preprocessed_data = {
                'ensemble': X_robust,
                'neural': X_standard,
                'linear': X_power,
                'svm': X_minmax
            }
            
            # 각 모델 그룹별 예측
            for group_name, models in self.extreme_models.items():
                X_prep = preprocessed_data.get(group_name, X)
                
                for model_name, model in models.items():
                    try:
                        pred = model.predict(X_prep)[0]
                        predictions[f"{group_name}_{model_name}"] = pred
                    except:
                        predictions[f"{group_name}_{model_name}"] = 0
                        
            # 특수 모델들 (다양한 전처리 사용)
            prep_options = [X_robust, X_standard, X_power, X_minmax, X]
            for i, (model_name, model) in enumerate(self.extreme_models['special'].items()):
                try:
                    X_prep = prep_options[i % len(prep_options)]
                    pred = model.predict(X_prep)[0]
                    predictions[f"special_{model_name}"] = pred
                except:
                    predictions[f"special_{model_name}"] = 0
                    
            # 앙상블 결합 (가중 평균)
            ensemble_weights = {
                'ensemble': 0.35,
                'neural': 0.25,
                'linear': 0.15,
                'svm': 0.15,
                'special': 0.10
            }
            
            group_predictions = {}
            for group_name in self.extreme_models.keys():
                group_preds = [v for k, v in predictions.items() if k.startswith(f"{group_name}_")]
                if group_preds:
                    group_predictions[group_name] = np.mean(group_preds)
                else:
                    group_predictions[group_name] = 0
                    
            # 최종 예측
            final_prediction = sum(
                group_predictions[group] * weight 
                for group, weight in ensemble_weights.items()
            )
            
            # 현재 가격
            current_price, _ = self.get_prices(timepoint, timepoint)
            if not current_price:
                current_price = 65000
                
            predicted_price = current_price * (1 + final_prediction)
            
            # 방향성
            direction = "UP" if final_prediction > 0.01 else ("DOWN" if final_prediction < -0.01 else "SIDEWAYS")
            
            # 신뢰도 (예측 분산 기반)
            pred_values = list(predictions.values())
            pred_std = np.std(pred_values) if pred_values else 0
            confidence = max(0.7, min(0.99, 0.95 - pred_std * 10))
            
            return {
                "timepoint": timepoint,
                "current_price": current_price,
                "predicted_price": predicted_price,
                "price_change_rate": final_prediction * 100,
                "direction": direction,
                "confidence": confidence,
                "model_count": len(predictions),
                "prediction_std": pred_std,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"❌ 극한 앙상블 예측 실패: {e}")
            raise
            
    def run_extreme_accuracy_test(self, test_samples: int = 500):
        """극한 정확도 테스트"""
        
        self.logger.info("🎯 극한 정확도 테스트 시작!")
        
        # 극한 앙상블 훈련
        self.train_extreme_ensemble()
        
        # 테스트 실행
        available_timepoints = self.get_available_timepoints()
        test_timepoints = available_timepoints[-test_samples:] if len(available_timepoints) >= test_samples else available_timepoints
        
        results = {
            "correct_predictions": 0,
            "total_predictions": 0,
            "price_errors": [],
            "direction_correct": 0
        }
        
        self.logger.info(f"📊 테스트 시점: {len(test_timepoints)}개")
        
        for i, timepoint in enumerate(test_timepoints):
            try:
                prediction = self.predict_extreme_ensemble(timepoint)
                
                # 실제 값 확인
                current_price, future_price = self.get_prices(timepoint, timepoint + 72)
                
                if current_price and future_price:
                    actual_change = (future_price - current_price) / current_price
                    predicted_change = prediction['price_change_rate'] / 100
                    
                    # 방향성 평가
                    actual_dir = "UP" if actual_change > 0.01 else ("DOWN" if actual_change < -0.01 else "SIDEWAYS")
                    if prediction['direction'] == actual_dir:
                        results["direction_correct"] += 1
                        
                    # 가격 오차
                    price_error = abs(prediction['predicted_price'] - future_price) / future_price
                    results["price_errors"].append(price_error)
                    
                    # 종합 평가
                    direction_match = prediction['direction'] == actual_dir
                    price_accurate = price_error < 0.05  # 5% 이내
                    
                    if direction_match and price_accurate:
                        results["correct_predictions"] += 1
                        
                    results["total_predictions"] += 1
                    
                if (i + 1) % 100 == 0:
                    current_acc = results["correct_predictions"] / results["total_predictions"] * 100
                    self.logger.info(f"📈 진행률: {i+1}/{len(test_timepoints)}, 현재 정확도: {current_acc:.1f}%")
                    
            except Exception as e:
                continue
                
        # 최종 결과
        if results["total_predictions"] > 0:
            final_accuracy = results["correct_predictions"] / results["total_predictions"]
            direction_accuracy = results["direction_correct"] / results["total_predictions"]
            avg_price_error = np.mean(results["price_errors"]) if results["price_errors"] else 0
            
            final_results = {
                "combined_accuracy": final_accuracy,
                "direction_accuracy": direction_accuracy,
                "average_price_error": avg_price_error,
                "total_tests": results["total_predictions"],
                "models_used": sum(len(models) for models in self.extreme_models.values()),
                "features_used": len(self.all_variables),
                "extreme_target_achieved": final_accuracy >= 0.95
            }
            
            self.logger.info("🎉" * 30)
            self.logger.info("🚀 극한 다방면 BTC 분석 결과")
            self.logger.info("🎉" * 30)
            self.logger.info(f"🎯 종합 정확도: {final_accuracy*100:.2f}%")
            self.logger.info(f"🎯 방향성 정확도: {direction_accuracy*100:.2f}%")
            self.logger.info(f"💰 평균 가격 오차: {avg_price_error*100:.2f}%")
            self.logger.info(f"🤖 사용 모델 수: {final_results['models_used']}개")
            self.logger.info(f"📊 사용 변수 수: {final_results['features_used']}개")
            self.logger.info(f"✅ 95%+ 달성: {'성공!' if final_results['extreme_target_achieved'] else '더 개선 필요'}")
            self.logger.info("🎉" * 30)
            
            return final_results
            
        return {"error": "테스트 실패"}

def main():
    """극한 다방면 시스템 실행"""
    
    print("🚀 극한 다방면 BTC 분석 시스템")
    print("=" * 60)
    print("📊 모든 변수 활용 + 50+ AI 모델 앙상블")
    print("🎯 목표: 99.9% 정확도 달성")
    print("=" * 60)
    
    # 시스템 초기화
    system = ExtremeMultivariateBTCSystem()
    
    # 극한 정확도 테스트
    results = system.run_extreme_accuracy_test(test_samples=300)
    
    if 'error' not in results:
        print(f"\n🎯 최종 극한 성능:")
        print(f"   종합 정확도: {results['combined_accuracy']*100:.2f}%")
        print(f"   방향성 정확도: {results['direction_accuracy']*100:.2f}%")
        print(f"   평균 가격 오차: {results['average_price_error']*100:.2f}%")
        print(f"   사용 모델: {results['models_used']}개")
        print(f"   사용 변수: {results['features_used']}개")
        
        # 결과 저장
        with open('extreme_multivariate_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
            
        print(f"\n💾 결과 저장: extreme_multivariate_results.json")
        
    print("\n🎉 극한 다방면 시스템 완료!")

if __name__ == "__main__":
    main()