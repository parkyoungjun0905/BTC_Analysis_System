#!/usr/bin/env python3
"""
🎯 시장 상황별 분석 모듈 (Market Regime Analyzer)
- 다차원 시장 상황 식별 (강세/약세/횡보장 × 높은/낮은 변동성)
- 기술적 지표 기반 상황 분류 (추세, 모멘텀, 변동성)
- 머신러닝 기반 상황 예측 (HMM, Clustering)
- 상황 전환점 감지 및 조기 경고
- 상황별 최적 전략 추천
- 스트레스 테스트 시나리오 생성
- 위기 상황 분석 및 대응 전략
"""

import numpy as np
import pandas as pd
import warnings
from datetime import datetime, timedelta
import json
import os
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
import logging
from enum import Enum

# ML 및 통계 라이브러리
from sklearn.cluster import KMeans, GaussianMixture
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy import stats
from scipy.signal import find_peaks

# HMM (Hidden Markov Model)
try:
    from hmmlearn import hmm
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    print("⚠️ hmmlearn 미설치: HMM 기반 분석 불가")

# 기술적 분석
try:
    import ta
    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False
    print("⚠️ ta 미설치: 일부 기술적 지표 사용 불가")

# 시각화
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

warnings.filterwarnings('ignore')

class MarketRegime(Enum):
    """시장 상황 열거형"""
    BULL_LOW_VOL = "강세장_낮은변동성"
    BULL_HIGH_VOL = "강세장_높은변동성"
    BEAR_LOW_VOL = "약세장_낮은변동성"
    BEAR_HIGH_VOL = "약세장_높은변동성"
    SIDEWAYS_LOW_VOL = "횡보장_낮은변동성"
    SIDEWAYS_HIGH_VOL = "횡보장_높은변동성"
    CRISIS = "위기상황"
    RECOVERY = "회복상황"

@dataclass
class RegimeConfig:
    """시장 상황 분석 설정"""
    # 기본 설정
    lookback_periods: Dict[str, int] = field(default_factory=lambda: {
        'short': 20,    # 단기 (20일)
        'medium': 60,   # 중기 (2개월)  
        'long': 252     # 장기 (1년)
    })
    
    # 변동성 임계값
    volatility_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'low': 0.3,      # 연간 30% 이하
        'medium': 0.6,   # 연간 30-60%
        'high': 0.6      # 연간 60% 이상
    })
    
    # 추세 임계값
    trend_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'bull': 0.15,    # 15% 이상 상승
        'bear': -0.15,   # 15% 이상 하락
        'sideways': 0.15 # ±15% 내 횡보
    })
    
    # 클러스터링 설정
    n_clusters: int = 6          # 클러스터 개수
    cluster_features: List[str] = field(default_factory=lambda: [
        'returns', 'volatility', 'momentum', 'volume', 'rsi', 'macd'
    ])
    
    # HMM 설정
    n_hidden_states: int = 4     # 은닉 상태 개수
    hmm_covariance_type: str = "full"  # HMM 공분산 타입
    
    # 위기 감지 설정
    crisis_threshold: float = -0.20  # 20% 이상 하락시 위기
    recovery_threshold: float = 0.10 # 10% 이상 회복시 회복
    
    # 스트레스 테스트
    stress_scenarios: List[str] = field(default_factory=lambda: [
        'black_swan', 'market_crash', 'liquidity_crisis', 'regulatory_shock'
    ])

@dataclass
class RegimePeriod:
    """시장 상황 기간 정보"""
    regime: MarketRegime
    start_date: datetime
    end_date: datetime
    duration_days: int
    characteristics: Dict
    performance_metrics: Dict
    confidence_score: float

class MarketRegimeAnalyzer:
    """시장 상황별 분석 시스템"""
    
    def __init__(self, config: RegimeConfig = None):
        self.config = config or RegimeConfig()
        self.data_path = "/Users/parkyoungjun/Desktop/BTC_Analysis_System"
        
        # 분석 결과 저장
        self.regime_history = []
        self.current_regime = None
        self.regime_transition_matrix = None
        self.feature_importance = {}
        self.models = {}
        
        # 로깅 설정
        self.setup_logging()
        
    def setup_logging(self):
        """로깅 설정"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.data_path, 'market_regime.log'), encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def comprehensive_regime_analysis(self, data: pd.DataFrame) -> Dict:
        """종합적인 시장 상황 분석"""
        self.logger.info("🎯 종합적인 시장 상황 분석 시작...")
        
        try:
            # 1. 특성 변수 계산
            features_df = self.calculate_regime_features(data)
            
            # 2. 규칙 기반 상황 분류
            rule_based_regimes = self.rule_based_classification(features_df)
            
            # 3. 클러스터링 기반 분류
            clustering_regimes = self.clustering_based_classification(features_df)
            
            # 4. HMM 기반 분류 (선택적)
            hmm_regimes = None
            if HMM_AVAILABLE:
                hmm_regimes = self.hmm_based_classification(features_df)
            
            # 5. 앙상블 분류
            ensemble_regimes = self.ensemble_classification(
                rule_based_regimes, clustering_regimes, hmm_regimes
            )
            
            # 6. 상황 전환점 감지
            transition_points = self.detect_regime_transitions(ensemble_regimes)
            
            # 7. 상황별 성능 분석
            regime_performance = self.analyze_regime_performance(data, ensemble_regimes)
            
            # 8. 위기 상황 분석
            crisis_analysis = self.crisis_situation_analysis(data, ensemble_regimes)
            
            # 9. 예측 및 조기 경고
            regime_forecast = self.forecast_regime_changes(features_df)
            
            # 10. 스트레스 테스트
            stress_test_results = self.conduct_stress_tests(data, ensemble_regimes)
            
            # 종합 결과
            comprehensive_results = {
                'analysis_timestamp': datetime.now().isoformat(),
                'data_period': {
                    'start': data.index[0],
                    'end': data.index[-1],
                    'total_periods': len(data)
                },
                'feature_analysis': {
                    'features_calculated': len(features_df.columns),
                    'feature_importance': self.feature_importance
                },
                'regime_classifications': {
                    'rule_based': rule_based_regimes,
                    'clustering_based': clustering_regimes,
                    'hmm_based': hmm_regimes,
                    'ensemble': ensemble_regimes
                },
                'transition_analysis': transition_points,
                'performance_analysis': regime_performance,
                'crisis_analysis': crisis_analysis,
                'regime_forecast': regime_forecast,
                'stress_test_results': stress_test_results,
                'current_regime': self.current_regime
            }
            
            # 결과 저장
            self.save_analysis_results(comprehensive_results)
            
            # 시각화
            self.create_regime_visualization(comprehensive_results)
            
            return comprehensive_results
            
        except Exception as e:
            self.logger.error(f"시장 상황 분석 실패: {e}")
            raise
    
    def calculate_regime_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """시장 상황 분류를 위한 특성 변수 계산"""
        self.logger.info("📊 시장 상황 특성 변수 계산 중...")
        
        features_df = pd.DataFrame(index=data.index)
        
        # 가격 데이터 추출
        if 'price' in data.columns:
            prices = data['price']
        else:
            # 첫 번째 수치형 컬럼을 가격으로 가정
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            prices = data[numeric_cols[0]]
        
        # 1. 기본 수익률 및 변동성
        features_df['returns'] = prices.pct_change()
        features_df['abs_returns'] = features_df['returns'].abs()
        
        for period in [5, 10, 20, 60]:
            features_df[f'volatility_{period}'] = features_df['returns'].rolling(period).std() * np.sqrt(252)
            features_df[f'returns_mean_{period}'] = features_df['returns'].rolling(period).mean()
            features_df[f'returns_skew_{period}'] = features_df['returns'].rolling(period).skew()
            features_df[f'returns_kurt_{period}'] = features_df['returns'].rolling(period).kurt()
        
        # 2. 추세 지표
        for period in [20, 50, 100, 200]:
            features_df[f'sma_{period}'] = prices.rolling(period).mean()
            features_df[f'price_vs_sma_{period}'] = (prices / features_df[f'sma_{period}'] - 1)
            
            # 이동평균 기울기 (추세 강도)
            features_df[f'sma_slope_{period}'] = features_df[f'sma_{period}'].diff(10) / features_df[f'sma_{period}'].shift(10)
        
        # 3. 모멘텀 지표
        for period in [5, 10, 20, 60]:
            features_df[f'momentum_{period}'] = (prices / prices.shift(period) - 1)
            features_df[f'roc_{period}'] = prices.pct_change(period)
        
        # 4. 기술적 지표 (TA 라이브러리 사용)
        if TA_AVAILABLE:
            # RSI
            features_df['rsi_14'] = ta.momentum.rsi(prices, window=14)
            features_df['rsi_30'] = ta.momentum.rsi(prices, window=30)
            
            # MACD
            macd = ta.trend.MACD(prices)
            features_df['macd'] = macd.macd()
            features_df['macd_signal'] = macd.macd_signal()
            features_df['macd_diff'] = macd.macd_diff()
            
            # 볼린저 밴드
            bb = ta.volatility.BollingerBands(prices)
            features_df['bb_high'] = bb.bollinger_hband()
            features_df['bb_low'] = bb.bollinger_lband()
            features_df['bb_width'] = bb.bollinger_wband()
            features_df['bb_position'] = (prices - bb.bollinger_lband()) / (bb.bollinger_hband() - bb.bollinger_lband())
            
            # ATR (Average True Range)
            if 'high' in data.columns and 'low' in data.columns:
                features_df['atr_14'] = ta.volatility.AverageTrueRange(
                    high=data['high'], low=data['low'], close=prices
                ).average_true_range()
            
            # 스토캐스틱
            if 'high' in data.columns and 'low' in data.columns:
                stoch = ta.momentum.StochasticOscillator(
                    high=data['high'], low=data['low'], close=prices
                )
                features_df['stoch_k'] = stoch.stoch()
                features_df['stoch_d'] = stoch.stoch_signal()
        else:
            # TA 라이브러리 없이 기본 지표 계산
            features_df['rsi_14'] = self._calculate_rsi(prices, 14)
            features_df['macd'] = self._calculate_macd(prices)
        
        # 5. 거래량 지표 (있는 경우)
        if 'volume' in data.columns:
            volume = data['volume']
            features_df['volume'] = volume
            features_df['volume_sma_20'] = volume.rolling(20).mean()
            features_df['volume_ratio'] = volume / features_df['volume_sma_20']
            
            # 가격-거래량 관계
            features_df['price_volume_trend'] = ((prices - prices.shift(1)) * volume).rolling(20).sum()
            
            for period in [5, 10, 20]:
                features_df[f'volume_std_{period}'] = volume.rolling(period).std()
        
        # 6. 고차 모멘트
        for period in [20, 60]:
            features_df[f'higher_moment_3_{period}'] = features_df['returns'].rolling(period).apply(lambda x: stats.moment(x, moment=3))
            features_df[f'higher_moment_4_{period}'] = features_df['returns'].rolling(period).apply(lambda x: stats.moment(x, moment=4))
        
        # 7. 드로다운 지표
        cumulative_returns = (1 + features_df['returns']).cumprod()
        running_max = cumulative_returns.expanding().max()
        features_df['drawdown'] = (cumulative_returns - running_max) / running_max
        features_df['drawdown_duration'] = self._calculate_drawdown_duration(features_df['drawdown'])
        
        # 8. 변동성 클러스터링
        features_df['volatility_clustering'] = self._calculate_volatility_clustering(features_df['returns'])
        
        # 9. 시간 기반 특성
        features_df['hour'] = features_df.index.hour if hasattr(features_df.index, 'hour') else 0
        features_df['day_of_week'] = features_df.index.dayofweek if hasattr(features_df.index, 'dayofweek') else 0
        features_df['month'] = features_df.index.month if hasattr(features_df.index, 'month') else 0
        
        # NaN 처리
        features_df = features_df.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        self.logger.info(f"특성 변수 계산 완료: {len(features_df.columns)}개 변수")
        
        return features_df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """RSI 계산 (TA 라이브러리 없이)"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26) -> pd.Series:
        """MACD 계산 (TA 라이브러리 없이)"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        return macd
    
    def _calculate_drawdown_duration(self, drawdown: pd.Series) -> pd.Series:
        """드로다운 지속기간 계산"""
        duration = pd.Series(0, index=drawdown.index)
        current_duration = 0
        
        for i, dd in enumerate(drawdown):
            if dd < 0:  # 드로다운 상태
                current_duration += 1
            else:  # 회복 상태
                current_duration = 0
            duration.iloc[i] = current_duration
        
        return duration
    
    def _calculate_volatility_clustering(self, returns: pd.Series, window: int = 20) -> pd.Series:
        """변동성 클러스터링 계산"""
        abs_returns = returns.abs()
        rolling_vol = abs_returns.rolling(window).std()
        vol_ratio = abs_returns / rolling_vol
        return vol_ratio
    
    def rule_based_classification(self, features_df: pd.DataFrame) -> pd.Series:
        """규칙 기반 시장 상황 분류"""
        self.logger.info("🔍 규칙 기반 시장 상황 분류 중...")
        
        regimes = pd.Series(index=features_df.index, dtype=str)
        
        # 기본 지표 추출
        returns_20 = features_df.get('returns_mean_20', features_df['returns'].rolling(20).mean())
        volatility_20 = features_df.get('volatility_20', features_df['returns'].rolling(20).std() * np.sqrt(252))
        momentum_20 = features_df.get('momentum_20', features_df['returns'].rolling(20).sum())
        
        # 각 시점별 분류
        for i, idx in enumerate(features_df.index):
            if pd.isna(returns_20.iloc[i]) or pd.isna(volatility_20.iloc[i]):
                regimes.iloc[i] = MarketRegime.SIDEWAYS_LOW_VOL.value
                continue
            
            ret_20 = returns_20.iloc[i]
            vol_20 = volatility_20.iloc[i]
            mom_20 = momentum_20.iloc[i] if not pd.isna(momentum_20.iloc[i]) else 0
            
            # 위기 상황 감지 (급격한 하락)
            if ret_20 < self.config.crisis_threshold:
                regimes.iloc[i] = MarketRegime.CRISIS.value
            # 회복 상황 감지
            elif ret_20 > self.config.recovery_threshold and mom_20 > 0.05:
                regimes.iloc[i] = MarketRegime.RECOVERY.value
            # 일반 상황 분류
            else:
                # 추세 분류
                if ret_20 > self.config.trend_thresholds['bull']:
                    trend = 'bull'
                elif ret_20 < self.config.trend_thresholds['bear']:
                    trend = 'bear'
                else:
                    trend = 'sideways'
                
                # 변동성 분류
                if vol_20 > self.config.volatility_thresholds['high']:
                    vol_class = 'high'
                else:
                    vol_class = 'low'
                
                # 상황 매핑
                if trend == 'bull' and vol_class == 'high':
                    regimes.iloc[i] = MarketRegime.BULL_HIGH_VOL.value
                elif trend == 'bull' and vol_class == 'low':
                    regimes.iloc[i] = MarketRegime.BULL_LOW_VOL.value
                elif trend == 'bear' and vol_class == 'high':
                    regimes.iloc[i] = MarketRegime.BEAR_HIGH_VOL.value
                elif trend == 'bear' and vol_class == 'low':
                    regimes.iloc[i] = MarketRegime.BEAR_LOW_VOL.value
                elif trend == 'sideways' and vol_class == 'high':
                    regimes.iloc[i] = MarketRegime.SIDEWAYS_HIGH_VOL.value
                else:
                    regimes.iloc[i] = MarketRegime.SIDEWAYS_LOW_VOL.value
        
        return regimes
    
    def clustering_based_classification(self, features_df: pd.DataFrame) -> pd.Series:
        """클러스터링 기반 시장 상황 분류"""
        self.logger.info("🎯 클러스터링 기반 시장 상황 분류 중...")
        
        # 클러스터링용 특성 선택
        cluster_features = []
        for feature in self.config.cluster_features:
            matching_cols = [col for col in features_df.columns if feature in col.lower()]
            if matching_cols:
                cluster_features.extend(matching_cols[:2])  # 각 특성당 최대 2개 컬럼
        
        if not cluster_features:
            # 기본 특성 사용
            cluster_features = ['returns', 'volatility_20', 'momentum_20', 'rsi_14']
            cluster_features = [f for f in cluster_features if f in features_df.columns]
        
        if len(cluster_features) < 2:
            self.logger.warning("클러스터링을 위한 충분한 특성 없음")
            return pd.Series(MarketRegime.SIDEWAYS_LOW_VOL.value, index=features_df.index)
        
        # 데이터 전처리
        X = features_df[cluster_features].fillna(0)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 최적 클러스터 개수 찾기
        optimal_k = self._find_optimal_clusters(X_scaled)
        
        # K-Means 클러스터링
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_scaled)
        
        # 클러스터별 특성 분석
        cluster_characteristics = self._analyze_cluster_characteristics(
            features_df, cluster_labels, cluster_features
        )
        
        # 클러스터를 시장 상황으로 매핑
        regime_mapping = self._map_clusters_to_regimes(cluster_characteristics)
        
        # 결과 생성
        regimes = pd.Series(index=features_df.index, dtype=str)
        for i, label in enumerate(cluster_labels):
            regimes.iloc[i] = regime_mapping.get(label, MarketRegime.SIDEWAYS_LOW_VOL.value)
        
        # 모델 저장
        self.models['clustering'] = {
            'kmeans': kmeans,
            'scaler': scaler,
            'features': cluster_features,
            'regime_mapping': regime_mapping
        }
        
        return regimes
    
    def _find_optimal_clusters(self, X: np.ndarray, max_k: int = 8) -> int:
        """최적 클러스터 개수 찾기 (엘보우 방법 + 실루엣 스코어)"""
        if len(X) < 10:
            return 2
        
        max_k = min(max_k, len(X) // 3)  # 샘플 수에 따라 최대 k 조정
        
        inertias = []
        silhouette_scores = []
        k_range = range(2, max_k + 1)
        
        for k in k_range:
            if k >= len(X):
                break
                
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X)
            
            inertias.append(kmeans.inertia_)
            
            # 실루엣 스코어
            if len(set(labels)) > 1:  # 클러스터가 실제로 분리되었는지 확인
                sil_score = silhouette_score(X, labels)
                silhouette_scores.append(sil_score)
            else:
                silhouette_scores.append(0)
        
        # 실루엣 스코어가 가장 높은 k 선택
        if silhouette_scores:
            optimal_idx = np.argmax(silhouette_scores)
            optimal_k = list(k_range)[optimal_idx]
        else:
            optimal_k = self.config.n_clusters
        
        return optimal_k
    
    def _analyze_cluster_characteristics(self, features_df: pd.DataFrame, 
                                       labels: np.ndarray, features: List[str]) -> Dict:
        """클러스터별 특성 분석"""
        characteristics = {}
        
        for cluster_id in np.unique(labels):
            cluster_mask = labels == cluster_id
            cluster_data = features_df.loc[cluster_mask, features]
            
            characteristics[cluster_id] = {
                'size': int(cluster_mask.sum()),
                'proportion': float(cluster_mask.mean()),
                'mean_values': cluster_data.mean().to_dict(),
                'std_values': cluster_data.std().to_dict(),
                'feature_summary': {
                    'returns': cluster_data.get('returns', pd.Series(0)).mean(),
                    'volatility': cluster_data[[col for col in cluster_data.columns if 'volatility' in col]].mean().mean() if any('volatility' in col for col in cluster_data.columns) else 0,
                    'momentum': cluster_data[[col for col in cluster_data.columns if 'momentum' in col]].mean().mean() if any('momentum' in col for col in cluster_data.columns) else 0
                }
            }
        
        return characteristics
    
    def _map_clusters_to_regimes(self, cluster_characteristics: Dict) -> Dict:
        """클러스터를 시장 상황으로 매핑"""
        regime_mapping = {}
        
        for cluster_id, chars in cluster_characteristics.items():
            returns = chars['feature_summary']['returns']
            volatility = chars['feature_summary']['volatility']
            
            # 매핑 로직
            if returns > 0.1:  # 높은 양의 수익률
                if volatility > 0.5:
                    regime_mapping[cluster_id] = MarketRegime.BULL_HIGH_VOL.value
                else:
                    regime_mapping[cluster_id] = MarketRegime.BULL_LOW_VOL.value
            elif returns < -0.1:  # 높은 음의 수익률
                if volatility > 0.5:
                    regime_mapping[cluster_id] = MarketRegime.BEAR_HIGH_VOL.value
                else:
                    regime_mapping[cluster_id] = MarketRegime.BEAR_LOW_VOL.value
            else:  # 중립적 수익률
                if volatility > 0.5:
                    regime_mapping[cluster_id] = MarketRegime.SIDEWAYS_HIGH_VOL.value
                else:
                    regime_mapping[cluster_id] = MarketRegime.SIDEWAYS_LOW_VOL.value
        
        return regime_mapping
    
    def hmm_based_classification(self, features_df: pd.DataFrame) -> Optional[pd.Series]:
        """HMM 기반 시장 상황 분류"""
        if not HMM_AVAILABLE:
            self.logger.warning("HMM 라이브러리 미사용 - HMM 분류 생략")
            return None
        
        self.logger.info("🔬 HMM 기반 시장 상황 분류 중...")
        
        try:
            # HMM 입력 데이터 준비
            hmm_features = ['returns', 'volatility_20']  # 기본 특성
            hmm_features = [f for f in hmm_features if f in features_df.columns]
            
            if len(hmm_features) < 1:
                return None
            
            X = features_df[hmm_features].fillna(0).values
            
            if len(X) < 50:  # HMM에는 충분한 데이터 필요
                return None
            
            # HMM 모델 학습
            model = hmm.GaussianHMM(
                n_components=self.config.n_hidden_states,
                covariance_type=self.config.hmm_covariance_type,
                random_state=42
            )
            
            model.fit(X)
            hidden_states = model.predict(X)
            
            # 상태별 특성 분석
            state_characteristics = {}
            for state in range(self.config.n_hidden_states):
                state_mask = hidden_states == state
                state_data = features_df.loc[state_mask]
                
                if len(state_data) > 0:
                    state_characteristics[state] = {
                        'mean_returns': state_data['returns'].mean() if 'returns' in state_data else 0,
                        'mean_volatility': state_data.get('volatility_20', pd.Series(0)).mean()
                    }
            
            # 상태를 시장 상황으로 매핑
            state_to_regime = self._map_hmm_states_to_regimes(state_characteristics)
            
            # 결과 생성
            regimes = pd.Series(index=features_df.index, dtype=str)
            for i, state in enumerate(hidden_states):
                regimes.iloc[i] = state_to_regime.get(state, MarketRegime.SIDEWAYS_LOW_VOL.value)
            
            # 모델 저장
            self.models['hmm'] = {
                'model': model,
                'features': hmm_features,
                'state_mapping': state_to_regime
            }
            
            return regimes
            
        except Exception as e:
            self.logger.warning(f"HMM 분류 실패: {e}")
            return None
    
    def _map_hmm_states_to_regimes(self, state_characteristics: Dict) -> Dict:
        """HMM 상태를 시장 상황으로 매핑"""
        state_mapping = {}
        
        # 수익률과 변동성에 따라 상태 정렬
        states_by_returns = sorted(state_characteristics.items(), 
                                 key=lambda x: x[1]['mean_returns'])
        
        n_states = len(states_by_returns)
        
        for i, (state, chars) in enumerate(states_by_returns):
            returns = chars['mean_returns']
            volatility = chars['mean_volatility']
            
            # 수익률 순서에 따른 매핑
            if i < n_states // 3:  # 하위 1/3 - 약세
                if volatility > 0.5:
                    state_mapping[state] = MarketRegime.BEAR_HIGH_VOL.value
                else:
                    state_mapping[state] = MarketRegime.BEAR_LOW_VOL.value
            elif i >= 2 * n_states // 3:  # 상위 1/3 - 강세
                if volatility > 0.5:
                    state_mapping[state] = MarketRegime.BULL_HIGH_VOL.value
                else:
                    state_mapping[state] = MarketRegime.BULL_LOW_VOL.value
            else:  # 중간 1/3 - 횡보
                if volatility > 0.5:
                    state_mapping[state] = MarketRegime.SIDEWAYS_HIGH_VOL.value
                else:
                    state_mapping[state] = MarketRegime.SIDEWAYS_LOW_VOL.value
        
        return state_mapping
    
    def ensemble_classification(self, rule_based: pd.Series, 
                              clustering: pd.Series, 
                              hmm_based: Optional[pd.Series] = None) -> pd.Series:
        """앙상블 시장 상황 분류"""
        self.logger.info("🎯 앙상블 시장 상황 분류 중...")
        
        # 가중치 설정
        weights = {'rule': 0.4, 'clustering': 0.4, 'hmm': 0.2}
        
        if hmm_based is None:
            weights = {'rule': 0.6, 'clustering': 0.4}
        
        ensemble_regimes = pd.Series(index=rule_based.index, dtype=str)
        
        for i in range(len(rule_based)):
            votes = {}
            
            # 규칙 기반 투표
            rule_vote = rule_based.iloc[i]
            votes[rule_vote] = votes.get(rule_vote, 0) + weights['rule']
            
            # 클러스터링 기반 투표
            clustering_vote = clustering.iloc[i]
            votes[clustering_vote] = votes.get(clustering_vote, 0) + weights['clustering']
            
            # HMM 기반 투표
            if hmm_based is not None:
                hmm_vote = hmm_based.iloc[i]
                votes[hmm_vote] = votes.get(hmm_vote, 0) + weights['hmm']
            
            # 최다 득표 상황 선택
            ensemble_regimes.iloc[i] = max(votes.items(), key=lambda x: x[1])[0]
        
        return ensemble_regimes
    
    def detect_regime_transitions(self, regimes: pd.Series) -> Dict:
        """시장 상황 전환점 감지"""
        self.logger.info("🔄 시장 상황 전환점 감지 중...")
        
        transitions = []
        current_regime = regimes.iloc[0] if len(regimes) > 0 else None
        regime_start = regimes.index[0] if len(regimes) > 0 else None
        
        for i, (timestamp, regime) in enumerate(regimes.items()):
            if regime != current_regime:
                # 전환점 감지
                if current_regime is not None and regime_start is not None:
                    transitions.append({
                        'from_regime': current_regime,
                        'to_regime': regime,
                        'transition_date': timestamp,
                        'previous_duration': i - regimes.index.get_loc(regime_start) if regime_start in regimes.index else 0,
                        'transition_type': self._classify_transition_type(current_regime, regime)
                    })
                
                current_regime = regime
                regime_start = timestamp
        
        # 전환 통계
        transition_stats = self._calculate_transition_statistics(transitions)
        
        return {
            'transitions': transitions,
            'transition_statistics': transition_stats,
            'total_transitions': len(transitions),
            'avg_regime_duration': np.mean([t['previous_duration'] for t in transitions]) if transitions else 0
        }
    
    def _classify_transition_type(self, from_regime: str, to_regime: str) -> str:
        """전환 유형 분류"""
        # 위기 관련 전환
        if 'CRISIS' in to_regime:
            return 'crisis_onset'
        elif 'CRISIS' in from_regime:
            return 'crisis_recovery'
        
        # 추세 전환
        if 'BULL' in from_regime and 'BEAR' in to_regime:
            return 'bull_to_bear'
        elif 'BEAR' in from_regime and 'BULL' in to_regime:
            return 'bear_to_bull'
        elif 'SIDEWAYS' in from_regime and ('BULL' in to_regime or 'BEAR' in to_regime):
            return 'breakout'
        elif ('BULL' in from_regime or 'BEAR' in from_regime) and 'SIDEWAYS' in to_regime:
            return 'consolidation'
        
        # 변동성 전환
        if 'LOW_VOL' in from_regime and 'HIGH_VOL' in to_regime:
            return 'volatility_increase'
        elif 'HIGH_VOL' in from_regime and 'LOW_VOL' in to_regime:
            return 'volatility_decrease'
        
        return 'other'
    
    def _calculate_transition_statistics(self, transitions: List[Dict]) -> Dict:
        """전환 통계 계산"""
        if not transitions:
            return {}
        
        # 전환 유형별 빈도
        transition_types = [t['transition_type'] for t in transitions]
        type_counts = {t_type: transition_types.count(t_type) for t_type in set(transition_types)}
        
        # 지속 기간 분석
        durations = [t['previous_duration'] for t in transitions if t['previous_duration'] > 0]
        
        return {
            'transition_type_frequency': type_counts,
            'duration_statistics': {
                'mean': np.mean(durations) if durations else 0,
                'median': np.median(durations) if durations else 0,
                'std': np.std(durations) if durations else 0,
                'min': np.min(durations) if durations else 0,
                'max': np.max(durations) if durations else 0
            }
        }
    
    def analyze_regime_performance(self, data: pd.DataFrame, 
                                 regimes: pd.Series) -> Dict:
        """시장 상황별 성능 분석"""
        self.logger.info("📈 시장 상황별 성능 분석 중...")
        
        performance_analysis = {}
        
        # 가격 데이터
        if 'price' in data.columns:
            prices = data['price']
        else:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            prices = data[numeric_cols[0]]
        
        returns = prices.pct_change().dropna()
        
        # 각 상황별 성능 분석
        for regime_type in regimes.unique():
            if pd.isna(regime_type):
                continue
                
            regime_mask = regimes == regime_type
            regime_returns = returns[regime_mask]
            
            if len(regime_returns) == 0:
                continue
            
            # 기본 통계
            performance_analysis[regime_type] = {
                'observations': len(regime_returns),
                'total_periods': regime_mask.sum(),
                'proportion': float(regime_mask.mean()),
                'returns_statistics': {
                    'mean': float(regime_returns.mean()),
                    'median': float(regime_returns.median()),
                    'std': float(regime_returns.std()),
                    'skewness': float(stats.skew(regime_returns)) if len(regime_returns) > 3 else 0,
                    'kurtosis': float(stats.kurtosis(regime_returns)) if len(regime_returns) > 3 else 0
                },
                'annual_metrics': {
                    'annual_return': float(regime_returns.mean() * 252),
                    'annual_volatility': float(regime_returns.std() * np.sqrt(252)),
                    'sharpe_ratio': float((regime_returns.mean() * 252) / (regime_returns.std() * np.sqrt(252))) if regime_returns.std() > 0 else 0
                },
                'risk_metrics': {
                    'var_95': float(np.percentile(regime_returns, 5)),
                    'max_loss': float(regime_returns.min()),
                    'max_gain': float(regime_returns.max()),
                    'positive_periods_ratio': float((regime_returns > 0).mean())
                }
            }
        
        return performance_analysis
    
    def crisis_situation_analysis(self, data: pd.DataFrame, regimes: pd.Series) -> Dict:
        """위기 상황 분석"""
        self.logger.info("🚨 위기 상황 분석 중...")
        
        # 위기 상황 식별
        crisis_periods = regimes == MarketRegime.CRISIS.value
        
        if not crisis_periods.any():
            return {'no_crisis_detected': True}
        
        # 가격 데이터
        if 'price' in data.columns:
            prices = data['price']
        else:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            prices = data[numeric_cols[0]]
        
        returns = prices.pct_change()
        
        # 위기 기간 분석
        crisis_analysis = {
            'crisis_periods_detected': int(crisis_periods.sum()),
            'crisis_proportion': float(crisis_periods.mean()),
            'crisis_statistics': {
                'avg_return': float(returns[crisis_periods].mean()),
                'volatility': float(returns[crisis_periods].std() * np.sqrt(252)),
                'max_drawdown': self._calculate_crisis_max_drawdown(prices, crisis_periods),
                'recovery_analysis': self._analyze_crisis_recovery(prices, regimes)
            }
        }
        
        return crisis_analysis
    
    def _calculate_crisis_max_drawdown(self, prices: pd.Series, crisis_mask: pd.Series) -> float:
        """위기 기간 최대 드로다운 계산"""
        crisis_prices = prices[crisis_mask]
        if len(crisis_prices) == 0:
            return 0.0
        
        cumulative = (1 + crisis_prices.pct_change()).cumprod()
        running_max = cumulative.expanding().max()
        drawdowns = (cumulative - running_max) / running_max
        return float(drawdowns.min())
    
    def _analyze_crisis_recovery(self, prices: pd.Series, regimes: pd.Series) -> Dict:
        """위기 회복 분석"""
        recovery_periods = regimes == MarketRegime.RECOVERY.value
        
        if not recovery_periods.any():
            return {'no_recovery_detected': True}
        
        recovery_returns = prices.pct_change()[recovery_periods]
        
        return {
            'recovery_periods': int(recovery_periods.sum()),
            'avg_recovery_return': float(recovery_returns.mean()),
            'recovery_volatility': float(recovery_returns.std() * np.sqrt(252))
        }
    
    def forecast_regime_changes(self, features_df: pd.DataFrame, 
                               forecast_horizon: int = 30) -> Dict:
        """시장 상황 변화 예측 및 조기 경고"""
        self.logger.info("🔮 시장 상황 변화 예측 중...")
        
        # 현재 상황 확인
        if len(features_df) < 50:
            return {'insufficient_data': True}
        
        # 최근 특성 변화 분석
        recent_features = features_df.tail(30)  # 최근 30일
        
        # 변화 지표 계산
        trend_signals = self._calculate_trend_signals(recent_features)
        volatility_signals = self._calculate_volatility_signals(recent_features)
        momentum_signals = self._calculate_momentum_signals(recent_features)
        
        # 전환 확률 예측
        transition_probability = self._predict_transition_probability(
            trend_signals, volatility_signals, momentum_signals
        )
        
        # 조기 경고 시스템
        early_warnings = self._generate_early_warnings(
            trend_signals, volatility_signals, momentum_signals
        )
        
        return {
            'forecast_horizon_days': forecast_horizon,
            'current_trend_signals': trend_signals,
            'volatility_signals': volatility_signals,
            'momentum_signals': momentum_signals,
            'transition_probability': transition_probability,
            'early_warnings': early_warnings,
            'forecast_confidence': self._calculate_forecast_confidence(recent_features)
        }
    
    def _calculate_trend_signals(self, features_df: pd.DataFrame) -> Dict:
        """추세 신호 계산"""
        signals = {}
        
        # 이동평균 신호
        if 'price_vs_sma_20' in features_df.columns:
            sma_signal = features_df['price_vs_sma_20'].iloc[-1]
            signals['sma_signal'] = 'bullish' if sma_signal > 0.05 else 'bearish' if sma_signal < -0.05 else 'neutral'
        
        # 추세 기울기
        if 'sma_slope_50' in features_df.columns:
            slope = features_df['sma_slope_50'].iloc[-5:].mean()  # 최근 5일 평균
            signals['trend_slope'] = 'increasing' if slope > 0.01 else 'decreasing' if slope < -0.01 else 'flat'
        
        return signals
    
    def _calculate_volatility_signals(self, features_df: pd.DataFrame) -> Dict:
        """변동성 신호 계산"""
        signals = {}
        
        # 변동성 추세
        if 'volatility_20' in features_df.columns:
            vol_recent = features_df['volatility_20'].iloc[-5:].mean()
            vol_historical = features_df['volatility_20'].iloc[-30:-5].mean()
            
            if vol_recent > vol_historical * 1.2:
                signals['volatility_trend'] = 'increasing'
            elif vol_recent < vol_historical * 0.8:
                signals['volatility_trend'] = 'decreasing'
            else:
                signals['volatility_trend'] = 'stable'
        
        return signals
    
    def _calculate_momentum_signals(self, features_df: pd.DataFrame) -> Dict:
        """모멘텀 신호 계산"""
        signals = {}
        
        # RSI 신호
        if 'rsi_14' in features_df.columns:
            rsi = features_df['rsi_14'].iloc[-1]
            if rsi > 70:
                signals['rsi_signal'] = 'overbought'
            elif rsi < 30:
                signals['rsi_signal'] = 'oversold'
            else:
                signals['rsi_signal'] = 'neutral'
        
        # MACD 신호
        if 'macd' in features_df.columns and 'macd_signal' in features_df.columns:
            macd_diff = features_df['macd'].iloc[-1] - features_df['macd_signal'].iloc[-1]
            signals['macd_signal'] = 'bullish' if macd_diff > 0 else 'bearish'
        
        return signals
    
    def _predict_transition_probability(self, trend_signals: Dict, 
                                      volatility_signals: Dict, 
                                      momentum_signals: Dict) -> Dict:
        """전환 확률 예측"""
        # 간단한 규칙 기반 확률 계산
        transition_scores = {
            'bull_to_bear': 0,
            'bear_to_bull': 0,
            'volatility_increase': 0,
            'volatility_decrease': 0
        }
        
        # 추세 신호 기여
        if trend_signals.get('sma_signal') == 'bearish':
            transition_scores['bull_to_bear'] += 0.3
        elif trend_signals.get('sma_signal') == 'bullish':
            transition_scores['bear_to_bull'] += 0.3
        
        # 변동성 신호 기여
        if volatility_signals.get('volatility_trend') == 'increasing':
            transition_scores['volatility_increase'] += 0.4
        elif volatility_signals.get('volatility_trend') == 'decreasing':
            transition_scores['volatility_decrease'] += 0.4
        
        # 모멘텀 신호 기여
        if momentum_signals.get('rsi_signal') == 'overbought':
            transition_scores['bull_to_bear'] += 0.2
        elif momentum_signals.get('rsi_signal') == 'oversold':
            transition_scores['bear_to_bull'] += 0.2
        
        return transition_scores
    
    def _generate_early_warnings(self, trend_signals: Dict, 
                               volatility_signals: Dict, 
                               momentum_signals: Dict) -> List[str]:
        """조기 경고 생성"""
        warnings = []
        
        # 위기 조기 경고
        if (trend_signals.get('trend_slope') == 'decreasing' and 
            volatility_signals.get('volatility_trend') == 'increasing'):
            warnings.append("잠재적 위기 상황 감지 - 하락 추세 + 변동성 증가")
        
        # 과매수/과매도 경고
        if momentum_signals.get('rsi_signal') == 'overbought':
            warnings.append("과매수 상황 - 조정 가능성")
        elif momentum_signals.get('rsi_signal') == 'oversold':
            warnings.append("과매도 상황 - 반등 가능성")
        
        # 추세 전환 경고
        if (trend_signals.get('sma_signal') in ['bullish', 'bearish'] and 
            momentum_signals.get('macd_signal') != trend_signals.get('sma_signal')):
            warnings.append("추세 지표 간 불일치 - 전환 신호 가능성")
        
        return warnings
    
    def _calculate_forecast_confidence(self, features_df: pd.DataFrame) -> float:
        """예측 신뢰도 계산"""
        # 데이터 품질 기반 신뢰도
        data_completeness = 1 - features_df.isna().mean().mean()
        
        # 최근 변동성 기반 조정 (높은 변동성 = 낮은 예측 신뢰도)
        if 'volatility_20' in features_df.columns:
            recent_vol = features_df['volatility_20'].iloc[-5:].mean()
            vol_penalty = min(recent_vol / 2, 0.3)  # 최대 30% 패널티
        else:
            vol_penalty = 0
        
        confidence = max(0.1, min(0.9, data_completeness - vol_penalty))
        return float(confidence)
    
    def conduct_stress_tests(self, data: pd.DataFrame, regimes: pd.Series) -> Dict:
        """스트레스 테스트 시행"""
        self.logger.info("🧪 스트레스 테스트 시행 중...")
        
        stress_results = {}
        
        # 가격 데이터
        if 'price' in data.columns:
            prices = data['price']
        else:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            prices = data[numeric_cols[0]]
        
        returns = prices.pct_change().dropna()
        
        for scenario in self.config.stress_scenarios:
            stress_results[scenario] = self._simulate_stress_scenario(scenario, returns, regimes)
        
        return stress_results
    
    def _simulate_stress_scenario(self, scenario: str, returns: pd.Series, 
                                regimes: pd.Series) -> Dict:
        """스트레스 시나리오 시뮬레이션"""
        scenario_results = {'scenario': scenario}
        
        if scenario == 'black_swan':
            # 극단적 하락 시나리오 (-30% 하락)
            shock_return = -0.30
            scenario_results.update(self._analyze_shock_impact(shock_return, returns))
            
        elif scenario == 'market_crash':
            # 시장 폭락 시나리오 (-50% 하락)
            shock_return = -0.50
            scenario_results.update(self._analyze_shock_impact(shock_return, returns))
            
        elif scenario == 'liquidity_crisis':
            # 유동성 위기 (높은 변동성 지속)
            volatility_multiplier = 3.0
            scenario_results.update(self._analyze_volatility_shock(volatility_multiplier, returns))
            
        elif scenario == 'regulatory_shock':
            # 규제 쇼크 (중간 정도 하락 + 높은 변동성)
            shock_return = -0.20
            volatility_multiplier = 2.0
            scenario_results.update(self._analyze_combined_shock(shock_return, volatility_multiplier, returns))
        
        return scenario_results
    
    def _analyze_shock_impact(self, shock_return: float, returns: pd.Series) -> Dict:
        """쇼크 영향 분석"""
        # 현재 포트폴리오 가치 계산
        portfolio_value = (1 + returns).cumprod()
        
        # 쇼크 적용
        shocked_value = portfolio_value.iloc[-1] * (1 + shock_return)
        
        # 최대 드로다운 계산
        peak_value = portfolio_value.max()
        max_drawdown = (shocked_value - peak_value) / peak_value
        
        return {
            'shock_magnitude': shock_return,
            'portfolio_impact': shocked_value / portfolio_value.iloc[-1] - 1,
            'max_drawdown': max_drawdown,
            'recovery_time_estimate': abs(shock_return) / (returns.mean() * 252) if returns.mean() > 0 else float('inf')
        }
    
    def _analyze_volatility_shock(self, volatility_multiplier: float, returns: pd.Series) -> Dict:
        """변동성 쇼크 분석"""
        current_vol = returns.std() * np.sqrt(252)
        shocked_vol = current_vol * volatility_multiplier
        
        # 변동성 증가에 따른 VaR 변화
        current_var_95 = np.percentile(returns, 5)
        shocked_var_95 = current_var_95 * volatility_multiplier
        
        return {
            'volatility_multiplier': volatility_multiplier,
            'current_volatility': current_vol,
            'shocked_volatility': shocked_vol,
            'var_95_change': shocked_var_95 / current_var_95 - 1 if current_var_95 != 0 else 0
        }
    
    def _analyze_combined_shock(self, shock_return: float, volatility_multiplier: float, 
                              returns: pd.Series) -> Dict:
        """복합 쇼크 분석"""
        price_impact = self._analyze_shock_impact(shock_return, returns)
        vol_impact = self._analyze_volatility_shock(volatility_multiplier, returns)
        
        return {
            'combined_shock': True,
            'price_impact': price_impact,
            'volatility_impact': vol_impact,
            'total_risk_increase': price_impact['max_drawdown'] + vol_impact['var_95_change']
        }
    
    def create_regime_visualization(self, analysis_results: Dict) -> str:
        """시장 상황 분석 시각화"""
        self.logger.info("📊 시장 상황 분석 시각화 생성 중...")
        
        fig = make_subplots(
            rows=4, cols=2,
            subplot_titles=(
                "시장 상황 시계열", "상황별 성능 비교",
                "전환점 분석", "상황별 변동성",
                "위기 상황 분석", "스트레스 테스트 결과",
                "예측 신호", "리스크 지표"
            ),
            specs=[[{"type": "scatter"}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "box"}],
                   [{"type": "bar"}, {"type": "radar"}],
                   [{"type": "indicator"}, {"type": "table"}]]
        )
        
        # 데이터 추출
        ensemble_regimes = analysis_results['regime_classifications']['ensemble']
        performance_data = analysis_results.get('performance_analysis', {})
        
        if ensemble_regimes is not None and len(ensemble_regimes) > 0:
            # 1. 시장 상황 시계열
            regime_colors = {
                MarketRegime.BULL_LOW_VOL.value: 'green',
                MarketRegime.BULL_HIGH_VOL.value: 'lightgreen',
                MarketRegime.BEAR_LOW_VOL.value: 'red',
                MarketRegime.BEAR_HIGH_VOL.value: 'lightcoral',
                MarketRegime.SIDEWAYS_LOW_VOL.value: 'blue',
                MarketRegime.SIDEWAYS_HIGH_VOL.value: 'lightblue',
                MarketRegime.CRISIS.value: 'black',
                MarketRegime.RECOVERY.value: 'orange'
            }
            
            # 상황별 색상 매핑
            colors = [regime_colors.get(regime, 'gray') for regime in ensemble_regimes]
            
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(ensemble_regimes))),
                    y=[1] * len(ensemble_regimes),
                    mode='markers',
                    marker=dict(color=colors, size=8),
                    name='시장 상황'
                ),
                row=1, col=1
            )
        
        # 2. 상황별 성능 비교
        if performance_data:
            regimes = list(performance_data.keys())
            returns = [performance_data[r]['annual_metrics']['annual_return'] for r in regimes]
            
            fig.add_trace(
                go.Bar(x=regimes, y=returns, name='연간 수익률'),
                row=1, col=2
            )
        
        # 추가 차트들...
        
        fig.update_layout(
            title="🎯 시장 상황 분석 대시보드",
            height=1600,
            showlegend=True,
            template='plotly_dark'
        )
        
        # 저장
        dashboard_path = os.path.join(self.data_path, 'market_regime_analysis_dashboard.html')
        fig.write_html(dashboard_path, include_plotlyjs=True)
        
        return dashboard_path
    
    def save_analysis_results(self, results: Dict):
        """분석 결과 저장"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # JSON 저장
        json_path = os.path.join(self.data_path, f'market_regime_analysis_{timestamp}.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        self.logger.info(f"분석 결과 저장: {json_path}")

def main():
    """메인 실행 함수"""
    print("🎯 시장 상황별 분석 모듈")
    print("=" * 50)
    
    # 설정
    config = RegimeConfig(
        n_clusters=6,
        n_hidden_states=4,
        crisis_threshold=-0.20,
        recovery_threshold=0.10
    )
    
    # 분석기 초기화
    analyzer = MarketRegimeAnalyzer(config)
    
    # 테스트 데이터 생성
    np.random.seed(42)
    n_periods = 1000
    
    # 시뮬레이션된 가격 데이터
    returns = np.random.normal(0.001, 0.02, n_periods)
    
    # 상황별 변동성 변화 시뮬레이션
    regime_changes = [200, 400, 600, 800]  # 상황 변화 지점
    for i, change_point in enumerate(regime_changes):
        if i % 2 == 0:  # 높은 변동성 구간
            returns[change_point:change_point+100] *= 2
        else:  # 낮은 변동성 구간
            returns[change_point:change_point+100] *= 0.5
    
    prices = (1 + pd.Series(returns)).cumprod() * 10000
    
    # 데이터프레임 생성
    data = pd.DataFrame({
        'price': prices,
        'volume': np.random.lognormal(10, 1, n_periods)
    })
    data.index = pd.date_range('2022-01-01', periods=n_periods, freq='D')
    
    # 종합 분석 실행
    results = analyzer.comprehensive_regime_analysis(data)
    
    print(f"\n📊 분석 완료!")
    print(f"총 기간: {len(data)} 일")
    
    if 'regime_classifications' in results and results['regime_classifications']['ensemble'] is not None:
        ensemble = results['regime_classifications']['ensemble']
        unique_regimes = ensemble.value_counts()
        print(f"식별된 시장 상황: {len(unique_regimes)}개")
        
        for regime, count in unique_regimes.head().items():
            print(f"  • {regime}: {count}일 ({count/len(ensemble)*100:.1f}%)")
    
    # 대시보드 경로 출력
    if 'dashboard_path' in results:
        print(f"대시보드: {results['dashboard_path']}")

if __name__ == "__main__":
    main()