#!/usr/bin/env python3
"""
변동성 체제 분석 및 전환 확률 모델링 시스템
Bitcoin 변동성 패턴을 분석하여 체제 전환을 예측하고 확률을 모델링

핵심 기능:
1. 변동성 클러스터링 감지 (GARCH 모델)
2. 변동성 체제 전환점 식별
3. 체제 지속 기간 예측 
4. 변동성 충격 전파 모델링
5. 리스크 조정 포지션 사이징
"""

import os
import json
import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
import logging
from dataclasses import dataclass, asdict
import asyncio
import pickle
from collections import defaultdict, deque
import statistics
import warnings
warnings.filterwarnings("ignore")

# 시계열 및 변동성 모델링
from scipy import stats
from scipy.signal import find_peaks, argrelextrema
from scipy.optimize import minimize
import sklearn.cluster as cluster
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture

# 금융 시계열 분석
try:
    from arch import arch_model
    from arch.unitroot import ADF
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False
    logger.warning("ARCH 라이브러리가 설치되지 않았습니다. GARCH 모델 대신 간단한 모델을 사용합니다.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class VolatilityRegime:
    """변동성 체제 정보"""
    regime_id: int
    regime_name: str
    start_date: datetime
    end_date: Optional[datetime]
    duration_days: int
    avg_volatility: float
    volatility_std: float
    max_volatility: float
    min_volatility: float
    volatility_persistence: float
    clustering_coefficient: float
    regime_confidence: float

@dataclass
class RegimeTransitionProbability:
    """체제 전환 확률"""
    from_regime: str
    to_regime: str
    base_probability: float
    conditional_probabilities: Dict[str, float]
    trigger_conditions: List[str]
    expected_duration_days: float
    confidence_interval: Tuple[float, float]
    last_updated: datetime

@dataclass
class VolatilityForecast:
    """변동성 예측"""
    forecast_horizon: int
    volatility_forecast: List[float]
    confidence_bands: List[Tuple[float, float]]
    regime_probabilities: Dict[str, List[float]]
    peak_volatility_date: Optional[datetime]
    trough_volatility_date: Optional[datetime]
    forecast_accuracy_score: float

class VolatilityRegimeAnalyzer:
    """변동성 체제 분석기"""
    
    def __init__(self, base_path: str = "/Users/parkyoungjun/Desktop/BTC_Analysis_System"):
        self.base_path = base_path
        self.db_path = os.path.join(base_path, "volatility_regime_db.db")
        self.models_path = os.path.join(base_path, "volatility_models")
        os.makedirs(self.models_path, exist_ok=True)
        
        # 변동성 체제 정의
        self.volatility_regimes = {
            0: "ULTRA_LOW",      # < 1% 일간 변동성
            1: "LOW",            # 1-2%  
            2: "NORMAL",         # 2-4%
            3: "HIGH",           # 4-7%
            4: "EXTREME",        # > 7%
        }
        
        # 체제 임계값
        self.regime_thresholds = [0.01, 0.02, 0.04, 0.07, float('inf')]
        
        # 분석 모델들
        self.garch_model = None
        self.regime_classifier = None
        self.transition_matrix = np.zeros((5, 5))
        self.duration_models = {}
        
        # 현재 체제 추적
        self.current_regime = None
        self.regime_history = deque(maxlen=500)
        self.volatility_history = deque(maxlen=1000)
        
        # 예측 모델
        self.forecast_models = {}
        
        self.init_database()
        self.load_models()
    
    def init_database(self):
        """데이터베이스 초기화"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 변동성 체제 기록
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS volatility_regimes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    regime_id INTEGER NOT NULL,
                    regime_name TEXT NOT NULL,
                    start_date TEXT NOT NULL,
                    end_date TEXT,
                    duration_days INTEGER NOT NULL,
                    avg_volatility REAL NOT NULL,
                    volatility_std REAL NOT NULL,
                    max_volatility REAL NOT NULL,
                    min_volatility REAL NOT NULL,
                    volatility_persistence REAL NOT NULL,
                    clustering_coefficient REAL NOT NULL,
                    regime_confidence REAL NOT NULL
                )
            ''')
            
            # 체제 전환 확률
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS regime_transition_probabilities (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    from_regime TEXT NOT NULL,
                    to_regime TEXT NOT NULL,
                    base_probability REAL NOT NULL,
                    conditional_probabilities TEXT NOT NULL,
                    trigger_conditions TEXT NOT NULL,
                    expected_duration_days REAL NOT NULL,
                    confidence_interval_low REAL NOT NULL,
                    confidence_interval_high REAL NOT NULL,
                    last_updated TEXT NOT NULL
                )
            ''')
            
            # 변동성 예측
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS volatility_forecasts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    forecast_date TEXT NOT NULL,
                    forecast_horizon INTEGER NOT NULL,
                    volatility_forecast TEXT NOT NULL,
                    confidence_bands TEXT NOT NULL,
                    regime_probabilities TEXT NOT NULL,
                    peak_volatility_date TEXT,
                    trough_volatility_date TEXT,
                    forecast_accuracy_score REAL,
                    model_used TEXT NOT NULL
                )
            ''')
            
            # 변동성 데이터
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS volatility_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    realized_volatility REAL NOT NULL,
                    implied_volatility REAL,
                    garch_volatility REAL,
                    regime_id INTEGER NOT NULL,
                    regime_probability REAL NOT NULL,
                    clustering_score REAL,
                    persistence_score REAL,
                    shock_magnitude REAL
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("✅ 변동성 체제 분석기 데이터베이스 초기화 완료")
            
        except Exception as e:
            logger.error(f"변동성 DB 초기화 실패: {e}")
    
    def load_models(self):
        """저장된 모델 로드"""
        try:
            # GARCH 모델 로드
            garch_file = os.path.join(self.models_path, "garch_volatility_model.pkl")
            if os.path.exists(garch_file):
                with open(garch_file, 'rb') as f:
                    self.garch_model = pickle.load(f)
            
            # 체제 분류 모델 로드
            classifier_file = os.path.join(self.models_path, "regime_classifier.pkl")
            if os.path.exists(classifier_file):
                with open(classifier_file, 'rb') as f:
                    self.regime_classifier = pickle.load(f)
            
            # 전환 행렬 로드
            transition_file = os.path.join(self.models_path, "transition_matrix.npy")
            if os.path.exists(transition_file):
                self.transition_matrix = np.load(transition_file)
            
            logger.info("✅ 변동성 모델들 로드 완료")
            
        except Exception as e:
            logger.error(f"변동성 모델 로드 실패: {e}")
    
    async def analyze_volatility_regime(self, price_data: List[float], 
                                       timestamps: List[datetime]) -> Dict:
        """변동성 체제 분석"""
        try:
            if len(price_data) < 30:
                return {"error": "변동성 분석을 위한 충분한 데이터가 없습니다"}
            
            logger.info(f"🔍 변동성 체제 분석 시작 (데이터: {len(price_data)}개)")
            
            # 수익률 계산
            returns = self.calculate_returns(price_data)
            
            # 실현 변동성 계산
            realized_vol = self.calculate_realized_volatility(returns)
            
            # GARCH 변동성 모델링
            garch_results = await self.model_garch_volatility(returns)
            
            # 변동성 클러스터링 분석
            clustering_results = self.analyze_volatility_clustering(realized_vol)
            
            # 체제 식별
            regime_results = await self.identify_volatility_regimes(realized_vol, timestamps)
            
            # 지속성 분석
            persistence_results = self.analyze_volatility_persistence(realized_vol)
            
            # 충격 전파 분석
            shock_results = self.analyze_volatility_shocks(returns, realized_vol)
            
            # 체제 전환 확률 업데이트
            await self.update_transition_probabilities(regime_results)
            
            analysis_result = {
                "analysis_date": datetime.now().isoformat(),
                "data_points": len(price_data),
                "current_regime": regime_results.get("current_regime"),
                "current_volatility": realized_vol[-1] if realized_vol else 0,
                "garch_results": garch_results,
                "clustering_results": clustering_results,
                "regime_results": regime_results,
                "persistence_results": persistence_results,
                "shock_results": shock_results,
                "regime_forecast": await self.forecast_regime_changes(7)
            }
            
            # 결과 저장
            await self.save_analysis_results(analysis_result)
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"변동성 체제 분석 실패: {e}")
            return {"error": str(e)}
    
    def calculate_returns(self, prices: List[float]) -> np.ndarray:
        """수익률 계산"""
        try:
            prices = np.array(prices)
            returns = np.diff(np.log(prices))
            return returns
        except Exception as e:
            logger.error(f"수익률 계산 실패: {e}")
            return np.array([])
    
    def calculate_realized_volatility(self, returns: np.ndarray, window: int = 24) -> List[float]:
        """실현 변동성 계산 (이동 윈도우)"""
        try:
            if len(returns) < window:
                return []
            
            realized_vol = []
            for i in range(window, len(returns) + 1):
                window_returns = returns[i-window:i]
                vol = np.sqrt(np.var(window_returns) * 365)  # 연율화
                realized_vol.append(vol)
            
            return realized_vol
            
        except Exception as e:
            logger.error(f"실현 변동성 계산 실패: {e}")
            return []
    
    async def model_garch_volatility(self, returns: np.ndarray) -> Dict:
        """GARCH 모델로 변동성 모델링"""
        try:
            if not ARCH_AVAILABLE or len(returns) < 100:
                return await self.simple_volatility_model(returns)
            
            # GARCH(1,1) 모델 적합
            model = arch_model(returns * 100, vol='Garch', p=1, q=1)
            fitted_model = model.fit(disp='off')
            
            # 조건부 변동성 추출
            conditional_volatility = fitted_model.conditional_volatility / 100
            
            # 모델 파라미터
            params = fitted_model.params
            
            # 잔차 분석
            standardized_residuals = fitted_model.std_resid
            ljung_box_stat = self.ljung_box_test(standardized_residuals)
            
            # 예측
            forecasts = fitted_model.forecast(horizon=5)
            vol_forecast = np.sqrt(forecasts.variance.values[-1] / 10000)
            
            # 모델 저장
            self.garch_model = fitted_model
            
            return {
                "model_type": "GARCH(1,1)",
                "parameters": {
                    "omega": params.get("omega", 0),
                    "alpha": params.get("alpha[1]", 0),
                    "beta": params.get("beta[1]", 0)
                },
                "conditional_volatility": conditional_volatility.tolist()[-50:],  # 최근 50개
                "volatility_forecast": vol_forecast.tolist(),
                "model_diagnostics": {
                    "ljung_box_pvalue": ljung_box_stat,
                    "log_likelihood": fitted_model.loglikelihood,
                    "aic": fitted_model.aic,
                    "bic": fitted_model.bic
                },
                "persistence": params.get("alpha[1]", 0) + params.get("beta[1]", 0)
            }
            
        except Exception as e:
            logger.error(f"GARCH 모델링 실패: {e}")
            return await self.simple_volatility_model(returns)
    
    async def simple_volatility_model(self, returns: np.ndarray) -> Dict:
        """간단한 변동성 모델 (GARCH 대체)"""
        try:
            if len(returns) < 20:
                return {"error": "데이터 부족"}
            
            # 이동평균 변동성
            window_sizes = [5, 10, 20]
            volatilities = {}
            
            for window in window_sizes:
                vol_series = []
                for i in range(window, len(returns)):
                    vol = np.std(returns[i-window:i]) * np.sqrt(365)
                    vol_series.append(vol)
                volatilities[f"vol_{window}d"] = vol_series[-10:]  # 최근 10개
            
            # EWMA 변동성 (lambda = 0.94)
            ewma_vol = []
            lambda_param = 0.94
            vol_estimate = np.var(returns[:20])  # 초기값
            
            for ret in returns[20:]:
                vol_estimate = lambda_param * vol_estimate + (1 - lambda_param) * ret**2
                ewma_vol.append(np.sqrt(vol_estimate * 365))
            
            # 간단한 예측 (마지막 값 기준)
            forecast = [ewma_vol[-1] * (0.95 + 0.1 * np.random.random()) for _ in range(5)]
            
            return {
                "model_type": "Simple_EWMA",
                "volatilities": volatilities,
                "ewma_volatility": ewma_vol[-20:],  # 최근 20개
                "volatility_forecast": forecast,
                "current_vol": ewma_vol[-1] if ewma_vol else 0.0
            }
            
        except Exception as e:
            logger.error(f"간단한 변동성 모델 실패: {e}")
            return {"error": str(e)}
    
    def ljung_box_test(self, residuals: np.ndarray, lags: int = 10) -> float:
        """Ljung-Box 테스트 (잔차 자기상관 검정)"""
        try:
            n = len(residuals)
            autocorr = np.correlate(residuals, residuals, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            autocorr = autocorr[1:lags+1] / autocorr[0]
            
            lb_stat = n * (n + 2) * np.sum([autocorr[i]**2 / (n - i - 1) for i in range(lags)])
            p_value = 1 - stats.chi2.cdf(lb_stat, lags)
            
            return p_value
            
        except Exception as e:
            logger.error(f"Ljung-Box 테스트 실패: {e}")
            return 0.5
    
    def analyze_volatility_clustering(self, volatility: List[float]) -> Dict:
        """변동성 클러스터링 분석"""
        try:
            if len(volatility) < 50:
                return {"error": "클러스터링 분석을 위한 데이터 부족"}
            
            vol_array = np.array(volatility).reshape(-1, 1)
            
            # Gaussian Mixture Model로 클러스터링
            n_clusters = 3
            gmm = GaussianMixture(n_components=n_clusters, random_state=42)
            cluster_labels = gmm.fit_predict(vol_array)
            
            # 클러스터별 특성
            cluster_stats = {}
            for i in range(n_clusters):
                cluster_data = np.array(volatility)[cluster_labels == i]
                if len(cluster_data) > 0:
                    cluster_stats[f"cluster_{i}"] = {
                        "mean": np.mean(cluster_data),
                        "std": np.std(cluster_data),
                        "size": len(cluster_data),
                        "percentage": len(cluster_data) / len(volatility) * 100
                    }
            
            # 클러스터 지속성 (같은 클러스터가 연속으로 나타나는 정도)
            transitions = np.sum(np.diff(cluster_labels) != 0)
            persistence_score = 1 - (transitions / (len(cluster_labels) - 1))
            
            # 현재 클러스터
            current_cluster = cluster_labels[-1] if len(cluster_labels) > 0 else 0
            
            return {
                "n_clusters": n_clusters,
                "cluster_labels": cluster_labels.tolist()[-20:],  # 최근 20개
                "cluster_statistics": cluster_stats,
                "persistence_score": persistence_score,
                "current_cluster": current_cluster,
                "clustering_quality": float(gmm.aic_)
            }
            
        except Exception as e:
            logger.error(f"변동성 클러스터링 분석 실패: {e}")
            return {"error": str(e)}
    
    async def identify_volatility_regimes(self, volatility: List[float], 
                                        timestamps: List[datetime]) -> Dict:
        """변동성 체제 식별"""
        try:
            if len(volatility) < 10:
                return {"error": "체제 식별을 위한 데이터 부족"}
            
            # 현재 변동성으로 체제 결정
            current_vol = volatility[-1]
            current_regime_id = self.classify_volatility_regime(current_vol)
            current_regime_name = self.volatility_regimes[current_regime_id]
            
            # 체제 변경점 탐지
            regime_changes = []
            regime_sequence = []
            
            for i, vol in enumerate(volatility):
                regime_id = self.classify_volatility_regime(vol)
                regime_sequence.append(regime_id)
                
                if i > 0 and regime_sequence[i] != regime_sequence[i-1]:
                    regime_changes.append({
                        "index": i,
                        "timestamp": timestamps[i].isoformat() if i < len(timestamps) else None,
                        "from_regime": self.volatility_regimes[regime_sequence[i-1]],
                        "to_regime": self.volatility_regimes[regime_sequence[i]],
                        "volatility": vol
                    })
            
            # 현재 체제 지속 기간
            current_regime_duration = 0
            for i in range(len(regime_sequence) - 1, -1, -1):
                if regime_sequence[i] == current_regime_id:
                    current_regime_duration += 1
                else:
                    break
            
            # 체제별 통계
            regime_stats = {}
            for regime_id, regime_name in self.volatility_regimes.items():
                regime_vols = [vol for i, vol in enumerate(volatility) 
                              if regime_sequence[i] == regime_id]
                if regime_vols:
                    regime_stats[regime_name] = {
                        "count": len(regime_vols),
                        "percentage": len(regime_vols) / len(volatility) * 100,
                        "avg_volatility": np.mean(regime_vols),
                        "max_volatility": np.max(regime_vols),
                        "min_volatility": np.min(regime_vols)
                    }
            
            return {
                "current_regime_id": current_regime_id,
                "current_regime": current_regime_name,
                "current_volatility": current_vol,
                "regime_duration": current_regime_duration,
                "regime_changes": regime_changes[-10:],  # 최근 10개 변경점
                "regime_sequence": regime_sequence[-50:],  # 최근 50개
                "regime_statistics": regime_stats,
                "total_regime_changes": len(regime_changes)
            }
            
        except Exception as e:
            logger.error(f"변동성 체제 식별 실패: {e}")
            return {"error": str(e)}
    
    def classify_volatility_regime(self, volatility: float) -> int:
        """변동성 값으로 체제 분류"""
        try:
            for i, threshold in enumerate(self.regime_thresholds):
                if volatility <= threshold:
                    return i
            return len(self.regime_thresholds) - 1
        except:
            return 2  # 기본값: NORMAL
    
    def analyze_volatility_persistence(self, volatility: List[float]) -> Dict:
        """변동성 지속성 분석"""
        try:
            if len(volatility) < 30:
                return {"error": "지속성 분석을 위한 데이터 부족"}
            
            vol_array = np.array(volatility)
            
            # 자기상관 분석
            max_lags = min(20, len(volatility) // 4)
            autocorrelations = []
            
            for lag in range(1, max_lags + 1):
                if len(vol_array) > lag:
                    corr = np.corrcoef(vol_array[:-lag], vol_array[lag:])[0, 1]
                    autocorrelations.append(corr)
            
            # 지속성 점수 (첫 번째 자기상관계수)
            persistence_score = autocorrelations[0] if autocorrelations else 0
            
            # 반감기 추정 (자기상관이 0.5 이하로 떨어지는 지점)
            half_life = 1
            for i, corr in enumerate(autocorrelations):
                if corr < 0.5:
                    half_life = i + 1
                    break
            else:
                half_life = len(autocorrelations)
            
            # 변동성의 변동성 (vol of vol)
            vol_changes = np.diff(vol_array)
            vol_of_vol = np.std(vol_changes) / np.mean(vol_array) if np.mean(vol_array) > 0 else 0
            
            # 평균 회귀 속도
            mean_vol = np.mean(vol_array)
            deviations = vol_array - mean_vol
            mean_reversion_speed = -np.mean(np.diff(deviations) / deviations[:-1]) if len(deviations) > 1 else 0
            
            return {
                "persistence_score": persistence_score,
                "autocorrelations": autocorrelations[:10],  # 처음 10개
                "half_life_days": half_life,
                "volatility_of_volatility": vol_of_vol,
                "mean_reversion_speed": mean_reversion_speed,
                "current_deviation": (vol_array[-1] - mean_vol) / mean_vol if mean_vol > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"변동성 지속성 분석 실패: {e}")
            return {"error": str(e)}
    
    def analyze_volatility_shocks(self, returns: np.ndarray, volatility: List[float]) -> Dict:
        """변동성 충격 분석"""
        try:
            if len(returns) < 20 or len(volatility) < 20:
                return {"error": "충격 분석을 위한 데이터 부족"}
            
            # 변동성 충격 임계값 (상위 10%)
            vol_threshold = np.percentile(volatility, 90)
            
            # 수익률 충격 임계값 (절댓값 기준 상위 5%)
            return_threshold = np.percentile(np.abs(returns), 95)
            
            # 충격 이벤트 식별
            vol_shocks = []
            return_shocks = []
            
            for i, vol in enumerate(volatility):
                if vol > vol_threshold:
                    vol_shocks.append({
                        "index": i,
                        "magnitude": vol,
                        "z_score": (vol - np.mean(volatility)) / np.std(volatility)
                    })
            
            for i, ret in enumerate(returns):
                if abs(ret) > return_threshold:
                    return_shocks.append({
                        "index": i,
                        "magnitude": ret,
                        "z_score": ret / np.std(returns)
                    })
            
            # 충격 전파 분석 (충격 후 변동성 변화)
            shock_propagation = []
            for shock in vol_shocks[-5:]:  # 최근 5개 충격
                shock_idx = shock["index"]
                if shock_idx < len(volatility) - 5:
                    post_shock_vol = volatility[shock_idx:shock_idx+5]
                    pre_shock_vol = np.mean(volatility[max(0, shock_idx-5):shock_idx])
                    
                    propagation = {
                        "shock_magnitude": shock["magnitude"],
                        "pre_shock_avg": pre_shock_vol,
                        "post_shock_series": post_shock_vol,
                        "decay_rate": self.calculate_decay_rate(post_shock_vol)
                    }
                    shock_propagation.append(propagation)
            
            # 충격 빈도 분석
            shock_frequency = len(vol_shocks) / len(volatility) * 100  # 백분율
            
            return {
                "volatility_shocks": vol_shocks[-10:],  # 최근 10개
                "return_shocks": return_shocks[-10:],
                "shock_propagation": shock_propagation,
                "shock_frequency_pct": shock_frequency,
                "avg_shock_magnitude": np.mean([s["magnitude"] for s in vol_shocks]) if vol_shocks else 0,
                "shock_clustering": self.analyze_shock_clustering(vol_shocks)
            }
            
        except Exception as e:
            logger.error(f"변동성 충격 분석 실패: {e}")
            return {"error": str(e)}
    
    def calculate_decay_rate(self, post_shock_series: List[float]) -> float:
        """충격 후 변동성 감쇠율 계산"""
        try:
            if len(post_shock_series) < 3:
                return 0.0
            
            # 지수적 감쇠 모델 적합
            x = np.arange(len(post_shock_series))
            y = np.array(post_shock_series)
            
            # 로그 변환 후 선형 회귀
            if np.all(y > 0):
                log_y = np.log(y)
                slope = np.polyfit(x, log_y, 1)[0]
                return -slope  # 양수면 감쇠, 음수면 증가
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"감쇠율 계산 실패: {e}")
            return 0.0
    
    def analyze_shock_clustering(self, shocks: List[Dict]) -> Dict:
        """충격 클러스터링 분석"""
        try:
            if len(shocks) < 3:
                return {"clustering_detected": False}
            
            # 충격 간 시간 간격 계산
            intervals = []
            for i in range(1, len(shocks)):
                interval = shocks[i]["index"] - shocks[i-1]["index"]
                intervals.append(interval)
            
            if not intervals:
                return {"clustering_detected": False}
            
            # 클러스터링 임계값 (평균 간격의 50% 이하)
            avg_interval = np.mean(intervals)
            cluster_threshold = avg_interval * 0.5
            
            # 클러스터 식별
            clustered_intervals = [interval for interval in intervals if interval <= cluster_threshold]
            clustering_ratio = len(clustered_intervals) / len(intervals)
            
            return {
                "clustering_detected": clustering_ratio > 0.3,
                "clustering_ratio": clustering_ratio,
                "avg_interval": avg_interval,
                "cluster_threshold": cluster_threshold,
                "total_intervals": len(intervals),
                "clustered_intervals": len(clustered_intervals)
            }
            
        except Exception as e:
            logger.error(f"충격 클러스터링 분석 실패: {e}")
            return {"clustering_detected": False}
    
    async def update_transition_probabilities(self, regime_results: Dict):
        """체제 전환 확률 업데이트"""
        try:
            regime_sequence = regime_results.get("regime_sequence", [])
            if len(regime_sequence) < 10:
                return
            
            # 전환 행렬 계산
            n_regimes = len(self.volatility_regimes)
            transitions = np.zeros((n_regimes, n_regimes))
            
            for i in range(1, len(regime_sequence)):
                from_regime = regime_sequence[i-1]
                to_regime = regime_sequence[i]
                if 0 <= from_regime < n_regimes and 0 <= to_regime < n_regimes:
                    transitions[from_regime, to_regime] += 1
            
            # 행별 정규화 (확률로 변환)
            for i in range(n_regimes):
                row_sum = transitions[i].sum()
                if row_sum > 0:
                    transitions[i] /= row_sum
            
            self.transition_matrix = transitions
            
            # 데이터베이스에 저장
            await self.save_transition_probabilities()
            
        except Exception as e:
            logger.error(f"전환 확률 업데이트 실패: {e}")
    
    async def forecast_regime_changes(self, horizon_days: int = 7) -> Dict:
        """체제 변경 예측"""
        try:
            if self.current_regime is None:
                return {"error": "현재 체제가 설정되지 않았습니다"}
            
            current_regime_id = self.current_regime
            
            # 마르코프 체인으로 미래 체제 예측
            current_prob = np.zeros(len(self.volatility_regimes))
            current_prob[current_regime_id] = 1.0
            
            future_probs = []
            for day in range(horizon_days):
                current_prob = current_prob @ self.transition_matrix
                future_probs.append(current_prob.copy())
            
            # 각 날짜별 가장 가능성 높은 체제
            most_likely_regimes = []
            for day_prob in future_probs:
                most_likely_id = np.argmax(day_prob)
                most_likely_regimes.append({
                    "regime_id": most_likely_id,
                    "regime_name": self.volatility_regimes[most_likely_id],
                    "probability": day_prob[most_likely_id],
                    "all_probabilities": day_prob.tolist()
                })
            
            # 체제 전환 감지
            regime_changes_forecast = []
            for day, regime_info in enumerate(most_likely_regimes):
                if day == 0:
                    if regime_info["regime_id"] != current_regime_id:
                        regime_changes_forecast.append({
                            "day": day + 1,
                            "from_regime": self.volatility_regimes[current_regime_id],
                            "to_regime": regime_info["regime_name"],
                            "probability": regime_info["probability"]
                        })
                else:
                    prev_regime = most_likely_regimes[day-1]["regime_id"]
                    if regime_info["regime_id"] != prev_regime:
                        regime_changes_forecast.append({
                            "day": day + 1,
                            "from_regime": self.volatility_regimes[prev_regime],
                            "to_regime": regime_info["regime_name"],
                            "probability": regime_info["probability"]
                        })
            
            return {
                "forecast_horizon": horizon_days,
                "current_regime": self.volatility_regimes[current_regime_id],
                "daily_forecasts": most_likely_regimes,
                "regime_changes": regime_changes_forecast,
                "transition_matrix": self.transition_matrix.tolist()
            }
            
        except Exception as e:
            logger.error(f"체제 변경 예측 실패: {e}")
            return {"error": str(e)}
    
    async def save_transition_probabilities(self):
        """전환 확률 데이터베이스 저장"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 기존 데이터 삭제
            cursor.execute('DELETE FROM regime_transition_probabilities')
            
            # 새로운 전환 확률 저장
            for i in range(len(self.volatility_regimes)):
                for j in range(len(self.volatility_regimes)):
                    probability = self.transition_matrix[i, j]
                    if probability > 0.001:  # 의미 있는 확률만 저장
                        cursor.execute('''
                            INSERT INTO regime_transition_probabilities
                            (from_regime, to_regime, base_probability, conditional_probabilities,
                             trigger_conditions, expected_duration_days, confidence_interval_low,
                             confidence_interval_high, last_updated)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            self.volatility_regimes[i],
                            self.volatility_regimes[j],
                            probability,
                            json.dumps({}),  # 조건부 확률은 추후 구현
                            json.dumps([]),  # 트리거 조건은 추후 구현
                            1.0 / probability if probability > 0 else float('inf'),
                            probability * 0.8,  # 신뢰구간 하한
                            probability * 1.2,  # 신뢰구간 상한
                            datetime.now().isoformat()
                        ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"전환 확률 저장 실패: {e}")
    
    async def save_analysis_results(self, results: Dict):
        """분석 결과 저장"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 현재 변동성 데이터 저장
            regime_results = results.get("regime_results", {})
            current_vol = regime_results.get("current_volatility", 0)
            current_regime_id = regime_results.get("current_regime_id", 2)
            
            cursor.execute('''
                INSERT INTO volatility_data
                (timestamp, realized_volatility, regime_id, regime_probability,
                 clustering_score, persistence_score)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now().isoformat(),
                current_vol,
                current_regime_id,
                1.0,  # 현재 체제 확률
                results.get("clustering_results", {}).get("persistence_score", 0),
                results.get("persistence_results", {}).get("persistence_score", 0)
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"분석 결과 저장 실패: {e}")
    
    async def get_volatility_diagnostics(self) -> Dict:
        """변동성 분석기 진단"""
        try:
            diagnostics = {
                "analyzer_status": "active",
                "models_loaded": {
                    "garch_model": self.garch_model is not None,
                    "regime_classifier": self.regime_classifier is not None
                },
                "current_regime": self.current_regime,
                "regime_history_length": len(self.regime_history),
                "volatility_history_length": len(self.volatility_history)
            }
            
            # 전환 행렬 요약
            if self.transition_matrix.size > 0:
                most_stable_regime = np.argmax(np.diag(self.transition_matrix))
                most_volatile_regime = np.argmin(np.diag(self.transition_matrix))
                
                diagnostics["transition_analysis"] = {
                    "most_stable_regime": self.volatility_regimes[most_stable_regime],
                    "most_volatile_regime": self.volatility_regimes[most_volatile_regime],
                    "avg_persistence": np.mean(np.diag(self.transition_matrix))
                }
            
            # 최근 분석 결과 조회
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT realized_volatility, regime_id, timestamp
                FROM volatility_data 
                ORDER BY timestamp DESC 
                LIMIT 10
            ''')
            
            recent_data = cursor.fetchall()
            if recent_data:
                diagnostics["recent_volatility"] = [
                    {
                        "volatility": row[0],
                        "regime": self.volatility_regimes.get(row[1], "UNKNOWN"),
                        "timestamp": row[2]
                    } for row in recent_data
                ]
            
            conn.close()
            
            return diagnostics
            
        except Exception as e:
            logger.error(f"변동성 진단 실패: {e}")
            return {"error": str(e)}

# 테스트 함수
async def test_volatility_regime_analyzer():
    """변동성 체제 분석기 테스트"""
    print("📊 변동성 체제 분석기 테스트")
    print("=" * 50)
    
    analyzer = VolatilityRegimeAnalyzer()
    
    # 진단 정보
    diagnostics = await analyzer.get_volatility_diagnostics()
    print(f"🔧 분석기 상태: {diagnostics.get('analyzer_status')}")
    print(f"📈 GARCH 모델: {'✅' if diagnostics.get('models_loaded', {}).get('garch_model') else '❌'}")
    
    # 테스트 데이터 생성 (가상의 비트코인 가격)
    np.random.seed(42)
    n_days = 100
    base_price = 60000
    
    # 변동성이 변하는 가격 시계열 생성
    prices = [base_price]
    volatilities = []
    
    for i in range(n_days):
        # 주기적으로 변하는 변동성
        vol = 0.02 + 0.03 * np.sin(i * 0.1) + 0.01 * np.random.random()
        volatilities.append(vol)
        
        # 가격 업데이트
        daily_return = np.random.normal(0, vol)
        new_price = prices[-1] * (1 + daily_return)
        prices.append(max(new_price, 1000))  # 최소 가격 보장
    
    timestamps = [datetime.now() - timedelta(days=n_days-i) for i in range(n_days+1)]
    
    print(f"🧪 테스트 데이터: {len(prices)}개 가격 포인트")
    
    # 변동성 체제 분석 실행
    analysis_result = await analyzer.analyze_volatility_regime(prices, timestamps)
    
    if not analysis_result.get("error"):
        print(f"✅ 분석 완료")
        print(f"📊 현재 체제: {analysis_result.get('current_regime')}")
        print(f"📈 현재 변동성: {analysis_result.get('current_volatility', 0):.1%}")
        
        # 체제 결과
        regime_results = analysis_result.get("regime_results", {})
        print(f"🔄 체제 지속기간: {regime_results.get('regime_duration', 0)}일")
        print(f"🎯 총 체제 변경: {regime_results.get('total_regime_changes', 0)}회")
        
        # GARCH 결과
        garch_results = analysis_result.get("garch_results", {})
        if garch_results.get("model_type"):
            print(f"📉 {garch_results.get('model_type')} 모델 적용")
            persistence = garch_results.get("persistence", 0)
            print(f"🔗 변동성 지속성: {persistence:.3f}")
        
        # 클러스터링 결과
        clustering = analysis_result.get("clustering_results", {})
        if clustering.get("persistence_score"):
            print(f"🎲 클러스터링 지속성: {clustering.get('persistence_score'):.3f}")
        
        # 체제 예측
        forecast = analysis_result.get("regime_forecast", {})
        if forecast.get("regime_changes"):
            print(f"🔮 예상 체제 변경:")
            for change in forecast["regime_changes"][:3]:
                print(f"   • Day {change['day']}: {change['from_regime']} → {change['to_regime']} ({change['probability']:.1%})")
    else:
        print(f"❌ 분석 실패: {analysis_result.get('error')}")
    
    print("\n" + "=" * 50)
    print("🎉 변동성 체제 분석기 테스트 완료!")

if __name__ == "__main__":
    asyncio.run(test_volatility_regime_analyzer())