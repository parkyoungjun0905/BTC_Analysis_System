#!/usr/bin/env python3
"""
체제별 적응형 모델 선택 및 가중치 시스템
각 시장 체제에 최적화된 예측 모델을 동적으로 선택하고 가중치를 조정

핵심 기능:
1. 체제별 최적 모델 자동 선택
2. 실시간 성능 기반 가중치 조정
3. 체제 전환시 모델 스위칭
4. 다중 모델 앙상블 최적화
5. 성능 기반 모델 순위 업데이트
"""

import os
import json
import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from dataclasses import dataclass, asdict
import asyncio
import pickle
from collections import defaultdict, deque
import statistics
from abc import ABC, abstractmethod
import warnings
warnings.filterwarnings("ignore")

# 성능 최적화
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelPerformanceMetrics:
    """모델 성능 지표"""
    model_id: str
    regime_type: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    directional_accuracy: float
    profit_factor: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    avg_return: float
    volatility_adjusted_return: float
    sample_size: int
    last_updated: datetime

@dataclass
class RegimeModelConfig:
    """체제별 모델 설정"""
    regime_type: str
    primary_model: str
    fallback_models: List[str]
    ensemble_weights: Dict[str, float]
    performance_threshold: float
    retraining_interval_days: int
    feature_importance_weights: Dict[str, float]
    risk_adjustment_factor: float

@dataclass
class PredictionResult:
    """예측 결과"""
    regime_type: str
    prediction: Dict[str, Any]
    model_used: str
    confidence: float
    contributing_models: Dict[str, float]
    performance_score: float
    timestamp: datetime

class BaseRegimeModel(ABC):
    """체제별 모델 기본 클래스"""
    
    def __init__(self, model_id: str, regime_type: str):
        self.model_id = model_id
        self.regime_type = regime_type
        self.performance_history = deque(maxlen=1000)
        self.is_trained = False
        self.last_training_date = None
        
    @abstractmethod
    async def train(self, training_data: List[Dict]) -> Dict:
        """모델 학습"""
        pass
    
    @abstractmethod
    async def predict(self, features: np.ndarray) -> Dict:
        """예측 수행"""
        pass
    
    @abstractmethod
    def get_feature_importance(self) -> Dict[str, float]:
        """특징 중요도 반환"""
        pass
    
    def update_performance(self, actual: float, predicted: float, metadata: Dict = None):
        """성능 업데이트"""
        self.performance_history.append({
            'timestamp': datetime.now(),
            'actual': actual,
            'predicted': predicted,
            'error': abs(actual - predicted),
            'direction_correct': (actual - predicted) * (predicted - actual) >= 0,
            'metadata': metadata or {}
        })

class TrendFollowingModel(BaseRegimeModel):
    """트렌드 추종 모델 (강세장 특화)"""
    
    def __init__(self, model_id: str = "trend_following"):
        super().__init__(model_id, "BULL_MARKET")
        self.momentum_weights = {
            'price_trend_7d': 0.25,
            'price_trend_30d': 0.20,
            'rsi_14': 0.15,
            'volume_trend': 0.15,
            'macd_signal': 0.10,
            'bollinger_position': 0.10,
            'fear_greed_index': 0.05
        }
    
    async def train(self, training_data: List[Dict]) -> Dict:
        """트렌드 추종 모델 학습"""
        try:
            # 강세장 데이터만 필터링
            bull_data = [d for d in training_data if d.get('regime') == 'BULL_MARKET']
            
            if len(bull_data) < 20:
                return {"error": "강세장 학습 데이터 부족"}
            
            # 모멘텀 기반 특징 가중치 최적화
            features_array = []
            targets_array = []
            
            for data in bull_data:
                features = data.get('features', [])
                target = data.get('target')
                
                if len(features) >= 7 and target is not None:
                    # 모멘텀 특징만 추출
                    momentum_features = [
                        features[1],  # price_trend_7d
                        features[2],  # price_trend_30d  
                        features[11], # rsi_14
                        features[8],  # volume_trend
                        features[12], # macd_signal
                        features[13], # bollinger_position
                        features[20]  # fear_greed_index
                    ]
                    features_array.append(momentum_features)
                    targets_array.append(target)
            
            if len(features_array) < 10:
                return {"error": "유효한 특징 데이터 부족"}
            
            features_array = np.array(features_array)
            targets_array = np.array(targets_array)
            
            # 선형 회귀로 최적 가중치 계산
            from sklearn.linear_model import Ridge
            model = Ridge(alpha=0.1)
            model.fit(features_array, targets_array)
            
            # 가중치 업데이트
            feature_names = list(self.momentum_weights.keys())
            for i, feature in enumerate(feature_names):
                if i < len(model.coef_):
                    self.momentum_weights[feature] = abs(model.coef_[i])
            
            # 정규화
            total_weight = sum(self.momentum_weights.values())
            if total_weight > 0:
                self.momentum_weights = {k: v/total_weight for k, v in self.momentum_weights.items()}
            
            self.is_trained = True
            self.last_training_date = datetime.now()
            
            # 성능 평가
            predictions = model.predict(features_array)
            accuracy = np.mean(np.abs(predictions - targets_array) < 0.02)  # 2% 이내 정확도
            
            return {
                "training_completed": True,
                "samples": len(features_array),
                "accuracy": accuracy,
                "optimal_weights": self.momentum_weights
            }
            
        except Exception as e:
            logger.error(f"트렌드 추종 모델 학습 실패: {e}")
            return {"error": str(e)}
    
    async def predict(self, features: np.ndarray) -> Dict:
        """트렌드 추종 예측"""
        try:
            if not self.is_trained or len(features) < 24:
                return {"error": "모델 미학습 또는 특징 부족"}
            
            # 모멘텀 특징 추출
            momentum_features = {
                'price_trend_7d': features[1],
                'price_trend_30d': features[2], 
                'rsi_14': features[11],
                'volume_trend': features[8],
                'macd_signal': features[12],
                'bollinger_position': features[13],
                'fear_greed_index': features[20]
            }
            
            # 가중 점수 계산
            momentum_score = sum(
                momentum_features[feature] * weight 
                for feature, weight in self.momentum_weights.items()
                if feature in momentum_features
            )
            
            # 트렌드 강도 분석
            trend_strength = (momentum_features['price_trend_7d'] + momentum_features['price_trend_30d']) / 2
            
            # 예측 신호 생성
            if momentum_score > 0.3 and trend_strength > 0.02:
                direction = "BULLISH"
                confidence = min(momentum_score * 2, 1.0)
                price_change_prediction = trend_strength * 1.2  # 트렌드 증폭
            elif momentum_score > 0.1:
                direction = "MILD_BULLISH"
                confidence = momentum_score
                price_change_prediction = trend_strength * 0.8
            else:
                direction = "NEUTRAL"
                confidence = 0.5
                price_change_prediction = 0.0
            
            return {
                "direction": direction,
                "confidence": confidence,
                "price_change_prediction": price_change_prediction,
                "momentum_score": momentum_score,
                "trend_strength": trend_strength,
                "model_type": "trend_following"
            }
            
        except Exception as e:
            logger.error(f"트렌드 추종 예측 실패: {e}")
            return {"error": str(e)}
    
    def get_feature_importance(self) -> Dict[str, float]:
        """특징 중요도 반환"""
        return self.momentum_weights.copy()

class MeanReversionModel(BaseRegimeModel):
    """평균 회귀 모델 (약세장/횡보장 특화)"""
    
    def __init__(self, model_id: str = "mean_reversion"):
        super().__init__(model_id, "BEAR_MARKET")
        self.reversion_weights = {
            'rsi_14': 0.30,
            'bollinger_position': 0.25,
            'fear_greed_index': 0.20,
            'whale_activity': 0.10,
            'exchange_flow': 0.10,
            'put_call_ratio': 0.05
        }
    
    async def train(self, training_data: List[Dict]) -> Dict:
        """평균 회귀 모델 학습"""
        try:
            # 약세장/횡보장 데이터 필터링
            reversion_data = [d for d in training_data 
                            if d.get('regime') in ['BEAR_MARKET', 'SIDEWAYS']]
            
            if len(reversion_data) < 20:
                return {"error": "평균 회귀 학습 데이터 부족"}
            
            features_array = []
            targets_array = []
            
            for data in reversion_data:
                features = data.get('features', [])
                target = data.get('target')
                
                if len(features) >= 21 and target is not None:
                    reversion_features = [
                        features[11],  # rsi_14
                        features[13],  # bollinger_position
                        features[20],  # fear_greed_index
                        features[14],  # whale_activity
                        features[15],  # exchange_flow
                        features[19] if len(features) > 19 else 0.5  # put_call_ratio
                    ]
                    features_array.append(reversion_features)
                    targets_array.append(target)
            
            if len(features_array) < 10:
                return {"error": "유효한 특징 데이터 부족"}
            
            features_array = np.array(features_array)
            targets_array = np.array(targets_array)
            
            # 평균 회귀 최적화
            from sklearn.linear_model import ElasticNet
            model = ElasticNet(alpha=0.1, l1_ratio=0.5)
            model.fit(features_array, targets_array)
            
            # 가중치 업데이트
            feature_names = list(self.reversion_weights.keys())
            for i, feature in enumerate(feature_names):
                if i < len(model.coef_):
                    self.reversion_weights[feature] = abs(model.coef_[i])
            
            # 정규화
            total_weight = sum(self.reversion_weights.values())
            if total_weight > 0:
                self.reversion_weights = {k: v/total_weight for k, v in self.reversion_weights.items()}
            
            self.is_trained = True
            self.last_training_date = datetime.now()
            
            predictions = model.predict(features_array)
            accuracy = np.mean(np.abs(predictions - targets_array) < 0.03)
            
            return {
                "training_completed": True,
                "samples": len(features_array),
                "accuracy": accuracy,
                "optimal_weights": self.reversion_weights
            }
            
        except Exception as e:
            logger.error(f"평균 회귀 모델 학습 실패: {e}")
            return {"error": str(e)}
    
    async def predict(self, features: np.ndarray) -> Dict:
        """평균 회귀 예측"""
        try:
            if not self.is_trained or len(features) < 21:
                return {"error": "모델 미학습 또는 특징 부족"}
            
            # 평균 회귀 특징 추출
            reversion_features = {
                'rsi_14': features[11],
                'bollinger_position': features[13],
                'fear_greed_index': features[20],
                'whale_activity': features[14],
                'exchange_flow': features[15],
                'put_call_ratio': features[19] if len(features) > 19 else 0.5
            }
            
            # 과매수/과매도 분석
            rsi = reversion_features['rsi_14']
            bb_pos = reversion_features['bollinger_position']
            fear_greed = reversion_features['fear_greed_index']
            
            # 평균 회귀 신호 생성
            oversold_score = 0
            overbought_score = 0
            
            # RSI 기반
            if rsi < 30:
                oversold_score += 0.4
            elif rsi > 70:
                overbought_score += 0.4
            
            # 볼린저밴드 기반
            if bb_pos < 0.2:
                oversold_score += 0.3
            elif bb_pos > 0.8:
                overbought_score += 0.3
            
            # 공포탐욕지수 기반
            if fear_greed < 25:
                oversold_score += 0.3
            elif fear_greed > 75:
                overbought_score += 0.3
            
            # 예측 생성
            if oversold_score > 0.6:
                direction = "BULLISH_REVERSAL"
                confidence = oversold_score
                price_change_prediction = 0.02 * oversold_score
            elif overbought_score > 0.6:
                direction = "BEARISH_REVERSAL"
                confidence = overbought_score
                price_change_prediction = -0.02 * overbought_score
            else:
                direction = "NEUTRAL"
                confidence = 0.5
                price_change_prediction = 0.0
            
            return {
                "direction": direction,
                "confidence": confidence,
                "price_change_prediction": price_change_prediction,
                "oversold_score": oversold_score,
                "overbought_score": overbought_score,
                "model_type": "mean_reversion"
            }
            
        except Exception as e:
            logger.error(f"평균 회귀 예측 실패: {e}")
            return {"error": str(e)}
    
    def get_feature_importance(self) -> Dict[str, float]:
        return self.reversion_weights.copy()

class VolatilityBreakoutModel(BaseRegimeModel):
    """변동성 돌파 모델 (고변동성/저변동성 특화)"""
    
    def __init__(self, model_id: str = "volatility_breakout"):
        super().__init__(model_id, "HIGH_VOLATILITY_SHOCK")
        self.volatility_weights = {
            'volatility_1d': 0.25,
            'volatility_7d': 0.20,
            'volume_volatility': 0.20,
            'futures_basis': 0.15,
            'funding_rate': 0.10,
            'whale_activity': 0.10
        }
    
    async def train(self, training_data: List[Dict]) -> Dict:
        """변동성 모델 학습"""
        try:
            # 고변동성/저변동성 데이터 필터링
            vol_data = [d for d in training_data 
                       if d.get('regime') in ['HIGH_VOLATILITY_SHOCK', 'LOW_VOLATILITY_ACCUMULATION']]
            
            if len(vol_data) < 15:
                return {"error": "변동성 학습 데이터 부족"}
            
            features_array = []
            targets_array = []
            
            for data in vol_data:
                features = data.get('features', [])
                target = data.get('target')
                
                if len(features) >= 19 and target is not None:
                    vol_features = [
                        features[4],   # volatility_1d
                        features[5],   # volatility_7d
                        features[9],   # volume_volatility
                        features[17],  # futures_basis
                        features[18],  # funding_rate
                        features[14]   # whale_activity
                    ]
                    features_array.append(vol_features)
                    targets_array.append(target)
            
            if len(features_array) < 10:
                return {"error": "유효한 특징 데이터 부족"}
            
            features_array = np.array(features_array)
            targets_array = np.array(targets_array)
            
            # 비선형 모델 (RandomForest) 사용
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
            model.fit(features_array, targets_array)
            
            # 특징 중요도 업데이트
            feature_names = list(self.volatility_weights.keys())
            importances = model.feature_importances_
            
            for i, feature in enumerate(feature_names):
                if i < len(importances):
                    self.volatility_weights[feature] = importances[i]
            
            self.is_trained = True
            self.last_training_date = datetime.now()
            
            predictions = model.predict(features_array)
            accuracy = np.mean(np.abs(predictions - targets_array) < 0.04)
            
            return {
                "training_completed": True,
                "samples": len(features_array),
                "accuracy": accuracy,
                "optimal_weights": self.volatility_weights
            }
            
        except Exception as e:
            logger.error(f"변동성 모델 학습 실패: {e}")
            return {"error": str(e)}
    
    async def predict(self, features: np.ndarray) -> Dict:
        """변동성 돌파 예측"""
        try:
            if not self.is_trained or len(features) < 19:
                return {"error": "모델 미학습 또는 특징 부족"}
            
            # 변동성 특징 추출
            vol_1d = features[4]
            vol_7d = features[5]
            vol_vol = features[9]
            futures_basis = features[17]
            funding_rate = features[18]
            whale_activity = features[14]
            
            # 변동성 체제 분석
            vol_regime_score = vol_1d * 0.4 + vol_7d * 0.3 + vol_vol * 0.3
            
            # 돌파 신호 감지
            basis_signal = abs(futures_basis) > 0.02
            funding_signal = abs(funding_rate) > 0.01
            whale_signal = whale_activity > 0.7
            
            breakout_signals = sum([basis_signal, funding_signal, whale_signal])
            
            # 예측 생성
            if vol_regime_score > 0.06 and breakout_signals >= 2:
                direction = "HIGH_VOLATILITY_BREAKOUT"
                confidence = min(vol_regime_score * 15, 1.0)
                price_change_prediction = np.sign(futures_basis + funding_rate) * 0.05
            elif vol_regime_score < 0.015 and whale_signal:
                direction = "LOW_VOLATILITY_ACCUMULATION"
                confidence = 0.7
                price_change_prediction = 0.01  # 작은 상승 기대
            else:
                direction = "NEUTRAL_VOLATILITY"
                confidence = 0.5
                price_change_prediction = 0.0
            
            return {
                "direction": direction,
                "confidence": confidence,
                "price_change_prediction": price_change_prediction,
                "volatility_regime_score": vol_regime_score,
                "breakout_signals": breakout_signals,
                "model_type": "volatility_breakout"
            }
            
        except Exception as e:
            logger.error(f"변동성 돌파 예측 실패: {e}")
            return {"error": str(e)}
    
    def get_feature_importance(self) -> Dict[str, float]:
        return self.volatility_weights.copy()

class AdaptiveRegimeModelSelector:
    """적응형 체제별 모델 선택기"""
    
    def __init__(self, base_path: str = "/Users/parkyoungjun/Desktop/BTC_Analysis_System"):
        self.base_path = base_path
        self.db_path = os.path.join(base_path, "adaptive_regime_db.db")
        self.models_path = os.path.join(base_path, "adaptive_models")
        os.makedirs(self.models_path, exist_ok=True)
        
        # 체제별 모델 등록
        self.regime_models = {
            "BULL_MARKET": [
                TrendFollowingModel(),
                VolatilityBreakoutModel("volatility_bull")
            ],
            "BEAR_MARKET": [
                MeanReversionModel(),
                TrendFollowingModel("trend_bear")
            ],
            "SIDEWAYS": [
                MeanReversionModel("mean_sideways"),
                VolatilityBreakoutModel("vol_sideways")
            ],
            "HIGH_VOLATILITY_SHOCK": [
                VolatilityBreakoutModel(),
                MeanReversionModel("mean_shock")
            ],
            "LOW_VOLATILITY_ACCUMULATION": [
                VolatilityBreakoutModel("vol_accumulation"),
                TrendFollowingModel("trend_accumulation")
            ]
        }
        
        # 체제별 설정
        self.regime_configs = {}
        self.init_regime_configs()
        
        # 성능 추적
        self.performance_tracker = {}
        self.model_rankings = {}
        
        # 동적 가중치
        self.dynamic_weights = defaultdict(lambda: defaultdict(float))
        
        # 예측 히스토리
        self.prediction_history = deque(maxlen=1000)
        
        self.init_database()
        self.load_configurations()
        
    def init_database(self):
        """데이터베이스 초기화"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 적응형 예측 기록
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS adaptive_predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    regime_type TEXT NOT NULL,
                    model_used TEXT NOT NULL,
                    prediction_data TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    contributing_models TEXT NOT NULL,
                    performance_score REAL,
                    actual_result REAL,
                    is_verified BOOLEAN DEFAULT FALSE
                )
            ''')
            
            # 모델 성능 추적
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS model_performance_tracking (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_id TEXT NOT NULL,
                    regime_type TEXT NOT NULL,
                    performance_period TEXT NOT NULL,
                    accuracy REAL NOT NULL,
                    precision_val REAL NOT NULL,
                    recall_val REAL NOT NULL,
                    f1_score REAL NOT NULL,
                    directional_accuracy REAL NOT NULL,
                    profit_factor REAL NOT NULL,
                    sharpe_ratio REAL,
                    max_drawdown REAL,
                    sample_size INTEGER NOT NULL,
                    last_updated TEXT NOT NULL
                )
            ''')
            
            # 동적 가중치 히스토리
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS dynamic_weights_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    regime_type TEXT NOT NULL,
                    model_weights TEXT NOT NULL,
                    performance_trigger TEXT,
                    adjustment_reason TEXT NOT NULL
                )
            ''')
            
            # 체제별 설정
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS regime_configurations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    regime_type TEXT NOT NULL,
                    primary_model TEXT NOT NULL,
                    fallback_models TEXT NOT NULL,
                    ensemble_weights TEXT NOT NULL,
                    performance_threshold REAL NOT NULL,
                    retraining_interval INTEGER NOT NULL,
                    risk_adjustment_factor REAL NOT NULL,
                    last_updated TEXT NOT NULL,
                    is_active BOOLEAN DEFAULT TRUE
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("✅ 적응형 모델 선택기 데이터베이스 초기화 완료")
            
        except Exception as e:
            logger.error(f"적응형 데이터베이스 초기화 실패: {e}")
    
    def init_regime_configs(self):
        """체제별 기본 설정 초기화"""
        try:
            self.regime_configs = {
                "BULL_MARKET": RegimeModelConfig(
                    regime_type="BULL_MARKET",
                    primary_model="trend_following",
                    fallback_models=["volatility_bull"],
                    ensemble_weights={"trend_following": 0.7, "volatility_bull": 0.3},
                    performance_threshold=0.65,
                    retraining_interval_days=7,
                    feature_importance_weights={},
                    risk_adjustment_factor=0.8
                ),
                "BEAR_MARKET": RegimeModelConfig(
                    regime_type="BEAR_MARKET",
                    primary_model="mean_reversion",
                    fallback_models=["trend_bear"],
                    ensemble_weights={"mean_reversion": 0.8, "trend_bear": 0.2},
                    performance_threshold=0.60,
                    retraining_interval_days=5,
                    feature_importance_weights={},
                    risk_adjustment_factor=1.2
                ),
                "SIDEWAYS": RegimeModelConfig(
                    regime_type="SIDEWAYS",
                    primary_model="mean_sideways",
                    fallback_models=["vol_sideways"],
                    ensemble_weights={"mean_sideways": 0.6, "vol_sideways": 0.4},
                    performance_threshold=0.55,
                    retraining_interval_days=10,
                    feature_importance_weights={},
                    risk_adjustment_factor=1.0
                ),
                "HIGH_VOLATILITY_SHOCK": RegimeModelConfig(
                    regime_type="HIGH_VOLATILITY_SHOCK",
                    primary_model="volatility_breakout",
                    fallback_models=["mean_shock"],
                    ensemble_weights={"volatility_breakout": 0.9, "mean_shock": 0.1},
                    performance_threshold=0.50,
                    retraining_interval_days=3,
                    feature_importance_weights={},
                    risk_adjustment_factor=1.5
                ),
                "LOW_VOLATILITY_ACCUMULATION": RegimeModelConfig(
                    regime_type="LOW_VOLATILITY_ACCUMULATION",
                    primary_model="vol_accumulation",
                    fallback_models=["trend_accumulation"],
                    ensemble_weights={"vol_accumulation": 0.7, "trend_accumulation": 0.3},
                    performance_threshold=0.60,
                    retraining_interval_days=14,
                    feature_importance_weights={},
                    risk_adjustment_factor=0.9
                )
            }
            
            logger.info("✅ 체제별 기본 설정 초기화 완료")
            
        except Exception as e:
            logger.error(f"체제별 설정 초기화 실패: {e}")
    
    def load_configurations(self):
        """저장된 설정 로드"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT regime_type, primary_model, fallback_models, ensemble_weights,
                       performance_threshold, retraining_interval, risk_adjustment_factor
                FROM regime_configurations 
                WHERE is_active = TRUE
            ''')
            
            configurations = cursor.fetchall()
            
            for config in configurations:
                regime_type = config[0]
                if regime_type in self.regime_configs:
                    self.regime_configs[regime_type].primary_model = config[1]
                    self.regime_configs[regime_type].fallback_models = json.loads(config[2])
                    self.regime_configs[regime_type].ensemble_weights = json.loads(config[3])
                    self.regime_configs[regime_type].performance_threshold = config[4]
                    self.regime_configs[regime_type].retraining_interval_days = config[5]
                    self.regime_configs[regime_type].risk_adjustment_factor = config[6]
            
            conn.close()
            logger.info("✅ 저장된 설정 로드 완료")
            
        except Exception as e:
            logger.error(f"설정 로드 실패: {e}")
    
    async def train_regime_models(self, training_data: List[Dict]) -> Dict:
        """모든 체제별 모델 학습"""
        try:
            logger.info(f"🧠 체제별 모델 학습 시작 (데이터: {len(training_data)}개)")
            
            training_results = {}
            
            for regime_type, models in self.regime_models.items():
                logger.info(f"📚 {regime_type} 모델들 학습 중...")
                
                regime_results = {}
                for model in models:
                    result = await model.train(training_data)
                    regime_results[model.model_id] = result
                
                training_results[regime_type] = regime_results
                
                # 성능 기반 가중치 업데이트
                await self.update_ensemble_weights(regime_type, regime_results)
            
            # 학습 결과 저장
            await self.save_training_results(training_results)
            
            return {
                "training_completed": True,
                "regimes_trained": len(training_results),
                "total_models": sum(len(models) for models in self.regime_models.values()),
                "results": training_results
            }
            
        except Exception as e:
            logger.error(f"체제별 모델 학습 실패: {e}")
            return {"error": str(e)}
    
    async def update_ensemble_weights(self, regime_type: str, training_results: Dict):
        """성능 기반 앙상블 가중치 업데이트"""
        try:
            if regime_type not in self.regime_configs:
                return
            
            # 모델별 성능 점수 계산
            performance_scores = {}
            for model_id, result in training_results.items():
                if isinstance(result, dict) and not result.get("error"):
                    accuracy = result.get("accuracy", 0)
                    samples = result.get("samples", 0)
                    
                    # 성능 점수 (정확도 + 샘플 가중치)
                    score = accuracy * (1 + min(samples / 100, 0.5))
                    performance_scores[model_id] = score
            
            if not performance_scores:
                return
            
            # 가중치 정규화
            total_score = sum(performance_scores.values())
            if total_score > 0:
                new_weights = {
                    model_id: score / total_score 
                    for model_id, score in performance_scores.items()
                }
                
                self.regime_configs[regime_type].ensemble_weights = new_weights
                
                # 동적 가중치 업데이트
                self.dynamic_weights[regime_type] = new_weights
                
                logger.info(f"✅ {regime_type} 앙상블 가중치 업데이트: {new_weights}")
            
        except Exception as e:
            logger.error(f"앙상블 가중치 업데이트 실패: {e}")
    
    async def predict_with_adaptive_selection(self, regime_type: str, features: np.ndarray) -> PredictionResult:
        """적응형 모델 선택으로 예측"""
        try:
            if regime_type not in self.regime_models:
                raise ValueError(f"지원하지 않는 체제 유형: {regime_type}")
            
            models = self.regime_models[regime_type]
            config = self.regime_configs[regime_type]
            
            # 개별 모델 예측
            model_predictions = {}
            model_confidences = {}
            
            for model in models:
                try:
                    prediction = await model.predict(features)
                    if not prediction.get("error"):
                        model_predictions[model.model_id] = prediction
                        model_confidences[model.model_id] = prediction.get("confidence", 0.5)
                except Exception as model_error:
                    logger.warning(f"모델 {model.model_id} 예측 실패: {model_error}")
                    continue
            
            if not model_predictions:
                raise ValueError("모든 모델 예측이 실패했습니다")
            
            # 성능 기반 모델 선택
            best_model_id = self.select_best_performing_model(regime_type, model_confidences)
            primary_prediction = model_predictions.get(best_model_id)
            
            if not primary_prediction:
                # Fallback: 가장 높은 신뢰도 모델 사용
                best_model_id = max(model_confidences.items(), key=lambda x: x[1])[0]
                primary_prediction = model_predictions[best_model_id]
            
            # 앙상블 예측 (가중 평균)
            ensemble_prediction = self.create_ensemble_prediction(
                model_predictions, config.ensemble_weights, regime_type
            )
            
            # 최종 예측 결정
            if ensemble_prediction.get("confidence", 0) > primary_prediction.get("confidence", 0):
                final_prediction = ensemble_prediction
                model_used = "ensemble"
            else:
                final_prediction = primary_prediction  
                model_used = best_model_id
            
            # 성능 점수 계산
            performance_score = self.calculate_prediction_performance_score(
                final_prediction, model_predictions, regime_type
            )
            
            # 기여 모델 가중치
            contributing_models = config.ensemble_weights.copy()
            
            result = PredictionResult(
                regime_type=regime_type,
                prediction=final_prediction,
                model_used=model_used,
                confidence=final_prediction.get("confidence", 0.5),
                contributing_models=contributing_models,
                performance_score=performance_score,
                timestamp=datetime.now()
            )
            
            # 예측 히스토리 업데이트
            self.prediction_history.append(result)
            
            # 결과 저장
            await self.save_prediction_result(result)
            
            return result
            
        except Exception as e:
            logger.error(f"적응형 예측 실패: {e}")
            # 기본 예측 반환
            return PredictionResult(
                regime_type=regime_type,
                prediction={"direction": "NEUTRAL", "confidence": 0.5, "error": str(e)},
                model_used="fallback",
                confidence=0.5,
                contributing_models={},
                performance_score=0.5,
                timestamp=datetime.now()
            )
    
    def select_best_performing_model(self, regime_type: str, model_confidences: Dict[str, float]) -> str:
        """성능 기반 최고 모델 선택"""
        try:
            # 최근 성능 히스토리 확인
            recent_performance = self.get_recent_model_performance(regime_type)
            
            if recent_performance:
                # 성능 점수와 신뢰도 조합
                combined_scores = {}
                for model_id, confidence in model_confidences.items():
                    performance = recent_performance.get(model_id, 0.5)
                    combined_score = 0.6 * performance + 0.4 * confidence
                    combined_scores[model_id] = combined_score
                
                return max(combined_scores.items(), key=lambda x: x[1])[0]
            else:
                # 성능 히스토리가 없으면 신뢰도 기준
                return max(model_confidences.items(), key=lambda x: x[1])[0]
                
        except Exception as e:
            logger.error(f"최고 모델 선택 실패: {e}")
            return list(model_confidences.keys())[0] if model_confidences else "default"
    
    def get_recent_model_performance(self, regime_type: str, days: int = 7) -> Dict[str, float]:
        """최근 모델 성능 조회"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
            
            cursor.execute('''
                SELECT model_id, AVG(accuracy) as avg_accuracy
                FROM model_performance_tracking 
                WHERE regime_type = ? AND last_updated > ?
                GROUP BY model_id
            ''', (regime_type, cutoff_date))
            
            results = cursor.fetchall()
            conn.close()
            
            return {model_id: accuracy for model_id, accuracy in results}
            
        except Exception as e:
            logger.error(f"모델 성능 조회 실패: {e}")
            return {}
    
    def create_ensemble_prediction(self, model_predictions: Dict, weights: Dict, regime_type: str) -> Dict:
        """앙상블 예측 생성"""
        try:
            if not model_predictions or not weights:
                return {"error": "예측 또는 가중치 데이터 없음"}
            
            # 방향성 투표
            direction_votes = {}
            confidence_weighted = 0
            price_change_weighted = 0
            total_weight = 0
            
            for model_id, prediction in model_predictions.items():
                weight = weights.get(model_id, 0)
                if weight <= 0:
                    continue
                
                direction = prediction.get("direction", "NEUTRAL")
                confidence = prediction.get("confidence", 0.5)
                price_change = prediction.get("price_change_prediction", 0)
                
                # 방향 투표 집계
                direction_votes[direction] = direction_votes.get(direction, 0) + weight
                
                # 가중 평균 계산
                confidence_weighted += confidence * weight
                price_change_weighted += price_change * weight
                total_weight += weight
            
            if total_weight == 0:
                return {"error": "유효한 가중치 없음"}
            
            # 최종 결과
            final_direction = max(direction_votes.items(), key=lambda x: x[1])[0] if direction_votes else "NEUTRAL"
            final_confidence = confidence_weighted / total_weight
            final_price_change = price_change_weighted / total_weight
            
            # 앙상블 보정
            ensemble_confidence = min(final_confidence * 1.1, 1.0)  # 앙상블 보너스
            
            return {
                "direction": final_direction,
                "confidence": ensemble_confidence,
                "price_change_prediction": final_price_change,
                "direction_votes": direction_votes,
                "total_weight": total_weight,
                "model_type": "ensemble"
            }
            
        except Exception as e:
            logger.error(f"앙상블 예측 생성 실패: {e}")
            return {"error": str(e)}
    
    def calculate_prediction_performance_score(self, prediction: Dict, 
                                             model_predictions: Dict, regime_type: str) -> float:
        """예측 성능 점수 계산"""
        try:
            base_confidence = prediction.get("confidence", 0.5)
            
            # 모델 합의도 (여러 모델이 같은 방향 예측시 가산점)
            directions = [p.get("direction", "NEUTRAL") for p in model_predictions.values()]
            consensus = directions.count(prediction.get("direction", "NEUTRAL")) / len(directions)
            
            # 체제 적합성 (각 체제별 기대 성능)
            regime_multiplier = {
                "BULL_MARKET": 1.1,
                "BEAR_MARKET": 1.0,
                "SIDEWAYS": 0.9,
                "HIGH_VOLATILITY_SHOCK": 0.8,
                "LOW_VOLATILITY_ACCUMULATION": 1.0
            }.get(regime_type, 1.0)
            
            # 최종 성능 점수
            performance_score = base_confidence * consensus * regime_multiplier
            
            return min(performance_score, 1.0)
            
        except Exception as e:
            logger.error(f"성능 점수 계산 실패: {e}")
            return 0.5
    
    async def save_prediction_result(self, result: PredictionResult):
        """예측 결과 저장"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO adaptive_predictions
                (timestamp, regime_type, model_used, prediction_data, confidence,
                 contributing_models, performance_score)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                result.timestamp.isoformat(),
                result.regime_type,
                result.model_used,
                json.dumps(result.prediction),
                result.confidence,
                json.dumps(result.contributing_models),
                result.performance_score
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"예측 결과 저장 실패: {e}")
    
    async def save_training_results(self, training_results: Dict):
        """학습 결과 저장"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for regime_type, results in training_results.items():
                for model_id, result in results.items():
                    if isinstance(result, dict) and not result.get("error"):
                        cursor.execute('''
                            INSERT INTO model_performance_tracking
                            (model_id, regime_type, performance_period, accuracy,
                             precision_val, recall_val, f1_score, directional_accuracy,
                             profit_factor, sample_size, last_updated)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            model_id,
                            regime_type,
                            "training",
                            result.get("accuracy", 0),
                            result.get("accuracy", 0),  # 단순화
                            result.get("accuracy", 0),
                            result.get("accuracy", 0),
                            result.get("accuracy", 0),
                            1.0,  # 기본값
                            result.get("samples", 0),
                            datetime.now().isoformat()
                        ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"학습 결과 저장 실패: {e}")
    
    async def get_adaptive_diagnostics(self) -> Dict:
        """적응형 시스템 진단"""
        try:
            diagnostics = {
                "system_status": "active",
                "total_regimes": len(self.regime_models),
                "total_models": sum(len(models) for models in self.regime_models.values()),
                "prediction_history_length": len(self.prediction_history)
            }
            
            # 체제별 모델 상태
            regime_status = {}
            for regime_type, models in self.regime_models.items():
                trained_models = sum(1 for model in models if model.is_trained)
                regime_status[regime_type] = {
                    "total_models": len(models),
                    "trained_models": trained_models,
                    "primary_model": self.regime_configs[regime_type].primary_model,
                    "ensemble_weights": self.regime_configs[regime_type].ensemble_weights,
                    "performance_threshold": self.regime_configs[regime_type].performance_threshold
                }
            
            diagnostics["regime_status"] = regime_status
            
            # 최근 예측 성능
            if self.prediction_history:
                recent_predictions = list(self.prediction_history)[-10:]
                avg_confidence = statistics.mean([p.confidence for p in recent_predictions])
                avg_performance = statistics.mean([p.performance_score for p in recent_predictions])
                
                diagnostics["recent_performance"] = {
                    "avg_confidence": avg_confidence,
                    "avg_performance_score": avg_performance,
                    "prediction_count": len(recent_predictions)
                }
            
            return diagnostics
            
        except Exception as e:
            logger.error(f"시스템 진단 실패: {e}")
            return {"error": str(e)}

# 테스트 함수
async def test_adaptive_regime_selector():
    """적응형 체제 모델 선택기 테스트"""
    print("🔄 적응형 체제 모델 선택기 테스트")
    print("=" * 60)
    
    selector = AdaptiveRegimeModelSelector()
    
    # 시스템 진단
    diagnostics = await selector.get_adaptive_diagnostics()
    print(f"📊 시스템 상태: {diagnostics.get('system_status')}")
    print(f"🎯 총 체제: {diagnostics.get('total_regimes')}개")
    print(f"🤖 총 모델: {diagnostics.get('total_models')}개")
    
    # 체제별 상태
    regime_status = diagnostics.get("regime_status", {})
    print(f"\n📈 체제별 모델 상태:")
    for regime, status in regime_status.items():
        trained = status["trained_models"]
        total = status["total_models"] 
        primary = status["primary_model"]
        print(f"   • {regime}: {trained}/{total} 학습됨, 주모델: {primary}")
    
    # 테스트 예측
    test_features = np.random.random(24)  # 24개 특징
    
    print(f"\n🧪 테스트 예측 수행:")
    for regime_type in ["BULL_MARKET", "BEAR_MARKET", "SIDEWAYS"]:
        try:
            result = await selector.predict_with_adaptive_selection(regime_type, test_features)
            
            print(f"\n🎯 {regime_type}:")
            print(f"   • 사용 모델: {result.model_used}")
            print(f"   • 예측: {result.prediction.get('direction', 'N/A')}")
            print(f"   • 신뢰도: {result.confidence:.1%}")
            print(f"   • 성능 점수: {result.performance_score:.3f}")
            
            if result.contributing_models:
                print(f"   • 기여 모델 가중치:")
                for model, weight in result.contributing_models.items():
                    print(f"     - {model}: {weight:.1%}")
                    
        except Exception as e:
            print(f"   ❌ {regime_type} 예측 실패: {e}")
    
    print("\n" + "=" * 60)
    print("🎉 적응형 체제 모델 선택기 테스트 완료!")

if __name__ == "__main__":
    asyncio.run(test_adaptive_regime_selector())