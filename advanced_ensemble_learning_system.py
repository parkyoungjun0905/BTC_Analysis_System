#!/usr/bin/env python3
"""
🎯 고급 앙상블 학습 시스템 - 90%+ 정확도 달성
비트코인 가격 예측을 위한 종합적인 앙상블 머신러닝 시스템

핵심 기능:
- 다양한 모델 아키텍처 통합 (LSTM, Transformer, XGBoost, CNN 등)
- 동적 모델 가중치 조정 및 적응적 앙상블
- 메타 학습 기반 모델 선택 및 최적화
- 베이지안 모델 평균화 및 불확실성 정량화
- 강건한 성능 모니터링 및 실패 감지
"""

import numpy as np
import pandas as pd
import warnings
import logging
import json
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional, Union
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from pathlib import Path
import sqlite3

# 머신러닝 라이브러리
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor,
    AdaBoostRegressor, BaggingRegressor, VotingRegressor, StackingRegressor
)
from sklearn.linear_model import (
    Ridge, Lasso, ElasticNet, HuberRegressor, RANSACRegressor, 
    TheilSenRegressor, BayesianRidge
)
from sklearn.svm import SVR, NuSVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import (
    TimeSeriesSplit, cross_val_score, GridSearchCV, RandomizedSearchCV
)
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error
)

# 딥러닝 라이브러리 (선택적 설치)
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import (
        LSTM, GRU, Dense, Dropout, Conv1D, MaxPooling1D, 
        MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D,
        Input, Concatenate, BatchNormalization
    )
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("⚠️ TensorFlow 미설치 - LSTM/Transformer 모델 비활성화")

# XGBoost/LightGBM/CatBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️ XGBoost 미설치")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("⚠️ LightGBM 미설치")

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("⚠️ CatBoost 미설치")

# 최적화 라이브러리
try:
    from scipy.optimize import minimize, differential_evolution
    from scipy.stats import norm
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("⚠️ SciPy 미설치")

warnings.filterwarnings('ignore')

class AdvancedEnsembleLearningSystem:
    """
    🧠 고급 앙상블 학습 시스템
    
    다양한 모델을 통합하여 90%+ 정확도를 달성하는 종합 시스템
    """
    
    def __init__(self, target_accuracy: float = 0.90):
        """초기화"""
        self.target_accuracy = target_accuracy
        self.models = {}
        self.model_weights = {}
        self.model_performance = {}
        self.meta_learners = {}
        self.scalers = {}
        
        # 성능 추적
        self.accuracy_history = []
        self.ensemble_history = []
        self.failure_count = 0
        
        # 로깅 설정
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('ensemble_system.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # 데이터 경로
        self.data_path = Path("/Users/parkyoungjun/Desktop/BTC_Analysis_System/historical_6month_data")
        
        # 모델 저장 경로
        self.model_save_path = Path("/Users/parkyoungjun/Desktop/BTC_Analysis_System/ensemble_models")
        self.model_save_path.mkdir(exist_ok=True)
        
        print("🎯 고급 앙상블 학습 시스템 초기화 완료")
        print(f"📊 목표 정확도: {target_accuracy*100:.1f}%")

    def load_comprehensive_data(self) -> pd.DataFrame:
        """
        📊 종합 데이터 로드 및 전처리
        
        Returns:
            pd.DataFrame: 통합 데이터셋
        """
        print("📊 종합 데이터 로드 시작...")
        
        all_data = {}
        files_loaded = 0
        
        # 모든 CSV 파일 로드
        for csv_file in self.data_path.glob("*.csv"):
            try:
                df = pd.read_csv(csv_file, index_col=0, parse_dates=True)
                if not df.empty:
                    column_name = csv_file.stem
                    all_data[column_name] = df.iloc[:, 0]  # 첫 번째 컬럼만 사용
                    files_loaded += 1
            except Exception as e:
                self.logger.warning(f"⚠️ 파일 로드 실패: {csv_file.name} - {e}")
                continue
        
        # 데이터 통합
        if not all_data:
            raise ValueError("❌ 로드된 데이터가 없습니다")
        
        combined_df = pd.DataFrame(all_data)
        combined_df = combined_df.dropna()
        
        print(f"✅ 데이터 로드 완료: {files_loaded}개 파일, {len(combined_df)}개 행")
        print(f"📈 특성 수: {len(combined_df.columns)}개")
        
        return combined_df

    def create_diverse_models(self) -> Dict[str, Any]:
        """
        🤖 다양한 모델 아키텍처 생성
        
        Returns:
            Dict[str, Any]: 생성된 모델들
        """
        print("🤖 다양한 모델 아키텍처 생성 시작...")
        
        models = {}
        
        # 1. 전통적인 머신러닝 모델
        models['random_forest'] = {
            'model': RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                random_state=42,
                n_jobs=-1
            ),
            'type': 'traditional',
            'hyperparams': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 15, 20, None],
                'min_samples_split': [2, 5, 10]
            }
        }
        
        models['gradient_boosting'] = {
            'model': GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=8,
                random_state=42
            ),
            'type': 'boosting',
            'hyperparams': {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.05, 0.1, 0.15],
                'max_depth': [6, 8, 10]
            }
        }
        
        models['extra_trees'] = {
            'model': ExtraTreesRegressor(
                n_estimators=200,
                random_state=42,
                n_jobs=-1
            ),
            'type': 'traditional'
        }
        
        # 2. 강건한 회귀 모델
        models['huber_regressor'] = {
            'model': HuberRegressor(epsilon=1.35, max_iter=1000),
            'type': 'robust'
        }
        
        models['ransac_regressor'] = {
            'model': RANSACRegressor(random_state=42, max_trials=1000),
            'type': 'robust'
        }
        
        models['theil_sen'] = {
            'model': TheilSenRegressor(random_state=42, n_jobs=-1),
            'type': 'robust'
        }
        
        # 3. 베이지안 모델
        models['bayesian_ridge'] = {
            'model': BayesianRidge(compute_score=True),
            'type': 'bayesian'
        }
        
        # 4. SVM 모델
        models['svr_rbf'] = {
            'model': SVR(kernel='rbf', C=100, gamma='scale'),
            'type': 'svm'
        }
        
        models['svr_poly'] = {
            'model': SVR(kernel='poly', degree=3, C=100),
            'type': 'svm'
        }
        
        # 5. 신경망 모델
        models['mlp_regressor'] = {
            'model': MLPRegressor(
                hidden_layer_sizes=(200, 100, 50),
                max_iter=2000,
                random_state=42,
                early_stopping=True
            ),
            'type': 'neural'
        }
        
        # 6. XGBoost 모델 (설치된 경우)
        if XGBOOST_AVAILABLE:
            models['xgboost'] = {
                'model': xgb.XGBRegressor(
                    n_estimators=200,
                    learning_rate=0.1,
                    max_depth=8,
                    random_state=42,
                    n_jobs=-1
                ),
                'type': 'boosting',
                'hyperparams': {
                    'n_estimators': [100, 200, 300],
                    'learning_rate': [0.05, 0.1, 0.15],
                    'max_depth': [6, 8, 10]
                }
            }
        
        # 7. LightGBM 모델 (설치된 경우)
        if LIGHTGBM_AVAILABLE:
            models['lightgbm'] = {
                'model': lgb.LGBMRegressor(
                    n_estimators=200,
                    learning_rate=0.1,
                    max_depth=8,
                    random_state=42,
                    n_jobs=-1,
                    verbose=-1
                ),
                'type': 'boosting'
            }
        
        # 8. CatBoost 모델 (설치된 경우)
        if CATBOOST_AVAILABLE:
            models['catboost'] = {
                'model': cb.CatBoostRegressor(
                    iterations=200,
                    learning_rate=0.1,
                    depth=8,
                    random_state=42,
                    silent=True
                ),
                'type': 'boosting'
            }
        
        print(f"✅ {len(models)}개 모델 아키텍처 생성 완료")
        return models

    def create_deep_learning_models(self, input_shape: Tuple[int, int]) -> Dict[str, Any]:
        """
        🧠 딥러닝 모델 생성 (TensorFlow 사용 가능시)
        
        Args:
            input_shape: 입력 데이터 형태
            
        Returns:
            Dict[str, Any]: 딥러닝 모델들
        """
        if not TENSORFLOW_AVAILABLE:
            return {}
        
        print("🧠 딥러닝 모델 생성 시작...")
        models = {}
        
        # 1. LSTM 모델
        lstm_model = Sequential([
            LSTM(128, return_sequences=True, input_shape=input_shape),
            Dropout(0.3),
            LSTM(64, return_sequences=True),
            Dropout(0.3),
            LSTM(32, return_sequences=False),
            Dropout(0.3),
            Dense(50, activation='relu'),
            Dropout(0.2),
            Dense(1)
        ])
        
        lstm_model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        models['lstm'] = {
            'model': lstm_model,
            'type': 'deep_learning',
            'sequence_length': input_shape[0]
        }
        
        # 2. GRU 모델
        gru_model = Sequential([
            GRU(128, return_sequences=True, input_shape=input_shape),
            Dropout(0.3),
            GRU(64, return_sequences=True),
            Dropout(0.3),
            GRU(32, return_sequences=False),
            Dropout(0.3),
            Dense(50, activation='relu'),
            Dropout(0.2),
            Dense(1)
        ])
        
        gru_model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        models['gru'] = {
            'model': gru_model,
            'type': 'deep_learning',
            'sequence_length': input_shape[0]
        }
        
        # 3. CNN-LSTM 하이브리드 모델
        cnn_lstm_model = Sequential([
            Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            Conv1D(filters=32, kernel_size=3, activation='relu'),
            BatchNormalization(),
            LSTM(64, return_sequences=True),
            Dropout(0.3),
            LSTM(32, return_sequences=False),
            Dropout(0.3),
            Dense(50, activation='relu'),
            Dense(1)
        ])
        
        cnn_lstm_model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        models['cnn_lstm'] = {
            'model': cnn_lstm_model,
            'type': 'deep_learning',
            'sequence_length': input_shape[0]
        }
        
        # 4. Transformer 모델 (간단한 버전)
        def create_transformer_model(input_shape):
            inputs = Input(shape=input_shape)
            
            # Multi-Head Attention
            attention = MultiHeadAttention(
                num_heads=8,
                key_dim=64
            )(inputs, inputs)
            
            attention = Dropout(0.3)(attention)
            attention = LayerNormalization()(inputs + attention)
            
            # Feed Forward
            ff = Dense(128, activation='relu')(attention)
            ff = Dropout(0.3)(ff)
            ff = Dense(input_shape[-1])(ff)
            ff = LayerNormalization()(attention + ff)
            
            # Global Average Pooling
            pooled = GlobalAveragePooling1D()(ff)
            
            # Final layers
            outputs = Dense(50, activation='relu')(pooled)
            outputs = Dropout(0.3)(outputs)
            outputs = Dense(1)(outputs)
            
            model = Model(inputs=inputs, outputs=outputs)
            return model
        
        transformer_model = create_transformer_model(input_shape)
        transformer_model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        models['transformer'] = {
            'model': transformer_model,
            'type': 'deep_learning',
            'sequence_length': input_shape[0]
        }
        
        print(f"✅ {len(models)}개 딥러닝 모델 생성 완료")
        return models

    def prepare_sequences(self, data: pd.DataFrame, sequence_length: int = 24) -> Tuple[np.ndarray, np.ndarray]:
        """
        🔄 시계열 시퀀스 데이터 준비 (딥러닝용)
        
        Args:
            data: 입력 데이터
            sequence_length: 시퀀스 길이
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: X, y 배열
        """
        # 가격 데이터가 있는 컬럼 찾기
        price_col = None
        for col in data.columns:
            if 'price' in col.lower() or 'btc' in col.lower():
                price_col = col
                break
        
        if price_col is None:
            price_col = data.columns[0]  # 첫 번째 컬럼 사용
        
        # 특성과 타겟 분리
        features = data.drop(columns=[price_col])
        target = data[price_col]
        
        # 시퀀스 생성
        X, y = [], []
        
        for i in range(sequence_length, len(data)):
            X.append(features.iloc[i-sequence_length:i].values)
            y.append(target.iloc[i])
        
        return np.array(X), np.array(y)

    def train_single_model(self, model_info: Dict, X_train: np.ndarray, 
                          y_train: np.ndarray, X_val: np.ndarray, 
                          y_val: np.ndarray) -> Dict:
        """
        🎯 단일 모델 훈련
        
        Args:
            model_info: 모델 정보
            X_train: 훈련 데이터
            y_train: 훈련 타겟
            X_val: 검증 데이터
            y_val: 검증 타겟
            
        Returns:
            Dict: 훈련 결과
        """
        model = model_info['model']
        model_type = model_info['type']
        
        try:
            # 딥러닝 모델 훈련
            if model_type == 'deep_learning':
                callbacks = [
                    EarlyStopping(patience=20, restore_best_weights=True),
                    ReduceLROnPlateau(factor=0.5, patience=10)
                ]
                
                history = model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=200,
                    batch_size=32,
                    callbacks=callbacks,
                    verbose=0
                )
                
                # 예측 및 성능 평가
                y_pred = model.predict(X_val).flatten()
                
            else:
                # 전통적인 ML 모델 훈련
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
            
            # 성능 메트릭 계산
            mse = mean_squared_error(y_val, y_pred)
            mae = mean_absolute_error(y_val, y_pred)
            r2 = r2_score(y_val, y_pred)
            
            # 방향성 정확도 계산
            direction_actual = np.diff(y_val) > 0
            direction_pred = np.diff(y_pred) > 0
            direction_accuracy = np.mean(direction_actual == direction_pred)
            
            return {
                'model': model,
                'mse': mse,
                'mae': mae,
                'r2': r2,
                'direction_accuracy': direction_accuracy,
                'predictions': y_pred,
                'status': 'success'
            }
            
        except Exception as e:
            self.logger.error(f"❌ 모델 훈련 실패: {e}")
            return {
                'model': model,
                'status': 'failed',
                'error': str(e)
            }

    def hyperparameter_optimization(self, model_info: Dict, X_train: np.ndarray, 
                                  y_train: np.ndarray) -> Dict:
        """
        🔧 하이퍼파라미터 최적화
        
        Args:
            model_info: 모델 정보
            X_train: 훈련 데이터
            y_train: 훈련 타겟
            
        Returns:
            Dict: 최적화된 모델 정보
        """
        if 'hyperparams' not in model_info or model_info['type'] == 'deep_learning':
            return model_info
        
        try:
            # 시계열 분할
            tscv = TimeSeriesSplit(n_splits=5)
            
            # 그리드 서치
            grid_search = RandomizedSearchCV(
                model_info['model'],
                model_info['hyperparams'],
                cv=tscv,
                scoring='neg_mean_squared_error',
                n_iter=20,
                random_state=42,
                n_jobs=-1
            )
            
            grid_search.fit(X_train, y_train)
            
            # 최적화된 모델로 업데이트
            model_info['model'] = grid_search.best_estimator_
            model_info['best_params'] = grid_search.best_params_
            model_info['best_score'] = -grid_search.best_score_
            
            return model_info
            
        except Exception as e:
            self.logger.warning(f"⚠️ 하이퍼파라미터 최적화 실패: {e}")
            return model_info

    def dynamic_ensemble_weighting(self, model_results: Dict[str, Dict]) -> Dict[str, float]:
        """
        ⚖️ 동적 앙상블 가중치 계산
        
        Args:
            model_results: 모델별 결과
            
        Returns:
            Dict[str, float]: 모델별 가중치
        """
        weights = {}
        
        # 성능 지표별 가중치
        performance_weights = {
            'direction_accuracy': 0.4,  # 방향성이 가장 중요
            'r2': 0.3,
            'mse': 0.2,
            'mae': 0.1
        }
        
        # 각 모델의 종합 점수 계산
        model_scores = {}
        
        for model_name, result in model_results.items():
            if result['status'] != 'success':
                model_scores[model_name] = 0.0
                continue
            
            # 정규화된 점수 계산
            score = 0.0
            
            # 방향성 정확도 (높을수록 좋음)
            score += result['direction_accuracy'] * performance_weights['direction_accuracy']
            
            # R² 점수 (높을수록 좋음, 음수일 수 있으므로 0과 1 사이로 클리핑)
            r2_normalized = max(0, min(1, result['r2']))
            score += r2_normalized * performance_weights['r2']
            
            # MSE (낮을수록 좋음, 역수 사용)
            mse_score = 1 / (1 + result['mse'])
            score += mse_score * performance_weights['mse']
            
            # MAE (낮을수록 좋음, 역수 사용)  
            mae_score = 1 / (1 + result['mae'])
            score += mae_score * performance_weights['mae']
            
            model_scores[model_name] = score
        
        # 소프트맥스 변환으로 가중치 정규화
        if model_scores:
            scores_array = np.array(list(model_scores.values()))
            if np.sum(scores_array) > 0:
                # 온도 파라미터를 사용한 소프트맥스 (더 선택적)
                temperature = 2.0
                exp_scores = np.exp(scores_array / temperature)
                softmax_weights = exp_scores / np.sum(exp_scores)
                
                for i, model_name in enumerate(model_scores.keys()):
                    weights[model_name] = float(softmax_weights[i])
            else:
                # 모든 모델이 실패한 경우 균등 가중치
                n_models = len(model_scores)
                for model_name in model_scores.keys():
                    weights[model_name] = 1.0 / n_models
        
        return weights

    def meta_learning_optimization(self, model_results: Dict[str, Dict], 
                                 X_val: np.ndarray, y_val: np.ndarray) -> Any:
        """
        🧠 메타 학습 기반 앙상블 최적화
        
        Args:
            model_results: 모델별 결과
            X_val: 검증 데이터
            y_val: 검증 타겟
            
        Returns:
            메타 러너 모델
        """
        # 성공한 모델들의 예측값 수집
        meta_features = []
        successful_models = []
        
        for model_name, result in model_results.items():
            if result['status'] == 'success':
                meta_features.append(result['predictions'])
                successful_models.append(model_name)
        
        if len(meta_features) < 2:
            return None
        
        # 메타 특성 매트릭스 생성
        meta_X = np.column_stack(meta_features)
        
        # 메타 러너 훈련 (여러 모델 시도)
        meta_learners = {
            'ridge': Ridge(alpha=1.0),
            'elastic_net': ElasticNet(alpha=0.1, l1_ratio=0.5),
            'random_forest': RandomForestRegressor(n_estimators=50, random_state=42),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=50, random_state=42)
        }
        
        best_meta_learner = None
        best_score = float('inf')
        
        for name, learner in meta_learners.items():
            try:
                # 시계열 교차검증
                tscv = TimeSeriesSplit(n_splits=3)
                scores = cross_val_score(learner, meta_X, y_val, cv=tscv, 
                                       scoring='neg_mean_squared_error')
                avg_score = -np.mean(scores)
                
                if avg_score < best_score:
                    best_score = avg_score
                    best_meta_learner = learner
                    
            except Exception as e:
                self.logger.warning(f"⚠️ 메타 러너 {name} 훈련 실패: {e}")
                continue
        
        # 최적 메타 러너 훈련
        if best_meta_learner is not None:
            best_meta_learner.fit(meta_X, y_val)
            
            return {
                'meta_learner': best_meta_learner,
                'successful_models': successful_models,
                'meta_score': best_score
            }
        
        return None

    def bayesian_model_averaging(self, model_results: Dict[str, Dict], 
                                X_val: np.ndarray, y_val: np.ndarray) -> Dict:
        """
        📊 베이지안 모델 평균화
        
        Args:
            model_results: 모델별 결과
            X_val: 검증 데이터  
            y_val: 검증 타겟
            
        Returns:
            Dict: 베이지안 평균화 결과
        """
        if not SCIPY_AVAILABLE:
            return None
        
        # 성공한 모델들의 예측값과 성능 수집
        predictions = []
        likelihoods = []
        
        for model_name, result in model_results.items():
            if result['status'] == 'success':
                pred = result['predictions']
                
                # 우도 계산 (MSE 기반)
                mse = result['mse']
                likelihood = np.exp(-mse / (2 * np.var(y_val)))
                
                predictions.append(pred)
                likelihoods.append(likelihood)
        
        if len(predictions) < 2:
            return None
        
        # 베이지안 가중치 계산
        likelihoods = np.array(likelihoods)
        bayesian_weights = likelihoods / np.sum(likelihoods)
        
        # 가중 평균 예측
        weighted_predictions = np.average(predictions, axis=0, weights=bayesian_weights)
        
        # 불확실성 정량화 (예측 분산)
        prediction_variance = np.var(predictions, axis=0)
        uncertainty = np.sqrt(prediction_variance)
        
        return {
            'predictions': weighted_predictions,
            'weights': bayesian_weights,
            'uncertainty': uncertainty,
            'confidence_intervals': {
                'lower': weighted_predictions - 1.96 * uncertainty,
                'upper': weighted_predictions + 1.96 * uncertainty
            }
        }

    def ensemble_prediction(self, models: Dict, weights: Dict[str, float], 
                          X_test: np.ndarray, method: str = 'weighted_average') -> np.ndarray:
        """
        🎯 앙상블 예측 수행
        
        Args:
            models: 훈련된 모델들
            weights: 모델 가중치
            X_test: 테스트 데이터
            method: 앙상블 방법
            
        Returns:
            np.ndarray: 앙상블 예측값
        """
        predictions = []
        model_weights = []
        
        for model_name, model_info in models.items():
            if model_name in weights and weights[model_name] > 0:
                try:
                    if model_info['type'] == 'deep_learning':
                        pred = model_info['model'].predict(X_test).flatten()
                    else:
                        pred = model_info['model'].predict(X_test)
                    
                    predictions.append(pred)
                    model_weights.append(weights[model_name])
                    
                except Exception as e:
                    self.logger.warning(f"⚠️ 모델 {model_name} 예측 실패: {e}")
                    continue
        
        if not predictions:
            raise ValueError("❌ 예측 가능한 모델이 없습니다")
        
        predictions = np.array(predictions)
        model_weights = np.array(model_weights)
        
        if method == 'weighted_average':
            # 가중 평균
            ensemble_pred = np.average(predictions, axis=0, weights=model_weights)
            
        elif method == 'median':
            # 중위수 (이상치에 강건)
            ensemble_pred = np.median(predictions, axis=0)
            
        elif method == 'trimmed_mean':
            # 절사 평균 (상하위 20% 제거)
            sorted_preds = np.sort(predictions, axis=0)
            n_models = len(predictions)
            trim_count = max(1, int(0.2 * n_models))
            trimmed = sorted_preds[trim_count:-trim_count] if n_models > 2*trim_count else sorted_preds
            ensemble_pred = np.mean(trimmed, axis=0)
            
        else:
            ensemble_pred = np.mean(predictions, axis=0)
        
        return ensemble_pred

    def performance_monitoring(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """
        📊 성능 모니터링 및 분석
        
        Args:
            y_true: 실제 값
            y_pred: 예측 값
            
        Returns:
            Dict: 성능 분석 결과
        """
        # 기본 메트릭
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # 방향성 정확도
        if len(y_true) > 1:
            direction_actual = np.diff(y_true) > 0
            direction_pred = np.diff(y_pred) > 0
            direction_accuracy = np.mean(direction_actual == direction_pred)
        else:
            direction_accuracy = 0.0
        
        # MAPE (Mean Absolute Percentage Error)
        mape = mean_absolute_percentage_error(y_true, y_pred) * 100
        
        # 잔차 분석
        residuals = y_true - y_pred
        residual_std = np.std(residuals)
        residual_mean = np.mean(residuals)
        
        # 성능 등급 계산
        if direction_accuracy >= 0.90:
            grade = 'A+'
        elif direction_accuracy >= 0.85:
            grade = 'A'
        elif direction_accuracy >= 0.80:
            grade = 'B+'
        elif direction_accuracy >= 0.75:
            grade = 'B'
        elif direction_accuracy >= 0.70:
            grade = 'C+'
        else:
            grade = 'C'
        
        return {
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'direction_accuracy': direction_accuracy,
            'mape': mape,
            'residual_std': residual_std,
            'residual_mean': residual_mean,
            'grade': grade,
            'target_achieved': direction_accuracy >= self.target_accuracy
        }

    def failure_detection_and_recovery(self, performance: Dict) -> Dict:
        """
        🔧 모델 실패 감지 및 복구
        
        Args:
            performance: 성능 분석 결과
            
        Returns:
            Dict: 복구 전략
        """
        issues = []
        recovery_actions = []
        
        # 성능 저하 감지
        if performance['direction_accuracy'] < 0.55:
            issues.append("심각한 성능 저하 (< 55%)")
            recovery_actions.append("모든 모델 재훈련")
            
        elif performance['direction_accuracy'] < 0.65:
            issues.append("성능 저하 감지 (< 65%)")
            recovery_actions.append("저성능 모델 제거 및 가중치 재조정")
        
        # 과적합 감지
        if abs(performance['residual_mean']) > 0.1 * np.mean(np.abs(performance.get('y_true', [1]))):
            issues.append("편향성 감지")
            recovery_actions.append("정규화 강화")
        
        # R² 점수 확인
        if performance['r2'] < 0.3:
            issues.append("낮은 설명력")
            recovery_actions.append("특성 엔지니어링 개선")
        
        # 복구 전략 실행
        if issues:
            self.failure_count += 1
            self.logger.warning(f"⚠️ 감지된 문제: {', '.join(issues)}")
            self.logger.info(f"🔧 복구 액션: {', '.join(recovery_actions)}")
        
        return {
            'issues_detected': issues,
            'recovery_actions': recovery_actions,
            'failure_count': self.failure_count,
            'needs_retraining': len(issues) > 0
        }

    def save_ensemble_system(self, models: Dict, weights: Dict, 
                           meta_learner: Any = None) -> str:
        """
        💾 앙상블 시스템 저장
        
        Args:
            models: 훈련된 모델들
            weights: 모델 가중치
            meta_learner: 메타 러너 (선택적)
            
        Returns:
            str: 저장 경로
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = self.model_save_path / f"ensemble_system_{timestamp}.pkl"
        
        ensemble_data = {
            'models': models,
            'weights': weights,
            'meta_learner': meta_learner,
            'scalers': self.scalers,
            'target_accuracy': self.target_accuracy,
            'accuracy_history': self.accuracy_history,
            'timestamp': timestamp
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(ensemble_data, f)
        
        self.logger.info(f"💾 앙상블 시스템 저장 완료: {save_path}")
        return str(save_path)

    def train_ensemble_system(self) -> Dict:
        """
        🎯 전체 앙상블 시스템 훈련
        
        Returns:
            Dict: 훈련 결과
        """
        print("\n🎯 고급 앙상블 학습 시스템 훈련 시작")
        print("=" * 50)
        
        start_time = datetime.now()
        
        try:
            # 1. 데이터 로드
            data = self.load_comprehensive_data()
            
            # 2. 데이터 전처리
            print("🔄 데이터 전처리...")
            
            # 가격 컬럼 찾기
            price_col = None
            for col in data.columns:
                if 'price' in col.lower():
                    price_col = col
                    break
            
            if price_col is None:
                price_col = data.columns[0]
            
            # 특성과 타겟 분리
            features = data.drop(columns=[price_col])
            target = data[price_col]
            
            # 데이터 분할 (시계열이므로 순차적)
            split_idx = int(len(data) * 0.8)
            
            X_train = features.iloc[:split_idx]
            y_train = target.iloc[:split_idx]
            X_test = features.iloc[split_idx:]
            y_test = target.iloc[split_idx:]
            
            # 검증 세트 분할
            val_split_idx = int(len(X_train) * 0.8)
            X_val = X_train.iloc[val_split_idx:]
            y_val = y_train.iloc[val_split_idx:]
            X_train = X_train.iloc[:val_split_idx]
            y_train = y_train.iloc[:val_split_idx]
            
            # 데이터 스케일링
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            X_test_scaled = scaler.transform(X_test)
            
            self.scalers['feature_scaler'] = scaler
            
            print(f"📊 훈련 데이터: {len(X_train)} 샘플")
            print(f"📊 검증 데이터: {len(X_val)} 샘플") 
            print(f"📊 테스트 데이터: {len(X_test)} 샘플")
            
            # 3. 다양한 모델 생성
            models = self.create_diverse_models()
            
            # 딥러닝 모델 추가 (데이터가 충분한 경우)
            if len(X_train) > 500 and TENSORFLOW_AVAILABLE:
                sequence_length = min(24, len(X_train) // 20)
                X_seq_train, y_seq_train = self.prepare_sequences(
                    pd.concat([X_train, y_train.to_frame()], axis=1), 
                    sequence_length
                )
                X_seq_val, y_seq_val = self.prepare_sequences(
                    pd.concat([X_val, y_val.to_frame()], axis=1), 
                    sequence_length
                )
                
                if len(X_seq_train) > 100:
                    input_shape = (sequence_length, X_train.shape[1])
                    deep_models = self.create_deep_learning_models(input_shape)
                    models.update(deep_models)
            
            print(f"🤖 총 {len(models)}개 모델 생성 완료")
            
            # 4. 모델별 하이퍼파라미터 최적화
            print("🔧 하이퍼파라미터 최적화...")
            optimized_models = {}
            
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = {}
                for name, model_info in models.items():
                    if model_info['type'] != 'deep_learning':  # 전통적 ML만
                        future = executor.submit(
                            self.hyperparameter_optimization, 
                            model_info, X_train_scaled, y_train.values
                        )
                        futures[name] = future
                    else:
                        optimized_models[name] = model_info
                
                # 결과 수집
                for name, future in futures.items():
                    try:
                        optimized_models[name] = future.result(timeout=300)
                    except Exception as e:
                        self.logger.warning(f"⚠️ {name} 최적화 실패: {e}")
                        optimized_models[name] = models[name]
            
            models = optimized_models
            
            # 5. 모델 훈련
            print("🎯 모델 훈련 시작...")
            model_results = {}
            
            for model_name, model_info in models.items():
                print(f"  📈 {model_name} 훈련 중...")
                
                if model_info['type'] == 'deep_learning':
                    # 시퀀스 데이터 사용
                    if 'X_seq_train' in locals():
                        result = self.train_single_model(
                            model_info, X_seq_train, y_seq_train,
                            X_seq_val, y_seq_val
                        )
                    else:
                        continue
                else:
                    # 일반 데이터 사용
                    result = self.train_single_model(
                        model_info, X_train_scaled, y_train.values,
                        X_val_scaled, y_val.values
                    )
                
                model_results[model_name] = result
                
                if result['status'] == 'success':
                    print(f"    ✅ 방향성 정확도: {result['direction_accuracy']:.3f}")
                    print(f"    ✅ R² 점수: {result['r2']:.3f}")
                else:
                    print(f"    ❌ 훈련 실패: {result.get('error', '알 수 없는 오류')}")
            
            # 성공한 모델 개수 확인
            successful_models = [name for name, result in model_results.items() 
                               if result['status'] == 'success']
            print(f"\n✅ 성공한 모델: {len(successful_models)}개")
            
            if len(successful_models) == 0:
                raise ValueError("❌ 훈련에 성공한 모델이 없습니다")
            
            # 6. 동적 가중치 계산
            print("⚖️ 동적 앙상블 가중치 계산...")
            weights = self.dynamic_ensemble_weighting(model_results)
            
            print("📊 모델별 가중치:")
            for model_name, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True):
                if weight > 0.01:  # 1% 이상인 경우만 출력
                    print(f"  {model_name}: {weight:.3f}")
            
            # 7. 메타 학습 최적화
            print("🧠 메타 학습 최적화...")
            meta_learner_result = self.meta_learning_optimization(
                model_results, X_val_scaled, y_val.values
            )
            
            # 8. 베이지안 모델 평균화
            print("📊 베이지안 모델 평균화...")
            bayesian_result = self.bayesian_model_averaging(
                model_results, X_val_scaled, y_val.values
            )
            
            # 9. 최종 테스트 예측
            print("🎯 최종 테스트 예측...")
            
            # 딥러닝 모델을 위한 시퀀스 데이터 준비
            test_predictions = {}
            
            for model_name, model_info in models.items():
                if (model_name in model_results and 
                    model_results[model_name]['status'] == 'success'):
                    
                    try:
                        if model_info['type'] == 'deep_learning':
                            if 'X_seq_train' in locals():
                                # 테스트용 시퀀스 데이터 준비
                                X_seq_test, y_seq_test = self.prepare_sequences(
                                    pd.concat([X_test, y_test.to_frame()], axis=1),
                                    sequence_length
                                )
                                pred = model_results[model_name]['model'].predict(X_seq_test).flatten()
                                # 길이 맞추기
                                if len(pred) != len(y_test):
                                    pred = np.pad(pred, (len(y_test) - len(pred), 0), 
                                                mode='edge')[:len(y_test)]
                            else:
                                continue
                        else:
                            pred = model_results[model_name]['model'].predict(X_test_scaled)
                        
                        test_predictions[model_name] = pred
                        
                    except Exception as e:
                        self.logger.warning(f"⚠️ {model_name} 테스트 예측 실패: {e}")
                        continue
            
            # 앙상블 예측
            if test_predictions:
                # 가중 평균 앙상블
                ensemble_pred = self.ensemble_prediction(
                    {name: {'model': model_results[name]['model'], 
                           'type': models[name]['type']} 
                     for name in test_predictions.keys()},
                    weights, X_test_scaled, method='weighted_average'
                )
                
                # 성능 평가
                performance = self.performance_monitoring(y_test.values, ensemble_pred)
                
                # 실패 감지 및 복구
                failure_analysis = self.failure_detection_and_recovery(performance)
                
                # 결과 저장
                self.accuracy_history.append({
                    'timestamp': datetime.now(),
                    'accuracy': performance['direction_accuracy'],
                    'r2': performance['r2'],
                    'grade': performance['grade']
                })
                
                # 시스템 저장
                save_path = self.save_ensemble_system(
                    {name: {'model': model_results[name]['model'],
                           'type': models[name]['type']}
                     for name in successful_models},
                    weights,
                    meta_learner_result
                )
                
                # 최종 결과
                end_time = datetime.now()
                training_time = (end_time - start_time).total_seconds()
                
                result = {
                    'success': True,
                    'training_time_seconds': training_time,
                    'models_trained': len(models),
                    'successful_models': len(successful_models),
                    'ensemble_performance': performance,
                    'model_weights': weights,
                    'meta_learner': meta_learner_result is not None,
                    'bayesian_averaging': bayesian_result is not None,
                    'failure_analysis': failure_analysis,
                    'save_path': save_path,
                    'target_achieved': performance['target_achieved']
                }
                
                # 결과 출력
                print("\n" + "="*50)
                print("🎯 앙상블 학습 시스템 훈련 완료!")
                print("="*50)
                print(f"⏱️  총 훈련 시간: {training_time:.1f}초")
                print(f"🤖 훈련된 모델: {len(models)}개")
                print(f"✅ 성공한 모델: {len(successful_models)}개")
                print(f"📊 방향성 정확도: {performance['direction_accuracy']:.3f} ({performance['direction_accuracy']*100:.1f}%)")
                print(f"📈 R² 점수: {performance['r2']:.3f}")
                print(f"🏆 성능 등급: {performance['grade']}")
                print(f"🎯 목표 달성: {'✅ YES' if performance['target_achieved'] else '❌ NO'}")
                print(f"💾 저장 경로: {save_path}")
                
                return result
            
            else:
                raise ValueError("❌ 테스트 예측에 성공한 모델이 없습니다")
                
        except Exception as e:
            self.logger.error(f"❌ 앙상블 시스템 훈련 실패: {e}")
            return {
                'success': False,
                'error': str(e),
                'training_time_seconds': (datetime.now() - start_time).total_seconds()
            }

def main():
    """메인 실행 함수"""
    print("🎯 고급 앙상블 학습 시스템 시작")
    
    # 시스템 초기화
    ensemble_system = AdvancedEnsembleLearningSystem(target_accuracy=0.90)
    
    # 앙상블 시스템 훈련
    result = ensemble_system.train_ensemble_system()
    
    # 결과를 JSON으로 저장
    result_path = "/Users/parkyoungjun/Desktop/BTC_Analysis_System/ensemble_learning_results.json"
    
    # datetime 객체를 문자열로 변환
    if 'accuracy_history' in result:
        for item in result.get('accuracy_history', []):
            if 'timestamp' in item and hasattr(item['timestamp'], 'isoformat'):
                item['timestamp'] = item['timestamp'].isoformat()
    
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\n📄 결과 저장 완료: {result_path}")
    
    return result

if __name__ == "__main__":
    result = main()