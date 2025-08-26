#!/usr/bin/env python3
"""
머신러닝 기반 실시간 시장 체제 분류기
다양한 ML 알고리즘을 앙상블하여 Bitcoin 시장 체제를 실시간으로 분류

지원 알고리즘:
1. Random Forest - 비선형 패턴 감지
2. Gradient Boosting - 순차적 특징 학습  
3. SVM - 고차원 경계 분류
4. Neural Network - 복잡한 관계 모델링
5. XGBoost - 고성능 부스팅
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
import joblib
from collections import defaultdict, deque
import warnings
warnings.filterwarnings("ignore")

# 머신러닝 라이브러리
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import xgboost as xgb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RegimeClassification:
    """체제 분류 결과"""
    regime_type: str
    confidence: float
    probability_distribution: Dict[str, float]
    feature_importance: Dict[str, float]
    ensemble_votes: Dict[str, str]
    prediction_timestamp: datetime
    
@dataclass
class ModelPerformance:
    """모델 성능 지표"""
    model_name: str
    accuracy: float
    precision: Dict[str, float]
    recall: Dict[str, float]
    f1_score: Dict[str, float]
    cross_val_score: float
    training_samples: int
    last_updated: datetime

class MLRegimeClassifier:
    """머신러닝 기반 시장 체제 분류기"""
    
    def __init__(self, base_path: str = "/Users/parkyoungjun/Desktop/BTC_Analysis_System"):
        self.base_path = base_path
        self.db_path = os.path.join(base_path, "ml_regime_db.db")
        self.models_path = os.path.join(base_path, "ml_regime_models")
        os.makedirs(self.models_path, exist_ok=True)
        
        # 체제 레이블
        self.regime_labels = {
            0: "LOW_VOLATILITY_ACCUMULATION",
            1: "BULL_MARKET",
            2: "SIDEWAYS", 
            3: "BEAR_MARKET",
            4: "HIGH_VOLATILITY_SHOCK"
        }
        
        # 특징 이름들
        self.feature_names = [
            "price_trend_1d", "price_trend_7d", "price_trend_30d", "trend_consistency",
            "volatility_1d", "volatility_7d", "volatility_30d", "volatility_regime_change", 
            "volume_trend", "volume_volatility", "volume_price_correlation",
            "rsi_14", "macd_signal", "bollinger_position",
            "whale_activity", "exchange_flow", "hodler_behavior",
            "futures_basis", "funding_rate", "put_call_ratio", "fear_greed_index",
            "correlation_gold", "correlation_stocks", "dxy_impact"
        ]
        
        # 개별 모델들
        self.models = {}
        self.scalers = {}
        self.feature_selectors = {}
        self.label_encoder = LabelEncoder()
        
        # 앙상블 모델
        self.ensemble_model = None
        self.meta_classifier = None
        
        # 학습 상태
        self.is_trained = False
        self.training_history = deque(maxlen=100)
        
        # 실시간 예측 히스토리
        self.prediction_history = deque(maxlen=200)
        self.current_regime = None
        self.regime_confidence = 0.0
        
        # 성능 추적
        self.performance_metrics = {}
        
        self.init_database()
        self.init_models()
        self.load_trained_models()
    
    def init_database(self):
        """데이터베이스 초기화"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # ML 예측 기록
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS ml_predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    predicted_regime TEXT NOT NULL,
                    regime_id INTEGER NOT NULL,
                    confidence REAL NOT NULL,
                    probability_distribution TEXT NOT NULL,
                    feature_importance TEXT NOT NULL,
                    ensemble_votes TEXT NOT NULL,
                    input_features TEXT NOT NULL,
                    model_version TEXT NOT NULL
                )
            ''')
            
            # 모델 성능 기록
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS ml_model_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT NOT NULL,
                    accuracy REAL NOT NULL,
                    precision_scores TEXT NOT NULL,
                    recall_scores TEXT NOT NULL,
                    f1_scores TEXT NOT NULL,
                    cross_val_score REAL NOT NULL,
                    training_samples INTEGER NOT NULL,
                    confusion_matrix TEXT NOT NULL,
                    feature_importance TEXT,
                    hyperparameters TEXT,
                    training_date TEXT NOT NULL
                )
            ''')
            
            # 학습 데이터
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS ml_training_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    features TEXT NOT NULL,
                    true_regime TEXT NOT NULL,
                    regime_id INTEGER NOT NULL,
                    data_source TEXT,
                    is_validated BOOLEAN DEFAULT FALSE,
                    validation_confidence REAL
                )
            ''')
            
            # 모델 비교 및 선택
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS model_comparison (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    comparison_date TEXT NOT NULL,
                    test_period_days INTEGER NOT NULL,
                    model_performances TEXT NOT NULL,
                    best_model TEXT NOT NULL,
                    ensemble_performance TEXT NOT NULL,
                    recommendation TEXT NOT NULL
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("✅ ML 체제 분류기 데이터베이스 초기화 완료")
            
        except Exception as e:
            logger.error(f"ML 데이터베이스 초기화 실패: {e}")
    
    def init_models(self):
        """ML 모델들 초기화"""
        try:
            # Random Forest
            self.models['random_forest'] = RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
                class_weight='balanced'
            )
            
            # Gradient Boosting
            self.models['gradient_boosting'] = GradientBoostingClassifier(
                n_estimators=150,
                learning_rate=0.1,
                max_depth=8,
                min_samples_split=5,
                random_state=42
            )
            
            # SVM
            self.models['svm'] = SVC(
                kernel='rbf',
                C=10.0,
                gamma='scale',
                probability=True,
                random_state=42,
                class_weight='balanced'
            )
            
            # Neural Network
            self.models['neural_network'] = MLPClassifier(
                hidden_layer_sizes=(128, 64, 32),
                activation='relu',
                solver='adam',
                alpha=0.001,
                learning_rate_init=0.001,
                max_iter=500,
                random_state=42,
                early_stopping=True
            )
            
            # XGBoost
            self.models['xgboost'] = xgb.XGBClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=8,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                use_label_encoder=False,
                eval_metric='mlogloss'
            )
            
            # 각 모델별 스케일러
            for model_name in self.models.keys():
                if model_name in ['svm', 'neural_network']:
                    self.scalers[model_name] = StandardScaler()
                else:
                    self.scalers[model_name] = RobustScaler()
                    
                # 특징 선택기
                self.feature_selectors[model_name] = SelectKBest(
                    score_func=f_classif, 
                    k=min(15, len(self.feature_names))
                )
            
            logger.info(f"✅ {len(self.models)}개 ML 모델 초기화 완료")
            
        except Exception as e:
            logger.error(f"ML 모델 초기화 실패: {e}")
    
    def load_trained_models(self):
        """학습된 모델들 로드"""
        try:
            models_loaded = 0
            
            for model_name in self.models.keys():
                model_file = os.path.join(self.models_path, f"{model_name}_regime_model.pkl")
                scaler_file = os.path.join(self.models_path, f"{model_name}_scaler.pkl")
                selector_file = os.path.join(self.models_path, f"{model_name}_selector.pkl")
                
                if all(os.path.exists(f) for f in [model_file, scaler_file, selector_file]):
                    try:
                        self.models[model_name] = joblib.load(model_file)
                        self.scalers[model_name] = joblib.load(scaler_file)
                        self.feature_selectors[model_name] = joblib.load(selector_file)
                        models_loaded += 1
                    except:
                        logger.warning(f"모델 {model_name} 로드 실패")
            
            # 앙상블 모델 로드
            ensemble_file = os.path.join(self.models_path, "ensemble_regime_model.pkl")
            if os.path.exists(ensemble_file):
                try:
                    self.ensemble_model = joblib.load(ensemble_file)
                    models_loaded += 1
                except:
                    logger.warning("앙상블 모델 로드 실패")
            
            # 레이블 인코더 로드
            encoder_file = os.path.join(self.models_path, "label_encoder.pkl")
            if os.path.exists(encoder_file):
                try:
                    self.label_encoder = joblib.load(encoder_file)
                except:
                    logger.warning("레이블 인코더 로드 실패")
            
            if models_loaded > 0:
                self.is_trained = True
                logger.info(f"✅ {models_loaded}개 학습된 모델 로드 완료")
            else:
                logger.info("학습된 모델이 없습니다. 새로운 학습이 필요합니다.")
                
        except Exception as e:
            logger.error(f"모델 로드 실패: {e}")
    
    async def train_models(self, training_data: List[Dict], test_size: float = 0.2) -> Dict:
        """모든 ML 모델 학습"""
        try:
            if not training_data or len(training_data) < 50:
                return {"error": "충분한 학습 데이터가 없습니다 (최소 50개 필요)"}
            
            logger.info(f"🧠 ML 모델 학습 시작 (데이터: {len(training_data)}개)")
            
            # 데이터 준비
            X, y = self.prepare_training_data(training_data)
            if X is None or y is None:
                return {"error": "학습 데이터 준비 실패"}
            
            # 레이블 인코딩
            y_encoded = self.label_encoder.fit_transform(y)
            
            # 학습/테스트 분할
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=test_size, stratify=y_encoded, random_state=42
            )
            
            # 개별 모델 학습
            model_results = {}
            for model_name, model in self.models.items():
                logger.info(f"📚 {model_name} 모델 학습 중...")
                result = await self.train_single_model(
                    model_name, model, X_train, X_test, y_train, y_test
                )
                model_results[model_name] = result
            
            # 앙상블 모델 구성
            ensemble_result = await self.create_ensemble_model(X_train, X_test, y_train, y_test)
            model_results['ensemble'] = ensemble_result
            
            # 최고 성능 모델 선정
            best_model = self.select_best_model(model_results)
            
            # 모델들 저장
            await self.save_all_models()
            
            # 학습 결과 저장
            await self.save_training_results(model_results, len(training_data))
            
            self.is_trained = True
            
            return {
                "training_completed": True,
                "training_samples": len(training_data),
                "test_samples": len(X_test),
                "model_results": model_results,
                "best_model": best_model,
                "ensemble_accuracy": ensemble_result.get('accuracy', 0)
            }
            
        except Exception as e:
            logger.error(f"ML 모델 학습 실패: {e}")
            return {"error": str(e)}
    
    def prepare_training_data(self, training_data: List[Dict]) -> Tuple[Optional[np.ndarray], Optional[List[str]]]:
        """학습 데이터 준비 및 전처리"""
        try:
            features = []
            labels = []
            
            for data in training_data:
                feature_vector = data.get('features')
                label = data.get('true_regime')
                
                if feature_vector and label and len(feature_vector) == len(self.feature_names):
                    features.append(feature_vector)
                    labels.append(label)
            
            if len(features) < 20:
                logger.error("유효한 학습 데이터가 너무 적습니다")
                return None, None
            
            X = np.array(features)
            y = labels
            
            # 이상치 제거
            X_clean, y_clean = self.remove_outliers(X, y)
            
            logger.info(f"✅ 학습 데이터 준비 완료: {len(X_clean)}개 샘플, {len(self.feature_names)}개 특징")
            
            return X_clean, y_clean
            
        except Exception as e:
            logger.error(f"학습 데이터 준비 실패: {e}")
            return None, None
    
    def remove_outliers(self, X: np.ndarray, y: List[str], threshold: float = 3.0) -> Tuple[np.ndarray, List[str]]:
        """이상치 제거"""
        try:
            # Z-score 기반 이상치 탐지
            z_scores = np.abs((X - np.mean(X, axis=0)) / np.std(X, axis=0))
            outlier_mask = np.any(z_scores > threshold, axis=1)
            
            # 이상치가 아닌 데이터만 선택
            clean_indices = ~outlier_mask
            X_clean = X[clean_indices]
            y_clean = [y[i] for i in range(len(y)) if clean_indices[i]]
            
            removed_count = len(X) - len(X_clean)
            if removed_count > 0:
                logger.info(f"이상치 {removed_count}개 제거됨")
            
            return X_clean, y_clean
            
        except Exception as e:
            logger.error(f"이상치 제거 실패: {e}")
            return X, y
    
    async def train_single_model(self, model_name: str, model, 
                                X_train: np.ndarray, X_test: np.ndarray,
                                y_train: np.ndarray, y_test: np.ndarray) -> Dict:
        """개별 모델 학습"""
        try:
            # 특징 스케일링
            scaler = self.scalers[model_name]
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # 특징 선택
            selector = self.feature_selectors[model_name]
            X_train_selected = selector.fit_transform(X_train_scaled, y_train)
            X_test_selected = selector.transform(X_test_scaled)
            
            # 하이퍼파라미터 튜닝 (일부 모델만)
            if model_name in ['random_forest', 'svm']:
                model = await self.hyperparameter_tuning(model_name, model, X_train_selected, y_train)
            
            # 모델 학습
            model.fit(X_train_selected, y_train)
            
            # 예측 및 평가
            y_pred = model.predict(X_test_selected)
            y_pred_proba = model.predict_proba(X_test_selected) if hasattr(model, 'predict_proba') else None
            
            # 성능 지표 계산
            accuracy = accuracy_score(y_test, y_pred)
            classification_rep = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            
            # 교차 검증
            cv_scores = cross_val_score(model, X_train_selected, y_train, cv=5, scoring='accuracy')
            
            # 특징 중요도
            feature_importance = {}
            if hasattr(model, 'feature_importances_'):
                selected_features = selector.get_support()
                feature_names_selected = [self.feature_names[i] for i in range(len(selected_features)) if selected_features[i]]
                importance_values = model.feature_importances_
                feature_importance = dict(zip(feature_names_selected, importance_values))
            
            # 업데이트된 모델 저장
            self.models[model_name] = model
            
            result = {
                'accuracy': accuracy,
                'cross_val_score': cv_scores.mean(),
                'classification_report': classification_rep,
                'feature_importance': feature_importance,
                'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
                'n_features_selected': X_train_selected.shape[1],
                'cv_std': cv_scores.std()
            }
            
            logger.info(f"✅ {model_name}: 정확도 {accuracy:.3f}, CV {cv_scores.mean():.3f}±{cv_scores.std():.3f}")
            
            return result
            
        except Exception as e:
            logger.error(f"{model_name} 모델 학습 실패: {e}")
            return {"error": str(e)}
    
    async def hyperparameter_tuning(self, model_name: str, model, X_train: np.ndarray, y_train: np.ndarray):
        """하이퍼파라미터 튜닝"""
        try:
            if model_name == 'random_forest':
                param_grid = {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 15, 20],
                    'min_samples_split': [2, 5, 10]
                }
            elif model_name == 'svm':
                param_grid = {
                    'C': [1, 10, 100],
                    'gamma': ['scale', 'auto', 0.001, 0.01]
                }
            else:
                return model
            
            # 그리드 서치 (3-fold CV로 빠르게)
            grid_search = GridSearchCV(
                model, param_grid, cv=3, scoring='accuracy', 
                n_jobs=-1, verbose=0
            )
            grid_search.fit(X_train, y_train)
            
            logger.info(f"{model_name} 최적 파라미터: {grid_search.best_params_}")
            
            return grid_search.best_estimator_
            
        except Exception as e:
            logger.error(f"{model_name} 하이퍼파라미터 튜닝 실패: {e}")
            return model
    
    async def create_ensemble_model(self, X_train: np.ndarray, X_test: np.ndarray,
                                  y_train: np.ndarray, y_test: np.ndarray) -> Dict:
        """앙상블 모델 생성"""
        try:
            # 성능 좋은 모델들만 선택
            good_models = []
            model_weights = []
            
            for model_name, model in self.models.items():
                if model_name in ['random_forest', 'gradient_boosting', 'xgboost']:
                    scaler = self.scalers[model_name]
                    selector = self.feature_selectors[model_name]
                    
                    X_train_processed = selector.transform(scaler.transform(X_train))
                    X_test_processed = selector.transform(scaler.transform(X_test))
                    
                    # 개별 모델 성능 확인
                    y_pred = model.predict(X_test_processed)
                    accuracy = accuracy_score(y_test, y_pred)
                    
                    if accuracy > 0.6:  # 60% 이상 정확도인 모델만
                        good_models.append((model_name, model))
                        model_weights.append(accuracy)
            
            if len(good_models) < 2:
                return {"error": "앙상블을 구성할 충분한 성능의 모델이 없습니다"}
            
            # 가중 투표 분류기
            estimators = [(name, model) for name, model in good_models]
            self.ensemble_model = VotingClassifier(
                estimators=estimators,
                voting='soft',  # 확률 기반 투표
                weights=model_weights
            )
            
            # 앙상블 학습 (전체 특징 사용)
            X_train_ensemble = StandardScaler().fit_transform(X_train)
            X_test_ensemble = StandardScaler().fit_transform(X_test)
            
            self.ensemble_model.fit(X_train_ensemble, y_train)
            
            # 앙상블 성능 평가
            y_pred_ensemble = self.ensemble_model.predict(X_test_ensemble)
            ensemble_accuracy = accuracy_score(y_test, y_pred_ensemble)
            
            # 교차 검증
            cv_scores = cross_val_score(
                self.ensemble_model, X_train_ensemble, y_train, cv=5, scoring='accuracy'
            )
            
            logger.info(f"✅ 앙상블 모델 생성: 정확도 {ensemble_accuracy:.3f}, CV {cv_scores.mean():.3f}")
            
            return {
                'accuracy': ensemble_accuracy,
                'cross_val_score': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'n_models': len(good_models),
                'model_names': [name for name, _ in good_models],
                'model_weights': model_weights
            }
            
        except Exception as e:
            logger.error(f"앙상블 모델 생성 실패: {e}")
            return {"error": str(e)}
    
    def select_best_model(self, model_results: Dict) -> str:
        """최고 성능 모델 선정"""
        try:
            best_model = "ensemble"
            best_score = 0.0
            
            for model_name, result in model_results.items():
                if isinstance(result, dict) and not result.get("error"):
                    # 정확도와 교차검증 점수의 조화평균
                    accuracy = result.get('accuracy', 0)
                    cv_score = result.get('cross_val_score', 0)
                    
                    if accuracy > 0 and cv_score > 0:
                        score = 2 * accuracy * cv_score / (accuracy + cv_score)
                        if score > best_score:
                            best_score = score
                            best_model = model_name
            
            logger.info(f"🏆 최고 성능 모델: {best_model} (점수: {best_score:.3f})")
            
            return best_model
            
        except Exception as e:
            logger.error(f"최고 모델 선정 실패: {e}")
            return "ensemble"
    
    async def predict_regime(self, features: np.ndarray) -> RegimeClassification:
        """시장 체제 예측"""
        try:
            if not self.is_trained:
                raise ValueError("모델이 학습되지 않았습니다")
            
            if len(features) != len(self.feature_names):
                raise ValueError(f"특징 개수가 맞지 않습니다: {len(features)} != {len(self.feature_names)}")
            
            features = features.reshape(1, -1)
            
            # 개별 모델 예측
            model_predictions = {}
            model_probabilities = {}
            
            for model_name, model in self.models.items():
                try:
                    scaler = self.scalers[model_name]
                    selector = self.feature_selectors[model_name]
                    
                    # 전처리
                    features_scaled = scaler.transform(features)
                    features_selected = selector.transform(features_scaled)
                    
                    # 예측
                    prediction = model.predict(features_selected)[0]
                    proba = model.predict_proba(features_selected)[0] if hasattr(model, 'predict_proba') else None
                    
                    # 레이블 디코딩
                    predicted_regime = self.label_encoder.inverse_transform([prediction])[0]
                    
                    model_predictions[model_name] = predicted_regime
                    if proba is not None:
                        model_probabilities[model_name] = dict(zip(
                            self.label_encoder.inverse_transform(range(len(proba))), proba
                        ))
                except Exception as model_error:
                    logger.warning(f"모델 {model_name} 예측 실패: {model_error}")
                    continue
            
            # 앙상블 예측
            ensemble_prediction = None
            ensemble_probabilities = None
            
            if self.ensemble_model:
                try:
                    features_scaled = StandardScaler().fit_transform(features)
                    ensemble_pred = self.ensemble_model.predict(features_scaled)[0]
                    ensemble_proba = self.ensemble_model.predict_proba(features_scaled)[0]
                    
                    ensemble_prediction = self.label_encoder.inverse_transform([ensemble_pred])[0]
                    ensemble_probabilities = dict(zip(
                        self.label_encoder.inverse_transform(range(len(ensemble_proba))), 
                        ensemble_proba
                    ))
                except Exception as ensemble_error:
                    logger.warning(f"앙상블 예측 실패: {ensemble_error}")
            
            # 최종 예측 결정 (앙상블 우선, 없으면 다수결)
            if ensemble_prediction and ensemble_probabilities:
                final_prediction = ensemble_prediction
                final_probabilities = ensemble_probabilities
                confidence = max(ensemble_probabilities.values())
            else:
                # 다수결 투표
                prediction_votes = list(model_predictions.values())
                if prediction_votes:
                    final_prediction = max(set(prediction_votes), key=prediction_votes.count)
                    
                    # 평균 확률 계산
                    if model_probabilities:
                        final_probabilities = defaultdict(float)
                        for proba_dict in model_probabilities.values():
                            for regime, prob in proba_dict.items():
                                final_probabilities[regime] += prob
                        
                        # 평균화
                        n_models = len(model_probabilities)
                        final_probabilities = {k: v/n_models for k, v in final_probabilities.items()}
                        confidence = final_probabilities.get(final_prediction, 0.5)
                    else:
                        final_probabilities = {final_prediction: 0.5}
                        confidence = 0.5
                else:
                    raise ValueError("모든 모델 예측이 실패했습니다")
            
            # 특징 중요도 계산 (Random Forest 기준)
            feature_importance = {}
            if 'random_forest' in self.models and hasattr(self.models['random_forest'], 'feature_importances_'):
                rf_model = self.models['random_forest']
                rf_selector = self.feature_selectors['random_forest']
                selected_features = rf_selector.get_support()
                feature_names_selected = [self.feature_names[i] for i in range(len(selected_features)) if selected_features[i]]
                
                if len(feature_names_selected) == len(rf_model.feature_importances_):
                    feature_importance = dict(zip(feature_names_selected, rf_model.feature_importances_))
            
            # 예측 결과 생성
            classification_result = RegimeClassification(
                regime_type=final_prediction,
                confidence=confidence,
                probability_distribution=dict(final_probabilities),
                feature_importance=feature_importance,
                ensemble_votes=model_predictions,
                prediction_timestamp=datetime.now()
            )
            
            # 예측 히스토리 업데이트
            self.prediction_history.append(classification_result)
            self.current_regime = final_prediction
            self.regime_confidence = confidence
            
            # 예측 결과 저장
            await self.save_prediction_result(classification_result, features.flatten())
            
            return classification_result
            
        except Exception as e:
            logger.error(f"체제 예측 실패: {e}")
            # 기본 예측 반환
            return RegimeClassification(
                regime_type="SIDEWAYS",
                confidence=0.5,
                probability_distribution={"SIDEWAYS": 0.5},
                feature_importance={},
                ensemble_votes={"error": "prediction_failed"},
                prediction_timestamp=datetime.now()
            )
    
    async def save_all_models(self):
        """모든 모델 저장"""
        try:
            # 개별 모델들 저장
            for model_name, model in self.models.items():
                joblib.dump(model, os.path.join(self.models_path, f"{model_name}_regime_model.pkl"))
                joblib.dump(self.scalers[model_name], os.path.join(self.models_path, f"{model_name}_scaler.pkl"))
                joblib.dump(self.feature_selectors[model_name], os.path.join(self.models_path, f"{model_name}_selector.pkl"))
            
            # 앙상블 모델 저장
            if self.ensemble_model:
                joblib.dump(self.ensemble_model, os.path.join(self.models_path, "ensemble_regime_model.pkl"))
            
            # 레이블 인코더 저장
            joblib.dump(self.label_encoder, os.path.join(self.models_path, "label_encoder.pkl"))
            
            logger.info("✅ 모든 모델 저장 완료")
            
        except Exception as e:
            logger.error(f"모델 저장 실패: {e}")
    
    async def save_training_results(self, model_results: Dict, training_samples: int):
        """학습 결과 저장"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for model_name, result in model_results.items():
                if isinstance(result, dict) and not result.get("error"):
                    cursor.execute('''
                        INSERT INTO ml_model_performance
                        (model_name, accuracy, precision_scores, recall_scores, f1_scores,
                         cross_val_score, training_samples, confusion_matrix, feature_importance,
                         hyperparameters, training_date)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        model_name,
                        result.get('accuracy', 0),
                        json.dumps(result.get('classification_report', {}).get('weighted avg', {})),
                        json.dumps(result.get('classification_report', {}).get('macro avg', {})),
                        json.dumps(result.get('classification_report', {}).get('macro avg', {})),
                        result.get('cross_val_score', 0),
                        training_samples,
                        json.dumps(result.get('confusion_matrix', [])),
                        json.dumps(result.get('feature_importance', {})),
                        json.dumps({}),  # 하이퍼파라미터는 별도 저장 필요시 구현
                        datetime.now().isoformat()
                    ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"학습 결과 저장 실패: {e}")
    
    async def save_prediction_result(self, classification: RegimeClassification, features: np.ndarray):
        """예측 결과 저장"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO ml_predictions
                (timestamp, predicted_regime, regime_id, confidence, probability_distribution,
                 feature_importance, ensemble_votes, input_features, model_version)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                classification.prediction_timestamp.isoformat(),
                classification.regime_type,
                list(self.regime_labels.values()).index(classification.regime_type),
                classification.confidence,
                json.dumps(classification.probability_distribution),
                json.dumps(classification.feature_importance),
                json.dumps(classification.ensemble_votes),
                json.dumps(features.tolist()),
                "v1.0"
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"예측 결과 저장 실패: {e}")
    
    async def get_model_diagnostics(self) -> Dict:
        """모델 진단 정보"""
        try:
            diagnostics = {
                "training_status": "trained" if self.is_trained else "untrained",
                "n_models": len(self.models),
                "current_regime": self.current_regime,
                "regime_confidence": self.regime_confidence,
                "prediction_history_length": len(self.prediction_history)
            }
            
            if self.is_trained:
                # 최근 성능 조회
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT model_name, accuracy, cross_val_score, training_date
                    FROM ml_model_performance 
                    ORDER BY training_date DESC 
                    LIMIT 10
                ''')
                
                recent_performance = cursor.fetchall()
                diagnostics["recent_performance"] = [
                    {
                        "model": row[0],
                        "accuracy": row[1],
                        "cross_val_score": row[2],
                        "training_date": row[3]
                    } for row in recent_performance
                ]
                
                # 최근 예측 히스토리
                if self.prediction_history:
                    recent_predictions = list(self.prediction_history)[-5:]
                    diagnostics["recent_predictions"] = [
                        {
                            "regime": pred.regime_type,
                            "confidence": pred.confidence,
                            "timestamp": pred.prediction_timestamp.isoformat()
                        } for pred in recent_predictions
                    ]
                
                conn.close()
            
            return diagnostics
            
        except Exception as e:
            logger.error(f"모델 진단 실패: {e}")
            return {"error": str(e)}

# 테스트 함수
async def test_ml_regime_classifier():
    """ML 체제 분류기 테스트"""
    print("🤖 머신러닝 체제 분류기 테스트")
    print("=" * 50)
    
    classifier = MLRegimeClassifier()
    
    # 진단 정보
    diagnostics = await classifier.get_model_diagnostics()
    print(f"📊 학습 상태: {diagnostics.get('training_status')}")
    print(f"🔢 모델 수: {diagnostics.get('n_models')}")
    
    if diagnostics.get('training_status') == 'trained':
        # 테스트 특징값
        test_features = np.array([
            0.03,   # price_trend_1d
            0.05,   # price_trend_7d  
            0.08,   # price_trend_30d
            0.7,    # trend_consistency
            0.04,   # volatility_1d
            0.03,   # volatility_7d
            0.035,  # volatility_30d
            0.1,    # volatility_regime_change
            0.2,    # volume_trend
            0.02,   # volume_volatility
            0.4,    # volume_price_correlation
            65,     # rsi_14
            0.05,   # macd_signal
            0.8,    # bollinger_position
            0.7,    # whale_activity
            0.1,    # exchange_flow
            0.6,    # hodler_behavior
            0.02,   # futures_basis
            0.01,   # funding_rate
            0.3,    # put_call_ratio
            65,     # fear_greed_index
            0.1,    # correlation_gold
            0.3,    # correlation_stocks
            -0.05   # dxy_impact
        ])
        
        # 예측 수행
        prediction = await classifier.predict_regime(test_features)
        
        print(f"\n🎯 예측된 체제: {prediction.regime_type}")
        print(f"🔥 신뢰도: {prediction.confidence:.1%}")
        print(f"📊 확률 분포:")
        for regime, prob in prediction.probability_distribution.items():
            print(f"   • {regime}: {prob:.1%}")
        
        print(f"🗳️ 모델별 투표:")
        for model, vote in prediction.ensemble_votes.items():
            print(f"   • {model}: {vote}")
        
        if prediction.feature_importance:
            print(f"🔍 주요 특징 (상위 5개):")
            sorted_features = sorted(prediction.feature_importance.items(), 
                                   key=lambda x: x[1], reverse=True)
            for feature, importance in sorted_features[:5]:
                print(f"   • {feature}: {importance:.3f}")
    else:
        print("⚠️ 모델이 학습되지 않았습니다")
        print("학습을 위해서는 충분한 레이블된 데이터가 필요합니다")
    
    print("\n" + "=" * 50)
    print("🎉 ML 체제 분류기 테스트 완료!")

if __name__ == "__main__":
    asyncio.run(test_ml_regime_classifier())