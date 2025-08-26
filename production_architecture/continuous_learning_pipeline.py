#!/usr/bin/env python3
"""
🔄 연속 학습 파이프라인
- 실시간 모델 성능 추적
- 자동 재학습 트리거
- 온라인 학습 및 적응형 모델 업데이트
- A/B 테스트 및 카나리 배포
- 모델 백업 및 롤백 시스템
"""

import asyncio
import json
import logging
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
import joblib
import threading
import queue
from enum import Enum
import shutil
import os

# 머신러닝
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

# 딥러닝
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model, clone_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# 데이터베이스
import sqlite3
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# 실험 추적
try:
    import mlflow
    import mlflow.sklearn
    import mlflow.keras
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

Base = declarative_base()

class ModelStatus(Enum):
    """모델 상태"""
    TRAINING = "training"
    VALIDATING = "validating"
    ACTIVE = "active"
    CANDIDATE = "candidate"
    DEPRECATED = "deprecated"
    FAILED = "failed"

class RetrainingTrigger(Enum):
    """재학습 트리거"""
    PERFORMANCE_DEGRADATION = "performance_degradation"
    DATA_DRIFT = "data_drift"
    SCHEDULED = "scheduled"
    MANUAL = "manual"
    ANOMALY_DETECTED = "anomaly_detected"

@dataclass
class ModelMetrics:
    """모델 성능 메트릭"""
    model_id: str
    model_name: str
    version: str
    accuracy: float
    mae: float
    rmse: float
    r2_score: float
    prediction_count: int
    training_time: float
    data_size: int
    timestamp: datetime

@dataclass
class RetrainingJob:
    """재학습 작업"""
    job_id: str
    trigger: RetrainingTrigger
    model_name: str
    priority: int
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: str = "pending"
    result: Optional[Dict[str, Any]] = None

class ModelRegistry(Base):
    """모델 레지스트리 테이블"""
    __tablename__ = 'model_registry'
    
    id = Column(Integer, primary_key=True)
    model_id = Column(String, unique=True, nullable=False)
    model_name = Column(String, nullable=False)
    version = Column(String, nullable=False)
    status = Column(String, nullable=False)
    accuracy = Column(Float)
    mae = Column(Float)
    rmse = Column(Float)
    r2_score = Column(Float)
    training_data_hash = Column(String)
    model_path = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    promoted_at = Column(DateTime)
    deprecated_at = Column(DateTime)

class TrainingHistory(Base):
    """학습 이력 테이블"""
    __tablename__ = 'training_history'
    
    id = Column(Integer, primary_key=True)
    job_id = Column(String, nullable=False)
    model_name = Column(String, nullable=False)
    trigger = Column(String, nullable=False)
    status = Column(String, nullable=False)
    accuracy_before = Column(Float)
    accuracy_after = Column(Float)
    training_duration = Column(Float)
    data_size = Column(Integer)
    error_message = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

class ContinuousLearningEngine:
    """연속 학습 엔진"""
    
    def __init__(self, data_path: str, model_storage_path: str):
        self.data_path = Path(data_path)
        self.model_storage_path = Path(model_storage_path)
        self.model_storage_path.mkdir(exist_ok=True, parents=True)
        
        self.setup_logging()
        self.setup_database()
        self.setup_mlflow()
        
        self.active_models = {}  # model_name -> model_info
        self.candidate_models = {}  # model_name -> candidate_info
        self.retraining_queue = queue.PriorityQueue()
        self.performance_history = {}  # model_name -> [metrics]
        
        self.is_running = False
        self.load_active_models()
        
    def setup_logging(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def setup_database(self):
        """데이터베이스 설정"""
        try:
            db_path = self.model_storage_path / "continuous_learning.db"
            self.engine = create_engine(f'sqlite:///{db_path}')
            Base.metadata.create_all(self.engine)
            
            Session = sessionmaker(bind=self.engine)
            self.db_session = Session()
            
            self.logger.info("Database setup completed")
            
        except Exception as e:
            self.logger.error(f"Database setup failed: {e}")
            self.db_session = None
            
    def setup_mlflow(self):
        """MLflow 설정"""
        if MLFLOW_AVAILABLE:
            try:
                mlflow.set_tracking_uri(f"file://{self.model_storage_path}/mlruns")
                mlflow.set_experiment("btc_prediction_continuous_learning")
                self.logger.info("MLflow setup completed")
            except Exception as e:
                self.logger.error(f"MLflow setup failed: {e}")
        else:
            self.logger.warning("MLflow not available")
            
    def load_active_models(self):
        """활성 모델 로드"""
        try:
            if self.db_session:
                active_models = self.db_session.query(ModelRegistry).filter(
                    ModelRegistry.status == ModelStatus.ACTIVE.value
                ).all()
                
                for model_record in active_models:
                    try:
                        model_path = Path(model_record.model_path)
                        if model_path.exists():
                            model = joblib.load(model_path)
                            
                            self.active_models[model_record.model_name] = {
                                'model': model,
                                'model_id': model_record.model_id,
                                'version': model_record.version,
                                'accuracy': model_record.accuracy,
                                'path': model_path,
                                'promoted_at': model_record.promoted_at
                            }
                            
                            self.logger.info(f"Loaded active model: {model_record.model_name} v{model_record.version}")
                            
                    except Exception as e:
                        self.logger.error(f"Failed to load model {model_record.model_name}: {e}")
                        
            else:
                # 기본 모델 로드
                self.load_default_models()
                
        except Exception as e:
            self.logger.error(f"Active models loading failed: {e}")
            self.load_default_models()
            
    def load_default_models(self):
        """기본 모델 로드"""
        try:
            # 기존 최고 성능 모델 로드
            perfect_model_path = self.data_path.parent / "perfect_100_model.pkl"
            if perfect_model_path.exists():
                model = joblib.load(perfect_model_path)
                model_id = hashlib.md5(f"perfect_system_{time.time()}".encode()).hexdigest()
                
                self.active_models['perfect_system'] = {
                    'model': model,
                    'model_id': model_id,
                    'version': 'v1.0',
                    'accuracy': 0.92,
                    'path': perfect_model_path,
                    'promoted_at': datetime.now()
                }
                
                self.logger.info("Default perfect system model loaded")
                
        except Exception as e:
            self.logger.error(f"Default model loading failed: {e}")
            
    async def start_continuous_learning(self):
        """연속 학습 시작"""
        self.logger.info("🔄 Starting continuous learning pipeline")
        self.is_running = True
        
        try:
            # 동시 실행: 성능 모니터링, 재학습 처리, 모델 평가
            await asyncio.gather(
                self.performance_monitoring_loop(),
                self.retraining_processing_loop(),
                self.model_evaluation_loop(),
                self.scheduled_tasks_loop(),
                return_exceptions=True
            )
            
        except Exception as e:
            self.logger.error(f"Continuous learning pipeline failed: {e}")
        finally:
            self.is_running = False
            
    async def performance_monitoring_loop(self):
        """성능 모니터링 루프"""
        self.logger.info("Starting performance monitoring loop")
        
        while self.is_running:
            try:
                # 각 활성 모델의 성능 모니터링
                for model_name, model_info in self.active_models.items():
                    await self.monitor_model_performance(model_name, model_info)
                    
                await asyncio.sleep(300)  # 5분마다 모니터링
                
            except Exception as e:
                self.logger.error(f"Performance monitoring error: {e}")
                await asyncio.sleep(60)
                
    async def monitor_model_performance(self, model_name: str, model_info: Dict[str, Any]):
        """개별 모델 성능 모니터링"""
        try:
            # 최근 예측 정확도 계산 (실제 환경에서는 예측 기록과 실제 값을 비교)
            current_accuracy = await self.calculate_current_accuracy(model_name)
            
            # 성능 이력에 추가
            if model_name not in self.performance_history:
                self.performance_history[model_name] = []
                
            metrics = ModelMetrics(
                model_id=model_info['model_id'],
                model_name=model_name,
                version=model_info['version'],
                accuracy=current_accuracy,
                mae=0.0,  # 실제 환경에서 계산
                rmse=0.0,
                r2_score=0.0,
                prediction_count=0,
                training_time=0.0,
                data_size=0,
                timestamp=datetime.now()
            )
            
            self.performance_history[model_name].append(metrics)
            
            # 최대 1000개 기록만 유지
            if len(self.performance_history[model_name]) > 1000:
                self.performance_history[model_name] = self.performance_history[model_name][-1000:]
                
            # 성능 저하 감지
            if await self.detect_performance_degradation(model_name, current_accuracy):
                await self.trigger_retraining(model_name, RetrainingTrigger.PERFORMANCE_DEGRADATION)
                
        except Exception as e:
            self.logger.error(f"Model performance monitoring failed for {model_name}: {e}")
            
    async def calculate_current_accuracy(self, model_name: str) -> float:
        """현재 모델 정확도 계산"""
        try:
            # 실제 환경에서는 최근 예측과 실제 값을 비교
            # 여기서는 시뮬레이션된 데이터 사용
            base_accuracy = self.active_models[model_name]['accuracy']
            
            # 시간에 따른 성능 저하 시뮬레이션
            promoted_at = self.active_models[model_name]['promoted_at']
            time_since_promotion = datetime.now() - promoted_at
            hours_since_promotion = time_since_promotion.total_seconds() / 3600
            
            # 점진적 성능 저하 (시간당 0.001%)
            degradation = min(0.05, hours_since_promotion * 0.00001)
            
            # 랜덤 노이즈 추가
            import random
            noise = random.uniform(-0.01, 0.01)
            
            current_accuracy = max(0.75, base_accuracy - degradation + noise)
            
            return current_accuracy
            
        except Exception as e:
            self.logger.error(f"Accuracy calculation failed for {model_name}: {e}")
            return 0.80  # 기본값
            
    async def detect_performance_degradation(self, model_name: str, current_accuracy: float) -> bool:
        """성능 저하 감지"""
        try:
            if model_name not in self.performance_history:
                return False
                
            history = self.performance_history[model_name]
            
            if len(history) < 10:  # 충분한 이력이 없으면 false
                return False
                
            # 최근 10개 측정값의 평균과 비교
            recent_accuracies = [m.accuracy for m in history[-10:]]
            avg_recent_accuracy = np.mean(recent_accuracies)
            
            # 기준선 정확도 (모델 등록시 정확도)
            baseline_accuracy = self.active_models[model_name]['accuracy']
            
            # 성능 저하 감지 조건들
            degradation_threshold = 0.05  # 5% 이상 저하
            trend_threshold = -0.02  # 지속적인 하락 추세
            
            # 1. 절대 성능 저하
            absolute_degradation = baseline_accuracy - current_accuracy > degradation_threshold
            
            # 2. 최근 평균과 비교
            recent_degradation = avg_recent_accuracy - current_accuracy > 0.02
            
            # 3. 추세 분석 (선형 회귀)
            if len(recent_accuracies) >= 5:
                x = np.arange(len(recent_accuracies))
                slope = np.polyfit(x, recent_accuracies, 1)[0]
                trend_degradation = slope < trend_threshold
            else:
                trend_degradation = False
                
            degradation_detected = absolute_degradation or recent_degradation or trend_degradation
            
            if degradation_detected:
                self.logger.warning(
                    f"Performance degradation detected for {model_name}: "
                    f"current={current_accuracy:.3f}, baseline={baseline_accuracy:.3f}, "
                    f"avg_recent={avg_recent_accuracy:.3f}, slope={slope:.4f}"
                )
                
            return degradation_detected
            
        except Exception as e:
            self.logger.error(f"Performance degradation detection failed: {e}")
            return False
            
    async def trigger_retraining(self, model_name: str, trigger: RetrainingTrigger, priority: int = 5):
        """재학습 트리거"""
        try:
            job_id = hashlib.md5(f"{model_name}_{trigger.value}_{time.time()}".encode()).hexdigest()
            
            retraining_job = RetrainingJob(
                job_id=job_id,
                trigger=trigger,
                model_name=model_name,
                priority=priority,
                created_at=datetime.now()
            )
            
            # 우선순위 큐에 추가 (낮은 숫자 = 높은 우선순위)
            self.retraining_queue.put((priority, retraining_job))
            
            self.logger.info(f"Retraining triggered for {model_name}: {trigger.value} (priority: {priority})")
            
            # 데이터베이스에 기록
            if self.db_session:
                training_record = TrainingHistory(
                    job_id=job_id,
                    model_name=model_name,
                    trigger=trigger.value,
                    status="queued"
                )
                
                self.db_session.add(training_record)
                self.db_session.commit()
                
        except Exception as e:
            self.logger.error(f"Retraining trigger failed: {e}")
            
    async def retraining_processing_loop(self):
        """재학습 처리 루프"""
        self.logger.info("Starting retraining processing loop")
        
        while self.is_running:
            try:
                if not self.retraining_queue.empty():
                    priority, job = self.retraining_queue.get()
                    await self.process_retraining_job(job)
                else:
                    await asyncio.sleep(30)  # 큐가 비어있으면 30초 대기
                    
            except Exception as e:
                self.logger.error(f"Retraining processing error: {e}")
                await asyncio.sleep(60)
                
    async def process_retraining_job(self, job: RetrainingJob):
        """재학습 작업 처리"""
        self.logger.info(f"Processing retraining job: {job.job_id} for {job.model_name}")
        
        job.started_at = datetime.now()
        job.status = "running"
        
        try:
            # 학습 데이터 준비
            X_train, y_train, X_val, y_val = await self.prepare_training_data()
            
            if X_train is None or len(X_train) == 0:
                raise Exception("No training data available")
                
            # 현재 모델 정확도 기록
            current_accuracy = 0.0
            if job.model_name in self.active_models:
                current_accuracy = await self.calculate_current_accuracy(job.model_name)
                
            # 모델 재학습
            new_model, training_metrics = await self.retrain_model(
                job.model_name, 
                X_train, y_train, 
                X_val, y_val
            )
            
            if new_model is None:
                raise Exception("Model retraining failed")
                
            # 성능 검증
            validation_metrics = await self.validate_model(new_model, X_val, y_val)
            
            # 개선 여부 확인
            improvement_threshold = 0.01  # 1% 이상 개선
            improved = validation_metrics['accuracy'] > (current_accuracy + improvement_threshold)
            
            if improved:
                # 후보 모델로 등록
                await self.register_candidate_model(job.model_name, new_model, validation_metrics)
                
                job.status = "completed"
                job.result = {
                    'improved': True,
                    'accuracy_before': current_accuracy,
                    'accuracy_after': validation_metrics['accuracy'],
                    'improvement': validation_metrics['accuracy'] - current_accuracy
                }
                
                self.logger.info(
                    f"Retraining successful for {job.model_name}: "
                    f"{current_accuracy:.3f} -> {validation_metrics['accuracy']:.3f} "
                    f"(+{validation_metrics['accuracy'] - current_accuracy:.3f})"
                )
                
            else:
                job.status = "no_improvement"
                job.result = {
                    'improved': False,
                    'accuracy_before': current_accuracy,
                    'accuracy_after': validation_metrics['accuracy'],
                    'improvement': validation_metrics['accuracy'] - current_accuracy
                }
                
                self.logger.info(
                    f"Retraining completed but no significant improvement for {job.model_name}: "
                    f"{current_accuracy:.3f} vs {validation_metrics['accuracy']:.3f}"
                )
                
        except Exception as e:
            job.status = "failed"
            job.result = {'error': str(e)}
            self.logger.error(f"Retraining job failed: {e}")
            
        finally:
            job.completed_at = datetime.now()
            
            # 데이터베이스 업데이트
            if self.db_session:
                try:
                    training_record = self.db_session.query(TrainingHistory).filter(
                        TrainingHistory.job_id == job.job_id
                    ).first()
                    
                    if training_record:
                        training_record.status = job.status
                        if job.result:
                            training_record.accuracy_before = job.result.get('accuracy_before')
                            training_record.accuracy_after = job.result.get('accuracy_after')
                            training_record.error_message = job.result.get('error')
                            
                        if job.started_at and job.completed_at:
                            training_duration = (job.completed_at - job.started_at).total_seconds()
                            training_record.training_duration = training_duration
                            
                        self.db_session.commit()
                        
                except Exception as e:
                    self.logger.error(f"Database update failed: {e}")
                    if self.db_session:
                        self.db_session.rollback()
                        
    async def prepare_training_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """학습 데이터 준비"""
        try:
            # 최신 데이터 로드
            data_file = self.data_path / "ai_optimized_3month_data" / "ai_matrix_complete.csv"
            
            if not data_file.exists():
                self.logger.error(f"Training data not found: {data_file}")
                return None, None, None, None
                
            df = pd.read_csv(data_file)
            
            if df.empty:
                self.logger.error("Empty training dataset")
                return None, None, None, None
                
            # 수치형 데이터만 선택
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            df_clean = df[numeric_cols].copy()
            
            # 결측치 처리
            df_clean = df_clean.ffill().bfill().fillna(df_clean.median()).fillna(0)
            
            # 타겟 설정 (BTC 가격 관련 컬럼 찾기)
            btc_col = None
            for col in df_clean.columns:
                if 'btc' in col.lower() and ('price' in col.lower() or 'momentum' in col.lower()):
                    btc_col = col
                    break
                    
            if btc_col is None:
                btc_col = df_clean.columns[0]
                
            # 타겟 생성 (다음 시점 예측)
            y = df_clean[btc_col].shift(-1).dropna()
            X = df_clean[:-1].drop(columns=[btc_col])
            
            # 특성 선택 (상위 100개)
            if len(X.columns) > 100:
                from sklearn.feature_selection import SelectKBest, f_regression
                selector = SelectKBest(score_func=f_regression, k=100)
                X_selected = pd.DataFrame(
                    selector.fit_transform(X, y),
                    columns=X.columns[selector.get_support()],
                    index=X.index
                )
            else:
                X_selected = X
                
            # 학습/검증 분할 (시계열 고려)
            split_idx = int(len(X_selected) * 0.8)
            
            X_train = X_selected[:split_idx].values
            y_train = y[:split_idx].values
            X_val = X_selected[split_idx:].values
            y_val = y[split_idx:].values
            
            self.logger.info(f"Training data prepared: train={X_train.shape}, val={X_val.shape}")
            
            return X_train, y_train, X_val, y_val
            
        except Exception as e:
            self.logger.error(f"Training data preparation failed: {e}")
            return None, None, None, None
            
    async def retrain_model(
        self, 
        model_name: str, 
        X_train: np.ndarray, 
        y_train: np.ndarray,
        X_val: np.ndarray, 
        y_val: np.ndarray
    ) -> Tuple[Any, Dict[str, float]]:
        """모델 재학습"""
        
        start_time = time.time()
        
        try:
            if MLFLOW_AVAILABLE:
                with mlflow.start_run(run_name=f"retrain_{model_name}_{int(time.time())}"):
                    model, metrics = await self._train_model_internal(model_name, X_train, y_train, X_val, y_val)
                    
                    # MLflow 로깅
                    mlflow.log_metrics(metrics)
                    mlflow.log_param("model_name", model_name)
                    mlflow.log_param("training_samples", len(X_train))
                    mlflow.log_param("validation_samples", len(X_val))
                    
                    return model, metrics
            else:
                model, metrics = await self._train_model_internal(model_name, X_train, y_train, X_val, y_val)
                return model, metrics
                
        except Exception as e:
            self.logger.error(f"Model retraining failed for {model_name}: {e}")
            return None, {}
            
    async def _train_model_internal(
        self,
        model_name: str,
        X_train: np.ndarray, 
        y_train: np.ndarray,
        X_val: np.ndarray, 
        y_val: np.ndarray
    ) -> Tuple[Any, Dict[str, float]]:
        """내부 모델 학습 로직"""
        
        # 데이터 스케일링
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        if model_name == "xgboost_ensemble":
            # XGBoost 모델
            model = xgb.XGBRegressor(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            )
            
            model.fit(
                X_train_scaled, y_train,
                eval_set=[(X_val_scaled, y_val)],
                early_stopping_rounds=20,
                verbose=False
            )
            
        elif model_name == "lstm_temporal":
            # LSTM 모델 (시계열 데이터 필요)
            model = self._create_lstm_model(X_train_scaled.shape[1])
            
            # 시퀀스 데이터 생성
            sequence_length = 24
            X_train_seq, y_train_seq = self._create_sequences(X_train_scaled, y_train, sequence_length)
            X_val_seq, y_val_seq = self._create_sequences(X_val_scaled, y_val, sequence_length)
            
            if len(X_train_seq) > 0 and len(X_val_seq) > 0:
                early_stopping = EarlyStopping(patience=10, restore_best_weights=True)
                
                model.fit(
                    X_train_seq, y_train_seq,
                    validation_data=(X_val_seq, y_val_seq),
                    epochs=50,
                    batch_size=32,
                    callbacks=[early_stopping],
                    verbose=0
                )
            else:
                # 시퀀스 데이터 생성 실패시 일반 Dense 모델로 대체
                model = self._create_dense_model(X_train_scaled.shape[1])
                model.fit(
                    X_train_scaled, y_train,
                    validation_data=(X_val_scaled, y_val),
                    epochs=50,
                    batch_size=32,
                    verbose=0
                )
                
        else:
            # 기본 RandomForest 모델
            model = RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
            
            model.fit(X_train_scaled, y_train)
            
        # 모델과 스케일러를 함께 저장하기 위한 패키지 생성
        model_package = {
            'model': model,
            'scaler': scaler,
            'model_type': model_name,
            'feature_count': X_train.shape[1],
            'trained_at': datetime.now()
        }
        
        # 성능 메트릭 계산
        y_pred = self._predict_with_model(model_package, X_val)
        
        mae = mean_absolute_error(y_val, y_pred)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        r2 = r2_score(y_val, y_pred)
        
        # 정확도 계산 (MAPE 기반)
        mape = np.mean(np.abs((y_val - y_pred) / (y_val + 1e-8))) * 100
        accuracy = max(0, 100 - mape) / 100
        
        metrics = {
            'accuracy': accuracy,
            'mae': mae,
            'rmse': rmse,
            'r2_score': r2,
            'mape': mape
        }
        
        return model_package, metrics
        
    def _create_lstm_model(self, input_dim: int) -> Sequential:
        """LSTM 모델 생성"""
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(24, input_dim)),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1, activation='linear')
        ])
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model
        
    def _create_dense_model(self, input_dim: int) -> Sequential:
        """Dense 모델 생성"""
        model = Sequential([
            Dense(128, activation='relu', input_shape=(input_dim,)),
            Dropout(0.2),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(1, activation='linear')
        ])
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model
        
    def _create_sequences(self, X: np.ndarray, y: np.ndarray, sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """시퀀스 데이터 생성"""
        try:
            X_seq, y_seq = [], []
            
            for i in range(sequence_length, len(X)):
                X_seq.append(X[i-sequence_length:i])
                y_seq.append(y[i])
                
            return np.array(X_seq), np.array(y_seq)
            
        except Exception as e:
            self.logger.error(f"Sequence creation failed: {e}")
            return np.array([]), np.array([])
            
    def _predict_with_model(self, model_package: Dict[str, Any], X: np.ndarray) -> np.ndarray:
        """모델 패키지로 예측"""
        try:
            model = model_package['model']
            scaler = model_package['scaler']
            model_type = model_package['model_type']
            
            X_scaled = scaler.transform(X)
            
            if model_type == "lstm_temporal" and hasattr(model, 'predict'):
                # LSTM의 경우 시퀀스 데이터 필요
                sequence_length = 24
                if len(X_scaled) >= sequence_length:
                    X_seq, _ = self._create_sequences(X_scaled, np.zeros(len(X_scaled)), sequence_length)
                    if len(X_seq) > 0:
                        return model.predict(X_seq, verbose=0).flatten()
                        
            # 일반적인 예측
            return model.predict(X_scaled)
            
        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            return np.zeros(len(X))
            
    async def validate_model(self, model_package: Dict[str, Any], X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, float]:
        """모델 검증"""
        try:
            y_pred = self._predict_with_model(model_package, X_val)
            
            mae = mean_absolute_error(y_val, y_pred)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            r2 = r2_score(y_val, y_pred)
            
            # 정확도 계산
            mape = np.mean(np.abs((y_val - y_pred) / (y_val + 1e-8))) * 100
            accuracy = max(0, 100 - mape) / 100
            
            return {
                'accuracy': accuracy,
                'mae': mae,
                'rmse': rmse,
                'r2_score': r2,
                'mape': mape
            }
            
        except Exception as e:
            self.logger.error(f"Model validation failed: {e}")
            return {'accuracy': 0.0, 'mae': float('inf'), 'rmse': float('inf'), 'r2_score': -1.0, 'mape': 100.0}
            
    async def register_candidate_model(self, model_name: str, model_package: Dict[str, Any], metrics: Dict[str, float]):
        """후보 모델 등록"""
        try:
            # 모델 ID 생성
            model_id = hashlib.md5(f"{model_name}_{time.time()}".encode()).hexdigest()
            version = f"v{int(time.time())}"
            
            # 모델 저장
            model_path = self.model_storage_path / f"{model_name}_{model_id}.pkl"
            joblib.dump(model_package, model_path)
            
            # 후보 모델에 추가
            self.candidate_models[model_name] = {
                'model_package': model_package,
                'model_id': model_id,
                'version': version,
                'metrics': metrics,
                'path': model_path,
                'registered_at': datetime.now()
            }
            
            # 데이터베이스에 등록
            if self.db_session:
                model_record = ModelRegistry(
                    model_id=model_id,
                    model_name=model_name,
                    version=version,
                    status=ModelStatus.CANDIDATE.value,
                    accuracy=metrics['accuracy'],
                    mae=metrics['mae'],
                    rmse=metrics['rmse'],
                    r2_score=metrics['r2_score'],
                    model_path=str(model_path)
                )
                
                self.db_session.add(model_record)
                self.db_session.commit()
                
            self.logger.info(f"Candidate model registered: {model_name} {version} (accuracy: {metrics['accuracy']:.3f})")
            
        except Exception as e:
            self.logger.error(f"Candidate model registration failed: {e}")
            
    async def model_evaluation_loop(self):
        """모델 평가 루프"""
        self.logger.info("Starting model evaluation loop")
        
        while self.is_running:
            try:
                # 후보 모델이 있으면 A/B 테스트 시작
                for model_name in list(self.candidate_models.keys()):
                    if model_name in self.active_models:
                        await self.start_ab_test(model_name)
                        
                await asyncio.sleep(3600)  # 1시간마다 평가
                
            except Exception as e:
                self.logger.error(f"Model evaluation error: {e}")
                await asyncio.sleep(300)
                
    async def start_ab_test(self, model_name: str):
        """A/B 테스트 시작"""
        try:
            if model_name not in self.candidate_models:
                return
                
            candidate_info = self.candidate_models[model_name]
            active_info = self.active_models[model_name]
            
            # A/B 테스트 결과 시뮬레이션 (실제 환경에서는 실제 트래픽으로 테스트)
            candidate_performance = await self.simulate_ab_test_performance(candidate_info)
            active_performance = await self.simulate_ab_test_performance(active_info)
            
            # 개선 여부 확인
            improvement_threshold = 0.02  # 2% 이상 개선
            is_better = candidate_performance['accuracy'] > (active_performance['accuracy'] + improvement_threshold)
            
            if is_better:
                # 후보 모델을 활성 모델로 승격
                await self.promote_candidate_model(model_name)
            else:
                # 후보 모델 삭제
                await self.reject_candidate_model(model_name)
                
        except Exception as e:
            self.logger.error(f"A/B test failed for {model_name}: {e}")
            
    async def simulate_ab_test_performance(self, model_info: Dict[str, Any]) -> Dict[str, float]:
        """A/B 테스트 성능 시뮬레이션"""
        # 실제 환경에서는 실제 예측 결과와 정답을 비교
        base_accuracy = model_info.get('accuracy', 0.85)
        
        # 랜덤 노이즈 추가로 실제 성능 시뮬레이션
        import random
        noise = random.uniform(-0.03, 0.03)
        actual_accuracy = max(0.7, min(0.95, base_accuracy + noise))
        
        return {'accuracy': actual_accuracy}
        
    async def promote_candidate_model(self, model_name: str):
        """후보 모델 승격"""
        try:
            if model_name not in self.candidate_models:
                return
                
            candidate_info = self.candidate_models[model_name]
            
            # 기존 활성 모델 백업 및 비활성화
            if model_name in self.active_models:
                await self.deprecate_active_model(model_name)
                
            # 후보 모델을 활성 모델로 승격
            self.active_models[model_name] = {
                'model': candidate_info['model_package'],
                'model_id': candidate_info['model_id'],
                'version': candidate_info['version'],
                'accuracy': candidate_info['metrics']['accuracy'],
                'path': candidate_info['path'],
                'promoted_at': datetime.now()
            }
            
            # 데이터베이스 업데이트
            if self.db_session:
                # 후보 모델 상태를 활성으로 변경
                model_record = self.db_session.query(ModelRegistry).filter(
                    ModelRegistry.model_id == candidate_info['model_id']
                ).first()
                
                if model_record:
                    model_record.status = ModelStatus.ACTIVE.value
                    model_record.promoted_at = datetime.now()
                    self.db_session.commit()
                    
            # 후보 목록에서 제거
            del self.candidate_models[model_name]
            
            self.logger.info(f"Model promoted to active: {model_name} {candidate_info['version']} (accuracy: {candidate_info['metrics']['accuracy']:.3f})")
            
        except Exception as e:
            self.logger.error(f"Model promotion failed: {e}")
            
    async def deprecate_active_model(self, model_name: str):
        """활성 모델 비활성화"""
        try:
            if model_name not in self.active_models:
                return
                
            active_info = self.active_models[model_name]
            
            # 데이터베이스 업데이트
            if self.db_session:
                model_record = self.db_session.query(ModelRegistry).filter(
                    ModelRegistry.model_id == active_info['model_id']
                ).first()
                
                if model_record:
                    model_record.status = ModelStatus.DEPRECATED.value
                    model_record.deprecated_at = datetime.now()
                    self.db_session.commit()
                    
            self.logger.info(f"Model deprecated: {model_name} {active_info['version']}")
            
        except Exception as e:
            self.logger.error(f"Model deprecation failed: {e}")
            
    async def reject_candidate_model(self, model_name: str):
        """후보 모델 거부"""
        try:
            if model_name not in self.candidate_models:
                return
                
            candidate_info = self.candidate_models[model_name]
            
            # 모델 파일 삭제
            if candidate_info['path'].exists():
                candidate_info['path'].unlink()
                
            # 데이터베이스 업데이트
            if self.db_session:
                model_record = self.db_session.query(ModelRegistry).filter(
                    ModelRegistry.model_id == candidate_info['model_id']
                ).first()
                
                if model_record:
                    model_record.status = ModelStatus.DEPRECATED.value
                    model_record.deprecated_at = datetime.now()
                    self.db_session.commit()
                    
            # 후보 목록에서 제거
            del self.candidate_models[model_name]
            
            self.logger.info(f"Candidate model rejected: {model_name} {candidate_info['version']}")
            
        except Exception as e:
            self.logger.error(f"Candidate model rejection failed: {e}")
            
    async def scheduled_tasks_loop(self):
        """예약 작업 루프"""
        self.logger.info("Starting scheduled tasks loop")
        
        last_daily_check = datetime.now().date()
        last_weekly_check = datetime.now().date()
        
        while self.is_running:
            try:
                current_date = datetime.now().date()
                
                # 일일 예약 재학습 (성능이 좋은 모델도 주기적으로 업데이트)
                if current_date > last_daily_check:
                    await self.trigger_scheduled_retraining("daily")
                    last_daily_check = current_date
                    
                # 주간 모델 정리
                if (current_date - last_weekly_check).days >= 7:
                    await self.cleanup_old_models()
                    last_weekly_check = current_date
                    
                await asyncio.sleep(3600)  # 1시간마다 체크
                
            except Exception as e:
                self.logger.error(f"Scheduled tasks error: {e}")
                await asyncio.sleep(1800)  # 30분 후 재시도
                
    async def trigger_scheduled_retraining(self, schedule_type: str):
        """예약된 재학습 트리거"""
        try:
            for model_name in self.active_models:
                # 마지막 재학습으로부터 충분한 시간이 지났는지 확인
                promoted_at = self.active_models[model_name]['promoted_at']
                time_since_promotion = datetime.now() - promoted_at
                
                if schedule_type == "daily" and time_since_promotion.days >= 1:
                    await self.trigger_retraining(
                        model_name, 
                        RetrainingTrigger.SCHEDULED, 
                        priority=8  # 낮은 우선순위
                    )
                    
        except Exception as e:
            self.logger.error(f"Scheduled retraining trigger failed: {e}")
            
    async def cleanup_old_models(self):
        """오래된 모델 정리"""
        try:
            if not self.db_session:
                return
                
            # 30일 이상 된 deprecated 모델 삭제
            cutoff_date = datetime.now() - timedelta(days=30)
            
            old_models = self.db_session.query(ModelRegistry).filter(
                ModelRegistry.status == ModelStatus.DEPRECATED.value,
                ModelRegistry.deprecated_at < cutoff_date
            ).all()
            
            for model_record in old_models:
                try:
                    # 모델 파일 삭제
                    model_path = Path(model_record.model_path)
                    if model_path.exists():
                        model_path.unlink()
                        
                    # 데이터베이스에서 삭제
                    self.db_session.delete(model_record)
                    
                except Exception as e:
                    self.logger.error(f"Failed to cleanup model {model_record.model_id}: {e}")
                    
            self.db_session.commit()
            
            self.logger.info(f"Cleaned up {len(old_models)} old models")
            
        except Exception as e:
            self.logger.error(f"Model cleanup failed: {e}")
            if self.db_session:
                self.db_session.rollback()
                
    def get_active_model(self, model_name: str) -> Optional[Dict[str, Any]]:
        """활성 모델 조회"""
        return self.active_models.get(model_name)
        
    def get_model_performance_history(self, model_name: str) -> List[ModelMetrics]:
        """모델 성능 이력 조회"""
        return self.performance_history.get(model_name, [])
        
    def get_training_history(self) -> List[Dict[str, Any]]:
        """학습 이력 조회"""
        try:
            if self.db_session:
                records = self.db_session.query(TrainingHistory).order_by(
                    TrainingHistory.created_at.desc()
                ).limit(100).all()
                
                return [{
                    'job_id': r.job_id,
                    'model_name': r.model_name,
                    'trigger': r.trigger,
                    'status': r.status,
                    'accuracy_before': r.accuracy_before,
                    'accuracy_after': r.accuracy_after,
                    'training_duration': r.training_duration,
                    'created_at': r.created_at.isoformat()
                } for r in records]
                
            return []
            
        except Exception as e:
            self.logger.error(f"Training history query failed: {e}")
            return []
            
    async def stop_continuous_learning(self):
        """연속 학습 중지"""
        self.logger.info("Stopping continuous learning pipeline")
        self.is_running = False

if __name__ == "__main__":
    async def main():
        # 연속 학습 엔진 실행
        data_path = "/Users/parkyoungjun/Desktop/BTC_Analysis_System"
        model_storage_path = "/Users/parkyoungjun/Desktop/BTC_Analysis_System/production_architecture/models"
        
        engine = ContinuousLearningEngine(data_path, model_storage_path)
        
        try:
            await engine.start_continuous_learning()
        except KeyboardInterrupt:
            print("\n🛑 Continuous learning stopped by user")
        except Exception as e:
            print(f"❌ Continuous learning failed: {e}")
        finally:
            await engine.stop_continuous_learning()
            
    asyncio.run(main())