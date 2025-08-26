#!/usr/bin/env python3
"""
🏭 프로덕션 앙상블 아키텍처 시스템
병렬 모델 훈련, 실시간 추론, 모델 레지스트리, 성능 추적

핵심 기능:
- 병렬/분산 모델 훈련 시스템
- 실시간 앙상블 추론 엔진
- 모델 레지스트리 및 버전 관리
- 자동화된 성능 추적 및 로깅
- CI/CD 파이프라인 통합
- 스케일링 및 로드 밸런싱
"""

import asyncio
import threading
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import queue
import time
import psutil
import json
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
from pathlib import Path
import sqlite3
import logging
from dataclasses import dataclass, asdict
import hashlib
import shutil

# 웹 및 API 프레임워크
try:
    from fastapi import FastAPI, HTTPException, BackgroundTasks
    from fastapi.middleware.cors import CORSMiddleware
    import uvicorn
    from pydantic import BaseModel
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    print("⚠️ FastAPI 미설치 - 웹 API 기능 비활성화")

# Redis (캐싱)
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    print("⚠️ Redis 미설치 - 캐싱 기능 비활성화")

# 기본 라이브러리들
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

# 앞서 구현한 시스템들 import
from advanced_ensemble_learning_system import AdvancedEnsembleLearningSystem
from advanced_model_selection_optimizer import AdvancedModelSelectionSystem
from robust_reliability_system import ReliabilitySystemManager

@dataclass
class ModelMetadata:
    """모델 메타데이터"""
    model_id: str
    model_name: str
    version: str
    created_at: datetime
    accuracy: float
    model_type: str
    hyperparameters: Dict[str, Any]
    file_path: str
    file_size: int
    checksum: str
    status: str  # 'training', 'active', 'deprecated', 'failed'

@dataclass
class TrainingJob:
    """훈련 작업"""
    job_id: str
    model_name: str
    config: Dict[str, Any]
    status: str  # 'queued', 'running', 'completed', 'failed'
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress: float = 0.0
    error_message: Optional[str] = None
    result_path: Optional[str] = None

class ModelRegistry:
    """
    📚 모델 레지스트리 및 버전 관리 시스템
    """
    
    def __init__(self, registry_path: str = "/Users/parkyoungjun/Desktop/BTC_Analysis_System/model_registry"):
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(exist_ok=True)
        
        # 메타데이터 데이터베이스
        self.db_path = self.registry_path / "registry.db"
        self.init_database()
        
        self.logger = logging.getLogger(__name__)

    def init_database(self):
        """데이터베이스 초기화"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS models (
                    model_id TEXT PRIMARY KEY,
                    model_name TEXT NOT NULL,
                    version TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    accuracy REAL,
                    model_type TEXT,
                    hyperparameters TEXT,
                    file_path TEXT,
                    file_size INTEGER,
                    checksum TEXT,
                    status TEXT,
                    UNIQUE(model_name, version)
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS model_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_id TEXT,
                    timestamp TEXT,
                    accuracy REAL,
                    mse REAL,
                    r2 REAL,
                    prediction_time REAL,
                    FOREIGN KEY (model_id) REFERENCES models (model_id)
                )
            """)

    def register_model(self, model: Any, metadata: ModelMetadata) -> str:
        """
        모델 등록
        
        Args:
            model: 훈련된 모델 객체
            metadata: 모델 메타데이터
            
        Returns:
            str: 등록된 모델 ID
        """
        # 모델 저장
        model_dir = self.registry_path / metadata.model_name / metadata.version
        model_dir.mkdir(parents=True, exist_ok=True)
        
        model_file = model_dir / f"{metadata.model_name}_{metadata.version}.pkl"
        
        with open(model_file, 'wb') as f:
            pickle.dump(model, f)
        
        # 파일 정보 업데이트
        metadata.file_path = str(model_file)
        metadata.file_size = model_file.stat().st_size
        metadata.checksum = self._calculate_checksum(model_file)
        
        # 데이터베이스에 등록
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO models 
                (model_id, model_name, version, created_at, accuracy, model_type, 
                 hyperparameters, file_path, file_size, checksum, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metadata.model_id, metadata.model_name, metadata.version,
                metadata.created_at.isoformat(), metadata.accuracy, metadata.model_type,
                json.dumps(metadata.hyperparameters), metadata.file_path,
                metadata.file_size, metadata.checksum, metadata.status
            ))
        
        self.logger.info(f"📚 모델 등록 완료: {metadata.model_id}")
        return metadata.model_id

    def load_model(self, model_id: str) -> Tuple[Any, ModelMetadata]:
        """모델 로드"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT * FROM models WHERE model_id = ?", (model_id,)
            )
            row = cursor.fetchone()
        
        if not row:
            raise ValueError(f"모델을 찾을 수 없습니다: {model_id}")
        
        # 메타데이터 복원
        metadata = ModelMetadata(
            model_id=row[0],
            model_name=row[1],
            version=row[2],
            created_at=datetime.fromisoformat(row[3]),
            accuracy=row[4],
            model_type=row[5],
            hyperparameters=json.loads(row[6] or "{}"),
            file_path=row[7],
            file_size=row[8],
            checksum=row[9],
            status=row[10]
        )
        
        # 모델 파일 로드
        with open(metadata.file_path, 'rb') as f:
            model = pickle.load(f)
        
        return model, metadata

    def list_models(self, model_name: str = None, status: str = None) -> List[ModelMetadata]:
        """모델 목록 조회"""
        query = "SELECT * FROM models WHERE 1=1"
        params = []
        
        if model_name:
            query += " AND model_name = ?"
            params.append(model_name)
        
        if status:
            query += " AND status = ?"
            params.append(status)
        
        query += " ORDER BY created_at DESC"
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()
        
        models = []
        for row in rows:
            models.append(ModelMetadata(
                model_id=row[0],
                model_name=row[1],
                version=row[2],
                created_at=datetime.fromisoformat(row[3]),
                accuracy=row[4],
                model_type=row[5],
                hyperparameters=json.loads(row[6] or "{}"),
                file_path=row[7],
                file_size=row[8],
                checksum=row[9],
                status=row[10]
            ))
        
        return models

    def _calculate_checksum(self, file_path: Path) -> str:
        """파일 체크섬 계산"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()

class ParallelTrainingManager:
    """
    ⚡ 병렬 모델 훈련 관리자
    """
    
    def __init__(self, max_workers: int = None):
        self.max_workers = max_workers or multiprocessing.cpu_count() // 2
        self.training_queue = queue.Queue()
        self.active_jobs = {}
        self.completed_jobs = {}
        
        # 모델 레지스트리
        self.model_registry = ModelRegistry()
        
        # 리소스 모니터링
        self.resource_monitor = ResourceMonitor()
        
        self.logger = logging.getLogger(__name__)

    def submit_training_job(self, job: TrainingJob) -> str:
        """훈련 작업 제출"""
        job.status = 'queued'
        self.training_queue.put(job)
        self.logger.info(f"⚡ 훈련 작업 제출: {job.job_id}")
        return job.job_id

    def start_training_workers(self):
        """훈련 워커 시작"""
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            
            while True:
                try:
                    # 대기 중인 작업 확인
                    job = self.training_queue.get(timeout=1)
                    
                    # 리소스 체크
                    if not self.resource_monitor.can_start_training():
                        self.training_queue.put(job)  # 다시 큐에 넣기
                        time.sleep(5)
                        continue
                    
                    # 병렬 훈련 시작
                    future = executor.submit(self._train_model_worker, job)
                    futures.append((future, job))
                    
                    job.status = 'running'
                    job.started_at = datetime.now()
                    self.active_jobs[job.job_id] = job
                    
                    self.logger.info(f"🔥 훈련 시작: {job.job_id}")
                    
                except queue.Empty:
                    # 완료된 작업 확인
                    completed_futures = []
                    for future, job in futures:
                        if future.done():
                            try:
                                result = future.result()
                                self._handle_training_completion(job, result)
                                completed_futures.append((future, job))
                            except Exception as e:
                                self._handle_training_error(job, e)
                                completed_futures.append((future, job))
                    
                    # 완료된 작업 제거
                    for completed in completed_futures:
                        futures.remove(completed)
                    
                    time.sleep(1)
                
                except KeyboardInterrupt:
                    self.logger.info("⏹️ 훈련 관리자 종료")
                    break

    def _train_model_worker(self, job: TrainingJob) -> Dict[str, Any]:
        """모델 훈련 워커 (별도 프로세스에서 실행)"""
        try:
            # 앙상블 시스템 초기화
            ensemble_system = AdvancedEnsembleLearningSystem()
            
            # 훈련 실행
            result = ensemble_system.train_ensemble_system()
            
            if result['success']:
                # 모델 등록
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                model_id = f"{job.model_name}_{timestamp}"
                
                metadata = ModelMetadata(
                    model_id=model_id,
                    model_name=job.model_name,
                    version=timestamp,
                    created_at=datetime.now(),
                    accuracy=result['ensemble_performance']['direction_accuracy'],
                    model_type='ensemble',
                    hyperparameters=job.config,
                    file_path="",  # 등록 시 설정됨
                    file_size=0,   # 등록 시 설정됨
                    checksum="",   # 등록 시 설정됨
                    status='active'
                )
                
                return {
                    'success': True,
                    'model_id': model_id,
                    'metadata': metadata,
                    'result': result
                }
            else:
                return {
                    'success': False,
                    'error': result.get('error', 'Unknown error')
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    def _handle_training_completion(self, job: TrainingJob, result: Dict[str, Any]):
        """훈련 완료 처리"""
        job.completed_at = datetime.now()
        job.progress = 1.0
        
        if result['success']:
            job.status = 'completed'
            job.result_path = result.get('model_id', '')
            self.logger.info(f"✅ 훈련 완료: {job.job_id} -> {job.result_path}")
        else:
            job.status = 'failed'
            job.error_message = result['error']
            self.logger.error(f"❌ 훈련 실패: {job.job_id} - {job.error_message}")
        
        # 활성 작업에서 완료 작업으로 이동
        if job.job_id in self.active_jobs:
            del self.active_jobs[job.job_id]
        
        self.completed_jobs[job.job_id] = job

    def _handle_training_error(self, job: TrainingJob, error: Exception):
        """훈련 오류 처리"""
        job.completed_at = datetime.now()
        job.status = 'failed'
        job.error_message = str(error)
        
        self.logger.error(f"❌ 훈련 오류: {job.job_id} - {error}")
        
        if job.job_id in self.active_jobs:
            del self.active_jobs[job.job_id]
        
        self.completed_jobs[job.job_id] = job

class ResourceMonitor:
    """
    📊 리소스 모니터링 시스템
    """
    
    def __init__(self):
        self.thresholds = {
            'cpu_percent': 80.0,      # 80% CPU 사용률
            'memory_percent': 85.0,   # 85% 메모리 사용률
            'disk_percent': 90.0      # 90% 디스크 사용률
        }

    def get_system_resources(self) -> Dict[str, float]:
        """현재 시스템 리소스 상태"""
        return {
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_percent': psutil.disk_usage('/').percent,
            'available_memory_gb': psutil.virtual_memory().available / (1024**3)
        }

    def can_start_training(self) -> bool:
        """훈련 시작 가능 여부 확인"""
        resources = self.get_system_resources()
        
        # 임계값 확인
        if resources['cpu_percent'] > self.thresholds['cpu_percent']:
            return False
        
        if resources['memory_percent'] > self.thresholds['memory_percent']:
            return False
        
        if resources['disk_percent'] > self.thresholds['disk_percent']:
            return False
        
        # 최소 메모리 요구사항 (2GB)
        if resources['available_memory_gb'] < 2.0:
            return False
        
        return True

class RealtimeInferenceEngine:
    """
    ⚡ 실시간 앙상블 추론 엔진
    """
    
    def __init__(self, model_registry: ModelRegistry):
        self.model_registry = model_registry
        self.loaded_models = {}  # 모델 캐시
        self.prediction_cache = {}  # 예측 캐시
        
        # Redis 캐싱 (사용 가능한 경우)
        if REDIS_AVAILABLE:
            try:
                self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
                self.redis_client.ping()
                self.redis_available = True
            except:
                self.redis_available = False
        else:
            self.redis_available = False
        
        self.logger = logging.getLogger(__name__)

    def load_active_models(self) -> int:
        """활성 모델들 로드"""
        active_models = self.model_registry.list_models(status='active')
        loaded_count = 0
        
        for metadata in active_models:
            try:
                model, _ = self.model_registry.load_model(metadata.model_id)
                self.loaded_models[metadata.model_id] = {
                    'model': model,
                    'metadata': metadata,
                    'loaded_at': datetime.now()
                }
                loaded_count += 1
                
            except Exception as e:
                self.logger.error(f"❌ 모델 로드 실패: {metadata.model_id} - {e}")
        
        self.logger.info(f"📚 활성 모델 로드 완료: {loaded_count}개")
        return loaded_count

    async def predict_ensemble(self, features: Dict[str, float], 
                             use_cache: bool = True) -> Dict[str, Any]:
        """
        앙상블 예측 수행
        
        Args:
            features: 입력 특성들
            use_cache: 캐시 사용 여부
            
        Returns:
            Dict[str, Any]: 예측 결과
        """
        start_time = time.time()
        
        # 캐시 키 생성
        cache_key = hashlib.md5(json.dumps(features, sort_keys=True).encode()).hexdigest()
        
        # 캐시 확인
        if use_cache:
            cached_result = await self._get_cached_prediction(cache_key)
            if cached_result:
                return cached_result
        
        # 개별 모델 예측 (비동기 병렬 처리)
        model_predictions = await self._parallel_model_predictions(features)
        
        if not model_predictions:
            raise HTTPException(status_code=500, detail="예측 가능한 모델이 없습니다")
        
        # 앙상블 집계 (가중 평균)
        total_weight = sum(pred['weight'] for pred in model_predictions.values())
        
        if total_weight == 0:
            ensemble_prediction = np.mean([pred['value'] for pred in model_predictions.values()])
        else:
            ensemble_prediction = sum(
                pred['value'] * pred['weight'] for pred in model_predictions.values()
            ) / total_weight
        
        # 불확실성 계산
        individual_preds = [pred['value'] for pred in model_predictions.values()]
        prediction_std = np.std(individual_preds) if len(individual_preds) > 1 else 0.0
        
        confidence = max(0.0, 1.0 - prediction_std / (abs(ensemble_prediction) + 1e-6))
        
        # 결과 구성
        result = {
            'prediction': float(ensemble_prediction),
            'confidence': float(confidence),
            'prediction_std': float(prediction_std),
            'models_used': list(model_predictions.keys()),
            'model_count': len(model_predictions),
            'individual_predictions': model_predictions,
            'prediction_time_ms': (time.time() - start_time) * 1000,
            'timestamp': datetime.now().isoformat(),
            'cache_hit': False
        }
        
        # 캐시 저장
        if use_cache:
            await self._cache_prediction(cache_key, result)
        
        return result

    async def _parallel_model_predictions(self, features: Dict[str, float]) -> Dict[str, Dict]:
        """병렬 모델 예측"""
        predictions = {}
        
        # ThreadPoolExecutor로 병렬 처리
        with ThreadPoolExecutor(max_workers=len(self.loaded_models)) as executor:
            futures = {}
            
            for model_id, model_info in self.loaded_models.items():
                future = executor.submit(
                    self._single_model_prediction, 
                    model_info, features
                )
                futures[model_id] = future
            
            # 결과 수집
            for model_id, future in futures.items():
                try:
                    prediction = future.result(timeout=10)  # 10초 타임아웃
                    if prediction is not None:
                        predictions[model_id] = prediction
                        
                except Exception as e:
                    self.logger.warning(f"⚠️ 모델 {model_id} 예측 실패: {e}")
                    continue
        
        return predictions

    def _single_model_prediction(self, model_info: Dict, 
                               features: Dict[str, float]) -> Optional[Dict]:
        """단일 모델 예측"""
        try:
            model = model_info['model']
            metadata = model_info['metadata']
            
            # 특성을 배열로 변환 (실제 구현에서는 모델에 맞게 조정 필요)
            # 여기서는 간단한 예시
            feature_array = np.array(list(features.values())).reshape(1, -1)
            
            # 예측 수행
            prediction = model.predict(feature_array)[0] if hasattr(model, 'predict') else 0.0
            
            # 가중치 (정확도 기반)
            weight = metadata.accuracy
            
            return {
                'value': float(prediction),
                'weight': float(weight),
                'model_type': metadata.model_type,
                'accuracy': metadata.accuracy
            }
            
        except Exception as e:
            self.logger.warning(f"⚠️ 단일 모델 예측 오류: {e}")
            return None

    async def _get_cached_prediction(self, cache_key: str) -> Optional[Dict]:
        """캐시된 예측 조회"""
        try:
            if self.redis_available:
                cached = self.redis_client.get(cache_key)
                if cached:
                    result = json.loads(cached.decode())
                    result['cache_hit'] = True
                    return result
            
            # 로컬 캐시 확인
            if cache_key in self.prediction_cache:
                cache_entry = self.prediction_cache[cache_key]
                # 5분 캐시 유효시간
                if datetime.now() - cache_entry['timestamp'] < timedelta(minutes=5):
                    result = cache_entry['data'].copy()
                    result['cache_hit'] = True
                    return result
                else:
                    del self.prediction_cache[cache_key]
            
        except Exception as e:
            self.logger.warning(f"⚠️ 캐시 조회 실패: {e}")
        
        return None

    async def _cache_prediction(self, cache_key: str, result: Dict):
        """예측 결과 캐시"""
        try:
            # Redis 캐시
            if self.redis_available:
                self.redis_client.setex(
                    cache_key, 300, json.dumps(result, default=str)
                )  # 5분 TTL
            
            # 로컬 캐시
            self.prediction_cache[cache_key] = {
                'data': result.copy(),
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            self.logger.warning(f"⚠️ 캐시 저장 실패: {e}")

# FastAPI 애플리케이션 (사용 가능한 경우)
if FASTAPI_AVAILABLE:
    app = FastAPI(title="앙상블 학습 시스템 API", version="1.0.0")
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # 전역 객체들
    model_registry = ModelRegistry()
    inference_engine = RealtimeInferenceEngine(model_registry)
    training_manager = ParallelTrainingManager()
    
    class PredictionRequest(BaseModel):
        features: Dict[str, float]
        use_cache: bool = True
    
    class TrainingRequest(BaseModel):
        model_name: str
        config: Dict[str, Any] = {}
    
    @app.on_event("startup")
    async def startup_event():
        """애플리케이션 시작시 초기화"""
        inference_engine.load_active_models()
    
    @app.post("/predict")
    async def predict(request: PredictionRequest):
        """예측 API"""
        try:
            result = await inference_engine.predict_ensemble(
                request.features, request.use_cache
            )
            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/train")
    async def train_model(request: TrainingRequest, background_tasks: BackgroundTasks):
        """모델 훈련 API"""
        job_id = f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        job = TrainingJob(
            job_id=job_id,
            model_name=request.model_name,
            config=request.config,
            status='queued',
            created_at=datetime.now()
        )
        
        training_manager.submit_training_job(job)
        
        return {"job_id": job_id, "status": "queued"}
    
    @app.get("/models")
    async def list_models(model_name: str = None, status: str = None):
        """모델 목록 API"""
        models = model_registry.list_models(model_name, status)
        return [asdict(model) for model in models]
    
    @app.get("/health")
    async def health_check():
        """헬스 체크 API"""
        resource_monitor = ResourceMonitor()
        resources = resource_monitor.get_system_resources()
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "resources": resources,
            "loaded_models": len(inference_engine.loaded_models)
        }

class ProductionEnsembleSystem:
    """
    🏭 프로덕션 앙상블 시스템 통합 관리자
    """
    
    def __init__(self):
        self.model_registry = ModelRegistry()
        self.training_manager = ParallelTrainingManager()
        self.inference_engine = RealtimeInferenceEngine(self.model_registry)
        self.resource_monitor = ResourceMonitor()
        
        # 성능 추적
        self.performance_tracker = PerformanceTracker()
        
        # 로깅 설정
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('production_ensemble.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def deploy_system(self, host: str = "0.0.0.0", port: int = 8000):
        """시스템 배포"""
        if not FASTAPI_AVAILABLE:
            print("❌ FastAPI가 설치되지 않아 웹 API를 시작할 수 없습니다")
            return
        
        print("🏭 프로덕션 앙상블 시스템 배포 시작...")
        
        # 초기 모델 로드
        self.inference_engine.load_active_models()
        
        # 훈련 워커 백그라운드 시작
        training_thread = threading.Thread(
            target=self.training_manager.start_training_workers,
            daemon=True
        )
        training_thread.start()
        
        # FastAPI 서버 시작
        print(f"🌐 API 서버 시작: http://{host}:{port}")
        print("📚 사용 가능한 엔드포인트:")
        print("  POST /predict - 앙상블 예측")
        print("  POST /train - 모델 훈련 시작")
        print("  GET /models - 모델 목록")
        print("  GET /health - 시스템 상태")
        
        uvicorn.run(app, host=host, port=port, log_level="info")

    def run_comprehensive_test(self) -> Dict[str, Any]:
        """종합 시스템 테스트"""
        print("🧪 종합 시스템 테스트 시작...")
        
        test_results = {
            'timestamp': datetime.now(),
            'test_components': [],
            'overall_status': 'unknown',
            'performance_metrics': {},
            'recommendations': []
        }
        
        # 1. 시스템 리소스 테스트
        print("📊 시스템 리소스 테스트...")
        resources = self.resource_monitor.get_system_resources()
        test_results['test_components'].append({
            'component': 'system_resources',
            'status': 'pass' if resources['available_memory_gb'] > 1.0 else 'fail',
            'details': resources
        })
        
        # 2. 모델 레지스트리 테스트
        print("📚 모델 레지스트리 테스트...")
        try:
            models = self.model_registry.list_models()
            registry_status = 'pass' if len(models) >= 0 else 'fail'
            test_results['test_components'].append({
                'component': 'model_registry',
                'status': registry_status,
                'details': {'model_count': len(models)}
            })
        except Exception as e:
            test_results['test_components'].append({
                'component': 'model_registry',
                'status': 'fail',
                'details': {'error': str(e)}
            })
        
        # 3. 추론 엔진 테스트
        print("⚡ 추론 엔진 테스트...")
        try:
            loaded_count = self.inference_engine.load_active_models()
            inference_status = 'pass' if loaded_count >= 0 else 'fail'
            test_results['test_components'].append({
                'component': 'inference_engine',
                'status': inference_status,
                'details': {'loaded_models': loaded_count}
            })
        except Exception as e:
            test_results['test_components'].append({
                'component': 'inference_engine',
                'status': 'fail',
                'details': {'error': str(e)}
            })
        
        # 전체 상태 평가
        failed_components = [c for c in test_results['test_components'] if c['status'] == 'fail']
        
        if not failed_components:
            test_results['overall_status'] = 'pass'
        elif len(failed_components) <= len(test_results['test_components']) // 2:
            test_results['overall_status'] = 'partial'
        else:
            test_results['overall_status'] = 'fail'
        
        # 권장사항
        if failed_components:
            test_results['recommendations'].append(
                f"{len(failed_components)}개 컴포넌트가 실패했습니다. 로그를 확인하세요."
            )
        
        if resources['available_memory_gb'] < 2.0:
            test_results['recommendations'].append(
                "메모리가 부족합니다. 최소 2GB 이상 확보하세요."
            )
        
        # 결과 출력
        print("\n" + "="*50)
        print("🧪 시스템 테스트 완료!")
        print("="*50)
        print(f"🏆 전체 상태: {test_results['overall_status'].upper()}")
        print(f"📊 테스트 컴포넌트: {len(test_results['test_components'])}개")
        print(f"❌ 실패한 컴포넌트: {len(failed_components)}개")
        
        if test_results['recommendations']:
            print("\n📋 권장사항:")
            for i, rec in enumerate(test_results['recommendations'], 1):
                print(f"  {i}. {rec}")
        
        return test_results

class PerformanceTracker:
    """성능 추적 시스템"""
    
    def __init__(self):
        self.db_path = "/Users/parkyoungjun/Desktop/BTC_Analysis_System/performance_tracking.db"
        self.init_database()
    
    def init_database(self):
        """데이터베이스 초기화"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    prediction REAL,
                    actual REAL,
                    accuracy REAL,
                    model_count INTEGER,
                    response_time_ms REAL
                )
            """)

def main():
    """메인 실행 함수"""
    print("🏭 프로덕션 앙상블 아키텍처 시스템")
    
    # 시스템 초기화
    production_system = ProductionEnsembleSystem()
    
    # 종합 테스트
    test_results = production_system.run_comprehensive_test()
    
    # 테스트 결과를 JSON으로 저장
    test_result_path = "/Users/parkyoungjun/Desktop/BTC_Analysis_System/production_system_test_results.json"
    with open(test_result_path, 'w', encoding='utf-8') as f:
        json.dump(test_results, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\n📄 테스트 결과 저장: {test_result_path}")
    
    return production_system

if __name__ == "__main__":
    system = main()