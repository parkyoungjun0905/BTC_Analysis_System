#!/usr/bin/env python3
"""
🎯 하이퍼파라미터 최적화 시스템
- 베이지안 최적화 (Gaussian Process, Tree-structured Parzen Estimator)
- 다목적 최적화 (Multi-objective Optimization)
- 분산 최적화 (Distributed Optimization)
- 조기 중단 (Early Stopping)
- 하이퍼파라미터 중요도 분석
- 최적화 과정 시각화
- 자동 하이퍼파라미터 튜닝 파이프라인
"""

import numpy as np
import pandas as pd
import warnings
from datetime import datetime, timedelta
import json
import os
from typing import Dict, List, Tuple, Optional, Union, Callable, Any
from dataclasses import dataclass, field
import logging
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp
import time

# ML 라이브러리
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb

# 베이지안 최적화
try:
    from skopt import gp_minimize, forest_minimize, gbrt_minimize
    from skopt.space import Real, Integer, Categorical
    from skopt.utils import use_named_args
    from skopt.acquisition import gaussian_ei, gaussian_pi, gaussian_lcb
    from skopt.callbacks import EarlyStopper
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False
    print("⚠️ scikit-optimize 미설치: 베이지안 최적화 불가")

# Optuna (대안 베이지안 최적화)
try:
    import optuna
    from optuna.samplers import TPESampler, CmaEsSampler
    from optuna.pruners import MedianPruner, HyperbandPruner
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("⚠️ Optuna 미설치: TPE 최적화 불가")

# 시각화
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

warnings.filterwarnings('ignore')

@dataclass
class OptimizationConfig:
    """최적화 설정"""
    # 최적화 방법
    optimization_method: str = 'bayesian_gp'  # 'grid', 'random', 'bayesian_gp', 'bayesian_rf', 'tpe'
    
    # 최적화 예산
    n_calls: int = 100              # 최적화 호출 수
    n_initial_points: int = 10      # 초기 랜덤 포인트 수
    
    # 교차 검증
    cv_folds: int = 5              # 교차 검증 폴드
    cv_scoring: str = 'neg_mean_squared_error'  # 교차 검증 스코어링
    
    # 조기 중단
    early_stopping: bool = True     # 조기 중단 사용 여부
    patience: int = 20             # 조기 중단 patience
    min_improvement: float = 0.001  # 최소 개선값
    
    # 다목적 최적화
    multi_objective: bool = False   # 다목적 최적화 여부
    objectives: List[str] = field(default_factory=lambda: ['accuracy', 'speed'])
    
    # 분산 최적화
    n_jobs: int = -1               # 병렬 처리 수
    distributed: bool = False       # 분산 처리 여부
    
    # 정규화
    normalize_objectives: bool = True  # 목적함수 정규화
    
    # 시각화
    plot_convergence: bool = True   # 수렴 플롯 생성
    plot_evaluations: bool = True   # 평가 플롯 생성
    
    # 하이퍼파라미터 중요도
    feature_importance: bool = True # 피처 중요도 분석
    
    # 저장 설정
    save_intermediate: bool = True  # 중간 결과 저장

class BaseOptimizer(ABC):
    """기본 최적화기 추상 클래스"""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.optimization_history = []
        self.best_params = None
        self.best_score = float('-inf')
        
    @abstractmethod
    def optimize(self, objective_func: Callable, search_space: Dict) -> Dict:
        """최적화 실행"""
        pass
    
    def _evaluate_objective(self, params: Dict, objective_func: Callable) -> float:
        """목적함수 평가"""
        try:
            score = objective_func(params)
            self.optimization_history.append({
                'params': params.copy(),
                'score': score,
                'timestamp': datetime.now()
            })
            
            if score > self.best_score:
                self.best_score = score
                self.best_params = params.copy()
            
            return score
        except Exception as e:
            logging.warning(f"목적함수 평가 실패: {e}")
            return float('-inf')

class BayesianOptimizer(BaseOptimizer):
    """베이지안 최적화기"""
    
    def __init__(self, config: OptimizationConfig):
        super().__init__(config)
        self.gp_model = None
        
    def optimize(self, objective_func: Callable, search_space: Dict) -> Dict:
        """베이지안 최적화 실행"""
        if not SKOPT_AVAILABLE:
            raise ImportError("scikit-optimize가 필요합니다")
        
        # 검색 공간 변환
        dimensions = self._convert_search_space(search_space)
        
        # 목적함수 래퍼
        @use_named_args(dimensions)
        def objective(**params):
            return -self._evaluate_objective(params, objective_func)  # 최소화를 위해 음수
        
        # 베이지안 최적화 방법 선택
        if self.config.optimization_method == 'bayesian_gp':
            result = gp_minimize(
                objective,
                dimensions,
                n_calls=self.config.n_calls,
                n_initial_points=self.config.n_initial_points,
                acquisition_func='EI',
                random_state=42
            )
        elif self.config.optimization_method == 'bayesian_rf':
            result = forest_minimize(
                objective,
                dimensions,
                n_calls=self.config.n_calls,
                n_initial_points=self.config.n_initial_points,
                random_state=42
            )
        else:
            result = gbrt_minimize(
                objective,
                dimensions,
                n_calls=self.config.n_calls,
                n_initial_points=self.config.n_initial_points,
                random_state=42
            )
        
        # 결과 정리
        best_params = dict(zip([d.name for d in dimensions], result.x))
        
        return {
            'best_params': best_params,
            'best_score': -result.fun,  # 다시 양수로
            'optimization_result': result,
            'n_evaluations': len(result.func_vals),
            'convergence_trace': [-val for val in result.func_vals]
        }
    
    def _convert_search_space(self, search_space: Dict) -> List:
        """검색 공간을 scikit-optimize 형식으로 변환"""
        dimensions = []
        
        for param_name, param_config in search_space.items():
            if isinstance(param_config, dict):
                param_type = param_config.get('type', 'real')
                
                if param_type == 'real':
                    dimensions.append(Real(
                        param_config['low'],
                        param_config['high'],
                        name=param_name,
                        prior=param_config.get('prior', 'uniform')
                    ))
                elif param_type == 'integer':
                    dimensions.append(Integer(
                        param_config['low'],
                        param_config['high'],
                        name=param_name
                    ))
                elif param_type == 'categorical':
                    dimensions.append(Categorical(
                        param_config['choices'],
                        name=param_name
                    ))
            else:
                # 간단한 형식 지원
                if isinstance(param_config, tuple) and len(param_config) == 2:
                    dimensions.append(Real(param_config[0], param_config[1], name=param_name))
                elif isinstance(param_config, list):
                    dimensions.append(Categorical(param_config, name=param_name))
        
        return dimensions

class OptunaOptimizer(BaseOptimizer):
    """Optuna TPE 최적화기"""
    
    def __init__(self, config: OptimizationConfig):
        super().__init__(config)
        self.study = None
        
    def optimize(self, objective_func: Callable, search_space: Dict) -> Dict:
        """Optuna 최적화 실행"""
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna가 필요합니다")
        
        # Optuna 스터디 생성
        sampler = TPESampler(seed=42)
        if self.config.early_stopping:
            pruner = MedianPruner(n_startup_trials=self.config.patience)
        else:
            pruner = optuna.pruners.NopPruner()
        
        self.study = optuna.create_study(
            direction='maximize',
            sampler=sampler,
            pruner=pruner
        )
        
        # 목적함수 래퍼
        def optuna_objective(trial):
            params = self._suggest_params(trial, search_space)
            return self._evaluate_objective(params, objective_func)
        
        # 최적화 실행
        self.study.optimize(
            optuna_objective,
            n_trials=self.config.n_calls,
            n_jobs=1 if self.config.n_jobs == -1 else self.config.n_jobs
        )
        
        return {
            'best_params': self.study.best_params,
            'best_score': self.study.best_value,
            'n_evaluations': len(self.study.trials),
            'study': self.study
        }
    
    def _suggest_params(self, trial, search_space: Dict) -> Dict:
        """파라미터 제안"""
        params = {}
        
        for param_name, param_config in search_space.items():
            if isinstance(param_config, dict):
                param_type = param_config.get('type', 'real')
                
                if param_type == 'real':
                    params[param_name] = trial.suggest_float(
                        param_name,
                        param_config['low'],
                        param_config['high']
                    )
                elif param_type == 'integer':
                    params[param_name] = trial.suggest_int(
                        param_name,
                        param_config['low'],
                        param_config['high']
                    )
                elif param_type == 'categorical':
                    params[param_name] = trial.suggest_categorical(
                        param_name,
                        param_config['choices']
                    )
            else:
                # 간단한 형식
                if isinstance(param_config, tuple) and len(param_config) == 2:
                    params[param_name] = trial.suggest_float(param_name, param_config[0], param_config[1])
                elif isinstance(param_config, list):
                    params[param_name] = trial.suggest_categorical(param_name, param_config)
        
        return params

class MultiObjectiveOptimizer(BaseOptimizer):
    """다목적 최적화기"""
    
    def __init__(self, config: OptimizationConfig):
        super().__init__(config)
        self.pareto_front = []
        
    def optimize(self, objective_funcs: List[Callable], search_space: Dict) -> Dict:
        """다목적 최적화 실행"""
        if not OPTUNA_AVAILABLE:
            raise ImportError("다목적 최적화는 Optuna가 필요합니다")
        
        # 다목적 스터디 생성
        study = optuna.create_study(
            directions=['maximize'] * len(objective_funcs),
            sampler=TPESampler(seed=42)
        )
        
        def multi_objective(trial):
            params = self._suggest_params(trial, search_space)
            
            objectives = []
            for obj_func in objective_funcs:
                try:
                    score = obj_func(params)
                    objectives.append(score)
                except Exception:
                    objectives.append(float('-inf'))
            
            return objectives
        
        # 최적화 실행
        study.optimize(multi_objective, n_trials=self.config.n_calls)
        
        # 파레토 프론트 추출
        pareto_trials = []
        for trial in study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                pareto_trials.append({
                    'params': trial.params,
                    'values': trial.values
                })
        
        return {
            'pareto_front': pareto_trials,
            'study': study,
            'n_evaluations': len(study.trials)
        }

class HyperparameterOptimizer:
    """하이퍼파라미터 최적화 메인 시스템"""
    
    def __init__(self, config: OptimizationConfig = None):
        self.config = config or OptimizationConfig()
        self.data_path = "/Users/parkyoungjun/Desktop/BTC_Analysis_System"
        
        # 최적화기 초기화
        self.optimizers = self._initialize_optimizers()
        
        # 결과 저장
        self.optimization_results = {}
        self.parameter_importance = {}
        
        # 로깅 설정
        self.setup_logging()
        
    def setup_logging(self):
        """로깅 설정"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.data_path, 'hyperparameter_optimization.log'), encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def _initialize_optimizers(self) -> Dict:
        """최적화기들 초기화"""
        optimizers = {}
        
        # 베이지안 최적화
        if SKOPT_AVAILABLE:
            optimizers['bayesian'] = BayesianOptimizer(self.config)
        
        # Optuna TPE
        if OPTUNA_AVAILABLE:
            optimizers['tpe'] = OptunaOptimizer(self.config)
            optimizers['multi_objective'] = MultiObjectiveOptimizer(self.config)
        
        return optimizers
    
    def optimize_model(self, model_class, X: pd.DataFrame, y: pd.Series, 
                      search_space: Dict) -> Dict:
        """모델 하이퍼파라미터 최적화"""
        self.logger.info(f"🎯 {model_class.__name__} 하이퍼파라미터 최적화 시작...")
        
        # 목적함수 정의
        def objective_function(params):
            return self._evaluate_model(model_class, params, X, y)
        
        # 최적화 방법 선택
        if self.config.optimization_method in self.optimizers:
            optimizer = self.optimizers[self.config.optimization_method]
        else:
            # 기본값: 사용 가능한 첫 번째 최적화기
            optimizer = list(self.optimizers.values())[0]
        
        # 최적화 실행
        start_time = time.time()
        optimization_result = optimizer.optimize(objective_function, search_space)
        optimization_time = time.time() - start_time
        
        # 결과 정리
        result = {
            'model_class': model_class.__name__,
            'optimization_method': self.config.optimization_method,
            'optimization_time_seconds': optimization_time,
            'best_params': optimization_result['best_params'],
            'best_score': optimization_result['best_score'],
            'n_evaluations': optimization_result.get('n_evaluations', len(optimizer.optimization_history)),
            'optimization_history': optimizer.optimization_history,
            'search_space': search_space,
            'data_shape': X.shape,
            'timestamp': datetime.now().isoformat()
        }
        
        # 파라미터 중요도 분석
        if self.config.feature_importance:
            importance = self._analyze_parameter_importance(optimizer.optimization_history, search_space)
            result['parameter_importance'] = importance
        
        # 결과 저장
        model_key = f"{model_class.__name__}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.optimization_results[model_key] = result
        
        self.logger.info(f"✅ 최적화 완료: {optimization_time:.2f}초, 최고 점수: {optimization_result['best_score']:.4f}")
        
        return result
    
    def _evaluate_model(self, model_class, params: Dict, X: pd.DataFrame, y: pd.Series) -> float:
        """모델 평가"""
        try:
            # 파라미터 전처리
            processed_params = self._preprocess_params(params)
            
            # 모델 생성
            model = model_class(**processed_params)
            
            # 교차 검증
            tscv = TimeSeriesSplit(n_splits=self.config.cv_folds)
            scores = cross_val_score(
                model, X, y, 
                cv=tscv, 
                scoring=self.config.cv_scoring,
                n_jobs=1  # 개별 모델 평가는 단일 스레드
            )
            
            # 평균 점수 반환 (음수 점수는 양수로 변환)
            mean_score = np.mean(scores)
            if self.config.cv_scoring.startswith('neg_'):
                mean_score = -mean_score
            
            return mean_score
            
        except Exception as e:
            self.logger.warning(f"모델 평가 실패: {e}")
            return float('-inf')
    
    def _preprocess_params(self, params: Dict) -> Dict:
        """파라미터 전처리 (타입 변환 등)"""
        processed = {}
        
        for key, value in params.items():
            # 정수형 파라미터 변환
            if key in ['n_estimators', 'max_depth', 'min_samples_split', 
                      'min_samples_leaf', 'max_features_int']:
                processed[key] = int(value)
            # 부동소수점 파라미터
            elif key in ['learning_rate', 'subsample', 'colsample_bytree', 
                        'alpha', 'l1_ratio']:
                processed[key] = float(value)
            # 문자열 파라미터
            else:
                processed[key] = value
        
        return processed
    
    def _analyze_parameter_importance(self, optimization_history: List[Dict], 
                                    search_space: Dict) -> Dict:
        """파라미터 중요도 분석"""
        if len(optimization_history) < 10:
            return {'insufficient_data': True}
        
        # 데이터 준비
        params_df = pd.DataFrame([h['params'] for h in optimization_history])
        scores = np.array([h['score'] for h in optimization_history])
        
        # 상관관계 분석
        importance_scores = {}
        
        for param_name in params_df.columns:
            param_values = params_df[param_name]
            
            # 수치형 파라미터의 경우 상관계수 계산
            if pd.api.types.is_numeric_dtype(param_values):
                correlation = np.corrcoef(param_values, scores)[0, 1]
                importance_scores[param_name] = abs(correlation) if not np.isnan(correlation) else 0
            else:
                # 범주형 파라미터의 경우 분산 분석
                unique_values = param_values.unique()
                if len(unique_values) > 1:
                    group_means = []
                    for value in unique_values:
                        group_scores = scores[param_values == value]
                        if len(group_scores) > 0:
                            group_means.append(np.mean(group_scores))
                    
                    importance_scores[param_name] = np.std(group_means) if len(group_means) > 1 else 0
                else:
                    importance_scores[param_name] = 0
        
        # 정규화
        max_importance = max(importance_scores.values()) if importance_scores else 1
        if max_importance > 0:
            importance_scores = {k: v/max_importance for k, v in importance_scores.items()}
        
        return importance_scores
    
    def optimize_multiple_models(self, model_configs: List[Dict], 
                                X: pd.DataFrame, y: pd.Series) -> Dict:
        """다중 모델 최적화"""
        self.logger.info("🚀 다중 모델 하이퍼파라미터 최적화 시작...")
        
        results = {}
        
        for config in model_configs:
            model_class = config['model_class']
            search_space = config['search_space']
            model_name = config.get('name', model_class.__name__)
            
            try:
                result = self.optimize_model(model_class, X, y, search_space)
                results[model_name] = result
                
                # 중간 결과 저장
                if self.config.save_intermediate:
                    self._save_intermediate_result(model_name, result)
                    
            except Exception as e:
                self.logger.error(f"모델 {model_name} 최적화 실패: {e}")
                results[model_name] = {'error': str(e)}
        
        # 최고 모델 선택
        best_model = self._select_best_model(results)
        
        comprehensive_result = {
            'individual_results': results,
            'best_model': best_model,
            'optimization_summary': self._generate_optimization_summary(results),
            'timestamp': datetime.now().isoformat()
        }
        
        return comprehensive_result
    
    def _select_best_model(self, results: Dict) -> Dict:
        """최고 모델 선택"""
        best_score = float('-inf')
        best_model = None
        best_name = None
        
        for model_name, result in results.items():
            if 'error' not in result and 'best_score' in result:
                if result['best_score'] > best_score:
                    best_score = result['best_score']
                    best_model = result
                    best_name = model_name
        
        if best_model:
            return {
                'model_name': best_name,
                'best_params': best_model['best_params'],
                'best_score': best_model['best_score'],
                'model_class': best_model['model_class']
            }
        else:
            return {'error': '유효한 최적화 결과 없음'}
    
    def _generate_optimization_summary(self, results: Dict) -> Dict:
        """최적화 요약 생성"""
        summary = {
            'total_models': len(results),
            'successful_optimizations': sum(1 for r in results.values() if 'error' not in r),
            'failed_optimizations': sum(1 for r in results.values() if 'error' in r),
            'total_evaluations': sum(r.get('n_evaluations', 0) for r in results.values() if 'error' not in r),
            'total_optimization_time': sum(r.get('optimization_time_seconds', 0) for r in results.values() if 'error' not in r)
        }
        
        # 모델별 성능 비교
        model_scores = {}
        for name, result in results.items():
            if 'error' not in result and 'best_score' in result:
                model_scores[name] = result['best_score']
        
        if model_scores:
            summary['model_performance_ranking'] = sorted(
                model_scores.items(), 
                key=lambda x: x[1], 
                reverse=True
            )
        
        return summary
    
    def create_optimization_dashboard(self, results: Dict) -> str:
        """최적화 대시보드 생성"""
        self.logger.info("📊 최적화 대시보드 생성 중...")
        
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=(
                "모델별 성능 비교", "최적화 수렴 곡선", "파라미터 중요도",
                "최적화 시간 비교", "평가 횟수 비교", "파라미터 분포",
                "교차 검증 점수", "최적 파라미터", "최적화 효율성"
            ),
            specs=[[{"type": "bar"}, {"type": "scatter"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "bar"}, {"type": "histogram"}],
                   [{"type": "box"}, {"type": "table"}, {"type": "scatter"}]]
        )
        
        # 데이터 추출
        if 'individual_results' in results:
            individual_results = results['individual_results']
            
            # 1. 모델별 성능 비교
            model_names = []
            model_scores = []
            
            for name, result in individual_results.items():
                if 'error' not in result and 'best_score' in result:
                    model_names.append(name)
                    model_scores.append(result['best_score'])
            
            if model_names:
                fig.add_trace(
                    go.Bar(x=model_names, y=model_scores, name='Best Score'),
                    row=1, col=1
                )
            
            # 2. 최적화 수렴 곡선 (첫 번째 모델)
            first_model = next((r for r in individual_results.values() if 'error' not in r), None)
            if first_model and 'optimization_history' in first_model:
                history = first_model['optimization_history']
                iterations = list(range(len(history)))
                scores = [h['score'] for h in history]
                
                # 누적 최대값 (수렴 곡선)
                cummax_scores = np.maximum.accumulate(scores)
                
                fig.add_trace(
                    go.Scatter(x=iterations, y=cummax_scores, mode='lines', name='Convergence'),
                    row=1, col=2
                )
            
            # 3. 파라미터 중요도 (첫 번째 모델)
            if first_model and 'parameter_importance' in first_model:
                importance = first_model['parameter_importance']
                if 'insufficient_data' not in importance:
                    params = list(importance.keys())
                    importances = list(importance.values())
                    
                    fig.add_trace(
                        go.Bar(x=params, y=importances, name='Parameter Importance'),
                        row=1, col=3
                    )
            
            # 4. 최적화 시간 비교
            opt_times = []
            for name, result in individual_results.items():
                if 'error' not in result and 'optimization_time_seconds' in result:
                    opt_times.append(result['optimization_time_seconds'])
            
            if opt_times and model_names:
                fig.add_trace(
                    go.Bar(x=model_names[:len(opt_times)], y=opt_times, name='Optimization Time'),
                    row=2, col=1
                )
            
            # 5. 평가 횟수 비교
            n_evals = []
            for name, result in individual_results.items():
                if 'error' not in result and 'n_evaluations' in result:
                    n_evals.append(result['n_evaluations'])
            
            if n_evals and model_names:
                fig.add_trace(
                    go.Bar(x=model_names[:len(n_evals)], y=n_evals, name='N Evaluations'),
                    row=2, col=2
                )
        
        fig.update_layout(
            title="🎯 하이퍼파라미터 최적화 대시보드",
            height=1200,
            showlegend=True,
            template='plotly_dark'
        )
        
        # 저장
        dashboard_path = os.path.join(self.data_path, 'hyperparameter_optimization_dashboard.html')
        fig.write_html(dashboard_path, include_plotlyjs=True)
        
        return dashboard_path
    
    def _save_intermediate_result(self, model_name: str, result: Dict):
        """중간 결과 저장"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'hyperopt_{model_name}_{timestamp}.json'
        filepath = os.path.join(self.data_path, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False, default=str)
    
    def save_optimization_results(self, results: Dict):
        """최적화 결과 저장"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 전체 결과 저장
        full_results_path = os.path.join(self.data_path, f'hyperparameter_optimization_results_{timestamp}.json')
        with open(full_results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        # 요약 보고서 저장
        summary_path = os.path.join(self.data_path, f'hyperparameter_optimization_summary_{timestamp}.txt')
        self._generate_summary_report(results, summary_path)
        
        self.logger.info(f"최적화 결과 저장: {full_results_path}")
        self.logger.info(f"요약 보고서 저장: {summary_path}")
    
    def _generate_summary_report(self, results: Dict, output_path: str):
        """요약 보고서 생성"""
        lines = [
            "🎯 하이퍼파라미터 최적화 결과 요약",
            "=" * 60,
            f"생성 일시: {datetime.now().strftime('%Y년 %m월 %d일 %H시 %M분')}",
            ""
        ]
        
        # 전체 요약
        if 'optimization_summary' in results:
            summary = results['optimization_summary']
            lines.extend([
                "📊 전체 요약",
                f"  • 최적화한 모델 수: {summary.get('total_models', 0)}개",
                f"  • 성공한 최적화: {summary.get('successful_optimizations', 0)}개",
                f"  • 실패한 최적화: {summary.get('failed_optimizations', 0)}개",
                f"  • 총 평가 횟수: {summary.get('total_evaluations', 0):,}회",
                f"  • 총 최적화 시간: {summary.get('total_optimization_time', 0):.1f}초",
                ""
            ])
            
            # 모델 순위
            if 'model_performance_ranking' in summary:
                lines.extend([
                    "🏆 모델 성능 순위",
                ])
                for i, (model_name, score) in enumerate(summary['model_performance_ranking'], 1):
                    lines.append(f"  {i}. {model_name}: {score:.4f}")
                lines.append("")
        
        # 최고 모델 정보
        if 'best_model' in results and 'error' not in results['best_model']:
            best = results['best_model']
            lines.extend([
                "🥇 최고 성능 모델",
                f"  • 모델: {best.get('model_name', 'Unknown')}",
                f"  • 점수: {best.get('best_score', 0):.4f}",
                f"  • 최적 파라미터:",
            ])
            
            for param, value in best.get('best_params', {}).items():
                lines.append(f"    - {param}: {value}")
            lines.append("")
        
        # 개별 모델 세부사항
        if 'individual_results' in results:
            lines.append("📋 개별 모델 세부사항")
            for model_name, result in results['individual_results'].items():
                if 'error' not in result:
                    lines.extend([
                        f"  • {model_name}",
                        f"    - 최고 점수: {result.get('best_score', 0):.4f}",
                        f"    - 평가 횟수: {result.get('n_evaluations', 0)}회",
                        f"    - 최적화 시간: {result.get('optimization_time_seconds', 0):.1f}초",
                    ])
                else:
                    lines.append(f"  • {model_name}: 최적화 실패 ({result['error']})")
        
        # 파일 저장
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

def create_default_search_spaces() -> Dict:
    """기본 검색 공간 정의"""
    return {
        'RandomForestRegressor': {
            'n_estimators': {'type': 'integer', 'low': 50, 'high': 500},
            'max_depth': {'type': 'integer', 'low': 3, 'high': 20},
            'min_samples_split': {'type': 'integer', 'low': 2, 'high': 20},
            'min_samples_leaf': {'type': 'integer', 'low': 1, 'high': 10},
            'max_features': {'type': 'categorical', 'choices': ['sqrt', 'log2', None]}
        },
        'XGBRegressor': {
            'n_estimators': {'type': 'integer', 'low': 50, 'high': 500},
            'max_depth': {'type': 'integer', 'low': 3, 'high': 10},
            'learning_rate': {'type': 'real', 'low': 0.01, 'high': 0.3},
            'subsample': {'type': 'real', 'low': 0.6, 'high': 1.0},
            'colsample_bytree': {'type': 'real', 'low': 0.6, 'high': 1.0}
        },
        'LGBMRegressor': {
            'n_estimators': {'type': 'integer', 'low': 50, 'high': 500},
            'max_depth': {'type': 'integer', 'low': 3, 'high': 10},
            'learning_rate': {'type': 'real', 'low': 0.01, 'high': 0.3},
            'feature_fraction': {'type': 'real', 'low': 0.6, 'high': 1.0},
            'bagging_fraction': {'type': 'real', 'low': 0.6, 'high': 1.0}
        }
    }

def main():
    """메인 실행 함수"""
    print("🎯 하이퍼파라미터 최적화 시스템")
    print("=" * 50)
    
    # 설정
    config = OptimizationConfig(
        optimization_method='bayesian_gp' if SKOPT_AVAILABLE else 'random',
        n_calls=50,
        cv_folds=5,
        early_stopping=True
    )
    
    # 최적화기 초기화
    optimizer = HyperparameterOptimizer(config)
    
    # 테스트 데이터 생성
    np.random.seed(42)
    n_samples, n_features = 1000, 20
    
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    y = pd.Series(
        np.sum(X.iloc[:, :5], axis=1) + np.random.randn(n_samples) * 0.1
    )
    
    # 다중 모델 최적화
    search_spaces = create_default_search_spaces()
    model_configs = [
        {
            'name': 'RandomForest',
            'model_class': RandomForestRegressor,
            'search_space': search_spaces['RandomForestRegressor']
        }
    ]
    
    # XGBoost 추가 (사용 가능한 경우)
    try:
        import xgboost
        model_configs.append({
            'name': 'XGBoost',
            'model_class': xgb.XGBRegressor,
            'search_space': search_spaces['XGBRegressor']
        })
    except ImportError:
        pass
    
    # 최적화 실행
    results = optimizer.optimize_multiple_models(model_configs, X, y)
    
    print(f"\n🎯 하이퍼파라미터 최적화 완료!")
    
    if 'best_model' in results and 'error' not in results['best_model']:
        best = results['best_model']
        print(f"최고 모델: {best['model_name']}")
        print(f"최고 점수: {best['best_score']:.4f}")
        print(f"최적 파라미터: {best['best_params']}")
    
    # 대시보드 생성
    dashboard_path = optimizer.create_optimization_dashboard(results)
    print(f"대시보드: {dashboard_path}")
    
    # 결과 저장
    optimizer.save_optimization_results(results)

if __name__ == "__main__":
    main()