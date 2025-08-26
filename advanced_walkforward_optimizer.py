#!/usr/bin/env python3
"""
🎯 고급 워크포워드 최적화 시스템
- 적응형 롤링 윈도우 (시장 변동성에 따라 조정)
- 다중 시간 프레임 최적화
- 아웃오브샘플 성능 추적
- 파라미터 안정성 분석
- 과적합 감지 및 방지
- 동적 리밸런싱 최적화
"""

import numpy as np
import pandas as pd
import warnings
import logging
from datetime import datetime, timedelta
import json
import os
from typing import Dict, List, Tuple, Optional, Callable
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
import pickle

# ML 라이브러리
from sklearn.model_selection import TimeSeriesSplit, ParameterGrid
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.feature_selection import SelectFromModel, RFE
import xgboost as xgb
import lightgbm as lgb

# 베이지안 최적화
try:
    from skopt import gp_minimize, forest_minimize, gbrt_minimize
    from skopt.space import Real, Integer, Categorical
    from skopt.utils import use_named_args
    from skopt.acquisition import gaussian_ei, gaussian_pi
    BAYESIAN_OPT_AVAILABLE = True
except ImportError:
    BAYESIAN_OPT_AVAILABLE = False

# 시각화
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

warnings.filterwarnings('ignore')

@dataclass
class OptimizationConfig:
    """최적화 설정"""
    # 롤링 윈도우 설정
    min_train_size: int = 500      # 최소 학습 데이터 크기
    max_train_size: int = 2000     # 최대 학습 데이터 크기
    test_size: int = 48            # 테스트 데이터 크기 (48시간)
    step_size: int = 24            # 이동 단계 (24시간)
    
    # 적응형 윈도우 설정
    volatility_lookback: int = 168  # 변동성 계산 기간 (1주일)
    volatility_threshold: float = 0.05  # 높은 변동성 임계값
    adaptive_window: bool = True    # 적응형 윈도우 사용
    
    # 최적화 설정
    optimization_method: str = 'bayesian'  # 'grid', 'random', 'bayesian'
    n_optimization_trials: int = 100   # 최적화 시도 횟수
    cv_folds: int = 5              # 교차 검증 fold 수
    
    # 성능 임계값
    min_r2_threshold: float = 0.1   # 최소 R² 임계값
    overfitting_threshold: float = 0.3  # 과적합 감지 임계값
    stability_threshold: float = 0.2    # 파라미터 안정성 임계값
    
    # 병렬 처리
    n_jobs: int = -1               # 병렬 처리 프로세스 수
    
    # 리밸런싱 설정
    rebalance_methods: List[str] = None  # 리밸런싱 방법들
    
    def __post_init__(self):
        if self.rebalance_methods is None:
            self.rebalance_methods = ['daily', 'weekly', 'volatility_based', 'performance_based']

class AdvancedWalkForwardOptimizer:
    """고급 워크포워드 최적화 시스템"""
    
    def __init__(self, config: OptimizationConfig = None):
        self.config = config or OptimizationConfig()
        self.data_path = "/Users/parkyoungjun/Desktop/BTC_Analysis_System"
        
        # 결과 저장
        self.optimization_history = []
        self.parameter_stability_tracker = {}
        self.overfitting_detector = {}
        self.performance_tracker = {}
        
        # 로깅 설정
        self.setup_logging()
        
    def setup_logging(self):
        """로깅 설정"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.data_path, 'walkforward_optimizer.log'), encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def adaptive_window_sizing(self, data: pd.DataFrame, current_idx: int) -> Dict[str, int]:
        """적응형 윈도우 크기 결정"""
        if not self.config.adaptive_window:
            return {
                'train_size': self.config.max_train_size,
                'test_size': self.config.test_size
            }
        
        # 현재 시점 기준 변동성 계산
        lookback_start = max(0, current_idx - self.config.volatility_lookback)
        recent_data = data.iloc[lookback_start:current_idx]
        
        if len(recent_data) < 10:  # 데이터 부족시 기본값
            return {
                'train_size': self.config.min_train_size,
                'test_size': self.config.test_size
            }
        
        # 가격 변동성 계산
        if 'price' in recent_data.columns:
            returns = recent_data['price'].pct_change().dropna()
        else:
            returns = recent_data.iloc[:, 0].pct_change().dropna()
        
        volatility = returns.std()
        
        # 변동성에 따른 윈도우 크기 조정
        if volatility > self.config.volatility_threshold:
            # 높은 변동성: 짧은 학습 윈도우, 더 자주 업데이트
            train_size = self.config.min_train_size
            test_size = max(12, self.config.test_size // 2)  # 더 짧은 테스트 기간
        else:
            # 낮은 변동성: 긴 학습 윈도우, 안정적인 예측
            train_size = self.config.max_train_size
            test_size = self.config.test_size
        
        return {
            'train_size': min(train_size, current_idx),  # 사용 가능한 데이터로 제한
            'test_size': test_size,
            'volatility': volatility
        }
    
    def comprehensive_walkforward_analysis(self, data: pd.DataFrame, model_configs: List[Dict]) -> Dict:
        """종합적인 워크포워드 분석"""
        self.logger.info("🚀 고급 워크포워드 최적화 시작...")
        self.logger.info(f"데이터 크기: {data.shape}, 기간: {data.index[0]} ~ {data.index[-1]}")
        
        results = []
        n_samples = len(data)
        
        # 초기 학습 데이터 크기
        current_pos = self.config.min_train_size
        
        while current_pos < n_samples - self.config.test_size:
            # 적응형 윈도우 크기 결정
            window_config = self.adaptive_window_sizing(data, current_pos)
            train_size = window_config['train_size']
            test_size = window_config['test_size']
            
            # 데이터 분할
            train_start = max(0, current_pos - train_size)
            train_data = data.iloc[train_start:current_pos]
            test_data = data.iloc[current_pos:current_pos + test_size]
            
            self.logger.info(f"Period {len(results)+1}: Train {len(train_data)} → Test {len(test_data)}")
            
            # 다중 모델 최적화 및 평가
            period_results = self.optimize_multiple_models(
                train_data, test_data, model_configs, len(results)+1
            )
            
            # 결과에 윈도우 정보 추가
            period_results.update(window_config)
            period_results['train_period'] = (train_data.index[0], train_data.index[-1])
            period_results['test_period'] = (test_data.index[0], test_data.index[-1])
            
            results.append(period_results)
            
            # 다음 위치로 이동
            current_pos += self.config.step_size
        
        # 결과 분석
        comprehensive_analysis = self.analyze_walkforward_results(results)
        
        # 저장
        self.save_optimization_results(results, comprehensive_analysis)
        
        return {
            'period_results': results,
            'comprehensive_analysis': comprehensive_analysis,
            'parameter_stability': self.parameter_stability_tracker,
            'overfitting_analysis': self.overfitting_detector
        }
    
    def optimize_multiple_models(self, train_data: pd.DataFrame, test_data: pd.DataFrame, 
                               model_configs: List[Dict], period_num: int) -> Dict:
        """다중 모델 최적화"""
        model_results = {}
        best_model = None
        best_score = -np.inf
        best_config = None
        
        # 각 모델에 대해 최적화 수행
        for model_config in model_configs:
            model_name = model_config['name']
            model_class = model_config['class']
            param_space = model_config['param_space']
            
            self.logger.info(f"   🔧 {model_name} 최적화 중...")
            
            # 모델별 최적화
            optimal_params = self.optimize_model_parameters(
                train_data, model_class, param_space
            )
            
            # 아웃오브샘플 평가
            oos_performance = self.evaluate_out_of_sample(
                model_class, optimal_params, train_data, test_data
            )
            
            # 과적합 검사
            overfitting_score = self.detect_overfitting(
                model_class, optimal_params, train_data, test_data
            )
            
            model_results[model_name] = {
                'optimal_params': optimal_params,
                'oos_performance': oos_performance,
                'overfitting_score': overfitting_score,
                'is_overfitted': overfitting_score > self.config.overfitting_threshold
            }
            
            # 최적 모델 선택
            if (oos_performance['r2'] > best_score and 
                oos_performance['r2'] > self.config.min_r2_threshold and
                not model_results[model_name]['is_overfitted']):
                best_score = oos_performance['r2']
                best_model = model_name
                best_config = optimal_params
        
        # 파라미터 안정성 추적
        self.track_parameter_stability(period_num, model_results)
        
        return {
            'period': period_num,
            'model_results': model_results,
            'best_model': best_model,
            'best_config': best_config,
            'best_score': best_score
        }
    
    def optimize_model_parameters(self, train_data: pd.DataFrame, model_class, 
                                param_space: Dict) -> Dict:
        """모델 파라미터 최적화"""
        if self.config.optimization_method == 'bayesian' and BAYESIAN_OPT_AVAILABLE:
            return self.bayesian_parameter_optimization(train_data, model_class, param_space)
        elif self.config.optimization_method == 'random':
            return self.random_search_optimization(train_data, model_class, param_space)
        else:
            return self.grid_search_optimization(train_data, model_class, param_space)
    
    def bayesian_parameter_optimization(self, train_data: pd.DataFrame, model_class, 
                                      param_space: Dict) -> Dict:
        """베이지안 최적화"""
        # 파라미터 공간 변환
        dimensions = []
        param_names = []
        
        for param_name, param_range in param_space.items():
            param_names.append(param_name)
            
            if isinstance(param_range, tuple) and len(param_range) == 2:
                # 연속형 파라미터
                dimensions.append(Real(param_range[0], param_range[1], name=param_name))
            elif isinstance(param_range, list):
                # 이산형 파라미터
                if all(isinstance(x, (int, float)) for x in param_range):
                    if all(isinstance(x, int) for x in param_range):
                        dimensions.append(Integer(min(param_range), max(param_range), name=param_name))
                    else:
                        dimensions.append(Real(min(param_range), max(param_range), name=param_name))
                else:
                    dimensions.append(Categorical(param_range, name=param_name))
        
        @use_named_args(dimensions)
        def objective(**params):
            try:
                score = self.cross_validate_model(train_data, model_class, params)
                return -score  # 최소화를 위해 음수 반환
            except Exception:
                return 1000  # 실패시 큰 값 반환
        
        # 베이지안 최적화 실행
        result = gp_minimize(
            objective,
            dimensions,
            n_calls=self.config.n_optimization_trials,
            n_initial_points=10,
            acquisition_func='EI',
            random_state=42
        )
        
        # 최적 파라미터 추출
        optimal_params = {}
        for i, param_name in enumerate(param_names):
            optimal_params[param_name] = result.x[i]
        
        return optimal_params
    
    def random_search_optimization(self, train_data: pd.DataFrame, model_class, 
                                 param_space: Dict) -> Dict:
        """랜덤 서치 최적화"""
        best_score = -np.inf
        best_params = {}
        
        for _ in range(self.config.n_optimization_trials):
            # 랜덤 파라미터 생성
            params = {}
            for param_name, param_range in param_space.items():
                if isinstance(param_range, tuple):
                    # 연속형
                    params[param_name] = np.random.uniform(param_range[0], param_range[1])
                elif isinstance(param_range, list):
                    # 이산형
                    params[param_name] = np.random.choice(param_range)
            
            # 정수형 파라미터 처리
            if 'n_estimators' in params:
                params['n_estimators'] = int(params['n_estimators'])
            if 'max_depth' in params:
                params['max_depth'] = int(params['max_depth'])
            if 'min_samples_split' in params:
                params['min_samples_split'] = int(params['min_samples_split'])
            
            try:
                score = self.cross_validate_model(train_data, model_class, params)
                if score > best_score:
                    best_score = score
                    best_params = params.copy()
            except Exception:
                continue
        
        return best_params
    
    def grid_search_optimization(self, train_data: pd.DataFrame, model_class, 
                               param_space: Dict) -> Dict:
        """그리드 서치 최적화"""
        # 파라미터 조합 생성
        param_combinations = list(ParameterGrid(param_space))
        
        # 조합이 너무 많으면 샘플링
        if len(param_combinations) > self.config.n_optimization_trials:
            param_combinations = np.random.choice(
                param_combinations, 
                self.config.n_optimization_trials, 
                replace=False
            ).tolist()
        
        best_score = -np.inf
        best_params = {}
        
        for params in param_combinations:
            try:
                score = self.cross_validate_model(train_data, model_class, params)
                if score > best_score:
                    best_score = score
                    best_params = params
            except Exception:
                continue
        
        return best_params
    
    def cross_validate_model(self, train_data: pd.DataFrame, model_class, params: Dict) -> float:
        """시계열 교차 검증"""
        if 'target' not in train_data.columns:
            return 0.0
        
        X = train_data.drop(columns=['target'])
        y = train_data['target']
        
        # 수치형 컬럼만 선택
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X = X[numeric_cols]
        
        # NaN 처리
        X = X.fillna(0)
        y = y.fillna(0)
        
        if len(X) < self.config.cv_folds * 2:
            # 데이터가 부족하면 단순 분할
            split_idx = len(X) // 2
            X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
            
            try:
                model = model_class(**params)
                model.fit(X_train, y_train)
                pred = model.predict(X_val)
                return r2_score(y_val, pred)
            except Exception:
                return 0.0
        
        # 시계열 교차 검증
        tscv = TimeSeriesSplit(n_splits=self.config.cv_folds)
        scores = []
        
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            try:
                model = model_class(**params)
                model.fit(X_train, y_train)
                pred = model.predict(X_val)
                score = r2_score(y_val, pred)
                scores.append(score)
            except Exception:
                scores.append(0.0)
        
        return np.mean(scores) if scores else 0.0
    
    def evaluate_out_of_sample(self, model_class, params: Dict, train_data: pd.DataFrame, 
                             test_data: pd.DataFrame) -> Dict:
        """아웃오브샘플 평가"""
        if 'target' not in train_data.columns or 'target' not in test_data.columns:
            return {'r2': 0, 'mae': np.inf, 'rmse': np.inf}
        
        # 데이터 준비
        X_train = train_data.drop(columns=['target'])
        y_train = train_data['target']
        X_test = test_data.drop(columns=['target'])
        y_test = test_data['target']
        
        # 수치형 컬럼만 선택
        numeric_cols = X_train.select_dtypes(include=[np.number]).columns
        X_train = X_train[numeric_cols].fillna(0)
        X_test = X_test[numeric_cols].fillna(0)
        y_train = y_train.fillna(0)
        y_test = y_test.fillna(0)
        
        # 공통 컬럼만 사용
        common_cols = X_train.columns.intersection(X_test.columns)
        X_train = X_train[common_cols]
        X_test = X_test[common_cols]
        
        try:
            # 모델 훈련
            model = model_class(**params)
            model.fit(X_train, y_train)
            
            # 예측
            predictions = model.predict(X_test)
            
            # 성능 지표 계산
            r2 = r2_score(y_test, predictions)
            mae = mean_absolute_error(y_test, predictions)
            rmse = np.sqrt(mean_squared_error(y_test, predictions))
            
            return {
                'r2': r2,
                'mae': mae,
                'rmse': rmse,
                'predictions': predictions.tolist(),
                'actuals': y_test.tolist()
            }
        except Exception:
            return {'r2': 0, 'mae': np.inf, 'rmse': np.inf}
    
    def detect_overfitting(self, model_class, params: Dict, train_data: pd.DataFrame, 
                          test_data: pd.DataFrame) -> float:
        """과적합 검사"""
        # 훈련 데이터 성능
        train_score = self.cross_validate_model(train_data, model_class, params)
        
        # 테스트 데이터 성능
        test_performance = self.evaluate_out_of_sample(model_class, params, train_data, test_data)
        test_score = test_performance['r2']
        
        # 과적합 점수 계산 (훈련 성능과 테스트 성능의 차이)
        if train_score > 0:
            overfitting_score = max(0, (train_score - test_score) / train_score)
        else:
            overfitting_score = 1.0  # 훈련도 실패했으면 완전 과적합으로 간주
        
        return overfitting_score
    
    def track_parameter_stability(self, period_num: int, model_results: Dict):
        """파라미터 안정성 추적"""
        for model_name, result in model_results.items():
            if model_name not in self.parameter_stability_tracker:
                self.parameter_stability_tracker[model_name] = {}
            
            params = result['optimal_params']
            for param_name, param_value in params.items():
                if param_name not in self.parameter_stability_tracker[model_name]:
                    self.parameter_stability_tracker[model_name][param_name] = []
                
                self.parameter_stability_tracker[model_name][param_name].append({
                    'period': period_num,
                    'value': param_value
                })
    
    def analyze_walkforward_results(self, results: List[Dict]) -> Dict:
        """워크포워드 결과 종합 분석"""
        self.logger.info("📊 워크포워드 결과 분석 중...")
        
        # 기본 통계
        best_scores = [r['best_score'] for r in results if r['best_score'] is not None]
        model_usage = {}
        parameter_evolution = {}
        
        for result in results:
            # 모델 사용 빈도
            best_model = result.get('best_model')
            if best_model:
                model_usage[best_model] = model_usage.get(best_model, 0) + 1
        
        # 파라미터 안정성 분석
        stability_analysis = {}
        for model_name, param_history in self.parameter_stability_tracker.items():
            stability_analysis[model_name] = {}
            
            for param_name, values in param_history.items():
                param_values = [v['value'] for v in values]
                if len(param_values) > 1:
                    stability_analysis[model_name][param_name] = {
                        'mean': np.mean(param_values),
                        'std': np.std(param_values),
                        'cv': np.std(param_values) / (np.mean(param_values) + 1e-8),
                        'stability_score': 1 / (1 + np.std(param_values))
                    }
        
        # 성능 추이 분석
        performance_trend = self.analyze_performance_trend(results)
        
        # 적응형 윈도우 효과 분석
        adaptive_window_analysis = self.analyze_adaptive_window_effect(results)
        
        analysis = {
            'summary_stats': {
                'total_periods': len(results),
                'avg_performance': np.mean(best_scores) if best_scores else 0,
                'std_performance': np.std(best_scores) if best_scores else 0,
                'best_performance': max(best_scores) if best_scores else 0,
                'worst_performance': min(best_scores) if best_scores else 0
            },
            'model_usage': model_usage,
            'parameter_stability': stability_analysis,
            'performance_trend': performance_trend,
            'adaptive_window_analysis': adaptive_window_analysis,
            'overfitting_summary': self.summarize_overfitting_detection(results)
        }
        
        return analysis
    
    def analyze_performance_trend(self, results: List[Dict]) -> Dict:
        """성능 추이 분석"""
        periods = [r['period'] for r in results]
        scores = [r['best_score'] if r['best_score'] is not None else 0 for r in results]
        
        if len(scores) < 2:
            return {'trend': 'insufficient_data'}
        
        # 선형 추세 분석
        from scipy import stats
        slope, intercept, r_value, p_value, std_err = stats.linregress(periods, scores)
        
        # 이동 평균
        window_size = min(5, len(scores) // 2)
        if window_size > 0:
            moving_avg = pd.Series(scores).rolling(window=window_size).mean().tolist()
        else:
            moving_avg = scores
        
        return {
            'trend': 'improving' if slope > 0 else 'declining',
            'slope': slope,
            'r_squared': r_value ** 2,
            'p_value': p_value,
            'moving_average': moving_avg,
            'volatility': np.std(scores)
        }
    
    def analyze_adaptive_window_effect(self, results: List[Dict]) -> Dict:
        """적응형 윈도우 효과 분석"""
        if not self.config.adaptive_window:
            return {'enabled': False}
        
        train_sizes = [r.get('train_size', 0) for r in results]
        test_sizes = [r.get('test_size', 0) for r in results]
        volatilities = [r.get('volatility', 0) for r in results if r.get('volatility') is not None]
        scores = [r['best_score'] if r['best_score'] is not None else 0 for r in results]
        
        analysis = {
            'enabled': True,
            'train_size_stats': {
                'mean': np.mean(train_sizes),
                'std': np.std(train_sizes),
                'min': min(train_sizes),
                'max': max(train_sizes)
            },
            'test_size_stats': {
                'mean': np.mean(test_sizes),
                'std': np.std(test_sizes),
                'min': min(test_sizes),
                'max': max(test_sizes)
            }
        }
        
        # 변동성과 성능의 관계
        if len(volatilities) > 1 and len(scores) > 1:
            corr_vol_perf = np.corrcoef(volatilities, scores[:len(volatilities)])[0, 1]
            analysis['volatility_performance_correlation'] = corr_vol_perf
        
        return analysis
    
    def summarize_overfitting_detection(self, results: List[Dict]) -> Dict:
        """과적합 검출 요약"""
        overfitted_counts = {}
        total_counts = {}
        
        for result in results:
            for model_name, model_result in result.get('model_results', {}).items():
                total_counts[model_name] = total_counts.get(model_name, 0) + 1
                if model_result.get('is_overfitted', False):
                    overfitted_counts[model_name] = overfitted_counts.get(model_name, 0) + 1
        
        overfitting_rates = {}
        for model_name, total in total_counts.items():
            overfitted = overfitted_counts.get(model_name, 0)
            overfitting_rates[model_name] = overfitted / total if total > 0 else 0
        
        return {
            'overfitting_rates': overfitting_rates,
            'total_evaluations': total_counts,
            'overfitted_evaluations': overfitted_counts
        }
    
    def save_optimization_results(self, results: List[Dict], analysis: Dict):
        """최적화 결과 저장"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # JSON 저장
        output_data = {
            'timestamp': timestamp,
            'config': self.config.__dict__,
            'results': results,
            'analysis': analysis
        }
        
        json_path = os.path.join(self.data_path, f'walkforward_optimization_{timestamp}.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False, default=str)
        
        self.logger.info(f"최적화 결과 저장: {json_path}")
    
    def create_optimization_dashboard(self, results: List[Dict], analysis: Dict) -> str:
        """최적화 대시보드 생성"""
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=(
                "성능 추이", "모델 사용 빈도", "파라미터 안정성",
                "적응형 윈도우 크기", "과적합 검출률", "성능 분포",
                "변동성 vs 성능", "학습 윈도우 효과", "예측 정확도"
            ),
            specs=[[{"type": "scatter"}, {"type": "pie"}, {"type": "heatmap"}],
                   [{"type": "scatter"}, {"type": "bar"}, {"type": "histogram"}],
                   [{"type": "scatter"}, {"type": "box"}, {"type": "scatter"}]]
        )
        
        # 성능 추이
        periods = [r['period'] for r in results]
        scores = [r['best_score'] if r['best_score'] is not None else 0 for r in results]
        
        fig.add_trace(
            go.Scatter(x=periods, y=scores, mode='lines+markers', name='성능'),
            row=1, col=1
        )
        
        # 모델 사용 빈도
        model_usage = analysis['model_usage']
        fig.add_trace(
            go.Pie(labels=list(model_usage.keys()), values=list(model_usage.values())),
            row=1, col=2
        )
        
        # 추가 차트들...
        
        fig.update_layout(
            title="🔧 워크포워드 최적화 대시보드",
            height=1200,
            showlegend=True,
            template='plotly_dark'
        )
        
        dashboard_path = os.path.join(self.data_path, 'walkforward_optimization_dashboard.html')
        fig.write_html(dashboard_path, include_plotlyjs=True)
        
        return dashboard_path

def create_default_model_configs() -> List[Dict]:
    """기본 모델 설정 생성"""
    return [
        {
            'name': 'RandomForest',
            'class': RandomForestRegressor,
            'param_space': {
                'n_estimators': [50, 100, 200, 300],
                'max_depth': [5, 10, 15, 20],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        },
        {
            'name': 'XGBoost',
            'class': xgb.XGBRegressor,
            'param_space': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0]
            }
        },
        {
            'name': 'LightGBM',
            'class': lgb.LGBMRegressor,
            'param_space': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'feature_fraction': [0.8, 0.9, 1.0]
            }
        },
        {
            'name': 'GradientBoosting',
            'class': GradientBoostingRegressor,
            'param_space': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0]
            }
        }
    ]

def main():
    """메인 실행 함수"""
    print("🔧 고급 워크포워드 최적화 시스템")
    print("="*60)
    
    # 설정
    config = OptimizationConfig(
        min_train_size=500,
        max_train_size=1500,
        test_size=48,
        adaptive_window=True,
        optimization_method='bayesian' if BAYESIAN_OPT_AVAILABLE else 'random',
        n_optimization_trials=50
    )
    
    # 최적화기 초기화
    optimizer = AdvancedWalkForwardOptimizer(config)
    
    # 데이터 로드 (임시 데이터)
    n_samples = 2000
    np.random.seed(42)
    data = pd.DataFrame({
        'feature1': np.random.randn(n_samples).cumsum(),
        'feature2': np.random.randn(n_samples),
        'feature3': np.random.randn(n_samples),
        'target': np.random.randn(n_samples) * 0.02
    })
    data.index = pd.date_range(start='2024-01-01', periods=n_samples, freq='H')
    
    # 모델 설정
    model_configs = create_default_model_configs()
    
    # 워크포워드 최적화 실행
    results = optimizer.comprehensive_walkforward_analysis(data, model_configs)
    
    print(f"\n✅ 최적화 완료!")
    print(f"총 기간: {len(results['period_results'])}")
    print(f"평균 성능: {results['comprehensive_analysis']['summary_stats']['avg_performance']:.4f}")
    
    # 대시보드 생성
    dashboard_path = optimizer.create_optimization_dashboard(
        results['period_results'], 
        results['comprehensive_analysis']
    )
    print(f"대시보드: {dashboard_path}")

if __name__ == "__main__":
    main()