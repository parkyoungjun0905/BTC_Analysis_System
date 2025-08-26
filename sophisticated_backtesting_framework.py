#!/usr/bin/env python3
"""
🎯 정교한 비트코인 예측 모델 백테스팅 프레임워크 v3.0
- 현실적인 시장 시뮬레이션 (거래비용, 슬리피지, 유동성 제약)
- 워크포워드 최적화 (롤링 윈도우, 아웃오브샘플 검증)
- 리스크 조정 성과 지표 (샤프/소르티노 비율, 최대낙폭, VaR)
- 시장 상황별 분석 (강세/약세/횡보장 성능 분석)
- 통계적 유의성 검정 (부트스트랩, 가설검정)
- 베이지안 최적화 포함 하이퍼파라미터 최적화
- 대화형 시각화 및 상세 리포팅
"""

import numpy as np
import pandas as pd
import warnings
import logging
from datetime import datetime, timedelta
import json
import os
from typing import Dict, List, Tuple, Optional, Union
import pickle
import asyncio
from dataclasses import dataclass
from abc import ABC, abstractmethod
import itertools

# 핵심 라이브러리
from sklearn.model_selection import TimeSeriesSplit, ParameterGrid
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.feature_selection import SelectFromModel, RFE
import xgboost as xgb
import lightgbm as lgb

# 시각화 라이브러리
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# 통계 및 최적화
from scipy import stats
from scipy.optimize import minimize
try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer
    from skopt.utils import use_named_args
    BAYESIAN_OPT_AVAILABLE = True
except ImportError:
    BAYESIAN_OPT_AVAILABLE = False
    print("⚠️ scikit-optimize 미설치: 그리드 서치만 사용")

warnings.filterwarnings('ignore')

@dataclass
class MarketCondition:
    """시장 상황 데이터 클래스"""
    name: str
    start_date: datetime
    end_date: datetime
    volatility: float
    trend: str  # 'bull', 'bear', 'sideways'
    description: str

@dataclass
class BacktestConfig:
    """백테스트 설정 클래스"""
    # 거래 비용
    transaction_cost: float = 0.001  # 0.1% 거래 수수료
    slippage: float = 0.0005  # 0.05% 슬리피지
    min_order_size: float = 100.0  # 최소 주문 크기 ($)
    
    # 백테스트 설정
    initial_capital: float = 10000.0  # 초기 자본
    max_position_size: float = 0.95  # 최대 포지션 크기
    lookback_window: int = 720  # 학습 데이터 윈도우 (시간)
    rebalance_frequency: int = 24  # 리밸런싱 주기 (시간)
    
    # 리스크 관리
    max_drawdown_limit: float = 0.20  # 20% 최대 낙폭 제한
    stop_loss: float = 0.05  # 5% 스탑로스
    take_profit: float = 0.15  # 15% 이익실현
    
    # 검증 설정
    bootstrap_samples: int = 1000  # 부트스트랩 샘플 수
    confidence_level: float = 0.95  # 신뢰구간
    min_samples: int = 100  # 최소 샘플 수

class MarketSimulator:
    """현실적인 시장 시뮬레이션 모듈"""
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.order_book = {'bid': [], 'ask': []}
        self.market_impact_cache = {}
        
    def calculate_transaction_costs(self, trade_size: float, price: float) -> float:
        """거래 비용 계산"""
        # 기본 거래 수수료
        base_cost = trade_size * price * self.config.transaction_cost
        
        # 시장 영향비용 (거래량에 따라 증가)
        market_impact = self.calculate_market_impact(trade_size, price)
        
        # 슬리피지
        slippage_cost = trade_size * price * self.config.slippage
        
        return base_cost + market_impact + slippage_cost
    
    def calculate_market_impact(self, trade_size: float, price: float) -> float:
        """시장 영향비용 계산 (Square-root law)"""
        # 거래 크기에 따른 비선형 영향
        notional = trade_size * price
        
        # 임시 영향 (일시적)
        temporary_impact = 0.01 * np.sqrt(notional / 100000)  # 스케일링 팩터
        
        # 영구 영향 (가격에 영구적으로 반영)
        permanent_impact = 0.005 * (notional / 100000)
        
        return (temporary_impact + permanent_impact) * price
    
    def simulate_order_execution(self, order_size: float, current_price: float, 
                                volume_24h: float) -> Dict[str, float]:
        """주문 실행 시뮬레이션"""
        # 유동성 제약 확인
        max_executable = volume_24h * 0.001  # 24시간 거래량의 0.1%까지만
        executable_size = min(order_size, max_executable)
        
        if executable_size < self.config.min_order_size:
            return {
                'executed_size': 0,
                'execution_price': current_price,
                'total_cost': 0,
                'slippage': 0,
                'rejected': True
            }
        
        # 실행 가격 계산 (슬리피지 포함)
        slippage_factor = (executable_size / volume_24h) * 100  # 거래량 대비 슬리피지
        execution_price = current_price * (1 + slippage_factor * self.config.slippage)
        
        # 총 거래 비용
        total_cost = self.calculate_transaction_costs(executable_size, execution_price)
        
        return {
            'executed_size': executable_size,
            'execution_price': execution_price,
            'total_cost': total_cost,
            'slippage': abs(execution_price - current_price) / current_price,
            'rejected': False
        }

class WalkForwardOptimizer:
    """워크포워드 최적화 시스템"""
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.optimization_history = []
        self.parameter_stability = {}
        
    def walk_forward_analysis(self, data: pd.DataFrame, model_class, 
                             param_grid: Dict) -> Dict:
        """워크포워드 분석 실행"""
        print("🚀 워크포워드 최적화 시작...")
        
        results = []
        optimal_params_history = []
        
        # 롤링 윈도우 설정
        n_samples = len(data)
        train_size = self.config.lookback_window
        test_size = self.config.rebalance_frequency
        
        # 워크포워드 루프
        for i in range(train_size, n_samples - test_size, test_size):
            # 훈련/테스트 데이터 분할
            train_data = data.iloc[i-train_size:i]
            test_data = data.iloc[i:i+test_size]
            
            print(f"   📊 Period {len(results)+1}: Train {len(train_data)} → Test {len(test_data)}")
            
            # 인샘플 최적화
            best_params = self.optimize_parameters(train_data, model_class, param_grid)
            optimal_params_history.append(best_params)
            
            # 아웃오브샘플 테스트
            model = model_class(**best_params)
            oos_performance = self.evaluate_out_of_sample(
                model, train_data, test_data
            )
            
            results.append({
                'period': len(results) + 1,
                'train_start': train_data.index[0],
                'train_end': train_data.index[-1], 
                'test_start': test_data.index[0],
                'test_end': test_data.index[-1],
                'optimal_params': best_params,
                'oos_performance': oos_performance
            })
        
        # 파라미터 안정성 분석
        self.analyze_parameter_stability(optimal_params_history)
        
        return {
            'results': results,
            'parameter_history': optimal_params_history,
            'stability_analysis': self.parameter_stability
        }
    
    def optimize_parameters(self, train_data: pd.DataFrame, model_class, 
                          param_grid: Dict) -> Dict:
        """파라미터 최적화"""
        if BAYESIAN_OPT_AVAILABLE and len(param_grid) > 3:
            return self.bayesian_optimization(train_data, model_class, param_grid)
        else:
            return self.grid_search_optimization(train_data, model_class, param_grid)
    
    def bayesian_optimization(self, train_data: pd.DataFrame, model_class, 
                            param_grid: Dict) -> Dict:
        """베이지안 최적화"""
        # 파라미터 공간 정의
        dimensions = []
        param_names = list(param_grid.keys())
        
        for param, values in param_grid.items():
            if isinstance(values[0], int):
                dimensions.append(Integer(min(values), max(values), name=param))
            else:
                dimensions.append(Real(min(values), max(values), name=param))
        
        @use_named_args(dimensions)
        def objective(**params):
            try:
                model = model_class(**params)
                score = self.cross_validate_model(model, train_data)
                return -score  # 최소화를 위해 음수화
            except:
                return 1000  # 실패시 큰 값 반환
        
        # 베이지안 최적화 실행
        result = gp_minimize(objective, dimensions, n_calls=50, random_state=42)
        
        # 최적 파라미터 반환
        optimal_params = {}
        for i, param_name in enumerate(param_names):
            optimal_params[param_name] = result.x[i]
        
        return optimal_params
    
    def grid_search_optimization(self, train_data: pd.DataFrame, model_class,
                               param_grid: Dict) -> Dict:
        """그리드 서치 최적화"""
        best_score = -np.inf
        best_params = {}
        
        # 모든 파라미터 조합 탐색
        param_combinations = list(ParameterGrid(param_grid))
        
        for params in param_combinations:
            try:
                model = model_class(**params)
                score = self.cross_validate_model(model, train_data)
                
                if score > best_score:
                    best_score = score
                    best_params = params
            except:
                continue
        
        return best_params
    
    def cross_validate_model(self, model, data: pd.DataFrame) -> float:
        """시계열 교차 검증"""
        tscv = TimeSeriesSplit(n_splits=5)
        scores = []
        
        X = data.drop(columns=['target'])
        y = data['target']
        
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            model.fit(X_train, y_train)
            pred = model.predict(X_val)
            score = r2_score(y_val, pred)
            scores.append(score)
        
        return np.mean(scores)
    
    def evaluate_out_of_sample(self, model, train_data: pd.DataFrame, 
                              test_data: pd.DataFrame) -> Dict:
        """아웃오브샘플 성능 평가"""
        X_train = train_data.drop(columns=['target'])
        y_train = train_data['target']
        X_test = test_data.drop(columns=['target'])
        y_test = test_data['target']
        
        # 모델 훈련
        model.fit(X_train, y_train)
        
        # 예측
        predictions = model.predict(X_test)
        
        # 성과 지표 계산
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)
        
        return {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'predictions': predictions,
            'actuals': y_test.values
        }
    
    def analyze_parameter_stability(self, param_history: List[Dict]):
        """파라미터 안정성 분석"""
        if not param_history:
            return
            
        # 각 파라미터별 통계
        all_params = list(param_history[0].keys())
        
        for param in all_params:
            values = [p[param] for p in param_history]
            
            self.parameter_stability[param] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'cv': np.std(values) / np.mean(values) if np.mean(values) != 0 else 0,
                'stability_score': 1 / (1 + np.std(values))  # 안정성 점수 (낮은 변동성일수록 높음)
            }

class RiskAdjustedMetrics:
    """리스크 조정 성과 지표 시스템"""
    
    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate
        
    def calculate_all_metrics(self, returns: np.ndarray, 
                            portfolio_values: np.ndarray) -> Dict:
        """모든 리스크 조정 지표 계산"""
        return {
            'sharpe_ratio': self.sharpe_ratio(returns),
            'sortino_ratio': self.sortino_ratio(returns),
            'calmar_ratio': self.calmar_ratio(returns, portfolio_values),
            'max_drawdown': self.max_drawdown(portfolio_values),
            'var_95': self.value_at_risk(returns, 0.95),
            'cvar_95': self.conditional_var(returns, 0.95),
            'omega_ratio': self.omega_ratio(returns),
            'tail_ratio': self.tail_ratio(returns),
            'skewness': stats.skew(returns),
            'kurtosis': stats.kurtosis(returns),
            'downside_deviation': self.downside_deviation(returns)
        }
    
    def sharpe_ratio(self, returns: np.ndarray) -> float:
        """샤프 비율"""
        excess_returns = returns - self.risk_free_rate / 365  # 일간 무위험 수익률
        return np.mean(excess_returns) / np.std(returns) * np.sqrt(365) if np.std(returns) > 0 else 0
    
    def sortino_ratio(self, returns: np.ndarray) -> float:
        """소르티노 비율"""
        excess_returns = returns - self.risk_free_rate / 365
        downside_returns = returns[returns < 0]
        downside_std = np.std(downside_returns) if len(downside_returns) > 0 else np.std(returns)
        return np.mean(excess_returns) / downside_std * np.sqrt(365) if downside_std > 0 else 0
    
    def calmar_ratio(self, returns: np.ndarray, portfolio_values: np.ndarray) -> float:
        """칼마 비율"""
        annual_return = (portfolio_values[-1] / portfolio_values[0]) ** (365 / len(returns)) - 1
        max_dd = self.max_drawdown(portfolio_values)
        return annual_return / abs(max_dd) if max_dd != 0 else 0
    
    def max_drawdown(self, portfolio_values: np.ndarray) -> float:
        """최대 낙폭"""
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - peak) / peak
        return np.min(drawdown)
    
    def value_at_risk(self, returns: np.ndarray, confidence_level: float) -> float:
        """VaR (Value at Risk)"""
        return np.percentile(returns, (1 - confidence_level) * 100)
    
    def conditional_var(self, returns: np.ndarray, confidence_level: float) -> float:
        """CVaR (Conditional Value at Risk)"""
        var = self.value_at_risk(returns, confidence_level)
        return np.mean(returns[returns <= var])
    
    def omega_ratio(self, returns: np.ndarray, threshold: float = 0) -> float:
        """오메가 비율"""
        gains = returns[returns > threshold] - threshold
        losses = threshold - returns[returns <= threshold]
        return np.sum(gains) / np.sum(losses) if np.sum(losses) > 0 else np.inf
    
    def tail_ratio(self, returns: np.ndarray) -> float:
        """테일 비율 (상위 5% 수익률 / 하위 5% 손실률)"""
        upper_tail = np.percentile(returns, 95)
        lower_tail = np.percentile(returns, 5)
        return abs(upper_tail) / abs(lower_tail) if lower_tail != 0 else np.inf
    
    def downside_deviation(self, returns: np.ndarray, target_return: float = 0) -> float:
        """하방 편차"""
        downside_returns = np.minimum(returns - target_return, 0)
        return np.sqrt(np.mean(downside_returns ** 2))

class MarketRegimeAnalyzer:
    """시장 상황별 분석 모듈"""
    
    def __init__(self):
        self.regimes = []
        self.regime_performance = {}
        
    def identify_market_regimes(self, data: pd.DataFrame, 
                              price_col: str = 'price') -> List[MarketCondition]:
        """시장 상황 식별"""
        prices = data[price_col]
        returns = prices.pct_change().dropna()
        
        # 60일 롤링 윈도우로 변동성과 추세 계산
        window = 60
        volatility = returns.rolling(window).std() * np.sqrt(365)  # 연율화 변동성
        trend = prices.rolling(window).apply(lambda x: self.calculate_trend(x))
        
        regimes = []
        current_regime = None
        regime_start = None
        
        for i, (date, vol, tr) in enumerate(zip(data.index, volatility, trend)):
            if pd.isna(vol) or pd.isna(tr):
                continue
                
            # 변동성 임계값 (중간값 기준)
            vol_median = volatility.median()
            
            # 시장 상황 분류
            if tr > 0.1 and vol < vol_median * 1.2:
                regime_type = 'bull_low_vol'
                description = '강세장 (낮은 변동성)'
            elif tr > 0.1 and vol >= vol_median * 1.2:
                regime_type = 'bull_high_vol'
                description = '강세장 (높은 변동성)'
            elif tr < -0.1 and vol < vol_median * 1.2:
                regime_type = 'bear_low_vol'
                description = '약세장 (낮은 변동성)'
            elif tr < -0.1 and vol >= vol_median * 1.2:
                regime_type = 'bear_high_vol'
                description = '약세장 (높은 변동성)'
            elif abs(tr) <= 0.1 and vol < vol_median:
                regime_type = 'sideways_low_vol'
                description = '횡보장 (낮은 변동성)'
            else:
                regime_type = 'sideways_high_vol'
                description = '횡보장 (높은 변동성)'
            
            # 상황 변경 감지
            if current_regime != regime_type:
                # 이전 구간 종료
                if current_regime and regime_start:
                    regimes.append(MarketCondition(
                        name=current_regime,
                        start_date=regime_start,
                        end_date=date,
                        volatility=vol,
                        trend='bull' if 'bull' in current_regime else 'bear' if 'bear' in current_regime else 'sideways',
                        description=description
                    ))
                
                # 새로운 구간 시작
                current_regime = regime_type
                regime_start = date
        
        # 마지막 구간 처리
        if current_regime and regime_start:
            regimes.append(MarketCondition(
                name=current_regime,
                start_date=regime_start,
                end_date=data.index[-1],
                volatility=volatility.iloc[-1],
                trend='bull' if 'bull' in current_regime else 'bear' if 'bear' in current_regime else 'sideways',
                description=description
            ))
        
        self.regimes = regimes
        return regimes
    
    def calculate_trend(self, prices: pd.Series) -> float:
        """추세 계산 (선형 회귀 기울기)"""
        if len(prices) < 2:
            return 0
        x = np.arange(len(prices))
        slope, _, _, _, _ = stats.linregress(x, prices)
        return slope / prices.mean()  # 정규화된 기울기
    
    def analyze_regime_performance(self, backtest_results: Dict, 
                                 regimes: List[MarketCondition]) -> Dict:
        """시장 상황별 성능 분석"""
        regime_performance = {}
        
        for regime in regimes:
            # 해당 기간 결과 필터링
            regime_results = self.filter_results_by_period(
                backtest_results, regime.start_date, regime.end_date
            )
            
            if regime_results:
                regime_performance[regime.name] = {
                    'period': f"{regime.start_date.strftime('%Y-%m-%d')} ~ {regime.end_date.strftime('%Y-%m-%d')}",
                    'trend': regime.trend,
                    'description': regime.description,
                    'performance': regime_results,
                    'duration_days': (regime.end_date - regime.start_date).days
                }
        
        self.regime_performance = regime_performance
        return regime_performance
    
    def filter_results_by_period(self, results: Dict, start_date: datetime, 
                               end_date: datetime) -> Dict:
        """기간별 결과 필터링"""
        # 구현 세부사항은 백테스트 결과 구조에 따라 조정
        return {}

class StatisticalSignificanceTester:
    """통계적 유의성 검정 시스템"""
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        
    def bootstrap_analysis(self, returns: np.ndarray, 
                         metric_func, n_bootstrap: int = 1000) -> Dict:
        """부트스트랩 분석"""
        n_samples = len(returns)
        bootstrap_metrics = []
        
        for _ in range(n_bootstrap):
            # 복원 추출
            bootstrap_sample = np.random.choice(returns, size=n_samples, replace=True)
            metric = metric_func(bootstrap_sample)
            bootstrap_metrics.append(metric)
        
        bootstrap_metrics = np.array(bootstrap_metrics)
        
        # 신뢰구간 계산
        alpha = 1 - self.config.confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        return {
            'mean': np.mean(bootstrap_metrics),
            'std': np.std(bootstrap_metrics),
            'confidence_interval': [
                np.percentile(bootstrap_metrics, lower_percentile),
                np.percentile(bootstrap_metrics, upper_percentile)
            ],
            'bootstrap_distribution': bootstrap_metrics
        }
    
    def significance_test(self, strategy_returns: np.ndarray, 
                         benchmark_returns: np.ndarray) -> Dict:
        """전략 vs 벤치마크 유의성 검정"""
        # t-검정
        t_stat, t_pvalue = stats.ttest_ind(strategy_returns, benchmark_returns)
        
        # Wilcoxon 순위합 검정 (비모수)
        wilcoxon_stat, wilcoxon_pvalue = stats.ranksums(strategy_returns, benchmark_returns)
        
        # Kolmogorov-Smirnov 검정
        ks_stat, ks_pvalue = stats.ks_2samp(strategy_returns, benchmark_returns)
        
        return {
            't_test': {'statistic': t_stat, 'p_value': t_pvalue},
            'wilcoxon_test': {'statistic': wilcoxon_stat, 'p_value': wilcoxon_pvalue},
            'ks_test': {'statistic': ks_stat, 'p_value': ks_pvalue},
            'significant': t_pvalue < 0.05 and wilcoxon_pvalue < 0.05
        }
    
    def multiple_comparisons_correction(self, p_values: List[float], 
                                      method: str = 'bonferroni') -> Dict:
        """다중비교 보정"""
        p_values = np.array(p_values)
        n_comparisons = len(p_values)
        
        if method == 'bonferroni':
            corrected_alpha = 0.05 / n_comparisons
            significant = p_values < corrected_alpha
        elif method == 'holm':
            sorted_indices = np.argsort(p_values)
            corrected_alpha = 0.05 / (n_comparisons - np.arange(n_comparisons))
            significant = np.zeros_like(p_values, dtype=bool)
            for i, idx in enumerate(sorted_indices):
                if p_values[idx] < corrected_alpha[i]:
                    significant[idx] = True
                else:
                    break
        else:
            significant = p_values < 0.05
        
        return {
            'method': method,
            'corrected_alpha': corrected_alpha if method == 'bonferroni' else None,
            'significant': significant,
            'adjusted_p_values': p_values * n_comparisons if method == 'bonferroni' else p_values
        }

class InteractiveVisualizationSystem:
    """대화형 시각화 시스템"""
    
    def __init__(self, output_path: str):
        self.output_path = output_path
        
    def create_comprehensive_dashboard(self, results: Dict) -> str:
        """종합 대시보드 생성"""
        fig = make_subplots(
            rows=4, cols=3,
            subplot_titles=(
                "포트폴리오 가치 변화", "월별 수익률", "드로다운 분석",
                "롤링 샤프 비율", "VaR 분석", "수익률 분포",
                "시장 상황별 성능", "파라미터 안정성", "예측 정확도",
                "거래 통계", "리스크 지표", "성능 비교"
            ),
            specs=[[{"secondary_y": True}, {"type": "bar"}, {"type": "scatter"}],
                   [{"type": "scatter"}, {"type": "box"}, {"type": "histogram"}],
                   [{"type": "bar"}, {"type": "heatmap"}, {"type": "scatter"}],
                   [{"type": "table"}, {"type": "bar"}, {"type": "bar"}]],
            vertical_spacing=0.08,
            horizontal_spacing=0.08
        )
        
        # 1. 포트폴리오 가치 변화
        if 'portfolio_values' in results:
            fig.add_trace(
                go.Scatter(
                    y=results['portfolio_values'],
                    mode='lines',
                    name='포트폴리오 가치',
                    line=dict(color='blue', width=2)
                ),
                row=1, col=1
            )
        
        # 2. 월별 수익률
        if 'monthly_returns' in results:
            fig.add_trace(
                go.Bar(
                    x=list(results['monthly_returns'].keys()),
                    y=list(results['monthly_returns'].values()),
                    name='월별 수익률',
                    marker_color=['green' if x > 0 else 'red' for x in results['monthly_returns'].values()]
                ),
                row=1, col=2
            )
        
        # 3. 드로다운 분석
        if 'drawdown_series' in results:
            fig.add_trace(
                go.Scatter(
                    y=results['drawdown_series'],
                    mode='lines',
                    fill='tozeroy',
                    name='드로다운',
                    line=dict(color='red'),
                    fillcolor='rgba(255,0,0,0.3)'
                ),
                row=1, col=3
            )
        
        # 추가 차트들...
        
        fig.update_layout(
            title="🎯 비트코인 예측 모델 백테스팅 종합 대시보드",
            height=1600,
            showlegend=True,
            template='plotly_dark'
        )
        
        output_file = os.path.join(self.output_path, "comprehensive_backtesting_dashboard.html")
        fig.write_html(output_file, include_plotlyjs=True)
        
        return output_file
    
    def create_regime_analysis_chart(self, regime_performance: Dict) -> str:
        """시장 상황별 분석 차트"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("상황별 수익률", "상황별 샤프비율", "상황별 최대낙폭", "상황 지속기간")
        )
        
        regimes = list(regime_performance.keys())
        
        # 차트 데이터 추출 및 시각화
        # ...
        
        output_file = os.path.join(self.output_path, "regime_analysis_chart.html")
        fig.write_html(output_file, include_plotlyjs=True)
        
        return output_file

class SophisticatedBacktestingFramework:
    """정교한 백테스팅 프레임워크 메인 클래스"""
    
    def __init__(self, config: BacktestConfig = None):
        self.config = config or BacktestConfig()
        self.data_path = "/Users/parkyoungjun/Desktop/BTC_Analysis_System"
        
        # 서브시스템 초기화
        self.market_simulator = MarketSimulator(self.config)
        self.walk_forward_optimizer = WalkForwardOptimizer(self.config)
        self.risk_metrics = RiskAdjustedMetrics()
        self.regime_analyzer = MarketRegimeAnalyzer()
        self.significance_tester = StatisticalSignificanceTester(self.config)
        self.visualizer = InteractiveVisualizationSystem(self.data_path)
        
        # 로깅 설정
        self.setup_logging()
        
    def setup_logging(self):
        """로깅 설정"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.data_path, 'sophisticated_backtest.log'), encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def load_data(self) -> pd.DataFrame:
        """데이터 로드 및 전처리"""
        try:
            # 데이터 로드 (기존 시스템과 호환)
            data_files = [
                "ai_matrix_complete.csv",
                "complete_indicators_data.csv",
                "btc_1h_data.csv"
            ]
            
            for filename in data_files:
                filepath = os.path.join(self.data_path, "historical_data", filename)
                if os.path.exists(filepath):
                    df = pd.read_csv(filepath)
                    self.logger.info(f"데이터 로드 완료: {filename} ({df.shape})")
                    return self.preprocess_data(df)
            
            raise FileNotFoundError("사용 가능한 데이터 파일이 없습니다.")
            
        except Exception as e:
            self.logger.error(f"데이터 로드 실패: {e}")
            raise
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """데이터 전처리"""
        # 수치형 컬럼만 선택
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        df_clean = df[numeric_columns].copy()
        
        # 결측값 처리
        df_clean = df_clean.ffill().bfill().fillna(0)
        
        # 무한대값 처리
        df_clean = df_clean.replace([np.inf, -np.inf], 0)
        
        # 타겟 변수 생성 (1시간 후 가격 변화율)
        if 'price' in df_clean.columns:
            df_clean['target'] = df_clean['price'].pct_change().shift(-1)
        else:
            # 첫 번째 컬럼을 가격으로 가정
            price_col = df_clean.columns[0]
            df_clean['target'] = df_clean[price_col].pct_change().shift(-1)
        
        # 마지막 행 제거 (타겟이 NaN)
        df_clean = df_clean[:-1]
        
        # 시간 인덱스 생성
        df_clean.index = pd.date_range(start='2024-01-01', periods=len(df_clean), freq='H')
        
        return df_clean
    
    def run_comprehensive_backtest(self, model_class=RandomForestRegressor, 
                                 param_grid: Dict = None) -> Dict:
        """종합적인 백테스트 실행"""
        self.logger.info("🚀 정교한 백테스팅 프레임워크 시작...")
        
        try:
            # 1. 데이터 로드
            data = self.load_data()
            self.logger.info(f"데이터 로드 완료: {data.shape}")
            
            # 2. 시장 상황 식별
            regimes = self.regime_analyzer.identify_market_regimes(data)
            self.logger.info(f"시장 상황 식별 완료: {len(regimes)}개 구간")
            
            # 3. 파라미터 그리드 설정
            if param_grid is None:
                param_grid = {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [5, 10, 15],
                    'min_samples_split': [2, 5, 10]
                }
            
            # 4. 워크포워드 최적화
            wf_results = self.walk_forward_optimizer.walk_forward_analysis(
                data, model_class, param_grid
            )
            
            # 5. 백테스트 실행
            backtest_results = self.execute_backtesting_simulation(data, wf_results)
            
            # 6. 리스크 조정 지표 계산
            risk_metrics = self.calculate_comprehensive_risk_metrics(backtest_results)
            
            # 7. 시장 상황별 성능 분석
            regime_performance = self.regime_analyzer.analyze_regime_performance(
                backtest_results, regimes
            )
            
            # 8. 통계적 유의성 검정
            significance_results = self.perform_significance_testing(backtest_results)
            
            # 9. 결과 통합
            comprehensive_results = {
                'backtest_config': self.config.__dict__,
                'data_info': {
                    'shape': data.shape,
                    'period': f"{data.index[0]} ~ {data.index[-1]}",
                    'features': len(data.columns) - 1  # target 제외
                },
                'walk_forward_results': wf_results,
                'backtest_performance': backtest_results,
                'risk_metrics': risk_metrics,
                'regime_analysis': regime_performance,
                'significance_tests': significance_results,
                'timestamp': datetime.now().isoformat()
            }
            
            # 10. 결과 저장 및 시각화
            self.save_comprehensive_results(comprehensive_results)
            self.create_comprehensive_visualizations(comprehensive_results)
            
            self.logger.info("✅ 정교한 백테스팅 프레임워크 완료!")
            return comprehensive_results
            
        except Exception as e:
            self.logger.error(f"백테스트 실행 실패: {e}")
            raise
    
    def execute_backtesting_simulation(self, data: pd.DataFrame, 
                                     wf_results: Dict) -> Dict:
        """백테스트 시뮬레이션 실행"""
        self.logger.info("📊 백테스트 시뮬레이션 시작...")
        
        portfolio_value = [self.config.initial_capital]
        positions = [0]  # BTC 포지션
        cash = [self.config.initial_capital]
        trades = []
        
        # 워크포워드 결과에서 최적 모델 사용
        for period_result in wf_results['results']:
            # 각 기간별 거래 시뮬레이션
            period_trades, period_portfolio = self.simulate_trading_period(
                data, period_result
            )
            trades.extend(period_trades)
            portfolio_value.extend(period_portfolio)
        
        # 수익률 계산
        returns = pd.Series(portfolio_value).pct_change().dropna()
        
        return {
            'portfolio_values': portfolio_value,
            'returns': returns.tolist(),
            'positions': positions,
            'cash': cash,
            'trades': trades,
            'total_return': (portfolio_value[-1] / portfolio_value[0]) - 1,
            'annualized_return': ((portfolio_value[-1] / portfolio_value[0]) ** (365 / len(portfolio_value))) - 1,
            'volatility': returns.std() * np.sqrt(365)
        }
    
    def simulate_trading_period(self, data: pd.DataFrame, period_result: Dict) -> Tuple[List, List]:
        """거래 기간 시뮬레이션"""
        trades = []
        portfolio_values = []
        
        # 실제 거래 로직 구현
        # (간단화된 버전 - 실제로는 더 복잡한 신호 생성 및 포지션 관리 필요)
        
        return trades, portfolio_values
    
    def calculate_comprehensive_risk_metrics(self, backtest_results: Dict) -> Dict:
        """종합 리스크 지표 계산"""
        returns = np.array(backtest_results['returns'])
        portfolio_values = np.array(backtest_results['portfolio_values'])
        
        return self.risk_metrics.calculate_all_metrics(returns, portfolio_values)
    
    def perform_significance_testing(self, backtest_results: Dict) -> Dict:
        """통계적 유의성 검정"""
        returns = np.array(backtest_results['returns'])
        
        # 벤치마크 수익률 (단순 보유 전략)
        benchmark_returns = np.random.normal(0.0001, 0.02, len(returns))  # 임시 벤치마크
        
        # 부트스트랩 분석
        bootstrap_sharpe = self.significance_tester.bootstrap_analysis(
            returns, lambda x: self.risk_metrics.sharpe_ratio(x)
        )
        
        # 유의성 검정
        significance = self.significance_tester.significance_test(returns, benchmark_returns)
        
        return {
            'bootstrap_sharpe': bootstrap_sharpe,
            'strategy_vs_benchmark': significance
        }
    
    def save_comprehensive_results(self, results: Dict):
        """종합 결과 저장"""
        # JSON 형태로 저장
        output_file = os.path.join(
            self.data_path, 
            f"sophisticated_backtest_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        self.logger.info(f"결과 저장 완료: {output_file}")
    
    def create_comprehensive_visualizations(self, results: Dict):
        """종합 시각화 생성"""
        # 대시보드 생성
        dashboard_file = self.visualizer.create_comprehensive_dashboard(results)
        self.logger.info(f"대시보드 생성 완료: {dashboard_file}")
        
        # 시장 상황별 분석 차트
        if 'regime_analysis' in results:
            regime_chart = self.visualizer.create_regime_analysis_chart(
                results['regime_analysis']
            )
            self.logger.info(f"시장 분석 차트 생성 완료: {regime_chart}")

def main():
    """메인 실행 함수"""
    print("🎯 정교한 비트코인 예측 모델 백테스팅 프레임워크 v3.0")
    print("="*80)
    
    # 설정
    config = BacktestConfig(
        transaction_cost=0.001,  # 0.1% 거래 수수료
        slippage=0.0005,        # 0.05% 슬리피지
        initial_capital=10000,   # $10,000 초기 자본
        max_drawdown_limit=0.20, # 20% 최대 낙폭 제한
        bootstrap_samples=1000   # 1000회 부트스트랩
    )
    
    # 프레임워크 초기화
    framework = SophisticatedBacktestingFramework(config)
    
    # 종합 백테스트 실행
    results = framework.run_comprehensive_backtest()
    
    print("\n🏆 백테스팅 완료!")
    print(f"총 수익률: {results['backtest_performance']['total_return']:.2%}")
    print(f"연율화 수익률: {results['backtest_performance']['annualized_return']:.2%}")
    print(f"샤프 비율: {results['risk_metrics']['sharpe_ratio']:.3f}")
    print(f"최대 낙폭: {results['risk_metrics']['max_drawdown']:.2%}")

if __name__ == "__main__":
    main()