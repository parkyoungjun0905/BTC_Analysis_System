#!/usr/bin/env python3
"""
🎯 리스크 조정 성과 지표 시스템
- 종합적인 리스크 측정 (샤프, 소르티노, 칼마 비율)
- 고급 VaR 및 CVaR 계산 (다양한 방법론)
- 드로다운 분석 (최대, 평균, 지속기간)
- 테일 리스크 분석 (극값 이론 적용)
- 리스크 기여도 분석 (포트폴리오 관점)
- 시장 상황별 리스크 조정 성과 측정
- 동적 리스크 지표 (시간 변화 추적)
"""

import numpy as np
import pandas as pd
import warnings
from datetime import datetime, timedelta
import json
import os
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import logging

# 통계 및 수치 계산
from scipy import stats
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import EmpiricalCovariance, LedoitWolf

# 시각화
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# 극값 이론 (옵셔널)
try:
    from scipy.stats import genpareto, genextreme
    EXTREME_VALUE_AVAILABLE = True
except ImportError:
    EXTREME_VALUE_AVAILABLE = False

warnings.filterwarnings('ignore')

@dataclass
class RiskConfig:
    """리스크 측정 설정"""
    # 기본 설정
    risk_free_rate: float = 0.02        # 연간 무위험 수익률 (2%)
    confidence_levels: List[float] = None  # VaR 신뢰수준들
    
    # 계산 기간
    lookback_periods: List[int] = None  # 롤링 계산 기간들
    min_periods: int = 30               # 최소 계산 기간
    
    # VaR 방법론
    var_methods: List[str] = None       # VaR 계산 방법들
    
    # 극값 분석
    extreme_percentile: float = 0.05    # 극값 분석 임계값 (5%)
    tail_threshold: float = 0.95        # 테일 리스크 임계값
    
    # 드로다운 분석
    dd_recovery_threshold: float = 0.9  # 회복 임계값 (90%)
    
    def __post_init__(self):
        if self.confidence_levels is None:
            self.confidence_levels = [0.90, 0.95, 0.99]
        if self.lookback_periods is None:
            self.lookback_periods = [30, 60, 120, 252]  # 1개월, 2개월, 4개월, 1년
        if self.var_methods is None:
            self.var_methods = ['historical', 'parametric', 'monte_carlo', 'extreme_value']

class RiskAdjustedMetricsSystem:
    """리스크 조정 성과 지표 시스템"""
    
    def __init__(self, config: RiskConfig = None):
        self.config = config or RiskConfig()
        self.data_path = "/Users/parkyoungjun/Desktop/BTC_Analysis_System"
        
        # 계산 결과 저장
        self.risk_metrics_history = {}
        self.drawdown_analysis = {}
        self.var_estimates = {}
        self.tail_risk_measures = {}
        
        # 로깅 설정
        self.setup_logging()
        
    def setup_logging(self):
        """로깅 설정"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.data_path, 'risk_metrics.log'), encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def calculate_comprehensive_risk_metrics(self, returns: Union[pd.Series, np.ndarray], 
                                           portfolio_values: Union[pd.Series, np.ndarray] = None,
                                           prices: Union[pd.Series, np.ndarray] = None) -> Dict:
        """종합적인 리스크 지표 계산"""
        self.logger.info("📊 종합적인 리스크 지표 계산 시작...")
        
        # 데이터 전처리
        if isinstance(returns, np.ndarray):
            returns = pd.Series(returns)
        
        if portfolio_values is not None and isinstance(portfolio_values, np.ndarray):
            portfolio_values = pd.Series(portfolio_values)
        
        # 기본 통계량
        basic_stats = self._calculate_basic_statistics(returns)
        
        # 리스크 조정 비율들
        risk_ratios = self._calculate_risk_ratios(returns, portfolio_values)
        
        # VaR 및 CVaR (다양한 방법론)
        var_metrics = self._calculate_var_metrics(returns)
        
        # 드로다운 분석
        drawdown_metrics = self._calculate_drawdown_metrics(portfolio_values or returns.cumsum())
        
        # 테일 리스크 분석
        tail_risk = self._calculate_tail_risk_measures(returns)
        
        # 고차 모멘트 분석
        higher_moments = self._calculate_higher_moments(returns)
        
        # 동적 리스크 지표
        dynamic_risk = self._calculate_dynamic_risk_metrics(returns)
        
        comprehensive_metrics = {
            'calculation_date': datetime.now().isoformat(),
            'data_period': {
                'start': returns.index[0] if hasattr(returns, 'index') else 'N/A',
                'end': returns.index[-1] if hasattr(returns, 'index') else 'N/A',
                'observations': len(returns)
            },
            'basic_statistics': basic_stats,
            'risk_adjusted_ratios': risk_ratios,
            'var_metrics': var_metrics,
            'drawdown_analysis': drawdown_metrics,
            'tail_risk_measures': tail_risk,
            'higher_moments': higher_moments,
            'dynamic_risk_metrics': dynamic_risk
        }
        
        return comprehensive_metrics
    
    def _calculate_basic_statistics(self, returns: pd.Series) -> Dict:
        """기본 통계량 계산"""
        annual_factor = 252  # 1년 = 252 거래일
        
        return {
            'mean_return': float(returns.mean()),
            'annual_return': float(returns.mean() * annual_factor),
            'volatility': float(returns.std()),
            'annual_volatility': float(returns.std() * np.sqrt(annual_factor)),
            'skewness': float(stats.skew(returns.dropna())),
            'kurtosis': float(stats.kurtosis(returns.dropna())),
            'min_return': float(returns.min()),
            'max_return': float(returns.max()),
            'positive_returns_ratio': float((returns > 0).mean()),
            'jarque_bera_test': self._jarque_bera_test(returns)
        }
    
    def _jarque_bera_test(self, returns: pd.Series) -> Dict:
        """정규성 검정 (Jarque-Bera)"""
        clean_returns = returns.dropna()
        if len(clean_returns) < 20:
            return {'statistic': None, 'p_value': None, 'is_normal': None}
        
        jb_stat, jb_pvalue = stats.jarque_bera(clean_returns)
        return {
            'statistic': float(jb_stat),
            'p_value': float(jb_pvalue),
            'is_normal': bool(jb_pvalue > 0.05)
        }
    
    def _calculate_risk_ratios(self, returns: pd.Series, portfolio_values: pd.Series = None) -> Dict:
        """리스크 조정 비율 계산"""
        annual_factor = 252
        risk_free_daily = self.config.risk_free_rate / annual_factor
        
        # 기본 지표들
        excess_returns = returns - risk_free_daily
        mean_excess = excess_returns.mean()
        volatility = returns.std()
        
        # 샤프 비율
        sharpe_ratio = (mean_excess * annual_factor) / (volatility * np.sqrt(annual_factor)) if volatility > 0 else 0
        
        # 소르티노 비율
        downside_returns = returns[returns < risk_free_daily]
        downside_std = downside_returns.std() if len(downside_returns) > 0 else volatility
        sortino_ratio = (mean_excess * annual_factor) / (downside_std * np.sqrt(annual_factor)) if downside_std > 0 else 0
        
        # 칼마 비율
        calmar_ratio = 0
        if portfolio_values is not None:
            max_dd = self._calculate_max_drawdown(portfolio_values)['max_drawdown_pct']
            annual_return = mean_excess * annual_factor
            calmar_ratio = annual_return / abs(max_dd) if max_dd != 0 else 0
        
        # 오메가 비율
        omega_ratio = self._calculate_omega_ratio(returns, risk_free_daily)
        
        # 정보 비율 (벤치마크 대비)
        information_ratio = self._calculate_information_ratio(returns)
        
        # Treynor 비율 (시장 베타 필요시)
        treynor_ratio = self._calculate_treynor_ratio(returns)
        
        # 최대 손실 대비 수익률
        max_loss = returns.min()
        gain_to_pain_ratio = mean_excess / abs(max_loss) if max_loss < 0 else float('inf')
        
        return {
            'sharpe_ratio': float(sharpe_ratio),
            'sortino_ratio': float(sortino_ratio),
            'calmar_ratio': float(calmar_ratio),
            'omega_ratio': float(omega_ratio),
            'information_ratio': float(information_ratio),
            'treynor_ratio': float(treynor_ratio),
            'gain_to_pain_ratio': float(gain_to_pain_ratio),
            'return_to_var_ratio': self._calculate_return_to_var_ratio(returns)
        }
    
    def _calculate_omega_ratio(self, returns: pd.Series, threshold: float = 0) -> float:
        """오메가 비율 계산"""
        gains = returns[returns > threshold] - threshold
        losses = threshold - returns[returns <= threshold]
        
        total_gains = gains.sum() if len(gains) > 0 else 0
        total_losses = losses.sum() if len(losses) > 0 else 1e-8  # 0 방지
        
        return total_gains / total_losses
    
    def _calculate_information_ratio(self, returns: pd.Series) -> float:
        """정보 비율 계산 (단순 벤치마크 대비)"""
        # 벤치마크를 0으로 가정 (절대 수익률 기준)
        benchmark_return = 0
        active_return = returns.mean() - benchmark_return
        tracking_error = returns.std()
        
        return (active_return * 252) / (tracking_error * np.sqrt(252)) if tracking_error > 0 else 0
    
    def _calculate_treynor_ratio(self, returns: pd.Series) -> float:
        """트레이너 비율 계산 (베타=1 가정)"""
        risk_free_daily = self.config.risk_free_rate / 252
        excess_return = (returns.mean() - risk_free_daily) * 252
        beta = 1.0  # 시장 베타를 1로 가정
        
        return excess_return / beta
    
    def _calculate_return_to_var_ratio(self, returns: pd.Series) -> Dict:
        """수익률 대 VaR 비율"""
        annual_return = returns.mean() * 252
        var_95 = self._calculate_historical_var(returns, 0.95)
        var_99 = self._calculate_historical_var(returns, 0.99)
        
        return {
            'return_to_var_95': float(annual_return / abs(var_95)) if var_95 != 0 else 0,
            'return_to_var_99': float(annual_return / abs(var_99)) if var_99 != 0 else 0
        }
    
    def _calculate_var_metrics(self, returns: pd.Series) -> Dict:
        """VaR 및 CVaR 계산 (다양한 방법론)"""
        var_results = {}
        
        for confidence_level in self.config.confidence_levels:
            level_results = {}
            
            # 1. 역사적 VaR
            if 'historical' in self.config.var_methods:
                level_results['historical_var'] = self._calculate_historical_var(returns, confidence_level)
                level_results['historical_cvar'] = self._calculate_historical_cvar(returns, confidence_level)
            
            # 2. 파라미터적 VaR (정규분포 가정)
            if 'parametric' in self.config.var_methods:
                level_results['parametric_var'] = self._calculate_parametric_var(returns, confidence_level)
                level_results['parametric_cvar'] = self._calculate_parametric_cvar(returns, confidence_level)
            
            # 3. 몬테카를로 VaR
            if 'monte_carlo' in self.config.var_methods:
                mc_results = self._calculate_monte_carlo_var(returns, confidence_level)
                level_results.update(mc_results)
            
            # 4. 극값 이론 VaR
            if 'extreme_value' in self.config.var_methods and EXTREME_VALUE_AVAILABLE:
                ev_results = self._calculate_extreme_value_var(returns, confidence_level)
                level_results.update(ev_results)
            
            var_results[f'confidence_{int(confidence_level*100)}'] = level_results
        
        return var_results
    
    def _calculate_historical_var(self, returns: pd.Series, confidence_level: float) -> float:
        """역사적 VaR 계산"""
        clean_returns = returns.dropna()
        if len(clean_returns) == 0:
            return 0.0
        return float(np.percentile(clean_returns, (1 - confidence_level) * 100))
    
    def _calculate_historical_cvar(self, returns: pd.Series, confidence_level: float) -> float:
        """역사적 CVaR 계산"""
        var = self._calculate_historical_var(returns, confidence_level)
        clean_returns = returns.dropna()
        tail_returns = clean_returns[clean_returns <= var]
        
        return float(tail_returns.mean()) if len(tail_returns) > 0 else var
    
    def _calculate_parametric_var(self, returns: pd.Series, confidence_level: float) -> float:
        """파라미터적 VaR (정규분포 가정)"""
        mean_return = returns.mean()
        std_return = returns.std()
        z_score = stats.norm.ppf(1 - confidence_level)
        
        return float(mean_return + z_score * std_return)
    
    def _calculate_parametric_cvar(self, returns: pd.Series, confidence_level: float) -> float:
        """파라미터적 CVaR (정규분포 가정)"""
        mean_return = returns.mean()
        std_return = returns.std()
        z_score = stats.norm.ppf(1 - confidence_level)
        
        # 조건부 기댓값 계산
        phi_z = stats.norm.pdf(z_score)
        cvar = mean_return - std_return * phi_z / (1 - confidence_level)
        
        return float(cvar)
    
    def _calculate_monte_carlo_var(self, returns: pd.Series, confidence_level: float, 
                                 n_simulations: int = 10000) -> Dict:
        """몬테카를로 VaR 계산"""
        clean_returns = returns.dropna()
        
        if len(clean_returns) < 30:
            return {'monte_carlo_var': 0.0, 'monte_carlo_cvar': 0.0}
        
        # 파라미터 추정
        mean_return = clean_returns.mean()
        std_return = clean_returns.std()
        
        # 몬테카를로 시뮬레이션
        np.random.seed(42)  # 재현 가능성
        simulated_returns = np.random.normal(mean_return, std_return, n_simulations)
        
        # VaR 및 CVaR 계산
        var_mc = np.percentile(simulated_returns, (1 - confidence_level) * 100)
        tail_returns_mc = simulated_returns[simulated_returns <= var_mc]
        cvar_mc = np.mean(tail_returns_mc) if len(tail_returns_mc) > 0 else var_mc
        
        return {
            'monte_carlo_var': float(var_mc),
            'monte_carlo_cvar': float(cvar_mc)
        }
    
    def _calculate_extreme_value_var(self, returns: pd.Series, confidence_level: float) -> Dict:
        """극값 이론 VaR 계산"""
        if not EXTREME_VALUE_AVAILABLE:
            return {'extreme_value_var': 0.0, 'extreme_value_cvar': 0.0}
        
        clean_returns = returns.dropna()
        
        if len(clean_returns) < 100:  # 극값 이론에는 충분한 데이터 필요
            return {'extreme_value_var': 0.0, 'extreme_value_cvar': 0.0}
        
        try:
            # 극값 (하위 5%)만 추출
            threshold = np.percentile(clean_returns, self.config.extreme_percentile * 100)
            exceedances = clean_returns[clean_returns <= threshold] - threshold
            
            if len(exceedances) < 20:
                return {'extreme_value_var': 0.0, 'extreme_value_cvar': 0.0}
            
            # GPD (Generalized Pareto Distribution) 피팅
            shape, loc, scale = genpareto.fit(-exceedances, floc=0)  # 음수 변환 (손실 기준)
            
            # VaR 계산
            prob = (1 - confidence_level) / self.config.extreme_percentile
            var_ev = threshold - scale * ((prob ** (-shape)) - 1) / shape
            
            # CVaR 계산 (근사)
            cvar_ev = var_ev - scale / (1 - shape) * (prob ** (-shape))
            
            return {
                'extreme_value_var': float(var_ev),
                'extreme_value_cvar': float(cvar_ev),
                'gpd_shape': float(shape),
                'gpd_scale': float(scale)
            }
        except Exception:
            return {'extreme_value_var': 0.0, 'extreme_value_cvar': 0.0}
    
    def _calculate_drawdown_metrics(self, cumulative_returns: pd.Series) -> Dict:
        """드로다운 분석"""
        if isinstance(cumulative_returns, np.ndarray):
            cumulative_returns = pd.Series(cumulative_returns)
        
        # 누적 최대값 (peak) 계산
        peaks = cumulative_returns.expanding().max()
        
        # 드로다운 계산
        drawdowns = (cumulative_returns - peaks) / peaks
        
        # 최대 드로다운
        max_drawdown = drawdowns.min()
        max_dd_start = drawdowns.idxmin()
        max_dd_peak = peaks.loc[:max_dd_start].idxmax()
        
        # 회복 지점 찾기
        max_dd_recovery = None
        recovery_threshold = peaks.loc[max_dd_peak] * self.config.dd_recovery_threshold
        
        post_dd_data = cumulative_returns.loc[max_dd_start:]
        recovery_candidates = post_dd_data[post_dd_data >= recovery_threshold]
        
        if len(recovery_candidates) > 0:
            max_dd_recovery = recovery_candidates.index[0]
        
        # 드로다운 기간 계산
        dd_duration = None
        if max_dd_recovery is not None and hasattr(cumulative_returns.index, 'to_pydatetime'):
            dd_duration = (max_dd_recovery - max_dd_peak).days
        elif max_dd_recovery is not None:
            # 인덱스가 정수인 경우
            dd_duration = int(max_dd_recovery - max_dd_peak)
        
        # 드로다운 통계
        drawdown_stats = {
            'max_drawdown_pct': float(max_drawdown * 100),
            'max_drawdown_value': float(max_drawdown),
            'max_dd_start_date': max_dd_peak,
            'max_dd_end_date': max_dd_start,
            'max_dd_recovery_date': max_dd_recovery,
            'max_dd_duration_days': dd_duration,
            'avg_drawdown_pct': float(drawdowns[drawdowns < 0].mean() * 100) if (drawdowns < 0).any() else 0,
            'drawdown_frequency': float((drawdowns < -0.05).sum() / len(drawdowns)),  # 5% 이상 하락 빈도
            'time_underwater_pct': float((drawdowns < -0.01).sum() / len(drawdowns) * 100)  # 1% 이상 하락 상태 비율
        }
        
        # 모든 드로다운 구간 분석
        dd_periods = self._analyze_all_drawdown_periods(drawdowns)
        drawdown_stats['all_drawdown_periods'] = dd_periods
        
        return drawdown_stats
    
    def _calculate_max_drawdown(self, values: pd.Series) -> Dict:
        """최대 드로다운 계산 (간단 버전)"""
        peaks = values.expanding().max()
        drawdowns = (values - peaks) / peaks
        max_dd = drawdowns.min()
        
        return {'max_drawdown_pct': float(max_dd * 100)}
    
    def _analyze_all_drawdown_periods(self, drawdowns: pd.Series, threshold: float = -0.01) -> List[Dict]:
        """모든 드로다운 기간 분석"""
        dd_periods = []
        in_drawdown = False
        current_dd = {}
        
        for idx, dd_value in drawdowns.items():
            if dd_value <= threshold and not in_drawdown:
                # 드로다운 시작
                in_drawdown = True
                current_dd = {
                    'start': idx,
                    'min_value': dd_value,
                    'min_date': idx
                }
            elif dd_value <= threshold and in_drawdown:
                # 드로다운 지속
                if dd_value < current_dd['min_value']:
                    current_dd['min_value'] = dd_value
                    current_dd['min_date'] = idx
            elif dd_value > threshold and in_drawdown:
                # 드로다운 종료
                current_dd['end'] = idx
                current_dd['duration'] = idx - current_dd['start']
                current_dd['magnitude_pct'] = current_dd['min_value'] * 100
                
                dd_periods.append(current_dd)
                in_drawdown = False
        
        # 현재 진행중인 드로다운 처리
        if in_drawdown:
            current_dd['end'] = drawdowns.index[-1]
            current_dd['duration'] = drawdowns.index[-1] - current_dd['start']
            current_dd['magnitude_pct'] = current_dd['min_value'] * 100
            dd_periods.append(current_dd)
        
        return dd_periods
    
    def _calculate_tail_risk_measures(self, returns: pd.Series) -> Dict:
        """테일 리스크 측정"""
        clean_returns = returns.dropna()
        
        if len(clean_returns) < 50:
            return {'insufficient_data': True}
        
        # 테일 비율 계산
        tail_ratios = {}
        for percentile in [90, 95, 99]:
            upper_tail = np.percentile(clean_returns, percentile)
            lower_tail = np.percentile(clean_returns, 100 - percentile)
            tail_ratios[f'tail_ratio_{percentile}'] = abs(upper_tail) / abs(lower_tail) if lower_tail != 0 else float('inf')
        
        # 극값 지수 계산 (Hill estimator)
        tail_index = self._calculate_tail_index(clean_returns)
        
        # 좌측/우측 테일 분석
        left_tail = clean_returns[clean_returns <= np.percentile(clean_returns, 5)]
        right_tail = clean_returns[clean_returns >= np.percentile(clean_returns, 95)]
        
        tail_measures = {
            'tail_ratios': tail_ratios,
            'tail_index': tail_index,
            'left_tail_mean': float(left_tail.mean()) if len(left_tail) > 0 else 0,
            'right_tail_mean': float(right_tail.mean()) if len(right_tail) > 0 else 0,
            'tail_dependency': self._calculate_tail_dependency(clean_returns),
            'expected_shortfall': {
                f'es_{int(conf*100)}': float(clean_returns[clean_returns <= np.percentile(clean_returns, (1-conf)*100)].mean())
                for conf in self.config.confidence_levels
            }
        }
        
        return tail_measures
    
    def _calculate_tail_index(self, returns: pd.Series) -> float:
        """테일 지수 계산 (Hill 추정량)"""
        k = int(len(returns) * 0.05)  # 상위 5% 사용
        sorted_returns = np.sort(returns)
        extreme_values = sorted_returns[-k:]
        
        if len(extreme_values) < 2:
            return 0.0
        
        # Hill 추정량
        log_ratios = np.log(extreme_values[1:] / extreme_values[0])
        tail_index = np.mean(log_ratios)
        
        return float(tail_index)
    
    def _calculate_tail_dependency(self, returns: pd.Series) -> Dict:
        """테일 의존성 계산"""
        # 단순화된 테일 의존성 (자기 상관)
        lagged_returns = returns.shift(1).dropna()
        aligned_returns = returns[lagged_returns.index]
        
        if len(aligned_returns) < 50:
            return {'upper': 0, 'lower': 0}
        
        # 상위/하위 10% 동시 발생 확률
        upper_10 = np.percentile(returns, 90)
        lower_10 = np.percentile(returns, 10)
        
        upper_dependency = np.mean((aligned_returns > upper_10) & (lagged_returns > upper_10))
        lower_dependency = np.mean((aligned_returns < lower_10) & (lagged_returns < lower_10))
        
        return {
            'upper': float(upper_dependency),
            'lower': float(lower_dependency)
        }
    
    def _calculate_higher_moments(self, returns: pd.Series) -> Dict:
        """고차 모멘트 분석"""
        clean_returns = returns.dropna()
        
        if len(clean_returns) < 30:
            return {'insufficient_data': True}
        
        return {
            'skewness': float(stats.skew(clean_returns)),
            'kurtosis': float(stats.kurtosis(clean_returns)),
            'excess_kurtosis': float(stats.kurtosis(clean_returns, fisher=False) - 3),
            'jarque_bera_stat': float(stats.jarque_bera(clean_returns)[0]),
            'jarque_bera_pvalue': float(stats.jarque_bera(clean_returns)[1]),
            'coskewness': self._calculate_coskewness(clean_returns),
            'cokurtosis': self._calculate_cokurtosis(clean_returns)
        }
    
    def _calculate_coskewness(self, returns: pd.Series) -> float:
        """공왜도 계산 (자기 지연과의)"""
        lagged_returns = returns.shift(1).dropna()
        aligned_returns = returns[lagged_returns.index]
        
        if len(aligned_returns) < 30:
            return 0.0
        
        # 표준화
        std_current = aligned_returns.std()
        std_lagged = lagged_returns.std()
        
        if std_current == 0 or std_lagged == 0:
            return 0.0
        
        normalized_current = (aligned_returns - aligned_returns.mean()) / std_current
        normalized_lagged = (lagged_returns - lagged_returns.mean()) / std_lagged
        
        coskewness = np.mean(normalized_current**2 * normalized_lagged)
        return float(coskewness)
    
    def _calculate_cokurtosis(self, returns: pd.Series) -> float:
        """공첨도 계산"""
        lagged_returns = returns.shift(1).dropna()
        aligned_returns = returns[lagged_returns.index]
        
        if len(aligned_returns) < 30:
            return 0.0
        
        # 표준화
        std_current = aligned_returns.std()
        std_lagged = lagged_returns.std()
        
        if std_current == 0 or std_lagged == 0:
            return 0.0
        
        normalized_current = (aligned_returns - aligned_returns.mean()) / std_current
        normalized_lagged = (lagged_returns - lagged_returns.mean()) / std_lagged
        
        cokurtosis = np.mean(normalized_current**3 * normalized_lagged)
        return float(cokurtosis)
    
    def _calculate_dynamic_risk_metrics(self, returns: pd.Series) -> Dict:
        """동적 리스크 지표 계산"""
        dynamic_metrics = {}
        
        for lookback in self.config.lookback_periods:
            if len(returns) < lookback + self.config.min_periods:
                continue
            
            rolling_sharpe = []
            rolling_var = []
            rolling_volatility = []
            
            for i in range(lookback, len(returns)):
                window_returns = returns.iloc[i-lookback:i]
                
                # 롤링 샤프 비율
                excess_returns = window_returns - self.config.risk_free_rate / 252
                sharpe = (excess_returns.mean() * 252) / (window_returns.std() * np.sqrt(252)) if window_returns.std() > 0 else 0
                rolling_sharpe.append(sharpe)
                
                # 롤링 VaR
                var_95 = self._calculate_historical_var(window_returns, 0.95)
                rolling_var.append(var_95)
                
                # 롤링 변동성
                rolling_volatility.append(window_returns.std() * np.sqrt(252))
            
            dynamic_metrics[f'lookback_{lookback}'] = {
                'rolling_sharpe': rolling_sharpe,
                'rolling_var_95': rolling_var,
                'rolling_volatility': rolling_volatility,
                'sharpe_stability': np.std(rolling_sharpe) if len(rolling_sharpe) > 1 else 0,
                'var_stability': np.std(rolling_var) if len(rolling_var) > 1 else 0
            }
        
        return dynamic_metrics
    
    def create_risk_dashboard(self, risk_metrics: Dict) -> str:
        """리스크 대시보드 생성"""
        self.logger.info("📊 리스크 분석 대시보드 생성 중...")
        
        fig = make_subplots(
            rows=4, cols=3,
            subplot_titles=(
                "리스크-수익 스캐터", "VaR 비교 (다양한 방법론)", "드로다운 분석",
                "테일 리스크 분포", "롤링 샤프 비율", "고차 모멘트 분석",
                "리스크 조정 비율", "CVaR vs VaR", "변동성 클러스터링",
                "극값 분석", "테일 의존성", "리스크 기여도"
            ),
            specs=[[{"type": "scatter"}, {"type": "bar"}, {"type": "scatter"}],
                   [{"type": "histogram"}, {"type": "scatter"}, {"type": "radar"}],
                   [{"type": "bar"}, {"type": "scatter"}, {"type": "heatmap"}],
                   [{"type": "scatter"}, {"type": "bar"}, {"type": "pie"}]],
            vertical_spacing=0.08,
            horizontal_spacing=0.08
        )
        
        # 1. 리스크-수익 스캐터 플롯
        if 'basic_statistics' in risk_metrics:
            stats = risk_metrics['basic_statistics']
            fig.add_trace(
                go.Scatter(
                    x=[stats['annual_volatility']],
                    y=[stats['annual_return']],
                    mode='markers',
                    marker=dict(size=15, color='blue'),
                    name='포트폴리오',
                    text=[f"샤프: {risk_metrics.get('risk_adjusted_ratios', {}).get('sharpe_ratio', 0):.3f}"],
                    textposition="top center"
                ),
                row=1, col=1
            )
        
        # 2. VaR 비교
        if 'var_metrics' in risk_metrics:
            var_data = risk_metrics['var_metrics'].get('confidence_95', {})
            methods = []
            values = []
            
            for method, value in var_data.items():
                if 'var' in method and not 'cvar' in method:
                    methods.append(method.replace('_var', '').title())
                    values.append(abs(value) * 100)  # 백분율로 변환
            
            if methods:
                fig.add_trace(
                    go.Bar(x=methods, y=values, name='VaR 95%', marker_color='red'),
                    row=1, col=2
                )
        
        # 3. 드로다운 시계열 (간단 버전)
        if 'drawdown_analysis' in risk_metrics:
            dd_stats = risk_metrics['drawdown_analysis']
            fig.add_trace(
                go.Scatter(
                    y=[dd_stats.get('max_drawdown_pct', 0)],
                    mode='markers',
                    marker=dict(size=20, color='red'),
                    name=f"최대 드로다운: {dd_stats.get('max_drawdown_pct', 0):.2f}%"
                ),
                row=1, col=3
            )
        
        # 추가 차트들은 데이터 가용성에 따라 구현...
        
        fig.update_layout(
            title="🎯 종합 리스크 분석 대시보드",
            height=1600,
            showlegend=True,
            template='plotly_dark'
        )
        
        # 저장
        dashboard_path = os.path.join(self.data_path, 'risk_analysis_dashboard.html')
        fig.write_html(dashboard_path, include_plotlyjs=True)
        
        return dashboard_path
    
    def save_risk_analysis_report(self, risk_metrics: Dict):
        """리스크 분석 보고서 저장"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # JSON 저장
        json_path = os.path.join(self.data_path, f'risk_analysis_report_{timestamp}.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(risk_metrics, f, indent=2, ensure_ascii=False, default=str)
        
        self.logger.info(f"리스크 분석 보고서 저장: {json_path}")
        
        # 요약 보고서 생성
        self._generate_summary_report(risk_metrics, timestamp)
    
    def _generate_summary_report(self, risk_metrics: Dict, timestamp: str):
        """요약 보고서 생성"""
        summary_lines = [
            "🎯 비트코인 투자 리스크 분석 요약 보고서",
            "=" * 60,
            f"분석 일시: {datetime.now().strftime('%Y년 %m월 %d일 %H시 %M분')}",
            ""
        ]
        
        # 기본 통계
        if 'basic_statistics' in risk_metrics:
            stats = risk_metrics['basic_statistics']
            summary_lines.extend([
                "📊 기본 통계",
                f"  • 연간 수익률: {stats.get('annual_return', 0)*100:.2f}%",
                f"  • 연간 변동성: {stats.get('annual_volatility', 0)*100:.2f}%",
                f"  • 왜도: {stats.get('skewness', 0):.3f}",
                f"  • 첨도: {stats.get('kurtosis', 0):.3f}",
                ""
            ])
        
        # 리스크 조정 비율
        if 'risk_adjusted_ratios' in risk_metrics:
            ratios = risk_metrics['risk_adjusted_ratios']
            summary_lines.extend([
                "⚖️ 리스크 조정 성과",
                f"  • 샤프 비율: {ratios.get('sharpe_ratio', 0):.3f}",
                f"  • 소르티노 비율: {ratios.get('sortino_ratio', 0):.3f}",
                f"  • 칼마 비율: {ratios.get('calmar_ratio', 0):.3f}",
                f"  • 오메가 비율: {ratios.get('omega_ratio', 0):.3f}",
                ""
            ])
        
        # VaR 정보
        if 'var_metrics' in risk_metrics:
            var_95 = risk_metrics['var_metrics'].get('confidence_95', {})
            summary_lines.extend([
                "🚨 Value at Risk (95% 신뢰수준)",
                f"  • 역사적 VaR: {var_95.get('historical_var', 0)*100:.2f}%",
                f"  • 파라미터적 VaR: {var_95.get('parametric_var', 0)*100:.2f}%",
                f"  • 몬테카를로 VaR: {var_95.get('monte_carlo_var', 0)*100:.2f}%",
                ""
            ])
        
        # 드로다운 분석
        if 'drawdown_analysis' in risk_metrics:
            dd = risk_metrics['drawdown_analysis']
            summary_lines.extend([
                "📉 드로다운 분석",
                f"  • 최대 드로다운: {dd.get('max_drawdown_pct', 0):.2f}%",
                f"  • 평균 드로다운: {dd.get('avg_drawdown_pct', 0):.2f}%",
                f"  • 수중 시간 비율: {dd.get('time_underwater_pct', 0):.1f}%",
                ""
            ])
        
        # 파일 저장
        summary_path = os.path.join(self.data_path, f'risk_summary_report_{timestamp}.txt')
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(summary_lines))
        
        self.logger.info(f"요약 보고서 저장: {summary_path}")

def main():
    """메인 실행 함수"""
    print("🎯 리스크 조정 성과 지표 시스템")
    print("=" * 50)
    
    # 설정
    config = RiskConfig(
        risk_free_rate=0.02,
        confidence_levels=[0.90, 0.95, 0.99],
        var_methods=['historical', 'parametric', 'monte_carlo']
    )
    
    # 시스템 초기화
    risk_system = RiskAdjustedMetricsSystem(config)
    
    # 테스트 데이터 생성
    np.random.seed(42)
    n_days = 1000
    
    # 실제 비트코인과 유사한 수익률 시뮬레이션
    returns = np.random.normal(0.001, 0.03, n_days)  # 평균 0.1%, 변동성 3%
    returns = pd.Series(returns, index=pd.date_range('2021-01-01', periods=n_days))
    
    # 포트폴리오 가치 계산
    portfolio_values = (1 + returns).cumprod() * 10000
    
    # 종합 리스크 분석
    risk_metrics = risk_system.calculate_comprehensive_risk_metrics(
        returns, portfolio_values
    )
    
    print("\n📊 리스크 분석 완료!")
    print(f"샤프 비율: {risk_metrics['risk_adjusted_ratios']['sharpe_ratio']:.3f}")
    print(f"최대 드로다운: {risk_metrics['drawdown_analysis']['max_drawdown_pct']:.2f}%")
    print(f"VaR (95%): {risk_metrics['var_metrics']['confidence_95']['historical_var']*100:.2f}%")
    
    # 대시보드 생성
    dashboard_path = risk_system.create_risk_dashboard(risk_metrics)
    print(f"대시보드 생성: {dashboard_path}")
    
    # 보고서 저장
    risk_system.save_risk_analysis_report(risk_metrics)

if __name__ == "__main__":
    main()