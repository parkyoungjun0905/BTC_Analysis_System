#!/usr/bin/env python3
"""
🎯 통계적 유의성 검정 시스템
- 부트스트랩 신뢰구간 (Bootstrap Confidence Intervals)
- 가설검정 (t-test, Wilcoxon, Kolmogorov-Smirnov)
- 다중비교 보정 (Bonferroni, Holm, FDR)
- 베이지안 통계 (베이지안 A/B 테스트)
- 시계열 특화 검정 (자기상관, 정상성)
- 효과 크기 분석 (Cohen's d, Hedges' g)
- 검정력 분석 (Power Analysis)
- 강건성 검정 (Robustness Testing)
"""

import numpy as np
import pandas as pd
import warnings
from datetime import datetime
import json
import os
from typing import Dict, List, Tuple, Optional, Union, Callable
from dataclasses import dataclass, field
import logging
from enum import Enum

# 통계 라이브러리
from scipy import stats
from scipy.stats import normaltest, shapiro, kstest, jarque_bera
from scipy.stats import ttest_ind, ttest_rel, wilcoxon, ranksums
from scipy.stats import pearsonr, spearmanr, kendalltau
from scipy.stats import chi2_contingency, fisher_exact
import statsmodels.api as sm
from statsmodels.stats.power import TTestPower, ttest_power
from statsmodels.stats.multitest import multipletests
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import acorr_ljungbox

# 베이지안 통계 (옵셔널)
try:
    import pymc3 as pm
    import arviz as az
    BAYESIAN_AVAILABLE = True
except ImportError:
    try:
        import pymc as pm
        import arviz as az
        BAYESIAN_AVAILABLE = True
    except ImportError:
        BAYESIAN_AVAILABLE = False
        print("⚠️ PyMC 미설치: 베이지안 분석 불가")

# 시각화
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

warnings.filterwarnings('ignore')

class TestType(Enum):
    """검정 유형"""
    PARAMETRIC = "parametric"
    NONPARAMETRIC = "nonparametric"
    BAYESIAN = "bayesian"
    BOOTSTRAP = "bootstrap"

@dataclass
class SignificanceConfig:
    """통계적 유의성 검정 설정"""
    # 기본 설정
    alpha: float = 0.05              # 유의수준
    confidence_level: float = 0.95   # 신뢰수준
    
    # 부트스트랩 설정
    bootstrap_samples: int = 10000   # 부트스트랩 샘플 수
    bootstrap_method: str = 'percentile'  # 'percentile', 'bias_corrected', 'accelerated'
    
    # 다중비교 보정
    multiple_comparison_method: str = 'holm'  # 'bonferroni', 'holm', 'fdr_bh'
    
    # 효과 크기 임계값
    small_effect_size: float = 0.2
    medium_effect_size: float = 0.5
    large_effect_size: float = 0.8
    
    # 검정력 분석
    desired_power: float = 0.8       # 원하는 검정력
    effect_sizes: List[float] = field(default_factory=lambda: [0.1, 0.2, 0.5, 0.8])
    
    # 시계열 검정
    max_lags: int = 10               # 최대 지연 차수
    trend_methods: List[str] = field(default_factory=lambda: ['c', 'ct', 'ctt'])  # ADF 검정 방법
    
    # 베이지안 설정
    mcmc_samples: int = 5000         # MCMC 샘플 수
    bayesian_chains: int = 4         # 베이지안 체인 수
    
    # 강건성 검정
    contamination_levels: List[float] = field(default_factory=lambda: [0.05, 0.1, 0.15])
    outlier_methods: List[str] = field(default_factory=lambda: ['iqr', 'zscore', 'isolation'])

class StatisticalSignificanceTester:
    """통계적 유의성 검정 시스템"""
    
    def __init__(self, config: SignificanceConfig = None):
        self.config = config or SignificanceConfig()
        self.data_path = "/Users/parkyoungjun/Desktop/BTC_Analysis_System"
        
        # 검정 결과 저장
        self.test_results = {}
        self.effect_sizes = {}
        self.power_analysis_results = {}
        
        # 로깅 설정
        self.setup_logging()
        
    def setup_logging(self):
        """로깅 설정"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.data_path, 'statistical_tests.log'), encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def comprehensive_significance_testing(self, strategy_returns: pd.Series, 
                                         benchmark_returns: pd.Series = None,
                                         additional_strategies: Dict[str, pd.Series] = None) -> Dict:
        """종합적인 통계적 유의성 검정"""
        self.logger.info("🧪 종합적인 통계적 유의성 검정 시작...")
        
        try:
            # 1. 기본 가정 검정
            assumption_tests = self.test_statistical_assumptions(strategy_returns, benchmark_returns)
            
            # 2. 전략 vs 벤치마크 비교 검정
            comparison_tests = {}
            if benchmark_returns is not None:
                comparison_tests = self.compare_strategies(strategy_returns, benchmark_returns)
            
            # 3. 다중 전략 비교 (있는 경우)
            multiple_comparison_tests = {}
            if additional_strategies:
                multiple_comparison_tests = self.multiple_strategy_comparison(
                    {'main_strategy': strategy_returns, **additional_strategies}
                )
            
            # 4. 부트스트랩 신뢰구간
            bootstrap_results = self.bootstrap_confidence_intervals(strategy_returns)
            
            # 5. 베이지안 분석 (가능한 경우)
            bayesian_results = {}
            if BAYESIAN_AVAILABLE and benchmark_returns is not None:
                bayesian_results = self.bayesian_comparison(strategy_returns, benchmark_returns)
            
            # 6. 시계열 특화 검정
            time_series_tests = self.time_series_specific_tests(strategy_returns)
            
            # 7. 효과 크기 분석
            effect_size_analysis = self.analyze_effect_sizes(strategy_returns, benchmark_returns)
            
            # 8. 검정력 분석
            power_analysis = self.conduct_power_analysis(strategy_returns, benchmark_returns)
            
            # 9. 강건성 검정
            robustness_tests = self.conduct_robustness_tests(strategy_returns, benchmark_returns)
            
            # 10. 종합 결론
            final_conclusions = self.generate_statistical_conclusions(
                assumption_tests, comparison_tests, multiple_comparison_tests,
                bootstrap_results, bayesian_results, time_series_tests,
                effect_size_analysis, power_analysis, robustness_tests
            )
            
            comprehensive_results = {
                'test_timestamp': datetime.now().isoformat(),
                'configuration': self.config.__dict__,
                'data_summary': {
                    'strategy_observations': len(strategy_returns),
                    'benchmark_observations': len(benchmark_returns) if benchmark_returns is not None else 0,
                    'additional_strategies': len(additional_strategies) if additional_strategies else 0,
                    'time_period': {
                        'start': strategy_returns.index[0] if hasattr(strategy_returns, 'index') else 'N/A',
                        'end': strategy_returns.index[-1] if hasattr(strategy_returns, 'index') else 'N/A'
                    }
                },
                'assumption_tests': assumption_tests,
                'strategy_comparison': comparison_tests,
                'multiple_comparisons': multiple_comparison_tests,
                'bootstrap_analysis': bootstrap_results,
                'bayesian_analysis': bayesian_results,
                'time_series_tests': time_series_tests,
                'effect_size_analysis': effect_size_analysis,
                'power_analysis': power_analysis,
                'robustness_tests': robustness_tests,
                'statistical_conclusions': final_conclusions
            }
            
            # 결과 저장
            self.save_test_results(comprehensive_results)
            
            # 시각화
            self.create_significance_testing_dashboard(comprehensive_results)
            
            return comprehensive_results
            
        except Exception as e:
            self.logger.error(f"통계적 유의성 검정 실패: {e}")
            raise
    
    def test_statistical_assumptions(self, data1: pd.Series, data2: pd.Series = None) -> Dict:
        """통계적 가정 검정"""
        self.logger.info("📊 통계적 가정 검정 중...")
        
        assumptions = {}
        
        # 1. 정규성 검정
        assumptions['normality_tests'] = self._test_normality(data1, data2)
        
        # 2. 등분산성 검정
        if data2 is not None:
            assumptions['equal_variance_tests'] = self._test_equal_variance(data1, data2)
        
        # 3. 독립성 검정
        assumptions['independence_tests'] = self._test_independence(data1, data2)
        
        # 4. 이상치 검정
        assumptions['outlier_tests'] = self._detect_outliers(data1, data2)
        
        # 5. 안정성 검정 (시계열)
        assumptions['stationarity_tests'] = self._test_stationarity(data1, data2)
        
        return assumptions
    
    def _test_normality(self, data1: pd.Series, data2: pd.Series = None) -> Dict:
        """정규성 검정"""
        results = {}
        
        # 데이터1 정규성 검정
        data1_clean = data1.dropna()
        if len(data1_clean) > 3:
            # Shapiro-Wilk 검정 (샘플 크기 < 5000)
            if len(data1_clean) < 5000:
                shapiro_stat, shapiro_p = shapiro(data1_clean)
                results['data1_shapiro'] = {
                    'statistic': float(shapiro_stat),
                    'p_value': float(shapiro_p),
                    'is_normal': shapiro_p > self.config.alpha
                }
            
            # Jarque-Bera 검정
            jb_stat, jb_p = jarque_bera(data1_clean)
            results['data1_jarque_bera'] = {
                'statistic': float(jb_stat),
                'p_value': float(jb_p),
                'is_normal': jb_p > self.config.alpha
            }
            
            # D'Agostino and Pearson's 검정
            try:
                da_stat, da_p = normaltest(data1_clean)
                results['data1_dagostino'] = {
                    'statistic': float(da_stat),
                    'p_value': float(da_p),
                    'is_normal': da_p > self.config.alpha
                }
            except:
                pass
        
        # 데이터2 정규성 검정
        if data2 is not None:
            data2_clean = data2.dropna()
            if len(data2_clean) > 3:
                if len(data2_clean) < 5000:
                    shapiro_stat, shapiro_p = shapiro(data2_clean)
                    results['data2_shapiro'] = {
                        'statistic': float(shapiro_stat),
                        'p_value': float(shapiro_p),
                        'is_normal': shapiro_p > self.config.alpha
                    }
                
                jb_stat, jb_p = jarque_bera(data2_clean)
                results['data2_jarque_bera'] = {
                    'statistic': float(jb_stat),
                    'p_value': float(jb_p),
                    'is_normal': jb_p > self.config.alpha
                }
        
        return results
    
    def _test_equal_variance(self, data1: pd.Series, data2: pd.Series) -> Dict:
        """등분산성 검정"""
        results = {}
        
        data1_clean = data1.dropna()
        data2_clean = data2.dropna()
        
        if len(data1_clean) > 1 and len(data2_clean) > 1:
            # Levene 검정
            try:
                levene_stat, levene_p = stats.levene(data1_clean, data2_clean)
                results['levene'] = {
                    'statistic': float(levene_stat),
                    'p_value': float(levene_p),
                    'equal_variance': levene_p > self.config.alpha
                }
            except:
                pass
            
            # Bartlett 검정
            try:
                bartlett_stat, bartlett_p = stats.bartlett(data1_clean, data2_clean)
                results['bartlett'] = {
                    'statistic': float(bartlett_stat),
                    'p_value': float(bartlett_p),
                    'equal_variance': bartlett_p > self.config.alpha
                }
            except:
                pass
            
            # F-검정 (분산비 검정)
            try:
                f_stat = np.var(data1_clean, ddof=1) / np.var(data2_clean, ddof=1)
                f_p = 2 * min(stats.f.cdf(f_stat, len(data1_clean)-1, len(data2_clean)-1),
                             1 - stats.f.cdf(f_stat, len(data1_clean)-1, len(data2_clean)-1))
                results['f_test'] = {
                    'statistic': float(f_stat),
                    'p_value': float(f_p),
                    'equal_variance': f_p > self.config.alpha
                }
            except:
                pass
        
        return results
    
    def _test_independence(self, data1: pd.Series, data2: pd.Series = None) -> Dict:
        """독립성 검정"""
        results = {}
        
        # 자기상관 검정 (Ljung-Box)
        data1_clean = data1.dropna()
        if len(data1_clean) > 10:
            try:
                ljung_box = acorr_ljungbox(data1_clean, lags=min(self.config.max_lags, len(data1_clean)//4))
                results['data1_ljung_box'] = {
                    'statistics': ljung_box['lb_stat'].to_dict(),
                    'p_values': ljung_box['lb_pvalue'].to_dict(),
                    'is_independent': all(ljung_box['lb_pvalue'] > self.config.alpha)
                }
            except:
                pass
        
        # 두 시리즈 간 독립성 (상관관계)
        if data2 is not None:
            data2_clean = data2.dropna()
            common_index = data1_clean.index.intersection(data2_clean.index)
            
            if len(common_index) > 3:
                aligned_data1 = data1_clean[common_index]
                aligned_data2 = data2_clean[common_index]
                
                # Pearson 상관계수
                try:
                    pearson_corr, pearson_p = pearsonr(aligned_data1, aligned_data2)
                    results['pearson_correlation'] = {
                        'correlation': float(pearson_corr),
                        'p_value': float(pearson_p),
                        'is_independent': pearson_p > self.config.alpha
                    }
                except:
                    pass
                
                # Spearman 순위 상관계수
                try:
                    spearman_corr, spearman_p = spearmanr(aligned_data1, aligned_data2)
                    results['spearman_correlation'] = {
                        'correlation': float(spearman_corr),
                        'p_value': float(spearman_p),
                        'is_independent': spearman_p > self.config.alpha
                    }
                except:
                    pass
        
        return results
    
    def _detect_outliers(self, data1: pd.Series, data2: pd.Series = None) -> Dict:
        """이상치 탐지"""
        results = {}
        
        for name, data in [('data1', data1), ('data2', data2)]:
            if data is None:
                continue
                
            data_clean = data.dropna()
            if len(data_clean) == 0:
                continue
            
            outlier_results = {}
            
            # IQR 방법
            Q1 = data_clean.quantile(0.25)
            Q3 = data_clean.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            iqr_outliers = data_clean[(data_clean < lower_bound) | (data_clean > upper_bound)]
            outlier_results['iqr_method'] = {
                'outlier_count': len(iqr_outliers),
                'outlier_percentage': len(iqr_outliers) / len(data_clean) * 100,
                'lower_bound': float(lower_bound),
                'upper_bound': float(upper_bound)
            }
            
            # Z-score 방법
            z_scores = np.abs(stats.zscore(data_clean))
            z_outliers = data_clean[z_scores > 3]
            outlier_results['zscore_method'] = {
                'outlier_count': len(z_outliers),
                'outlier_percentage': len(z_outliers) / len(data_clean) * 100,
                'threshold': 3.0
            }
            
            # Modified Z-score 방법
            median = data_clean.median()
            mad = np.median(np.abs(data_clean - median))
            modified_z_scores = 0.6745 * (data_clean - median) / mad
            modified_z_outliers = data_clean[np.abs(modified_z_scores) > 3.5]
            outlier_results['modified_zscore_method'] = {
                'outlier_count': len(modified_z_outliers),
                'outlier_percentage': len(modified_z_outliers) / len(data_clean) * 100,
                'threshold': 3.5
            }
            
            results[name] = outlier_results
        
        return results
    
    def _test_stationarity(self, data1: pd.Series, data2: pd.Series = None) -> Dict:
        """안정성 검정"""
        results = {}
        
        for name, data in [('data1', data1), ('data2', data2)]:
            if data is None:
                continue
                
            data_clean = data.dropna()
            if len(data_clean) < 10:
                continue
            
            stationarity_results = {}
            
            # Augmented Dickey-Fuller 검정
            for trend in self.config.trend_methods:
                try:
                    adf_result = adfuller(data_clean, regression=trend)
                    stationarity_results[f'adf_{trend}'] = {
                        'statistic': float(adf_result[0]),
                        'p_value': float(adf_result[1]),
                        'critical_values': {k: float(v) for k, v in adf_result[4].items()},
                        'is_stationary': adf_result[1] < self.config.alpha
                    }
                except:
                    continue
            
            # KPSS 검정
            try:
                kpss_result = kpss(data_clean)
                stationarity_results['kpss'] = {
                    'statistic': float(kpss_result[0]),
                    'p_value': float(kpss_result[1]),
                    'critical_values': {k: float(v) for k, v in kpss_result[3].items()},
                    'is_stationary': kpss_result[1] > self.config.alpha
                }
            except:
                pass
            
            results[name] = stationarity_results
        
        return results
    
    def compare_strategies(self, strategy_returns: pd.Series, 
                          benchmark_returns: pd.Series) -> Dict:
        """전략 vs 벤치마크 비교 검정"""
        self.logger.info("⚖️ 전략 vs 벤치마크 비교 검정 중...")
        
        # 데이터 정렬
        common_index = strategy_returns.index.intersection(benchmark_returns.index)
        if len(common_index) == 0:
            return {'error': '공통 시점 없음'}
        
        strategy_aligned = strategy_returns[common_index]
        benchmark_aligned = benchmark_returns[common_index]
        
        comparison_results = {}
        
        # 1. 독립 t-검정
        try:
            t_stat, t_p = ttest_ind(strategy_aligned, benchmark_aligned)
            comparison_results['independent_ttest'] = {
                'statistic': float(t_stat),
                'p_value': float(t_p),
                'significant': t_p < self.config.alpha,
                'strategy_better': t_stat > 0
            }
        except:
            pass
        
        # 2. 대응 t-검정 (쌍체 비교)
        try:
            paired_t_stat, paired_t_p = ttest_rel(strategy_aligned, benchmark_aligned)
            comparison_results['paired_ttest'] = {
                'statistic': float(paired_t_stat),
                'p_value': float(paired_t_p),
                'significant': paired_t_p < self.config.alpha,
                'strategy_better': paired_t_stat > 0
            }
        except:
            pass
        
        # 3. Wilcoxon 순위합 검정 (비모수)
        try:
            wilcoxon_stat, wilcoxon_p = ranksums(strategy_aligned, benchmark_aligned)
            comparison_results['wilcoxon_ranksum'] = {
                'statistic': float(wilcoxon_stat),
                'p_value': float(wilcoxon_p),
                'significant': wilcoxon_p < self.config.alpha,
                'strategy_better': wilcoxon_stat > 0
            }
        except:
            pass
        
        # 4. Wilcoxon 부호순위 검정 (쌍체 비교)
        try:
            differences = strategy_aligned - benchmark_aligned
            wilcoxon_signed_stat, wilcoxon_signed_p = wilcoxon(differences)
            comparison_results['wilcoxon_signed_rank'] = {
                'statistic': float(wilcoxon_signed_stat),
                'p_value': float(wilcoxon_signed_p),
                'significant': wilcoxon_signed_p < self.config.alpha,
                'strategy_better': np.median(differences) > 0
            }
        except:
            pass
        
        # 5. Kolmogorov-Smirnov 검정
        try:
            ks_stat, ks_p = stats.ks_2samp(strategy_aligned, benchmark_aligned)
            comparison_results['kolmogorov_smirnov'] = {
                'statistic': float(ks_stat),
                'p_value': float(ks_p),
                'significant': ks_p < self.config.alpha,
                'distributions_different': ks_p < self.config.alpha
            }
        except:
            pass
        
        return comparison_results
    
    def multiple_strategy_comparison(self, strategies: Dict[str, pd.Series]) -> Dict:
        """다중 전략 비교"""
        self.logger.info("🔍 다중 전략 비교 검정 중...")
        
        strategy_names = list(strategies.keys())
        n_strategies = len(strategy_names)
        
        if n_strategies < 2:
            return {'error': '최소 2개 전략 필요'}
        
        # 데이터 정렬
        common_index = strategies[strategy_names[0]].index
        for name in strategy_names[1:]:
            common_index = common_index.intersection(strategies[name].index)
        
        if len(common_index) == 0:
            return {'error': '공통 시점 없음'}
        
        aligned_strategies = {name: strategies[name][common_index] for name in strategy_names}
        
        multiple_results = {}
        
        # 1. 일원분산분석 (ANOVA)
        try:
            strategy_arrays = [aligned_strategies[name].values for name in strategy_names]
            f_stat, f_p = stats.f_oneway(*strategy_arrays)
            multiple_results['anova'] = {
                'f_statistic': float(f_stat),
                'p_value': float(f_p),
                'significant': f_p < self.config.alpha,
                'strategies_different': f_p < self.config.alpha
            }
        except:
            pass
        
        # 2. Kruskal-Wallis 검정 (비모수)
        try:
            kw_stat, kw_p = stats.kruskal(*strategy_arrays)
            multiple_results['kruskal_wallis'] = {
                'statistic': float(kw_stat),
                'p_value': float(kw_p),
                'significant': kw_p < self.config.alpha,
                'strategies_different': kw_p < self.config.alpha
            }
        except:
            pass
        
        # 3. 쌍별 비교 (모든 전략 쌍)
        pairwise_results = {}
        p_values = []
        
        for i in range(n_strategies):
            for j in range(i+1, n_strategies):
                name1, name2 = strategy_names[i], strategy_names[j]
                
                # t-검정
                try:
                    t_stat, t_p = ttest_ind(aligned_strategies[name1], aligned_strategies[name2])
                    pairwise_results[f'{name1}_vs_{name2}'] = {
                        'ttest_statistic': float(t_stat),
                        'ttest_p_value': float(t_p),
                        'significant': t_p < self.config.alpha
                    }
                    p_values.append(t_p)
                except:
                    pass
        
        # 4. 다중비교 보정
        if p_values:
            rejected, p_adjusted, alpha_sidak, alpha_bonf = multipletests(
                p_values, alpha=self.config.alpha, method=self.config.multiple_comparison_method
            )
            
            multiple_results['multiple_comparison_correction'] = {
                'method': self.config.multiple_comparison_method,
                'original_alpha': self.config.alpha,
                'adjusted_alpha_bonferroni': float(alpha_bonf),
                'adjusted_alpha_sidak': float(alpha_sidak),
                'rejected_hypotheses': rejected.tolist(),
                'adjusted_p_values': p_adjusted.tolist(),
                'significant_comparisons': int(np.sum(rejected))
            }
        
        multiple_results['pairwise_comparisons'] = pairwise_results
        
        return multiple_results
    
    def bootstrap_confidence_intervals(self, data: pd.Series, 
                                     statistic_functions: Dict[str, Callable] = None) -> Dict:
        """부트스트랩 신뢰구간"""
        self.logger.info("🔄 부트스트랩 신뢰구간 계산 중...")
        
        if statistic_functions is None:
            statistic_functions = {
                'mean': np.mean,
                'median': np.median,
                'std': np.std,
                'sharpe_ratio': lambda x: np.mean(x) / np.std(x) * np.sqrt(252) if np.std(x) > 0 else 0,
                'skewness': lambda x: stats.skew(x),
                'kurtosis': lambda x: stats.kurtosis(x)
            }
        
        data_clean = data.dropna()
        n_samples = len(data_clean)
        
        if n_samples < 10:
            return {'error': '부트스트랩에 충분하지 않은 데이터'}
        
        bootstrap_results = {}
        
        for stat_name, stat_func in statistic_functions.items():
            try:
                # 원본 통계량
                original_stat = stat_func(data_clean)
                
                # 부트스트랩 샘플링
                bootstrap_stats = []
                np.random.seed(42)  # 재현 가능성
                
                for _ in range(self.config.bootstrap_samples):
                    bootstrap_sample = np.random.choice(data_clean, size=n_samples, replace=True)
                    bootstrap_stat = stat_func(bootstrap_sample)
                    bootstrap_stats.append(bootstrap_stat)
                
                bootstrap_stats = np.array(bootstrap_stats)
                bootstrap_stats = bootstrap_stats[~np.isnan(bootstrap_stats)]  # NaN 제거
                
                if len(bootstrap_stats) == 0:
                    continue
                
                # 신뢰구간 계산
                alpha = 1 - self.config.confidence_level
                lower_percentile = (alpha / 2) * 100
                upper_percentile = (1 - alpha / 2) * 100
                
                ci_lower = np.percentile(bootstrap_stats, lower_percentile)
                ci_upper = np.percentile(bootstrap_stats, upper_percentile)
                
                # 편향 보정된 신뢰구간 (BCa)
                bias_correction = self._calculate_bias_correction(original_stat, bootstrap_stats)
                acceleration = self._calculate_acceleration(data_clean, stat_func, original_stat)
                
                bca_lower, bca_upper = self._calculate_bca_interval(
                    bootstrap_stats, bias_correction, acceleration, alpha
                )
                
                bootstrap_results[stat_name] = {
                    'original_statistic': float(original_stat),
                    'bootstrap_mean': float(np.mean(bootstrap_stats)),
                    'bootstrap_std': float(np.std(bootstrap_stats)),
                    'confidence_interval_percentile': {
                        'lower': float(ci_lower),
                        'upper': float(ci_upper)
                    },
                    'confidence_interval_bca': {
                        'lower': float(bca_lower),
                        'upper': float(bca_upper)
                    },
                    'bias': float(np.mean(bootstrap_stats) - original_stat),
                    'bootstrap_distribution': bootstrap_stats.tolist()[:1000]  # 처음 1000개만 저장
                }
            except Exception as e:
                self.logger.warning(f"부트스트랩 실패 ({stat_name}): {e}")
                continue
        
        return bootstrap_results
    
    def _calculate_bias_correction(self, original_stat: float, bootstrap_stats: np.ndarray) -> float:
        """편향 보정 계산"""
        proportion = np.mean(bootstrap_stats < original_stat)
        if proportion == 0:
            proportion = 1e-7
        elif proportion == 1:
            proportion = 1 - 1e-7
        
        bias_correction = stats.norm.ppf(proportion)
        return bias_correction
    
    def _calculate_acceleration(self, data: np.ndarray, stat_func: Callable, original_stat: float) -> float:
        """가속도 계산 (Jackknife)"""
        n = len(data)
        jackknife_stats = []
        
        for i in range(n):
            jackknife_sample = np.concatenate([data[:i], data[i+1:]])
            jackknife_stat = stat_func(jackknife_sample)
            jackknife_stats.append(jackknife_stat)
        
        jackknife_stats = np.array(jackknife_stats)
        jackknife_mean = np.mean(jackknife_stats)
        
        numerator = np.sum((jackknife_mean - jackknife_stats) ** 3)
        denominator = 6 * (np.sum((jackknife_mean - jackknife_stats) ** 2)) ** 1.5
        
        if denominator == 0:
            return 0
        
        acceleration = numerator / denominator
        return acceleration
    
    def _calculate_bca_interval(self, bootstrap_stats: np.ndarray, bias_correction: float, 
                              acceleration: float, alpha: float) -> Tuple[float, float]:
        """편향 보정 가속도 신뢰구간 계산"""
        z_alpha_2 = stats.norm.ppf(alpha / 2)
        z_1_alpha_2 = stats.norm.ppf(1 - alpha / 2)
        
        # 하한
        numerator_lower = bias_correction + z_alpha_2
        denominator_lower = 1 - acceleration * (bias_correction + z_alpha_2)
        alpha_1 = stats.norm.cdf(bias_correction + numerator_lower / denominator_lower)
        
        # 상한
        numerator_upper = bias_correction + z_1_alpha_2
        denominator_upper = 1 - acceleration * (bias_correction + z_1_alpha_2)
        alpha_2 = stats.norm.cdf(bias_correction + numerator_upper / denominator_upper)
        
        # 백분위수 계산
        alpha_1 = max(0, min(1, alpha_1))
        alpha_2 = max(0, min(1, alpha_2))
        
        ci_lower = np.percentile(bootstrap_stats, alpha_1 * 100)
        ci_upper = np.percentile(bootstrap_stats, alpha_2 * 100)
        
        return ci_lower, ci_upper
    
    def bayesian_comparison(self, strategy_returns: pd.Series, 
                          benchmark_returns: pd.Series) -> Dict:
        """베이지안 전략 비교"""
        if not BAYESIAN_AVAILABLE:
            return {'error': 'PyMC 라이브러리 없음'}
        
        self.logger.info("🔮 베이지안 전략 비교 중...")
        
        try:
            # 데이터 정렬
            common_index = strategy_returns.index.intersection(benchmark_returns.index)
            strategy_aligned = strategy_returns[common_index].values
            benchmark_aligned = benchmark_returns[common_index].values
            
            # 베이지안 모델 구축
            with pm.Model() as model:
                # 전략 수익률 모델
                strategy_mu = pm.Normal('strategy_mu', mu=0, sigma=0.1)
                strategy_sigma = pm.HalfNormal('strategy_sigma', sigma=0.1)
                
                # 벤치마크 수익률 모델
                benchmark_mu = pm.Normal('benchmark_mu', mu=0, sigma=0.1)
                benchmark_sigma = pm.HalfNormal('benchmark_sigma', sigma=0.1)
                
                # 관측 데이터
                strategy_obs = pm.Normal('strategy_obs', mu=strategy_mu, sigma=strategy_sigma, observed=strategy_aligned)
                benchmark_obs = pm.Normal('benchmark_obs', mu=benchmark_mu, sigma=benchmark_sigma, observed=benchmark_aligned)
                
                # 차이
                mu_diff = pm.Deterministic('mu_diff', strategy_mu - benchmark_mu)
                
                # MCMC 샘플링
                trace = pm.sample(self.config.mcmc_samples, chains=self.config.bayesian_chains, return_inferencedata=True)
            
            # 결과 분석
            posterior_summary = az.summary(trace)
            
            # 전략이 더 좋을 확률
            mu_diff_samples = trace.posterior['mu_diff'].values.flatten()
            prob_strategy_better = np.mean(mu_diff_samples > 0)
            
            # 95% 신뢰구간
            hdi_95 = az.hdi(trace, hdi_prob=0.95)
            
            bayesian_results = {
                'probability_strategy_better': float(prob_strategy_better),
                'mu_difference_posterior': {
                    'mean': float(np.mean(mu_diff_samples)),
                    'std': float(np.std(mu_diff_samples)),
                    'hdi_95': {
                        'lower': float(hdi_95['mu_diff'].values[0]),
                        'upper': float(hdi_95['mu_diff'].values[1])
                    }
                },
                'posterior_summary': {
                    param: {
                        'mean': float(row['mean']),
                        'std': float(row['sd']),
                        'hdi_3%': float(row['hdi_3%']),
                        'hdi_97%': float(row['hdi_97%'])
                    }
                    for param, row in posterior_summary.iterrows()
                },
                'convergence_diagnostics': {
                    'r_hat_max': float(posterior_summary['r_hat'].max()),
                    'ess_bulk_min': float(posterior_summary['ess_bulk'].min()),
                    'ess_tail_min': float(posterior_summary['ess_tail'].min())
                }
            }
            
            return bayesian_results
            
        except Exception as e:
            self.logger.error(f"베이지안 분석 실패: {e}")
            return {'error': str(e)}
    
    def time_series_specific_tests(self, data: pd.Series) -> Dict:
        """시계열 특화 검정"""
        self.logger.info("⏱️ 시계열 특화 검정 중...")
        
        data_clean = data.dropna()
        ts_results = {}
        
        # 1. 자기상관 검정 (이미 독립성 검정에 포함됨)
        
        # 2. ARCH 효과 검정 (이분산성)
        try:
            # Engle의 ARCH 검정
            residuals = data_clean - data_clean.mean()
            squared_residuals = residuals ** 2
            
            # 회귀분석: squared_residuals ~ lagged_squared_residuals
            lags = min(self.config.max_lags, len(squared_residuals) // 4)
            if lags > 0:
                lagged_data = np.column_stack([squared_residuals.shift(i).dropna() for i in range(1, lags+1)])
                y = squared_residuals[lags:]
                
                if len(y) > lags:
                    # OLS 회귀
                    X = sm.add_constant(lagged_data)
                    model = sm.OLS(y, X).fit()
                    
                    # LM 검정 통계량
                    lm_stat = len(y) * model.rsquared
                    lm_p = 1 - stats.chi2.cdf(lm_stat, lags)
                    
                    ts_results['arch_test'] = {
                        'lm_statistic': float(lm_stat),
                        'p_value': float(lm_p),
                        'has_arch_effect': lm_p < self.config.alpha,
                        'lags_tested': lags
                    }
        except Exception as e:
            self.logger.warning(f"ARCH 검정 실패: {e}")
        
        # 3. 단위근 검정 (이미 안정성 검정에 포함됨)
        
        # 4. 공적분 검정 (두 시리즈가 있는 경우만)
        
        # 5. 변동점 검정
        try:
            ts_results['change_point_detection'] = self._detect_change_points(data_clean)
        except Exception as e:
            self.logger.warning(f"변동점 검정 실패: {e}")
        
        return ts_results
    
    def _detect_change_points(self, data: pd.Series, min_size: int = 30) -> Dict:
        """변동점 탐지 (PELT 알고리즘 간단 버전)"""
        n = len(data)
        if n < min_size * 2:
            return {'change_points': [], 'method': 'insufficient_data'}
        
        # 간단한 분산 변화 탐지
        change_points = []
        
        for i in range(min_size, n - min_size):
            before = data[:i]
            after = data[i:]
            
            # F-검정으로 분산 변화 검정
            f_stat = np.var(after) / np.var(before) if np.var(before) > 0 else 1
            p_value = 2 * min(stats.f.cdf(f_stat, len(after)-1, len(before)-1),
                             1 - stats.f.cdf(f_stat, len(after)-1, len(before)-1))
            
            if p_value < 0.01:  # 엄격한 기준
                change_points.append({
                    'position': i,
                    'date': data.index[i] if hasattr(data, 'index') else i,
                    'f_statistic': float(f_stat),
                    'p_value': float(p_value)
                })
        
        return {
            'change_points': change_points,
            'method': 'variance_change_f_test',
            'total_change_points': len(change_points)
        }
    
    def analyze_effect_sizes(self, strategy_returns: pd.Series, 
                           benchmark_returns: pd.Series = None) -> Dict:
        """효과 크기 분석"""
        self.logger.info("📏 효과 크기 분석 중...")
        
        effect_size_results = {}
        
        if benchmark_returns is not None:
            # 데이터 정렬
            common_index = strategy_returns.index.intersection(benchmark_returns.index)
            strategy_aligned = strategy_returns[common_index]
            benchmark_aligned = benchmark_returns[common_index]
            
            # Cohen's d
            pooled_std = np.sqrt(((len(strategy_aligned) - 1) * np.var(strategy_aligned, ddof=1) + 
                                 (len(benchmark_aligned) - 1) * np.var(benchmark_aligned, ddof=1)) / 
                                (len(strategy_aligned) + len(benchmark_aligned) - 2))
            
            cohens_d = (np.mean(strategy_aligned) - np.mean(benchmark_aligned)) / pooled_std if pooled_std > 0 else 0
            
            # Hedges' g (편향 보정된 Cohen's d)
            correction_factor = 1 - (3 / (4 * (len(strategy_aligned) + len(benchmark_aligned) - 2) - 1))
            hedges_g = cohens_d * correction_factor
            
            # Glass's delta
            glass_delta = (np.mean(strategy_aligned) - np.mean(benchmark_aligned)) / np.std(benchmark_aligned, ddof=1)
            
            # 효과 크기 해석
            def interpret_effect_size(effect_size):
                abs_effect = abs(effect_size)
                if abs_effect < self.config.small_effect_size:
                    return 'negligible'
                elif abs_effect < self.config.medium_effect_size:
                    return 'small'
                elif abs_effect < self.config.large_effect_size:
                    return 'medium'
                else:
                    return 'large'
            
            effect_size_results['cohens_d'] = {
                'value': float(cohens_d),
                'interpretation': interpret_effect_size(cohens_d),
                'favors': 'strategy' if cohens_d > 0 else 'benchmark'
            }
            
            effect_size_results['hedges_g'] = {
                'value': float(hedges_g),
                'interpretation': interpret_effect_size(hedges_g),
                'favors': 'strategy' if hedges_g > 0 else 'benchmark'
            }
            
            effect_size_results['glass_delta'] = {
                'value': float(glass_delta),
                'interpretation': interpret_effect_size(glass_delta),
                'favors': 'strategy' if glass_delta > 0 else 'benchmark'
            }
        
        # 전략 자체의 효과 크기 (리스크 대비 수익률)
        strategy_clean = strategy_returns.dropna()
        if len(strategy_clean) > 0:
            # 샤프 비율을 효과 크기로 사용
            sharpe_ratio = np.mean(strategy_clean) / np.std(strategy_clean) * np.sqrt(252) if np.std(strategy_clean) > 0 else 0
            
            effect_size_results['strategy_sharpe_ratio'] = {
                'value': float(sharpe_ratio),
                'interpretation': interpret_effect_size(sharpe_ratio),
                'annualized': True
            }
        
        return effect_size_results
    
    def conduct_power_analysis(self, strategy_returns: pd.Series, 
                             benchmark_returns: pd.Series = None) -> Dict:
        """검정력 분석"""
        self.logger.info("⚡ 검정력 분석 중...")
        
        power_results = {}
        
        if benchmark_returns is not None:
            # 현재 샘플 크기
            common_index = strategy_returns.index.intersection(benchmark_returns.index)
            n_samples = len(common_index)
            
            if n_samples > 0:
                strategy_aligned = strategy_returns[common_index]
                benchmark_aligned = benchmark_returns[common_index]
                
                # 풀링된 표준편차
                pooled_std = np.sqrt(((len(strategy_aligned) - 1) * np.var(strategy_aligned, ddof=1) + 
                                     (len(benchmark_aligned) - 1) * np.var(benchmark_aligned, ddof=1)) / 
                                    (len(strategy_aligned) + len(benchmark_aligned) - 2))
                
                # 다양한 효과 크기에 대한 검정력 계산
                power_analysis = {}
                for effect_size in self.config.effect_sizes:
                    try:
                        power = TTestPower().solve_power(effect_size=effect_size, 
                                                        nobs=n_samples, 
                                                        alpha=self.config.alpha)
                        power_analysis[f'effect_size_{effect_size}'] = {
                            'power': float(power),
                            'adequate_power': power >= self.config.desired_power
                        }
                    except:
                        pass
                
                power_results['current_sample_power'] = power_analysis
                
                # 원하는 검정력을 위한 필요 샘플 크기
                required_samples = {}
                for effect_size in self.config.effect_sizes:
                    if effect_size > 0:  # 0 효과 크기는 제외
                        try:
                            required_n = TTestPower().solve_power(effect_size=effect_size,
                                                                 power=self.config.desired_power,
                                                                 alpha=self.config.alpha)
                            required_samples[f'effect_size_{effect_size}'] = {
                                'required_n': int(required_n),
                                'current_n': n_samples,
                                'additional_samples_needed': max(0, int(required_n) - n_samples)
                            }
                        except:
                            pass
                
                power_results['required_sample_sizes'] = required_samples
                
                # 현재 데이터로 탐지 가능한 최소 효과 크기
                try:
                    min_detectable_effect = TTestPower().solve_power(power=self.config.desired_power,
                                                                    nobs=n_samples,
                                                                    alpha=self.config.alpha)
                    power_results['minimum_detectable_effect'] = {
                        'effect_size': float(min_detectable_effect),
                        'interpretation': 'small' if min_detectable_effect < 0.5 else 'medium' if min_detectable_effect < 0.8 else 'large'
                    }
                except:
                    pass
        
        return power_results
    
    def conduct_robustness_tests(self, strategy_returns: pd.Series, 
                               benchmark_returns: pd.Series = None) -> Dict:
        """강건성 검정"""
        self.logger.info("🛡️ 강건성 검정 중...")
        
        robustness_results = {}
        
        # 1. 아웃라이어 제거 후 검정
        robustness_results['outlier_removal_tests'] = self._test_with_outlier_removal(
            strategy_returns, benchmark_returns
        )
        
        # 2. 부트스트랩 안정성 검정
        robustness_results['bootstrap_stability'] = self._bootstrap_stability_test(strategy_returns)
        
        # 3. 서브샘플 검정
        robustness_results['subsample_tests'] = self._subsample_robustness_test(
            strategy_returns, benchmark_returns
        )
        
        return robustness_results
    
    def _test_with_outlier_removal(self, strategy_returns: pd.Series, 
                                 benchmark_returns: pd.Series = None) -> Dict:
        """이상치 제거 후 검정"""
        results = {}
        
        for method in self.config.outlier_methods:
            try:
                if method == 'iqr':
                    strategy_clean = self._remove_outliers_iqr(strategy_returns)
                    benchmark_clean = self._remove_outliers_iqr(benchmark_returns) if benchmark_returns is not None else None
                elif method == 'zscore':
                    strategy_clean = self._remove_outliers_zscore(strategy_returns)
                    benchmark_clean = self._remove_outliers_zscore(benchmark_returns) if benchmark_returns is not None else None
                else:
                    continue  # isolation forest는 구현 생략
                
                # 이상치 제거 후 비교 검정
                if benchmark_clean is not None:
                    comparison_results = self.compare_strategies(strategy_clean, benchmark_clean)
                    results[f'{method}_removal'] = {
                        'strategy_observations_removed': len(strategy_returns) - len(strategy_clean),
                        'benchmark_observations_removed': len(benchmark_returns) - len(benchmark_clean) if benchmark_returns is not None else 0,
                        'test_results': comparison_results
                    }
            except Exception as e:
                self.logger.warning(f"이상치 제거 검정 실패 ({method}): {e}")
                continue
        
        return results
    
    def _remove_outliers_iqr(self, data: pd.Series) -> pd.Series:
        """IQR 방법으로 이상치 제거"""
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return data[(data >= lower_bound) & (data <= upper_bound)]
    
    def _remove_outliers_zscore(self, data: pd.Series, threshold: float = 3) -> pd.Series:
        """Z-score 방법으로 이상치 제거"""
        z_scores = np.abs(stats.zscore(data))
        return data[z_scores < threshold]
    
    def _bootstrap_stability_test(self, data: pd.Series, n_bootstrap: int = 1000) -> Dict:
        """부트스트랩 안정성 검정"""
        data_clean = data.dropna()
        
        if len(data_clean) < 30:
            return {'error': '부트스트랩에 충분하지 않은 데이터'}
        
        # 다양한 통계량의 부트스트랩 안정성
        statistics = {
            'mean': np.mean,
            'std': np.std,
            'skewness': lambda x: stats.skew(x),
            'kurtosis': lambda x: stats.kurtosis(x)
        }
        
        stability_results = {}
        
        for stat_name, stat_func in statistics.items():
            bootstrap_stats = []
            np.random.seed(42)
            
            for _ in range(n_bootstrap):
                bootstrap_sample = np.random.choice(data_clean, size=len(data_clean), replace=True)
                bootstrap_stat = stat_func(bootstrap_sample)
                bootstrap_stats.append(bootstrap_stat)
            
            bootstrap_stats = np.array(bootstrap_stats)
            bootstrap_stats = bootstrap_stats[~np.isnan(bootstrap_stats)]
            
            if len(bootstrap_stats) > 0:
                # 안정성 측정 (변동계수)
                stability_cv = np.std(bootstrap_stats) / abs(np.mean(bootstrap_stats)) if np.mean(bootstrap_stats) != 0 else np.inf
                
                stability_results[stat_name] = {
                    'bootstrap_mean': float(np.mean(bootstrap_stats)),
                    'bootstrap_std': float(np.std(bootstrap_stats)),
                    'coefficient_of_variation': float(stability_cv),
                    'is_stable': stability_cv < 0.1  # 10% 미만이면 안정
                }
        
        return stability_results
    
    def _subsample_robustness_test(self, strategy_returns: pd.Series, 
                                 benchmark_returns: pd.Series = None) -> Dict:
        """서브샘플 강건성 검정"""
        if benchmark_returns is None:
            return {'error': '벤치마크 데이터 필요'}
        
        common_index = strategy_returns.index.intersection(benchmark_returns.index)
        if len(common_index) < 100:
            return {'error': '서브샘플링에 충분하지 않은 데이터'}
        
        strategy_aligned = strategy_returns[common_index]
        benchmark_aligned = benchmark_returns[common_index]
        
        # 다양한 크기의 서브샘플로 검정
        sample_sizes = [0.5, 0.7, 0.9]  # 50%, 70%, 90%
        subsample_results = {}
        
        for sample_ratio in sample_sizes:
            sample_size = int(len(common_index) * sample_ratio)
            
            # 여러 번의 랜덤 서브샘플링
            n_subsamples = 50
            significant_tests = 0
            
            np.random.seed(42)
            for _ in range(n_subsamples):
                subsample_idx = np.random.choice(len(common_index), size=sample_size, replace=False)
                subsample_strategy = strategy_aligned.iloc[subsample_idx]
                subsample_benchmark = benchmark_aligned.iloc[subsample_idx]
                
                # t-검정 수행
                try:
                    _, p_value = ttest_ind(subsample_strategy, subsample_benchmark)
                    if p_value < self.config.alpha:
                        significant_tests += 1
                except:
                    continue
            
            subsample_results[f'sample_ratio_{sample_ratio}'] = {
                'sample_size': sample_size,
                'n_subsamples': n_subsamples,
                'significant_tests': significant_tests,
                'significance_ratio': significant_tests / n_subsamples,
                'is_robust': significant_tests / n_subsamples >= 0.8  # 80% 이상에서 유의하면 강건
            }
        
        return subsample_results
    
    def generate_statistical_conclusions(self, *test_results) -> Dict:
        """통계적 결론 생성"""
        self.logger.info("📋 통계적 결론 생성 중...")
        
        conclusions = {
            'overall_significance': 'inconclusive',
            'confidence_level': 'low',
            'recommendations': [],
            'limitations': [],
            'key_findings': []
        }
        
        # 각 검정 결과를 종합하여 결론 도출
        # (구현 생략 - 실제로는 복잡한 논리가 필요)
        
        return conclusions
    
    def create_significance_testing_dashboard(self, results: Dict) -> str:
        """통계적 유의성 검정 대시보드 생성"""
        self.logger.info("📊 통계적 유의성 검정 대시보드 생성 중...")
        
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=(
                "가정 검정 결과", "효과 크기 분석", "검정력 분석",
                "부트스트랩 신뢰구간", "베이지안 분석", "강건성 검정",
                "p-value 분포", "다중비교 결과", "시계열 특화 검정"
            ),
            specs=[[{"type": "table"}, {"type": "bar"}, {"type": "scatter"}],
                   [{"type": "box"}, {"type": "violin"}, {"type": "heatmap"}],
                   [{"type": "histogram"}, {"type": "bar"}, {"type": "indicator"}]]
        )
        
        # 대시보드 구현 (차트별로 구현)
        # ...
        
        fig.update_layout(
            title="🧪 통계적 유의성 검정 종합 대시보드",
            height=1200,
            showlegend=True,
            template='plotly_dark'
        )
        
        dashboard_path = os.path.join(self.data_path, 'statistical_significance_dashboard.html')
        fig.write_html(dashboard_path, include_plotlyjs=True)
        
        return dashboard_path
    
    def save_test_results(self, results: Dict):
        """검정 결과 저장"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # JSON 저장
        json_path = os.path.join(self.data_path, f'statistical_significance_results_{timestamp}.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        self.logger.info(f"검정 결과 저장: {json_path}")

def main():
    """메인 실행 함수"""
    print("🧪 통계적 유의성 검정 시스템")
    print("=" * 50)
    
    # 설정
    config = SignificanceConfig(
        alpha=0.05,
        bootstrap_samples=5000,
        multiple_comparison_method='holm',
        desired_power=0.8
    )
    
    # 검정 시스템 초기화
    tester = StatisticalSignificanceTester(config)
    
    # 테스트 데이터 생성
    np.random.seed(42)
    n_days = 500
    
    # 전략 수익률 (약간 더 좋은 성능)
    strategy_returns = pd.Series(
        np.random.normal(0.0015, 0.02, n_days),
        index=pd.date_range('2023-01-01', periods=n_days)
    )
    
    # 벤치마크 수익률
    benchmark_returns = pd.Series(
        np.random.normal(0.001, 0.02, n_days),
        index=pd.date_range('2023-01-01', periods=n_days)
    )
    
    # 종합 통계적 유의성 검정
    results = tester.comprehensive_significance_testing(strategy_returns, benchmark_returns)
    
    print(f"\n🧪 통계적 검정 완료!")
    print(f"데이터 기간: {results['data_summary']['time_period']['start']} ~ {results['data_summary']['time_period']['end']}")
    print(f"전략 관찰값: {results['data_summary']['strategy_observations']}개")
    
    # 주요 결과 출력
    if 'strategy_comparison' in results:
        comparison = results['strategy_comparison']
        if 'paired_ttest' in comparison:
            t_test = comparison['paired_ttest']
            print(f"대응 t-검정: t={t_test['statistic']:.3f}, p={t_test['p_value']:.4f}")
            print(f"전략이 더 우수함: {'예' if t_test['strategy_better'] and t_test['significant'] else '아니오'}")

if __name__ == "__main__":
    main()