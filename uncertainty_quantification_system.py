#!/usr/bin/env python3
"""
🎯 Uncertainty Quantification System
불확실성 정량화 시스템 - 예측 신뢰도 측정 및 리스크 관리

주요 기능:
1. Monte Carlo Dropout - 드롭아웃 기반 불확실성 추정
2. Ensemble Methods - 앙상블 기반 신뢰도 측정
3. Bayesian Neural Networks - 베이지안 뉴럴 네트워크
4. Prediction Intervals - 예측 구간 추정
5. Risk Assessment - 위험도 평가 및 관리
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, MultivariateNormal
from scipy import stats
from scipy.stats import norm, t
import warnings
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from collections import defaultdict
import logging
import joblib

warnings.filterwarnings('ignore')

@dataclass
class UncertaintyMetrics:
    """불확실성 메트릭"""
    mean: float
    std: float
    lower_ci: float
    upper_ci: float
    confidence: float
    epistemic_uncertainty: float  # 모델 불확실성
    aleatoric_uncertainty: float  # 데이터 불확실성
    total_uncertainty: float

class BayesianLinearLayer(nn.Module):
    """베이지안 선형 레이어"""
    
    def __init__(self, in_features: int, out_features: int, prior_std: float = 1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # 가중치 평균과 분산 파라미터
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features))
        
        # 편향 평균과 분산 파라미터
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features))
        
        # 사전 분포 설정
        self.prior_std = prior_std
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """파라미터 초기화"""
        # 가중치 초기화
        nn.init.normal_(self.weight_mu, 0, 0.1)
        nn.init.constant_(self.weight_rho, -3)  # 작은 분산으로 시작
        
        # 편향 초기화
        nn.init.normal_(self.bias_mu, 0, 0.1)
        nn.init.constant_(self.bias_rho, -3)
    
    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # 가중치 표준편차 계산 (softplus로 양수 보장)
        weight_std = torch.log1p(torch.exp(self.weight_rho))
        bias_std = torch.log1p(torch.exp(self.bias_rho))
        
        # 가중치와 편향 샘플링
        if self.training:
            weight = self.weight_mu + weight_std * torch.randn_like(weight_std)
            bias = self.bias_mu + bias_std * torch.randn_like(bias_std)
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        
        # 출력 계산
        output = F.linear(input, weight, bias)
        
        # KL 발산 계산 (정규화 항)
        weight_kl = self._kl_divergence(self.weight_mu, weight_std)
        bias_kl = self._kl_divergence(self.bias_mu, bias_std)
        kl_loss = weight_kl + bias_kl
        
        return output, kl_loss
    
    def _kl_divergence(self, mu: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        """가우시안 KL 발산 계산"""
        prior_std = self.prior_std
        var = std ** 2
        
        kl = torch.log(prior_std / std) + (var + mu ** 2) / (2 * prior_std ** 2) - 0.5
        return kl.sum()

class BayesianNeuralNetwork(nn.Module):
    """베이지안 뉴럴 네트워크"""
    
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int, 
                 prior_std: float = 1.0, dropout_rate: float = 0.1):
        super().__init__()
        
        self.layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        
        # 입력 레이어
        prev_dim = input_dim
        
        # 숨겨진 레이어들
        for hidden_dim in hidden_dims:
            self.layers.append(BayesianLinearLayer(prev_dim, hidden_dim, prior_std))
            self.dropout_layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        # 출력 레이어
        self.output_layer = BayesianLinearLayer(prev_dim, output_dim, prior_std)
        
        self.kl_weight = 1.0
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        total_kl = 0.0
        
        # 순전파
        for layer, dropout in zip(self.layers, self.dropout_layers):
            x, kl = layer(x)
            x = F.relu(x)
            x = dropout(x)
            total_kl += kl
        
        # 출력 레이어
        output, kl = self.output_layer(x)
        total_kl += kl
        
        return output, total_kl * self.kl_weight

class MonteCarloDropout:
    """몬테카를로 드롭아웃 불확실성 추정"""
    
    def __init__(self, model: nn.Module, num_samples: int = 100):
        self.model = model
        self.num_samples = num_samples
    
    def predict_with_uncertainty(self, X: torch.Tensor) -> UncertaintyMetrics:
        """불확실성과 함께 예측"""
        self.model.train()  # 드롭아웃 활성화
        
        predictions = []
        
        with torch.no_grad():
            for _ in range(self.num_samples):
                pred = self.model(X)
                if isinstance(pred, tuple):
                    pred = pred[0]  # 베이지안 모델인 경우
                predictions.append(pred.cpu().numpy())
        
        predictions = np.array(predictions)
        
        # 통계 계산
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        
        # 신뢰 구간 (95%)
        lower_ci = np.percentile(predictions, 2.5, axis=0)
        upper_ci = np.percentile(predictions, 97.5, axis=0)
        
        # 신뢰도 계산 (변이 계수의 역수)
        confidence = 1.0 / (1.0 + np.mean(std_pred) / (np.abs(np.mean(mean_pred)) + 1e-8))
        
        # 불확실성 분해
        epistemic_uncertainty = np.mean(std_pred)  # 모델 불확실성
        aleatoric_uncertainty = self._estimate_aleatoric_uncertainty(predictions)
        total_uncertainty = np.sqrt(epistemic_uncertainty**2 + aleatoric_uncertainty**2)
        
        return UncertaintyMetrics(
            mean=float(np.mean(mean_pred)),
            std=float(np.mean(std_pred)),
            lower_ci=float(np.mean(lower_ci)),
            upper_ci=float(np.mean(upper_ci)),
            confidence=float(confidence),
            epistemic_uncertainty=float(epistemic_uncertainty),
            aleatoric_uncertainty=float(aleatoric_uncertainty),
            total_uncertainty=float(total_uncertainty)
        )
    
    def _estimate_aleatoric_uncertainty(self, predictions: np.ndarray) -> float:
        """데이터 불확실성 추정"""
        # 예측의 분산을 이용한 간단한 추정
        sample_vars = np.var(predictions, axis=1)
        return float(np.sqrt(np.mean(sample_vars)))

class EnsembleUncertainty:
    """앙상블 기반 불확실성 추정"""
    
    def __init__(self, models: List, model_types: List[str] = None):
        self.models = models
        self.model_types = model_types or ['model'] * len(models)
        self.weights = None
    
    def fit_ensemble_weights(self, X_val: np.ndarray, y_val: np.ndarray):
        """검증 데이터로 앙상블 가중치 학습"""
        predictions = []
        
        for model in self.models:
            if hasattr(model, 'predict'):
                pred = model.predict(X_val)
            else:
                pred = model(torch.FloatTensor(X_val)).detach().cpu().numpy()
            predictions.append(pred.flatten())
        
        predictions = np.array(predictions).T  # (samples, models)
        
        # 각 모델의 성능 기반 가중치 계산
        weights = []
        for i, pred in enumerate(predictions.T):
            mae = mean_absolute_error(y_val, pred)
            weight = 1.0 / (mae + 1e-8)  # MAE의 역수
            weights.append(weight)
        
        # 정규화
        weights = np.array(weights)
        self.weights = weights / np.sum(weights)
        
        return self.weights
    
    def predict_with_uncertainty(self, X: Union[np.ndarray, torch.Tensor]) -> UncertaintyMetrics:
        """앙상블 불확실성 예측"""
        predictions = []
        
        for i, model in enumerate(self.models):
            if hasattr(model, 'predict'):
                pred = model.predict(X)
            else:
                if isinstance(X, np.ndarray):
                    X_tensor = torch.FloatTensor(X)
                else:
                    X_tensor = X
                pred = model(X_tensor)
                if isinstance(pred, tuple):
                    pred = pred[0]
                pred = pred.detach().cpu().numpy()
            
            predictions.append(pred.flatten())
        
        predictions = np.array(predictions)  # (models, samples)
        
        # 가중 평균 (가중치가 있는 경우)
        if self.weights is not None:
            mean_pred = np.average(predictions, axis=0, weights=self.weights)
        else:
            mean_pred = np.mean(predictions, axis=0)
        
        # 앙상블 불확실성 계산
        ensemble_std = np.std(predictions, axis=0)
        
        # 신뢰 구간
        lower_ci = np.percentile(predictions, 2.5, axis=0)
        upper_ci = np.percentile(predictions, 97.5, axis=0)
        
        # 신뢰도 (일치도 기반)
        agreement = 1.0 - np.mean(ensemble_std) / (np.abs(np.mean(mean_pred)) + 1e-8)
        confidence = max(0.0, min(1.0, agreement))
        
        return UncertaintyMetrics(
            mean=float(np.mean(mean_pred)),
            std=float(np.mean(ensemble_std)),
            lower_ci=float(np.mean(lower_ci)),
            upper_ci=float(np.mean(upper_ci)),
            confidence=float(confidence),
            epistemic_uncertainty=float(np.mean(ensemble_std)),  # 모델 간 불일치
            aleatoric_uncertainty=0.0,  # 앙상블에서는 직접 측정 어려움
            total_uncertainty=float(np.mean(ensemble_std))
        )

class GaussianProcessUncertainty:
    """가우시안 프로세스 불확실성 추정"""
    
    def __init__(self, kernel=None, alpha: float = 1e-10):
        if kernel is None:
            kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
        
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=alpha,
            n_restarts_optimizer=10,
            normalize_y=True
        )
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """가우시안 프로세스 훈련"""
        self.gp.fit(X, y)
        return self
    
    def predict_with_uncertainty(self, X: np.ndarray) -> UncertaintyMetrics:
        """가우시안 프로세스 불확실성 예측"""
        mean_pred, std_pred = self.gp.predict(X, return_std=True)
        
        # 신뢰 구간 (95%)
        lower_ci = mean_pred - 1.96 * std_pred
        upper_ci = mean_pred + 1.96 * std_pred
        
        # 신뢰도 (불확실성의 역수)
        confidence = 1.0 / (1.0 + np.mean(std_pred))
        
        return UncertaintyMetrics(
            mean=float(np.mean(mean_pred)),
            std=float(np.mean(std_pred)),
            lower_ci=float(np.mean(lower_ci)),
            upper_ci=float(np.mean(upper_ci)),
            confidence=float(confidence),
            epistemic_uncertainty=float(np.mean(std_pred)),  # GP 불확실성
            aleatoric_uncertainty=0.0,  # GP에서는 노이즈 모델링 필요
            total_uncertainty=float(np.mean(std_pred))
        )

class QuantileRegression:
    """분위수 회귀 기반 예측 구간"""
    
    def __init__(self, quantiles: List[float] = [0.025, 0.5, 0.975]):
        self.quantiles = quantiles
        self.models = {}
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """분위수 회귀 모델 훈련"""
        from sklearn.ensemble import GradientBoostingRegressor
        
        for q in self.quantiles:
            model = GradientBoostingRegressor(
                loss='quantile',
                alpha=q,
                n_estimators=100,
                random_state=42
            )
            model.fit(X, y)
            self.models[q] = model
        
        return self
    
    def predict_with_uncertainty(self, X: np.ndarray) -> UncertaintyMetrics:
        """분위수 기반 불확실성 예측"""
        predictions = {}
        
        for q, model in self.models.items():
            predictions[q] = model.predict(X)
        
        # 중앙값을 평균으로 사용
        mean_pred = predictions[0.5]
        
        # 신뢰 구간
        lower_ci = predictions[0.025]
        upper_ci = predictions[0.975]
        
        # 표준편차 추정 (IQR 기반)
        std_pred = (upper_ci - lower_ci) / 3.92  # 95% CI를 표준편차로 변환
        
        # 신뢰도
        interval_width = upper_ci - lower_ci
        confidence = 1.0 / (1.0 + np.mean(interval_width) / (np.abs(np.mean(mean_pred)) + 1e-8))
        
        return UncertaintyMetrics(
            mean=float(np.mean(mean_pred)),
            std=float(np.mean(std_pred)),
            lower_ci=float(np.mean(lower_ci)),
            upper_ci=float(np.mean(upper_ci)),
            confidence=float(confidence),
            epistemic_uncertainty=float(np.mean(std_pred)),
            aleatoric_uncertainty=0.0,
            total_uncertainty=float(np.mean(std_pred))
        )

class RiskAssessment:
    """위험도 평가 시스템"""
    
    def __init__(self):
        self.risk_thresholds = {
            'low': 0.05,      # 5% 변동
            'medium': 0.10,   # 10% 변동
            'high': 0.20,     # 20% 변동
            'extreme': 0.30   # 30% 변동
        }
    
    def assess_prediction_risk(self, uncertainty_metrics: UncertaintyMetrics, 
                              current_price: float) -> Dict:
        """예측 위험도 평가"""
        # 상대적 불확실성 계산
        relative_uncertainty = uncertainty_metrics.std / current_price if current_price > 0 else 0
        
        # 위험 등급 결정
        risk_level = 'low'
        for level, threshold in sorted(self.risk_thresholds.items(), 
                                     key=lambda x: x[1], reverse=True):
            if relative_uncertainty >= threshold:
                risk_level = level
                break
        
        # VaR (Value at Risk) 계산 (95% 신뢰도)
        var_95 = current_price - uncertainty_metrics.lower_ci
        
        # 손실 확률 추정
        if uncertainty_metrics.std > 0:
            loss_prob = 1 - stats.norm.cdf(0, 
                                          uncertainty_metrics.mean - current_price, 
                                          uncertainty_metrics.std)
        else:
            loss_prob = 0.0
        
        # 극단 위험 지표
        extreme_risk_indicators = {
            'high_uncertainty': relative_uncertainty > self.risk_thresholds['high'],
            'low_confidence': uncertainty_metrics.confidence < 0.5,
            'wide_prediction_interval': (uncertainty_metrics.upper_ci - uncertainty_metrics.lower_ci) / current_price > 0.2,
            'high_epistemic_uncertainty': uncertainty_metrics.epistemic_uncertainty / current_price > 0.1
        }
        
        return {
            'risk_level': risk_level,
            'relative_uncertainty': float(relative_uncertainty),
            'var_95': float(var_95),
            'loss_probability': float(loss_prob),
            'confidence_score': uncertainty_metrics.confidence,
            'extreme_risk_indicators': extreme_risk_indicators,
            'risk_score': self._calculate_composite_risk_score(uncertainty_metrics, current_price)
        }
    
    def _calculate_composite_risk_score(self, uncertainty_metrics: UncertaintyMetrics, 
                                      current_price: float) -> float:
        """복합 위험 점수 계산 (0-1, 높을수록 위험)"""
        factors = [
            uncertainty_metrics.std / current_price if current_price > 0 else 0,  # 상대 불확실성
            1 - uncertainty_metrics.confidence,  # 신뢰도의 역수
            uncertainty_metrics.epistemic_uncertainty / current_price if current_price > 0 else 0,  # 모델 불확실성
            (uncertainty_metrics.upper_ci - uncertainty_metrics.lower_ci) / current_price if current_price > 0 else 0  # 예측 구간 폭
        ]
        
        # 가중 평균 (동일 가중치)
        risk_score = np.mean(factors)
        return float(min(1.0, risk_score))

class UncertaintyQuantificationSystem:
    """완전한 불확실성 정량화 시스템"""
    
    def __init__(self):
        self.methods = {}
        self.risk_assessor = RiskAssessment()
        self.calibration_data = []
        self.setup_logging()
    
    def setup_logging(self):
        """로깅 설정"""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def add_uncertainty_method(self, name: str, method):
        """불확실성 추정 방법 추가"""
        self.methods[name] = method
        self.logger.info(f"불확실성 방법 추가: {name}")
    
    def create_bayesian_ensemble(self, input_dim: int, horizons: List[int]) -> Dict:
        """베이지안 앙상블 생성"""
        ensemble_models = {}
        
        for horizon in horizons:
            models = []
            
            # 베이지안 뉴럴 네트워크
            bnn = BayesianNeuralNetwork(
                input_dim=input_dim,
                hidden_dims=[128, 64, 32],
                output_dim=1,
                prior_std=1.0,
                dropout_rate=0.2
            )
            models.append(bnn)
            
            # 다양한 랜덤 포레스트
            rf_models = [
                RandomForestRegressor(n_estimators=100, max_depth=10, random_state=i)
                for i in range(3)
            ]
            models.extend(rf_models)
            
            # 그래디언트 부스팅
            gb_models = [
                GradientBoostingRegressor(n_estimators=100, max_depth=6, random_state=i)
                for i in range(2)
            ]
            models.extend(gb_models)
            
            # 앙상블 불확실성 추정기
            ensemble_uncertainty = EnsembleUncertainty(
                models=models,
                model_types=['bayesian'] + ['rf']*3 + ['gb']*2
            )
            
            ensemble_models[horizon] = ensemble_uncertainty
        
        return ensemble_models
    
    def fit_uncertainty_methods(self, X_train: np.ndarray, y_train: np.ndarray,
                               X_val: np.ndarray, y_val: np.ndarray):
        """불확실성 추정 방법들 훈련"""
        self.logger.info("🎯 불확실성 추정 방법 훈련 시작")
        
        # 가우시안 프로세스
        gp_uncertainty = GaussianProcessUncertainty()
        gp_uncertainty.fit(X_train, y_train)
        self.add_uncertainty_method('gaussian_process', gp_uncertainty)
        
        # 분위수 회귀
        quantile_regression = QuantileRegression()
        quantile_regression.fit(X_train, y_train)
        self.add_uncertainty_method('quantile_regression', quantile_regression)
        
        # 베이지안 앙상블
        bayesian_ensemble = self.create_bayesian_ensemble(X_train.shape[1], [1, 24, 168])
        
        # 앙상블 훈련 (간단한 경우만)
        for horizon, ensemble in bayesian_ensemble.items():
            # 검증 데이터로 가중치 학습
            sklearn_models = [m for m in ensemble.models if hasattr(m, 'fit')]
            for model in sklearn_models:
                model.fit(X_train, y_train)
            
            ensemble.fit_ensemble_weights(X_val, y_val)
            self.add_uncertainty_method(f'bayesian_ensemble_{horizon}h', ensemble)
        
        self.logger.info("✅ 불확실성 추정 방법 훈련 완료")
    
    def predict_with_full_uncertainty(self, X: np.ndarray, current_price: float) -> Dict:
        """완전한 불확실성 분석으로 예측"""
        self.logger.info(f"🔮 불확실성 분석 예측 시작 - 샘플: {X.shape[0]}")
        
        method_results = {}
        
        # 각 방법으로 불확실성 추정
        for method_name, method in self.methods.items():
            try:
                uncertainty_metrics = method.predict_with_uncertainty(X)
                risk_assessment = self.risk_assessor.assess_prediction_risk(
                    uncertainty_metrics, current_price
                )
                
                method_results[method_name] = {
                    'uncertainty_metrics': uncertainty_metrics,
                    'risk_assessment': risk_assessment
                }
                
            except Exception as e:
                self.logger.warning(f"방법 {method_name} 실행 중 오류: {str(e)}")
                continue
        
        # 메타 불확실성 분석
        meta_analysis = self._meta_uncertainty_analysis(method_results)
        
        # 최종 통합 결과
        final_result = self._integrate_uncertainty_results(method_results, meta_analysis, current_price)
        
        self.logger.info("✅ 불확실성 분석 예측 완료")
        
        return final_result
    
    def _meta_uncertainty_analysis(self, method_results: Dict) -> Dict:
        """메타 불확실성 분석"""
        if not method_results:
            return {'consensus': 'low', 'agreement_score': 0.0}
        
        # 모든 방법의 예측 수집
        predictions = []
        confidences = []
        risk_levels = []
        
        for method_name, result in method_results.items():
            metrics = result['uncertainty_metrics']
            risk = result['risk_assessment']
            
            predictions.append(metrics.mean)
            confidences.append(metrics.confidence)
            risk_levels.append(risk['risk_score'])
        
        # 방법 간 일치도
        pred_std = np.std(predictions) if len(predictions) > 1 else 0.0
        conf_mean = np.mean(confidences)
        risk_mean = np.mean(risk_levels)
        
        # 합의 수준
        if pred_std / (np.abs(np.mean(predictions)) + 1e-8) < 0.05:
            consensus = 'high'
        elif pred_std / (np.abs(np.mean(predictions)) + 1e-8) < 0.15:
            consensus = 'medium'
        else:
            consensus = 'low'
        
        # 일치도 점수
        agreement_score = 1.0 / (1.0 + pred_std)
        
        return {
            'consensus': consensus,
            'agreement_score': float(agreement_score),
            'prediction_variance': float(pred_std),
            'average_confidence': float(conf_mean),
            'average_risk': float(risk_mean),
            'method_count': len(method_results)
        }
    
    def _integrate_uncertainty_results(self, method_results: Dict, meta_analysis: Dict, 
                                     current_price: float) -> Dict:
        """불확실성 결과 통합"""
        if not method_results:
            return self._create_empty_uncertainty_result()
        
        # 가중 평균 계산
        total_weight = 0.0
        weighted_mean = 0.0
        weighted_std = 0.0
        weighted_confidence = 0.0
        
        method_weights = {
            'gaussian_process': 0.3,
            'quantile_regression': 0.2,
            'bayesian_ensemble_1h': 0.2,
            'bayesian_ensemble_24h': 0.2,
            'bayesian_ensemble_168h': 0.1
        }
        
        individual_results = {}
        
        for method_name, result in method_results.items():
            metrics = result['uncertainty_metrics']
            risk = result['risk_assessment']
            
            # 방법별 가중치 (신뢰도와 사전 정의 가중치 결합)
            base_weight = method_weights.get(method_name, 0.1)
            confidence_weight = metrics.confidence
            final_weight = base_weight * confidence_weight
            
            weighted_mean += metrics.mean * final_weight
            weighted_std += metrics.std * final_weight
            weighted_confidence += metrics.confidence * final_weight
            total_weight += final_weight
            
            # 개별 결과 저장
            individual_results[method_name] = {
                'prediction': metrics.mean,
                'uncertainty': metrics.std,
                'confidence': metrics.confidence,
                'risk_level': risk['risk_level'],
                'risk_score': risk['risk_score']
            }
        
        # 가중 평균 정규화
        if total_weight > 0:
            final_mean = weighted_mean / total_weight
            final_std = weighted_std / total_weight
            final_confidence = weighted_confidence / total_weight
        else:
            final_mean = final_std = final_confidence = 0.0
        
        # 최종 위험 평가
        final_uncertainty_metrics = UncertaintyMetrics(
            mean=final_mean,
            std=final_std,
            lower_ci=final_mean - 1.96 * final_std,
            upper_ci=final_mean + 1.96 * final_std,
            confidence=final_confidence,
            epistemic_uncertainty=final_std * 0.7,  # 추정
            aleatoric_uncertainty=final_std * 0.3,   # 추정
            total_uncertainty=final_std
        )
        
        final_risk = self.risk_assessor.assess_prediction_risk(
            final_uncertainty_metrics, current_price
        )
        
        return {
            'integrated_prediction': {
                'mean': final_mean,
                'std': final_std,
                'lower_ci': final_mean - 1.96 * final_std,
                'upper_ci': final_mean + 1.96 * final_std,
                'confidence': final_confidence
            },
            'risk_assessment': final_risk,
            'meta_analysis': meta_analysis,
            'individual_methods': individual_results,
            'analysis_timestamp': datetime.now().isoformat(),
            'summary': {
                'prediction_quality': 'high' if final_confidence > 0.8 else 'medium' if final_confidence > 0.5 else 'low',
                'uncertainty_level': final_risk['risk_level'],
                'recommendation': self._generate_recommendation(final_uncertainty_metrics, final_risk)
            }
        }
    
    def _generate_recommendation(self, uncertainty_metrics: UncertaintyMetrics, 
                               risk_assessment: Dict) -> str:
        """투자 권고 생성"""
        confidence = uncertainty_metrics.confidence
        risk_level = risk_assessment['risk_level']
        
        if confidence > 0.8 and risk_level == 'low':
            return "높은 신뢰도, 낮은 위험 - 투자 적극 고려"
        elif confidence > 0.6 and risk_level in ['low', 'medium']:
            return "중간 신뢰도, 적정 위험 - 신중한 투자 고려"
        elif risk_level in ['high', 'extreme']:
            return "높은 위험 - 투자 주의 또는 회피"
        else:
            return "낮은 신뢰도 - 추가 분석 필요"
    
    def _create_empty_uncertainty_result(self) -> Dict:
        """빈 불확실성 결과 생성"""
        return {
            'integrated_prediction': {
                'mean': 0.0,
                'std': 0.0,
                'lower_ci': 0.0,
                'upper_ci': 0.0,
                'confidence': 0.0
            },
            'risk_assessment': {
                'risk_level': 'unknown',
                'relative_uncertainty': 0.0,
                'confidence_score': 0.0
            },
            'meta_analysis': {
                'consensus': 'none',
                'agreement_score': 0.0
            },
            'individual_methods': {},
            'analysis_timestamp': datetime.now().isoformat(),
            'summary': {
                'prediction_quality': 'unknown',
                'uncertainty_level': 'unknown',
                'recommendation': '데이터 부족으로 분석 불가'
            }
        }
    
    def calibrate_uncertainty(self, X_test: np.ndarray, y_test: np.ndarray):
        """불확실성 보정"""
        self.logger.info("🎯 불확실성 보정 시작")
        
        calibration_results = {}
        
        for method_name, method in self.methods.items():
            try:
                uncertainty_metrics = method.predict_with_uncertainty(X_test)
                
                # 보정 메트릭 계산
                predictions = np.full(len(X_test), uncertainty_metrics.mean)
                errors = np.abs(y_test - predictions)
                
                # 예측 구간 포함 비율
                lower_bound = np.full(len(X_test), uncertainty_metrics.lower_ci)
                upper_bound = np.full(len(X_test), uncertainty_metrics.upper_ci)
                
                coverage = np.mean((y_test >= lower_bound) & (y_test <= upper_bound))
                
                calibration_results[method_name] = {
                    'coverage': float(coverage),
                    'mean_error': float(np.mean(errors)),
                    'calibration_score': float(abs(coverage - 0.95))  # 95% 구간 대비
                }
                
            except Exception as e:
                self.logger.warning(f"보정 중 오류 {method_name}: {str(e)}")
                continue
        
        self.calibration_data.append(calibration_results)
        self.logger.info("✅ 불확실성 보정 완료")
        
        return calibration_results
    
    def save_system(self, filepath: str):
        """시스템 저장"""
        save_data = {
            'calibration_data': self.calibration_data,
            'risk_thresholds': self.risk_assessor.risk_thresholds,
            'system_timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        
        # 모델들은 별도 저장
        model_data = {}
        for name, method in self.methods.items():
            if hasattr(method, 'gp'):  # 가우시안 프로세스
                joblib.dump(method.gp, f'{filepath}_{name}_gp.pkl')
            elif hasattr(method, 'models') and hasattr(method.models, 'items'):  # 분위수 회귀
                joblib.dump(method.models, f'{filepath}_{name}_models.pkl')
        
        self.logger.info(f"불확실성 시스템 저장 완료: {filepath}")

def main():
    """메인 테스트 함수"""
    # 테스트 데이터 생성
    np.random.seed(42)
    n_samples = 1000
    n_features = 20
    
    # 합성 특성 데이터
    X = np.random.randn(n_samples, n_features)
    
    # 비선형 타겟 (노이즈 포함)
    true_function = lambda x: np.sum(x[:, :5] ** 2, axis=1) + 0.1 * np.sum(x[:, 5:10], axis=1)
    noise = np.random.normal(0, 0.1, n_samples)
    y = true_function(X) + noise
    
    # 훈련/검증/테스트 분할
    train_size = int(0.6 * n_samples)
    val_size = int(0.2 * n_samples)
    
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_val = X[train_size:train_size+val_size]
    y_val = y[train_size:train_size+val_size]
    X_test = X[train_size+val_size:]
    y_test = y[train_size+val_size:]
    
    # 불확실성 시스템 초기화
    uncertainty_system = UncertaintyQuantificationSystem()
    
    print("🎯 Uncertainty Quantification System Test")
    print("="*60)
    
    # 시스템 훈련
    uncertainty_system.fit_uncertainty_methods(X_train, y_train, X_val, y_val)
    
    # 예측 및 불확실성 분석
    current_price = 55000.0  # 현재 BTC 가격 가정
    test_sample = X_test[:5]  # 5개 샘플 테스트
    
    uncertainty_results = uncertainty_system.predict_with_full_uncertainty(test_sample, current_price)
    
    # 결과 출력
    print(f"\n📊 통합 불확실성 분석 결과:")
    pred = uncertainty_results['integrated_prediction']
    print(f"  예측값: {pred['mean']:.4f} ± {pred['std']:.4f}")
    print(f"  신뢰구간: [{pred['lower_ci']:.4f}, {pred['upper_ci']:.4f}]")
    print(f"  신뢰도: {pred['confidence']:.3f}")
    
    risk = uncertainty_results['risk_assessment']
    print(f"\n⚠️ 위험 평가:")
    print(f"  위험 수준: {risk['risk_level']}")
    print(f"  위험 점수: {risk['risk_score']:.3f}")
    print(f"  손실 확률: {risk['loss_probability']:.3f}")
    
    meta = uncertainty_results['meta_analysis']
    print(f"\n🤝 메타 분석:")
    print(f"  합의 수준: {meta['consensus']}")
    print(f"  일치도: {meta['agreement_score']:.3f}")
    print(f"  방법 수: {meta['method_count']}")
    
    summary = uncertainty_results['summary']
    print(f"\n💡 요약:")
    print(f"  예측 품질: {summary['prediction_quality']}")
    print(f"  불확실성: {summary['uncertainty_level']}")
    print(f"  권고사항: {summary['recommendation']}")
    
    # 개별 방법 결과
    print(f"\n🔍 개별 방법별 결과:")
    for method_name, result in uncertainty_results['individual_methods'].items():
        print(f"  {method_name}:")
        print(f"    예측: {result['prediction']:.4f}")
        print(f"    불확실성: {result['uncertainty']:.4f}")
        print(f"    신뢰도: {result['confidence']:.3f}")
        print(f"    위험: {result['risk_level']}")
    
    # 보정 테스트
    calibration_results = uncertainty_system.calibrate_uncertainty(X_test, y_test)
    
    print(f"\n📈 불확실성 보정 결과:")
    for method_name, calibration in calibration_results.items():
        print(f"  {method_name}:")
        print(f"    커버리지: {calibration['coverage']:.3f}")
        print(f"    평균 오차: {calibration['mean_error']:.4f}")
        print(f"    보정 점수: {calibration['calibration_score']:.4f}")
    
    # 결과 저장
    with open('uncertainty_quantification_results.json', 'w', encoding='utf-8') as f:
        json.dump(uncertainty_results, f, indent=2, ensure_ascii=False, default=str)
    
    uncertainty_system.save_system('uncertainty_system.json')
    
    print(f"\n💾 결과 저장 완료")
    print(f"  - uncertainty_quantification_results.json")
    print(f"  - uncertainty_system.json")
    
    return uncertainty_results

if __name__ == "__main__":
    main()