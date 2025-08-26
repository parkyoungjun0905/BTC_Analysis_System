#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
고급 시계열 교차 검증 시스템
- 워크포워드 검증
- 퍼지드 그룹 시계열 분할
- 블록킹 시계열 검증
- 샘플 외 테스트 프로토콜
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

import os
import sys
import json
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union, Iterator, Generator
import logging
import pickle
from dataclasses import dataclass
from abc import ABC, abstractmethod

# 과학 계산
from scipy import stats
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d

# 머신러닝
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, BaseCrossValidator
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

# 통계 및 시각화
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.dates import DateFormatter
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """검증 결과 데이터 클래스"""
    train_scores: List[float]
    test_scores: List[float]
    train_indices: List[np.ndarray]
    test_indices: List[np.ndarray]
    fold_metadata: List[Dict]
    overall_metrics: Dict[str, float]
    timestamp: datetime

class CustomTimeSeriesSplit(BaseCrossValidator):
    """커스텀 시계열 분할 기본 클래스"""
    
    def __init__(self, n_splits: int = 5):
        self.n_splits = n_splits
    
    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits

class WalkForwardValidation(CustomTimeSeriesSplit):
    """
    워크포워드 검증
    - 시간에 따라 훈련 윈도우가 확장되는 방식
    - 실제 거래 환경을 가장 잘 모방
    """
    
    def __init__(self, 
                 n_splits: int = 5,
                 min_train_size: int = None,
                 max_train_size: int = None,
                 test_size: int = None,
                 gap: int = 0):
        """
        Args:
            n_splits: 분할 수
            min_train_size: 최소 훈련 크기
            max_train_size: 최대 훈련 크기
            test_size: 테스트 크기
            gap: 훈련과 테스트 간 간격 (데이터 유출 방지)
        """
        super().__init__(n_splits)
        self.min_train_size = min_train_size
        self.max_train_size = max_train_size
        self.test_size = test_size
        self.gap = gap
    
    def split(self, X, y=None, groups=None) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        데이터를 워크포워드 방식으로 분할
        
        Yields:
            (train_indices, test_indices) 튜플
        """
        n_samples = len(X)
        
        # 기본값 설정
        test_size = self.test_size or max(1, n_samples // (self.n_splits + 1))
        min_train_size = self.min_train_size or max(50, n_samples // 5)
        
        # 첫 번째 분할의 시작점
        start_idx = min_train_size
        
        for i in range(self.n_splits):
            # 테스트 시작점과 끝점
            test_start = start_idx + i * test_size + self.gap
            test_end = min(test_start + test_size, n_samples)
            
            if test_end - test_start < test_size // 2:
                break
            
            # 훈련 데이터 범위
            train_end = test_start - self.gap
            train_start = 0
            
            # 최대 훈련 크기 제한
            if self.max_train_size:
                train_start = max(0, train_end - self.max_train_size)
            
            if train_end - train_start < min_train_size:
                break
            
            train_indices = np.arange(train_start, train_end)
            test_indices = np.arange(test_start, test_end)
            
            yield train_indices, test_indices

class PurgedGroupTimeSeriesSplit(CustomTimeSeriesSplit):
    """
    퍼지드 그룹 시계열 분할
    - 그룹 기반 분할로 데이터 유출 방지
    - 시간적 종속성을 고려한 purge 적용
    """
    
    def __init__(self, 
                 n_splits: int = 5,
                 group_gap: timedelta = None,
                 purge_gap: timedelta = None):
        """
        Args:
            n_splits: 분할 수
            group_gap: 그룹 간 최소 간격
            purge_gap: 퍼지 간격
        """
        super().__init__(n_splits)
        self.group_gap = group_gap or timedelta(hours=24)
        self.purge_gap = purge_gap or timedelta(hours=6)
    
    def _create_groups(self, timestamps: pd.DatetimeIndex) -> np.ndarray:
        """타임스탬프를 기반으로 그룹 생성"""
        groups = np.zeros(len(timestamps), dtype=int)
        current_group = 0
        
        for i, ts in enumerate(timestamps):
            if i == 0:
                groups[i] = current_group
            else:
                if ts - timestamps[i-1] > self.group_gap:
                    current_group += 1
                groups[i] = current_group
        
        return groups
    
    def split(self, X, y=None, groups=None) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        그룹 기반 퍼지드 분할
        
        Yields:
            (train_indices, test_indices) 튜플
        """
        if isinstance(X, pd.DataFrame) and isinstance(X.index, pd.DatetimeIndex):
            timestamps = X.index
        else:
            # 인덱스가 타임스탬프가 아니면 순차적 그룹 생성
            timestamps = pd.date_range('2024-01-01', periods=len(X), freq='H')
        
        if groups is None:
            groups = self._create_groups(timestamps)
        
        unique_groups = np.unique(groups)
        n_groups = len(unique_groups)
        
        test_size = max(1, n_groups // (self.n_splits + 1))
        
        for i in range(self.n_splits):
            # 테스트 그룹 선택
            test_start_group = i * test_size
            test_end_group = min((i + 1) * test_size, n_groups)
            
            if test_end_group <= test_start_group:
                break
            
            test_groups = unique_groups[test_start_group:test_end_group]
            test_indices = np.where(np.isin(groups, test_groups))[0]
            
            # 훈련 그룹 (테스트 이전)
            train_groups = unique_groups[:test_start_group]
            train_indices = np.where(np.isin(groups, train_groups))[0]
            
            # 퍼지 적용 (테스트 시작 전 일정 시간 제거)
            if len(test_indices) > 0 and len(train_indices) > 0:
                test_start_time = timestamps[test_indices[0]]
                purge_cutoff = test_start_time - self.purge_gap
                
                train_indices = train_indices[timestamps[train_indices] <= purge_cutoff]
            
            if len(train_indices) > 0 and len(test_indices) > 0:
                yield train_indices, test_indices

class BlockingTimeSeriesCV(CustomTimeSeriesSplit):
    """
    블록킹 시계열 교차 검증
    - 시간 블록 단위로 분할
    - 각 블록 내 독립성 보장
    """
    
    def __init__(self, 
                 n_splits: int = 5,
                 block_size: int = None,
                 separation_size: int = None):
        """
        Args:
            n_splits: 분할 수
            block_size: 블록 크기
            separation_size: 블록 간 분리 크기
        """
        super().__init__(n_splits)
        self.block_size = block_size
        self.separation_size = separation_size or 0
    
    def split(self, X, y=None, groups=None) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        블록킹 분할
        
        Yields:
            (train_indices, test_indices) 튜플
        """
        n_samples = len(X)
        
        # 기본 블록 크기 설정
        if self.block_size is None:
            self.block_size = max(50, n_samples // (self.n_splits * 3))
        
        # 사용 가능한 총 길이 (분리 고려)
        total_used = self.n_splits * self.block_size + (self.n_splits - 1) * self.separation_size
        
        if total_used > n_samples:
            # 블록 크기 자동 조정
            self.block_size = (n_samples - (self.n_splits - 1) * self.separation_size) // self.n_splits
        
        for i in range(self.n_splits):
            # 테스트 블록 위치
            test_start = i * (self.block_size + self.separation_size)
            test_end = test_start + self.block_size
            
            if test_end > n_samples:
                break
            
            test_indices = np.arange(test_start, test_end)
            
            # 훈련 블록들 (테스트 블록 제외)
            train_indices = []
            
            for j in range(self.n_splits):
                if j != i:
                    block_start = j * (self.block_size + self.separation_size)
                    block_end = block_start + self.block_size
                    
                    if block_end <= n_samples:
                        train_indices.extend(range(block_start, block_end))
            
            train_indices = np.array(train_indices)
            
            if len(train_indices) > 0:
                yield train_indices, test_indices

class OutOfSampleTestProtocol:
    """
    샘플 외 테스트 프로토콜
    - 완전히 미래 데이터로 최종 검증
    - 시간적 일관성 보장
    """
    
    def __init__(self, 
                 holdout_ratio: float = 0.2,
                 validation_ratio: float = 0.2,
                 purge_gap: timedelta = None):
        """
        Args:
            holdout_ratio: 홀드아웃 비율
            validation_ratio: 검증 세트 비율
            purge_gap: 퍼지 간격
        """
        self.holdout_ratio = holdout_ratio
        self.validation_ratio = validation_ratio
        self.purge_gap = purge_gap or timedelta(hours=1)
    
    def split_temporal_holdout(self, X: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        시간적 홀드아웃 분할
        
        Args:
            X: 전체 데이터
            
        Returns:
            (train, validation, test) 데이터프레임 튜플
        """
        n_samples = len(X)
        
        # 분할 지점 계산
        test_size = int(n_samples * self.holdout_ratio)
        val_size = int(n_samples * self.validation_ratio)
        train_size = n_samples - test_size - val_size
        
        # 시간순 분할
        train_data = X.iloc[:train_size]
        val_data = X.iloc[train_size:train_size + val_size]
        test_data = X.iloc[train_size + val_size:]
        
        return train_data, val_data, test_data

class AdvancedCrossValidationSystem:
    """
    🔬 고급 시계열 교차 검증 시스템
    
    주요 기능:
    1. 다양한 시계열 교차 검증 전략
    2. 데이터 유출 방지 메커니즘
    3. 성능 메트릭 종합 평가
    4. 검증 결과 시각화 및 분석
    """
    
    def __init__(self):
        """시스템 초기화"""
        self.logger = logging.getLogger(__name__)
        
        # 검증 전략들
        self.validators = {
            'walk_forward': WalkForwardValidation(n_splits=5),
            'purged_group': PurgedGroupTimeSeriesSplit(n_splits=5),
            'blocking': BlockingTimeSeriesCV(n_splits=5),
            'standard_ts': TimeSeriesSplit(n_splits=5)
        }
        
        # 결과 저장소
        self.validation_results = {}
        self.oos_protocol = OutOfSampleTestProtocol()
        
        self.logger.info("🔬 고급 교차 검증 시스템 초기화 완료")
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        예측 성능 메트릭 계산
        
        Args:
            y_true: 실제값
            y_pred: 예측값
            
        Returns:
            메트릭 딕셔너리
        """
        metrics = {}
        
        try:
            metrics['mse'] = mean_squared_error(y_true, y_pred)
            metrics['mae'] = mean_absolute_error(y_true, y_pred)
            metrics['rmse'] = np.sqrt(metrics['mse'])
            metrics['r2'] = r2_score(y_true, y_pred)
            
            # 방향성 정확도 (금융에서 중요)
            y_true_diff = np.diff(y_true)
            y_pred_diff = np.diff(y_pred)
            direction_accuracy = np.mean(np.sign(y_true_diff) == np.sign(y_pred_diff))
            metrics['direction_accuracy'] = direction_accuracy
            
            # 최대 오차
            metrics['max_error'] = np.max(np.abs(y_true - y_pred))
            
            # 평균 절대 백분율 오차 (MAPE)
            mask = y_true != 0
            if np.sum(mask) > 0:
                mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
                metrics['mape'] = mape
            else:
                metrics['mape'] = float('inf')
            
        except Exception as e:
            self.logger.warning(f"메트릭 계산 오류: {e}")
            metrics = {'error': str(e)}
        
        return metrics
    
    def run_cross_validation(self, 
                           X: pd.DataFrame, 
                           y: pd.Series,
                           model: BaseEstimator,
                           cv_method: str = 'walk_forward',
                           **cv_params) -> ValidationResult:
        """
        교차 검증 실행
        
        Args:
            X: 특성 데이터
            y: 타겟 데이터
            model: 모델 객체
            cv_method: 교차 검증 방법
            **cv_params: 교차 검증 파라미터
            
        Returns:
            검증 결과
        """
        self.logger.info(f"🔬 {cv_method} 교차 검증 실행...")
        
        # 검증자 선택
        if cv_method not in self.validators:
            raise ValueError(f"지원되지 않는 검증 방법: {cv_method}")
        
        validator = self.validators[cv_method]
        
        # 파라미터 업데이트
        if cv_params:
            for param, value in cv_params.items():
                if hasattr(validator, param):
                    setattr(validator, param, value)
        
        # 교차 검증 실행
        train_scores = []
        test_scores = []
        train_indices_list = []
        test_indices_list = []
        fold_metadata = []
        
        fold_num = 0
        for train_idx, test_idx in validator.split(X, y):
            fold_num += 1
            
            # 데이터 분할
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            try:
                # 모델 훈련
                model_copy = pickle.loads(pickle.dumps(model))  # 딥 카피
                model_copy.fit(X_train, y_train)
                
                # 예측
                y_train_pred = model_copy.predict(X_train)
                y_test_pred = model_copy.predict(X_test)
                
                # 메트릭 계산
                train_metrics = self.calculate_metrics(y_train.values, y_train_pred)
                test_metrics = self.calculate_metrics(y_test.values, y_test_pred)
                
                train_scores.append(train_metrics)
                test_scores.append(test_metrics)
                train_indices_list.append(train_idx)
                test_indices_list.append(test_idx)
                
                # 폴드 메타데이터
                metadata = {
                    'fold': fold_num,
                    'train_period': (X_train.index[0], X_train.index[-1]),
                    'test_period': (X_test.index[0], X_test.index[-1]),
                    'train_size': len(train_idx),
                    'test_size': len(test_idx)
                }
                fold_metadata.append(metadata)
                
                self.logger.info(f"폴드 {fold_num}: 훈련 R² = {train_metrics.get('r2', 0):.3f}, "
                               f"테스트 R² = {test_metrics.get('r2', 0):.3f}")
                
            except Exception as e:
                self.logger.error(f"폴드 {fold_num} 실행 오류: {e}")
                continue
        
        # 전체 메트릭 계산
        overall_metrics = self._calculate_overall_metrics(train_scores, test_scores)
        
        # 결과 객체 생성
        result = ValidationResult(
            train_scores=train_scores,
            test_scores=test_scores,
            train_indices=train_indices_list,
            test_indices=test_indices_list,
            fold_metadata=fold_metadata,
            overall_metrics=overall_metrics,
            timestamp=datetime.now()
        )
        
        self.validation_results[cv_method] = result
        self.logger.info(f"✅ {cv_method} 교차 검증 완료")
        
        return result
    
    def _calculate_overall_metrics(self, 
                                 train_scores: List[Dict], 
                                 test_scores: List[Dict]) -> Dict[str, float]:
        """전체 메트릭 계산"""
        overall = {}
        
        if not train_scores or not test_scores:
            return overall
        
        # 메트릭 이름 추출
        metric_names = set()
        for score_dict in train_scores + test_scores:
            metric_names.update(score_dict.keys())
        
        metric_names.discard('error')  # 오류 제외
        
        for metric in metric_names:
            # 훈련 메트릭
            train_values = [s.get(metric, np.nan) for s in train_scores]
            train_values = [v for v in train_values if not np.isnan(v) and v != float('inf')]
            
            if train_values:
                overall[f'train_{metric}_mean'] = np.mean(train_values)
                overall[f'train_{metric}_std'] = np.std(train_values)
            
            # 테스트 메트릭
            test_values = [s.get(metric, np.nan) for s in test_scores]
            test_values = [v for v in test_values if not np.isnan(v) and v != float('inf')]
            
            if test_values:
                overall[f'test_{metric}_mean'] = np.mean(test_values)
                overall[f'test_{metric}_std'] = np.std(test_values)
                
                # 오버피팅 감지
                if train_values:
                    if metric in ['r2', 'direction_accuracy']:
                        # 높을수록 좋은 메트릭
                        overall[f'{metric}_overfitting'] = np.mean(train_values) - np.mean(test_values)
                    else:
                        # 낮을수록 좋은 메트릭
                        overall[f'{metric}_overfitting'] = np.mean(test_values) - np.mean(train_values)
        
        return overall
    
    def compare_validation_methods(self, 
                                 X: pd.DataFrame, 
                                 y: pd.Series,
                                 model: BaseEstimator) -> Dict[str, ValidationResult]:
        """
        여러 검증 방법 비교 실행
        
        Args:
            X: 특성 데이터
            y: 타겟 데이터
            model: 모델 객체
            
        Returns:
            방법별 검증 결과
        """
        self.logger.info("🔬 여러 검증 방법 비교 실행...")
        
        results = {}
        
        for method_name in self.validators.keys():
            try:
                self.logger.info(f"📊 {method_name} 실행 중...")
                result = self.run_cross_validation(X, y, model, method_name)
                results[method_name] = result
            except Exception as e:
                self.logger.error(f"{method_name} 실행 오류: {e}")
                continue
        
        self.logger.info(f"✅ {len(results)}개 방법 비교 완료")
        return results
    
    def out_of_sample_test(self, 
                         X: pd.DataFrame, 
                         y: pd.Series,
                         model: BaseEstimator) -> Dict[str, float]:
        """
        샘플 외 테스트 실행
        
        Args:
            X: 전체 특성 데이터
            y: 전체 타겟 데이터
            model: 모델 객체
            
        Returns:
            OOS 테스트 결과
        """
        self.logger.info("🎯 샘플 외 테스트 실행...")
        
        # 데이터 분할
        combined_data = pd.concat([X, y], axis=1)
        train_data, val_data, test_data = self.oos_protocol.split_temporal_holdout(combined_data)
        
        # 분할 후 X, y 재구성
        X_train = train_data.iloc[:, :-1]
        y_train = train_data.iloc[:, -1]
        X_val = val_data.iloc[:, :-1]
        y_val = val_data.iloc[:, -1]
        X_test = test_data.iloc[:, :-1]
        y_test = test_data.iloc[:, -1]
        
        # 모델 훈련 (훈련 + 검증 데이터)
        X_train_val = pd.concat([X_train, X_val])
        y_train_val = pd.concat([y_train, y_val])
        
        model.fit(X_train_val, y_train_val)
        
        # OOS 예측
        y_test_pred = model.predict(X_test)
        
        # 메트릭 계산
        oos_metrics = self.calculate_metrics(y_test.values, y_test_pred)
        
        self.logger.info(f"✅ OOS 테스트 완료 - R²: {oos_metrics.get('r2', 0):.3f}")
        
        return oos_metrics
    
    def visualize_cv_results(self, 
                           results: Dict[str, ValidationResult],
                           save_path: str = None) -> str:
        """
        교차 검증 결과 시각화
        
        Args:
            results: 검증 결과들
            save_path: 저장 경로
            
        Returns:
            HTML 차트 문자열
        """
        if not results:
            return "<p>시각화할 결과가 없습니다.</p>"
        
        # Plotly 서브플롯 생성
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('R² 점수', '방향성 정확도', 'RMSE', 'MAE'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        
        for i, (method, result) in enumerate(results.items()):
            color = colors[i % len(colors)]
            
            # R² 점수
            test_r2_scores = [s.get('r2', 0) for s in result.test_scores]
            fig.add_trace(
                go.Box(y=test_r2_scores, name=f'{method} R²', 
                      marker_color=color, boxpoints='all'),
                row=1, col=1
            )
            
            # 방향성 정확도
            direction_scores = [s.get('direction_accuracy', 0) for s in result.test_scores]
            fig.add_trace(
                go.Box(y=direction_scores, name=f'{method} 방향성', 
                      marker_color=color, boxpoints='all'),
                row=1, col=2
            )
            
            # RMSE
            rmse_scores = [s.get('rmse', 0) for s in result.test_scores]
            fig.add_trace(
                go.Box(y=rmse_scores, name=f'{method} RMSE', 
                      marker_color=color, boxpoints='all'),
                row=2, col=1
            )
            
            # MAE
            mae_scores = [s.get('mae', 0) for s in result.test_scores]
            fig.add_trace(
                go.Box(y=mae_scores, name=f'{method} MAE', 
                      marker_color=color, boxpoints='all'),
                row=2, col=2
            )
        
        fig.update_layout(
            title="교차 검증 방법 비교",
            height=800,
            showlegend=True
        )
        
        # HTML 생성
        html_str = fig.to_html(include_plotlyjs='cdn')
        
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(html_str)
            self.logger.info(f"시각화 결과 저장: {save_path}")
        
        return html_str
    
    def generate_validation_report(self, 
                                 results: Dict[str, ValidationResult],
                                 oos_results: Dict[str, float] = None) -> str:
        """
        검증 보고서 생성
        
        Args:
            results: 검증 결과들
            oos_results: OOS 테스트 결과
            
        Returns:
            HTML 보고서
        """
        html_report = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>고급 교차 검증 보고서</title>
            <meta charset="utf-8">
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .header { background: #2c3e50; color: white; padding: 20px; border-radius: 5px; }
                .section { margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }
                .metric { display: inline-block; margin: 10px; padding: 10px; background: #f8f9fa; border-radius: 3px; }
                .good { background: #d4edda; color: #155724; }
                .medium { background: #fff3cd; color: #856404; }
                .poor { background: #f8d7da; color: #721c24; }
                table { width: 100%; border-collapse: collapse; margin: 10px 0; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background: #f2f2f2; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🔬 고급 교차 검증 시스템 보고서</h1>
                <p>생성 시간: {timestamp}</p>
            </div>
            
            <div class="section">
                <h2>📊 검증 방법 비교 요약</h2>
                <table>
                    <tr>
                        <th>검증 방법</th>
                        <th>평균 R²</th>
                        <th>평균 RMSE</th>
                        <th>방향성 정확도</th>
                        <th>오버피팅 정도</th>
                        <th>안정성</th>
                    </tr>
                    {method_comparison_rows}
                </table>
            </div>
            
            <div class="section">
                <h2>🎯 샘플 외 테스트 결과</h2>
                {oos_results_section}
            </div>
            
            <div class="section">
                <h2>📈 상세 분석</h2>
                {detailed_analysis}
            </div>
            
            <div class="section">
                <h2>✅ 권장사항</h2>
                <ul>
                    {recommendations}
                </ul>
            </div>
        </body>
        </html>
        """.format(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            method_comparison_rows=self._generate_method_comparison_rows(results),
            oos_results_section=self._generate_oos_section(oos_results),
            detailed_analysis=self._generate_detailed_analysis(results),
            recommendations=self._generate_recommendations(results, oos_results)
        )
        
        return html_report
    
    def _generate_method_comparison_rows(self, results: Dict[str, ValidationResult]) -> str:
        """방법 비교 테이블 행 생성"""
        rows = []
        
        for method, result in results.items():
            overall = result.overall_metrics
            
            r2_mean = overall.get('test_r2_mean', 0)
            rmse_mean = overall.get('test_rmse_mean', 0)
            direction_mean = overall.get('test_direction_accuracy_mean', 0)
            overfitting = overall.get('r2_overfitting', 0)
            r2_std = overall.get('test_r2_std', 0)
            
            row = f"""
            <tr>
                <td>{method}</td>
                <td>{r2_mean:.3f}</td>
                <td>{rmse_mean:.2f}</td>
                <td>{direction_mean:.3f}</td>
                <td>{overfitting:.3f}</td>
                <td>{r2_std:.3f}</td>
            </tr>
            """
            rows.append(row)
        
        return "".join(rows)
    
    def _generate_oos_section(self, oos_results: Dict[str, float] = None) -> str:
        """OOS 결과 섹션 생성"""
        if not oos_results:
            return "<p>OOS 테스트 결과가 없습니다.</p>"
        
        r2 = oos_results.get('r2', 0)
        rmse = oos_results.get('rmse', 0)
        direction = oos_results.get('direction_accuracy', 0)
        mae = oos_results.get('mae', 0)
        
        quality_class = 'good' if r2 > 0.7 else 'medium' if r2 > 0.5 else 'poor'
        
        return f"""
        <div class="metric {quality_class}">R²: {r2:.3f}</div>
        <div class="metric">RMSE: {rmse:.2f}</div>
        <div class="metric">MAE: {mae:.2f}</div>
        <div class="metric">방향성 정확도: {direction:.3f}</div>
        """
    
    def _generate_detailed_analysis(self, results: Dict[str, ValidationResult]) -> str:
        """상세 분석 섹션 생성"""
        analysis = []
        
        if not results:
            return "<p>분석할 결과가 없습니다.</p>"
        
        best_method = max(results.keys(), 
                         key=lambda k: results[k].overall_metrics.get('test_r2_mean', 0))
        
        best_r2 = results[best_method].overall_metrics.get('test_r2_mean', 0)
        
        analysis.append(f"<p><strong>최고 성능 방법:</strong> {best_method} (R² = {best_r2:.3f})</p>")
        
        # 안정성 분석
        stability_analysis = []
        for method, result in results.items():
            std = result.overall_metrics.get('test_r2_std', float('inf'))
            stability_analysis.append((method, std))
        
        most_stable = min(stability_analysis, key=lambda x: x[1])
        analysis.append(f"<p><strong>가장 안정적:</strong> {most_stable[0]} (표준편차 = {most_stable[1]:.3f})</p>")
        
        return "".join(analysis)
    
    def _generate_recommendations(self, 
                                results: Dict[str, ValidationResult],
                                oos_results: Dict[str, float] = None) -> str:
        """권장사항 생성"""
        recommendations = []
        
        if not results:
            return "<li>분석할 결과가 없습니다.</li>"
        
        # 성능 기반 권장사항
        best_method = max(results.keys(), 
                         key=lambda k: results[k].overall_metrics.get('test_r2_mean', 0))
        
        recommendations.append(f"<li><strong>성능 중심:</strong> {best_method} 방법 사용 권장</li>")
        
        # 안정성 기반 권장사항
        stability_analysis = [(method, result.overall_metrics.get('test_r2_std', float('inf'))) 
                            for method, result in results.items()]
        most_stable = min(stability_analysis, key=lambda x: x[1])
        
        recommendations.append(f"<li><strong>안정성 중심:</strong> {most_stable[0]} 방법 고려</li>")
        
        # 일반적인 권장사항
        recommendations.append("<li>실제 운용에서는 워크포워드 검증이 가장 현실적</li>")
        recommendations.append("<li>데이터 유출 방지를 위해 퍼지드 방법 고려</li>")
        recommendations.append("<li>모델 복잡도에 따라 적절한 검증 방법 선택</li>")
        
        return "".join(recommendations)
    
    def save_results(self, output_dir: str = "cv_results") -> None:
        """결과 저장"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 검증 결과 저장
        results_file = os.path.join(output_dir, "validation_results.json")
        serializable_results = {}
        
        for method, result in self.validation_results.items():
            serializable_results[method] = {
                'overall_metrics': result.overall_metrics,
                'fold_metadata': result.fold_metadata,
                'timestamp': result.timestamp.isoformat()
            }
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False, default=str)
        
        self.logger.info(f"✅ 결과 저장 완료: {output_dir}")


def main():
    """메인 실행 함수"""
    print("🔬 고급 교차 검증 시스템 시작")
    
    # 시스템 초기화
    cv_system = AdvancedCrossValidationSystem()
    
    # 예제 데이터 생성
    print("\n📊 예제 데이터 생성...")
    np.random.seed(42)
    
    # 시계열 특성을 가진 데이터 생성
    n_samples = 1000
    n_features = 5
    
    # 특성 데이터 (자기상관 있는 시계열)
    X_data = []
    for i in range(n_features):
        series = np.cumsum(np.random.randn(n_samples) * 0.1)
        X_data.append(series)
    
    X = pd.DataFrame(np.column_stack(X_data), 
                    columns=[f'feature_{i}' for i in range(n_features)])
    X.index = pd.date_range('2024-01-01', periods=n_samples, freq='H')
    
    # 타겟 데이터 (특성들과 약간의 관계)
    y = (X.sum(axis=1) + np.random.randn(n_samples) * 0.5)
    
    print(f"데이터 크기: X {X.shape}, y {y.shape}")
    
    # 모델 정의
    model = RandomForestRegressor(n_estimators=10, random_state=42)  # 테스트용 작은 모델
    
    # 여러 검증 방법 비교
    print("\n🔬 여러 검증 방법 비교...")
    comparison_results = cv_system.compare_validation_methods(X, y, model)
    
    # OOS 테스트
    print("\n🎯 샘플 외 테스트...")
    oos_results = cv_system.out_of_sample_test(X, y, model)
    
    # 결과 시각화
    print("\n📈 결과 시각화...")
    chart_html = cv_system.visualize_cv_results(comparison_results, "cv_comparison_chart.html")
    
    # 보고서 생성
    print("\n📋 보고서 생성...")
    report_html = cv_system.generate_validation_report(comparison_results, oos_results)
    
    with open("advanced_cv_report.html", "w", encoding="utf-8") as f:
        f.write(report_html)
    
    # 결과 저장
    print("\n💾 결과 저장...")
    cv_system.save_results()
    
    print("\n✅ 고급 교차 검증 시스템 완료!")
    print("📋 상세 보고서: advanced_cv_report.html")
    print("📊 비교 차트: cv_comparison_chart.html")


if __name__ == "__main__":
    main()