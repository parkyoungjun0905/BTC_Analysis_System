#!/usr/bin/env python3
"""
⚡ 고도화된 특성 최적화 시스템
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎯 핵심 기능:
• 특성 중요도 자동 학습 및 순위
• 실시간 특성 선택 최적화
• 다중 모델 기반 특성 평가
• 메모리 효율적 계산
• 병렬 처리 가속화

🔧 최적화 기법:
• Mutual Information 기반 선택
• Recursive Feature Elimination
• SHAP 값 기반 중요도
• 상관관계 제거
• 시계열 안정성 검증
"""

import asyncio
import pandas as pd
import numpy as np
import json
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import logging
import warnings
from concurrent.futures import ProcessPoolExecutor
import joblib

# 기계학습 라이브러리
try:
    from sklearn.feature_selection import (
        SelectKBest, SelectPercentile, RFE, RFECV, 
        f_regression, mutual_info_regression, chi2
    )
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler, RobustScaler
    from sklearn.decomposition import PCA, FastICA
    from sklearn.manifold import TSNE
    from sklearn.model_selection import cross_val_score, TimeSeriesSplit
    from sklearn.metrics import mean_squared_error, r2_score
    import shap
    SKLEARN_AVAILABLE = True
except ImportError:
    print("⚠️ scikit-learn 또는 shap 미설치")
    SKLEARN_AVAILABLE = False

warnings.filterwarnings('ignore')

class AdvancedFeatureOptimizer:
    """고도화된 특성 최적화 시스템"""
    
    def __init__(self, n_features_target: int = 1000):
        self.n_features_target = n_features_target
        self.logger = self._setup_logger()
        self.db_path = "feature_optimization.db"
        
        # 모델 앙상블
        self.models = {
            'rf': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            'gbm': GradientBoostingRegressor(n_estimators=100, random_state=42),
        } if SKLEARN_AVAILABLE else {}
        
        self.feature_scores_cache = {}
        self.stability_cache = {}
        
        self._init_database()
    
    def _setup_logger(self) -> logging.Logger:
        """로깅 설정"""
        logger = logging.getLogger(f"{__name__}.FeatureOptimizer")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _init_database(self):
        """최적화 결과 데이터베이스 초기화"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 특성 점수 테이블
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS feature_scores (
            feature_name TEXT PRIMARY KEY,
            mutual_info_score REAL,
            f_score REAL,
            shap_importance REAL,
            stability_score REAL,
            correlation_penalty REAL,
            final_score REAL,
            rank INTEGER,
            last_updated TIMESTAMP
        )
        ''')
        
        # 최적화 히스토리
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS optimization_history (
            timestamp TIMESTAMP,
            n_features_before INTEGER,
            n_features_after INTEGER,
            optimization_method TEXT,
            performance_score REAL,
            execution_time REAL
        )
        ''')
        
        conn.commit()
        conn.close()
    
    async def optimize_features(self, features_df: pd.DataFrame, 
                              target: Optional[np.ndarray] = None,
                              method: str = 'comprehensive') -> pd.DataFrame:
        """포괄적 특성 최적화"""
        
        self.logger.info(f"🚀 특성 최적화 시작: {len(features_df.columns)}개 → {self.n_features_target}개 목표")
        start_time = datetime.now()
        
        # 1단계: 기본 정리 (NaN, 상수, 중복)
        features_clean = await self._basic_cleanup(features_df)
        self.logger.info(f"✅ 기본 정리: {len(features_clean.columns)}개 특성 유지")
        
        # 2단계: 목표 변수 생성 (제공되지 않은 경우)
        if target is None:
            target = await self._generate_synthetic_target(features_clean)
        
        # 3단계: 다중 기법 적용
        if method == 'comprehensive':
            optimized_features = await self._comprehensive_optimization(features_clean, target)
        elif method == 'fast':
            optimized_features = await self._fast_optimization(features_clean, target)
        elif method == 'stability_focused':
            optimized_features = await self._stability_optimization(features_clean, target)
        else:
            optimized_features = await self._mutual_info_optimization(features_clean, target)
        
        # 4단계: 최종 검증
        final_features = await self._final_validation(optimized_features, target)
        
        # 결과 저장
        execution_time = (datetime.now() - start_time).total_seconds()
        await self._save_optimization_result(
            len(features_df.columns), len(final_features.columns),
            method, execution_time
        )
        
        self.logger.info(f"🎯 최적화 완료: {len(final_features.columns)}개 특성, {execution_time:.2f}초")
        
        return final_features
    
    async def _basic_cleanup(self, df: pd.DataFrame) -> pd.DataFrame:
        """기본 정리: NaN, 상수, 중복 제거"""
        
        # 1. NaN 비율이 높은 특성 제거 (50% 이상)
        nan_threshold = 0.5
        nan_ratios = df.isnull().sum() / len(df)
        valid_features = nan_ratios[nan_ratios <= nan_threshold].index
        df = df[valid_features]
        
        # 2. 상수 특성 제거 (분산이 매우 낮은 특성)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            variances = df[numeric_cols].var()
            non_constant = variances[variances > 1e-8].index
            df = df[non_constant]
        
        # 3. 무한값 처리
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(df.median())
        
        # 4. 완전 중복 특성 제거
        df = df.T.drop_duplicates().T
        
        return df
    
    async def _generate_synthetic_target(self, features_df: pd.DataFrame) -> np.ndarray:
        """합성 목표 변수 생성 (실제로는 가격 변화율 사용)"""
        # 가격 관련 특성 찾기
        price_features = [col for col in features_df.columns 
                         if any(keyword in col.lower() for keyword in ['price', 'btc', 'close'])]
        
        if price_features:
            # 주요 가격 특성으로 목표 변수 생성
            main_price = features_df[price_features[0]].values
            # 간단한 노이즈가 있는 미래 가격 시뮬레이션
            target = main_price * (1 + np.random.normal(0, 0.02, len(main_price)))
            return (target - main_price) / main_price  # 변화율
        else:
            # 기본 노이즈 목표 변수
            return np.random.randn(len(features_df))
    
    async def _comprehensive_optimization(self, features_df: pd.DataFrame, 
                                        target: np.ndarray) -> pd.DataFrame:
        """포괄적 최적화 (모든 기법 적용)"""
        
        # 1. Mutual Information 기반 초기 선택
        features_mi = await self._select_by_mutual_info(features_df, target, top_k=2000)
        
        # 2. 상관관계 제거
        features_uncorr = await self._remove_high_correlation(features_mi, threshold=0.95)
        
        # 3. 안정성 기반 필터링
        features_stable = await self._filter_by_stability(features_uncorr, target)
        
        # 4. 모델 기반 중요도
        features_model = await self._model_based_selection(features_stable, target)
        
        # 5. SHAP 기반 최종 선택
        if SKLEARN_AVAILABLE and len(features_model.columns) > self.n_features_target:
            features_final = await self._shap_based_selection(features_model, target)
        else:
            features_final = features_model
        
        return features_final
    
    async def _fast_optimization(self, features_df: pd.DataFrame, 
                               target: np.ndarray) -> pd.DataFrame:
        """빠른 최적화 (계산량 최소화)"""
        
        # F-점수 기반 빠른 선택
        if SKLEARN_AVAILABLE:
            selector = SelectKBest(score_func=f_regression, k=min(self.n_features_target, len(features_df.columns)))
            
            try:
                features_selected = selector.fit_transform(features_df.fillna(0), target)
                selected_columns = selector.get_feature_names_out()
                return pd.DataFrame(features_selected, columns=selected_columns)
            except Exception as e:
                self.logger.warning(f"F-점수 선택 실패: {e}")
        
        # 대안: 분산 기반 선택
        variances = features_df.var()
        top_variance = variances.nlargest(self.n_features_target).index
        return features_df[top_variance]
    
    async def _stability_optimization(self, features_df: pd.DataFrame,
                                    target: np.ndarray) -> pd.DataFrame:
        """안정성 중심 최적화"""
        
        # 시계열 분할로 안정성 테스트
        stable_features = []
        
        if len(features_df) > 10:
            # 시계열 분할
            n_splits = min(5, len(features_df) // 2)
            split_size = len(features_df) // n_splits
            
            feature_stability = {}
            
            for feature in features_df.columns:
                stability_scores = []
                feature_values = features_df[feature].fillna(feature_values.median())
                
                # 각 분할에서 평균과 분산 계산
                for i in range(n_splits - 1):
                    start_idx = i * split_size
                    end_idx = (i + 1) * split_size
                    
                    segment1 = feature_values.iloc[start_idx:end_idx]
                    segment2 = feature_values.iloc[end_idx:end_idx + split_size]
                    
                    if len(segment1) > 0 and len(segment2) > 0:
                        # 평균의 안정성
                        mean_diff = abs(segment1.mean() - segment2.mean())
                        mean_stability = 1 / (1 + mean_diff)
                        
                        # 분산의 안정성  
                        std_ratio = min(segment1.std(), segment2.std()) / max(segment1.std(), segment2.std())
                        std_stability = std_ratio if not np.isnan(std_ratio) else 0
                        
                        combined_stability = (mean_stability + std_stability) / 2
                        stability_scores.append(combined_stability)
                
                feature_stability[feature] = np.mean(stability_scores) if stability_scores else 0
            
            # 상위 안정성 특성 선택
            sorted_features = sorted(feature_stability.items(), key=lambda x: x[1], reverse=True)
            top_features = [f[0] for f in sorted_features[:self.n_features_target]]
            
            return features_df[top_features]
        
        return features_df.iloc[:, :self.n_features_target]
    
    async def _mutual_info_optimization(self, features_df: pd.DataFrame,
                                      target: np.ndarray) -> pd.DataFrame:
        """상호 정보량 기반 최적화"""
        
        if not SKLEARN_AVAILABLE:
            return features_df.iloc[:, :self.n_features_target]
        
        try:
            # 상호 정보량 계산
            features_array = features_df.fillna(0).values
            
            # 배치 처리로 메모리 효율성 향상
            mi_scores = {}
            batch_size = 100
            
            for i in range(0, len(features_df.columns), batch_size):
                batch_end = min(i + batch_size, len(features_df.columns))
                batch_features = features_array[:, i:batch_end]
                batch_columns = features_df.columns[i:batch_end]
                
                batch_scores = mutual_info_regression(batch_features, target, random_state=42)
                
                for j, col in enumerate(batch_columns):
                    mi_scores[col] = batch_scores[j]
            
            # 상위 점수 특성 선택
            sorted_features = sorted(mi_scores.items(), key=lambda x: x[1], reverse=True)
            top_features = [f[0] for f in sorted_features[:self.n_features_target]]
            
            return features_df[top_features]
            
        except Exception as e:
            self.logger.error(f"상호 정보량 최적화 실패: {e}")
            return features_df.iloc[:, :self.n_features_target]
    
    async def _select_by_mutual_info(self, features_df: pd.DataFrame, 
                                   target: np.ndarray, top_k: int) -> pd.DataFrame:
        """상호 정보량 기반 특성 선택"""
        
        if not SKLEARN_AVAILABLE:
            return features_df.iloc[:, :top_k]
        
        try:
            selector = SelectKBest(score_func=mutual_info_regression, k=min(top_k, len(features_df.columns)))
            features_selected = selector.fit_transform(features_df.fillna(0), target)
            selected_columns = selector.get_feature_names_out()
            
            return pd.DataFrame(features_selected, columns=selected_columns, index=features_df.index)
            
        except Exception as e:
            self.logger.warning(f"상호 정보량 선택 실패: {e}")
            return features_df.iloc[:, :top_k]
    
    async def _remove_high_correlation(self, features_df: pd.DataFrame, 
                                     threshold: float = 0.95) -> pd.DataFrame:
        """높은 상관관계 특성 제거"""
        
        # 상관관계 행렬 계산
        corr_matrix = features_df.corr().abs()
        
        # 상삼각행렬에서 높은 상관관계 쌍 찾기
        upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        # 높은 상관관계를 가진 특성 제거
        to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > threshold)]
        
        features_filtered = features_df.drop(columns=to_drop)
        
        self.logger.info(f"🔄 상관관계 필터링: {len(to_drop)}개 특성 제거")
        
        return features_filtered
    
    async def _filter_by_stability(self, features_df: pd.DataFrame,
                                 target: np.ndarray) -> pd.DataFrame:
        """안정성 기반 필터링"""
        
        if len(features_df) < 20:
            return features_df
        
        stable_features = []
        
        for feature in features_df.columns:
            feature_values = features_df[feature].fillna(features_df[feature].median())
            
            # 이동 통계의 안정성 체크
            window_size = max(5, len(feature_values) // 10)
            rolling_mean = feature_values.rolling(window=window_size).mean()
            rolling_std = feature_values.rolling(window=window_size).std()
            
            # 안정성 점수 (변동계수의 역수)
            if rolling_std.mean() > 0 and rolling_mean.mean() > 0:
                cv = rolling_std.mean() / abs(rolling_mean.mean())
                stability_score = 1 / (1 + cv)
            else:
                stability_score = 0
            
            if stability_score > 0.3:  # 임계값
                stable_features.append(feature)
        
        if len(stable_features) > 0:
            return features_df[stable_features]
        else:
            return features_df
    
    async def _model_based_selection(self, features_df: pd.DataFrame,
                                   target: np.ndarray) -> pd.DataFrame:
        """모델 기반 특성 선택"""
        
        if not SKLEARN_AVAILABLE or len(features_df.columns) == 0:
            return features_df
        
        try:
            # Random Forest 중요도
            rf_model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
            rf_model.fit(features_df.fillna(0), target)
            
            # 중요도 점수
            importances = rf_model.feature_importances_
            feature_importance = pd.Series(importances, index=features_df.columns)
            
            # 상위 중요도 특성 선택
            n_select = min(self.n_features_target, len(features_df.columns))
            top_features = feature_importance.nlargest(n_select).index
            
            return features_df[top_features]
            
        except Exception as e:
            self.logger.warning(f"모델 기반 선택 실패: {e}")
            return features_df.iloc[:, :min(self.n_features_target, len(features_df.columns))]
    
    async def _shap_based_selection(self, features_df: pd.DataFrame,
                                  target: np.ndarray) -> pd.DataFrame:
        """SHAP 값 기반 특성 선택"""
        
        try:
            # SHAP explainer 생성 (Random Forest 기반)
            model = RandomForestRegressor(n_estimators=50, random_state=42)
            model.fit(features_df.fillna(0), target)
            
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(features_df.fillna(0))
            
            # SHAP 중요도 계산 (절댓값의 평균)
            shap_importance = pd.Series(
                np.abs(shap_values).mean(axis=0),
                index=features_df.columns
            )
            
            # 상위 SHAP 중요도 특성 선택
            top_features = shap_importance.nlargest(self.n_features_target).index
            
            return features_df[top_features]
            
        except Exception as e:
            self.logger.warning(f"SHAP 선택 실패: {e}")
            return features_df.iloc[:, :self.n_features_target]
    
    async def _final_validation(self, features_df: pd.DataFrame,
                              target: np.ndarray) -> pd.DataFrame:
        """최종 검증 및 품질 확인"""
        
        # 1. 최종 크기 조정
        if len(features_df.columns) > self.n_features_target:
            # 분산 기반 마지막 선택
            variances = features_df.var()
            top_variance = variances.nlargest(self.n_features_target).index
            features_df = features_df[top_variance]
        
        # 2. 데이터 품질 최종 확인
        final_features = features_df.copy()
        
        # NaN 처리
        final_features = final_features.fillna(final_features.median())
        
        # 무한값 처리
        final_features = final_features.replace([np.inf, -np.inf], 0)
        
        # 3. 특성 점수 저장
        await self._save_feature_scores(final_features, target)
        
        return final_features
    
    async def _save_feature_scores(self, features_df: pd.DataFrame, target: np.ndarray):
        """특성 점수 데이터베이스 저장"""
        
        if not SKLEARN_AVAILABLE:
            return
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 상호 정보량 계산
            mi_scores = mutual_info_regression(features_df.fillna(0), target, random_state=42)
            
            # F-점수 계산
            f_scores = f_regression(features_df.fillna(0), target)[0]
            
            for i, feature in enumerate(features_df.columns):
                cursor.execute('''
                INSERT OR REPLACE INTO feature_scores 
                (feature_name, mutual_info_score, f_score, final_score, rank, last_updated)
                VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    feature,
                    float(mi_scores[i]),
                    float(f_scores[i]),
                    float(mi_scores[i] + f_scores[i]),
                    i + 1,
                    datetime.now()
                ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"특성 점수 저장 실패: {e}")
    
    async def _save_optimization_result(self, n_before: int, n_after: int,
                                      method: str, execution_time: float):
        """최적화 결과 저장"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        INSERT INTO optimization_history 
        (timestamp, n_features_before, n_features_after, optimization_method, execution_time)
        VALUES (?, ?, ?, ?, ?)
        ''', (datetime.now(), n_before, n_after, method, execution_time))
        
        conn.commit()
        conn.close()
    
    def get_feature_ranking(self) -> pd.DataFrame:
        """특성 순위 조회"""
        
        conn = sqlite3.connect(self.db_path)
        
        query = '''
        SELECT feature_name, mutual_info_score, f_score, final_score, rank, last_updated
        FROM feature_scores 
        ORDER BY final_score DESC, rank ASC
        '''
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        return df
    
    def get_optimization_history(self) -> pd.DataFrame:
        """최적화 히스토리 조회"""
        
        conn = sqlite3.connect(self.db_path)
        
        query = '''
        SELECT * FROM optimization_history 
        ORDER BY timestamp DESC
        '''
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        return df

class RealTimeFeatureMonitor:
    """실시간 특성 모니터링 시스템"""
    
    def __init__(self, optimizer: AdvancedFeatureOptimizer):
        self.optimizer = optimizer
        self.monitoring_active = False
        self.performance_history = []
        
    async def start_monitoring(self, features_df: pd.DataFrame, 
                             target: np.ndarray, interval: int = 3600):
        """실시간 모니터링 시작 (1시간 간격)"""
        
        self.monitoring_active = True
        
        while self.monitoring_active:
            try:
                # 성능 평가
                performance = await self._evaluate_performance(features_df, target)
                
                # 히스토리 저장
                self.performance_history.append({
                    'timestamp': datetime.now(),
                    'performance': performance,
                    'n_features': len(features_df.columns)
                })
                
                print(f"📊 특성 성능 모니터링: {performance:.4f} (특성 {len(features_df.columns)}개)")
                
                # 성능이 크게 떨어지면 재최적화 제안
                if len(self.performance_history) > 5:
                    recent_avg = np.mean([h['performance'] for h in self.performance_history[-3:]])
                    older_avg = np.mean([h['performance'] for h in self.performance_history[-6:-3]])
                    
                    if recent_avg < older_avg * 0.9:  # 10% 이상 성능 저하
                        print("⚠️ 성능 저하 감지: 특성 재최적화 필요")
                
                await asyncio.sleep(interval)
                
            except Exception as e:
                print(f"❌ 모니터링 오류: {e}")
                await asyncio.sleep(60)
    
    async def _evaluate_performance(self, features_df: pd.DataFrame, 
                                  target: np.ndarray) -> float:
        """특성 성능 평가"""
        
        if not SKLEARN_AVAILABLE:
            return 0.5
        
        try:
            # 빠른 모델로 성능 평가
            model = RandomForestRegressor(n_estimators=20, random_state=42, n_jobs=-1)
            
            # 시계열 분할로 교차 검증
            tscv = TimeSeriesSplit(n_splits=3)
            scores = cross_val_score(model, features_df.fillna(0), target, cv=tscv, scoring='r2')
            
            return scores.mean()
            
        except Exception as e:
            print(f"성능 평가 오류: {e}")
            return 0.0
    
    def stop_monitoring(self):
        """모니터링 중지"""
        self.monitoring_active = False
    
    def get_performance_history(self) -> pd.DataFrame:
        """성능 히스토리 반환"""
        return pd.DataFrame(self.performance_history)

# 사용 예제
async def main():
    """메인 실행 함수"""
    print("⚡ 고도화된 특성 최적화 시스템 테스트")
    
    # 테스트 데이터 생성
    np.random.seed(42)
    n_samples = 1000
    n_features = 1500
    
    # 가상의 특성 데이터
    feature_names = [f"feature_{i}" for i in range(n_features)]
    features_df = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=feature_names
    )
    
    # 목표 변수 (일부 특성과 연관)
    important_features = np.random.choice(n_features, 50, replace=False)
    target = np.sum(features_df.iloc[:, important_features] * np.random.randn(50), axis=1) + np.random.randn(n_samples) * 0.1
    
    # 최적화 실행
    optimizer = AdvancedFeatureOptimizer(n_features_target=1000)
    
    print(f"\n🔍 최적화 전: {len(features_df.columns)}개 특성")
    
    # 다양한 최적화 방법 테스트
    methods = ['comprehensive', 'fast', 'stability_focused']
    
    for method in methods:
        print(f"\n🚀 {method} 최적화 실행...")
        
        optimized_features = await optimizer.optimize_features(
            features_df.copy(), 
            target, 
            method=method
        )
        
        print(f"✅ {method} 결과: {len(optimized_features.columns)}개 특성")
        
        # 성능 평가
        if SKLEARN_AVAILABLE:
            model = RandomForestRegressor(n_estimators=50, random_state=42)
            model.fit(optimized_features.fillna(0), target)
            score = model.score(optimized_features.fillna(0), target)
            print(f"📈 R² 점수: {score:.4f}")
    
    # 특성 순위 조회
    print("\n📊 최고 성능 특성 Top 10:")
    ranking = optimizer.get_feature_ranking()
    if len(ranking) > 0:
        print(ranking.head(10))
    
    # 최적화 히스토리
    print("\n📈 최적화 히스토리:")
    history = optimizer.get_optimization_history()
    print(history)

if __name__ == "__main__":
    # 실행
    asyncio.run(main())