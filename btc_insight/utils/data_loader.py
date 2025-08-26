#!/usr/bin/env python3
"""
📊 데이터 로더
- ai_optimized_3month_data 폴더의 1시간 단위 통합 데이터 로드
- 데이터 전처리 및 검증
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
from typing import Dict, List, Optional

from .logger import get_logger

class DataLoader:
    """BTC 분석용 데이터 로더"""
    
    def __init__(self, data_path: str):
        self.logger = get_logger(__name__)
        self.data_path = Path(data_path)
        self.data = None
        self.data_info = {}
        
        print(f"📊 데이터 로더 초기화: {self.data_path}")
        
    def load_data(self) -> bool:
        """
        3개월치 1시간 단위 통합 데이터 로드
        
        Returns:
            로드 성공 여부
        """
        print("📥 3개월치 통합 데이터 로딩 시작...")
        
        try:
            # ai_matrix_complete.csv 파일 로드
            csv_file = self.data_path / "ai_matrix_complete.csv"
            
            if not csv_file.exists():
                self.logger.error(f"데이터 파일 없음: {csv_file}")
                return False
            
            # 데이터 읽기
            self.data = pd.read_csv(csv_file)
            print(f"📊 원본 데이터 로드: {self.data.shape}")
            
            # 데이터 검증 및 전처리
            if not self._validate_and_preprocess():
                return False
            
            # 데이터 정보 수집
            self._collect_data_info()
            
            print(f"✅ 데이터 로드 완료:")
            print(f"   📏 크기: {self.data.shape[0]}행 x {self.data.shape[1]}열")
            print(f"   📅 기간: {self.data_info.get('time_range', 'N/A')}")
            print(f"   🔢 지표 수: {self.data_info.get('indicator_count', 0)}개")
            
            return True
            
        except Exception as e:
            self.logger.error(f"데이터 로드 오류: {e}")
            print(f"❌ 데이터 로드 실패: {e}")
            return False
    
    def _validate_and_preprocess(self) -> bool:
        """데이터 검증 및 전처리"""
        print("🔍 데이터 검증 및 전처리...")
        
        # 빈 데이터 체크
        if self.data.empty:
            self.logger.error("빈 데이터셋")
            return False
        
        # timestamp 컬럼 처리
        if 'timestamp' in self.data.columns:
            try:
                self.data['timestamp'] = pd.to_datetime(self.data['timestamp'])
                # 시간순 정렬
                self.data = self.data.sort_values('timestamp').reset_index(drop=True)
                print("✅ timestamp 처리 및 시간순 정렬 완료")
            except Exception as e:
                self.logger.warning(f"timestamp 처리 오류: {e}")
        
        # 숫자형 컬럼만 추출
        numeric_columns = self.data.select_dtypes(include=[np.number]).columns.tolist()
        
        # timestamp가 있으면 포함
        if 'timestamp' in self.data.columns:
            columns_to_keep = ['timestamp'] + numeric_columns
        else:
            columns_to_keep = numeric_columns
            
        self.data = self.data[columns_to_keep]
        
        # 결측치 처리
        missing_before = self.data.isnull().sum().sum()
        if missing_before > 0:
            print(f"⚠️ 결측치 발견: {missing_before}개")
            
            # 시계열 특성을 고려한 결측치 처리
            # 1. 전진 채움 (forward fill)
            self.data = self.data.fillna(method='ffill')
            
            # 2. 후진 채움 (backward fill)
            self.data = self.data.fillna(method='bfill')
            
            # 3. 여전히 남은 결측치는 0으로 처리
            self.data = self.data.fillna(0)
            
            missing_after = self.data.isnull().sum().sum()
            print(f"✅ 결측치 처리 완료: {missing_before} → {missing_after}")
        
        # 무한값 처리
        inf_count = np.isinf(self.data.select_dtypes(include=[np.number])).sum().sum()
        if inf_count > 0:
            print(f"⚠️ 무한값 발견: {inf_count}개")
            self.data = self.data.replace([np.inf, -np.inf], 0)
            print("✅ 무한값 처리 완료")
        
        # 데이터 타입 최적화
        self._optimize_data_types()
        
        print("✅ 데이터 검증 및 전처리 완료")
        return True
    
    def _optimize_data_types(self):
        """메모리 사용량 최적화를 위한 데이터 타입 변환"""
        numeric_columns = self.data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            if self.data[col].dtype == 'float64':
                # float32로 다운캐스트 시도
                if self.data[col].min() >= np.finfo(np.float32).min and \
                   self.data[col].max() <= np.finfo(np.float32).max:
                    self.data[col] = self.data[col].astype(np.float32)
            
            elif self.data[col].dtype == 'int64':
                # int32로 다운캐스트 시도
                if self.data[col].min() >= np.iinfo(np.int32).min and \
                   self.data[col].max() <= np.iinfo(np.int32).max:
                    self.data[col] = self.data[col].astype(np.int32)
    
    def _collect_data_info(self):
        """데이터 정보 수집"""
        self.data_info = {
            'shape': self.data.shape,
            'columns': list(self.data.columns),
            'numeric_columns': list(self.data.select_dtypes(include=[np.number]).columns),
            'indicator_count': len(self.data.select_dtypes(include=[np.number]).columns),
            'memory_usage_mb': self.data.memory_usage(deep=True).sum() / 1024 / 1024
        }
        
        # 시간 범위 정보
        if 'timestamp' in self.data.columns:
            self.data_info['time_range'] = f"{self.data['timestamp'].min()} ~ {self.data['timestamp'].max()}"
            self.data_info['time_span_hours'] = int((self.data['timestamp'].max() - self.data['timestamp'].min()).total_seconds() / 3600)
        
        # BTC 가격 컬럼 식별
        btc_price_col = self._identify_btc_price_column()
        if btc_price_col:
            self.data_info['btc_price_column'] = btc_price_col
            self.data_info['btc_price_range'] = {
                'min': float(self.data[btc_price_col].min()),
                'max': float(self.data[btc_price_col].max()),
                'current': float(self.data[btc_price_col].iloc[-1])
            }
    
    def _identify_btc_price_column(self) -> Optional[str]:
        """BTC 가격 컬럼 식별"""
        candidates = [
            'onchain_blockchain_info_network_stats_market_price_usd',
            'btc_price', 'price', 'close', 'market_price_usd'
        ]
        
        # 후보 컬럼 우선 검색
        for candidate in candidates:
            if candidate in self.data.columns:
                return candidate
        
        # 큰 수치를 가진 컬럼 찾기 (BTC 가격 특성)
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if self.data[col].mean() > 10000:  # BTC 가격 범위
                return col
                
        return None
    
    def get_data(self) -> pd.DataFrame:
        """전체 데이터 반환"""
        return self.data
    
    def get_latest_data(self, hours: int = 168) -> pd.DataFrame:
        """
        최근 N시간 데이터 반환
        
        Args:
            hours: 가져올 시간 (기본 168시간 = 1주일)
            
        Returns:
            최근 N시간 데이터
        """
        if self.data is None:
            self.logger.error("데이터가 로드되지 않음")
            return pd.DataFrame()
        
        return self.data.tail(hours).copy()
    
    def get_btc_price_series(self) -> pd.Series:
        """BTC 가격 시계열 반환"""
        btc_col = self._identify_btc_price_column()
        if btc_col and self.data is not None:
            return self.data[btc_col]
        else:
            return pd.Series()
    
    def get_data_info(self) -> Dict:
        """데이터 정보 반환"""
        return self.data_info.copy()
    
    def get_feature_columns(self) -> List[str]:
        """피처 컬럼 목록 반환 (BTC 가격 제외)"""
        if self.data is None:
            return []
        
        btc_col = self._identify_btc_price_column()
        all_numeric = self.data.select_dtypes(include=[np.number]).columns.tolist()
        
        if btc_col and btc_col in all_numeric:
            all_numeric.remove(btc_col)
        
        return all_numeric
    
    def validate_data_continuity(self) -> Dict:
        """데이터 연속성 검증"""
        if self.data is None or 'timestamp' not in self.data.columns:
            return {'continuity': False, 'reason': 'No timestamp data'}
        
        # 시간 간격 확인
        time_diffs = self.data['timestamp'].diff().dt.total_seconds() / 3600  # 시간 단위
        
        # 1시간 간격이 아닌 부분 찾기
        irregular_intervals = time_diffs[(time_diffs != 1.0) & (~time_diffs.isna())]
        
        continuity_info = {
            'continuity': len(irregular_intervals) == 0,
            'total_points': len(self.data),
            'irregular_intervals': len(irregular_intervals),
            'expected_hours': int((self.data['timestamp'].max() - self.data['timestamp'].min()).total_seconds() / 3600),
            'actual_hours': len(self.data),
            'data_completeness': len(self.data) / int((self.data['timestamp'].max() - self.data['timestamp'].min()).total_seconds() / 3600) * 100
        }
        
        return continuity_info
    
    def get_data_quality_report(self) -> Dict:
        """데이터 품질 리포트 생성"""
        if self.data is None:
            return {'status': 'No data loaded'}
        
        numeric_data = self.data.select_dtypes(include=[np.number])
        
        quality_report = {
            'data_shape': self.data.shape,
            'missing_values': self.data.isnull().sum().sum(),
            'duplicate_rows': self.data.duplicated().sum(),
            'numeric_columns': len(numeric_data.columns),
            'zero_variance_columns': (numeric_data.var() == 0).sum(),
            'high_correlation_pairs': self._find_high_correlations(numeric_data),
            'data_continuity': self.validate_data_continuity(),
            'memory_usage_mb': round(self.data.memory_usage(deep=True).sum() / 1024 / 1024, 2)
        }
        
        return quality_report
    
    def _find_high_correlations(self, numeric_data: pd.DataFrame, threshold: float = 0.95) -> List[tuple]:
        """높은 상관관계를 가진 컬럼 쌍 찾기"""
        if numeric_data.shape[1] < 2:
            return []
        
        corr_matrix = numeric_data.corr().abs()
        
        # 상삼각 행렬만 고려 (중복 제거)
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # 높은 상관관계 쌍 찾기
        high_corr_pairs = []
        for col1 in upper_triangle.columns:
            for col2 in upper_triangle.index:
                corr_val = upper_triangle.loc[col2, col1]
                if not np.isnan(corr_val) and corr_val > threshold:
                    high_corr_pairs.append((col1, col2, round(corr_val, 3)))
        
        return high_corr_pairs[:10]  # 상위 10개만 반환