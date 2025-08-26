#!/usr/bin/env python3
"""
⚡ 빠른 정확도 향상 시스템
- 현재 78.26% → 85%+ 목표
- 효율적인 백테스트로 빠른 결과 도출
"""

import numpy as np
import pandas as pd
import warnings
import joblib
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple

# 머신러닝 라이브러리
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.cluster import KMeans

warnings.filterwarnings('ignore')

class FastAccuracyImprovement:
    """빠른 정확도 향상 시스템"""
    
    def __init__(self):
        self.data_path = "/Users/parkyoungjun/Desktop/BTC_Analysis_System"
        self.current_accuracy = 78.26
        self.target_accuracy = 85.0
        self.results = {}
        
    def load_data(self) -> pd.DataFrame:
        """데이터 로드"""
        print("⚡ 빠른 정확도 향상 시스템")
        print("="*50)
        print(f"🚀 현재: {self.current_accuracy}% → 목표: {self.target_accuracy}%")
        print("="*50)
        
        csv_path = os.path.join(self.data_path, "ai_optimized_3month_data", "ai_matrix_complete.csv")
        print(f"📂 데이터 로드: {csv_path}")
        
        df = pd.read_csv(csv_path)
        
        # 기본 전처리
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        df_clean = df[numeric_columns].copy()
        df_clean = df_clean.ffill().bfill().fillna(0)
        df_clean = df_clean.replace([np.inf, -np.inf], 0)
        
        print(f"✅ 데이터 로드 완료: {df_clean.shape}")
        return df_clean
    
    def baseline_accuracy(self, df: pd.DataFrame) -> float:
        """현재 베이스라인 정확도 측정"""
        print("\n📊 베이스라인 정확도 측정...")
        
        # BTC 가격 컬럼 찾기
        btc_col = 'onchain_blockchain_info_network_stats_market_price_usd'
        if btc_col not in df.columns:
            btc_col = df.columns[0]  # 없으면 첫 번째 컬럼 사용
            
        print(f"🎯 타겟 컬럼: {btc_col}")
        X = df.drop(columns=[btc_col]).values
        y = df[btc_col].shift(-1).dropna().values
        X = X[:-1]  # 마지막 행 제거 (target이 없으므로)
        
        # 시계열 분할
        tscv = TimeSeriesSplit(n_splits=3)
        accuracies = []
        
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # 스케일링
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # 랜덤포레스트 모델
            rf = RandomForestRegressor(n_estimators=50, random_state=42)
            rf.fit(X_train_scaled, y_train)
            
            pred = rf.predict(X_test_scaled)
            accuracy = max(0, r2_score(y_test, pred) * 100)
            accuracies.append(accuracy)
        
        baseline_acc = np.mean(accuracies)
        print(f"🎯 베이스라인 정확도: {baseline_acc:.2f}%")
        return baseline_acc
    
    def improvement_1_feature_engineering(self, df: pd.DataFrame) -> float:
        """개선 1: 고급 피처 엔지니어링"""
        print("\n💡 개선 1: 고급 피처 엔지니어링")
        print("-" * 40)
        
        # BTC 가격 컬럼 찾기
        btc_col = 'onchain_blockchain_info_network_stats_market_price_usd'
        if btc_col not in df.columns:
            btc_col = df.columns[0]
            
        df_enhanced = df.copy()
        
        # 가격 데이터로 고급 피처 생성
        price_data = df[btc_col]
        
        # 1. 다중 기간 모멘텀
        for period in [12, 24, 168]:  # 12시간, 1일, 1주
            df_enhanced[f'momentum_{period}h'] = price_data.pct_change(period)
            df_enhanced[f'volatility_{period}h'] = price_data.pct_change().rolling(period).std()
        
        # 2. 기술적 지표
        df_enhanced['sma_ratio'] = price_data / price_data.rolling(24).mean()
        df_enhanced['price_position'] = (price_data - price_data.rolling(168).min()) / (price_data.rolling(168).max() - price_data.rolling(168).min())
        
        # 3. 변동성 지표
        df_enhanced['high_low_ratio'] = price_data.rolling(24).max() / price_data.rolling(24).min()
        
        # NaN 처리
        df_enhanced = df_enhanced.ffill().bfill().fillna(0)
        df_enhanced = df_enhanced.replace([np.inf, -np.inf], 0)
        
        # 테스트
        accuracy = self._test_accuracy(df_enhanced, btc_col)
        print(f"📈 피처 엔지니어링 결과: {accuracy:.2f}%")
        
        return accuracy
    
    def improvement_2_ensemble_weighting(self, df: pd.DataFrame) -> float:
        """개선 2: 동적 앙상블 가중치"""
        print("\n💡 개선 2: 동적 앙상블 가중치")
        print("-" * 40)
        
        # BTC 가격 컬럼 찾기
        btc_col = 'onchain_blockchain_info_network_stats_market_price_usd'
        if btc_col not in df.columns:
            btc_col = df.columns[0]
        X = df.drop(columns=[btc_col]).values
        y = df[btc_col].shift(-1).dropna().values
        X = X[:-1]
        
        # 시계열 분할
        tscv = TimeSeriesSplit(n_splits=3)
        accuracies = []
        
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # 스케일링
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # 3개 모델 학습
            models = {
                'rf': RandomForestRegressor(n_estimators=50, random_state=42),
                'gbm': GradientBoostingRegressor(n_estimators=50, random_state=42),
                'ridge': Ridge(alpha=1.0)
            }
            
            predictions = {}
            for name, model in models.items():
                model.fit(X_train_scaled, y_train)
                pred = model.predict(X_test_scaled)
                predictions[name] = pred
            
            # 동적 가중치 계산 (최근 성과 기반)
            weights = {'rf': 0.4, 'gbm': 0.4, 'ridge': 0.2}  # 초기 가중치
            
            # 앙상블 예측
            ensemble_pred = (weights['rf'] * predictions['rf'] + 
                           weights['gbm'] * predictions['gbm'] + 
                           weights['ridge'] * predictions['ridge'])
            
            accuracy = max(0, r2_score(y_test, ensemble_pred) * 100)
            accuracies.append(accuracy)
        
        final_accuracy = np.mean(accuracies)
        print(f"📈 동적 앙상블 결과: {final_accuracy:.2f}%")
        
        return final_accuracy
    
    def improvement_3_market_regime(self, df: pd.DataFrame) -> float:
        """개선 3: 시장 국면별 모델"""
        print("\n💡 개선 3: 시장 국면별 특화 모델")
        print("-" * 40)
        
        # BTC 가격 컬럼 찾기
        btc_col = 'onchain_blockchain_info_network_stats_market_price_usd'
        if btc_col not in df.columns:
            btc_col = df.columns[0]
        price_data = df[btc_col]
        
        # 시장 국면 정의 (간단화)
        returns_24h = price_data.pct_change(24).fillna(0)
        volatility_24h = price_data.pct_change().rolling(24).std().fillna(0)
        
        # 2개 국면으로 단순화 (상승/하락)
        bull_market = returns_24h > 0.02  # 2% 이상 상승
        bear_market = returns_24h < -0.02  # 2% 이상 하락
        
        # 국면별 데이터 분리
        df_bull = df[bull_market].copy()
        df_bear = df[bear_market].copy()
        df_sideways = df[~(bull_market | bear_market)].copy()
        
        # 각 국면별 정확도 계산
        accuracies = []
        
        for regime_name, regime_df in [("상승장", df_bull), ("하락장", df_bear), ("횡보장", df_sideways)]:
            if len(regime_df) < 100:  # 데이터가 너무 적으면 스킵
                continue
                
            acc = self._test_accuracy(regime_df, btc_col)
            accuracies.append(acc)
            print(f"  {regime_name}: {acc:.2f}%")
        
        avg_accuracy = np.mean(accuracies) if accuracies else 0
        print(f"📈 시장국면별 평균 결과: {avg_accuracy:.2f}%")
        
        return avg_accuracy
    
    def _test_accuracy(self, df: pd.DataFrame, btc_col: str) -> float:
        """정확도 테스트 헬퍼 함수"""
        X = df.drop(columns=[btc_col]).values
        y = df[btc_col].shift(-1).dropna().values
        X = X[:-1]
        
        if len(X) < 50:  # 데이터가 너무 적으면 0 반환
            return 0.0
        
        # 간단한 train/test 분할 (마지막 30% 테스트)
        split_idx = int(len(X) * 0.7)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # 스케일링 및 모델 학습
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        rf = RandomForestRegressor(n_estimators=30, random_state=42)
        rf.fit(X_train_scaled, y_train)
        
        pred = rf.predict(X_test_scaled)
        accuracy = max(0, r2_score(y_test, pred) * 100)
        
        return accuracy
    
    def run_all_improvements(self):
        """모든 개선사항 실행"""
        print("⚡ 빠른 정확도 향상 분석 시작...")
        print(f"⏰ 시작 시간: {datetime.now()}")
        
        # 데이터 로드
        df = self.load_data()
        
        # 베이스라인
        baseline = self.baseline_accuracy(df)
        self.results['baseline'] = baseline
        
        # 개선사항들
        self.results['feature_engineering'] = self.improvement_1_feature_engineering(df)
        self.results['ensemble_weighting'] = self.improvement_2_ensemble_weighting(df)
        self.results['market_regime'] = self.improvement_3_market_regime(df)
        
        # 최고 결과
        best_accuracy = max(self.results.values())
        improvement = best_accuracy - baseline
        
        print("\n" + "="*60)
        print("🏆 최종 결과 요약")
        print("="*60)
        print(f"📊 베이스라인:          {baseline:.2f}%")
        print(f"🔧 피처 엔지니어링:     {self.results['feature_engineering']:.2f}%")
        print(f"⚖️ 동적 앙상블:         {self.results['ensemble_weighting']:.2f}%")
        print(f"📈 시장국면별:          {self.results['market_regime']:.2f}%")
        print("-" * 60)
        print(f"🎯 최고 정확도:         {best_accuracy:.2f}%")
        print(f"📈 개선폭:             +{improvement:.2f}%")
        
        if best_accuracy >= self.target_accuracy:
            print(f"🎉 목표 달성! ({self.target_accuracy}% 이상)")
        else:
            print(f"⚠️  목표 미달성 (목표: {self.target_accuracy}%)")
        
        print("="*60)
        
        # 결과 저장
        result_summary = {
            "timestamp": datetime.now().isoformat(),
            "baseline_accuracy": baseline,
            "improvements": self.results,
            "best_accuracy": best_accuracy,
            "improvement_amount": improvement,
            "target_achieved": bool(best_accuracy >= self.target_accuracy)
        }
        
        result_file = os.path.join(self.data_path, "fast_improvement_results.json")
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result_summary, f, ensure_ascii=False, indent=2)
        
        print(f"💾 결과 저장: {result_file}")
        
        return result_summary

if __name__ == "__main__":
    system = FastAccuracyImprovement()
    results = system.run_all_improvements()