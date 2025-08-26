#!/usr/bin/env python3
"""
🔍 디버그 테스트 - 왜 정확도가 0%인지 분석
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

def debug_test():
    print("🔍 디버그 테스트 - 정확도 0% 원인 분석")
    print("=" * 50)
    
    # 데이터 로드
    df = pd.read_csv("ai_optimized_3month_data/ai_matrix_complete.csv")
    btc_col = 'onchain_blockchain_info_network_stats_market_price_usd'
    
    print(f"📊 원본 데이터: {df.shape}")
    print(f"🎯 타겟 컬럼: {btc_col}")
    
    # 기본 전처리
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df_clean = df[numeric_cols].fillna(0).replace([np.inf, -np.inf], 0)
    
    # BTC 가격 분석
    btc_price = df_clean[btc_col]
    print(f"\n📈 BTC 가격 분석:")
    print(f"  - 평균: {btc_price.mean():.2f}")
    print(f"  - 표준편차: {btc_price.std():.2f}")  
    print(f"  - 최소값: {btc_price.min():.2f}")
    print(f"  - 최대값: {btc_price.max():.2f}")
    
    # 연속성 체크 (시계열 순서가 맞는지)
    price_changes = btc_price.diff().abs()
    print(f"  - 평균 변화량: {price_changes.mean():.2f}")
    print(f"  - 큰 변화 횟수: {(price_changes > 100).sum()}")
    
    # Target 생성 (1시간 후 가격 예측)
    y = btc_price.shift(-1).dropna()
    X = df_clean.drop(columns=[btc_col]).iloc[:-1]  # 마지막 행 제거
    
    print(f"\n🔢 모델링 데이터:")
    print(f"  - X shape: {X.shape}")
    print(f"  - y shape: {y.shape}")
    print(f"  - X에 NaN/inf: {np.isinf(X.values).sum() + np.isnan(X.values).sum()}")
    print(f"  - y에 NaN/inf: {np.isinf(y.values).sum() + np.isnan(y.values).sum()}")
    
    # Train/Test 분할
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"\n✂️ 데이터 분할:")
    print(f"  - Train: {len(X_train)}")
    print(f"  - Test: {len(X_test)}")
    
    # 스케일링
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 모델 학습
    print(f"\n🤖 모델 학습...")
    rf = RandomForestRegressor(n_estimators=10, random_state=42, max_depth=5)
    rf.fit(X_train_scaled, y_train)
    
    # 예측
    pred = rf.predict(X_test_scaled)
    
    print(f"\n📊 예측 결과:")
    print(f"  - 실제 y_test 평균: {y_test.mean():.2f}")
    print(f"  - 예측 pred 평균: {pred.mean():.2f}")
    print(f"  - 실제 y_test 표준편차: {y_test.std():.2f}")
    print(f"  - 예측 pred 표준편차: {pred.std():.2f}")
    
    # 다양한 메트릭 계산
    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(np.mean((y_test - pred) ** 2))
    r2 = r2_score(y_test, pred)
    
    print(f"\n📈 성능 지표:")
    print(f"  - MAE: {mae:.4f}")
    print(f"  - RMSE: {rmse:.4f}")
    print(f"  - R²: {r2:.4f}")
    print(f"  - R² * 100: {r2 * 100:.2f}%")
    
    # R² 스코어가 음수인 경우 확인
    if r2 < 0:
        print(f"\n⚠️ R² 음수 원인 분석:")
        
        # 베이스라인 예측 (평균값으로 예측)
        baseline_pred = np.full_like(y_test, y_train.mean())
        baseline_mse = np.mean((y_test - baseline_pred) ** 2)
        model_mse = np.mean((y_test - pred) ** 2)
        
        print(f"  - 베이스라인 MSE: {baseline_mse:.4f}")
        print(f"  - 모델 MSE: {model_mse:.4f}")
        print(f"  - 모델이 베이스라인보다 {model_mse/baseline_mse:.2f}배 나쁨")
        
        # 상관관계 확인
        correlation = np.corrcoef(y_test, pred)[0, 1]
        print(f"  - 실제값과 예측값 상관관계: {correlation:.4f}")
    
    # 간단한 베이스라인과 비교
    print(f"\n🎯 베이스라인 비교:")
    
    # 베이스라인 1: 평균값 예측
    baseline1 = np.full_like(y_test, y_train.mean())
    r2_baseline1 = r2_score(y_test, baseline1)
    print(f"  - 평균값 예측 R²: {r2_baseline1:.4f} ({r2_baseline1*100:.2f}%)")
    
    # 베이스라인 2: 마지막 값 예측 (no-change)
    baseline2 = y_test.shift(1).fillna(y_train.iloc[-1])
    r2_baseline2 = r2_score(y_test[1:], baseline2[1:])
    print(f"  - 마지막값 예측 R²: {r2_baseline2:.4f} ({r2_baseline2*100:.2f}%)")
    
    print("=" * 50)
    
    return {
        'r2': r2,
        'mae': mae,
        'rmse': rmse,
        'correlation': np.corrcoef(y_test, pred)[0, 1] if r2 < 0 else None
    }

if __name__ == "__main__":
    results = debug_test()