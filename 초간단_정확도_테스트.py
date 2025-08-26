#!/usr/bin/env python3
"""
⚡ 초간단 정확도 테스트
- 현재 모델 성능 빠르게 확인
- 주요 개선 아이디어만 테스트
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

def quick_test():
    print("⚡ 초간단 정확도 테스트")
    print("=" * 40)
    
    # 데이터 로드
    print("📂 데이터 로딩...")
    df = pd.read_csv("ai_optimized_3month_data/ai_matrix_complete.csv")
    
    # BTC 가격 컬럼
    btc_col = 'onchain_blockchain_info_network_stats_market_price_usd'
    if btc_col not in df.columns:
        btc_col = df.columns[0]
    
    print(f"🎯 타겟: {btc_col}")
    print(f"📊 데이터 크기: {df.shape}")
    
    # 기본 전처리
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df_clean = df[numeric_cols].fillna(0).replace([np.inf, -np.inf], 0)
    
    # X, y 분리
    X = df_clean.drop(columns=[btc_col]).values
    y = df_clean[btc_col].shift(-1).dropna().values
    X = X[:-1]
    
    print(f"📈 X shape: {X.shape}, y shape: {y.shape}")
    
    # 간단한 train/test 분할 (최근 20% 테스트)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"🔄 Train: {len(X_train)}, Test: {len(X_test)}")
    
    # 1. 베이스라인 모델
    print("\n1️⃣ 베이스라인 테스트...")
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    rf = RandomForestRegressor(n_estimators=30, random_state=42)
    rf.fit(X_train_scaled, y_train)
    pred_baseline = rf.predict(X_test_scaled)
    
    r2_baseline = r2_score(y_test, pred_baseline)
    acc_baseline = max(0, r2_baseline * 100)
    print(f"📊 베이스라인 정확도: {acc_baseline:.2f}%")
    
    # 2. 피처 중요도 기반 선택
    print("\n2️⃣ 상위 피처만 사용...")
    feature_importance = rf.feature_importances_
    top_indices = np.argsort(feature_importance)[-200:]  # 상위 200개
    
    X_train_top = X_train_scaled[:, top_indices]
    X_test_top = X_test_scaled[:, top_indices]
    
    rf_top = RandomForestRegressor(n_estimators=30, random_state=42)
    rf_top.fit(X_train_top, y_train)
    pred_top = rf_top.predict(X_test_top)
    
    r2_top = r2_score(y_test, pred_top)
    acc_top = max(0, r2_top * 100)
    print(f"📊 상위 피처 정확도: {acc_top:.2f}%")
    
    # 3. 앙상블 (2개 모델)
    print("\n3️⃣ 간단 앙상블...")
    from sklearn.ensemble import GradientBoostingRegressor
    
    gbm = GradientBoostingRegressor(n_estimators=30, random_state=42)
    gbm.fit(X_train_scaled, y_train)
    pred_gbm = gbm.predict(X_test_scaled)
    
    # 앙상블 (50:50)
    pred_ensemble = (pred_baseline + pred_gbm) / 2
    
    r2_ensemble = r2_score(y_test, pred_ensemble)
    acc_ensemble = max(0, r2_ensemble * 100)
    print(f"📊 앙상블 정확도: {acc_ensemble:.2f}%")
    
    # 결과 요약
    print("\n" + "=" * 50)
    print("🏆 결과 요약")
    print("=" * 50)
    print(f"📊 베이스라인:     {acc_baseline:.2f}%")
    print(f"🔝 상위 피처:     {acc_top:.2f}%")  
    print(f"⚖️ 앙상블:         {acc_ensemble:.2f}%")
    print("-" * 50)
    
    best_acc = max(acc_baseline, acc_top, acc_ensemble)
    print(f"🎯 최고 정확도:   {best_acc:.2f}%")
    
    if best_acc >= 85:
        print("🎉 목표 달성! (85% 이상)")
    else:
        need_improvement = 85 - best_acc
        print(f"⚠️ 목표 미달 (추가 {need_improvement:.1f}% 필요)")
    
    print("=" * 50)
    
    return {
        'baseline': acc_baseline,
        'top_features': acc_top, 
        'ensemble': acc_ensemble,
        'best': best_acc
    }

if __name__ == "__main__":
    results = quick_test()