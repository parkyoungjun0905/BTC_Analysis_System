#!/usr/bin/env python3
"""
📅 시계열 정렬 후 정확도 테스트
- timestamp로 정렬하여 실제 시계열 만들기
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

def timeseries_sorted_test():
    print("📅 시계열 정렬 후 정확도 테스트")
    print("=" * 50)
    
    # 데이터 로드
    df = pd.read_csv("ai_optimized_3month_data/ai_matrix_complete.csv")
    btc_col = 'onchain_blockchain_info_network_stats_market_price_usd'
    
    print(f"📊 원본 데이터: {df.shape}")
    
    # timestamp로 정렬
    if 'timestamp' in df.columns:
        print("📅 timestamp로 정렬 중...")
        df = df.sort_values('timestamp').reset_index(drop=True)
    else:
        print("⚠️ timestamp 컬럼이 없습니다. 원본 순서 사용")
    
    # 전처리
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df_clean = df[numeric_cols].fillna(0).replace([np.inf, -np.inf], 0)
    
    # BTC 가격 변화량 확인
    btc_price = df_clean[btc_col]
    price_changes = btc_price.diff().abs()
    print(f"\n📈 정렬 후 BTC 가격 분석:")
    print(f"  - 평균 변화량: {price_changes.mean():.2f}")
    print(f"  - 큰 변화 횟수: {(price_changes > 100).sum()}")
    print(f"  - 연속성 비율: {((price_changes < 50).sum() / len(price_changes) * 100):.1f}%")
    
    # 시계열 특성 추가
    print("\n🛠️ 시계열 피처 추가...")
    df_enhanced = df_clean.copy()
    
    # 가격 기반 피처
    df_enhanced['price_lag_1'] = btc_price.shift(1)
    df_enhanced['price_lag_2'] = btc_price.shift(2)
    df_enhanced['price_change_1h'] = btc_price.pct_change(1)
    df_enhanced['price_change_6h'] = btc_price.pct_change(6)
    df_enhanced['price_ma_12h'] = btc_price.rolling(12).mean()
    df_enhanced['price_ma_24h'] = btc_price.rolling(24).mean()
    df_enhanced['price_std_12h'] = btc_price.rolling(12).std()
    
    # NaN 처리
    df_enhanced = df_enhanced.fillna(method='bfill').fillna(0)
    
    # Target 생성 (1시간 후)
    y = btc_price.shift(-1).dropna()
    X = df_enhanced.drop(columns=[btc_col]).iloc[:-1]
    
    print(f"📊 향상된 데이터: X{X.shape}, y{y.shape}")
    
    # 시계열 분할 (과거 80%로 학습, 최근 20%로 테스트)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"✂️ 시계열 분할: Train{len(X_train)}, Test{len(X_test)}")
    
    # 스케일링
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 모델 학습 (시계열에 적합한 설정)
    print("\n🤖 시계열 모델 학습...")
    rf = RandomForestRegressor(
        n_estimators=50,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42
    )
    rf.fit(X_train_scaled, y_train)
    
    # 예측
    pred = rf.predict(X_test_scaled)
    
    # 성능 평가
    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(np.mean((y_test - pred) ** 2))
    r2 = r2_score(y_test, pred)
    correlation = np.corrcoef(y_test, pred)[0, 1]
    
    print(f"\n📈 성능 결과:")
    print(f"  - MAE: {mae:.2f}")
    print(f"  - RMSE: {rmse:.2f}")
    print(f"  - R²: {r2:.4f}")
    print(f"  - 정확도: {max(0, r2 * 100):.2f}%")
    print(f"  - 상관관계: {correlation:.4f}")
    
    # 피처 중요도 확인
    print(f"\n🔝 상위 피처:")
    feature_names = X.columns
    importance = rf.feature_importances_
    top_indices = np.argsort(importance)[-10:]
    
    for i, idx in enumerate(reversed(top_indices)):
        print(f"  {i+1:2d}. {feature_names[idx][:50]:50s} {importance[idx]:.4f}")
    
    # 방향성 정확도 (상승/하락 예측)
    actual_direction = (y_test.diff() > 0).astype(int)
    pred_direction = (pd.Series(pred).diff() > 0).astype(int)
    direction_accuracy = (actual_direction == pred_direction).mean()
    
    print(f"\n🎯 방향성 정확도: {direction_accuracy * 100:.1f}%")
    
    # 베이스라인 비교
    baseline_pred = y_test.shift(1).fillna(y_train.iloc[-1])  # 직전 값으로 예측
    baseline_r2 = r2_score(y_test[1:], baseline_pred[1:])
    
    print(f"📊 베이스라인(직전값) R²: {baseline_r2:.4f} ({max(0, baseline_r2*100):.2f}%)")
    
    improvement = max(0, r2 * 100) - max(0, baseline_r2 * 100)
    print(f"🚀 베이스라인 대비 향상: +{improvement:.2f}%")
    
    print("\n" + "=" * 60)
    print("🏆 최종 결과")
    print("=" * 60)
    print(f"🎯 시계열 정렬 후 정확도: {max(0, r2 * 100):.2f}%")
    
    if max(0, r2 * 100) >= 75:
        print("🎉 우수한 성능!")
    elif max(0, r2 * 100) >= 50:
        print("✅ 양호한 성능")
    else:
        print("⚠️ 추가 개선 필요")
    
    print("=" * 60)
    
    return {
        'r2': r2,
        'accuracy': max(0, r2 * 100),
        'correlation': correlation,
        'direction_accuracy': direction_accuracy * 100,
        'improvement': improvement
    }

if __name__ == "__main__":
    results = timeseries_sorted_test()