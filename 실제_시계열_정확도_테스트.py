#!/usr/bin/env python3
"""
🎯 실제 시계열 데이터로 정확도 측정
- historical_6month_data의 실제 hourly 데이터 사용
- 진짜 시계열 예측 성능 확인
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

def real_timeseries_test():
    print("🎯 실제 시계열 데이터 정확도 테스트")
    print("=" * 60)
    
    # 1. BTC 가격 데이터 로드
    print("📂 BTC 가격 데이터 로딩...")
    btc_df = pd.read_csv("historical_6month_data/btc_price_hourly.csv")
    btc_df['timestamp'] = pd.to_datetime(btc_df['timestamp'])
    btc_df = btc_df.sort_values('timestamp').reset_index(drop=True)
    
    print(f"📈 BTC 데이터: {btc_df.shape}")
    print(f"📅 기간: {btc_df['timestamp'].min()} ~ {btc_df['timestamp'].max()}")
    
    # 2. 주요 지표들 로드 및 병합
    print("\n📊 주요 지표 로딩...")
    
    indicator_files = [
        "fear_greed_index_hourly.csv",
        "onchain_mvrv_hourly.csv", 
        "onchain_whale_ratio_hourly.csv",
        "derivatives_funding_rate_hourly.csv",
        "macro_SPX_hourly.csv",
        "macro_VIX_hourly.csv"
    ]
    
    # BTC 데이터를 기준으로 시작
    merged_df = btc_df.copy()
    
    # 각 지표 파일 병합
    for file in indicator_files:
        try:
            df = pd.read_csv(f"historical_6month_data/{file}")
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            # timestamp로 병합
            merged_df = merged_df.merge(df, on='timestamp', how='left')
            print(f"  ✅ {file} 병합 완료")
            
        except Exception as e:
            print(f"  ⚠️ {file} 스킵: {e}")
    
    print(f"\n🔗 병합 완료: {merged_df.shape}")
    
    # 3. 전처리
    print("🛠️ 데이터 전처리...")
    
    # timestamp 제외하고 숫자형만
    numeric_cols = merged_df.select_dtypes(include=[np.number]).columns
    df_clean = merged_df[['timestamp'] + list(numeric_cols)].copy()
    
    # 결측치 처리 (forward fill)
    df_clean = df_clean.fillna(method='ffill').fillna(method='bfill')
    
    # BTC 가격이 있는지 확인
    price_col = None
    for col in df_clean.columns:
        if 'price' in col.lower() and 'btc' in col.lower():
            price_col = col
            break
    
    if price_col is None:
        # 첫 번째 숫자 컬럼을 가격으로 사용
        price_col = numeric_cols[0]
    
    print(f"🎯 가격 컬럼: {price_col}")
    
    # 가격 데이터 연속성 확인
    price_data = df_clean[price_col]
    price_changes = price_data.pct_change().abs()
    continuity = (price_changes < 0.05).mean()  # 5% 미만 변화율 비율
    
    print(f"📈 가격 연속성: {continuity*100:.1f}%")
    print(f"📊 평균 가격: ${price_data.mean():,.0f}")
    print(f"📊 가격 범위: ${price_data.min():,.0f} ~ ${price_data.max():,.0f}")
    
    # 4. 시계열 피처 생성
    print("\n🔧 시계열 피처 생성...")
    df_enhanced = df_clean.copy()
    
    # 가격 기반 피처들
    df_enhanced['price_lag1'] = price_data.shift(1)
    df_enhanced['price_lag6'] = price_data.shift(6)
    df_enhanced['price_lag24'] = price_data.shift(24)
    
    df_enhanced['price_change_1h'] = price_data.pct_change(1)
    df_enhanced['price_change_6h'] = price_data.pct_change(6)
    df_enhanced['price_change_24h'] = price_data.pct_change(24)
    
    df_enhanced['price_ma_12h'] = price_data.rolling(12).mean()
    df_enhanced['price_ma_24h'] = price_data.rolling(24).mean()
    df_enhanced['price_ma_168h'] = price_data.rolling(168).mean()  # 1주
    
    df_enhanced['price_std_12h'] = price_data.rolling(12).std()
    df_enhanced['price_std_24h'] = price_data.rolling(24).std()
    
    df_enhanced['price_min_24h'] = price_data.rolling(24).min()
    df_enhanced['price_max_24h'] = price_data.rolling(24).max()
    df_enhanced['price_range_24h'] = df_enhanced['price_max_24h'] - df_enhanced['price_min_24h']
    
    # RSI (14기간)
    delta = price_data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df_enhanced['rsi_14'] = 100 - (100 / (1 + rs))
    
    # NaN 처리
    df_enhanced = df_enhanced.fillna(method='bfill').fillna(0)
    
    print(f"🚀 향상된 데이터: {df_enhanced.shape}")
    
    # 5. 모델링
    print("\n🤖 모델링 준비...")
    
    # X, y 준비
    y = price_data.shift(-1).dropna()  # 1시간 후 가격 예측
    X = df_enhanced.drop(columns=['timestamp', price_col]).iloc[:-1]
    
    print(f"📊 X: {X.shape}, y: {y.shape}")
    
    # 시계열 분할 (최근 20%를 테스트)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"✂️ Train: {len(X_train)}, Test: {len(X_test)}")
    
    # 스케일링
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 6. 모델 학습 및 평가
    print("\n🏆 모델 성능 평가")
    print("-" * 40)
    
    models = {
        'RandomForest': RandomForestRegressor(n_estimators=50, max_depth=15, random_state=42),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=50, max_depth=8, random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\n🔬 {name} 테스트...")
        
        # 학습
        model.fit(X_train_scaled, y_train)
        
        # 예측
        pred = model.predict(X_test_scaled)
        
        # 성능 계산
        mae = mean_absolute_error(y_test, pred)
        rmse = np.sqrt(np.mean((y_test - pred) ** 2))
        r2 = r2_score(y_test, pred)
        accuracy = max(0, r2 * 100)
        
        # 방향성 정확도 
        actual_direction = (y_test.diff() > 0).astype(int).iloc[1:]
        pred_direction = (pd.Series(pred).diff() > 0).astype(int).iloc[1:]
        direction_acc = (actual_direction == pred_direction).mean() * 100
        
        # 상관관계
        correlation = np.corrcoef(y_test, pred)[0, 1]
        
        results[name] = {
            'mae': mae,
            'rmse': rmse, 
            'r2': r2,
            'accuracy': accuracy,
            'direction_accuracy': direction_acc,
            'correlation': correlation
        }
        
        print(f"  📈 정확도(R²): {accuracy:.2f}%")
        print(f"  🎯 방향성: {direction_acc:.1f}%")
        print(f"  📊 상관관계: {correlation:.3f}")
        print(f"  💰 MAE: ${mae:.0f}")
    
    # 7. 앙상블
    print(f"\n🎭 앙상블 테스트...")
    
    rf_pred = models['RandomForest'].predict(X_test_scaled)
    gb_pred = models['GradientBoosting'].predict(X_test_scaled)
    
    # 가중 평균 (RF 60%, GB 40%)
    ensemble_pred = 0.6 * rf_pred + 0.4 * gb_pred
    
    ensemble_r2 = r2_score(y_test, ensemble_pred)
    ensemble_accuracy = max(0, ensemble_r2 * 100)
    ensemble_mae = mean_absolute_error(y_test, ensemble_pred)
    ensemble_corr = np.corrcoef(y_test, ensemble_pred)[0, 1]
    
    # 앙상블 방향성
    ensemble_direction = (pd.Series(ensemble_pred).diff() > 0).astype(int).iloc[1:]
    ensemble_dir_acc = (actual_direction == ensemble_direction).mean() * 100
    
    print(f"  📈 앙상블 정확도: {ensemble_accuracy:.2f}%")
    print(f"  🎯 앙상블 방향성: {ensemble_dir_acc:.1f}%")
    print(f"  📊 앙상블 상관관계: {ensemble_corr:.3f}")
    print(f"  💰 앙상블 MAE: ${ensemble_mae:.0f}")
    
    # 8. 최종 결과
    best_accuracy = max(results['RandomForest']['accuracy'], 
                       results['GradientBoosting']['accuracy'], 
                       ensemble_accuracy)
    
    print("\n" + "=" * 70)
    print("🏆 최종 결과 - 실제 시계열 데이터")
    print("=" * 70)
    print(f"🎯 최고 정확도: {best_accuracy:.2f}%")
    print(f"📊 Random Forest: {results['RandomForest']['accuracy']:.2f}%")
    print(f"📊 Gradient Boosting: {results['GradientBoosting']['accuracy']:.2f}%")  
    print(f"📊 앙상블: {ensemble_accuracy:.2f}%")
    print("-" * 70)
    
    if best_accuracy >= 80:
        status = "🎉 매우 우수!"
    elif best_accuracy >= 70:
        status = "✅ 우수함"
    elif best_accuracy >= 60:
        status = "👍 양호함" 
    elif best_accuracy >= 50:
        status = "⚠️ 개선 필요"
    else:
        status = "❌ 심각한 문제"
    
    print(f"📈 성능 평가: {status}")
    print("=" * 70)
    
    return {
        'best_accuracy': best_accuracy,
        'rf_accuracy': results['RandomForest']['accuracy'],
        'gb_accuracy': results['GradientBoosting']['accuracy'],
        'ensemble_accuracy': ensemble_accuracy,
        'data_continuity': continuity
    }

if __name__ == "__main__":
    results = real_timeseries_test()