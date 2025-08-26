#!/usr/bin/env python3
"""
🎯 최종 정확도 테스트 (오류 수정)
- 실제 시계열 데이터로 백테스트
- 78.26% → 85%+ 목표 달성 확인
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

def final_accuracy_test():
    print("🎯 최종 정확도 테스트")
    print("=" * 50)
    
    # BTC 가격 데이터 로드
    print("📂 데이터 로딩...")
    btc_df = pd.read_csv("historical_6month_data/btc_price_hourly.csv")
    btc_df['timestamp'] = pd.to_datetime(btc_df['timestamp'])
    btc_df = btc_df.sort_values('timestamp').reset_index(drop=True)
    
    print(f"📈 BTC 데이터: {btc_df.shape}")
    print(f"📅 기간: {btc_df['timestamp'].min().strftime('%Y-%m-%d')} ~ {btc_df['timestamp'].max().strftime('%Y-%m-%d')}")
    
    # Fear & Greed Index 추가
    try:
        fg_df = pd.read_csv("historical_6month_data/fear_greed_index_hourly.csv")
        fg_df['timestamp'] = pd.to_datetime(fg_df['timestamp'])
        btc_df = btc_df.merge(fg_df, on='timestamp', how='left')
        print("✅ Fear & Greed Index 추가")
    except:
        print("⚠️ Fear & Greed Index 스킵")
    
    # MVRV 추가 
    try:
        mvrv_df = pd.read_csv("historical_6month_data/onchain_mvrv_hourly.csv")
        mvrv_df['timestamp'] = pd.to_datetime(mvrv_df['timestamp'])
        btc_df = btc_df.merge(mvrv_df, on='timestamp', how='left', suffixes=('', '_mvrv'))
        print("✅ MVRV Ratio 추가")
    except:
        print("⚠️ MVRV 스킵")
    
    # 숫자형 컬럼만 추출
    numeric_cols = btc_df.select_dtypes(include=[np.number]).columns
    df_clean = btc_df[['timestamp'] + list(numeric_cols)].copy()
    df_clean = df_clean.fillna(method='ffill').fillna(method='bfill').fillna(0)
    
    # 가격 컬럼 찾기
    price_col = 'open'  # BTC 가격
    price_data = df_clean[price_col]
    
    print(f"🎯 가격: ${price_data.mean():,.0f} (${price_data.min():,.0f} ~ ${price_data.max():,.0f})")
    
    # 시계열 피처 생성
    print("🔧 피처 생성...")
    df_enhanced = df_clean.copy()
    
    # 가격 기반 기술적 지표
    df_enhanced['price_lag1'] = price_data.shift(1)
    df_enhanced['price_change_1h'] = price_data.pct_change(1) * 100
    df_enhanced['price_change_6h'] = price_data.pct_change(6) * 100
    df_enhanced['price_change_24h'] = price_data.pct_change(24) * 100
    
    # 이동평균
    df_enhanced['sma_12h'] = price_data.rolling(12).mean()
    df_enhanced['sma_24h'] = price_data.rolling(24).mean()
    df_enhanced['sma_168h'] = price_data.rolling(168).mean()  # 1주
    
    # 볼린저 밴드
    sma_20 = price_data.rolling(20).mean()
    std_20 = price_data.rolling(20).std()
    df_enhanced['bb_upper'] = sma_20 + (std_20 * 2)
    df_enhanced['bb_lower'] = sma_20 - (std_20 * 2)
    df_enhanced['bb_position'] = (price_data - df_enhanced['bb_lower']) / (df_enhanced['bb_upper'] - df_enhanced['bb_lower'])
    
    # RSI
    delta = price_data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df_enhanced['rsi'] = 100 - (100 / (1 + rs))
    
    # 변동성
    df_enhanced['volatility_24h'] = price_data.pct_change().rolling(24).std() * 100
    df_enhanced['volatility_168h'] = price_data.pct_change().rolling(168).std() * 100
    
    # 거래량 분석 (있다면)
    if 'volume' in df_enhanced.columns:
        df_enhanced['volume_ma_24h'] = df_enhanced['volume'].rolling(24).mean()
        df_enhanced['volume_ratio'] = df_enhanced['volume'] / df_enhanced['volume_ma_24h']
    
    # NaN 처리
    df_enhanced = df_enhanced.fillna(method='bfill').fillna(0)
    
    print(f"🚀 향상된 피처: {df_enhanced.shape}")
    
    # 타겟 생성 (1시간 후 가격 예측)
    y = price_data.shift(-1).dropna()
    X = df_enhanced.drop(columns=['timestamp', price_col]).iloc[:-1]
    
    print(f"📊 모델링 데이터: X{X.shape}, y{len(y)}")
    
    # 시계열 분할 (최근 20% 테스트)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"✂️ 분할: 학습{len(X_train)} | 테스트{len(X_test)}")
    
    # 스케일링
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 모델 테스트
    print("\n🏆 성능 평가")
    print("-" * 30)
    
    models = {
        'RandomForest': RandomForestRegressor(
            n_estimators=100, 
            max_depth=20,
            min_samples_split=5,
            random_state=42
        ),
        'GradientBoosting': GradientBoostingRegressor(
            n_estimators=100,
            max_depth=8,
            learning_rate=0.1,
            random_state=42
        )
    }
    
    results = {}
    predictions = {}
    
    for name, model in models.items():
        print(f"\n🔬 {name}...")
        
        # 학습 및 예측
        model.fit(X_train_scaled, y_train)
        pred = model.predict(X_test_scaled)
        predictions[name] = pred
        
        # 성능 지표
        r2 = r2_score(y_test, pred)
        accuracy = max(0, r2 * 100)
        mae = mean_absolute_error(y_test, pred)
        
        # 방향성 정확도 (numpy로 계산)
        actual_changes = np.diff(y_test.values)
        pred_changes = np.diff(pred)
        direction_correct = (actual_changes > 0) == (pred_changes > 0)
        direction_accuracy = np.mean(direction_correct) * 100
        
        # 상관관계
        correlation = np.corrcoef(y_test.values, pred)[0, 1]
        
        results[name] = {
            'accuracy': accuracy,
            'mae': mae,
            'direction': direction_accuracy,
            'correlation': correlation
        }
        
        print(f"  📈 정확도: {accuracy:.2f}%")
        print(f"  🎯 방향성: {direction_accuracy:.1f}%")
        print(f"  📊 상관관계: {correlation:.3f}")
    
    # 앙상블
    print(f"\n🎭 앙상블...")
    ensemble_pred = (predictions['RandomForest'] + predictions['GradientBoosting']) / 2
    
    ensemble_r2 = r2_score(y_test, ensemble_pred)
    ensemble_accuracy = max(0, ensemble_r2 * 100)
    ensemble_mae = mean_absolute_error(y_test, ensemble_pred)
    
    # 앙상블 방향성
    ensemble_changes = np.diff(ensemble_pred)
    ensemble_direction = np.mean((actual_changes > 0) == (ensemble_changes > 0)) * 100
    ensemble_correlation = np.corrcoef(y_test.values, ensemble_pred)[0, 1]
    
    print(f"  📈 정확도: {ensemble_accuracy:.2f}%")
    print(f"  🎯 방향성: {ensemble_direction:.1f}%")
    print(f"  📊 상관관계: {ensemble_correlation:.3f}")
    
    # 최종 결과
    best_accuracy = max(
        results['RandomForest']['accuracy'],
        results['GradientBoosting']['accuracy'], 
        ensemble_accuracy
    )
    
    print("\n" + "=" * 60)
    print("🏆 최종 백테스트 결과")
    print("=" * 60)
    print(f"🎯 목표: 78.26% → 85%+ 달성")
    print("-" * 60)
    print(f"📊 Random Forest:     {results['RandomForest']['accuracy']:6.2f}%")
    print(f"📊 Gradient Boosting: {results['GradientBoosting']['accuracy']:6.2f}%") 
    print(f"📊 앙상블:            {ensemble_accuracy:6.2f}%")
    print("-" * 60)
    print(f"🏆 최고 성능:         {best_accuracy:6.2f}%")
    
    # 목표 달성 여부
    if best_accuracy >= 85:
        status = "🎉 목표 달성! (85% 이상)"
        improvement = best_accuracy - 78.26
        print(f"🚀 개선폭: +{improvement:.2f}%")
    elif best_accuracy >= 80:
        status = "🔥 거의 달성! (80% 이상)"
        needed = 85 - best_accuracy
        print(f"⚠️ 추가 필요: +{needed:.1f}%")
    elif best_accuracy >= 75:
        status = "✅ 양호한 성능 (75% 이상)"
        needed = 85 - best_accuracy
        print(f"⚠️ 추가 필요: +{needed:.1f}%")
    else:
        status = "⚠️ 추가 개선 필요"
        needed = 85 - best_accuracy
        print(f"⚠️ 부족한 성능: -{needed:.1f}%")
    
    print(f"📈 평가: {status}")
    print("=" * 60)
    
    return {
        'best_accuracy': best_accuracy,
        'target_achieved': best_accuracy >= 85,
        'improvement_needed': max(0, 85 - best_accuracy),
        'results': results
    }

if __name__ == "__main__":
    final_results = final_accuracy_test()