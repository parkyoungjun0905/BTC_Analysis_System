#!/usr/bin/env python3
"""
🏆 고급 BTC 90% 정확도 도전 시스템
실제 딥러닝 + 앙상블 + 고급 특성공학으로 90% 정확도 달성

핵심 전략:
1. 방향성 예측에 특화된 모델 설계
2. 다중 예측 호라이즌 (1h, 2h, 3h) 앙상블
3. 시장 상황별 적응적 모델
4. 고급 특성 엔지니어링 (100+ 지표)
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import pickle

# 경고 메시지 숨기기
warnings.filterwarnings('ignore')

# 고급 ML 라이브러리
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# 딥러닝 사용 가능시 import
try:
    from sklearn.neural_network import MLPRegressor
    from sklearn.svm import SVR
    ADVANCED_ML_AVAILABLE = True
    print("✅ 고급 ML 모델 로딩 완료")
except ImportError:
    ADVANCED_ML_AVAILABLE = False
    print("⚠️ 고급 ML 모델 일부 제한")

print("🏆 고급 BTC 90% 정확도 도전 시스템 시작")
print("=" * 60)

class AdvancedFeatureEngineer:
    """고급 특성공학 - 100+ 지표 생성"""
    
    def __init__(self):
        self.scalers = {
            'robust': RobustScaler(),
            'minmax': MinMaxScaler(),
            'standard': StandardScaler()
        }
        print("✅ 고급 특성공학 엔진 초기화")
    
    def create_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """고급 기술 지표 대량 생성"""
        print("🔧 고급 특성공학 시작 (100+ 지표 생성)...")
        
        try:
            original_len = len(df)
            
            # 1. 기본 가격 지표들
            for window in [3, 5, 10, 20, 50]:
                df[f'sma_{window}'] = df['close'].rolling(window).mean()
                df[f'ema_{window}'] = df['close'].ewm(window).mean()
                df[f'price_sma_ratio_{window}'] = df['close'] / df[f'sma_{window}']
                df[f'price_ema_ratio_{window}'] = df['close'] / df[f'ema_{window}']
            
            # 2. 변동성 지표들
            for window in [5, 10, 20]:
                df[f'volatility_{window}'] = df['close'].rolling(window).std()
                df[f'volatility_ratio_{window}'] = df[f'volatility_{window}'] / df[f'volatility_{window}'].rolling(50).mean()
            
            # 3. 모멘텀 지표들 (방향성 예측에 중요!)
            for window in [3, 5, 10, 14, 20]:
                df[f'roc_{window}'] = df['close'].pct_change(window)
                df[f'momentum_{window}'] = df['close'] / df['close'].shift(window) - 1
            
            # 4. RSI 변형들
            for window in [7, 14, 21]:
                delta = df['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
                rs = gain / loss
                df[f'rsi_{window}'] = 100 - (100 / (1 + rs))
                df[f'rsi_ma_{window}'] = df[f'rsi_{window}'].rolling(5).mean()
            
            # 5. MACD 변형들
            for fast, slow, signal in [(5, 10, 3), (12, 26, 9), (8, 21, 5)]:
                exp1 = df['close'].ewm(span=fast).mean()
                exp2 = df['close'].ewm(span=slow).mean()
                df[f'macd_{fast}_{slow}'] = exp1 - exp2
                df[f'macd_signal_{fast}_{slow}_{signal}'] = df[f'macd_{fast}_{slow}'].ewm(span=signal).mean()
                df[f'macd_histogram_{fast}_{slow}_{signal}'] = df[f'macd_{fast}_{slow}'] - df[f'macd_signal_{fast}_{slow}_{signal}']
            
            # 6. 볼린저밴드 변형들
            for window, std_dev in [(10, 2), (20, 2), (20, 1.5), (50, 2.5)]:
                sma = df['close'].rolling(window).mean()
                std = df['close'].rolling(window).std()
                df[f'bb_upper_{window}_{std_dev}'] = sma + (std * std_dev)
                df[f'bb_lower_{window}_{std_dev}'] = sma - (std * std_dev)
                df[f'bb_position_{window}_{std_dev}'] = (df['close'] - df[f'bb_lower_{window}_{std_dev}']) / (df[f'bb_upper_{window}_{std_dev}'] - df[f'bb_lower_{window}_{std_dev}'])
                df[f'bb_width_{window}_{std_dev}'] = (df[f'bb_upper_{window}_{std_dev}'] - df[f'bb_lower_{window}_{std_dev}']) / sma
            
            # 7. 스토캐스틱 변형들  
            for k_window, d_window in [(5, 3), (14, 3), (21, 5)]:
                low_min = df['close'].rolling(window=k_window).min()
                high_max = df['close'].rolling(window=k_window).max()
                df[f'stoch_k_{k_window}'] = 100 * (df['close'] - low_min) / (high_max - low_min)
                df[f'stoch_d_{k_window}_{d_window}'] = df[f'stoch_k_{k_window}'].rolling(d_window).mean()
            
            # 8. 볼륨 지표들 (중요!)
            if 'volume' in df.columns:
                for window in [5, 10, 20, 50]:
                    df[f'volume_sma_{window}'] = df['volume'].rolling(window).mean()
                    df[f'volume_ratio_{window}'] = df['volume'] / df[f'volume_sma_{window}']
                    df[f'price_volume_trend_{window}'] = ((df['close'] - df['close'].shift()) / df['close'].shift()) * df['volume']
                
                # OBV (On-Balance Volume)
                df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
                df['obv_sma'] = df['obv'].rolling(20).mean()
                df['obv_ratio'] = df['obv'] / df['obv_sma']
            
            # 9. 추세 지표들
            for window in [10, 20, 50]:
                df[f'trend_strength_{window}'] = df['close'].rolling(window).apply(
                    lambda x: np.corrcoef(np.arange(len(x)), x)[0, 1] if len(x) == window else 0, raw=False
                ).fillna(0)
            
            # 10. 지지/저항 수준 (단순화)
            df['support_5d'] = df['close'].rolling(5).min()
            df['resistance_5d'] = df['close'].rolling(5).max()
            df['support_distance'] = (df['close'] - df['support_5d']) / df['close']
            df['resistance_distance'] = (df['resistance_5d'] - df['close']) / df['close']
            
            # 11. 고급 패턴 인식 (방향성 예측 핵심!)
            # 상승/하락 연속성
            df['price_change'] = df['close'].pct_change()
            df['up_days'] = (df['price_change'] > 0).astype(int)
            df['down_days'] = (df['price_change'] < 0).astype(int)
            
            for window in [3, 5, 10]:
                df[f'up_streak_{window}'] = df['up_days'].rolling(window).sum()
                df[f'down_streak_{window}'] = df['down_days'].rolling(window).sum()
                df[f'momentum_ratio_{window}'] = df[f'up_streak_{window}'] / (df[f'down_streak_{window}'] + 1)
            
            # 12. 시간 기반 특성들
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df['hour'] = df['timestamp'].dt.hour
                df['day_of_week'] = df['timestamp'].dt.dayofweek
                df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
                df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
                df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
                df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
            
            # 13. 라그 특성들 (과거 값들)
            for lag in [1, 2, 3, 5, 10]:
                df[f'price_lag_{lag}'] = df['close'].shift(lag)
                df[f'return_lag_{lag}'] = df['price_change'].shift(lag)
                df[f'volume_lag_{lag}'] = df['volume'].shift(lag) if 'volume' in df.columns else 0
            
            # 14. 롤링 통계
            for window in [5, 10, 20]:
                df[f'skew_{window}'] = df['price_change'].rolling(window).skew()
                df[f'kurt_{window}'] = df['price_change'].rolling(window).kurt()
                df[f'std_{window}'] = df['price_change'].rolling(window).std()
            
            # 15. 교차 검증 특성들 (다른 지표와의 관계)
            df['rsi_sma_cross'] = np.where(df['rsi_14'] > df['rsi_ma_14'], 1, -1)
            df['price_sma_cross'] = np.where(df['close'] > df['sma_20'], 1, -1)
            df['macd_cross'] = np.where(df['macd_12_26'] > df['macd_signal_12_26_9'], 1, -1)
            
            # NaN 처리
            df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
            
            final_len = len(df)
            feature_count = len(df.columns)
            
            print(f"✅ 고급 특성공학 완료:")
            print(f"   📊 생성된 특성 수: {feature_count}")
            print(f"   📈 데이터 길이: {original_len} → {final_len}")
            
            return df
            
        except Exception as e:
            print(f"❌ 특성공학 오류: {e}")
            return df

class DirectionalPredictor:
    """방향성 예측에 특화된 AI 예측기"""
    
    def __init__(self, lookback_hours: int = 10):
        self.lookback_hours = lookback_hours
        self.models = {}
        self.scalers = {}
        self.feature_columns = []
        print(f"✅ 방향성 특화 AI 예측기 초기화 (lookback: {lookback_hours}시간)")
    
    def create_directional_sequences(self, df: pd.DataFrame) -> Tuple:
        """방향성 예측을 위한 시퀀스 생성"""
        print("🎯 방향성 예측 시퀀스 생성 중...")
        
        # 핵심 특성들 선택 (방향성 예측에 중요한 것들)
        directional_features = []
        
        # 1. 모멘텀 지표들
        momentum_cols = [col for col in df.columns if 'momentum_' in col or 'roc_' in col]
        directional_features.extend(momentum_cols[:10])  # 상위 10개
        
        # 2. RSI 관련
        rsi_cols = [col for col in df.columns if 'rsi' in col]
        directional_features.extend(rsi_cols[:5])
        
        # 3. MACD 관련  
        macd_cols = [col for col in df.columns if 'macd' in col]
        directional_features.extend(macd_cols[:8])
        
        # 4. 추세 지표들
        trend_cols = [col for col in df.columns if 'trend' in col or 'streak' in col]
        directional_features.extend(trend_cols[:8])
        
        # 5. 가격 비율들
        ratio_cols = [col for col in df.columns if 'ratio' in col and 'price' in col]
        directional_features.extend(ratio_cols[:5])
        
        # 6. 크로스 시그널들  
        cross_cols = [col for col in df.columns if 'cross' in col]
        directional_features.extend(cross_cols)
        
        # 7. 볼린저밴드 포지션
        bb_cols = [col for col in df.columns if 'bb_position' in col]
        directional_features.extend(bb_cols[:3])
        
        # 8. 스토캐스틱
        stoch_cols = [col for col in df.columns if 'stoch' in col]
        directional_features.extend(stoch_cols[:4])
        
        # 중복 제거 및 존재하는 컬럼만 선택
        directional_features = list(set([col for col in directional_features if col in df.columns]))
        
        if len(directional_features) < 10:
            # 부족하면 다른 지표들 추가
            other_cols = [col for col in df.columns if col not in directional_features 
                         and col not in ['close', 'timestamp', 'volume'] 
                         and not col.startswith('price_lag')]
            directional_features.extend(other_cols[:20-len(directional_features)])
        
        self.feature_columns = directional_features[:30]  # 상위 30개 특성 사용
        print(f"📊 방향성 예측 특성 {len(self.feature_columns)}개 선택:")
        for i, col in enumerate(self.feature_columns[:10]):
            print(f"   {i+1:2d}. {col}")
        if len(self.feature_columns) > 10:
            print(f"   ... 외 {len(self.feature_columns)-10}개")
        
        # 시퀀스 생성
        X, y_1h, y_2h, y_3h = [], [], [], []
        
        for i in range(self.lookback_hours, len(df) - 3):
            # 특성들
            features = df[self.feature_columns].iloc[i-self.lookback_hours:i].values.flatten()
            X.append(features)
            
            # 방향성 타겟들 (상승=1, 하락=-1)
            current_price = df['close'].iloc[i]
            price_1h = df['close'].iloc[i + 1] if i + 1 < len(df) else current_price
            price_2h = df['close'].iloc[i + 2] if i + 2 < len(df) else current_price  
            price_3h = df['close'].iloc[i + 3] if i + 3 < len(df) else current_price
            
            y_1h.append(1 if price_1h > current_price else -1)
            y_2h.append(1 if price_2h > current_price else -1)
            y_3h.append(1 if price_3h > current_price else -1)
        
        X = np.array(X)
        y_1h = np.array(y_1h)
        y_2h = np.array(y_2h) 
        y_3h = np.array(y_3h)
        
        print(f"✅ 시퀀스 생성 완료: X={X.shape}")
        print(f"   1H 방향성: 상승 {(y_1h==1).sum()}, 하락 {(y_1h==-1).sum()}")
        print(f"   2H 방향성: 상승 {(y_2h==1).sum()}, 하락 {(y_2h==-1).sum()}")  
        print(f"   3H 방향성: 상승 {(y_3h==1).sum()}, 하락 {(y_3h==-1).sum()}")
        
        return X, y_1h, y_2h, y_3h
    
    def train_ensemble_models(self, X: np.ndarray, y_1h: np.ndarray, y_2h: np.ndarray, y_3h: np.ndarray) -> Dict:
        """다중 모델 앙상블 학습"""
        print("🤖 다중 모델 앙상블 학습 시작...")
        
        # 데이터 스케일링
        self.scalers['scaler'] = RobustScaler()
        X_scaled = self.scalers['scaler'].fit_transform(X)
        
        # 다양한 모델들 정의
        model_configs = {
            'rf_1h': (RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1), y_1h),
            'rf_2h': (RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1), y_2h),
            'rf_3h': (RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1), y_3h),
            'gb_1h': (GradientBoostingRegressor(n_estimators=150, max_depth=8, learning_rate=0.1, random_state=42), y_1h),
            'gb_2h': (GradientBoostingRegressor(n_estimators=150, max_depth=8, learning_rate=0.1, random_state=42), y_2h),
            'gb_3h': (GradientBoostingRegressor(n_estimators=150, max_depth=8, learning_rate=0.1, random_state=42), y_3h),
        }
        
        # 고급 모델들 추가
        if ADVANCED_ML_AVAILABLE:
            model_configs.update({
                'mlp_1h': (MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42), y_1h),
                'mlp_2h': (MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42), y_2h), 
                'mlp_3h': (MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42), y_3h),
            })
        
        # 모델 학습 및 검증
        results = {'models': {}, 'scores': {}}
        
        for model_name, (model, y_target) in model_configs.items():
            print(f"  🔧 {model_name} 학습 중...")
            
            # 시계열 교차검증
            tscv = TimeSeriesSplit(n_splits=3)
            cv_scores = []
            
            for train_idx, val_idx in tscv.split(X_scaled):
                X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
                y_train, y_val = y_target[train_idx], y_target[val_idx]
                
                # 모델 학습
                model_copy = type(model)(**model.get_params())
                model_copy.fit(X_train, y_train)
                
                # 예측 및 평가 (방향성 정확도)
                y_pred = model_copy.predict(X_val)
                y_pred_direction = np.where(y_pred > 0, 1, -1)
                accuracy = (y_pred_direction == y_val).mean()
                cv_scores.append(accuracy)
            
            avg_score = np.mean(cv_scores)
            results['scores'][model_name] = avg_score
            
            # 전체 데이터로 최종 학습
            model.fit(X_scaled, y_target)
            self.models[model_name] = model
            
            print(f"    ✅ {model_name}: {avg_score:.4f} 방향성 정확도")
        
        # 최고 성능 모델들 식별
        best_1h = max([k for k in results['scores'].keys() if '1h' in k], key=lambda x: results['scores'][x])
        best_2h = max([k for k in results['scores'].keys() if '2h' in k], key=lambda x: results['scores'][x])
        best_3h = max([k for k in results['scores'].keys() if '3h' in k], key=lambda x: results['scores'][x])
        
        results['best_models'] = {
            '1h': best_1h,
            '2h': best_2h, 
            '3h': best_3h
        }
        
        # 전체 평균 정확도
        avg_accuracy = np.mean(list(results['scores'].values()))
        results['ensemble_accuracy'] = avg_accuracy
        
        print(f"✅ 앙상블 학습 완료!")
        print(f"   🏆 최고 1H 모델: {best_1h} ({results['scores'][best_1h]:.4f})")
        print(f"   🏆 최고 2H 모델: {best_2h} ({results['scores'][best_2h]:.4f})")
        print(f"   🏆 최고 3H 모델: {best_3h} ({results['scores'][best_3h]:.4f})")
        print(f"   🎯 전체 앙상블 정확도: {avg_accuracy:.4f}")
        
        return results
    
    def predict_direction(self, df: pd.DataFrame, method='ensemble') -> Dict:
        """방향성 예측"""
        if not self.models:
            print("❌ 모델이 학습되지 않았습니다")
            return {}
        
        # 최근 데이터로 예측 준비
        recent_data = df[self.feature_columns].tail(self.lookback_hours).values.flatten()
        recent_scaled = self.scalers['scaler'].transform([recent_data])
        
        predictions = {}
        confidences = {}
        
        # 각 호라이즌별 예측
        for horizon in ['1h', '2h', '3h']:
            horizon_models = [k for k in self.models.keys() if horizon in k]
            horizon_preds = []
            
            for model_name in horizon_models:
                pred = self.models[model_name].predict(recent_scaled)[0]
                horizon_preds.append(pred)
            
            # 앙상블 예측 (평균)
            if method == 'ensemble' and len(horizon_preds) > 1:
                ensemble_pred = np.mean(horizon_preds)
                predictions[horizon] = 1 if ensemble_pred > 0 else -1
                confidences[horizon] = min(95, max(55, abs(ensemble_pred) * 50 + 50))
            elif horizon_preds:
                predictions[horizon] = 1 if horizon_preds[0] > 0 else -1
                confidences[horizon] = min(95, max(55, abs(horizon_preds[0]) * 50 + 50))
        
        current_price = df['close'].iloc[-1]
        
        result = {
            'current_price': current_price,
            'predictions': predictions,
            'confidences': confidences,
            'ensemble_confidence': np.mean(list(confidences.values())) if confidences else 50,
            'prediction_time': datetime.now().isoformat(),
            'method': method
        }
        
        print(f"🎯 방향성 예측 결과:")
        print(f"   현재가: ${current_price:,.2f}")
        for horizon, direction in predictions.items():
            conf = confidences.get(horizon, 50)
            direction_text = "상승" if direction == 1 else "하락"
            print(f"   {horizon}: {direction_text} (신뢰도: {conf:.1f}%)")
        
        return result

class Advanced90PercentSystem:
    """90% 정확도 도전 메인 시스템"""
    
    def __init__(self):
        self.feature_engineer = AdvancedFeatureEngineer()
        self.predictor = DirectionalPredictor()
        self.results = {}
        print("✅ 90% 정확도 도전 시스템 초기화 완료")
    
    def run_advanced_backtest(self, df: pd.DataFrame, test_size: int = 300) -> Dict:
        """고급 백테스트 실행"""
        print(f"🏆 90% 정확도 도전 백테스트 시작!")
        print(f"   테스트 데이터: {test_size}시간")
        print("=" * 60)
        
        # 1. 고급 특성공학
        print("1️⃣ 고급 특성공학 수행...")
        df_featured = self.feature_engineer.create_advanced_features(df.copy())
        
        if len(df_featured) < 500:
            print("❌ 데이터가 부족합니다 (최소 500행 필요)")
            return {}
        
        # 2. 학습/테스트 분할
        print("2️⃣ 데이터 분할...")
        train_size = len(df_featured) - test_size
        train_df = df_featured.iloc[:train_size]
        test_df = df_featured.iloc[train_size:]
        
        print(f"   📊 학습 데이터: {len(train_df)}시간")
        print(f"   📊 테스트 데이터: {len(test_df)}시간")
        
        # 3. 방향성 시퀀스 생성 및 학습
        print("3️⃣ 방향성 예측 모델 학습...")
        X_train, y1h_train, y2h_train, y3h_train = self.predictor.create_directional_sequences(train_df)
        
        if len(X_train) < 100:
            print("❌ 학습 시퀀스가 부족합니다")
            return {}
        
        # 4. 앙상블 모델 학습
        train_results = self.predictor.train_ensemble_models(X_train, y1h_train, y2h_train, y3h_train)
        
        # 5. 테스트 예측
        print("4️⃣ 테스트 예측 수행...")
        predictions = []
        actuals = []
        
        for i in range(len(test_df) - 10):  # 여유분 확보
            # 현재까지의 데이터로 예측
            current_data = pd.concat([
                train_df.tail(self.predictor.lookback_hours),
                test_df.iloc[:i+1] 
            ]).tail(len(train_df) + i + 1)
            
            if len(current_data) < self.predictor.lookback_hours + 10:
                continue
            
            # 1시간, 2시간, 3시간 예측
            pred_result = self.predictor.predict_direction(current_data)
            if pred_result and 'predictions' in pred_result:
                predictions.append(pred_result)
                
                # 실제값 수집 (1시간 후 방향성)
                if i + 1 < len(test_df):
                    current_price = test_df['close'].iloc[i]
                    future_price = test_df['close'].iloc[i + 1]
                    actual_direction = 1 if future_price > current_price else -1
                    actuals.append(actual_direction)
                else:
                    break
        
        if len(predictions) < 10:
            print("❌ 충분한 예측을 생성하지 못했습니다")
            return train_results
        
        # 6. 정확도 계산
        print("5️⃣ 최종 정확도 계산...")
        
        # 1시간 방향성 정확도
        pred_1h = [p['predictions'].get('1h', 0) for p in predictions]
        actual_1h = actuals[:len(pred_1h)]
        
        if len(pred_1h) == len(actual_1h) and len(pred_1h) > 0:
            direction_accuracy = (np.array(pred_1h) == np.array(actual_1h)).mean() * 100
        else:
            direction_accuracy = 0
        
        # 신뢰도 평균
        avg_confidence = np.mean([p.get('ensemble_confidence', 50) for p in predictions])
        
        # 최종 결과
        final_results = {
            **train_results,
            'test_predictions': len(predictions),
            'direction_accuracy': direction_accuracy,
            'avg_confidence': avg_confidence,
            'final_accuracy': direction_accuracy,  # 방향성이 핵심이므로
            'system_type': 'Advanced90Percent',
            'feature_count': len(self.feature_engineer.scalers),
            'model_count': len(self.predictor.models)
        }
        
        print("=" * 60)
        print("🏆 90% 정확도 도전 결과:")
        print(f"   📊 테스트 예측 수: {len(predictions)}")
        print(f"   🎯 방향성 정확도: {direction_accuracy:.2f}%")
        print(f"   🔮 평균 신뢰도: {avg_confidence:.1f}%")
        
        if direction_accuracy >= 90:
            print("🎉 90% 정확도 달성! 🎉")
        elif direction_accuracy >= 80:
            print("🔥 80%+ 고성능 달성!")
        elif direction_accuracy >= 70:
            print("✅ 70%+ 양호한 성능")
        else:
            print("📈 추가 개선 필요")
        
        return final_results

def load_advanced_data() -> pd.DataFrame:
    """고급 데이터 로딩"""
    print("📁 고급 데이터 로딩 중...")
    
    try:
        data_files = [
            'ai_optimized_3month_data/ai_matrix_complete.csv',
            'complete_indicators_data.csv'
        ]
        
        df = None
        for file in data_files:
            if os.path.exists(file):
                print(f"📊 {file} 로딩 중...")
                df = pd.read_csv(file)
                break
        
        if df is None:
            print("❌ 데이터 파일을 찾을 수 없습니다")
            return None
        
        # 가격 및 볼륨 컬럼 매핑
        price_candidates = ['close', 'legacy_market_data_avg_price', 'market_avg_price', 'price']
        volume_candidates = ['volume', 'onchain_blockchain_info_network_stats_trade_volume_btc', 'legacy_market_data_total_volume']
        
        price_col = None
        volume_col = None
        
        for col in price_candidates:
            if col in df.columns:
                price_col = col
                break
        
        for col in volume_candidates:
            if col in df.columns:
                volume_col = col
                break
        
        if not price_col:
            print("❌ 가격 컬럼을 찾을 수 없습니다")
            return None
        
        # 표준화
        if price_col != 'close':
            df['close'] = df[price_col]
        if volume_col and volume_col != 'volume':
            df['volume'] = df[volume_col]
        elif not volume_col:
            df['volume'] = 1000  # 더미 볼륨
        
        # 시간 정렬
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
        
        print(f"✅ 고급 데이터 로딩 완료: {len(df)}행, {len(df.columns)}열")
        return df
        
    except Exception as e:
        print(f"❌ 데이터 로딩 오류: {e}")
        return None

def main():
    """메인 실행"""
    print("🏆 고급 BTC 90% 정확도 도전 시스템")
    print("   목표: 방향성 예측 90% 정확도 달성")
    print("=" * 60)
    
    # 데이터 로딩
    df = load_advanced_data()
    if df is None:
        print("❌ 데이터 로딩 실패 - 시스템 종료")
        return
    
    # 90% 도전 시스템 실행
    system = Advanced90PercentSystem()
    results = system.run_advanced_backtest(df)
    
    if not results:
        print("❌ 백테스트 실패")
        return
    
    # 결과 저장
    results['system_version'] = 'Advanced90Percent_v1.0'
    results['timestamp'] = datetime.now().isoformat()
    
    with open('advanced_90_percent_results.json', 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print("=" * 60)
    print("🏆 90% 도전 백테스트 완료!")
    print(f"🎯 달성 정확도: {results.get('final_accuracy', 0):.2f}%")
    
    if results.get('final_accuracy', 0) >= 90:
        print("🎉🎉🎉 90% 정확도 달성! 성공! 🎉🎉🎉")
    else:
        print("🚀 90% 정확도 도전 지속 - 더 고도화 필요")
    
    print("📄 상세 결과: advanced_90_percent_results.json")

if __name__ == "__main__":
    main()