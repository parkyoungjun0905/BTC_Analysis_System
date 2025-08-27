#!/usr/bin/env python3
"""
💰 정확한 BTC 가격 예측 시스템
방향성이 아닌 구체적인 달러 가격을 예측

목표: $109,742 → 1시간 후 $108,xxx, 2시간 후 $107,xxx 등
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import warnings
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

print("💰 정확한 BTC 가격 예측 시스템 시작")
print("=" * 60)

class PrecisePricePredictionSystem:
    """정확한 가격 예측 시스템"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_cols = []
        print("✅ 정확한 가격 예측 시스템 초기화")
    
    def load_and_prepare_data(self):
        """데이터 로딩 및 준비"""
        print("📁 데이터 로딩 중...")
        
        try:
            # 3개월 데이터 로드
            if os.path.exists('ai_optimized_3month_data/ai_matrix_complete.csv'):
                df = pd.read_csv('ai_optimized_3month_data/ai_matrix_complete.csv')
                print(f"✅ 데이터 로딩: {len(df)}행")
                
                # 가격 컬럼 설정
                if 'legacy_market_data_avg_price' in df.columns:
                    df['close'] = df['legacy_market_data_avg_price']
                    print(f"✅ 가격 범위: ${df['close'].min():,.0f} ~ ${df['close'].max():,.0f}")
                
                return df
            else:
                print("❌ 데이터 파일을 찾을 수 없습니다")
                return None
                
        except Exception as e:
            print(f"❌ 데이터 로딩 오류: {e}")
            return None
    
    def create_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """가격 예측을 위한 고급 특성 생성"""
        print("🔧 가격 예측 특성 생성 중...")
        
        try:
            # 1. 가격 변화율 패턴
            for window in [1, 2, 3, 5, 10, 24]:
                df[f'price_change_{window}h'] = df['close'].pct_change(window)
                df[f'price_momentum_{window}h'] = df['close'] / df['close'].shift(window) - 1
            
            # 2. 이동평균 및 가격 관계
            for window in [3, 6, 12, 24, 48]:
                df[f'sma_{window}'] = df['close'].rolling(window).mean()
                df[f'price_to_sma_{window}'] = df['close'] / df[f'sma_{window}']
                df[f'sma_slope_{window}'] = df[f'sma_{window}'].diff(3)
            
            # 3. 변동성 지표
            for window in [6, 12, 24]:
                df[f'volatility_{window}'] = df['close'].rolling(window).std()
                df[f'volatility_ratio_{window}'] = df[f'volatility_{window}'] / df[f'volatility_{window}'].rolling(48).mean()
            
            # 4. 고급 기술 지표
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            df['rsi_slope'] = df['rsi'].diff(3)
            
            # 볼린저 밴드
            df['bb_middle'] = df['close'].rolling(20).mean()
            df['bb_std'] = df['close'].rolling(20).std()
            df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
            df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            
            # 5. 시간 패턴 (시간대별 가격 경향)
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df['hour'] = df['timestamp'].dt.hour
                df['day_of_week'] = df['timestamp'].dt.dayofweek
                # 순환 인코딩
                df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
                df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
            
            # 6. 추가 패턴들
            # 가격 가속도
            df['price_acceleration'] = df['close'].diff().diff()
            
            # 최근 고점/저점까지의 거리
            df['distance_to_high_24h'] = df['close'] / df['close'].rolling(24).max()
            df['distance_to_low_24h'] = df['close'] / df['close'].rolling(24).min()
            
            # NaN 처리
            df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
            
            print(f"✅ 특성 생성 완료: {len(df.columns)}개 특성")
            return df
            
        except Exception as e:
            print(f"❌ 특성 생성 오류: {e}")
            return df
    
    def prepare_price_prediction_data(self, df: pd.DataFrame) -> dict:
        """가격 예측용 데이터 준비"""
        print("📊 가격 예측 데이터 준비 중...")
        
        try:
            # 가격 예측에 유효한 특성들 선택
            feature_candidates = []
            for col in df.columns:
                if any(keyword in col for keyword in [
                    'price_change', 'price_momentum', 'price_to_sma', 'sma_slope',
                    'volatility', 'rsi', 'bb_position', 'bb_width', 
                    'hour_sin', 'hour_cos', 'price_acceleration',
                    'distance_to_high', 'distance_to_low'
                ]):
                    if col != 'close' and not col.startswith('timestamp'):
                        feature_candidates.append(col)
            
            # 상위 30개 특성만 사용 (과적합 방지)
            self.feature_cols = feature_candidates[:30]
            
            print(f"📊 사용 특성 {len(self.feature_cols)}개:")
            for i, col in enumerate(self.feature_cols[:10]):
                print(f"   {i+1:2d}. {col}")
            if len(self.feature_cols) > 10:
                print(f"   ... 외 {len(self.feature_cols)-10}개")
            
            # 시계열 데이터 생성
            lookback = 12  # 12시간 lookback (더 많은 패턴 학습)
            X, y1h, y2h, y3h = [], [], [], []
            
            for i in range(lookback, len(df) - 3):
                # 특성 벡터 (12시간 평탄화)
                features = df[self.feature_cols].iloc[i-lookback:i].values.flatten()
                X.append(features)
                
                # 타겟: 실제 미래 가격 (방향성이 아닌!)
                current_price = df['close'].iloc[i]
                price_1h = df['close'].iloc[i + 1] if i + 1 < len(df) else current_price
                price_2h = df['close'].iloc[i + 2] if i + 2 < len(df) else current_price
                price_3h = df['close'].iloc[i + 3] if i + 3 < len(df) else current_price
                
                y1h.append(price_1h)  # 실제 1시간 후 가격
                y2h.append(price_2h)  # 실제 2시간 후 가격
                y3h.append(price_3h)  # 실제 3시간 후 가격
            
            X = np.array(X)
            y1h = np.array(y1h)
            y2h = np.array(y2h)
            y3h = np.array(y3h)
            
            print(f"✅ 데이터 준비 완료: X={X.shape}")
            print(f"   가격 범위: ${y1h.min():,.0f} ~ ${y1h.max():,.0f}")
            
            return {
                'X': X, 'y1h': y1h, 'y2h': y2h, 'y3h': y3h,
                'recent_data': df.tail(50)
            }
            
        except Exception as e:
            print(f"❌ 데이터 준비 오류: {e}")
            return {}
    
    def train_price_models(self, data: dict) -> dict:
        """정확한 가격 예측 모델 학습"""
        print("🤖 정확한 가격 예측 모델 학습 중...")
        
        try:
            X = data['X']
            y1h = data['y1h']
            y2h = data['y2h']
            y3h = data['y3h']
            
            # 스케일링 (특성과 타겟 모두)
            self.scalers['X'] = StandardScaler()
            X_scaled = self.scalers['X'].fit_transform(X)
            
            # 타겟 스케일링 (가격 범위가 크므로)
            self.scalers['y1h'] = StandardScaler()
            self.scalers['y2h'] = StandardScaler()  
            self.scalers['y3h'] = StandardScaler()
            
            y1h_scaled = self.scalers['y1h'].fit_transform(y1h.reshape(-1, 1)).flatten()
            y2h_scaled = self.scalers['y2h'].fit_transform(y2h.reshape(-1, 1)).flatten()
            y3h_scaled = self.scalers['y3h'].fit_transform(y3h.reshape(-1, 1)).flatten()
            
            # 학습/검증 분할
            train_size = int(len(X_scaled) * 0.85)  # 85% 학습용
            
            X_train = X_scaled[:train_size]
            X_val = X_scaled[train_size:]
            
            # 각 호라이즌별 모델 학습
            models = {}
            validation_scores = {}
            
            for horizon, y_scaled, y_original in [
                ('1h', y1h_scaled, y1h), 
                ('2h', y2h_scaled, y2h), 
                ('3h', y3h_scaled, y3h)
            ]:
                print(f"  🔧 {horizon} 모델 학습 중...")
                
                y_train = y_scaled[:train_size]
                y_val_scaled = y_scaled[train_size:]
                y_val_original = y_original[train_size:]
                
                # RandomForest 회귀모델 (가격 예측용)
                rf = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=15,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1
                )
                
                rf.fit(X_train, y_train)
                
                # 검증 성능 평가
                y_pred_scaled = rf.predict(X_val)
                y_pred_original = self.scalers[f'y{horizon}'].inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
                
                mae = mean_absolute_error(y_val_original, y_pred_original)
                rmse = np.sqrt(mean_squared_error(y_val_original, y_pred_original))
                mape = np.mean(np.abs((y_val_original - y_pred_original) / y_val_original)) * 100
                
                models[horizon] = rf
                validation_scores[horizon] = {
                    'mae': mae,
                    'rmse': rmse, 
                    'mape': mape
                }
                
                print(f"    ✅ {horizon}: MAE=${mae:.0f}, RMSE=${rmse:.0f}, MAPE={mape:.2f}%")
            
            self.models = models
            
            return {
                'models': models,
                'validation_scores': validation_scores,
                'feature_importance': self._get_feature_importance()
            }
            
        except Exception as e:
            print(f"❌ 모델 학습 오류: {e}")
            return {}
    
    def _get_feature_importance(self):
        """특성 중요도 계산"""
        importance = {}
        for horizon, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                feature_imp = list(zip(
                    [f"feat_{i}" for i in range(len(model.feature_importances_))],
                    model.feature_importances_
                ))
                feature_imp.sort(key=lambda x: x[1], reverse=True)
                importance[horizon] = feature_imp[:10]  # 상위 10개만
        return importance
    
    def predict_exact_prices(self, data: dict) -> dict:
        """현재 시점에서 정확한 미래 가격 예측"""
        print("💰 현재 시점에서 정확한 미래 가격 예측 중...")
        
        try:
            recent_df = data['recent_data']
            current_price = recent_df['close'].iloc[-1]
            current_time = datetime.now()
            
            # 최근 특성 데이터 준비
            lookback = 12
            recent_features = recent_df[self.feature_cols].tail(lookback).values.flatten()
            recent_scaled = self.scalers['X'].transform([recent_features])
            
            # 각 호라이즌별 가격 예측
            price_predictions = {}
            
            for horizon in ['1h', '2h', '3h']:
                model = self.models[horizon]
                pred_scaled = model.predict(recent_scaled)[0]
                
                # 스케일 복원하여 실제 달러 가격으로 변환
                pred_price = self.scalers[f'y{horizon}'].inverse_transform([[pred_scaled]])[0][0]
                
                price_change = pred_price - current_price
                change_percent = (price_change / current_price) * 100
                
                price_predictions[horizon] = {
                    'predicted_price': round(pred_price, 2),
                    'price_change': round(price_change, 2),
                    'change_percent': round(change_percent, 3),
                    'confidence': min(95, max(60, 80 + abs(change_percent)))  # 변화량에 따른 신뢰도
                }
            
            result = {
                'timestamp': current_time.isoformat(),
                'current_price': current_price,
                'price_predictions': price_predictions,
                'validation_times': {
                    '1h': (current_time + timedelta(hours=1)).isoformat(),
                    '2h': (current_time + timedelta(hours=2)).isoformat(), 
                    '3h': (current_time + timedelta(hours=3)).isoformat()
                }
            }
            
            print(f"💰 정확한 가격 예측 완료:")
            print(f"   현재가: ${current_price:,.2f}")
            print(f"   현재시간: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print()
            
            for horizon, pred in price_predictions.items():
                print(f"   {horizon}: ${pred['predicted_price']:,.2f} "
                      f"({pred['change_percent']:+.2f}%, "
                      f"${pred['price_change']:+,.0f})")
            
            return result
            
        except Exception as e:
            print(f"❌ 가격 예측 오류: {e}")
            return {}
    
    def save_prediction(self, prediction: dict):
        """예측 결과 저장"""
        try:
            with open('precise_price_predictions.json', 'w') as f:
                json.dump([prediction], f, indent=2, ensure_ascii=False)
            print("✅ 정확한 가격 예측 저장: precise_price_predictions.json")
        except Exception as e:
            print(f"❌ 저장 오류: {e}")

def main():
    """메인 실행"""
    print("💰 정확한 BTC 가격 예측 시스템")
    print("=" * 60)
    
    system = PrecisePricePredictionSystem()
    
    # 1. 데이터 로딩
    df = system.load_and_prepare_data()
    if df is None:
        return
    
    # 2. 특성 생성
    df_featured = system.create_price_features(df)
    
    # 3. 데이터 준비
    data = system.prepare_price_prediction_data(df_featured)
    if not data:
        return
    
    # 4. 모델 학습
    training_results = system.train_price_models(data)
    if not training_results:
        return
    
    # 5. 현재 시점 예측
    prediction = system.predict_exact_prices(data)
    if not prediction:
        return
    
    # 6. 결과 저장
    system.save_prediction(prediction)
    
    print("=" * 60)
    print("💰 정확한 가격 예측 완료!")
    print("📄 결과: precise_price_predictions.json")
    print()
    print("🕐 검증 일정:")
    for horizon, time_str in prediction['validation_times'].items():
        pred_price = prediction['price_predictions'][horizon]['predicted_price']
        print(f"   {horizon}: {time_str} → ${pred_price:,.2f}")

if __name__ == "__main__":
    main()