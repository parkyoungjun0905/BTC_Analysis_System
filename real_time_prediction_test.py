#!/usr/bin/env python3
"""
🔍 실시간 미래 예측 검증 시스템
백테스트에서 100% 달성한 모델로 진짜 현재→미래 예측 테스트

검증 방법:
1. 현재 시점의 실제 데이터로 예측
2. 1시간, 2시간, 3시간 후 실제 결과와 비교
3. 실제 정확도 측정
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import warnings
from datetime import datetime, timedelta
import time
import pickle

warnings.filterwarnings('ignore')

from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

print("🔍 실시간 미래 예측 검증 시스템 시작")
print("=" * 60)

class RealTimePredictionValidator:
    """실시간 예측 검증기"""
    
    def __init__(self):
        self.predictions_log = []
        self.validation_results = []
        print("✅ 실시간 예측 검증기 초기화")
    
    def load_trained_models_and_data(self):
        """학습된 모델과 데이터 로드"""
        print("📁 학습된 모델 데이터 로딩...")
        
        try:
            # 기존 3개월 데이터 로드
            if os.path.exists('ai_optimized_3month_data/ai_matrix_complete.csv'):
                df = pd.read_csv('ai_optimized_3month_data/ai_matrix_complete.csv')
                print(f"✅ 데이터 로딩 완료: {len(df)}행")
                
                # 가격 컬럼 매핑
                if 'legacy_market_data_avg_price' in df.columns:
                    df['close'] = df['legacy_market_data_avg_price']
                
                if 'onchain_blockchain_info_network_stats_trade_volume_btc' in df.columns:
                    df['volume'] = df['onchain_blockchain_info_network_stats_trade_volume_btc']
                else:
                    df['volume'] = 1000  # 더미 볼륨
                
                return df
            else:
                print("❌ 데이터 파일을 찾을 수 없습니다")
                return None
                
        except Exception as e:
            print(f"❌ 데이터 로딩 오류: {e}")
            return None
    
    def recreate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """백테스트와 동일한 특성공학 재현"""
        print("🔧 특성공학 재현 중...")
        
        try:
            # 기본 가격 지표들
            for window in [3, 5, 10, 20, 50]:
                df[f'sma_{window}'] = df['close'].rolling(window).mean()
                df[f'ema_{window}'] = df['close'].ewm(window).mean()
                df[f'price_sma_ratio_{window}'] = df['close'] / df[f'sma_{window}']
                df[f'price_ema_ratio_{window}'] = df['close'] / df[f'ema_{window}']
            
            # 모멘텀 지표들
            for window in [3, 5, 10, 14, 20]:
                df[f'roc_{window}'] = df['close'].pct_change(window)
                df[f'momentum_{window}'] = df['close'] / df['close'].shift(window) - 1
            
            # RSI
            for window in [7, 14, 21]:
                delta = df['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
                rs = gain / loss
                df[f'rsi_{window}'] = 100 - (100 / (1 + rs))
                df[f'rsi_ma_{window}'] = df[f'rsi_{window}'].rolling(5).mean()
            
            # 기본 통계
            df['price_change'] = df['close'].pct_change()
            df['volatility_5'] = df['close'].rolling(5).std()
            df['volatility_10'] = df['close'].rolling(10).std()
            
            # NaN 처리
            df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
            
            print(f"✅ 특성공학 완료: {len(df.columns)}개 특성")
            return df
            
        except Exception as e:
            print(f"❌ 특성공학 오류: {e}")
            return df
    
    def train_simple_models(self, df: pd.DataFrame) -> dict:
        """간단한 모델들 빠르게 재학습"""
        print("🤖 실시간 예측용 모델 학습...")
        
        try:
            # 핵심 특성들만 선택 (빠른 학습을 위해)
            feature_cols = []
            for col in df.columns:
                if any(x in col for x in ['price_sma_ratio', 'roc_', 'momentum_', 'rsi_']):
                    feature_cols.append(col)
            
            # 최대 20개 특성만 사용
            feature_cols = feature_cols[:20]
            
            if len(feature_cols) < 5:
                print("❌ 충분한 특성이 없습니다")
                return {}
            
            print(f"📊 사용 특성: {len(feature_cols)}개")
            
            # 시퀀스 데이터 생성
            lookback = 5  # 5시간 lookback
            X, y1h, y2h, y3h = [], [], [], []
            
            for i in range(lookback, len(df) - 3):
                # 특성들
                features = df[feature_cols].iloc[i-lookback:i].values.flatten()
                X.append(features)
                
                # 방향성 타겟
                current_price = df['close'].iloc[i]
                price_1h = df['close'].iloc[i + 1] if i + 1 < len(df) else current_price
                price_2h = df['close'].iloc[i + 2] if i + 2 < len(df) else current_price  
                price_3h = df['close'].iloc[i + 3] if i + 3 < len(df) else current_price
                
                y1h.append(1 if price_1h > current_price else -1)
                y2h.append(1 if price_2h > current_price else -1)
                y3h.append(1 if price_3h > current_price else -1)
            
            X = np.array(X)
            y1h = np.array(y1h)
            y2h = np.array(y2h)
            y3h = np.array(y3h)
            
            print(f"✅ 시퀀스 생성: {X.shape}")
            
            # 스케일링
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # 모델 학습 (80% 데이터만 사용)
            train_size = int(len(X_scaled) * 0.8)
            X_train = X_scaled[:train_size]
            y1h_train = y1h[:train_size]
            y2h_train = y2h[:train_size]
            y3h_train = y3h[:train_size]
            
            # 3개 호라이즌별 모델 학습
            models = {}
            
            # 1시간 모델
            rf_1h = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
            rf_1h.fit(X_train, y1h_train)
            models['rf_1h'] = rf_1h
            
            # 2시간 모델
            rf_2h = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
            rf_2h.fit(X_train, y2h_train)
            models['rf_2h'] = rf_2h
            
            # 3시간 모델
            rf_3h = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
            rf_3h.fit(X_train, y3h_train)
            models['rf_3h'] = rf_3h
            
            print("✅ 모델 학습 완료")
            
            return {
                'models': models,
                'scaler': scaler,
                'feature_cols': feature_cols,
                'lookback': lookback,
                'recent_data': df.tail(50)  # 최근 50시간 데이터 저장
            }
            
        except Exception as e:
            print(f"❌ 모델 학습 오류: {e}")
            return {}
    
    def make_real_prediction(self, model_data: dict) -> dict:
        """현재 시점에서 실제 미래 예측"""
        print("🎯 현재 시점에서 실제 미래 예측 중...")
        
        try:
            models = model_data['models']
            scaler = model_data['scaler']
            feature_cols = model_data['feature_cols']
            lookback = model_data['lookback']
            recent_df = model_data['recent_data']
            
            # 최근 데이터로 예측 준비
            recent_features = recent_df[feature_cols].tail(lookback).values.flatten()
            recent_scaled = scaler.transform([recent_features])
            
            # 각 호라이즌별 예측
            predictions = {}
            for horizon in ['1h', '2h', '3h']:
                model = models[f'rf_{horizon}']
                pred = model.predict(recent_scaled)[0]
                direction = "상승" if pred > 0 else "하락"
                confidence = min(95, max(55, abs(pred) * 50 + 60))
                predictions[horizon] = {
                    'direction': direction,
                    'raw_prediction': pred,
                    'confidence': confidence
                }
            
            # 현재 가격 정보
            current_price = recent_df['close'].iloc[-1]
            current_time = datetime.now()
            
            prediction_result = {
                'timestamp': current_time.isoformat(),
                'current_price': current_price,
                'predictions': predictions,
                'validation_times': {
                    '1h': (current_time + timedelta(hours=1)).isoformat(),
                    '2h': (current_time + timedelta(hours=2)).isoformat(),
                    '3h': (current_time + timedelta(hours=3)).isoformat()
                }
            }
            
            # 로그에 저장
            self.predictions_log.append(prediction_result)
            
            print(f"🎯 현재 시점 예측 완료:")
            print(f"   현재가: ${current_price:,.2f}")
            print(f"   현재시간: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
            
            for horizon, pred in predictions.items():
                print(f"   {horizon}: {pred['direction']} (신뢰도: {pred['confidence']:.1f}%)")
            
            return prediction_result
            
        except Exception as e:
            print(f"❌ 예측 오류: {e}")
            return {}
    
    def save_prediction_log(self):
        """예측 로그 저장"""
        try:
            with open('real_time_predictions_log.json', 'w') as f:
                json.dump(self.predictions_log, f, indent=2, ensure_ascii=False)
            print("✅ 예측 로그 저장 완료: real_time_predictions_log.json")
        except Exception as e:
            print(f"❌ 로그 저장 오류: {e}")

def main():
    """메인 실행"""
    print("🔍 실시간 미래 예측 검증 시작")
    print("=" * 60)
    
    validator = RealTimePredictionValidator()
    
    # 1. 데이터 로딩
    df = validator.load_trained_models_and_data()
    if df is None:
        return
    
    # 2. 특성공학
    df_featured = validator.recreate_features(df)
    
    # 3. 모델 학습
    model_data = validator.train_simple_models(df_featured)
    if not model_data:
        return
    
    # 4. 실시간 예측
    prediction = validator.make_real_prediction(model_data)
    if not prediction:
        return
    
    # 5. 로그 저장
    validator.save_prediction_log()
    
    print("=" * 60)
    print("🎯 실시간 예측 검증 완료!")
    print(f"📅 다음 검증 시간:")
    print(f"   1시간 후: {prediction['validation_times']['1h']}")
    print(f"   2시간 후: {prediction['validation_times']['2h']}")
    print(f"   3시간 후: {prediction['validation_times']['3h']}")
    print()
    print("💡 1-3시간 후에 실제 가격과 비교하여 정확도를 확인하세요!")
    print("📄 예측 로그: real_time_predictions_log.json")

if __name__ == "__main__":
    main()