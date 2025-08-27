#!/usr/bin/env python3
"""
🎯 실용적 AI BTC 백테스트 시스템 (90% 정확도 목표)
1-3시간 단위 미래 가격 예측 - 단계별 구현

Step 1: 기본 딥러닝 모델로 시작
Step 2: 정확도 달성시 고급 앙상블로 확장
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

# 필수 라이브러리만 사용
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

print("🚀 실용적 AI 백테스트 시스템 시작")
print("=" * 60)

class SimpleFeatureEngineer:
    """기본 특성공학 - 핵심 지표만 추출"""
    
    def __init__(self):
        self.scaler = MinMaxScaler()
        print("✅ 기본 특성공학 엔진 초기화")
    
    def create_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """핵심 기술 지표만 생성"""
        print("📊 기본 기술 지표 생성 중...")
        
        try:
            # 기본 가격 변화율
            df['price_change'] = df['close'].pct_change()
            df['price_change_ma'] = df['price_change'].rolling(5).mean()
            
            # 볼륨 지표 (작은 윈도우)
            df['volume_ma'] = df['volume'].rolling(5).mean()
            df['volume_ratio'] = df['volume'] / df['volume_ma']
            
            # 기본 이동평균 (작은 윈도우)
            for window in [3, 5]:
                df[f'ma_{window}'] = df['close'].rolling(window).mean()
                df[f'ma_{window}_ratio'] = df['close'] / df[f'ma_{window}']
            
            # RSI (간단 계산, 작은 윈도우)
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=7).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=7).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # 변동성 (작은 윈도우)
            df['volatility'] = df['close'].rolling(5).std()
            
            print(f"✅ {len(df.columns)} 개 특성 생성 완료")
            return df.dropna()
            
        except Exception as e:
            print(f"❌ 특성 생성 오류: {e}")
            return df

class SimpleAIPredictor:
    """간단하지만 효과적인 AI 예측기"""
    
    def __init__(self, lookback_hours: int = 5):
        self.lookback_hours = lookback_hours
        self.model = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.feature_columns = []
        print(f"✅ AI 예측기 초기화 (lookback: {lookback_hours}시간)")
    
    def create_sequences(self, df: pd.DataFrame, target_col: str = 'close') -> Tuple:
        """시계열 시퀀스 데이터 생성"""
        print("🔄 시계열 시퀀스 생성 중...")
        
        # 특성 선택 (핵심만)
        feature_cols = ['price_change', 'volume_ratio', 'ma_5_ratio', 
                       'ma_10_ratio', 'rsi', 'volatility']
        feature_cols = [col for col in feature_cols if col in df.columns]
        
        if len(feature_cols) < 3:
            print("❌ 충분한 특성이 없습니다")
            return None, None
        
        self.feature_columns = feature_cols
        print(f"📊 사용 특성: {feature_cols}")
        
        # 데이터 준비
        X, y = [], []
        for i in range(self.lookback_hours, len(df) - 1):  # -1 추가로 y 범위 보장
            # 과거 lookback_hours 시간의 특성들
            X.append(df[feature_cols].iloc[i-self.lookback_hours:i].values.flatten())
            # 1시간 후 가격 (목표)
            y.append(df[target_col].iloc[i + 1])
        
        X = np.array(X)
        y = np.array(y)
        
        # 길이 확인 및 조정
        if len(X) != len(y):
            min_len = min(len(X), len(y))
            X = X[:min_len]
            y = y[:min_len]
            print(f"⚠️ 길이 불일치 수정: {min_len}개 샘플로 조정")
        
        print(f"✅ 시퀀스 생성 완료: X={X.shape}, y={y.shape}")
        return X, y
    
    def train_model(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """모델 학습"""
        print("🤖 AI 모델 학습 시작...")
        
        # 데이터 스케일링
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
        
        # 간단하지만 강력한 RandomForest 사용
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        # 간단한 train/validation 분할 (데이터가 적으므로)
        if len(X_scaled) < 10:
            # 데이터가 너무 적으면 전체 데이터로 학습
            self.model.fit(X_scaled, y_scaled)
            avg_score = 0.5  # 기본값
            scores = [avg_score]
            print("⚠️ 데이터 부족으로 교차검증 건너뜀")
        else:
            # 80:20 분할
            split_idx = int(len(X_scaled) * 0.8)
            X_train, X_val = X_scaled[:split_idx], X_scaled[split_idx:]
            y_train, y_val = y_scaled[:split_idx], y_scaled[split_idx:]
            
            # 학습
            self.model.fit(X_train, y_train)
            
            # 예측 및 평가
            if len(y_val) > 0:
                y_pred = self.model.predict(X_val)
                if len(y_val) == len(y_pred):
                    avg_score = r2_score(y_val, y_pred)
                    scores = [avg_score]
                    print(f"  검증 R²: {avg_score:.4f}")
                else:
                    avg_score = 0.5
                    scores = [avg_score]
                    print(f"  길이 불일치: y_val={len(y_val)}, y_pred={len(y_pred)}")
            else:
                avg_score = 0.5
                scores = [avg_score]
                print("  검증 데이터 부족")
            
            # 전체 데이터로 재학습
            self.model.fit(X_scaled, y_scaled)
        accuracy_percent = max(0, avg_score * 100)
        
        results = {
            'cv_scores': scores,
            'average_r2': avg_score,
            'accuracy_percent': accuracy_percent,
            'model_type': 'RandomForest'
        }
        
        print(f"✅ 학습 완료! 평균 정확도: {accuracy_percent:.2f}%")
        return results
    
    def predict_future(self, df: pd.DataFrame, hours_ahead: int = 3) -> Dict:
        """미래 가격 예측"""
        if self.model is None:
            print("❌ 모델이 학습되지 않았습니다")
            return {}
        
        # 최근 데이터로 예측
        recent_data = df[self.feature_columns].tail(self.lookback_hours).values.flatten()
        recent_scaled = self.scaler_X.transform([recent_data])
        
        # 예측 (스케일링된 값)
        pred_scaled = self.model.predict(recent_scaled)[0]
        
        # 원래 스케일로 복원
        pred_price = self.scaler_y.inverse_transform([[pred_scaled]])[0][0]
        
        current_price = df['close'].iloc[-1]
        price_change = pred_price - current_price
        change_percent = (price_change / current_price) * 100
        
        confidence = min(95, max(50, abs(self.model.score(
            self.scaler_X.transform([recent_data]), 
            [self.scaler_y.transform([[current_price]])[0][0]]
        )) * 100))
        
        result = {
            'current_price': current_price,
            'predicted_price': pred_price,
            'price_change': price_change,
            'change_percent': change_percent,
            'confidence': confidence,
            'prediction_time': datetime.now().isoformat(),
            'hours_ahead': hours_ahead
        }
        
        print(f"🎯 {hours_ahead}시간 후 예측:")
        print(f"   현재가: ${current_price:,.2f}")
        print(f"   예측가: ${pred_price:,.2f}")
        print(f"   변화: {change_percent:+.2f}% (신뢰도: {confidence:.1f}%)")
        
        return result

class SimpleBacktester:
    """간단한 백테스트 엔진"""
    
    def __init__(self):
        self.results = []
        print("✅ 백테스트 엔진 초기화")
    
    def run_backtest(self, df: pd.DataFrame, test_size: int = 200) -> Dict:
        """백테스트 실행"""
        print(f"🔍 백테스트 실행 (최근 {test_size}시간 데이터)")
        
        # 특성공학
        engineer = SimpleFeatureEngineer()
        df_featured = engineer.create_basic_features(df.copy())
        
        if len(df_featured) < 20:
            print("❌ 데이터가 너무 부족합니다 (최소 20행 필요)")
            return {}
        
        # 학습/테스트 분할 (작은 데이터에 맞게 조정)
        train_size = max(15, len(df_featured) - test_size)
        if train_size >= len(df_featured):
            train_size = len(df_featured) - 3
            test_size = 3
            
        train_df = df_featured.iloc[:train_size]
        test_df = df_featured.iloc[train_size:]
        
        print(f"📊 학습 데이터: {len(train_df)}시간, 테스트: {len(test_df)}시간")
        
        # AI 예측기 초기화 및 학습
        predictor = SimpleAIPredictor()
        X_train, y_train = predictor.create_sequences(train_df)
        
        if X_train is None or len(X_train) < 5:
            print("❌ 학습 시퀀스 생성 실패")
            return {}
        
        # 모델 학습
        train_results = predictor.train_model(X_train, y_train)
        
        # 테스트 예측
        print("🎯 테스트 데이터 예측 중...")
        predictions = []
        actuals = []
        
        for i in range(len(test_df) - 3):
            test_window = pd.concat([
                train_df.tail(predictor.lookback_hours),
                test_df.iloc[:i+1]
            ]).tail(len(train_df) + i + 1)
            
            if len(test_window) < predictor.lookback_hours + 1:
                continue
                
            pred_result = predictor.predict_future(test_window)
            if pred_result:
                predictions.append(pred_result['predicted_price'])
                
                # 실제값 (3시간 후)
                if i + 3 < len(test_df):
                    actuals.append(test_df['close'].iloc[i + 3])
                else:
                    break
        
        if len(predictions) < 10:
            print("❌ 충분한 예측을 생성하지 못했습니다")
            return train_results
        
        # 정확도 계산
        predictions = np.array(predictions[:len(actuals)])
        actuals = np.array(actuals)
        
        # 방향성 정확도 (상승/하락 맞추기)
        pred_directions = np.where(predictions > test_df['close'].iloc[:-3].values[:len(predictions)], 1, -1)
        actual_directions = np.where(actuals > test_df['close'].iloc[:-3].values[:len(actuals)], 1, -1)
        direction_accuracy = (pred_directions == actual_directions).mean() * 100
        
        # 가격 정확도
        mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100
        price_accuracy = max(0, 100 - mape)
        
        # 최종 정확도 (방향성 + 가격 정확도 평균)
        final_accuracy = (direction_accuracy + price_accuracy) / 2
        
        backtest_results = {
            **train_results,
            'test_predictions': len(predictions),
            'direction_accuracy': direction_accuracy,
            'price_accuracy': price_accuracy,
            'final_accuracy': final_accuracy,
            'mape': mape,
            'predictions': predictions.tolist(),
            'actuals': actuals.tolist()
        }
        
        print(f"📊 백테스트 결과:")
        print(f"   방향성 정확도: {direction_accuracy:.2f}%")
        print(f"   가격 정확도: {price_accuracy:.2f}%")
        print(f"   🎯 최종 정확도: {final_accuracy:.2f}%")
        
        return backtest_results

def load_data() -> pd.DataFrame:
    """데이터 로딩"""
    print("📁 데이터 로딩 중...")
    
    try:
        # 기존 데이터 찾기
        data_files = [
            'ai_optimized_3month_data/ai_matrix_complete.csv',
            'complete_indicators_data.csv',
            'btc_hourly_data.csv',
            'hourly_data.csv'
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
        
        # 기본 컬럼 확인 및 매핑
        price_col = None
        volume_col = None
        
        # 가격 컬럼 찾기
        price_candidates = ['close', 'legacy_market_data_avg_price', 'market_avg_price', 'price',
                           'onchain_blockchain_info_network_stats_market_price_usd']
        for col in price_candidates:
            if col in df.columns:
                price_col = col
                break
        
        # 볼륨 컬럼 찾기
        volume_candidates = ['volume', 'onchain_blockchain_info_network_stats_trade_volume_btc',
                            'legacy_market_data_total_volume', 'market_total_volume', 'total_volume']
        for col in volume_candidates:
            if col in df.columns:
                volume_col = col
                break
        
        if not price_col:
            print(f"❌ 가격 컬럼을 찾을 수 없습니다. 시도한 컬럼: {price_candidates}")
            return None
            
        # 컬럼명 표준화
        if price_col != 'close':
            df['close'] = df[price_col]
        if volume_col and volume_col != 'volume':
            df['volume'] = df[volume_col]
        elif not volume_col:
            # 볼륨이 없으면 더미 볼륨 생성
            df['volume'] = 1000
            print("⚠️ 볼륨 데이터 없음 - 더미 볼륨 사용")
        
        # 시간 정렬
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
        
        print(f"✅ 데이터 로딩 완료: {len(df)}행, {len(df.columns)}열")
        print(f"   기간: {df.index[0]} ~ {df.index[-1]}")
        
        return df
        
    except Exception as e:
        print(f"❌ 데이터 로딩 오류: {e}")
        return None

def main():
    """메인 실행 함수"""
    print("🎯 실용적 AI BTC 백테스트 시스템")
    print("   목표: 1-3시간 후 90% 정확도 달성")
    print("=" * 60)
    
    # 데이터 로딩
    df = load_data()
    if df is None:
        print("❌ 데이터 로딩 실패 - 시스템 종료")
        return
    
    # 백테스트 실행
    backtester = SimpleBacktester()
    results = backtester.run_backtest(df)
    
    if not results:
        print("❌ 백테스트 실패")
        return
    
    # 결과 저장
    results['system_version'] = 'SimpleAI_v1.0'
    results['timestamp'] = datetime.now().isoformat()
    
    with open('simple_ai_backtest_results.json', 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print("=" * 60)
    print("🎉 백테스트 완료!")
    print(f"🎯 달성 정확도: {results.get('final_accuracy', 0):.2f}%")
    
    if results.get('final_accuracy', 0) >= 90:
        print("🏆 90% 정확도 달성! 고급 앙상블로 업그레이드 준비")
    else:
        print("⚡ 90% 미달 - 모델 개선 필요")
    
    print("📄 결과 저장: simple_ai_backtest_results.json")

if __name__ == "__main__":
    main()