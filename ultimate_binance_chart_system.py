"""
🚀 Ultimate Binance-Style BTC Chart System
- 바이낸스처럼 1m, 5m, 15m, 1h, 4h, 1d, 1w 시간축 지원
- 6개월 학습 데이터 기반 고정확도 예측
- 실시간 API 연동
- 95%+ 정확도 목표
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# 필수 라이브러리
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("❌ Plotly 미설치 - pip install plotly")
    exit()

try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
    from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
    from sklearn.svm import SVR
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler, RobustScaler
    from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
    from sklearn.model_selection import TimeSeriesSplit
    import ta  # 기술적 지표
    import yfinance as yf  # 실시간 데이터
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("❌ 필수 라이브러리 미설치")
    print("pip install scikit-learn ta yfinance")
    exit()

class UltimateBinanceChart:
    """Ultimate Binance-Style Chart System"""
    
    def __init__(self):
        self.base_path = "/Users/parkyoungjun/Desktop/BTC_Analysis_System"
        self.timeseries_path = os.path.join(self.base_path, "timeseries_data")
        self.historical_path = os.path.join(self.base_path, "historical_data")
        
        # 시간축 정의 (바이낸스 스타일)
        self.timeframes = {
            '1m': {'minutes': 1, 'points': 60, 'label': '1분'},
            '5m': {'minutes': 5, 'points': 60, 'label': '5분'},
            '15m': {'minutes': 15, 'points': 96, 'label': '15분'},
            '1h': {'minutes': 60, 'points': 168, 'label': '1시간'},
            '4h': {'minutes': 240, 'points': 180, 'label': '4시간'},
            '1d': {'minutes': 1440, 'points': 365, 'label': '1일'},
            '1w': {'minutes': 10080, 'points': 52, 'label': '1주'}
        }
        
        # 고급 모델 앙상블
        self.models = {}
        self.accuracy_scores = {}
        
        # 시스템 정확도 추적
        self.system_accuracy = {
            'price_accuracy': 0,
            'direction_accuracy': 0,
            'trend_accuracy': 0,
            'volatility_accuracy': 0
        }
    
    def get_realtime_data(self, symbol: str = "BTC-USD", period: str = "6mo", interval: str = "1h") -> pd.DataFrame:
        """실시간 데이터 가져오기"""
        try:
            print(f"📡 실시간 데이터 가져오는 중... ({symbol}, {interval})")
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)
            
            if df.empty:
                print("⚠️ yfinance 데이터 없음, 로컬 데이터 사용")
                return self.load_local_data()
            
            df = df.reset_index()
            df.columns = [col.lower() for col in df.columns]
            
            # 필수 컬럼 변환
            if 'date' in df.columns:
                df['timestamp'] = pd.to_datetime(df['date'])
            elif 'datetime' in df.columns:
                df['timestamp'] = pd.to_datetime(df['datetime'])
            
            df['price'] = df['close']
            
            print(f"✅ 실시간 데이터: {len(df)}개 포인트")
            return df
            
        except Exception as e:
            print(f"⚠️ 실시간 데이터 실패: {e}")
            return self.load_local_data()
    
    def load_local_data(self) -> pd.DataFrame:
        """로컬 6개월 데이터 로드"""
        try:
            print("📊 로컬 6개월 데이터 로드 중...")
            
            data_files = {
                'price': 'btc_price.csv',
                'volume': 'btc_volume.csv',
                'market_cap': 'btc_market_cap.csv',
                'active_addresses': 'active_addresses.csv'
            }
            
            master_df = None
            
            for data_type, filename in data_files.items():
                filepath = os.path.join(self.timeseries_path, filename)
                if os.path.exists(filepath):
                    df = pd.read_csv(filepath)
                    if 'timestamp' in df.columns and 'value' in df.columns:
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                        df = df.rename(columns={'value': data_type})
                        df = df[['timestamp', data_type]].dropna()
                        
                        if master_df is None:
                            master_df = df
                        else:
                            master_df = master_df.merge(df, on='timestamp', how='outer')
            
            if master_df is not None:
                master_df = master_df.sort_values('timestamp').reset_index(drop=True)
                for col in master_df.columns:
                    if col != 'timestamp':
                        master_df[col] = master_df[col].interpolate(method='linear')
                
                # 가상의 OHLC 데이터 생성 (실제 데이터가 없을 경우)
                if 'open' not in master_df.columns:
                    master_df['open'] = master_df['price'] * (1 + np.random.normal(0, 0.001, len(master_df)))
                    master_df['high'] = master_df['price'] * (1 + np.abs(np.random.normal(0, 0.005, len(master_df))))
                    master_df['low'] = master_df['price'] * (1 - np.abs(np.random.normal(0, 0.005, len(master_df))))
                    master_df['close'] = master_df['price']
                
                print(f"✅ 로컬 데이터: {len(master_df)}개 포인트")
                return master_df
            
            return pd.DataFrame()
            
        except Exception as e:
            print(f"❌ 로컬 데이터 로드 실패: {e}")
            return pd.DataFrame()
    
    def calculate_ultra_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """초고급 기술적 지표 계산 (정확도 향상용)"""
        try:
            print("🔧 초고급 지표 계산 중...")
            
            # 기본 지표들
            for period in [7, 14, 21, 30, 50, 100, 200]:
                df[f'sma_{period}'] = ta.trend.sma_indicator(df['price'], window=period)
                df[f'ema_{period}'] = ta.trend.ema_indicator(df['price'], window=period)
            
            # RSI 다중 기간
            for period in [7, 14, 21, 28]:
                df[f'rsi_{period}'] = ta.momentum.rsi(df['price'], window=period)
            
            # MACD 변형
            for fast, slow in [(12, 26), (5, 35), (8, 21)]:
                macd = ta.trend.MACD(df['price'], window_fast=fast, window_slow=slow)
                df[f'macd_{fast}_{slow}'] = macd.macd()
                df[f'macd_signal_{fast}_{slow}'] = macd.macd_signal()
                df[f'macd_hist_{fast}_{slow}'] = macd.macd_diff()
            
            # 볼린저 밴드 다중
            for period in [20, 30]:
                bb = ta.volatility.BollingerBands(df['price'], window=period)
                df[f'bb_upper_{period}'] = bb.bollinger_hband()
                df[f'bb_lower_{period}'] = bb.bollinger_lband()
                df[f'bb_width_{period}'] = bb.bollinger_wband()
                df[f'bb_pband_{period}'] = bb.bollinger_pband()
            
            # Stochastic
            stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'])
            df['stoch_k'] = stoch.stoch()
            df['stoch_d'] = stoch.stoch_signal()
            
            # Williams %R
            for period in [14, 28]:
                df[f'williams_r_{period}'] = ta.momentum.williams_r(df['high'], df['low'], df['close'], lbp=period)
            
            # ATR (Average True Range)
            for period in [14, 21]:
                df[f'atr_{period}'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=period)
            
            # ADX (Average Directional Index)
            adx = ta.trend.ADXIndicator(df['high'], df['low'], df['close'])
            df['adx'] = adx.adx()
            df['adx_pos'] = adx.adx_pos()
            df['adx_neg'] = adx.adx_neg()
            
            # CCI (Commodity Channel Index)
            for period in [20, 40]:
                df[f'cci_{period}'] = ta.trend.cci(df['high'], df['low'], df['close'], window=period)
            
            # OBV (On Balance Volume)
            if 'volume' in df.columns:
                df['obv'] = ta.volume.on_balance_volume(df['close'], df['volume'])
                df['obv_ma'] = df['obv'].rolling(20).mean()
                
                # Volume indicators
                df['volume_sma'] = df['volume'].rolling(20).mean()
                df['volume_ratio'] = df['volume'] / df['volume_sma']
                
                # VWAP
                df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
            
            # Ichimoku
            ichimoku = ta.trend.IchimokuIndicator(df['high'], df['low'])
            df['ichimoku_a'] = ichimoku.ichimoku_a()
            df['ichimoku_b'] = ichimoku.ichimoku_b()
            df['ichimoku_base'] = ichimoku.ichimoku_base_line()
            df['ichimoku_conv'] = ichimoku.ichimoku_conversion_line()
            
            # 파생 지표들
            df['price_ma_ratio'] = df['price'] / df['sma_50']
            df['rsi_divergence'] = df['rsi_14'] - df['rsi_14'].rolling(14).mean()
            df['momentum'] = df['price'].pct_change(10)
            df['volatility'] = df['price'].rolling(20).std() / df['price'].rolling(20).mean()
            
            # 트렌드 강도
            df['trend_strength'] = abs(df['price'] - df['sma_50']) / df['sma_50']
            
            # 지지/저항 레벨
            df['resistance'] = df['high'].rolling(20).max()
            df['support'] = df['low'].rolling(20).min()
            df['price_position'] = (df['price'] - df['support']) / (df['resistance'] - df['support'])
            
            print(f"✅ 초고급 지표 계산 완료: {len(df.columns)}개 피처")
            return df
            
        except Exception as e:
            print(f"❌ 지표 계산 실패: {e}")
            return df
    
    def train_ultra_models(self, df: pd.DataFrame, timeframe: str) -> Dict:
        """초정밀 모델 훈련 (95%+ 정확도 목표)"""
        try:
            print(f"🤖 {timeframe} 초정밀 모델 훈련 중...")
            
            # 피처 선택
            feature_cols = []
            for col in df.columns:
                if col not in ['timestamp', 'date', 'price', 'close', 'open', 'high', 'low'] and df[col].notna().sum() > len(df) * 0.7:
                    feature_cols.append(col)
            
            if len(feature_cols) < 10:
                print("⚠️ 피처 부족")
                return {}
            
            print(f"📊 선택된 피처: {len(feature_cols)}개")
            
            # 데이터 준비
            df_clean = df[['price'] + feature_cols].dropna()
            
            if len(df_clean) < 50:
                print("❌ 훈련 데이터 부족")
                return {}
            
            # 숫자가 아닌 컬럼 제거
            numeric_features = []
            for col in feature_cols:
                if pd.api.types.is_numeric_dtype(df_clean[col]):
                    numeric_features.append(col)
            
            feature_cols = numeric_features
            
            # 타겟 생성 (다음 캔들 가격)
            X = df_clean[feature_cols].iloc[:-1].reset_index(drop=True)
            y = df_clean['price'].iloc[1:].reset_index(drop=True)
            
            # TimeSeriesSplit으로 교차 검증
            tscv = TimeSeriesSplit(n_splits=5)
            
            # 다양한 스케일러 테스트
            scalers = {
                'standard': StandardScaler(),
                'robust': RobustScaler()
            }
            
            # 다양한 모델들
            models = {
                'RandomForest': RandomForestRegressor(
                    n_estimators=200, 
                    max_depth=20,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1
                ),
                'ExtraTrees': ExtraTreesRegressor(
                    n_estimators=200,
                    max_depth=20,
                    random_state=42,
                    n_jobs=-1
                ),
                'GradientBoosting': GradientBoostingRegressor(
                    n_estimators=200,
                    learning_rate=0.05,
                    max_depth=10,
                    random_state=42
                ),
                'SVR': SVR(kernel='rbf', C=100, gamma=0.001),
                'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5),
                'MLP': MLPRegressor(
                    hidden_layer_sizes=(100, 50, 25),
                    activation='relu',
                    solver='adam',
                    max_iter=1000,
                    random_state=42
                )
            }
            
            best_models = []
            
            for scaler_name, scaler in scalers.items():
                X_scaled = scaler.fit_transform(X)
                
                for model_name, model in models.items():
                    try:
                        # 교차 검증
                        scores = []
                        direction_scores = []
                        
                        for train_idx, test_idx in tscv.split(X_scaled):
                            X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
                            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                            
                            model_copy = model.__class__(**model.get_params())
                            model_copy.fit(X_train, y_train)
                            y_pred = model_copy.predict(X_test)
                            
                            # 평가 지표
                            mae = mean_absolute_error(y_test, y_pred)
                            mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
                            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                            r2 = r2_score(y_test, y_pred)
                            
                            # 방향 정확도
                            if len(y_test) > 1:
                                actual_direction = np.diff(y_test.values)
                                pred_direction = np.diff(y_pred)
                                direction_acc = np.mean(np.sign(actual_direction) == np.sign(pred_direction))
                                direction_scores.append(direction_acc)
                            
                            scores.append({
                                'mae': mae,
                                'mape': mape,
                                'rmse': rmse,
                                'r2': r2
                            })
                        
                        # 평균 점수
                        avg_mae = np.mean([s['mae'] for s in scores])
                        avg_mape = np.mean([s['mape'] for s in scores])
                        avg_r2 = np.mean([s['r2'] for s in scores])
                        avg_direction = np.mean(direction_scores) if direction_scores else 0.5
                        
                        # 최종 모델 훈련
                        model.fit(X_scaled, y)
                        
                        best_models.append({
                            'name': f'{model_name}_{scaler_name}',
                            'model': model,
                            'scaler': scaler,
                            'mae': avg_mae,
                            'mape': avg_mape,
                            'r2': avg_r2,
                            'direction_accuracy': avg_direction,
                            'features': feature_cols,
                            'score': avg_r2 * avg_direction  # 종합 점수
                        })
                        
                        print(f"  • {model_name}_{scaler_name}: MAPE={avg_mape:.2f}%, R²={avg_r2:.3f}, 방향={avg_direction:.1%}")
                        
                    except Exception as e:
                        continue
            
            # 상위 3개 모델 선택
            best_models = sorted(best_models, key=lambda x: x['score'], reverse=True)[:3]
            
            if best_models:
                best = best_models[0]
                print(f"🏆 최고 모델: {best['name']} (MAPE={best['mape']:.2f}%, 방향정확도={best['direction_accuracy']:.1%})")
                
                # 정확도 저장
                self.accuracy_scores[timeframe] = {
                    'mape': best['mape'],
                    'direction': best['direction_accuracy'],
                    'r2': best['r2']
                }
            
            return {
                'models': best_models,
                'timeframe': timeframe,
                'data_points': len(X)
            }
            
        except Exception as e:
            print(f"❌ 모델 훈련 실패: {e}")
            return {}
    
    def predict_future(self, df: pd.DataFrame, model_info: Dict, periods: int) -> List[Dict]:
        """미래 예측 생성"""
        try:
            if not model_info or 'models' not in model_info:
                return []
            
            predictions = []
            models = model_info['models']
            
            # 최신 데이터
            latest = df[models[0]['features']].iloc[-1:].values
            current_price = df['price'].iloc[-1]
            current_time = df['timestamp'].iloc[-1] if 'timestamp' in df.columns else datetime.now()
            
            for i in range(periods):
                pred_time = current_time + timedelta(minutes=self.timeframes[model_info['timeframe']]['minutes'] * (i + 1))
                
                # 앙상블 예측
                all_predictions = []
                for model_data in models:
                    try:
                        if model_data['scaler']:
                            features_scaled = model_data['scaler'].transform(latest)
                        else:
                            features_scaled = latest
                        
                        pred = model_data['model'].predict(features_scaled)[0]
                        all_predictions.append(pred)
                    except:
                        continue
                
                if all_predictions:
                    # 가중 평균 (성능 기반)
                    weights = [m['score'] for m in models[:len(all_predictions)]]
                    total_weight = sum(weights)
                    weights = [w/total_weight for w in weights]
                    
                    predicted_price = np.average(all_predictions, weights=weights)
                    price_std = np.std(all_predictions)
                    
                    # 신뢰도 계산
                    avg_accuracy = np.mean([m['direction_accuracy'] for m in models])
                    time_decay = 0.95 ** i  # 시간에 따른 신뢰도 감소
                    confidence = avg_accuracy * time_decay * 100
                    
                    predictions.append({
                        'time': pred_time,
                        'price': predicted_price,
                        'upper': predicted_price + price_std * 2,
                        'lower': predicted_price - price_std * 2,
                        'confidence': confidence
                    })
            
            return predictions
            
        except Exception as e:
            print(f"❌ 예측 생성 실패: {e}")
            return []
    
    def create_binance_chart(self, timeframe: str = '1h', predict_periods: int = 24):
        """바이낸스 스타일 차트 생성"""
        try:
            print(f"\n📊 {timeframe} 바이낸스 스타일 차트 생성 중...")
            
            # 데이터 로드
            interval_map = {
                '1m': '1m', '5m': '5m', '15m': '15m',
                '1h': '1h', '4h': '4h', '1d': '1d', '1w': '1wk'
            }
            
            df = self.get_realtime_data(interval=interval_map.get(timeframe, '1h'))
            if df.empty:
                print("❌ 데이터 없음")
                return None
            
            # 지표 계산
            df = self.calculate_ultra_indicators(df)
            
            # 모델 훈련
            model_info = self.train_ultra_models(df, timeframe)
            
            # 예측 생성
            predictions = self.predict_future(df, model_info, predict_periods)
            
            # 차트 생성
            fig = make_subplots(
                rows=4, cols=1,
                subplot_titles=(
                    f"BTC/USDT {self.timeframes[timeframe]['label']} 차트",
                    "거래량",
                    "RSI & 스토캐스틱",
                    "MACD"
                ),
                vertical_spacing=0.05,
                row_heights=[0.5, 0.15, 0.15, 0.15],
                specs=[[{"secondary_y": False}],
                       [{"secondary_y": False}],
                       [{"secondary_y": False}],
                       [{"secondary_y": False}]]
            )
            
            # 1. 캔들스틱 차트
            fig.add_trace(
                go.Candlestick(
                    x=df['timestamp'] if 'timestamp' in df.columns else df.index,
                    open=df['open'],
                    high=df['high'],
                    low=df['low'],
                    close=df['close'],
                    name='BTC',
                    increasing_line_color='#26a69a',
                    decreasing_line_color='#ef5350'
                ),
                row=1, col=1
            )
            
            # 이동평균선
            for ma in [20, 50, 200]:
                if f'sma_{ma}' in df.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=df['timestamp'] if 'timestamp' in df.columns else df.index,
                            y=df[f'sma_{ma}'],
                            name=f'MA{ma}',
                            line=dict(width=1)
                        ),
                        row=1, col=1
                    )
            
            # 볼린저 밴드
            if 'bb_upper_20' in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df['timestamp'] if 'timestamp' in df.columns else df.index,
                        y=df['bb_upper_20'],
                        name='BB Upper',
                        line=dict(color='rgba(250,128,114,0.5)', width=1)
                    ),
                    row=1, col=1
                )
                fig.add_trace(
                    go.Scatter(
                        x=df['timestamp'] if 'timestamp' in df.columns else df.index,
                        y=df['bb_lower_20'],
                        name='BB Lower',
                        line=dict(color='rgba(250,128,114,0.5)', width=1),
                        fill='tonexty',
                        fillcolor='rgba(250,128,114,0.1)'
                    ),
                    row=1, col=1
                )
            
            # 예측 추가
            if predictions:
                pred_times = [p['time'] for p in predictions]
                pred_prices = [p['price'] for p in predictions]
                pred_upper = [p['upper'] for p in predictions]
                pred_lower = [p['lower'] for p in predictions]
                
                fig.add_trace(
                    go.Scatter(
                        x=pred_times,
                        y=pred_prices,
                        mode='lines+markers',
                        name='AI 예측',
                        line=dict(color='yellow', width=2, dash='dot'),
                        marker=dict(size=4)
                    ),
                    row=1, col=1
                )
                
                # 예측 신뢰구간
                fig.add_trace(
                    go.Scatter(
                        x=pred_times + pred_times[::-1],
                        y=pred_upper + pred_lower[::-1],
                        fill='toself',
                        fillcolor='rgba(255,255,0,0.1)',
                        line=dict(color='rgba(255,255,255,0)'),
                        name='예측 범위',
                        hoverinfo='skip'
                    ),
                    row=1, col=1
                )
            
            # 2. 거래량
            if 'volume' in df.columns:
                colors = ['red' if row['close'] < row['open'] else 'green' 
                         for _, row in df.iterrows()]
                fig.add_trace(
                    go.Bar(
                        x=df['timestamp'] if 'timestamp' in df.columns else df.index,
                        y=df['volume'],
                        name='Volume',
                        marker_color=colors,
                        opacity=0.7
                    ),
                    row=2, col=1
                )
            
            # 3. RSI & 스토캐스틱
            if 'rsi_14' in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df['timestamp'] if 'timestamp' in df.columns else df.index,
                        y=df['rsi_14'],
                        name='RSI',
                        line=dict(color='purple', width=1)
                    ),
                    row=3, col=1
                )
                
                # RSI 과매수/과매도 선
                fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
            
            if 'stoch_k' in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df['timestamp'] if 'timestamp' in df.columns else df.index,
                        y=df['stoch_k'],
                        name='Stoch %K',
                        line=dict(color='orange', width=1)
                    ),
                    row=3, col=1
                )
            
            # 4. MACD
            if 'macd_12_26' in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df['timestamp'] if 'timestamp' in df.columns else df.index,
                        y=df['macd_12_26'],
                        name='MACD',
                        line=dict(color='blue', width=1)
                    ),
                    row=4, col=1
                )
                fig.add_trace(
                    go.Scatter(
                        x=df['timestamp'] if 'timestamp' in df.columns else df.index,
                        y=df['macd_signal_12_26'],
                        name='Signal',
                        line=dict(color='red', width=1)
                    ),
                    row=4, col=1
                )
                fig.add_trace(
                    go.Bar(
                        x=df['timestamp'] if 'timestamp' in df.columns else df.index,
                        y=df['macd_hist_12_26'],
                        name='Histogram',
                        marker_color='gray',
                        opacity=0.3
                    ),
                    row=4, col=1
                )
            
            # 레이아웃 설정 (바이낸스 다크 테마)
            fig.update_layout(
                title={
                    'text': f"🚀 BTC/USDT Professional Trading Chart - {timeframe}",
                    'x': 0.5,
                    'xanchor': 'center'
                },
                template='plotly_dark',
                height=1000,
                showlegend=True,
                hovermode='x unified',
                xaxis_rangeslider_visible=False,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            # Y축 설정
            fig.update_yaxes(title_text="Price (USDT)", row=1, col=1)
            fig.update_yaxes(title_text="Volume", row=2, col=1)
            fig.update_yaxes(title_text="RSI/Stoch", row=3, col=1)
            fig.update_yaxes(title_text="MACD", row=4, col=1)
            
            # X축 설정
            fig.update_xaxes(title_text="Time", row=4, col=1)
            
            # 차트 저장
            chart_path = os.path.join(self.base_path, f"binance_chart_{timeframe}.html")
            fig.write_html(chart_path, include_plotlyjs=True)
            
            print(f"✅ 차트 저장: {chart_path}")
            
            # 정확도 리포트
            if timeframe in self.accuracy_scores:
                acc = self.accuracy_scores[timeframe]
                print(f"\n🎯 시스템 정확도:")
                print(f"  • MAPE: {acc['mape']:.2f}% (가격 오차)")
                print(f"  • 방향 정확도: {acc['direction']:.1%}")
                print(f"  • R² Score: {acc['r2']:.3f}")
                print(f"  • 종합 정확도: {(100 - acc['mape'] + acc['direction']*100) / 2:.1f}%")
            
            return chart_path
            
        except Exception as e:
            print(f"❌ 차트 생성 실패: {e}")
            return None
    
    def create_all_timeframes(self):
        """모든 시간축 차트 생성"""
        results = {}
        
        for tf in ['1h', '4h', '1d']:  # 주요 시간축만
            print(f"\n{'='*60}")
            print(f"📊 {tf} 차트 생성 중...")
            print(f"{'='*60}")
            
            path = self.create_binance_chart(timeframe=tf, predict_periods=24)
            if path:
                results[tf] = path
        
        return results

def main():
    """메인 실행"""
    print("🚀 Ultimate Binance-Style BTC Chart System")
    print("="*80)
    
    system = UltimateBinanceChart()
    
    # 원하는 시간축 선택
    print("\n시간축 선택:")
    print("1. 1시간 (1h)")
    print("2. 4시간 (4h)")
    print("3. 1일 (1d)")
    print("4. 모든 시간축")
    
    # 기본값: 1시간
    choice = "1"
    
    if choice == "1":
        chart = system.create_binance_chart(timeframe='1h', predict_periods=48)
    elif choice == "2":
        chart = system.create_binance_chart(timeframe='4h', predict_periods=30)
    elif choice == "3":
        chart = system.create_binance_chart(timeframe='1d', predict_periods=30)
    else:
        charts = system.create_all_timeframes()
        print(f"\n✅ 생성된 차트: {list(charts.keys())}")
    
    # 브라우저 열기
    try:
        import subprocess
        if choice in ["1", "2", "3"]:
            subprocess.run(["open", chart], check=True)
        else:
            for path in charts.values():
                subprocess.run(["open", path], check=True)
        print("\n🌐 차트가 브라우저에서 열렸습니다!")
    except:
        pass
    
    print("\n" + "="*80)
    print("🎉 Ultimate Chart System 완료!")
    print("="*80)

if __name__ == "__main__":
    main()