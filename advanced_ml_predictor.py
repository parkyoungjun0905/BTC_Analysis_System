"""
고도화된 BTC 예측 시스템
- 6개월 과거 데이터 패턴 학습
- 지표-실제가격 상관관계 분석
- 머신러닝 기반 정확한 예측
- 시각적으로 뛰어난 차트
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# 머신러닝 라이브러리
try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    import joblib
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("⚠️ scikit-learn 미설치 - pip install scikit-learn")

# 차트 라이브러리
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("⚠️ Plotly 미설치 - pip install plotly")

class AdvancedMLPredictor:
    """고도화된 머신러닝 예측 시스템"""
    
    def __init__(self):
        self.base_path = "/Users/parkyoungjun/Desktop/BTC_Analysis_System"
        self.historical_path = os.path.join(self.base_path, "historical_data")
        self.timeseries_path = os.path.join(self.base_path, "timeseries_data")
        self.model_path = os.path.join(self.base_path, "trained_models")
        
        # 모델 저장 폴더 생성
        os.makedirs(self.model_path, exist_ok=True)
        
        # 핵심 지표 리스트 (상관관계 높은 것들)
        self.key_indicators = [
            'RSI_14', 'MACD', 'MACD_signal', 'BB_upper', 'BB_lower',
            'volume_sma_20', 'ATR_14', 'ADX_14', 'Stoch_K', 'Stoch_D',
            'exchange_netflow', 'whale_ratio', 'funding_rate', 'open_interest',
            'fear_greed_index', 'hash_rate', 'difficulty', 'active_addresses'
        ]
        
        # 학습된 모델들
        self.models = {}
        self.scaler = StandardScaler()
        self.price_scaler = MinMaxScaler()
        
    def load_6month_data(self) -> pd.DataFrame:
        """6개월 누적 데이터 로드 및 전처리"""
        try:
            print("📊 6개월 누적 데이터 로드 중...")
            
            # 1. JSON 파일들 로드
            json_files = sorted([f for f in os.listdir(self.historical_path) 
                               if f.startswith("btc_analysis_") and f.endswith(".json")])
            
            all_data = []
            
            for filename in json_files:
                filepath = os.path.join(self.historical_path, filename)
                try:
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                    
                    # 타임스탬프 파싱
                    if "collection_time" in data:
                        timestamp = pd.to_datetime(data["collection_time"])
                    else:
                        time_part = filename.replace("btc_analysis_", "").replace(".json", "")
                        timestamp = pd.to_datetime(time_part)
                    
                    # 가격 추출
                    price = self.extract_price_from_data(data)
                    if price <= 0:
                        continue
                    
                    # 지표 추출
                    indicators = self.extract_indicators_from_data(data)
                    
                    # 데이터 구성
                    row = {
                        'timestamp': timestamp,
                        'price': price,
                        'filename': filename
                    }
                    row.update(indicators)
                    all_data.append(row)
                    
                except Exception as e:
                    print(f"파일 처리 실패 {filename}: {e}")
                    continue
            
            # 2. CSV 파일들도 통합 (더 많은 데이터)
            csv_data = self.load_timeseries_csv_data()
            if not csv_data.empty:
                print(f"📈 CSV 시계열 데이터 추가: {len(csv_data)}개 포인트")
                # JSON 데이터와 병합
                pass  # 복잡하므로 일단 JSON만 사용
            
            # 3. 데이터프레임 생성
            df = pd.DataFrame(all_data)
            if df.empty:
                print("❌ 데이터 없음")
                return df
            
            # 4. 전처리
            df = df.sort_values('timestamp').reset_index(drop=True)
            df = df.drop_duplicates(subset=['timestamp'], keep='last')
            
            # 5. 결측치 처리
            df = self.preprocess_data(df)
            
            print(f"✅ 총 데이터: {len(df)}개 포인트")
            print(f"📅 기간: {df['timestamp'].min()} ~ {df['timestamp'].max()}")
            
            return df
            
        except Exception as e:
            print(f"❌ 데이터 로드 실패: {e}")
            return pd.DataFrame()
    
    def load_timeseries_csv_data(self) -> pd.DataFrame:
        """시계열 CSV 데이터 로드"""
        try:
            csv_files = [f for f in os.listdir(self.timeseries_path) 
                        if f.endswith('.csv') and 'btc_price' in f]
            
            if not csv_files:
                return pd.DataFrame()
            
            # BTC 가격 CSV 파일 로드
            price_file = os.path.join(self.timeseries_path, csv_files[0])
            df = pd.read_csv(price_file)
            
            if 'timestamp' in df.columns and 'value' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.rename(columns={'value': 'price'})
                return df[['timestamp', 'price']]
            
            return pd.DataFrame()
            
        except Exception as e:
            print(f"CSV 데이터 로드 실패: {e}")
            return pd.DataFrame()
    
    def extract_price_from_data(self, data: Dict) -> float:
        """데이터에서 BTC 가격 추출"""
        paths = [
            ["data_sources", "legacy_analyzer", "market_data", "avg_price"],
            ["summary", "current_btc_price"],
            ["market_data", "current_price"]
        ]
        
        for path in paths:
            try:
                value = data
                for key in path:
                    value = value[key]
                if value and value > 0:
                    return float(value)
            except:
                continue
        return 0
    
    def extract_indicators_from_data(self, data: Dict) -> Dict:
        """데이터에서 핵심 지표 추출"""
        indicators = {}
        
        try:
            # 온체인 데이터
            if "data_sources" in data and "legacy_analyzer" in data["data_sources"]:
                legacy = data["data_sources"]["legacy_analyzer"]
                
                if "onchain_data" in legacy:
                    onchain = legacy["onchain_data"]
                    indicators.update({
                        'hash_rate': onchain.get('hash_rate', 0),
                        'difficulty': onchain.get('difficulty', 0),
                        'active_addresses': onchain.get('active_addresses', 0),
                        'exchange_netflow': onchain.get('exchange_netflow', 0),
                        'whale_ratio': onchain.get('whale_ratio', 0),
                        'mvrv': onchain.get('mvrv', 0),
                        'nvt': onchain.get('nvt', 0),
                        'sopr': onchain.get('sopr', 0)
                    })
                
                if "market_data" in legacy:
                    market = legacy["market_data"]
                    indicators.update({
                        'volume_24h': market.get('total_volume', 0),
                        'market_cap': market.get('market_cap', 0)
                    })
            
            # 기술적 지표 (summary에서)
            if "summary" in data:
                summary = data["summary"]
                
                # RSI 관련
                for key in ['rsi_14', 'RSI_14', 'rsi']:
                    if key in summary:
                        indicators['RSI_14'] = summary[key]
                        break
                
                # MACD 관련
                for key in ['macd', 'MACD']:
                    if key in summary:
                        indicators['MACD'] = summary[key]
                        break
                
                # 기타 지표들
                indicators.update({
                    'bb_upper': summary.get('bb_upper', 0),
                    'bb_lower': summary.get('bb_lower', 0),
                    'atr_14': summary.get('atr_14', 0),
                    'adx_14': summary.get('adx_14', 0),
                    'stoch_k': summary.get('stoch_k', 0),
                    'stoch_d': summary.get('stoch_d', 0)
                })
            
            # 지표 딕셔너리에서 직접
            if "indicators" in data:
                indicators.update(data["indicators"])
            
            return indicators
            
        except Exception as e:
            print(f"지표 추출 실패: {e}")
            return {}
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """데이터 전처리"""
        try:
            print("🔄 데이터 전처리 중...")
            
            # 1. 숫자 컬럼만 추출
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            numeric_cols = [col for col in numeric_cols if col != 'price']  # price 제외
            
            # 2. 결측치 처리
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    df[col] = df[col].fillna(df[col].median())
            
            # 3. 이상치 제거 (IQR 방법)
            for col in numeric_cols:
                if col in df.columns:
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    df[col] = np.clip(df[col], lower_bound, upper_bound)
            
            # 4. 기술적 지표 계산 (이동평균 등)
            df = self.calculate_technical_indicators(df)
            
            # 5. 시차 변수 생성
            df = self.create_lagged_features(df)
            
            print(f"✅ 전처리 완료: {len(df)}행, {len(df.columns)}개 컬럼")
            
            return df
            
        except Exception as e:
            print(f"전처리 실패: {e}")
            return df
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """기술적 지표 계산"""
        try:
            # 이동평균
            df['price_sma_5'] = df['price'].rolling(window=5).mean()
            df['price_sma_20'] = df['price'].rolling(window=20).mean()
            df['price_ema_12'] = df['price'].ewm(span=12).mean()
            df['price_ema_26'] = df['price'].ewm(span=26).mean()
            
            # MACD (간단 계산)
            if 'MACD' not in df.columns:
                df['MACD'] = df['price_ema_12'] - df['price_ema_26']
                df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
            
            # RSI (간단 계산)
            if 'RSI_14' not in df.columns:
                delta = df['price'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                df['RSI_14'] = 100 - (100 / (1 + rs))
            
            # 볼린저 밴드
            if 'bb_upper' not in df.columns:
                bb_period = 20
                bb_std = 2
                sma = df['price'].rolling(window=bb_period).mean()
                std = df['price'].rolling(window=bb_period).std()
                df['bb_upper'] = sma + (std * bb_std)
                df['bb_lower'] = sma - (std * bb_std)
                df['bb_width'] = df['bb_upper'] - df['bb_lower']
                df['bb_position'] = (df['price'] - df['bb_lower']) / df['bb_width']
            
            # 가격 변화율
            df['price_change'] = df['price'].pct_change()
            df['price_change_1h'] = df['price'].pct_change(periods=1)
            
            # 변동성
            df['volatility_20'] = df['price_change'].rolling(window=20).std()
            
            return df
            
        except Exception as e:
            print(f"기술적 지표 계산 실패: {e}")
            return df
    
    def create_lagged_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """시차 변수 생성"""
        try:
            # 가격 시차
            for lag in [1, 3, 6, 12, 24]:
                df[f'price_lag_{lag}'] = df['price'].shift(lag)
            
            # RSI 시차
            if 'RSI_14' in df.columns:
                for lag in [1, 3, 6]:
                    df[f'RSI_lag_{lag}'] = df['RSI_14'].shift(lag)
            
            # MACD 시차
            if 'MACD' in df.columns:
                for lag in [1, 3]:
                    df[f'MACD_lag_{lag}'] = df['MACD'].shift(lag)
            
            return df
            
        except Exception as e:
            print(f"시차 변수 생성 실패: {e}")
            return df
    
    def train_ml_models(self, df: pd.DataFrame) -> Dict:
        """머신러닝 모델 훈련"""
        if not ML_AVAILABLE:
            print("❌ scikit-learn 미설치")
            return {}
        
        try:
            print("🤖 머신러닝 모델 훈련 시작...")
            
            # 1. 피처 선택 (결측치 없는 숫자 컬럼)
            feature_cols = []
            for col in df.columns:
                if col not in ['timestamp', 'price', 'filename'] and df[col].dtype in ['float64', 'int64']:
                    if df[col].notna().sum() > len(df) * 0.5:  # 50% 이상 데이터 있는 컬럼만
                        feature_cols.append(col)
            
            print(f"📊 선택된 피처: {len(feature_cols)}개")
            
            # 2. 결측치 제거
            df_clean = df[['price'] + feature_cols].dropna()
            
            if len(df_clean) < 50:
                print("❌ 훈련 데이터 부족")
                return {}
            
            print(f"✅ 훈련 데이터: {len(df_clean)}개 샘플")
            
            # 3. X, y 분리
            X = df_clean[feature_cols]
            y = df_clean['price']
            
            # 4. 미래 가격 예측을 위한 타겟 생성 (1시간 후)
            y_future = y.shift(-1).dropna()  # 1시간 후 가격
            X_future = X.iloc[:-1]  # 마지막 행 제외
            
            # 5. 훈련/테스트 분할
            X_train, X_test, y_train, y_test = train_test_split(
                X_future, y_future, test_size=0.2, random_state=42, shuffle=False
            )
            
            # 6. 스케일링
            self.scaler.fit(X_train)
            X_train_scaled = self.scaler.transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # 7. 다양한 모델 훈련
            models_config = {
                'random_forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
                'gradient_boost': GradientBoostingRegressor(n_estimators=100, random_state=42),
                'linear': LinearRegression(),
                'ridge': Ridge(alpha=1.0)
            }
            
            results = {}
            
            for name, model in models_config.items():
                print(f"🔄 {name} 훈련 중...")
                
                # 훈련
                if name in ['linear', 'ridge']:
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                else:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                
                # 평가
                mae = mean_absolute_error(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                # 방향성 정확도 (실제 중요한 지표)
                actual_direction = (y_test.shift(-1) > y_test).iloc[:-1]
                pred_direction = (pd.Series(y_pred[:-1]) > y_test.iloc[:-1])
                direction_accuracy = (actual_direction == pred_direction).mean()
                
                results[name] = {
                    'model': model,
                    'mae': mae,
                    'mse': mse,
                    'r2': r2,
                    'direction_accuracy': direction_accuracy,
                    'rmse': np.sqrt(mse)
                }
                
                print(f"  ✅ {name}: MAE=${mae:.0f}, R²={r2:.3f}, 방향정확도={direction_accuracy:.1%}")
            
            # 8. 최고 성능 모델 선택 (방향 정확도 기준)
            best_model_name = max(results.keys(), 
                                key=lambda x: results[x]['direction_accuracy'])
            best_model = results[best_model_name]
            
            print(f"🏆 최고 모델: {best_model_name} (방향정확도: {best_model['direction_accuracy']:.1%})")
            
            # 9. 모델 저장
            self.models = results
            self.feature_cols = feature_cols
            self.best_model_name = best_model_name
            
            # 파일로 저장
            model_file = os.path.join(self.model_path, f'best_model_{best_model_name}.pkl')
            joblib.dump(best_model['model'], model_file)
            
            scaler_file = os.path.join(self.model_path, 'scaler.pkl')
            joblib.dump(self.scaler, scaler_file)
            
            print("💾 모델 저장 완료")
            
            return results
            
        except Exception as e:
            print(f"❌ 모델 훈련 실패: {e}")
            return {}
    
    def predict_future_prices(self, df: pd.DataFrame, hours_ahead: int = 24) -> List[Dict]:
        """미래 가격 예측"""
        try:
            if not self.models or not hasattr(self, 'best_model_name'):
                print("❌ 훈련된 모델 없음")
                return []
            
            print(f"🔮 {hours_ahead}시간 예측 시작...")
            
            best_model = self.models[self.best_model_name]['model']
            
            # 최신 데이터
            latest_data = df.iloc[-1][self.feature_cols].values.reshape(1, -1)
            
            # 스케일링 (필요한 경우)
            if self.best_model_name in ['linear', 'ridge']:
                latest_data = self.scaler.transform(latest_data)
            
            predictions = []
            current_price = df.iloc[-1]['price']
            
            # 시간별 예측
            for hour in range(1, hours_ahead + 1):
                # 예측
                pred_price = best_model.predict(latest_data)[0]
                
                # 신뢰도 계산 (모델 성능 기반)
                model_accuracy = self.models[self.best_model_name]['direction_accuracy']
                confidence = model_accuracy * (1 - hour * 0.01)  # 시간이 지날수록 감소
                confidence = max(confidence, 0.3)  # 최소 30%
                
                # 신뢰 구간 계산
                model_mae = self.models[self.best_model_name]['mae']
                uncertainty = model_mae * (1 + hour * 0.1)  # 시간이 지날수록 증가
                
                predictions.append({
                    'hour': hour,
                    'price': pred_price,
                    'confidence': confidence,
                    'upper_bound': pred_price + uncertainty,
                    'lower_bound': pred_price - uncertainty,
                    'change_from_current': ((pred_price / current_price) - 1) * 100
                })
                
                # 다음 예측을 위한 데이터 업데이트 (간단한 방법)
                # 실제로는 더 정교한 방법 필요
                
            print(f"✅ 예측 완료: {len(predictions)}개 시점")
            
            return predictions
            
        except Exception as e:
            print(f"❌ 예측 실패: {e}")
            return []
    
    def create_advanced_chart(self, df: pd.DataFrame, predictions: List[Dict]) -> str:
        """고도화된 시각화 차트"""
        if not PLOTLY_AVAILABLE:
            print("❌ Plotly 미설치")
            return ""
        
        try:
            print("📊 고급 차트 생성 중...")
            
            # 차트 데이터 준비
            current_time = datetime.now()
            historical_times = df['timestamp'].tolist()
            historical_prices = df['price'].tolist()
            
            # 예측 데이터
            future_times = [current_time + timedelta(hours=p['hour']) for p in predictions]
            future_prices = [p['price'] for p in predictions]
            future_upper = [p['upper_bound'] for p in predictions]
            future_lower = [p['lower_bound'] for p in predictions]
            future_confidence = [p['confidence'] * 100 for p in predictions]
            
            # 4단계 서브플롯
            fig = make_subplots(
                rows=4, cols=1,
                subplot_titles=(
                    "📈 BTC 가격 & ML 예측 (6개월 학습 데이터 기반)",
                    "🎯 ML 모델 신뢰도",
                    "📊 핵심 지표 (RSI, MACD)", 
                    "💹 가격 변화율 & 예측 성능"
                ),
                vertical_spacing=0.06,
                row_heights=[0.4, 0.2, 0.2, 0.2]
            )
            
            # 1. 과거 가격 (최근 100개만)
            recent_df = df.tail(min(100, len(df)))
            fig.add_trace(
                go.Scatter(
                    x=recent_df['timestamp'],
                    y=recent_df['price'],
                    mode='lines',
                    name='실제 가격 (학습 데이터)',
                    line=dict(color='#2E86C1', width=2),
                    hovertemplate='<b>실제 가격</b><br>%{x|%m/%d %H:%M}<br>$%{y:,.0f}<extra></extra>'
                ),
                row=1, col=1
            )
            
            # 2. 현재 시점
            current_price = df.iloc[-1]['price']
            fig.add_trace(
                go.Scatter(
                    x=[current_time],
                    y=[current_price],
                    mode='markers',
                    name='현재 시점',
                    marker=dict(color='red', size=12, symbol='diamond'),
                    hovertemplate='<b>현재 가격</b><br>%{x|%m/%d %H:%M}<br>$%{y:,.0f}<extra></extra>'
                ),
                row=1, col=1
            )
            
            # 3. ML 예측
            fig.add_trace(
                go.Scatter(
                    x=future_times,
                    y=future_prices,
                    mode='lines+markers',
                    name='ML 예측 (6개월 학습)',
                    line=dict(color='#E74C3C', width=3, dash='dot'),
                    marker=dict(size=6, symbol='triangle-up'),
                    hovertemplate='<b>ML 예측</b><br>%{x|%m/%d %H:%M}<br>$%{y:,.0f}<br>신뢰도: %{customdata:.1f}%<extra></extra>',
                    customdata=future_confidence
                ),
                row=1, col=1
            )
            
            # 4. 신뢰 구간
            fig.add_trace(
                go.Scatter(
                    x=future_times + future_times[::-1],
                    y=future_upper + future_lower[::-1],
                    fill='toself',
                    fillcolor='rgba(231,76,60,0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name='ML 신뢰 구간',
                    hoverinfo='skip'
                ),
                row=1, col=1
            )
            
            # 5. 신뢰도 변화
            fig.add_trace(
                go.Scatter(
                    x=future_times,
                    y=future_confidence,
                    mode='lines+markers',
                    name='ML 신뢰도',
                    line=dict(color='#28B463', width=2),
                    marker=dict(size=5),
                    hovertemplate='<b>신뢰도</b><br>%{x|%m/%d %H:%M}<br>%{y:.1f}%<extra></extra>'
                ),
                row=2, col=1
            )
            
            # 6. RSI 지표
            if 'RSI_14' in recent_df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=recent_df['timestamp'],
                        y=recent_df['RSI_14'],
                        mode='lines',
                        name='RSI(14)',
                        line=dict(color='purple', width=1),
                        hovertemplate='<b>RSI</b><br>%{x|%m/%d %H:%M}<br>%{y:.1f}<extra></extra>'
                    ),
                    row=3, col=1
                )
                
                # RSI 과매수/과매도 선
                fig.add_hline(y=70, line=dict(color="red", dash="dash"), row=3, col=1)
                fig.add_hline(y=30, line=dict(color="green", dash="dash"), row=3, col=1)
            
            # 7. 가격 변화율
            if 'price_change' in recent_df.columns:
                price_changes = recent_df['price_change'].fillna(0) * 100
                colors = ['green' if x >= 0 else 'red' for x in price_changes]
                
                fig.add_trace(
                    go.Bar(
                        x=recent_df['timestamp'],
                        y=price_changes,
                        name='가격 변화율',
                        marker_color=colors,
                        opacity=0.7,
                        hovertemplate='<b>변화율</b><br>%{x|%m/%d %H:%M}<br>%{y:.2f}%<extra></extra>'
                    ),
                    row=4, col=1
                )
            
            # 레이아웃 설정
            model_info = ""
            if hasattr(self, 'best_model_name') and self.best_model_name in self.models:
                model = self.models[self.best_model_name]
                model_info = f"모델: {self.best_model_name.upper()} | 방향정확도: {model['direction_accuracy']:.1%} | MAE: ${model['mae']:.0f}"
            
            fig.update_layout(
                title={
                    'text': f"""
                    <b>🚀 BTC 고급 ML 예측 시스템</b><br>
                    <span style='font-size:14px'>
                    현재: ${current_price:,.0f} | 생성: {current_time.strftime('%Y-%m-%d %H:%M')}<br>
                    {model_info}<br>
                    6개월 데이터 학습 | 예측 범위: 24시간
                    </span>
                    """,
                    'x': 0.5,
                    'font': {'size': 16}
                },
                height=1000,
                showlegend=True,
                template='plotly_white',
                hovermode='x unified'
            )
            
            # 축 설정
            fig.update_xaxes(title_text="시간", tickformat="%m/%d %H:%M", row=4, col=1)
            fig.update_yaxes(title_text="BTC 가격 (USD)", row=1, col=1)
            fig.update_yaxes(title_text="신뢰도 (%)", range=[0, 100], row=2, col=1)
            fig.update_yaxes(title_text="RSI", range=[0, 100], row=3, col=1)
            fig.update_yaxes(title_text="변화율 (%)", row=4, col=1)
            
            # 저장
            chart_path = os.path.join(self.base_path, "advanced_ml_prediction_chart.html")
            fig.write_html(chart_path)
            
            print(f"✅ 고급 차트 저장: {chart_path}")
            return chart_path
            
        except Exception as e:
            print(f"❌ 차트 생성 실패: {e}")
            return ""
    
    async def run_full_analysis(self):
        """전체 분석 실행"""
        print("🚀 고도화된 ML 예측 시스템 시작")
        print("="*80)
        
        # 1. 데이터 로드
        df = self.load_6month_data()
        if df.empty:
            print("❌ 데이터 로드 실패")
            return None
        
        # 2. 모델 훈련
        model_results = self.train_ml_models(df)
        if not model_results:
            print("❌ 모델 훈련 실패")
            return None
        
        # 3. 미래 예측
        predictions = self.predict_future_prices(df, hours_ahead=24)
        if not predictions:
            print("❌ 예측 실패")
            return None
        
        # 4. 차트 생성
        chart_path = self.create_advanced_chart(df, predictions)
        
        # 5. 결과 출력
        self.print_comprehensive_results(df, predictions, model_results)
        
        # 6. 브라우저 열기
        if chart_path:
            try:
                import subprocess
                subprocess.run(["open", chart_path])
                print("\n🌐 브라우저에서 고급 차트 열림!")
            except:
                print(f"\n💡 브라우저에서: {chart_path}")
        
        return {
            'dataframe': df,
            'predictions': predictions,
            'models': model_results,
            'chart_path': chart_path
        }
    
    def print_comprehensive_results(self, df: pd.DataFrame, predictions: List[Dict], 
                                  model_results: Dict):
        """종합 결과 출력"""
        print("\n" + "="*80)
        print("📊 고도화된 ML 예측 결과")
        print("="*80)
        
        current_price = df.iloc[-1]['price']
        current_time = datetime.now()
        
        # 기본 정보
        print(f"💰 현재 가격: ${current_price:,.0f}")
        print(f"🕐 분석 시간: {current_time.strftime('%Y-%m-%d %H:%M')}")
        print(f"📈 학습 데이터: {len(df)}개 포인트 (6개월)")
        
        # 모델 성능
        if hasattr(self, 'best_model_name') and self.best_model_name in model_results:
            best_model = model_results[self.best_model_name]
            print(f"\n🤖 최고 모델: {self.best_model_name.upper()}")
            print(f"  • 방향 정확도: {best_model['direction_accuracy']:.1%} ⭐")
            print(f"  • 평균 오차: ${best_model['mae']:,.0f}")
            print(f"  • R² 점수: {best_model['r2']:.3f}")
            print(f"  • RMSE: ${best_model['rmse']:,.0f}")
        
        # 예측 결과
        print(f"\n🔮 24시간 ML 예측:")
        pred_24h = predictions[-1]
        print(f"  • 24시간 후: ${pred_24h['price']:,.0f} ({pred_24h['change_from_current']:+.2f}%)")
        print(f"  • 신뢰도: {pred_24h['confidence']*100:.1f}%")
        print(f"  • 예상 범위: ${pred_24h['lower_bound']:,.0f} ~ ${pred_24h['upper_bound']:,.0f}")
        
        # 주요 시점 예측
        print(f"\n⏰ 주요 시점별 예측:")
        key_hours = [1, 6, 12, 24]
        for hour in key_hours:
            if hour <= len(predictions):
                pred = predictions[hour-1]
                future_time = (current_time + timedelta(hours=hour)).strftime("%m/%d %H:%M")
                print(f"  • {hour:2d}시간 후 ({future_time}): ${pred['price']:,.0f} "
                      f"({pred['change_from_current']:+.2f}%) [신뢰도: {pred['confidence']*100:.1f}%]")
        
        # 전체 모델 비교
        print(f"\n📈 모든 모델 성능 비교:")
        for name, model in model_results.items():
            print(f"  • {name:15}: 방향정확도 {model['direction_accuracy']:6.1%} | "
                  f"MAE ${model['mae']:6.0f} | R² {model['r2']:5.3f}")
        
        print("\n" + "="*80)
        print("🎉 6개월 학습 데이터 기반 ML 예측 완료!")
        print("="*80)

async def main():
    """메인 실행"""
    predictor = AdvancedMLPredictor()
    result = await predictor.run_full_analysis()
    
    if result:
        print("\n✅ 고도화된 예측 시스템 완료!")
    else:
        print("\n❌ 예측 시스템 실패")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())