"""
🚀 BTC 한 달(30일) 장기 예측 시스템
- 6개월 학습 데이터 활용
- 다층 예측 모델 (일간/주간/월간)
- 불확실성 증가 모델링
- 시나리오 기반 예측
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
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression, Ridge, Lasso
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_absolute_error, r2_score
    import ta  # 기술적 지표
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("❌ 필수 라이브러리 미설치")
    exit()

class MonthlyBTCPredictor:
    """한 달 BTC 예측 시스템"""
    
    def __init__(self):
        self.base_path = "/Users/parkyoungjun/Desktop/BTC_Analysis_System"
        self.timeseries_path = os.path.join(self.base_path, "timeseries_data")
        
        # 다층 모델 구조
        self.models = {
            'short_term': {},  # 1-7일
            'medium_term': {}, # 7-14일  
            'long_term': {}    # 14-30일
        }
        
        # 시간대별 가중치 (불확실성 증가)
        self.time_weights = {
            'daily': 1.0,      # 1-7일: 높은 신뢰도
            'weekly': 0.8,     # 7-14일: 중간 신뢰도
            'monthly': 0.6     # 14-30일: 낮은 신뢰도
        }
        
    def load_6month_data(self) -> pd.DataFrame:
        """6개월 시계열 데이터 로드"""
        try:
            print("📊 6개월 시계열 데이터 로드 중...")
            
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
                        
                        print(f"  ✅ {data_type}: {len(df)}개 포인트")
            
            if master_df is None or len(master_df) == 0:
                print("❌ 데이터 로드 실패")
                return pd.DataFrame()
            
            # 정렬 및 결측치 처리
            master_df = master_df.sort_values('timestamp').reset_index(drop=True)
            for col in master_df.columns:
                if col != 'timestamp':
                    master_df[col] = master_df[col].interpolate(method='linear')
            
            # 최근 6개월 데이터
            six_months_ago = datetime.now() - timedelta(days=180)
            master_df = master_df[master_df['timestamp'] >= six_months_ago].reset_index(drop=True)
            
            print(f"✅ 통합 데이터: {len(master_df)}개 포인트")
            print(f"📅 기간: {master_df['timestamp'].min()} ~ {master_df['timestamp'].max()}")
            
            return master_df
            
        except Exception as e:
            print(f"❌ 데이터 로드 실패: {e}")
            return pd.DataFrame()
    
    def calculate_advanced_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """고급 기술적 지표 계산"""
        try:
            print("🔧 30일 예측용 고급 지표 계산 중...")
            
            # 기본 가격 지표
            df['price_ma7'] = df['price'].rolling(7).mean()
            df['price_ma14'] = df['price'].rolling(14).mean() 
            df['price_ma30'] = df['price'].rolling(30).mean()
            df['price_std7'] = df['price'].rolling(7).std()
            df['price_std14'] = df['price'].rolling(14).std()
            
            # 변동성 지표
            df['volatility_7d'] = df['price'].rolling(7).std() / df['price'].rolling(7).mean()
            df['volatility_14d'] = df['price'].rolling(14).std() / df['price'].rolling(14).mean()
            df['volatility_30d'] = df['price'].rolling(30).std() / df['price'].rolling(30).mean()
            
            # RSI (다중 기간)
            df['rsi_7'] = ta.momentum.rsi(df['price'], window=7)
            df['rsi_14'] = ta.momentum.rsi(df['price'], window=14)
            df['rsi_21'] = ta.momentum.rsi(df['price'], window=21)
            
            # MACD
            macd = ta.trend.MACD(df['price'])
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            df['macd_histogram'] = macd.macd_diff()
            
            # 볼린저 밴드
            bb = ta.volatility.BollingerBands(df['price'], window=20)
            df['bb_upper'] = bb.bollinger_hband()
            df['bb_lower'] = bb.bollinger_lband()
            df['bb_middle'] = bb.bollinger_mavg()
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            df['bb_position'] = (df['price'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            # 거래량 지표
            if 'volume' in df.columns:
                df['volume_ma7'] = df['volume'].rolling(7).mean()
                df['volume_ratio'] = df['volume'] / df['volume_ma7']
                df['price_volume'] = df['price'] * df['volume']
            
            # 추세 지표
            df['price_trend_7d'] = (df['price'] - df['price'].shift(7)) / df['price'].shift(7)
            df['price_trend_14d'] = (df['price'] - df['price'].shift(14)) / df['price'].shift(14)
            df['price_trend_30d'] = (df['price'] - df['price'].shift(30)) / df['price'].shift(30)
            
            # 모멘텀 지표
            df['momentum_3d'] = df['price'] / df['price'].shift(3) - 1
            df['momentum_7d'] = df['price'] / df['price'].shift(7) - 1
            df['momentum_14d'] = df['price'] / df['price'].shift(14) - 1
            
            # Williams %R
            df['williams_r'] = ta.momentum.williams_r(df['price'], df['price'], df['price'], window=14)
            
            # 일간 수익률
            df['daily_return'] = df['price'].pct_change()
            df['daily_return_ma7'] = df['daily_return'].rolling(7).mean()
            
            # 시장 상태 지표
            df['market_state'] = np.where(df['price'] > df['price_ma30'], 1, 
                                 np.where(df['price'] < df['price_ma30'], -1, 0))
            
            print(f"✅ 고급 지표 계산 완료: {len(df.columns)}개 컬럼")
            return df
            
        except Exception as e:
            print(f"❌ 지표 계산 실패: {e}")
            return df
    
    def train_multilayer_models(self, df: pd.DataFrame) -> Dict:
        """다층 예측 모델 훈련"""
        try:
            print("🤖 다층 예측 모델 훈련 중...")
            
            # 피처 선택 (NaN이 적은 컬럼들)
            feature_cols = []
            for col in df.columns:
                if col not in ['timestamp', 'price'] and df[col].notna().sum() > len(df) * 0.7:
                    feature_cols.append(col)
            
            print(f"📊 선택된 피처: {len(feature_cols)}개")
            
            # 결측치 제거
            df_clean = df[['price'] + feature_cols].dropna()
            
            if len(df_clean) < 60:
                print("❌ 훈련 데이터 부족")
                return {}
            
            print(f"✅ 훈련 데이터: {len(df_clean)}개 샘플")
            
            model_results = {}
            
            # 3가지 예측 기간별 모델 훈련
            prediction_periods = {
                'short_term': [1, 3, 7],      # 1-7일
                'medium_term': [7, 10, 14],   # 7-14일
                'long_term': [14, 21, 30]     # 14-30일
            }
            
            for period_name, days in prediction_periods.items():
                print(f"  🎯 {period_name} 모델 훈련 중...")
                period_models = {}
                
                for target_days in days:
                    if len(df_clean) <= target_days:
                        continue
                        
                    # 타겟 생성 (N일 후 가격)
                    X = df_clean[feature_cols].iloc[:-target_days].reset_index(drop=True)
                    y = df_clean['price'].iloc[target_days:].reset_index(drop=True)
                    
                    # 훈련/테스트 분할
                    split_idx = int(len(X) * 0.8)
                    X_train = X.iloc[:split_idx]
                    X_test = X.iloc[split_idx:]
                    y_train = y.iloc[:split_idx]
                    y_test = y.iloc[split_idx:]
                    
                    # 스케일링
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
                    
                    # 모델들
                    models = {
                        'RandomForest': RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1),
                        'GradientBoosting': GradientBoostingRegressor(n_estimators=50, random_state=42),
                        'Ridge': Ridge(alpha=1.0),
                        'Linear': LinearRegression()
                    }
                    
                    best_model = None
                    best_score = float('inf')
                    
                    for name, model in models.items():
                        try:
                            if name in ['Ridge', 'Linear']:
                                model.fit(X_train_scaled, y_train)
                                y_pred = model.predict(X_test_scaled)
                            else:
                                model.fit(X_train, y_train)
                                y_pred = model.predict(X_test)
                            
                            mae = mean_absolute_error(y_test, y_pred)
                            r2 = r2_score(y_test, y_pred)
                            
                            # 방향 정확도
                            direction_accuracy = np.mean((y_test.values[1:] > y_test.values[:-1]) == 
                                                       (y_pred[1:] > y_pred[:-1])) if len(y_test) > 1 else 0.5
                            
                            if mae < best_score:
                                best_score = mae
                                best_model = {
                                    'name': name,
                                    'model': model,
                                    'scaler': scaler if name in ['Ridge', 'Linear'] else None,
                                    'mae': mae,
                                    'r2': r2,
                                    'direction_accuracy': direction_accuracy,
                                    'features': feature_cols
                                }
                            
                        except Exception as e:
                            continue
                    
                    if best_model:
                        period_models[f'{target_days}d'] = best_model
                        print(f"    • {target_days}일: {best_model['name']} (MAE=${best_model['mae']:,.0f})")
                
                if period_models:
                    model_results[period_name] = period_models
            
            print(f"✅ 다층 모델 훈련 완료: {len(model_results)}개 기간")
            return model_results
            
        except Exception as e:
            print(f"❌ 모델 훈련 실패: {e}")
            return {}
    
    def generate_monthly_predictions(self, df: pd.DataFrame, models: Dict) -> List[Dict]:
        """30일 장기 예측 생성"""
        try:
            print("🔮 30일 장기 예측 생성 중...")
            
            if not models:
                return []
            
            current_price = df['price'].iloc[-1]
            current_time = datetime.now()
            
            predictions = []
            
            # 30일간 일별 예측
            for day in range(1, 31):
                pred_time = current_time + timedelta(days=day)
                
                # 기간별 모델 선택 및 예측
                if day <= 7 and 'short_term' in models:
                    # 단기 모델 사용
                    period_models = models['short_term']
                    confidence_base = self.time_weights['daily']
                    
                elif day <= 14 and 'medium_term' in models:
                    # 중기 모델 사용
                    period_models = models['medium_term']
                    confidence_base = self.time_weights['weekly']
                    
                else:
                    # 장기 모델 사용
                    period_models = models.get('long_term', models.get('medium_term', models.get('short_term', {})))
                    confidence_base = self.time_weights['monthly']
                
                if not period_models:
                    continue
                
                # 가장 적합한 모델 선택
                selected_model = None
                for model_key in sorted(period_models.keys()):
                    target_days = int(model_key.replace('d', ''))
                    if target_days >= day:
                        selected_model = period_models[model_key]
                        break
                
                if not selected_model:
                    # 가장 가까운 모델 사용
                    selected_model = list(period_models.values())[-1]
                
                # 예측 실행
                try:
                    features = df[selected_model['features']].iloc[-1:].values
                    
                    if selected_model['scaler']:
                        features = selected_model['scaler'].transform(features)
                    
                    base_pred = selected_model['model'].predict(features)[0]
                    
                    # 시간에 따른 불확실성 증가
                    uncertainty_factor = 1 + (day / 30) * 0.3
                    noise = np.random.normal(0, selected_model['mae'] * uncertainty_factor * 0.1)
                    
                    predicted_price = base_pred + noise
                    
                    # 신뢰도 계산 (시간이 지날수록 감소)
                    confidence = confidence_base * selected_model['direction_accuracy'] * (1 - day * 0.02)
                    confidence = max(confidence, 0.1)
                    
                    # 신뢰 구간
                    margin = selected_model['mae'] * uncertainty_factor
                    upper_bound = predicted_price + margin
                    lower_bound = predicted_price - margin
                    
                    # 변화율 계산
                    change_pct = ((predicted_price - current_price) / current_price) * 100
                    
                    prediction = {
                        'day': day,
                        'date': pred_time.strftime('%Y-%m-%d'),
                        'timestamp': pred_time,
                        'price': predicted_price,
                        'upper_bound': upper_bound,
                        'lower_bound': lower_bound,
                        'confidence': confidence * 100,
                        'change_pct': change_pct,
                        'model_used': selected_model['name'],
                        'period_type': 'short' if day <= 7 else 'medium' if day <= 14 else 'long'
                    }
                    
                    predictions.append(prediction)
                    
                except Exception as e:
                    continue
            
            print(f"✅ 30일 예측 완료: {len(predictions)}개 시점")
            return predictions
            
        except Exception as e:
            print(f"❌ 예측 생성 실패: {e}")
            return []
    
    def create_monthly_chart(self, df: pd.DataFrame, predictions: List[Dict]) -> str:
        """30일 예측 차트 생성"""
        try:
            print("📊 30일 궁극 예측 차트 생성 중...")
            
            if not predictions:
                return ""
            
            # 차트 생성
            fig = make_subplots(
                rows=4, cols=1,
                subplot_titles=(
                    "🪙 BTC 30일 장기 예측 차트",
                    "📊 예측 신뢰도 변화", 
                    "📈 일간 변화율 예상",
                    "🎯 주요 가격대 분포"
                ),
                vertical_spacing=0.08,
                row_heights=[0.4, 0.2, 0.2, 0.2]
            )
            
            # 1. 과거 30일 데이터
            recent_30d = df.tail(30)
            fig.add_trace(
                go.Scatter(
                    x=recent_30d['timestamp'],
                    y=recent_30d['price'],
                    mode='lines+markers',
                    name='과거 30일 실제',
                    line=dict(color='#1f77b4', width=2),
                    marker=dict(size=4)
                ),
                row=1, col=1
            )
            
            # 2. 현재 시점
            current_time = datetime.now()
            current_price = df['price'].iloc[-1]
            
            fig.add_trace(
                go.Scatter(
                    x=[current_time],
                    y=[current_price],
                    mode='markers',
                    name='현재 시점',
                    marker=dict(color='red', size=15, symbol='diamond')
                ),
                row=1, col=1
            )
            
            # 3. 미래 30일 예측
            pred_times = [p['timestamp'] for p in predictions]
            pred_prices = [p['price'] for p in predictions]
            pred_upper = [p['upper_bound'] for p in predictions]
            pred_lower = [p['lower_bound'] for p in predictions]
            pred_confidence = [p['confidence'] for p in predictions]
            
            # 예측 라인
            fig.add_trace(
                go.Scatter(
                    x=pred_times,
                    y=pred_prices,
                    mode='lines+markers',
                    name='30일 AI 예측',
                    line=dict(color='#ff7f0e', width=3, dash='dot'),
                    marker=dict(size=5, symbol='triangle-up')
                ),
                row=1, col=1
            )
            
            # 4. 신뢰 구간
            fig.add_trace(
                go.Scatter(
                    x=pred_times + pred_times[::-1],
                    y=pred_upper + pred_lower[::-1],
                    fill='toself',
                    fillcolor='rgba(255,127,14,0.15)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name='예측 신뢰구간',
                    hoverinfo='skip'
                ),
                row=1, col=1
            )
            
            # 5. 신뢰도 차트
            fig.add_trace(
                go.Scatter(
                    x=pred_times,
                    y=pred_confidence,
                    mode='lines+markers',
                    name='AI 신뢰도',
                    line=dict(color='#2ca02c', width=2),
                    marker=dict(size=4)
                ),
                row=2, col=1
            )
            
            # 6. 일간 변화율
            daily_changes = [p['change_pct'] for p in predictions]
            fig.add_trace(
                go.Bar(
                    x=pred_times,
                    y=daily_changes,
                    name='일간 변화율',
                    marker_color=['green' if x >= 0 else 'red' for x in daily_changes]
                ),
                row=3, col=1
            )
            
            # 7. 가격 분포 히스토그램
            fig.add_trace(
                go.Histogram(
                    x=pred_prices,
                    nbinsx=20,
                    name='가격 분포',
                    marker_color='lightblue',
                    opacity=0.7
                ),
                row=4, col=1
            )
            
            # 레이아웃 설정
            title_text = f"""
            <b>🚀 BTC 30일 장기 예측 분석 시스템</b><br>
            <span style='font-size:14px'>
            현재: ${current_price:,.0f} | 생성시간: {current_time.strftime('%Y-%m-%d %H:%M')}<br>
            예측범위: 30일 | 평균 신뢰도: {np.mean(pred_confidence):.1f}% | 학습데이터: 6개월
            </span>
            """
            
            fig.update_layout(
                title={
                    'text': title_text,
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 16}
                },
                height=1200,
                showlegend=True,
                template='plotly_white',
                hovermode='x unified'
            )
            
            # 축 설정
            fig.update_yaxes(title_text="BTC 가격 (USD)", row=1, col=1)
            fig.update_yaxes(title_text="신뢰도 (%)", range=[0, 100], row=2, col=1)
            fig.update_yaxes(title_text="변화율 (%)", row=3, col=1)
            fig.update_yaxes(title_text="빈도", row=4, col=1)
            fig.update_xaxes(title_text="날짜", row=4, col=1)
            
            # 저장
            chart_path = os.path.join(self.base_path, "btc_monthly_prediction_chart.html")
            fig.write_html(chart_path, include_plotlyjs=True)
            
            print(f"✅ 30일 차트 저장: {chart_path}")
            return chart_path
            
        except Exception as e:
            print(f"❌ 차트 생성 실패: {e}")
            return ""
    
    def save_predictions_json(self, predictions: List[Dict], current_price: float) -> str:
        """예측 결과를 JSON으로 저장"""
        try:
            result = {
                "generation_time": datetime.now().isoformat(),
                "current_price": current_price,
                "prediction_period": "30_days",
                "total_predictions": len(predictions),
                "predictions": predictions,
                "summary": {
                    "avg_confidence": np.mean([p['confidence'] for p in predictions]),
                    "price_range": {
                        "min": min([p['lower_bound'] for p in predictions]),
                        "max": max([p['upper_bound'] for p in predictions])
                    },
                    "final_price": predictions[-1]['price'] if predictions else current_price,
                    "total_change_pct": predictions[-1]['change_pct'] if predictions else 0
                }
            }
            
            json_path = os.path.join(self.base_path, "monthly_predictions.json")
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False, default=str)
            
            print(f"✅ 예측 데이터 저장: {json_path}")
            return json_path
            
        except Exception as e:
            print(f"❌ JSON 저장 실패: {e}")
            return ""

async def main():
    """메인 실행"""
    print("🚀 BTC 30일 장기 예측 시스템 시작")
    print("=" * 80)
    
    predictor = MonthlyBTCPredictor()
    
    # 1. 데이터 로드
    df = predictor.load_6month_data()
    if df.empty:
        print("❌ 데이터 로드 실패")
        return
    
    # 2. 지표 계산
    df = predictor.calculate_advanced_indicators(df)
    
    # 3. 모델 훈련
    models = predictor.train_multilayer_models(df)
    if not models:
        print("❌ 모델 훈련 실패")
        return
    
    # 4. 예측 생성
    predictions = predictor.generate_monthly_predictions(df, models)
    if not predictions:
        print("❌ 예측 생성 실패")
        return
    
    # 5. 차트 생성
    chart_path = predictor.create_monthly_chart(df, predictions)
    
    # 6. 결과 저장
    current_price = df['price'].iloc[-1]
    json_path = predictor.save_predictions_json(predictions, current_price)
    
    # 7. 결과 출력
    print("\n" + "=" * 80)
    print("🚀 BTC 30일 장기 예측 결과")
    print("=" * 80)
    print(f"💰 현재 가격: ${current_price:,.0f}")
    print(f"🕐 분석 시간: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"📊 학습 데이터: {len(df)}일 (6개월)")
    
    if predictions:
        # 주요 예측 결과
        pred_1w = next((p for p in predictions if p['day'] == 7), None)
        pred_2w = next((p for p in predictions if p['day'] == 14), None) 
        pred_1m = predictions[-1]
        
        print(f"\n🔮 주요 예측 결과:")
        if pred_1w:
            print(f"  • 1주일 후: ${pred_1w['price']:,.0f} ({pred_1w['change_pct']:+.2f}%) [신뢰도: {pred_1w['confidence']:.1f}%]")
        if pred_2w:
            print(f"  • 2주일 후: ${pred_2w['price']:,.0f} ({pred_2w['change_pct']:+.2f}%) [신뢰도: {pred_2w['confidence']:.1f}%]")
        print(f"  • 1개월 후: ${pred_1m['price']:,.0f} ({pred_1m['change_pct']:+.2f}%) [신뢰도: {pred_1m['confidence']:.1f}%]")
        
        # 통계
        avg_confidence = np.mean([p['confidence'] for p in predictions])
        price_volatility = np.std([p['price'] for p in predictions])
        
        print(f"\n📊 예측 통계:")
        print(f"  • 평균 신뢰도: {avg_confidence:.1f}%")
        print(f"  • 예측 변동성: ${price_volatility:,.0f}")
        print(f"  • 최고 예상: ${max([p['upper_bound'] for p in predictions]):,.0f}")
        print(f"  • 최저 예상: ${min([p['lower_bound'] for p in predictions]):,.0f}")
    
    # 브라우저 열기
    if chart_path:
        try:
            import subprocess
            subprocess.run(["open", chart_path], check=True)
            print(f"\n🌐 30일 예측 차트가 브라우저에서 열렸습니다!")
        except:
            print(f"\n💡 브라우저에서 열어보세요: {chart_path}")
    
    print("\n" + "=" * 80)
    print("🎉 30일 장기 예측 완료!")
    print("=" * 80)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())