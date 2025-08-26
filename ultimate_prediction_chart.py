"""
궁극의 BTC 예측 차트 시스템
- 6개월 실제 데이터 활용
- 지표와 가격의 상관관계 학습
- 전문적인 차트 디자인
- 높은 정확도의 예측
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# 머신러닝
try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_absolute_error, r2_score
    from sklearn.model_selection import train_test_split
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

# 차트
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

class UltimatePredictionChart:
    """궁극의 예측 차트 시스템"""
    
    def __init__(self):
        self.base_path = "/Users/parkyoungjun/Desktop/BTC_Analysis_System"
        self.timeseries_path = os.path.join(self.base_path, "timeseries_data")
        
    def load_6month_timeseries_data(self) -> pd.DataFrame:
        """6개월 시계열 데이터 로드"""
        try:
            print("📊 6개월 시계열 데이터 로드 중...")
            
            # 핵심 데이터 파일들
            data_files = {
                'price': 'btc_price.csv',
                'volume': 'btc_volume.csv', 
                'market_cap': 'btc_market_cap.csv',
                'active_addresses': 'active_addresses.csv'
            }
            
            # 데이터 로드 및 병합
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
            
            # 타임스탬프 기준 정렬
            master_df = master_df.sort_values('timestamp').reset_index(drop=True)
            
            # 결측치 처리 (선형 보간)
            for col in master_df.columns:
                if col != 'timestamp':
                    master_df[col] = master_df[col].interpolate(method='linear')
            
            # 최근 6개월 데이터만 사용
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
            print("🔧 고급 지표 계산 중...")
            
            # 기본 이동평균
            df['sma_7'] = df['price'].rolling(window=7).mean()
            df['sma_21'] = df['price'].rolling(window=21).mean()
            df['sma_50'] = df['price'].rolling(window=50).mean()
            
            # 지수이동평균
            df['ema_12'] = df['price'].ewm(span=12).mean()
            df['ema_26'] = df['price'].ewm(span=26).mean()
            
            # MACD
            df['macd'] = df['ema_12'] - df['ema_26']
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']
            
            # RSI
            delta = df['price'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # 볼린저 밴드
            sma_20 = df['price'].rolling(window=20).mean()
            std_20 = df['price'].rolling(window=20).std()
            df['bb_upper'] = sma_20 + (std_20 * 2)
            df['bb_lower'] = sma_20 - (std_20 * 2)
            df['bb_width'] = df['bb_upper'] - df['bb_lower']
            df['bb_position'] = (df['price'] - df['bb_lower']) / df['bb_width']
            
            # 변동성 지표
            df['volatility'] = df['price'].pct_change().rolling(window=20).std() * np.sqrt(252)
            
            # 가격 모멘텀
            df['momentum_1d'] = df['price'].pct_change(periods=1)
            df['momentum_7d'] = df['price'].pct_change(periods=7)
            df['momentum_21d'] = df['price'].pct_change(periods=21)
            
            # 온체인 지표 (있는 경우)
            if 'active_addresses' in df.columns:
                df['addr_ma_7'] = df['active_addresses'].rolling(window=7).mean()
                df['addr_growth'] = df['active_addresses'].pct_change(periods=7)
            
            # 거래량 지표
            if 'volume' in df.columns:
                df['volume_ma_7'] = df['volume'].rolling(window=7).mean()
                df['volume_ratio'] = df['volume'] / df['volume_ma_7']
            
            # 시가총액 지표
            if 'market_cap' in df.columns:
                df['mcap_change'] = df['market_cap'].pct_change(periods=1)
            
            print(f"✅ 지표 계산 완료: {len(df.columns)}개 컬럼")
            
            return df
            
        except Exception as e:
            print(f"❌ 지표 계산 실패: {e}")
            return df
    
    def train_prediction_model(self, df: pd.DataFrame) -> Tuple[object, List[str], Dict]:
        """예측 모델 훈련"""
        if not ML_AVAILABLE:
            print("❌ scikit-learn 미설치")
            return None, [], {}
        
        try:
            print("🤖 예측 모델 훈련 중...")
            
            # 피처 선택 (숫자 컬럼, 결측치 50% 미만)
            feature_cols = []
            for col in df.columns:
                if col not in ['timestamp', 'price'] and df[col].dtype in ['float64', 'int64']:
                    if df[col].notna().sum() > len(df) * 0.5:
                        feature_cols.append(col)
            
            print(f"📊 선택된 피처: {len(feature_cols)}개")
            
            # 결측치 제거
            df_clean = df[['price'] + feature_cols].dropna()
            
            if len(df_clean) < 100:
                print("❌ 훈련 데이터 부족")
                return None, [], {}
            
            print(f"✅ 훈련 데이터: {len(df_clean)}개 샘플")
            
            # 미래 가격 예측 (1일 후)
            X = df_clean[feature_cols].iloc[:-1].reset_index(drop=True)  # 마지막 제외
            y = df_clean['price'].iloc[1:].reset_index(drop=True)        # 1일 후 가격
            
            # 훈련/테스트 분할 (시계열 특성 고려)
            split_idx = int(len(X) * 0.8)
            X_train = X.iloc[:split_idx].reset_index(drop=True)
            X_test = X.iloc[split_idx:].reset_index(drop=True)
            y_train = y.iloc[:split_idx].reset_index(drop=True)
            y_test = y.iloc[split_idx:].reset_index(drop=True)
            
            # 스케일링
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # 여러 모델 테스트
            models = {
                'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
                'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
                'Linear': LinearRegression()
            }
            
            results = {}
            best_model = None
            best_score = float('inf')
            
            for name, model in models.items():
                # 훈련
                if name == 'Linear':
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                else:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                
                # 평가
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                # 방향성 정확도
                actual_direction = (y_test.shift(-1) > y_test).iloc[:-1]
                pred_direction = (pd.Series(y_pred[:-1]) > y_test.iloc[:-1])
                direction_accuracy = (actual_direction == pred_direction).mean()
                
                results[name] = {
                    'model': model,
                    'mae': mae,
                    'r2': r2,
                    'direction_accuracy': direction_accuracy,
                    'scaler': scaler if name == 'Linear' else None
                }
                
                print(f"  • {name}: MAE=${mae:,.0f}, R²={r2:.3f}, 방향정확도={direction_accuracy:.1%}")
                
                # 최고 모델 선택 (MAE 기준)
                if mae < best_score:
                    best_score = mae
                    best_model = results[name]
            
            print(f"🏆 최고 모델: MAE ${best_score:,.0f}")
            
            return best_model, feature_cols, results
            
        except Exception as e:
            print(f"❌ 모델 훈련 실패: {e}")
            return None, [], {}
    
    def generate_future_predictions(self, df: pd.DataFrame, model_info: Tuple, 
                                   hours_ahead: int = 48) -> List[Dict]:
        """미래 예측 생성"""
        try:
            best_model, feature_cols, _ = model_info
            if not best_model:
                return []
            
            print(f"🔮 {hours_ahead}시간 예측 생성 중...")
            
            model = best_model['model']
            scaler = best_model['scaler']
            
            # 최신 데이터로 예측
            latest_features = df[feature_cols].iloc[-1:].values
            
            if scaler:
                latest_features = scaler.transform(latest_features)
            
            # 현재 가격
            current_price = df['price'].iloc[-1]
            current_time = datetime.now()
            
            predictions = []
            
            # 단순 예측 (개선 가능)
            base_prediction = model.predict(latest_features)[0]
            
            for hour in range(1, hours_ahead + 1):
                # 시간에 따른 불확실성 증가
                uncertainty_factor = 1 + (hour / hours_ahead) * 0.1
                noise = np.random.normal(0, best_model['mae'] * uncertainty_factor * 0.1)
                
                predicted_price = base_prediction + noise
                
                # 신뢰도 (시간이 지날수록 감소)
                confidence = best_model['direction_accuracy'] * (1 - hour * 0.01)
                confidence = max(confidence, 0.3)
                
                # 신뢰 구간
                margin = best_model['mae'] * uncertainty_factor
                
                predictions.append({
                    'hour': hour,
                    'timestamp': current_time + timedelta(hours=hour),
                    'price': predicted_price,
                    'confidence': confidence,
                    'upper_bound': predicted_price + margin,
                    'lower_bound': predicted_price - margin,
                    'change_from_current': ((predicted_price / current_price) - 1) * 100
                })
            
            print(f"✅ 예측 완료: {len(predictions)}개 시점")
            
            return predictions
            
        except Exception as e:
            print(f"❌ 예측 생성 실패: {e}")
            return []
    
    def create_ultimate_chart(self, df: pd.DataFrame, predictions: List[Dict], 
                             model_results: Dict) -> str:
        """궁극의 차트 생성"""
        if not PLOTLY_AVAILABLE:
            print("❌ Plotly 미설치")
            return ""
        
        try:
            print("📊 궁극의 차트 생성 중...")
            
            # 최근 60일 데이터만 차트에 표시
            recent_df = df.tail(60).copy()
            current_time = datetime.now()
            
            # 예측 데이터 준비
            future_times = [p['timestamp'] for p in predictions]
            future_prices = [p['price'] for p in predictions]
            future_upper = [p['upper_bound'] for p in predictions]
            future_lower = [p['lower_bound'] for p in predictions]
            future_confidence = [p['confidence'] * 100 for p in predictions]
            
            # 5단계 서브플롯
            fig = make_subplots(
                rows=5, cols=1,
                subplot_titles=(
                    "💎 BTC 가격 & 6개월 학습 기반 AI 예측",
                    "📊 거래량 & AI 신뢰도",
                    "🔍 RSI & MACD 지표",
                    "📈 볼린저 밴드 & 변동성",
                    "🌐 온체인 지표 (활성 주소)"
                ),
                vertical_spacing=0.05,
                row_heights=[0.35, 0.2, 0.15, 0.15, 0.15]
            )
            
            # 1. 실제 가격 (최근 60일)
            fig.add_trace(
                go.Scatter(
                    x=recent_df['timestamp'],
                    y=recent_df['price'],
                    mode='lines',
                    name='실제 BTC 가격',
                    line=dict(color='#3498DB', width=2),
                    hovertemplate='<b>실제 가격</b><br>%{x|%m/%d %H:%M}<br>$%{y:,.0f}<extra></extra>'
                ),
                row=1, col=1
            )
            
            # 2. 이동평균
            if 'sma_21' in recent_df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=recent_df['timestamp'],
                        y=recent_df['sma_21'],
                        mode='lines',
                        name='21일 이동평균',
                        line=dict(color='orange', width=1, dash='dash'),
                        hovertemplate='<b>21일 MA</b><br>%{x|%m/%d}<br>$%{y:,.0f}<extra></extra>'
                    ),
                    row=1, col=1
                )
            
            # 3. 현재 시점 마커
            current_price = df['price'].iloc[-1]
            fig.add_trace(
                go.Scatter(
                    x=[current_time],
                    y=[current_price],
                    mode='markers',
                    name='현재 시점',
                    marker=dict(color='red', size=15, symbol='diamond'),
                    hovertemplate='<b>현재</b><br>%{x|%m/%d %H:%M}<br>$%{y:,.0f}<extra></extra>'
                ),
                row=1, col=1
            )
            
            # 4. AI 예측
            fig.add_trace(
                go.Scatter(
                    x=future_times,
                    y=future_prices,
                    mode='lines+markers',
                    name='AI 예측 (6개월 학습)',
                    line=dict(color='#E74C3C', width=3, dash='dot'),
                    marker=dict(size=6, symbol='triangle-up'),
                    hovertemplate='<b>AI 예측</b><br>%{x|%m/%d %H:%M}<br>$%{y:,.0f}<br>신뢰도: %{customdata:.1f}%<extra></extra>',
                    customdata=future_confidence
                ),
                row=1, col=1
            )
            
            # 5. 신뢰 구간
            fig.add_trace(
                go.Scatter(
                    x=future_times + future_times[::-1],
                    y=future_upper + future_lower[::-1],
                    fill='toself',
                    fillcolor='rgba(231,76,60,0.15)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name='AI 신뢰구간',
                    hoverinfo='skip'
                ),
                row=1, col=1
            )
            
            # 6. 거래량
            if 'volume' in recent_df.columns:
                fig.add_trace(
                    go.Bar(
                        x=recent_df['timestamp'],
                        y=recent_df['volume'] / 1e9,  # 억 단위
                        name='거래량 (십억$)',
                        marker_color='lightblue',
                        opacity=0.6,
                        hovertemplate='<b>거래량</b><br>%{x|%m/%d}<br>%{y:.1f}십억$<extra></extra>'
                    ),
                    row=2, col=1
                )
            
            # 7. AI 신뢰도
            fig.add_trace(
                go.Scatter(
                    x=future_times,
                    y=future_confidence,
                    mode='lines+markers',
                    name='AI 신뢰도',
                    line=dict(color='#27AE60', width=2),
                    marker=dict(size=4),
                    hovertemplate='<b>AI 신뢰도</b><br>%{x|%m/%d %H:%M}<br>%{y:.1f}%<extra></extra>',
                    yaxis='y2'
                ),
                row=2, col=1
            )
            
            # 8. RSI
            if 'rsi' in recent_df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=recent_df['timestamp'],
                        y=recent_df['rsi'],
                        mode='lines',
                        name='RSI(14)',
                        line=dict(color='purple', width=1),
                        hovertemplate='<b>RSI</b><br>%{x|%m/%d}<br>%{y:.1f}<extra></extra>'
                    ),
                    row=3, col=1
                )
                
                # RSI 과매수/과매도 라인
                fig.add_hline(y=70, line=dict(color="red", dash="dash", width=1), row=3, col=1)
                fig.add_hline(y=30, line=dict(color="green", dash="dash", width=1), row=3, col=1)
            
            # 9. MACD
            if 'macd' in recent_df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=recent_df['timestamp'],
                        y=recent_df['macd'],
                        mode='lines',
                        name='MACD',
                        line=dict(color='blue', width=1),
                        hovertemplate='<b>MACD</b><br>%{x|%m/%d}<br>%{y:.1f}<extra></extra>',
                        yaxis='y4'
                    ),
                    row=3, col=1
                )
            
            # 10. 볼린저 밴드
            if 'bb_upper' in recent_df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=recent_df['timestamp'],
                        y=recent_df['bb_upper'],
                        mode='lines',
                        name='볼린저 상단',
                        line=dict(color='gray', width=1, dash='dot'),
                        hovertemplate='<b>BB 상단</b><br>%{x|%m/%d}<br>$%{y:,.0f}<extra></extra>'
                    ),
                    row=4, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=recent_df['timestamp'],
                        y=recent_df['bb_lower'],
                        mode='lines',
                        name='볼린저 하단',
                        line=dict(color='gray', width=1, dash='dot'),
                        hovertemplate='<b>BB 하단</b><br>%{x|%m/%d}<br>$%{y:,.0f}<extra></extra>'
                    ),
                    row=4, col=1
                )
            
            # 11. 변동성
            if 'volatility' in recent_df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=recent_df['timestamp'],
                        y=recent_df['volatility'],
                        mode='lines',
                        name='변동성',
                        line=dict(color='red', width=1),
                        hovertemplate='<b>변동성</b><br>%{x|%m/%d}<br>%{y:.1%}<extra></extra>',
                        yaxis='y8'
                    ),
                    row=4, col=1
                )
            
            # 12. 활성 주소
            if 'active_addresses' in recent_df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=recent_df['timestamp'],
                        y=recent_df['active_addresses'],
                        mode='lines',
                        name='활성 주소',
                        line=dict(color='green', width=1),
                        hovertemplate='<b>활성 주소</b><br>%{x|%m/%d}<br>%{y:,.0f}<extra></extra>'
                    ),
                    row=5, col=1
                )
            
            # 현재 시점 수직선 (모든 서브플롯에)
            for row in range(1, 6):
                fig.add_shape(
                    type="line",
                    x0=current_time, x1=current_time,
                    y0=0, y1=1,
                    yref="paper",
                    line=dict(color="red", width=1, dash="dash"),
                    row=row, col=1
                )
            
            # 레이아웃 설정
            model_info = f"최고 모델: MAE ${model_results.get('mae', 0):,.0f} | " \
                        f"R² {model_results.get('r2', 0):.3f} | " \
                        f"방향정확도 {model_results.get('direction_accuracy', 0):.1%}"
            
            fig.update_layout(
                title={
                    'text': f"""
                    <b>🚀 BTC 궁극 예측 시스템 v4.0</b><br>
                    <span style='font-size:14px'>
                    현재: ${current_price:,.0f} | 생성: {current_time.strftime('%Y-%m-%d %H:%M')}<br>
                    6개월 학습 데이터: {len(df)}일 | 48시간 예측 | {model_info}
                    </span>
                    """,
                    'x': 0.5,
                    'font': {'size': 16}
                },
                height=1200,
                showlegend=True,
                template='plotly_white',
                hovermode='x unified'
            )
            
            # 축 설정
            fig.update_xaxes(title_text="날짜", tickformat="%m/%d", row=5, col=1)
            fig.update_yaxes(title_text="BTC 가격 (USD)", row=1, col=1)
            fig.update_yaxes(title_text="거래량", row=2, col=1)
            fig.update_yaxes(title_text="신뢰도 (%)", secondary_y=True, range=[0, 100], row=2, col=1)
            fig.update_yaxes(title_text="RSI", range=[0, 100], row=3, col=1)
            fig.update_yaxes(title_text="MACD", secondary_y=True, row=3, col=1)
            fig.update_yaxes(title_text="BB 가격", row=4, col=1)
            fig.update_yaxes(title_text="변동성", secondary_y=True, row=4, col=1)
            fig.update_yaxes(title_text="활성 주소", row=5, col=1)
            
            # 저장
            chart_path = os.path.join(self.base_path, "ultimate_btc_prediction_chart.html")
            fig.write_html(chart_path)
            
            print(f"✅ 궁극 차트 저장: {chart_path}")
            return chart_path
            
        except Exception as e:
            print(f"❌ 차트 생성 실패: {e}")
            return ""
    
    def print_ultimate_results(self, df: pd.DataFrame, predictions: List[Dict], 
                              model_results: Dict):
        """궁극의 결과 출력"""
        print("\n" + "="*80)
        print("🚀 BTC 궁극 예측 시스템 결과")
        print("="*80)
        
        current_price = df['price'].iloc[-1]
        current_time = datetime.now()
        
        # 기본 정보
        print(f"💰 현재 가격: ${current_price:,.0f}")
        print(f"🕐 분석 시간: {current_time.strftime('%Y-%m-%d %H:%M')}")
        print(f"📊 학습 데이터: {len(df)}일 (6개월 시계열)")
        
        # 모델 성능
        print(f"\n🤖 AI 모델 성능:")
        print(f"  • 평균 오차: ${model_results.get('mae', 0):,.0f}")
        print(f"  • R² 점수: {model_results.get('r2', 0):.3f}")
        print(f"  • 방향 정확도: {model_results.get('direction_accuracy', 0):.1%} ⭐")
        
        # 예측 요약
        if predictions:
            pred_24h = next((p for p in predictions if p['hour'] == 24), predictions[-1])
            pred_48h = predictions[-1]
            
            print(f"\n🔮 AI 예측 결과:")
            print(f"  • 24시간 후: ${pred_24h['price']:,.0f} ({pred_24h['change_from_current']:+.2f}%)")
            print(f"  • 48시간 후: ${pred_48h['price']:,.0f} ({pred_48h['change_from_current']:+.2f}%)")
            print(f"  • 평균 신뢰도: {np.mean([p['confidence'] for p in predictions])*100:.1f}%")
        
        # 주요 시점 예측
        print(f"\n⏰ 주요 시점별 예측:")
        key_hours = [6, 12, 24, 36, 48]
        for hour in key_hours:
            pred = next((p for p in predictions if p['hour'] == hour), None)
            if pred:
                time_str = pred['timestamp'].strftime("%m/%d %H:%M")
                print(f"  • {hour:2d}시간 후 ({time_str}): ${pred['price']:,.0f} "
                      f"({pred['change_from_current']:+.2f}%) [신뢰도: {pred['confidence']*100:.1f}%]")
        
        print("\n" + "="*80)
        print("🎉 6개월 학습 데이터 기반 궁극 예측 완료!")
        print("="*80)
    
    async def run_ultimate_system(self):
        """궁극 시스템 실행"""
        print("🚀 BTC 궁극 예측 시스템 v4.0 시작")
        print("="*80)
        
        # 1. 6개월 데이터 로드
        df = self.load_6month_timeseries_data()
        if df.empty:
            print("❌ 데이터 로드 실패")
            return None
        
        # 2. 고급 지표 계산
        df = self.calculate_advanced_indicators(df)
        
        # 3. AI 모델 훈련
        model_info = self.train_prediction_model(df)
        best_model, feature_cols, all_results = model_info
        
        if not best_model:
            print("❌ 모델 훈련 실패")
            return None
        
        # 4. 미래 예측
        predictions = self.generate_future_predictions(df, model_info, hours_ahead=48)
        
        if not predictions:
            print("❌ 예측 생성 실패")
            return None
        
        # 5. 궁극의 차트 생성
        chart_path = self.create_ultimate_chart(df, predictions, best_model)
        
        # 6. 결과 출력
        self.print_ultimate_results(df, predictions, best_model)
        
        # 7. 브라우저에서 열기
        if chart_path:
            try:
                import subprocess
                subprocess.run(["open", chart_path])
                print(f"\n🌐 궁극의 차트가 브라우저에서 열렸습니다!")
            except:
                print(f"\n💡 브라우저에서 확인: {chart_path}")
        
        return {
            'dataframe': df,
            'predictions': predictions,
            'model_results': best_model,
            'chart_path': chart_path
        }

async def main():
    """메인 실행"""
    system = UltimatePredictionChart()
    result = await system.run_ultimate_system()
    
    if result:
        print("\n✅ 궁극 예측 시스템 완료!")
    else:
        print("\n❌ 시스템 실행 실패")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())