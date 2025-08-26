"""
🎯 BTC 예측 백테스팅 검증 시스템
- 6개월 과거 데이터로 시간여행 시뮬레이션
- 과거 시점에서 예측 → 실제 결과와 비교
- 실제 정확도 측정 및 학습
- 검증된 모델로만 미래 예측
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import Ridge, LinearRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_absolute_error, r2_score
    import ta
except ImportError:
    print("❌ 필수 라이브러리 미설치")
    exit()

class BacktestingValidationSystem:
    """백테스팅 검증 시스템"""
    
    def __init__(self):
        self.base_path = "/Users/parkyoungjun/Desktop/BTC_Analysis_System"
        self.timeseries_path = os.path.join(self.base_path, "timeseries_data")
        
        # 백테스트 결과 저장
        self.backtest_results = []
        self.validated_accuracy = {}
        self.best_models = {}
        
    def load_6month_data(self) -> pd.DataFrame:
        """6개월 전체 데이터 로드"""
        try:
            print("📊 6개월 전체 데이터 로드 중...")
            
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
                
                print(f"✅ 전체 데이터: {len(master_df)}개 포인트")
                print(f"📅 기간: {master_df['timestamp'].min()} ~ {master_df['timestamp'].max()}")
                return master_df
            
            return pd.DataFrame()
            
        except Exception as e:
            print(f"❌ 데이터 로드 실패: {e}")
            return pd.DataFrame()
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """기술적 지표 계산"""
        try:
            # 기본 지표들
            df['sma_7'] = df['price'].rolling(7).mean()
            df['sma_14'] = df['price'].rolling(14).mean()
            df['sma_30'] = df['price'].rolling(30).mean()
            df['ema_12'] = df['price'].ewm(span=12).mean()
            df['ema_26'] = df['price'].ewm(span=26).mean()
            
            # RSI
            df['rsi_14'] = ta.momentum.rsi(df['price'], window=14)
            
            # MACD
            macd = ta.trend.MACD(df['price'])
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            df['macd_hist'] = macd.macd_diff()
            
            # 볼린저 밴드
            bb = ta.volatility.BollingerBands(df['price'], window=20)
            df['bb_upper'] = bb.bollinger_hband()
            df['bb_lower'] = bb.bollinger_lband()
            df['bb_width'] = bb.bollinger_wband()
            
            # 변화율
            df['price_change_1d'] = df['price'].pct_change()
            df['price_change_7d'] = df['price'].pct_change(7)
            df['volatility'] = df['price'].rolling(14).std()
            
            # 거래량 지표
            if 'volume' in df.columns:
                df['volume_sma'] = df['volume'].rolling(20).mean()
                df['volume_ratio'] = df['volume'] / df['volume_sma']
            
            return df
            
        except Exception as e:
            print(f"❌ 지표 계산 실패: {e}")
            return df
    
    def run_historical_simulation(self, df: pd.DataFrame) -> List[Dict]:
        """과거 시점 시뮬레이션"""
        try:
            print("🕐 과거 시점 시뮬레이션 시작...")
            print("="*60)
            
            # 지표 계산
            df = self.calculate_indicators(df)
            
            # 피처 선택
            feature_cols = []
            for col in df.columns:
                if col not in ['timestamp', 'price'] and pd.api.types.is_numeric_dtype(df[col]):
                    if df[col].notna().sum() > len(df) * 0.7:
                        feature_cols.append(col)
            
            print(f"📊 사용 피처: {len(feature_cols)}개")
            
            # 시뮬레이션 시점들 (과거 여러 시점에서 예측)
            simulation_points = []
            total_days = len(df)
            
            # 30일마다 시뮬레이션 (충분한 학습 데이터 확보)
            for i in range(60, total_days - 30, 30):  # 60일 이후부터, 30일 간격
                simulation_points.append(i)
            
            print(f"🎯 시뮬레이션 시점: {len(simulation_points)}개")
            
            backtest_results = []
            
            for sim_idx, current_idx in enumerate(simulation_points):
                current_date = df.iloc[current_idx]['timestamp']
                print(f"\n📅 시뮬레이션 {sim_idx+1}/{len(simulation_points)}: {current_date.strftime('%Y-%m-%d')}")
                
                # 현재 시점까지의 데이터만 사용 (미래 데이터 사용 금지!)
                train_df = df.iloc[:current_idx].copy()
                
                if len(train_df) < 50:
                    continue
                
                # 예측 대상 기간 (1일, 7일, 14일 후)
                prediction_periods = [1, 7, 14]
                
                for pred_days in prediction_periods:
                    future_idx = current_idx + pred_days
                    
                    if future_idx >= len(df):
                        continue
                    
                    # 실제 미래 가격 (정답)
                    actual_future_price = df.iloc[future_idx]['price']
                    actual_current_price = df.iloc[current_idx]['price']
                    actual_change = ((actual_future_price - actual_current_price) / actual_current_price) * 100
                    
                    # 과거 데이터로만 모델 훈련
                    result = self.train_and_predict_historical(train_df, feature_cols, pred_days)
                    
                    if result:
                        predicted_price = result['prediction']
                        predicted_change = ((predicted_price - actual_current_price) / actual_current_price) * 100
                        
                        # 예측 정확도 계산
                        price_error = abs(predicted_price - actual_future_price)
                        price_mape = abs(price_error / actual_future_price) * 100
                        
                        # 방향 정확도
                        direction_correct = (predicted_change > 0 and actual_change > 0) or \
                                          (predicted_change < 0 and actual_change < 0)
                        
                        backtest_result = {
                            'simulation_date': current_date,
                            'prediction_period': f'{pred_days}d',
                            'current_price': actual_current_price,
                            'predicted_price': predicted_price,
                            'actual_price': actual_future_price,
                            'predicted_change': predicted_change,
                            'actual_change': actual_change,
                            'price_error': price_error,
                            'price_mape': price_mape,
                            'direction_correct': direction_correct,
                            'model_name': result['model_name'],
                            'train_data_size': len(train_df)
                        }
                        
                        backtest_results.append(backtest_result)
                        
                        print(f"    {pred_days}일 후: 예측 ${predicted_price:,.0f} vs 실제 ${actual_future_price:,.0f} "
                              f"(오차: {price_mape:.1f}%, 방향: {'✅' if direction_correct else '❌'})")
            
            print(f"\n✅ 백테스팅 완료: {len(backtest_results)}개 시뮬레이션")
            return backtest_results
            
        except Exception as e:
            print(f"❌ 시뮬레이션 실패: {e}")
            return []
    
    def train_and_predict_historical(self, train_df: pd.DataFrame, feature_cols: List[str], pred_days: int) -> Dict:
        """과거 데이터로만 훈련하고 예측"""
        try:
            # 훈련 데이터 준비
            df_clean = train_df[['price'] + feature_cols].dropna()
            
            if len(df_clean) < 30:
                return None
            
            # X: 피처, y: pred_days일 후 가격
            if len(df_clean) <= pred_days:
                return None
            
            X = df_clean[feature_cols].iloc[:-pred_days].values
            y = df_clean['price'].iloc[pred_days:].values
            
            # 훈련/검증 분할
            split_idx = int(len(X) * 0.8)
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            if len(X_train) < 10:
                return None
            
            # 여러 모델 테스트
            models = {
                'RandomForest': RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42),
                'GradientBoosting': GradientBoostingRegressor(n_estimators=50, random_state=42),
                'Ridge': Ridge(alpha=1.0),
                'Linear': LinearRegression()
            }
            
            best_model = None
            best_score = float('inf')
            best_name = ""
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            
            for name, model in models.items():
                try:
                    if name in ['Ridge', 'Linear']:
                        model.fit(X_train_scaled, y_train)
                        val_pred = model.predict(X_val_scaled)
                    else:
                        model.fit(X_train, y_train)
                        val_pred = model.predict(X_val)
                    
                    val_mae = mean_absolute_error(y_val, val_pred)
                    
                    if val_mae < best_score:
                        best_score = val_mae
                        best_model = model
                        best_name = name
                
                except Exception:
                    continue
            
            if best_model is None:
                return None
            
            # 최신 데이터로 예측
            latest_features = df_clean[feature_cols].iloc[-1:].values
            
            if best_name in ['Ridge', 'Linear']:
                latest_scaled = scaler.transform(latest_features)
                prediction = best_model.predict(latest_scaled)[0]
            else:
                prediction = best_model.predict(latest_features)[0]
            
            return {
                'prediction': prediction,
                'model_name': best_name,
                'validation_mae': best_score
            }
            
        except Exception as e:
            return None
    
    def analyze_backtest_results(self, results: List[Dict]) -> Dict:
        """백테스트 결과 분석"""
        try:
            print("\n📊 백테스트 결과 분석")
            print("="*60)
            
            if not results:
                print("❌ 분석할 결과 없음")
                return {}
            
            # 기간별 분석
            periods = ['1d', '7d', '14d']
            analysis = {}
            
            for period in periods:
                period_results = [r for r in results if r['prediction_period'] == period]
                
                if not period_results:
                    continue
                
                # 정확도 지표 계산
                direction_accuracy = np.mean([r['direction_correct'] for r in period_results])
                avg_mape = np.mean([r['price_mape'] for r in period_results])
                median_mape = np.median([r['price_mape'] for r in period_results])
                
                # 예측 vs 실제 상관관계
                predicted_changes = [r['predicted_change'] for r in period_results]
                actual_changes = [r['actual_change'] for r in period_results]
                correlation = np.corrcoef(predicted_changes, actual_changes)[0,1] if len(predicted_changes) > 1 else 0
                
                analysis[period] = {
                    'samples': len(period_results),
                    'direction_accuracy': direction_accuracy,
                    'avg_mape': avg_mape,
                    'median_mape': median_mape,
                    'correlation': correlation,
                    'results': period_results
                }
                
                print(f"\n🎯 {period} 예측 결과:")
                print(f"  • 시뮬레이션 횟수: {len(period_results)}회")
                print(f"  • 방향 정확도: {direction_accuracy:.1%}")
                print(f"  • 평균 가격 오차: {avg_mape:.2f}%")
                print(f"  • 중간값 오차: {median_mape:.2f}%")
                print(f"  • 예측-실제 상관관계: {correlation:.3f}")
            
            # 전체 통계
            all_direction = np.mean([r['direction_correct'] for r in results])
            all_mape = np.mean([r['price_mape'] for r in results])
            
            print(f"\n📈 전체 성능:")
            print(f"  • 전체 방향 정확도: {all_direction:.1%}")
            print(f"  • 전체 평균 가격 오차: {all_mape:.2f}%")
            
            # 시간별 성능 변화
            results_by_time = sorted(results, key=lambda x: x['simulation_date'])
            if len(results_by_time) > 10:
                recent_results = results_by_time[-len(results_by_time)//2:]  # 최근 절반
                old_results = results_by_time[:len(results_by_time)//2]     # 과거 절반
                
                recent_accuracy = np.mean([r['direction_correct'] for r in recent_results])
                old_accuracy = np.mean([r['direction_correct'] for r in old_results])
                
                print(f"\n⏰ 시간별 성능 변화:")
                print(f"  • 과거 절반 정확도: {old_accuracy:.1%}")
                print(f"  • 최근 절반 정확도: {recent_accuracy:.1%}")
                print(f"  • 성능 변화: {recent_accuracy - old_accuracy:+.1%}")
            
            self.validated_accuracy = analysis
            return analysis
            
        except Exception as e:
            print(f"❌ 결과 분석 실패: {e}")
            return {}
    
    def save_backtest_results(self, results: List[Dict], analysis: Dict):
        """백테스트 결과 저장"""
        try:
            # 상세 결과
            detailed_results = {
                'generation_time': datetime.now().isoformat(),
                'total_simulations': len(results),
                'backtest_results': results,
                'performance_analysis': analysis
            }
            
            results_path = os.path.join(self.base_path, "backtest_results.json")
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(detailed_results, f, indent=2, ensure_ascii=False, default=str)
            
            print(f"✅ 백테스트 결과 저장: {results_path}")
            
            # 요약 리포트
            summary = {
                'validated_at': datetime.now().isoformat(),
                'real_world_accuracy': {
                    '1d': analysis.get('1d', {}).get('direction_accuracy', 0),
                    '7d': analysis.get('7d', {}).get('direction_accuracy', 0), 
                    '14d': analysis.get('14d', {}).get('direction_accuracy', 0)
                },
                'price_error_rates': {
                    '1d': analysis.get('1d', {}).get('avg_mape', 0),
                    '7d': analysis.get('7d', {}).get('avg_mape', 0),
                    '14d': analysis.get('14d', {}).get('avg_mape', 0)
                },
                'reliability_score': np.mean([
                    analysis.get('1d', {}).get('direction_accuracy', 0),
                    analysis.get('7d', {}).get('direction_accuracy', 0),
                    analysis.get('14d', {}).get('direction_accuracy', 0)
                ])
            }
            
            summary_path = os.path.join(self.base_path, "validated_accuracy.json")
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
            
            print(f"✅ 검증된 정확도 저장: {summary_path}")
            
        except Exception as e:
            print(f"❌ 결과 저장 실패: {e}")
    
    def create_backtest_visualization(self, results: List[Dict]) -> str:
        """백테스트 시각화"""
        try:
            print("📊 백테스트 결과 시각화 중...")
            
            if not results:
                return ""
            
            # 데이터 준비
            df_results = pd.DataFrame(results)
            df_results['simulation_date'] = pd.to_datetime(df_results['simulation_date'])
            
            fig = make_subplots(
                rows=3, cols=2,
                subplot_titles=(
                    "방향 정확도 (기간별)",
                    "가격 오차 (MAPE)",
                    "예측 vs 실제 가격 (1일)",
                    "예측 vs 실제 가격 (7일)", 
                    "시간별 성능 변화",
                    "모델별 성능"
                ),
                specs=[[{"type": "bar"}, {"type": "box"}],
                       [{"type": "scatter"}, {"type": "scatter"}],
                       [{"type": "scatter"}, {"type": "bar"}]]
            )
            
            # 1. 기간별 방향 정확도
            periods = ['1d', '7d', '14d']
            accuracies = []
            for period in periods:
                period_data = df_results[df_results['prediction_period'] == period]
                if len(period_data) > 0:
                    accuracy = period_data['direction_correct'].mean()
                    accuracies.append(accuracy)
                else:
                    accuracies.append(0)
            
            fig.add_trace(
                go.Bar(x=periods, y=accuracies, name='방향 정확도',
                      marker_color=['green' if x > 0.5 else 'red' for x in accuracies]),
                row=1, col=1
            )
            fig.add_hline(y=0.5, line_dash="dash", line_color="gray", row=1, col=1)
            
            # 2. 가격 오차 분포
            for period in periods:
                period_data = df_results[df_results['prediction_period'] == period]
                if len(period_data) > 0:
                    fig.add_trace(
                        go.Box(y=period_data['price_mape'], name=f'{period} MAPE'),
                        row=1, col=2
                    )
            
            # 3. 예측 vs 실제 (1일)
            day1_data = df_results[df_results['prediction_period'] == '1d']
            if len(day1_data) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=day1_data['actual_price'],
                        y=day1_data['predicted_price'],
                        mode='markers',
                        name='1일 예측',
                        marker=dict(color=['green' if x else 'red' for x in day1_data['direction_correct']])
                    ),
                    row=2, col=1
                )
                # 완벽한 예측선
                min_price = min(day1_data['actual_price'].min(), day1_data['predicted_price'].min())
                max_price = max(day1_data['actual_price'].max(), day1_data['predicted_price'].max())
                fig.add_trace(
                    go.Scatter(x=[min_price, max_price], y=[min_price, max_price],
                              mode='lines', line=dict(dash='dash', color='gray'),
                              name='완벽한 예측', showlegend=False),
                    row=2, col=1
                )
            
            # 4. 예측 vs 실제 (7일)
            day7_data = df_results[df_results['prediction_period'] == '7d']
            if len(day7_data) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=day7_data['actual_price'],
                        y=day7_data['predicted_price'],
                        mode='markers',
                        name='7일 예측',
                        marker=dict(color=['green' if x else 'red' for x in day7_data['direction_correct']])
                    ),
                    row=2, col=2
                )
            
            # 5. 시간별 성능 변화
            df_results_sorted = df_results.sort_values('simulation_date')
            df_results_sorted['rolling_accuracy'] = df_results_sorted['direction_correct'].rolling(10, min_periods=1).mean()
            
            fig.add_trace(
                go.Scatter(
                    x=df_results_sorted['simulation_date'],
                    y=df_results_sorted['rolling_accuracy'],
                    mode='lines',
                    name='10회 이동평균 정확도'
                ),
                row=3, col=1
            )
            fig.add_hline(y=0.5, line_dash="dash", line_color="gray", row=3, col=1)
            
            # 6. 모델별 성능
            model_performance = df_results.groupby('model_name')['direction_correct'].mean()
            fig.add_trace(
                go.Bar(
                    x=list(model_performance.index),
                    y=list(model_performance.values),
                    name='모델별 정확도'
                ),
                row=3, col=2
            )
            
            # 레이아웃 설정
            fig.update_layout(
                title="🎯 BTC 예측 백테스팅 검증 결과",
                height=1000,
                showlegend=True,
                template='plotly_dark'
            )
            
            # 축 설정
            fig.update_yaxes(title_text="정확도", range=[0, 1], row=1, col=1)
            fig.update_yaxes(title_text="MAPE (%)", row=1, col=2)
            fig.update_xaxes(title_text="실제 가격", row=2, col=1)
            fig.update_yaxes(title_text="예측 가격", row=2, col=1)
            fig.update_xaxes(title_text="실제 가격", row=2, col=2)
            fig.update_yaxes(title_text="예측 가격", row=2, col=2)
            fig.update_xaxes(title_text="시뮬레이션 날짜", row=3, col=1)
            fig.update_yaxes(title_text="정확도", range=[0, 1], row=3, col=1)
            fig.update_yaxes(title_text="정확도", range=[0, 1], row=3, col=2)
            
            # 저장
            chart_path = os.path.join(self.base_path, "backtest_validation_chart.html")
            fig.write_html(chart_path, include_plotlyjs=True)
            
            print(f"✅ 백테스트 차트 저장: {chart_path}")
            return chart_path
            
        except Exception as e:
            print(f"❌ 시각화 실패: {e}")
            return ""

def main():
    """메인 실행"""
    print("🎯 BTC 예측 백테스팅 검증 시스템")
    print("="*80)
    print("6개월 과거 데이터로 시간여행하여 예측 정확도를 실제 검증합니다.")
    print("="*80)
    
    system = BacktestingValidationSystem()
    
    # 1. 데이터 로드
    df = system.load_6month_data()
    if df.empty:
        print("❌ 데이터 로드 실패")
        return
    
    # 2. 과거 시점 시뮬레이션 실행
    backtest_results = system.run_historical_simulation(df)
    if not backtest_results:
        print("❌ 시뮬레이션 실패")
        return
    
    # 3. 결과 분석
    analysis = system.analyze_backtest_results(backtest_results)
    
    # 4. 결과 저장
    system.save_backtest_results(backtest_results, analysis)
    
    # 5. 시각화
    chart_path = system.create_backtest_visualization(backtest_results)
    
    # 6. 최종 결과 출력
    print("\n" + "="*80)
    print("🏆 실제 검증된 예측 정확도")
    print("="*80)
    
    if analysis:
        print("📊 검증 결과 요약:")
        for period, data in analysis.items():
            print(f"  • {period:3s} 예측: {data['direction_accuracy']:.1%} 방향 정확도, "
                  f"{data['avg_mape']:.2f}% 가격 오차 ({data['samples']}회 검증)")
    
    print(f"\n💡 결론:")
    print(f"이것이 6개월 실제 데이터로 검증한 진짜 예측 정확도입니다.")
    print(f"이제 이 검증된 성능을 바탕으로만 미래를 예측해야 합니다.")
    
    # 브라우저 열기
    if chart_path:
        try:
            import subprocess
            subprocess.run(["open", chart_path], check=True)
            print(f"\n🌐 백테스트 결과 차트가 브라우저에서 열렸습니다!")
        except:
            print(f"\n💡 브라우저에서 확인: {chart_path}")
    
    print("\n" + "="*80)
    print("🎉 백테스팅 검증 완료!")
    print("="*80)

if __name__ == "__main__":
    main()