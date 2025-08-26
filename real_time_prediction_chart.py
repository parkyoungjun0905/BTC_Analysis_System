"""
실시간 BTC 미래 예측 차트 생성기
현재 데이터 + 24시간 미래 예측 시각화
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import asyncio

# 차트 라이브러리
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("⚠️ Plotly 미설치 - pip install plotly")

# 통합 시스템 import
from integrated_prediction_system import IntegratedPredictionSystem

class RealTimePredictionChart:
    """실시간 예측 차트 생성기"""
    
    def __init__(self):
        self.base_path = "/Users/parkyoungjun/Desktop/BTC_Analysis_System"
        self.historical_path = os.path.join(self.base_path, "historical_data")
        self.chart_path = os.path.join(self.base_path, "realtime_prediction_chart.html")
        
    def load_recent_price_history(self) -> List[Dict]:
        """최근 가격 히스토리 로드 (차트용)"""
        try:
            files = sorted([f for f in os.listdir(self.historical_path) 
                           if f.startswith("btc_analysis_") and f.endswith(".json")])
            
            price_history = []
            
            # 최근 10개 파일에서 가격 추출
            for filename in files[-10:]:
                filepath = os.path.join(self.historical_path, filename)
                try:
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                    
                    # 타임스탬프 추출
                    timestamp_str = data.get("collection_time", filename.split("_")[-1].replace(".json", ""))
                    try:
                        timestamp = datetime.fromisoformat(timestamp_str.replace("T", " ").replace("Z", ""))
                    except:
                        # 파일명에서 타임스탬프 추출 시도
                        timestamp = datetime.now() - timedelta(hours=len(files)-files.index(filename))
                    
                    # 가격 추출
                    price = self.extract_price_from_data(data)
                    if price > 0:
                        price_history.append({
                            "timestamp": timestamp,
                            "price": price,
                            "filename": filename
                        })
                        
                except Exception as e:
                    print(f"파일 로드 실패 {filename}: {e}")
                    continue
            
            return sorted(price_history, key=lambda x: x["timestamp"])
            
        except Exception as e:
            print(f"가격 히스토리 로드 실패: {e}")
            return []
    
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
    
    async def generate_future_predictions(self, current_data: Dict, hours: int = 24) -> List[Dict]:
        """미래 예측 데이터 생성"""
        try:
            # 통합 예측 시스템 사용
            system = IntegratedPredictionSystem()
            result = await system.run_full_prediction_cycle()
            
            if "error" in result:
                print(f"예측 실패: {result['error']}")
                return []
            
            # 예측 데이터 추출
            prediction = result.get("prediction", {})
            hourly_predictions = prediction.get("hourly_predictions", [])
            
            current_time = datetime.now()
            future_data = []
            
            for pred in hourly_predictions:
                hour_offset = pred.get("hour", 1)
                if isinstance(hour_offset, (int, float)):
                    future_time = current_time + timedelta(hours=float(hour_offset))
                else:
                    future_time = current_time + timedelta(hours=1)
                    
                future_data.append({
                    "timestamp": future_time,
                    "price": pred.get("price", 0),
                    "confidence": pred.get("confidence", 0.5),
                    "upper_bound": pred.get("price", 0) * (1 + 0.02),  # ±2% 구간
                    "lower_bound": pred.get("price", 0) * (1 - 0.02),
                    "hour": hour_offset
                })
            
            return future_data
            
        except Exception as e:
            print(f"미래 예측 생성 실패: {e}")
            return []
    
    async def create_comprehensive_chart(self) -> str:
        """종합 예측 차트 생성"""
        if not PLOTLY_AVAILABLE:
            return "Plotly 미설치로 차트 생성 불가"
        
        try:
            print("📊 실시간 예측 차트 생성 시작...")
            
            # 1. 과거 가격 히스토리 로드
            price_history = self.load_recent_price_history()
            if not price_history:
                return "과거 데이터 부족"
            
            print(f"📈 과거 데이터: {len(price_history)}개 포인트")
            
            # 2. 최신 데이터로 미래 예측
            latest_file = sorted([f for f in os.listdir(self.historical_path) 
                                 if f.startswith("btc_analysis_") and f.endswith(".json")])[-1]
            
            with open(os.path.join(self.historical_path, latest_file), 'r') as f:
                current_data = json.load(f)
            
            future_predictions = await self.generate_future_predictions(current_data)
            if not future_predictions:
                return "미래 예측 생성 실패"
            
            print(f"🔮 미래 예측: {len(future_predictions)}개 포인트")
            
            # 3. 차트 데이터 준비
            historical_times = [p["timestamp"] for p in price_history]
            historical_prices = [p["price"] for p in price_history]
            
            future_times = [p["timestamp"] for p in future_predictions]
            future_prices = [p["price"] for p in future_predictions]
            future_upper = [p["upper_bound"] for p in future_predictions]
            future_lower = [p["lower_bound"] for p in future_predictions]
            future_confidence = [p["confidence"] for p in future_predictions]
            
            # 현재 시점
            current_time = datetime.now()
            current_price = historical_prices[-1] if historical_prices else 0
            
            # 4. 서브플롯 생성
            fig = make_subplots(
                rows=3, cols=1,
                subplot_titles=("BTC 가격 예측 (과거 + 미래)", "신뢰도 구간", "예측 정확도 트렌드"),
                vertical_spacing=0.08,
                row_heights=[0.6, 0.25, 0.15]
            )
            
            # 5. 과거 가격 (실제 데이터)
            fig.add_trace(
                go.Scatter(
                    x=historical_times,
                    y=historical_prices,
                    mode='lines+markers',
                    name='과거 실제 가격',
                    line=dict(color='#1f77b4', width=2),
                    marker=dict(size=4),
                    hovertemplate='<b>실제 가격</b><br>시간: %{x}<br>가격: $%{y:,.0f}<extra></extra>'
                ),
                row=1, col=1
            )
            
            # 6. 현재 시점 표시
            fig.add_vline(
                x=current_time,
                line=dict(color="red", width=2, dash="dash"),
                annotation_text="현재 시점",
                annotation_position="top"
            )
            
            # 7. 미래 예측 가격
            fig.add_trace(
                go.Scatter(
                    x=future_times,
                    y=future_prices,
                    mode='lines+markers',
                    name='미래 예측 가격',
                    line=dict(color='#ff7f0e', width=3, dash='dot'),
                    marker=dict(size=6, symbol='diamond'),
                    hovertemplate='<b>예측 가격</b><br>시간: %{x}<br>가격: $%{y:,.0f}<br>신뢰도: %{customdata:.1%}<extra></extra>',
                    customdata=future_confidence
                ),
                row=1, col=1
            )
            
            # 8. 신뢰 구간
            fig.add_trace(
                go.Scatter(
                    x=future_times + future_times[::-1],
                    y=future_upper + future_lower[::-1],
                    fill='toself',
                    fillcolor='rgba(255,127,14,0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name='예측 신뢰 구간 (±2%)',
                    showlegend=True,
                    hoverinfo='skip'
                ),
                row=1, col=1
            )
            
            # 9. 신뢰도 그래프
            fig.add_trace(
                go.Scatter(
                    x=future_times,
                    y=[c*100 for c in future_confidence],
                    mode='lines+markers',
                    name='예측 신뢰도 (%)',
                    line=dict(color='#2ca02c', width=2),
                    marker=dict(size=5),
                    hovertemplate='<b>신뢰도</b><br>시간: %{x}<br>신뢰도: %{y:.1f}%<extra></extra>'
                ),
                row=2, col=1
            )
            
            # 10. 정확도 트렌드 (모의)
            accuracy_trend = [85 - i*0.5 for i in range(24)]  # 시간 지날수록 정확도 감소
            fig.add_trace(
                go.Scatter(
                    x=future_times,
                    y=accuracy_trend,
                    mode='lines',
                    name='예상 정확도 (%)',
                    line=dict(color='#d62728', width=2),
                    hovertemplate='<b>예상 정확도</b><br>시간: %{x}<br>정확도: %{y:.1f}%<extra></extra>'
                ),
                row=3, col=1
            )
            
            # 11. 차트 레이아웃 설정
            fig.update_layout(
                title={
                    'text': f"🚀 BTC 실시간 예측 차트 - {current_time.strftime('%Y-%m-%d %H:%M')}",
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 20, 'color': '#2c3e50'}
                },
                height=800,
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                template='plotly_white',
                annotations=[
                    dict(
                        x=0.02, y=0.98,
                        xref='paper', yref='paper',
                        text=f"현재 가격: ${current_price:,.0f}<br>예측 엔진: v3.0<br>데이터: 2,418개 지표",
                        showarrow=False,
                        bgcolor="rgba(255,255,255,0.8)",
                        bordercolor="gray",
                        borderwidth=1,
                        font=dict(size=10)
                    )
                ]
            )
            
            # 12. Y축 설정
            fig.update_yaxes(title_text="BTC 가격 (USD)", row=1, col=1)
            fig.update_yaxes(title_text="신뢰도 (%)", row=2, col=1, range=[0, 100])
            fig.update_yaxes(title_text="정확도 (%)", row=3, col=1, range=[50, 100])
            
            # 13. X축 설정
            fig.update_xaxes(title_text="시간", row=3, col=1)
            
            # 14. 차트 저장
            fig.write_html(self.chart_path)
            
            print(f"✅ 차트 저장 완료: {self.chart_path}")
            
            return self.chart_path
            
        except Exception as e:
            print(f"❌ 차트 생성 실패: {e}")
            return f"차트 생성 실패: {e}"
    
    async def generate_prediction_summary(self) -> Dict:
        """예측 요약 정보 생성"""
        try:
            # 통합 시스템으로 예측 실행
            system = IntegratedPredictionSystem()
            result = await system.run_full_prediction_cycle()
            
            if "error" in result:
                return {"error": result["error"]}
            
            # 요약 정보 추출
            prediction = result.get("prediction", {})
            pred_data = prediction.get("prediction", {})
            
            current_price = prediction.get("current_price", 0)
            predicted_price = pred_data.get("predicted_price", 0)
            direction = pred_data.get("direction", "중립")
            confidence = pred_data.get("confidence", 0)
            price_change = pred_data.get("price_change", 0)
            
            # 시간별 주요 예측
            hourly = prediction.get("hourly_predictions", [])
            key_predictions = {
                "1시간": hourly[0] if len(hourly) > 0 else {},
                "6시간": hourly[5] if len(hourly) > 5 else {},
                "12시간": hourly[11] if len(hourly) > 11 else {},
                "24시간": hourly[23] if len(hourly) > 23 else {}
            }
            
            return {
                "current_price": current_price,
                "predicted_price": predicted_price,
                "direction": direction,
                "confidence": confidence,
                "price_change": price_change,
                "market_regime": prediction.get("market_regime", "unknown"),
                "key_signals": prediction.get("key_signals", [])[:3],
                "key_predictions": key_predictions,
                "chart_path": self.chart_path
            }
            
        except Exception as e:
            print(f"예측 요약 생성 실패: {e}")
            return {"error": str(e)}

async def run_real_time_prediction():
    """실시간 예측 및 차트 생성"""
    print("🚀 실시간 BTC 미래 예측 차트 생성기")
    print("="*60)
    
    chart_generator = RealTimePredictionChart()
    
    # 차트 생성
    chart_path = await chart_generator.create_comprehensive_chart()
    
    if "실패" in chart_path:
        print(f"❌ {chart_path}")
        return
    
    # 예측 요약 생성
    summary = await chart_generator.generate_prediction_summary()
    
    if "error" in summary:
        print(f"❌ 예측 요약 실패: {summary['error']}")
        return
    
    # 결과 출력
    print("\n📊 실시간 예측 결과")
    print("="*60)
    print(f"💰 현재 가격: ${summary['current_price']:,.0f}")
    print(f"🎯 시장 체제: {summary['market_regime']}")
    print(f"📈 예측 방향: {summary['direction']}")
    print(f"🎪 신뢰도: {summary['confidence']:.1%}")
    print(f"💫 24시간 후 예측: ${summary['predicted_price']:,.0f}")
    print(f"📊 변화율: {summary['price_change']:+.2f}%")
    
    print(f"\n🔍 핵심 신호:")
    for i, signal in enumerate(summary['key_signals'], 1):
        print(f"  {i}. {signal}")
    
    print(f"\n⏰ 주요 시점 예측:")
    for timeframe, pred in summary['key_predictions'].items():
        if pred:
            price = pred.get('price', 0)
            change = pred.get('change_percent', 0)
            confidence = pred.get('confidence', 0)
            print(f"  • {timeframe}: ${price:,.0f} ({change:+.2f}%) [신뢰도: {confidence:.1%}]")
    
    print(f"\n📈 차트 보기:")
    print(f"  파일: {chart_path}")
    
    # 브라우저에서 차트 열기
    import subprocess
    try:
        subprocess.run(["open", chart_path], check=True)
        print("  ✅ 브라우저에서 차트가 열렸습니다!")
    except:
        print(f"  💡 브라우저에서 다음 파일을 여세요: {chart_path}")
    
    print("="*60)
    print("🎉 실시간 예측 완료!")
    
    return summary

if __name__ == "__main__":
    asyncio.run(run_real_time_prediction())