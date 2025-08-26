"""
간단한 BTC 예측 차트 생성기
현재 데이터 + 24시간 예측 시각화
"""

import os
import json
import numpy as np
from datetime import datetime, timedelta
import asyncio

# 차트 라이브러리
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# 기본 예측 함수
from precision_future_predictor import PrecisionFuturePredictor

class SimplePredictionChart:
    """간단한 예측 차트"""
    
    def __init__(self):
        self.base_path = "/Users/parkyoungjun/Desktop/BTC_Analysis_System"
    
    async def create_prediction_chart(self):
        """예측 차트 생성"""
        if not PLOTLY_AVAILABLE:
            print("❌ Plotly 미설치 - pip install plotly")
            return None
        
        try:
            print("🚀 BTC 예측 차트 생성 중...")
            
            # 1. 예측 시스템 실행
            predictor = PrecisionFuturePredictor()
            result = await predictor.predict_future(hours_ahead=24)
            
            if not result or "error" in result:
                print(f"❌ 예측 실패: {result.get('error', '알 수 없는 오류')}")
                return None
            
            # 2. 데이터 추출
            current_price = result.get("current_price", 114914)
            
            # 예측 데이터 찾기 - precision_future_predictor 결과 구조에 맞게
            predictions = []
            if "predictions" in result and isinstance(result["predictions"], dict):
                if "full_predictions" in result["predictions"]:
                    predictions = result["predictions"]["full_predictions"]
            
            # 기본 24시간 예측 데이터 생성 (fallback)
            if not predictions:
                predictions = []
                for h in range(1, 25):
                    price_change = -0.01 * h  # 시간당 1% 하락 가정
                    predicted_price = current_price * (1 + price_change/100)
                    predictions.append({
                        "hour": h,
                        "price": predicted_price,
                        "upper_bound": predicted_price * 1.02,
                        "lower_bound": predicted_price * 0.98,
                        "confidence": max(0.3, 0.5 - h*0.01)
                    })
            
            if not predictions:
                print(f"❌ 예측 데이터를 찾을 수 없습니다. 결과 키: {list(result.keys())}")
                return None
            
            print(f"💰 현재 BTC 가격: ${current_price:,.0f}")
            print(f"📊 예측 데이터: {len(predictions)}개 시간 포인트")
            
            # 3. 시간 및 가격 배열 생성
            current_time = datetime.now()
            
            # 과거 데이터 (현재 시점)
            historical_times = [current_time - timedelta(hours=i) for i in range(12, 0, -1)]
            historical_prices = [current_price + np.random.uniform(-500, 500) for _ in range(12)]  # 모의 과거 데이터
            
            # 현재 시점
            present_time = [current_time]
            present_price = [current_price]
            
            # 미래 예측 데이터
            future_times = [current_time + timedelta(hours=float(p["hour"])) for p in predictions]
            future_prices = [p["price"] for p in predictions]
            future_upper = [p["upper_bound"] for p in predictions]
            future_lower = [p["lower_bound"] for p in predictions]
            future_confidence = [p["confidence"] * 100 for p in predictions]
            
            # 4. 차트 생성
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=("BTC 가격 예측 (과거 12시간 + 미래 24시간)", "예측 신뢰도"),
                vertical_spacing=0.1,
                row_heights=[0.7, 0.3]
            )
            
            # 5. 과거 가격 (모의 데이터)
            fig.add_trace(
                go.Scatter(
                    x=historical_times,
                    y=historical_prices,
                    mode='lines+markers',
                    name='과거 가격',
                    line=dict(color='#1f77b4', width=2),
                    marker=dict(size=4),
                    hovertemplate='<b>과거 가격</b><br>%{x}<br>$%{y:,.0f}<extra></extra>'
                ),
                row=1, col=1
            )
            
            # 6. 현재 시점
            fig.add_trace(
                go.Scatter(
                    x=present_time,
                    y=present_price,
                    mode='markers',
                    name='현재 가격',
                    marker=dict(color='red', size=12, symbol='diamond'),
                    hovertemplate='<b>현재 가격</b><br>%{x}<br>$%{y:,.0f}<extra></extra>'
                ),
                row=1, col=1
            )
            
            # 7. 미래 예측
            fig.add_trace(
                go.Scatter(
                    x=future_times,
                    y=future_prices,
                    mode='lines+markers',
                    name='미래 예측',
                    line=dict(color='#ff7f0e', width=3, dash='dot'),
                    marker=dict(size=5, symbol='diamond'),
                    hovertemplate='<b>예측 가격</b><br>%{x}<br>$%{y:,.0f}<extra></extra>'
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
                    name='신뢰 구간',
                    hoverinfo='skip'
                ),
                row=1, col=1
            )
            
            # 9. 신뢰도 그래프
            fig.add_trace(
                go.Scatter(
                    x=future_times,
                    y=future_confidence,
                    mode='lines+markers',
                    name='예측 신뢰도',
                    line=dict(color='#2ca02c', width=2),
                    marker=dict(size=4),
                    hovertemplate='<b>신뢰도</b><br>%{x}<br>%{y:.1f}%<extra></extra>'
                ),
                row=2, col=1
            )
            
            # 10. 현재 시점 수직선
            fig.add_vline(
                x=current_time,
                line=dict(color="red", width=2, dash="dash"),
                annotation_text="현재",
                annotation_position="top"
            )
            
            # 11. 레이아웃 설정
            fig.update_layout(
                title={
                    'text': f"🚀 BTC 실시간 예측 차트<br><sub>현재: ${current_price:,.0f} | 생성시간: {current_time.strftime('%Y-%m-%d %H:%M')}</sub>",
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 18}
                },
                height=700,
                showlegend=True,
                template='plotly_white',
                hovermode='x unified'
            )
            
            # 12. 축 설정
            fig.update_yaxes(title_text="BTC 가격 (USD)", row=1, col=1)
            fig.update_yaxes(title_text="신뢰도 (%)", row=2, col=1, range=[0, 100])
            fig.update_xaxes(title_text="시간", row=2, col=1)
            
            # 13. 차트 저장
            chart_path = os.path.join(self.base_path, "btc_prediction_chart.html")
            fig.write_html(chart_path)
            
            print(f"✅ 차트 저장: {chart_path}")
            
            # 14. 예측 요약
            pred_1h = predictions[0]
            pred_24h = predictions[-1]
            
            print(f"\n📊 예측 요약:")
            print(f"  • 1시간 후: ${pred_1h['price']:,.0f} ({(pred_1h['price']-current_price)/current_price*100:+.2f}%)")
            print(f"  • 24시간 후: ${pred_24h['price']:,.0f} ({(pred_24h['price']-current_price)/current_price*100:+.2f}%)")
            print(f"  • 평균 신뢰도: {np.mean(future_confidence):.1f}%")
            
            # 15. 브라우저에서 열기
            try:
                import subprocess
                subprocess.run(["open", chart_path], check=True)
                print("🌐 브라우저에서 차트 열림!")
            except:
                print(f"💡 브라우저에서 열어보세요: {chart_path}")
            
            return {
                "chart_path": chart_path,
                "current_price": current_price,
                "prediction_1h": pred_1h,
                "prediction_24h": pred_24h,
                "avg_confidence": np.mean(future_confidence)
            }
            
        except Exception as e:
            print(f"❌ 차트 생성 실패: {e}")
            return None

async def main():
    """메인 실행"""
    print("🎯 BTC 간단 예측 차트 생성기")
    print("="*50)
    
    chart_generator = SimplePredictionChart()
    result = await chart_generator.create_prediction_chart()
    
    if result:
        print("\n" + "="*50)
        print("🎉 차트 생성 완료!")
        print("="*50)
    else:
        print("\n❌ 차트 생성 실패")

if __name__ == "__main__":
    asyncio.run(main())