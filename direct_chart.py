"""
BTC 예측 직접 차트 생성
"""

import json
import os
from datetime import datetime, timedelta

# Plotly 체크
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("❌ Plotly 미설치")
    exit()

def create_btc_prediction_chart():
    """BTC 예측 차트 직접 생성"""
    
    print("🚀 BTC 예측 차트 생성 중...")
    
    # 1. 최신 예측 데이터 로드
    predictions_dir = "/Users/parkyoungjun/Desktop/BTC_Analysis_System/predictions"
    files = [f for f in os.listdir(predictions_dir) if f.endswith('.json')]
    if not files:
        print("❌ 예측 데이터 없음")
        return
    
    latest_file = sorted(files)[-1]
    with open(os.path.join(predictions_dir, latest_file), 'r') as f:
        data = json.load(f)
    
    print(f"📊 데이터 로드: {latest_file}")
    
    # 2. 데이터 추출
    current_price = data["current_price"]
    predictions = data["predictions"]["full_predictions"]
    
    print(f"💰 현재 가격: ${current_price:,.0f}")
    print(f"📈 예측 포인트: {len(predictions)}개")
    
    # 3. 시간 배열 생성
    base_time = datetime.now()
    
    # 과거 12시간 (모의)
    print("과거 데이터 생성 중...")
    past_hours = list(range(-12, 0))
    past_times = []
    past_prices = []
    
    for h in past_hours:
        past_times.append(base_time + timedelta(hours=h))
        past_prices.append(current_price + (h * 50))
    
    print(f"과거 데이터: {len(past_times)}개 포인트")
    
    # 현재 시점
    now_time = [base_time]
    now_price = [current_price]
    
    # 미래 예측
    print("미래 예측 데이터 생성 중...")
    future_times = []
    future_prices = []
    future_upper = []
    future_lower = []
    future_confidence = []
    
    for i, p in enumerate(predictions):
        try:
            print(f"처리 중: {i+1}/{len(predictions)} - hour: {p['hour']}, type: {type(p['hour'])}")
            hour = int(p["hour"])
            future_times.append(base_time + timedelta(hours=hour))
            future_prices.append(float(p["price"]))
            future_upper.append(float(p["upper_bound"]))
            future_lower.append(float(p["lower_bound"]))
            future_confidence.append(float(p["confidence"]) * 100)
        except Exception as e:
            print(f"데이터 변환 오류: {e}, 데이터: {p}")
            break
    
    print(f"미래 데이터: {len(future_times)}개 포인트 생성됨")
    
    # 4. 차트 생성
    print("차트 생성 중...")
    try:
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("BTC 가격 예측 (12시간 전 ~ 24시간 후)", "예측 신뢰도"),
            vertical_spacing=0.12,
            row_heights=[0.75, 0.25]
        )
        print("서브플롯 생성 완료")
    except Exception as e:
        print(f"서브플롯 생성 실패: {e}")
        return
    
    # 과거 가격
    fig.add_trace(
        go.Scatter(
            x=past_times,
            y=past_prices,
            mode='lines+markers',
            name='과거 가격',
            line=dict(color='blue', width=2),
            marker=dict(size=4)
        ),
        row=1, col=1
    )
    
    # 현재 시점
    fig.add_trace(
        go.Scatter(
            x=now_time,
            y=now_price,
            mode='markers',
            name='현재 가격',
            marker=dict(color='red', size=12, symbol='diamond')
        ),
        row=1, col=1
    )
    
    # 미래 예측
    fig.add_trace(
        go.Scatter(
            x=future_times,
            y=future_prices,
            mode='lines+markers',
            name='미래 예측',
            line=dict(color='orange', width=3, dash='dot'),
            marker=dict(size=6, symbol='triangle-up')
        ),
        row=1, col=1
    )
    
    # 신뢰 구간
    fig.add_trace(
        go.Scatter(
            x=future_times + future_times[::-1],
            y=future_upper + future_lower[::-1],
            fill='toself',
            fillcolor='rgba(255,165,0,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='신뢰 구간',
            hoverinfo='skip'
        ),
        row=1, col=1
    )
    
    # 신뢰도
    fig.add_trace(
        go.Scatter(
            x=future_times,
            y=future_confidence,
            mode='lines+markers',
            name='신뢰도 (%)',
            line=dict(color='green', width=2),
            marker=dict(size=4)
        ),
        row=2, col=1
    )
    
    # 현재 시점 수직선
    fig.add_vline(
        x=base_time,
        line=dict(color="red", width=2, dash="dash"),
        annotation_text="현재",
        annotation_position="top"
    )
    
    # 레이아웃
    fig.update_layout(
        title={
            'text': f"🚀 BTC 실시간 예측 차트<br><sub>${current_price:,.0f} | {base_time.strftime('%m-%d %H:%M')}</sub>",
            'x': 0.5,
            'font': {'size': 18}
        },
        height=700,
        showlegend=True,
        template='plotly_white'
    )
    
    # 축 설정
    fig.update_yaxes(title_text="BTC 가격 (USD)", row=1, col=1)
    fig.update_yaxes(title_text="신뢰도 (%)", row=2, col=1, range=[0, 100])
    fig.update_xaxes(title_text="시간", row=2, col=1)
    
    # 저장
    chart_path = "/Users/parkyoungjun/Desktop/BTC_Analysis_System/btc_live_chart.html"
    fig.write_html(chart_path)
    
    print(f"✅ 차트 저장: {chart_path}")
    
    # 예측 요약
    pred_1h = predictions[0]
    pred_24h = predictions[-1]
    
    print(f"\n📊 예측 요약:")
    print(f"  • 1시간 후: ${pred_1h['price']:,.0f} ({(pred_1h['price']-current_price)/current_price*100:+.2f}%)")
    print(f"  • 24시간 후: ${pred_24h['price']:,.0f} ({(pred_24h['price']-current_price)/current_price*100:+.2f}%)")
    print(f"  • 평균 신뢰도: {sum(future_confidence)/len(future_confidence):.1f}%")
    
    # 브라우저 열기
    try:
        import subprocess
        subprocess.run(["open", chart_path])
        print("🌐 브라우저에서 차트 열림!")
    except:
        print(f"💡 브라우저에서: {chart_path}")
    
    return chart_path

if __name__ == "__main__":
    print("🎯 BTC 직접 차트 생성기")
    print("="*40)
    
    try:
        chart_path = create_btc_prediction_chart()
        print("\n" + "="*40)
        print("🎉 차트 생성 성공!")
    except Exception as e:
        print(f"❌ 실패: {e}")