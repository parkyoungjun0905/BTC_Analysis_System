"""
전문적인 BTC 예측 차트 생성기
- 명확한 날짜/시간 표시
- 실제 과거 데이터 활용
- 깔끔한 디자인
- 상세한 정보 표시
"""

import os
import json
import pandas as pd
from datetime import datetime, timedelta
import asyncio

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("❌ Plotly 미설치 - pip install plotly")
    exit()

from precision_future_predictor import PrecisionFuturePredictor

class ProfessionalBTCChart:
    """전문적인 BTC 예측 차트"""
    
    def __init__(self):
        self.base_path = "/Users/parkyoungjun/Desktop/BTC_Analysis_System"
        self.historical_path = os.path.join(self.base_path, "historical_data")
    
    def load_historical_price_data(self, hours_back: int = 24) -> pd.DataFrame:
        """실제 과거 가격 데이터 로드"""
        try:
            files = sorted([f for f in os.listdir(self.historical_path) 
                           if f.startswith("btc_analysis_") and f.endswith(".json")])
            
            price_data = []
            
            # 최근 파일들에서 가격 데이터 추출
            for filename in files[-hours_back:]:
                filepath = os.path.join(self.historical_path, filename)
                try:
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                    
                    # 타임스탬프 파싱
                    if "collection_time" in data:
                        timestamp_str = data["collection_time"]
                        timestamp = pd.to_datetime(timestamp_str)
                    else:
                        # 파일명에서 타임스탬프 추출
                        time_part = filename.replace("btc_analysis_", "").replace(".json", "")
                        timestamp = pd.to_datetime(time_part)
                    
                    # 가격 추출
                    price = self.extract_price_from_data(data)
                    if price > 0:
                        price_data.append({
                            "timestamp": timestamp,
                            "price": price,
                            "volume": data.get("data_sources", {}).get("legacy_analyzer", {}).get("market_data", {}).get("total_volume", 0)
                        })
                        
                except Exception as e:
                    continue
            
            df = pd.DataFrame(price_data)
            if not df.empty:
                df = df.sort_values('timestamp').reset_index(drop=True)
                print(f"📊 과거 데이터: {len(df)}개 포인트 ({df.iloc[0]['timestamp']} ~ {df.iloc[-1]['timestamp']})")
            
            return df
            
        except Exception as e:
            print(f"과거 데이터 로드 실패: {e}")
            return pd.DataFrame()
    
    def extract_price_from_data(self, data: dict) -> float:
        """데이터에서 가격 추출"""
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
    
    async def create_professional_chart(self):
        """전문적인 예측 차트 생성"""
        print("🚀 전문 BTC 예측 차트 생성 시작...")
        
        # 1. 과거 데이터 로드
        historical_df = self.load_historical_price_data(hours_back=30)
        if historical_df.empty:
            print("❌ 과거 데이터 부족")
            return None
        
        # 2. 최신 예측 실행
        predictor = PrecisionFuturePredictor()
        result = await predictor.predict_future(hours_ahead=24)
        
        if not result or "error" in result:
            print("❌ 예측 실패")
            return None
        
        current_price = result.get("current_price", historical_df.iloc[-1]['price'])
        
        # 3. 예측 데이터 준비
        predictions = []
        if "predictions" in result and "full_predictions" in result["predictions"]:
            predictions = result["predictions"]["full_predictions"]
        
        print(f"💰 현재 가격: ${current_price:,.0f}")
        print(f"🔮 예측 포인트: {len(predictions)}개")
        
        # 4. 차트용 데이터프레임 생성
        now = datetime.now()
        
        # 미래 예측 데이터프레임
        future_data = []
        for pred in predictions:
            future_time = now + timedelta(hours=int(pred["hour"]))
            future_data.append({
                "timestamp": future_time,
                "price": pred["price"],
                "upper": pred["upper_bound"],
                "lower": pred["lower_bound"],
                "confidence": pred["confidence"] * 100
            })
        
        future_df = pd.DataFrame(future_data)
        
        # 5. 전문적인 차트 생성
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=(
                "🪙 BTC 가격 예측 차트",
                "📊 거래량 & 신뢰도", 
                "📈 가격 변화율"
            ),
            vertical_spacing=0.08,
            row_heights=[0.5, 0.25, 0.25],
            specs=[[{"secondary_y": False}],
                   [{"secondary_y": True}],
                   [{"secondary_y": False}]]
        )
        
        # 6. 과거 실제 가격
        fig.add_trace(
            go.Scatter(
                x=historical_df['timestamp'],
                y=historical_df['price'],
                mode='lines+markers',
                name='실제 가격',
                line=dict(color='#1f77b4', width=2),
                marker=dict(size=4),
                hovertemplate='<b>실제 가격</b><br>' +
                             '시간: %{x|%m/%d %H:%M}<br>' +
                             '가격: $%{y:,.0f}<br>' +
                             '<extra></extra>',
                showlegend=True
            ),
            row=1, col=1
        )
        
        # 7. 현재 시점 마커
        fig.add_trace(
            go.Scatter(
                x=[now],
                y=[current_price],
                mode='markers',
                name='현재 시점',
                marker=dict(
                    color='red', 
                    size=15, 
                    symbol='diamond',
                    line=dict(color='darkred', width=2)
                ),
                hovertemplate='<b>현재 시점</b><br>' +
                             '시간: %{x|%m/%d %H:%M}<br>' +
                             '가격: $%{y:,.0f}<br>' +
                             '<extra></extra>',
                showlegend=True
            ),
            row=1, col=1
        )
        
        # 8. 미래 예측 라인
        fig.add_trace(
            go.Scatter(
                x=future_df['timestamp'],
                y=future_df['price'],
                mode='lines+markers',
                name='AI 예측',
                line=dict(color='#ff7f0e', width=3, dash='dot'),
                marker=dict(size=5, symbol='triangle-up'),
                hovertemplate='<b>AI 예측</b><br>' +
                             '시간: %{x|%m/%d %H:%M}<br>' +
                             '예측가: $%{y:,.0f}<br>' +
                             '신뢰도: %{customdata:.1f}%<br>' +
                             '<extra></extra>',
                customdata=future_df['confidence'],
                showlegend=True
            ),
            row=1, col=1
        )
        
        # 9. 신뢰 구간 (채움)
        fig.add_trace(
            go.Scatter(
                x=list(future_df['timestamp']) + list(future_df['timestamp'][::-1]),
                y=list(future_df['upper']) + list(future_df['lower'][::-1]),
                fill='toself',
                fillcolor='rgba(255,127,14,0.15)',
                line=dict(color='rgba(255,255,255,0)'),
                name='예측 신뢰구간',
                hoverinfo='skip',
                showlegend=True
            ),
            row=1, col=1
        )
        
        # 10. 현재 시점 수직선
        fig.add_shape(
            type="line",
            x0=now, x1=now, y0=0, y1=1,
            yref="paper",
            line=dict(color="red", width=2, dash="dash"),
            row=1, col=1
        )
        
        # 현재 시점 주석
        fig.add_annotation(
            x=now,
            y=current_price,
            text=f"현재<br>{now.strftime('%m/%d %H:%M')}",
            showarrow=True,
            arrowhead=2,
            arrowcolor="red",
            bgcolor="white",
            bordercolor="red",
            row=1, col=1
        )
        
        # 11. 거래량 (과거)
        if 'volume' in historical_df.columns:
            fig.add_trace(
                go.Bar(
                    x=historical_df['timestamp'],
                    y=historical_df['volume'] / 1e9,  # 억 단위로 변환
                    name='거래량 (억$)',
                    marker_color='lightblue',
                    opacity=0.6,
                    hovertemplate='<b>거래량</b><br>' +
                                 '시간: %{x|%m/%d %H:%M}<br>' +
                                 '거래량: %{y:.1f}억$<br>' +
                                 '<extra></extra>',
                    yaxis='y2'
                ),
                row=2, col=1
            )
        
        # 12. 신뢰도 라인
        fig.add_trace(
            go.Scatter(
                x=future_df['timestamp'],
                y=future_df['confidence'],
                mode='lines+markers',
                name='AI 신뢰도',
                line=dict(color='#2ca02c', width=2),
                marker=dict(size=4),
                hovertemplate='<b>AI 신뢰도</b><br>' +
                             '시간: %{x|%m/%d %H:%M}<br>' +
                             '신뢰도: %{y:.1f}%<br>' +
                             '<extra></extra>',
                yaxis='y1'
            ),
            row=2, col=1
        )
        
        # 13. 가격 변화율 계산 및 표시
        historical_pct = historical_df['price'].pct_change() * 100
        future_pct = future_df['price'].pct_change() * 100
        
        fig.add_trace(
            go.Bar(
                x=historical_df['timestamp'][1:],  # 첫 번째 값은 NaN이므로 제외
                y=historical_pct[1:],
                name='과거 변화율',
                marker_color=['green' if x >= 0 else 'red' for x in historical_pct[1:]],
                hovertemplate='<b>변화율</b><br>' +
                             '시간: %{x|%m/%d %H:%M}<br>' +
                             '변화: %{y:.2f}%<br>' +
                             '<extra></extra>',
                showlegend=True
            ),
            row=3, col=1
        )
        
        # 14. 레이아웃 설정
        current_time_str = now.strftime("%Y년 %m월 %d일 %H시 %M분")
        prediction_end = (now + timedelta(hours=24)).strftime("%m월 %d일 %H시")
        
        fig.update_layout(
            title={
                'text': f"""
                <b>🚀 BTC 전문 예측 분석 차트</b><br>
                <span style='font-size:14px'>
                현재: ${current_price:,.0f} | 생성시간: {current_time_str}<br>
                예측범위: 24시간 ({prediction_end}까지) | AI신뢰도: 평균 {future_df['confidence'].mean():.1f}%
                </span>
                """,
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 16}
            },
            height=900,
            showlegend=True,
            template='plotly_white',
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.15,
                xanchor="center",
                x=0.5
            )
        )
        
        # 15. 축 설정
        fig.update_xaxes(
            title_text="날짜 및 시간",
            tickformat="%m/%d<br>%H:%M",
            row=3, col=1
        )
        
        fig.update_yaxes(title_text="BTC 가격 (USD)", row=1, col=1)
        fig.update_yaxes(title_text="신뢰도 (%)", range=[0, 100], row=2, col=1)
        fig.update_yaxes(title_text="거래량 (억$)", secondary_y=True, row=2, col=1)
        fig.update_yaxes(title_text="변화율 (%)", row=3, col=1)
        
        # 16. 그리드 설정
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        
        # 17. 차트 저장
        chart_path = os.path.join(self.base_path, "professional_btc_chart.html")
        fig.write_html(chart_path, include_plotlyjs=True)
        
        print(f"✅ 전문 차트 저장: {chart_path}")
        
        # 18. 예측 요약 출력
        print(f"\n📊 전문 예측 요약:")
        print(f"  🕐 현재 시간: {current_time_str}")
        print(f"  💰 현재 가격: ${current_price:,.0f}")
        print(f"  🔮 24시간 예측: ${future_df.iloc[-1]['price']:,.0f} ({((future_df.iloc[-1]['price']/current_price)-1)*100:+.2f}%)")
        print(f"  📈 최고 예상: ${future_df['upper'].max():,.0f}")
        print(f"  📉 최저 예상: ${future_df['lower'].min():,.0f}")
        print(f"  🎯 평균 신뢰도: {future_df['confidence'].mean():.1f}%")
        
        # 19. 주요 시점 예측
        key_hours = [1, 6, 12, 24]
        print(f"\n⏰ 주요 시점 예측:")
        for hour in key_hours:
            if hour <= len(future_df):
                row = future_df.iloc[hour-1]
                time_str = row['timestamp'].strftime("%m/%d %H:%M")
                change_pct = ((row['price']/current_price)-1)*100
                print(f"  • {hour:2d}시간 후 ({time_str}): ${row['price']:,.0f} ({change_pct:+.2f}%) [신뢰도: {row['confidence']:.1f}%]")
        
        # 20. 브라우저에서 열기
        try:
            import subprocess
            subprocess.run(["open", chart_path], check=True)
            print(f"\n🌐 브라우저에서 전문 차트 열림!")
        except:
            print(f"\n💡 브라우저에서 열어보세요: {chart_path}")
        
        return {
            "chart_path": chart_path,
            "current_price": current_price,
            "predictions": future_df.to_dict('records'),
            "summary": {
                "avg_confidence": future_df['confidence'].mean(),
                "price_24h": future_df.iloc[-1]['price'],
                "change_24h": ((future_df.iloc[-1]['price']/current_price)-1)*100
            }
        }

async def main():
    """메인 실행"""
    print("🎯 전문 BTC 예측 차트 생성기")
    print("="*60)
    
    chart_gen = ProfessionalBTCChart()
    result = await chart_gen.create_professional_chart()
    
    if result:
        print("\n" + "="*60)
        print("🎉 전문 차트 생성 완료!")
        print("="*60)
    else:
        print("\n❌ 차트 생성 실패")

if __name__ == "__main__":
    asyncio.run(main())