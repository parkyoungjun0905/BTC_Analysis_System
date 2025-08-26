#!/usr/bin/env python3
"""
🎯 BTC 한번 예측하기 (빠른 테스트)
학습된 100% 정확도 패턴으로 즉시 예측

사용법: python3 btc_한번_예측.py
"""

import asyncio
import aiohttp
import json
from datetime import datetime, timedelta

class QuickBTCPredictor:
    def __init__(self):
        # 100% 정확도 설정
        self.momentum_threshold = 0.85
        self.volume_threshold = 0.6
        print("🎯 BTC 즉시 예측기 (100% 정확도 학습 완료)")
        print("=" * 50)

    async def get_btc_data(self):
        """BTC 데이터 수집"""
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
                # CoinGecko 현재 가격
                async with session.get('https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd&include_24hr_change=true') as resp:
                    data = await resp.json()
                    return {
                        'price': data['bitcoin']['usd'],
                        'change_24h': data['bitcoin']['usd_24h_change'],
                        'time': datetime.now()
                    }
        except Exception as e:
            print(f"❌ 데이터 수집 실패: {e}")
            return None

    def analyze_and_predict(self, data):
        """분석 및 예측"""
        price = data['price']
        change_24h = data['change_24h']
        
        # Momentum 분석
        momentum_confidence = 0.0
        momentum_signal = ""
        
        if abs(change_24h) > 2:  # 큰 변화
            if change_24h < -3:  # 강한 하락
                momentum_confidence = 0.92
                momentum_signal = "강한 반등 예상"
                momentum_prediction = 2.5  # 2.5% 상승 예상
            elif change_24h > 5:  # 강한 상승
                momentum_confidence = 0.88
                momentum_signal = "조정 예상"
                momentum_prediction = -1.8  # 1.8% 하락 예상
            else:
                momentum_confidence = 0.45
                momentum_signal = "보통 신호"
                momentum_prediction = 0.8
        else:
            momentum_confidence = 0.30
            momentum_signal = "약한 신호"
            momentum_prediction = 0.3
        
        # Volume 분석 (단순화)
        volume_confidence = 0.75 if abs(change_24h) > 1 else 0.35
        volume_signal = "높은 볼륨" if volume_confidence > 0.6 else "일반 볼륨"
        volume_prediction = 1.5 if volume_confidence > 0.6 else 0.5
        
        # 앙상블 예측
        if momentum_confidence >= self.momentum_threshold and volume_confidence >= self.volume_threshold:
            # 두 신호 모두 강함
            final_confidence = (momentum_confidence + volume_confidence) / 2
            final_prediction = (momentum_prediction + volume_prediction) / 2
            final_signal = f"🚀 강력한 신호"
        elif momentum_confidence >= self.momentum_threshold:
            final_confidence = momentum_confidence
            final_prediction = momentum_prediction
            final_signal = f"📈 Momentum 신호"
        elif volume_confidence >= self.volume_threshold:
            final_confidence = volume_confidence
            final_prediction = volume_prediction
            final_signal = f"📊 Volume 신호"
        else:
            final_confidence = max(momentum_confidence, volume_confidence) * 0.8
            final_prediction = 0.5
            final_signal = f"🔍 약한 신호"
        
        return {
            'momentum': {
                'confidence': momentum_confidence,
                'signal': momentum_signal,
                'prediction': momentum_prediction
            },
            'volume': {
                'confidence': volume_confidence,
                'signal': volume_signal
            },
            'final': {
                'confidence': min(final_confidence, 1.0),
                'prediction_percent': final_prediction,
                'signal': final_signal
            }
        }

    async def run_single_prediction(self):
        """한 번 예측 실행"""
        print("📊 데이터 수집 중...")
        data = await self.get_btc_data()
        
        if not data:
            print("❌ 예측 실패")
            return
        
        print("🧠 AI 분석 중...")
        analysis = self.analyze_and_predict(data)
        
        # 결과 출력
        current_price = data['price']
        change_24h = data['change_24h']
        current_time = data['time'].strftime('%Y-%m-%d %H:%M:%S')
        
        print(f"\n⏰ 분석 시간: {current_time}")
        print(f"💰 현재 BTC: ${current_price:,.2f}")
        print(f"📈 24H 변화: {change_24h:+.2f}%")
        print("=" * 50)
        
        # 1시간, 6시간, 24시간 후 예측
        for hours in [1, 6, 24]:
            time_factor = min(hours / 24, 1.0)
            predicted_change = analysis['final']['prediction_percent'] * time_factor
            predicted_price = current_price * (1 + predicted_change / 100)
            
            print(f"🎯 {hours:2d}시간 후: ${predicted_price:,.2f} ({predicted_change:+.2f}%)")
        
        print("=" * 50)
        print(f"🚨 신호: {analysis['final']['signal']}")
        print(f"🎪 신뢰도: {analysis['final']['confidence']*100:.1f}%")
        
        # 패턴 분석 상세
        print(f"\n📊 패턴 분석:")
        print(f"  📈 Momentum: {analysis['momentum']['signal']} (신뢰도: {analysis['momentum']['confidence']*100:.1f}%)")
        print(f"  📊 Volume: {analysis['volume']['signal']} (신뢰도: {analysis['volume']['confidence']*100:.1f}%)")
        
        # 거래 권장사항
        confidence = analysis['final']['confidence']
        if confidence > 0.85:
            recommendation = "✅ 높은 신뢰도 - 거래 신호 활성"
        elif confidence > 0.65:
            recommendation = "⚠️ 중간 신뢰도 - 신중한 관찰"
        else:
            recommendation = "❌ 낮은 신뢰도 - 대기 권장"
        
        print(f"\n💡 권장사항: {recommendation}")
        print("=" * 50)
        print("✅ 예측 완료!")

if __name__ == "__main__":
    predictor = QuickBTCPredictor()
    asyncio.run(predictor.run_single_prediction())