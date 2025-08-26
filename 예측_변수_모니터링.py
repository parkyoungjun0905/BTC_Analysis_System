#!/usr/bin/env python3
"""
🎯 BTC 예측 변수 모니터링 시스템
예측에 영향을 주는 핵심 변수들과 변화량 실시간 추적

사용법: python3 예측_변수_모니터링.py
"""

import asyncio
import aiohttp
import json
from datetime import datetime, timedelta
import math

class PredictionVariableMonitor:
    def __init__(self):
        self.logger_name = "Variable Monitor"
        print("🔍 BTC 예측 변수 모니터링 시스템")
        print("=" * 60)
        print("📊 예측에 영향을 주는 핵심 변수들을 실시간 추적합니다")
        print("⚡ 변수 변화시 예측 가격 변화량도 계산합니다")
        print("=" * 60)

    async def get_market_data(self):
        """시장 데이터 수집"""
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                # 현재 가격 및 24시간 데이터
                async with session.get('https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd&include_24hr_change=true&include_24hr_vol=true') as resp:
                    price_data = await resp.json()
                
                # Binance 상세 데이터
                async with session.get('https://api.binance.com/api/v3/ticker/24hr?symbol=BTCUSDT') as resp:
                    binance_data = await resp.json()
                
                # Fear & Greed Index (단순 추정)
                fear_greed = self.estimate_fear_greed(price_data['bitcoin']['usd_24h_change'])
                
                return {
                    'current_price': price_data['bitcoin']['usd'],
                    'price_change_24h': price_data['bitcoin']['usd_24h_change'],
                    'volume_24h_usd': price_data['bitcoin']['usd_24h_vol'],
                    'volume_24h_btc': float(binance_data['volume']),
                    'high_24h': float(binance_data['highPrice']),
                    'low_24h': float(binance_data['lowPrice']),
                    'price_change_percent': float(binance_data['priceChangePercent']),
                    'fear_greed_index': fear_greed,
                    'timestamp': datetime.now()
                }
        except Exception as e:
            print(f"❌ 데이터 수집 실패: {e}")
            return None

    def estimate_fear_greed(self, price_change):
        """Fear & Greed Index 추정"""
        if price_change > 10: return 80  # Extreme Greed
        elif price_change > 5: return 70   # Greed
        elif price_change > 0: return 55   # Neutral-Greed
        elif price_change > -5: return 45  # Neutral-Fear
        elif price_change > -10: return 30 # Fear
        else: return 15  # Extreme Fear

    def calculate_variable_impacts(self, current_data):
        """각 변수의 예측에 미치는 영향력 계산"""
        base_price = current_data['current_price']
        
        # 1. 24시간 가격 변화 변수
        price_change = current_data['price_change_24h']
        price_impact = self.analyze_price_change_impact(price_change, base_price)
        
        # 2. 볼륨 변수
        volume_btc = current_data['volume_24h_btc']
        volume_impact = self.analyze_volume_impact(volume_btc, base_price)
        
        # 3. 변동성 변수
        volatility = (current_data['high_24h'] - current_data['low_24h']) / base_price * 100
        volatility_impact = self.analyze_volatility_impact(volatility, base_price)
        
        # 4. 심리 지표 변수
        fear_greed = current_data['fear_greed_index']
        sentiment_impact = self.analyze_sentiment_impact(fear_greed, base_price)
        
        return {
            'price_change': price_impact,
            'volume': volume_impact,
            'volatility': volatility_impact,
            'sentiment': sentiment_impact,
            'base_price': base_price
        }

    def analyze_price_change_impact(self, price_change, base_price):
        """가격 변화 변수 분석"""
        current_impact = abs(price_change) * 0.1  # 현재 영향력
        
        # 시나리오별 예측 변화
        scenarios = {}
        test_changes = [-15, -10, -5, -2, 0, 2, 5, 10, 15]
        
        for test_change in test_changes:
            if test_change < -10:
                # 강한 하락시 반등 예상
                prediction_change = 2.5
                confidence = 0.90
            elif test_change < -3:
                # 중간 하락시 반등 예상  
                prediction_change = 1.8
                confidence = 0.85
            elif test_change < -1:
                # 약한 하락시
                prediction_change = 0.8
                confidence = 0.60
            elif test_change < 1:
                # 보합
                prediction_change = 0.3
                confidence = 0.40
            elif test_change < 3:
                # 약한 상승
                prediction_change = 0.5
                confidence = 0.55
            elif test_change < 7:
                # 중간 상승
                prediction_change = -0.5
                confidence = 0.70
            else:
                # 강한 상승시 조정 예상
                prediction_change = -2.0
                confidence = 0.85
            
            predicted_price = base_price * (1 + prediction_change / 100)
            scenarios[f"{test_change:+.1f}%"] = {
                'predicted_price': round(predicted_price, 2),
                'change_from_current': round(prediction_change, 2),
                'confidence': round(confidence * 100, 1)
            }
        
        return {
            'variable_name': '24시간 가격변화율',
            'current_value': round(price_change, 2),
            'current_impact': f"{current_impact:.1f}% 영향",
            'sensitivity': 'HIGH (가격 예측에 가장 큰 영향)',
            'scenarios': scenarios,
            'monitoring_alert': f"±3% 변화시 예측 ±1-2% 변동"
        }

    def analyze_volume_impact(self, volume_btc, base_price):
        """거래량 변수 분석"""
        current_volume_score = min(volume_btc / 50000, 5.0)  # 정규화
        
        scenarios = {}
        test_volumes = [10000, 30000, 50000, 80000, 120000, 200000, 500000]
        
        for test_vol in test_volumes:
            volume_ratio = test_vol / 50000  # 기준값 대비
            
            if volume_ratio > 4:
                # 매우 높은 거래량
                prediction_change = 2.8
                confidence = 0.92
                signal = "강한 돌파"
            elif volume_ratio > 2:
                # 높은 거래량
                prediction_change = 1.5
                confidence = 0.78
                signal = "볼륨 증가"
            elif volume_ratio > 1:
                # 보통 거래량
                prediction_change = 0.5
                confidence = 0.55
                signal = "일반"
            else:
                # 낮은 거래량
                prediction_change = 0.2
                confidence = 0.35
                signal = "저조"
            
            predicted_price = base_price * (1 + prediction_change / 100)
            scenarios[f"{test_vol:,}"] = {
                'predicted_price': round(predicted_price, 2),
                'change_from_current': round(prediction_change, 2),
                'confidence': round(confidence * 100, 1),
                'signal': signal
            }
        
        return {
            'variable_name': '24시간 거래량 (BTC)',
            'current_value': f"{volume_btc:,.0f} BTC",
            'sensitivity': 'MEDIUM (거래량 2배 증가시 예측 +1-2% 상승)',
            'scenarios': scenarios,
            'monitoring_alert': f"100K BTC 돌파시 강한 신호, 30K 미만시 약한 신호"
        }

    def analyze_volatility_impact(self, volatility, base_price):
        """변동성 변수 분석"""
        scenarios = {}
        test_volatilities = [1, 3, 5, 8, 12, 20, 30]
        
        for test_vol in test_volatilities:
            if test_vol < 2:
                # 낮은 변동성
                prediction_change = 0.1
                confidence = 0.30
                signal = "횡보 지속"
            elif test_vol < 5:
                # 보통 변동성
                prediction_change = 0.8
                confidence = 0.65
                signal = "정상 범위"
            elif test_vol < 10:
                # 높은 변동성
                prediction_change = 1.8
                confidence = 0.80
                signal = "활발한 거래"
            else:
                # 극도로 높은 변동성
                prediction_change = 3.2
                confidence = 0.85
                signal = "극단적 움직임"
            
            predicted_price = base_price * (1 + prediction_change / 100)
            scenarios[f"{test_vol}%"] = {
                'predicted_price': round(predicted_price, 2),
                'change_from_current': round(prediction_change, 2),
                'confidence': round(confidence * 100, 1),
                'signal': signal
            }
        
        return {
            'variable_name': '24시간 변동성',
            'current_value': f"{volatility:.1f}%",
            'sensitivity': 'MEDIUM (변동성 10% 초과시 큰 움직임 예상)',
            'scenarios': scenarios,
            'monitoring_alert': f"5% 이상시 주의, 15% 이상시 극단적 움직임"
        }

    def analyze_sentiment_impact(self, fear_greed, base_price):
        """심리 지표 변수 분석"""
        scenarios = {}
        test_sentiments = [10, 25, 40, 50, 60, 75, 90]
        
        for sentiment in test_sentiments:
            if sentiment < 20:
                # 극도의 공포
                prediction_change = 3.0  # 반등 예상
                confidence = 0.88
                signal = "극도 공포 → 반등"
            elif sentiment < 40:
                # 공포
                prediction_change = 1.5
                confidence = 0.75
                signal = "공포 → 상승"
            elif sentiment < 60:
                # 중립
                prediction_change = 0.3
                confidence = 0.45
                signal = "중립"
            elif sentiment < 80:
                # 탐욕
                prediction_change = -1.0
                confidence = 0.70
                signal = "탐욕 → 조정"
            else:
                # 극도의 탐욕
                prediction_change = -2.5
                confidence = 0.85
                signal = "극도 탐욕 → 급락"
            
            predicted_price = base_price * (1 + prediction_change / 100)
            scenarios[f"{sentiment}"] = {
                'predicted_price': round(predicted_price, 2),
                'change_from_current': round(prediction_change, 2),
                'confidence': round(confidence * 100, 1),
                'signal': signal
            }
        
        return {
            'variable_name': 'Fear & Greed Index',
            'current_value': f"{fear_greed}",
            'sensitivity': 'HIGH (극단값에서 반전 신호 강함)',
            'scenarios': scenarios,
            'monitoring_alert': f"20 미만 또는 80 초과시 강한 반전 신호"
        }

    async def run_variable_monitoring(self):
        """변수 모니터링 실행"""
        print("🚀 실시간 변수 모니터링 시작!\n")
        
        while True:
            try:
                # 데이터 수집
                market_data = await self.get_market_data()
                if not market_data:
                    print("❌ 데이터 수집 실패, 1분 후 재시도...")
                    await asyncio.sleep(60)
                    continue
                
                # 변수 영향 분석
                impacts = self.calculate_variable_impacts(market_data)
                
                # 결과 출력
                current_time = market_data['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
                current_price = market_data['current_price']
                
                print(f"⏰ 분석 시간: {current_time}")
                print(f"💰 현재 BTC: ${current_price:,.2f}")
                print("=" * 80)
                
                # 각 변수별 상세 분석
                for var_name, impact_data in impacts.items():
                    if var_name == 'base_price':
                        continue
                    
                    print(f"\n📊 {impact_data['variable_name']}")
                    print(f"   현재값: {impact_data['current_value']}")
                    print(f"   민감도: {impact_data['sensitivity']}")
                    print(f"   ⚠️  모니터링 알림: {impact_data['monitoring_alert']}")
                    
                    print(f"   📈 변수 변화시 예측 가격 시나리오:")
                    for scenario, result in list(impact_data['scenarios'].items())[:7]:  # 상위 7개만 표시
                        print(f"      {scenario:>8} → ${result['predicted_price']:>8,.2f} ({result['change_from_current']:+.1f}%) [{result['confidence']}%]")
                
                print("\n" + "=" * 80)
                
                # 핵심 모니터링 포인트 요약
                print("🎯 현재 주요 모니터링 포인트:")
                
                # 가격 변화 알림
                price_change = market_data['price_change_24h']
                if abs(price_change) > 5:
                    print(f"   🚨 가격변화 주의: {price_change:+.1f}% (큰 변화 감지)")
                else:
                    print(f"   ✅ 가격변화 안정: {price_change:+.1f}% (정상 범위)")
                
                # 거래량 알림
                volume_btc = market_data['volume_24h_btc']
                if volume_btc > 100000:
                    print(f"   🚨 거래량 급증: {volume_btc:,.0f} BTC (강한 신호)")
                elif volume_btc < 30000:
                    print(f"   ⚠️ 거래량 저조: {volume_btc:,.0f} BTC (약한 신호)")
                else:
                    print(f"   ✅ 거래량 정상: {volume_btc:,.0f} BTC (일반 수준)")
                
                # 변동성 알림
                volatility = (market_data['high_24h'] - market_data['low_24h']) / current_price * 100
                if volatility > 10:
                    print(f"   🚨 높은 변동성: {volatility:.1f}% (큰 움직임 예상)")
                else:
                    print(f"   ✅ 변동성 보통: {volatility:.1f}% (안정적)")
                
                # 심리 지표 알림
                fear_greed = market_data['fear_greed_index']
                if fear_greed < 25 or fear_greed > 75:
                    print(f"   🚨 극단적 심리: {fear_greed} (반전 신호 주의)")
                else:
                    print(f"   ✅ 심리 지표 정상: {fear_greed} (중립 범위)")
                
                print("\n" + "=" * 80)
                print("🔄 3분 후 업데이트... (Ctrl+C로 종료)")
                
                # 3분 대기
                await asyncio.sleep(180)
                
            except KeyboardInterrupt:
                print("\n\n👋 변수 모니터링 종료")
                break
            except Exception as e:
                print(f"❌ 오류 발생: {e}")
                await asyncio.sleep(60)

if __name__ == "__main__":
    monitor = PredictionVariableMonitor()
    asyncio.run(monitor.run_variable_monitoring())