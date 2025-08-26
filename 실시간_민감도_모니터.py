#!/usr/bin/env python3
"""
🚨 실시간 민감도 모니터링 시스템
학습된 민감도 결과를 기반으로 실시간 예측 변화 추적

목적: "지금 이 지표가 변하면 예측가격이 얼마나 변할지" 실시간 안내
"""

import asyncio
import aiohttp
import json
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class RealTimeSensitivityMonitor:
    def __init__(self):
        """실시간 민감도 모니터 초기화"""
        self.sensitivity_data = self.load_sensitivity_results()
        self.logger = logging.getLogger(__name__)
        
        print("🚨 실시간 민감도 모니터링 시스템")
        print("=" * 60)
        print("📊 학습된 민감도를 기반으로 실시간 예측 변화 추적")
        print("⚡ 지표 변화시 예측가격 변화량 실시간 안내")
        print("=" * 60)
        
    def load_sensitivity_results(self):
        """학습된 민감도 결과 로드"""
        try:
            with open('prediction_sensitivity_results.json', 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print("❌ 민감도 학습 결과를 찾을 수 없습니다!")
            print("👉 먼저 'python3 prediction_sensitivity_learner.py'를 실행하세요")
            return None
            
    async def get_current_market_data(self):
        """현재 시장 데이터 수집"""
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                # CoinGecko 데이터
                async with session.get('https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd&include_24hr_change=true&include_24hr_vol=true') as resp:
                    price_data = await resp.json()
                
                # Binance 볼륨 데이터
                async with session.get('https://api.binance.com/api/v3/ticker/24hr?symbol=BTCUSDT') as resp:
                    binance_data = await resp.json()
                
                return {
                    'price': price_data['bitcoin']['usd'],
                    'price_change_24h': price_data['bitcoin']['usd_24h_change'],
                    'volume_btc': float(binance_data['volume']),
                    'timestamp': datetime.now()
                }
        except Exception as e:
            self.logger.error(f"데이터 수집 실패: {e}")
            return None
    
    def calculate_current_prediction(self, market_data):
        """현재 데이터로 예측 계산"""
        price = market_data['price']
        price_change_24h = market_data['price_change_24h']
        volume_btc = market_data['volume_btc']
        
        # 학습된 로직으로 예측 (간단화)
        reversal_strength = abs(price_change_24h) / 100
        volume_ratio = volume_btc / 50000
        
        # Momentum 계산
        if reversal_strength > 0.05:
            if price_change_24h < -3:
                momentum_change = 2.5
                momentum_confidence = 0.92
            elif price_change_24h > 5:
                momentum_change = -1.8
                momentum_confidence = 0.88
            else:
                momentum_change = 0.8
                momentum_confidence = 0.45
        else:
            momentum_change = 0.3
            momentum_confidence = 0.30
        
        # Volume 계산
        if volume_ratio >= 2.5:
            if abs(price_change_24h) >= 3:
                volume_change = 2.8
                volume_confidence = 0.94
            else:
                volume_change = 1.5
                volume_confidence = 0.72
        else:
            volume_change = 0.5
            volume_confidence = 0.35
        
        # 앙상블
        if momentum_confidence >= 0.85 and volume_confidence >= 0.6:
            final_change = (momentum_change + volume_change) / 2
            final_confidence = (momentum_confidence + volume_confidence) / 2
        elif momentum_confidence >= 0.85:
            final_change = momentum_change
            final_confidence = momentum_confidence
        elif volume_confidence >= 0.6:
            final_change = volume_change
            final_confidence = volume_confidence
        else:
            final_change = 0.5
            final_confidence = max(momentum_confidence, volume_confidence) * 0.8
        
        predicted_price = price * (1 + final_change / 100)
        
        return {
            'predicted_price': predicted_price,
            'confidence': final_confidence,
            'change_percent': final_change,
            'momentum_component': momentum_change,
            'volume_component': volume_change
        }
    
    def analyze_sensitivity_alerts(self, current_data, prediction):
        """현재 상황에서 주의해야 할 민감도 알림 생성"""
        price = current_data['price']
        price_change_24h = current_data['price_change_24h']
        volume_btc = current_data['volume_btc']
        
        alerts = []
        
        # 1. 가격변화율 민감도 알림
        if self.sensitivity_data:
            price_sens = self.sensitivity_data['price_sensitivity']['learning_results']
            
            # 현재 가격변화에서 임계점까지의 거리 계산
            critical_thresholds = [-7, 7]  # 학습 결과에서 큰 변화가 있는 지점들
            
            for threshold in critical_thresholds:
                distance = abs(price_change_24h - threshold)
                if distance < 2:  # 2% 이내 근접
                    expected_change = price_sens.get(f"{threshold:+.1f}%", {}).get('price_difference', 0)
                    alerts.append({
                        'type': 'PRICE_CRITICAL',
                        'message': f"🚨 가격변화 {threshold}% 임계점 근접! ({distance:.1f}% 차이)",
                        'impact': f"도달시 예측가격 {expected_change:+.0f}$ 변동 예상",
                        'urgency': 'HIGH' if distance < 1 else 'MEDIUM'
                    })
        
        # 2. 볼륨 민감도 알림
        volume_thresholds = [125000, 200000]  # 학습에서 중요한 볼륨 임계점들
        
        for threshold in volume_thresholds:
            if volume_btc < threshold and volume_btc > threshold * 0.8:  # 80% 이상 근접
                distance_pct = (threshold - volume_btc) / threshold * 100
                alerts.append({
                    'type': 'VOLUME_APPROACHING',
                    'message': f"📊 거래량 {threshold:,} BTC 임계점 접근 중! ({distance_pct:.1f}% 남음)",
                    'impact': f"돌파시 예측가격 +1000$+ 상승 예상",
                    'urgency': 'MEDIUM'
                })
        
        # 3. 복합 패턴 알림
        if abs(price_change_24h) > 5 and volume_btc > 100000:
            if price_change_24h < -5:
                pattern = "강한 하락 + 높은 볼륨"
                impact = "+2400$ 상승"
            else:
                pattern = "강한 상승 + 높은 볼륨" 
                impact = "-2500$ 하락"
            
            alerts.append({
                'type': 'PATTERN_ACTIVE',
                'message': f"🎯 고영향 패턴 활성: {pattern}",
                'impact': f"예측가격 {impact} 예상",
                'urgency': 'VERY HIGH'
            })
        
        return alerts
    
    def generate_what_if_scenarios(self, current_data, prediction):
        """현재 상황에서 "만약 X가 변한다면" 시나리오 생성"""
        scenarios = []
        base_price = current_data['price']
        base_predicted = prediction['predicted_price']
        
        # 가격변화 시나리오
        price_scenarios = [
            (current_data['price_change_24h'] - 3, "3% 더 하락한다면"),
            (current_data['price_change_24h'] + 3, "3% 더 상승한다면"),
            (-8, "8% 급락한다면"),
            (8, "8% 급등한다면")
        ]
        
        for test_change, description in price_scenarios:
            # 간단한 영향 계산 (학습 결과 기반)
            if test_change < -7:
                price_impact = 2245  # 학습 결과에서
            elif test_change > 7:
                price_impact = -2582
            else:
                price_impact = 0
            
            new_predicted = base_predicted + price_impact
            
            scenarios.append({
                'type': 'PRICE_SCENARIO',
                'condition': description,
                'new_predicted_price': new_predicted,
                'price_difference': price_impact,
                'probability': self.estimate_probability(test_change, current_data)
            })
        
        # 볼륨 시나리오
        volume_scenarios = [
            (current_data['volume_btc'] * 2, "거래량이 2배가 된다면"),
            (150000, "거래량이 15만 BTC가 된다면"),
            (200000, "거래량이 20만 BTC를 돌파한다면")
        ]
        
        for test_volume, description in volume_scenarios:
            if test_volume > 125000:
                volume_impact = 1122  # 학습 결과에서
            else:
                volume_impact = 0
            
            new_predicted = base_predicted + volume_impact
            
            scenarios.append({
                'type': 'VOLUME_SCENARIO',
                'condition': description,
                'new_predicted_price': new_predicted,
                'price_difference': volume_impact,
                'probability': self.estimate_volume_probability(test_volume, current_data)
            })
        
        return scenarios
    
    def estimate_probability(self, target_change, current_data):
        """가격변화 확률 추정"""
        current_change = current_data['price_change_24h']
        diff = abs(target_change - current_change)
        
        if diff < 2: return "높음"
        elif diff < 5: return "중간"
        else: return "낮음"
    
    def estimate_volume_probability(self, target_volume, current_data):
        """볼륨 확률 추정"""
        current_volume = current_data['volume_btc']
        ratio = target_volume / current_volume
        
        if ratio < 1.5: return "높음"
        elif ratio < 3: return "중간"
        else: return "낮음"
    
    async def run_monitoring(self):
        """실시간 모니터링 실행"""
        if not self.sensitivity_data:
            print("❌ 민감도 데이터가 없습니다. 종료합니다.")
            return
            
        print("🚀 실시간 민감도 모니터링 시작!\n")
        
        iteration = 0
        
        while True:
            try:
                iteration += 1
                
                # 현재 데이터 수집
                current_data = await self.get_current_market_data()
                if not current_data:
                    print("❌ 데이터 수집 실패, 1분 후 재시도...")
                    await asyncio.sleep(60)
                    continue
                
                # 현재 예측 계산
                current_prediction = self.calculate_current_prediction(current_data)
                
                # 민감도 알림 분석
                alerts = self.analyze_sensitivity_alerts(current_data, current_prediction)
                
                # What-if 시나리오 생성
                scenarios = self.generate_what_if_scenarios(current_data, current_prediction)
                
                # 결과 출력
                current_time = current_data['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
                print(f"🔄 업데이트 #{iteration} - {current_time}")
                print("=" * 80)
                
                # 현재 상태
                print(f"💰 현재 BTC: ${current_data['price']:,.2f}")
                print(f"📈 24H 변화: {current_data['price_change_24h']:+.2f}%")
                print(f"📊 거래량: {current_data['volume_btc']:,.0f} BTC")
                print(f"🎯 현재 예측: ${current_prediction['predicted_price']:,.2f} (신뢰도: {current_prediction['confidence']*100:.1f}%)")
                
                # 민감도 알림
                if alerts:
                    print(f"\n🚨 민감도 알림 ({len(alerts)}개):")
                    for alert in alerts:
                        urgency_icon = {"VERY HIGH": "🔴", "HIGH": "🟠", "MEDIUM": "🟡"}.get(alert['urgency'], "⚪")
                        print(f"   {urgency_icon} {alert['message']}")
                        print(f"      💡 영향: {alert['impact']}")
                else:
                    print(f"\n✅ 현재 민감도 알림 없음 (안정 상태)")
                
                # What-if 시나리오 (상위 3개만)
                high_impact_scenarios = [s for s in scenarios if abs(s['price_difference']) > 500][:3]
                if high_impact_scenarios:
                    print(f"\n🎭 주요 시나리오 예측:")
                    for i, scenario in enumerate(high_impact_scenarios, 1):
                        print(f"   {i}. {scenario['condition']}")
                        print(f"      → ${scenario['new_predicted_price']:,.2f} ({scenario['price_difference']:+.0f}$) [발생확률: {scenario['probability']}]")
                
                # 핵심 모니터링 포인트
                print(f"\n👀 지금 주시해야 할 지표:")
                
                # 가격변화 모니터링
                price_change = current_data['price_change_24h']
                if price_change > 5:
                    print(f"   🔸 가격변화 {price_change:+.1f}% → 7% 도달시 예측 -2582$ 변동")
                elif price_change < -5:
                    print(f"   🔸 가격변화 {price_change:+.1f}% → -7% 도달시 예측 +2245$ 변동")
                else:
                    print(f"   🔸 가격변화 {price_change:+.1f}% (안정 범위, 임계점 ±7%)")
                
                # 볼륨 모니터링
                volume = current_data['volume_btc']
                if volume > 100000:
                    print(f"   🔸 거래량 {volume:,.0f} BTC (높음) → 125K 돌파시 예측 +1122$ 상승")
                else:
                    print(f"   🔸 거래량 {volume:,.0f} BTC → 125K BTC 돌파 대기 중")
                
                print("=" * 80)
                print("⏰ 2분 후 업데이트... (Ctrl+C로 종료)")
                
                # 2분 대기
                await asyncio.sleep(120)
                
            except KeyboardInterrupt:
                print("\n\n👋 민감도 모니터링 종료")
                break
            except Exception as e:
                print(f"❌ 오류 발생: {e}")
                await asyncio.sleep(60)

if __name__ == "__main__":
    monitor = RealTimeSensitivityMonitor()
    asyncio.run(monitor.run_monitoring())