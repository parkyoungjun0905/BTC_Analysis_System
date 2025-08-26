#!/usr/bin/env python3
"""
💰 실제 현재 BTC 가격 확인
- 여러 거래소 현재가 조회
- 실시간 데이터로 예측 테스트
"""

import requests
import json
from datetime import datetime

def get_current_btc_price():
    """실제 현재 BTC 가격 조회"""
    print("💰 실제 현재 BTC 가격 조회")
    print("=" * 40)
    
    exchanges = {
        "바이낸스": "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT",
        "코인베이스": "https://api.coinbase.com/v2/exchange-rates?currency=BTC",
        "업비트": "https://api.upbit.com/v1/ticker?markets=KRW-BTC"
    }
    
    prices = {}
    
    for exchange, url in exchanges.items():
        try:
            response = requests.get(url, timeout=5)
            data = response.json()
            
            if exchange == "바이낸스":
                price = float(data['price'])
                prices[exchange] = price
                print(f"📊 {exchange:8s}: ${price:8,.0f}")
                
            elif exchange == "코인베이스":
                price = float(data['data']['rates']['USD'])
                prices[exchange] = price
                print(f"📊 {exchange:8s}: ${price:8,.0f}")
                
            elif exchange == "업비트":
                price_krw = data[0]['trade_price']
                # 환율 1350원 가정
                price_usd = price_krw / 1350
                prices[exchange] = price_usd
                print(f"📊 {exchange:8s}: ${price_usd:8,.0f} (₩{price_krw:,})")
                
        except Exception as e:
            print(f"❌ {exchange} 오류: {e}")
    
    if prices:
        avg_price = sum(prices.values()) / len(prices)
        print("-" * 40)
        print(f"📈 평균 현재가: ${avg_price:,.0f}")
        print(f"⏰ 조회 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 가격 차이 확인
        if len(prices) > 1:
            max_price = max(prices.values())
            min_price = min(prices.values())
            spread = ((max_price - min_price) / avg_price) * 100
            print(f"💸 거래소간 차이: {spread:.2f}%")
        
        return avg_price, prices
    else:
        print("❌ 모든 거래소 조회 실패")
        return None, {}

def compare_with_test_data():
    """테스트 데이터와 실제 가격 비교"""
    real_price, exchanges = get_current_btc_price()
    test_price = 94000
    
    print("\n🔍 테스트 vs 실제 비교")
    print("=" * 40)
    print(f"🧪 테스트 데이터:  ${test_price:,}")
    
    if real_price:
        print(f"💰 실제 현재가:    ${real_price:,.0f}")
        
        diff = real_price - test_price
        diff_pct = (diff / test_price) * 100
        
        print(f"📊 차이:          ${diff:+,.0f} ({diff_pct:+.1f}%)")
        
        if abs(diff_pct) > 10:
            print("⚠️  큰 차이! 실제 데이터 사용 권장")
        elif abs(diff_pct) > 5:
            print("⚠️  중간 차이 - 업데이트 고려")
        else:
            print("✅ 비슷한 수준")
    
    return real_price

def get_real_market_data():
    """실제 시장 데이터 가져오기 (간단 버전)"""
    print("\n📊 실제 시장 데이터 수집")
    print("=" * 40)
    
    try:
        # 바이낸스 24시간 데이터
        url = "https://api.binance.com/api/v3/ticker/24hr?symbol=BTCUSDT"
        response = requests.get(url, timeout=10)
        data = response.json()
        
        current_price = float(data['lastPrice'])
        high_24h = float(data['highPrice'])
        low_24h = float(data['lowPrice'])
        volume_24h = float(data['volume'])
        price_change_24h = float(data['priceChangePercent'])
        
        print(f"💰 현재가:        ${current_price:,.0f}")
        print(f"📈 24h 고가:      ${high_24h:,.0f}")
        print(f"📉 24h 저가:      ${low_24h:,.0f}")
        print(f"📊 24h 거래량:    {volume_24h:,.0f} BTC")
        print(f"📊 24h 변동:      {price_change_24h:+.2f}%")
        
        market_data = {
            'current_price': current_price,
            'high': high_24h,
            'low': low_24h,
            'close': current_price,
            'volume': volume_24h,
            'price_change_24h': price_change_24h
        }
        
        print("✅ 실제 시장 데이터 수집 완료")
        return market_data
        
    except Exception as e:
        print(f"❌ 시장 데이터 수집 실패: {e}")
        return None

if __name__ == "__main__":
    print("🚀 실제 BTC 가격 vs 테스트 데이터 비교")
    print("=" * 60)
    
    # 1. 현재 가격 조회
    real_price = compare_with_test_data()
    
    # 2. 실제 시장 데이터 수집
    market_data = get_real_market_data()
    
    # 3. 권장사항
    print("\n💡 권장사항")
    print("=" * 40)
    print("1. 실제 예측 시에는 실시간 API 데이터 사용")
    print("2. 주요 거래소 API 연동 (바이낸스, 코인베이스 등)")
    print("3. 5분~1시간마다 데이터 업데이트")
    print("4. 여러 거래소 평균가 사용으로 정확도 향상")
    
    if market_data:
        print("\n🎯 실제 데이터로 예측 테스트 준비 완료!")
        print("   - 이 데이터를 98% 모델에 입력하면 정확한 예측 가능")