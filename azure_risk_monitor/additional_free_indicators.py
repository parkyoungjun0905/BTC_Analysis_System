"""
추가 무료 선행지표 모듈
정확도 향상을 위한 고급 무료 지표들
"""

import asyncio
import aiohttp
import json
from typing import Dict, Optional
from datetime import datetime, timedelta
import logging
import hashlib
import hmac

logger = logging.getLogger(__name__)

class AdditionalFreeIndicators:
    """정확도 향상을 위한 추가 무료 선행지표"""
    
    def __init__(self):
        self.logger = logger
        
    async def collect_additional_indicators(self) -> Dict:
        """추가 선행지표 수집"""
        try:
            tasks = [
                self.get_mempool_analysis(),        # 1. 멤풀 분석
                self.get_exchange_order_book(),     # 2. 거래소 오더북 불균형
                self.get_stablecoin_flows(),        # 3. 스테이블코인 플로우
                self.get_options_data(),            # 4. 옵션 데이터
                self.get_social_sentiment(),        # 5. 소셜 센티먼트
                self.get_mining_difficulty(),       # 6. 채굴 난이도 조정
                self.get_lightning_network(),       # 7. 라이트닝 네트워크
                self.get_defi_tvl_changes()         # 8. DeFi TVL 변화
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            indicators = {}
            names = [
                "mempool_pressure",
                "orderbook_imbalance", 
                "stablecoin_dynamics",
                "options_structure",
                "social_momentum",
                "mining_economics",
                "lightning_adoption",
                "defi_flows"
            ]
            
            for name, result in zip(names, results):
                if isinstance(result, Exception):
                    self.logger.error(f"{name} 수집 실패: {result}")
                    indicators[name] = self.get_default_indicator(name)
                else:
                    indicators[name] = result
                    
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "indicators": indicators,
                "analysis": self.analyze_additional_signals(indicators)
            }
            
        except Exception as e:
            self.logger.error(f"추가 지표 수집 실패: {e}")
            return {}
    
    async def get_mempool_analysis(self) -> Dict:
        """멤풀 분석 - 거래 대기 압력"""
        try:
            async with aiohttp.ClientSession() as session:
                # Mempool.space API (무료)
                url = "https://mempool.space/api/v1/fees/recommended"
                async with session.get(url) as response:
                    if response.status == 200:
                        fees = await response.json()
                        
                # 멤풀 사이즈
                url2 = "https://mempool.space/api/mempool"
                async with session.get(url2) as response:
                    if response.status == 200:
                        mempool = await response.json()
                        
                        # 수수료 압력 분석
                        fee_pressure = fees.get("fastestFee", 0)
                        mempool_size = mempool.get("vsize", 0) / 1000000  # MB
                        
                        # 급격한 수수료 상승 = 온체인 활동 급증 = 가격 변동 신호
                        if fee_pressure > 50:  # 50 sat/vB 이상
                            signal = "BULLISH"
                            strength = min(fee_pressure / 100, 1.0)
                        elif fee_pressure < 10:
                            signal = "BEARISH"
                            strength = 0.3
                        else:
                            signal = "NEUTRAL"
                            strength = 0.5
                            
                        return {
                            "fee_pressure": fee_pressure,
                            "mempool_size_mb": mempool_size,
                            "congestion_level": "high" if mempool_size > 50 else "normal",
                            "signal": signal,
                            "strength": strength,
                            "interpretation": "높은 수수료 = 급한 거래 = 큰 움직임 예상"
                        }
                        
        except Exception as e:
            self.logger.error(f"멤풀 분석 실패: {e}")
            return self.get_default_indicator("mempool_pressure")
    
    async def get_exchange_order_book(self) -> Dict:
        """거래소 오더북 불균형 분석"""
        try:
            async with aiohttp.ClientSession() as session:
                # Binance 오더북 깊이
                url = "https://api.binance.com/api/v3/depth?symbol=BTCUSDT&limit=100"
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # 매수/매도 벽 분석
                        bids = data.get("bids", [])
                        asks = data.get("asks", [])
                        
                        # 상위 10개 호가 총량
                        bid_volume = sum(float(b[1]) for b in bids[:10])
                        ask_volume = sum(float(a[1]) for a in asks[:10])
                        
                        # 불균형 비율
                        imbalance_ratio = (bid_volume - ask_volume) / (bid_volume + ask_volume)
                        
                        # 매수벽이 크면 지지, 매도벽이 크면 저항
                        if imbalance_ratio > 0.2:
                            signal = "BULLISH"
                            strength = min(abs(imbalance_ratio) * 2, 1.0)
                        elif imbalance_ratio < -0.2:
                            signal = "BEARISH"
                            strength = min(abs(imbalance_ratio) * 2, 1.0)
                        else:
                            signal = "NEUTRAL"
                            strength = 0.5
                            
                        return {
                            "bid_volume": bid_volume,
                            "ask_volume": ask_volume,
                            "imbalance_ratio": imbalance_ratio,
                            "buy_wall": bid_volume > ask_volume * 1.5,
                            "sell_wall": ask_volume > bid_volume * 1.5,
                            "signal": signal,
                            "strength": strength
                        }
                        
        except Exception as e:
            self.logger.error(f"오더북 분석 실패: {e}")
            return self.get_default_indicator("orderbook_imbalance")
    
    async def get_stablecoin_flows(self) -> Dict:
        """스테이블코인 유입/유출 분석"""
        try:
            async with aiohttp.ClientSession() as session:
                # CoinGecko에서 스테이블코인 시총 변화
                stablecoins = ["tether", "usd-coin", "dai"]
                total_mcap_change = 0
                
                for stable in stablecoins:
                    url = f"https://api.coingecko.com/api/v3/coins/{stable}"
                    async with session.get(url) as response:
                        if response.status == 200:
                            data = await response.json()
                            # 24시간 시총 변화
                            mcap_change = data.get("market_data", {}).get("market_cap_change_percentage_24h", 0)
                            total_mcap_change += mcap_change
                
                # 스테이블코인 시총 증가 = 자금 유입 = 매수 대기
                avg_change = total_mcap_change / len(stablecoins)
                
                if avg_change > 2:  # 2% 이상 증가
                    signal = "BULLISH"
                    strength = min(avg_change / 5, 1.0)
                elif avg_change < -2:  # 2% 이상 감소
                    signal = "BEARISH"
                    strength = min(abs(avg_change) / 5, 1.0)
                else:
                    signal = "NEUTRAL"
                    strength = 0.5
                    
                return {
                    "stablecoin_mcap_change": avg_change,
                    "money_flow": "inflow" if avg_change > 0 else "outflow",
                    "signal": signal,
                    "strength": strength,
                    "interpretation": "스테이블코인 증가 = 매수 대기 자금"
                }
                
        except Exception as e:
            self.logger.error(f"스테이블코인 분석 실패: {e}")
            return self.get_default_indicator("stablecoin_dynamics")
    
    async def get_options_data(self) -> Dict:
        """옵션 데이터 분석 (Deribit 공개 데이터)"""
        try:
            async with aiohttp.ClientSession() as session:
                # Deribit 공개 API
                url = "https://www.deribit.com/api/v2/public/get_book_summary_by_currency?currency=BTC&kind=option"
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        result = data.get("result", [])
                        
                        # Put/Call 비율 계산
                        calls = [r for r in result if "C" in r.get("instrument_name", "")]
                        puts = [r for r in result if "P" in r.get("instrument_name", "")]
                        
                        call_volume = sum(r.get("volume", 0) for r in calls)
                        put_volume = sum(r.get("volume", 0) for r in puts)
                        
                        pc_ratio = put_volume / call_volume if call_volume > 0 else 1
                        
                        # Put/Call 비율로 시장 심리 판단
                        if pc_ratio > 1.2:  # 풋이 많음 = 헤지 = 불안
                            signal = "BEARISH"
                            strength = min((pc_ratio - 1) * 2, 1.0)
                        elif pc_ratio < 0.7:  # 콜이 많음 = 낙관
                            signal = "BULLISH"
                            strength = min((1 - pc_ratio) * 2, 1.0)
                        else:
                            signal = "NEUTRAL"
                            strength = 0.5
                            
                        return {
                            "put_call_ratio": pc_ratio,
                            "call_volume": call_volume,
                            "put_volume": put_volume,
                            "market_sentiment": "fearful" if pc_ratio > 1 else "greedy",
                            "signal": signal,
                            "strength": strength
                        }
                        
        except Exception as e:
            self.logger.error(f"옵션 데이터 분석 실패: {e}")
            return self.get_default_indicator("options_structure")
    
    async def get_social_sentiment(self) -> Dict:
        """소셜 미디어 센티먼트 (Reddit, Twitter 대체)"""
        try:
            # LunarCrush 무료 API 또는 Alternative.me 센티먼트
            async with aiohttp.ClientSession() as session:
                # Alternative.me 소셜 볼륨
                url = "https://api.alternative.me/v2/ticker/bitcoin/"
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        btc_data = data.get("data", {}).get("1", {})
                        
                        # 소셜 신호들
                        social_volume = btc_data.get("quotes", {}).get("USD", {}).get("volume_24h", 0)
                        percent_change = btc_data.get("quotes", {}).get("USD", {}).get("percent_change_24h", 0)
                        
                        # 거래량 급증 + 가격 정체 = 돌파 임박
                        volume_spike = social_volume > 30000000000  # 300억 달러 이상
                        
                        if volume_spike and abs(percent_change) < 2:
                            signal = "BULLISH"  # 축적 단계
                            strength = 0.8
                        elif not volume_spike and percent_change < -3:
                            signal = "BEARISH"  # 관심 감소
                            strength = 0.7
                        else:
                            signal = "NEUTRAL"
                            strength = 0.5
                            
                        return {
                            "social_volume": social_volume,
                            "volume_spike": volume_spike,
                            "sentiment_shift": "positive" if percent_change > 0 else "negative",
                            "signal": signal,
                            "strength": strength
                        }
                        
        except Exception as e:
            self.logger.error(f"소셜 센티먼트 분석 실패: {e}")
            return self.get_default_indicator("social_momentum")
    
    async def get_mining_difficulty(self) -> Dict:
        """채굴 난이도 및 해시레이트 분석"""
        try:
            async with aiohttp.ClientSession() as session:
                # Blockchain.com API
                url = "https://api.blockchain.info/stats"
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        difficulty = data.get("difficulty", 0)
                        hash_rate = data.get("hash_rate", 0)
                        
                        # 난이도 조정 예측 (2주마다)
                        # 해시레이트 증가 = 채굴자 신뢰 = 강세
                        
                        # 간단한 추세 (실제로는 14일 평균 필요)
                        if hash_rate > 500000000:  # 500 EH/s 이상
                            signal = "BULLISH"
                            strength = 0.7
                        else:
                            signal = "NEUTRAL"
                            strength = 0.5
                            
                        return {
                            "difficulty": difficulty,
                            "hash_rate": hash_rate,
                            "miner_confidence": "high" if hash_rate > 500000000 else "normal",
                            "signal": signal,
                            "strength": strength
                        }
                        
        except Exception as e:
            self.logger.error(f"채굴 데이터 분석 실패: {e}")
            return self.get_default_indicator("mining_economics")
    
    async def get_lightning_network(self) -> Dict:
        """라이트닝 네트워크 성장 분석"""
        try:
            async with aiohttp.ClientSession() as session:
                # 1ML API (무료)
                url = "https://1ml.com/statistics?json=true"
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # 라이트닝 네트워크 용량
                        capacity = data.get("total_capacity", 0)
                        node_count = data.get("number_of_nodes", 0)
                        
                        # 네트워크 성장 = 장기 채택 = 강세
                        if capacity > 5000:  # 5000 BTC 이상
                            signal = "BULLISH"
                            strength = 0.6
                        else:
                            signal = "NEUTRAL"
                            strength = 0.5
                            
                        return {
                            "ln_capacity_btc": capacity,
                            "node_count": node_count,
                            "adoption_trend": "growing" if capacity > 5000 else "stable",
                            "signal": signal,
                            "strength": strength
                        }
                        
        except Exception as e:
            self.logger.error(f"라이트닝 네트워크 분석 실패: {e}")
            return self.get_default_indicator("lightning_adoption")
    
    async def get_defi_tvl_changes(self) -> Dict:
        """DeFi TVL 변화 - BTC 관련"""
        try:
            async with aiohttp.ClientSession() as session:
                # DefiLlama API (무료)
                url = "https://api.llama.fi/protocols"
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # WBTC, renBTC 등 BTC 관련 프로토콜
                        btc_protocols = [p for p in data if "BTC" in p.get("name", "").upper()]
                        total_tvl = sum(p.get("tvl", 0) for p in btc_protocols)
                        
                        # TVL 증가 = BTC DeFi 사용 증가
                        if total_tvl > 1000000000:  # 10억 달러 이상
                            signal = "BULLISH"
                            strength = 0.6
                        else:
                            signal = "NEUTRAL"
                            strength = 0.5
                            
                        return {
                            "btc_defi_tvl": total_tvl,
                            "protocols_count": len(btc_protocols),
                            "defi_adoption": "high" if total_tvl > 1000000000 else "moderate",
                            "signal": signal,
                            "strength": strength
                        }
                        
        except Exception as e:
            self.logger.error(f"DeFi TVL 분석 실패: {e}")
            return self.get_default_indicator("defi_flows")
    
    def analyze_additional_signals(self, indicators: Dict) -> Dict:
        """추가 지표 종합 분석"""
        try:
            bullish_count = 0
            bearish_count = 0
            total_strength = 0
            
            for name, indicator in indicators.items():
                if isinstance(indicator, dict):
                    signal = indicator.get("signal", "NEUTRAL")
                    strength = indicator.get("strength", 0.5)
                    
                    if signal == "BULLISH":
                        bullish_count += 1
                        total_strength += strength
                    elif signal == "BEARISH":
                        bearish_count += 1
                        total_strength -= strength
                        
            # 종합 신호
            if bullish_count > bearish_count + 2:
                overall_signal = "STRONG_BULLISH"
            elif bullish_count > bearish_count:
                overall_signal = "BULLISH"
            elif bearish_count > bullish_count + 2:
                overall_signal = "STRONG_BEARISH"
            elif bearish_count > bullish_count:
                overall_signal = "BEARISH"
            else:
                overall_signal = "NEUTRAL"
                
            return {
                "overall_signal": overall_signal,
                "bullish_indicators": bullish_count,
                "bearish_indicators": bearish_count,
                "signal_strength": total_strength / len(indicators) if indicators else 0,
                "confidence": min(abs(total_strength) / len(indicators) * 100, 100) if indicators else 0
            }
            
        except Exception as e:
            self.logger.error(f"신호 분석 실패: {e}")
            return {"overall_signal": "NEUTRAL", "confidence": 0}
    
    def get_default_indicator(self, name: str) -> Dict:
        """기본 지표 값"""
        return {
            "signal": "NEUTRAL",
            "strength": 0.5,
            "error": "데이터 수집 실패"
        }

async def test_additional_indicators():
    """추가 지표 테스트"""
    print("🧪 추가 무료 선행지표 테스트...")
    
    collector = AdditionalFreeIndicators()
    result = await collector.collect_additional_indicators()
    
    if result:
        print("\n✅ 추가 지표 수집 성공!")
        print(f"수집된 지표: {len(result.get('indicators', {}))}")
        
        analysis = result.get("analysis", {})
        print(f"\n📊 종합 분석:")
        print(f"  • 신호: {analysis.get('overall_signal')}")
        print(f"  • 신뢰도: {analysis.get('confidence', 0):.1f}%")
        print(f"  • 강세 지표: {analysis.get('bullish_indicators', 0)}")
        print(f"  • 약세 지표: {analysis.get('bearish_indicators', 0)}")
        
        print(f"\n📋 개별 지표:")
        for name, indicator in result.get("indicators", {}).items():
            if isinstance(indicator, dict):
                print(f"  • {name}: {indicator.get('signal')} ({indicator.get('strength', 0):.2f})")
    else:
        print("❌ 지표 수집 실패")

if __name__ == "__main__":
    asyncio.run(test_additional_indicators())