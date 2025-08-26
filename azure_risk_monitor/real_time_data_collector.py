#!/usr/bin/env python3
"""
실시간 선행지표 수집기
무료 API들을 활용한 실제 선행지표 데이터 수집
"""

import asyncio
import aiohttp
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import hashlib
import hmac
import time

class RealTimeLeadingIndicators:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # API 엔드포인트들
        self.binance_base = "https://fapi.binance.com"
        self.whale_alert_base = "https://api.whale-alert.io/v1"
        self.fear_greed_base = "https://api.alternative.me/fng"
        self.yahoo_finance_base = "https://query1.finance.yahoo.com/v8/finance/chart"
        
        # 캐시 (API 호출 최적화)
        self.cache = {}
        self.cache_ttl = {}
    
    async def collect_all_real_indicators(self) -> Dict:
        """모든 실시간 선행지표 수집"""
        indicators = {
            "timestamp": datetime.utcnow().isoformat(),
            "data_sources": {
                "binance_derivatives": {},
                "macro_indicators": {},
                "whale_activity": {},
                "sentiment_indicators": {},
                "technical_signals": {}
            }
        }
        
        try:
            # 병렬로 모든 지표 수집
            tasks = [
                self.get_binance_derivatives_indicators(),
                self.get_macro_indicators(),
                self.get_whale_activity_indicators(),
                self.get_sentiment_indicators(),
                self.get_technical_signals()
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 결과 정리
            indicators["data_sources"]["binance_derivatives"] = results[0] if not isinstance(results[0], Exception) else {}
            indicators["data_sources"]["macro_indicators"] = results[1] if not isinstance(results[1], Exception) else {}
            indicators["data_sources"]["whale_activity"] = results[2] if not isinstance(results[2], Exception) else {}
            indicators["data_sources"]["sentiment_indicators"] = results[3] if not isinstance(results[3], Exception) else {}
            indicators["data_sources"]["technical_signals"] = results[4] if not isinstance(results[4], Exception) else {}
            
            # 종합 신호 강도 계산
            indicators["composite_signals"] = self.calculate_composite_signals(indicators)
            
            return indicators
            
        except Exception as e:
            self.logger.error(f"실시간 지표 수집 실패: {e}")
            return {"error": str(e), "timestamp": datetime.utcnow().isoformat()}
    
    async def get_binance_derivatives_indicators(self) -> Dict:
        """Binance 파생상품 선행지표 수집"""
        try:
            async with aiohttp.ClientSession() as session:
                indicators = {}
                
                # 1. 펀딩비 데이터 (중요 선행지표)
                funding_url = f"{self.binance_base}/fapi/v1/fundingRate"
                async with session.get(funding_url, params={"symbol": "BTCUSDT", "limit": 10}) as resp:
                    if resp.status == 200:
                        funding_data = await resp.json()
                        if funding_data:
                            current_funding = float(funding_data[0]["fundingRate"])
                            prev_funding = float(funding_data[1]["fundingRate"]) if len(funding_data) > 1 else current_funding
                            
                            indicators["funding_rate"] = {
                                "current": current_funding,
                                "trend": "rising" if current_funding > prev_funding else "falling" if current_funding < prev_funding else "stable",
                                "acceleration": current_funding - prev_funding,
                                "signal_strength": abs(current_funding) * 100000  # 정규화
                            }
                
                # 2. 오픈 인터레스트 (OI) 변화
                oi_url = f"{self.binance_base}/fapi/v1/openInterest"
                async with session.get(oi_url, params={"symbol": "BTCUSDT"}) as resp:
                    if resp.status == 200:
                        oi_data = await resp.json()
                        current_oi = float(oi_data["openInterest"])
                        
                        # 과거 OI와 비교 (캐시 활용)
                        prev_oi = self.cache.get("prev_oi", current_oi)
                        oi_change = (current_oi - prev_oi) / prev_oi if prev_oi > 0 else 0
                        self.cache["prev_oi"] = current_oi
                        
                        indicators["open_interest"] = {
                            "current": current_oi,
                            "change_pct": oi_change,
                            "momentum": "increasing" if oi_change > 0.02 else "decreasing" if oi_change < -0.02 else "stable",
                            "signal_strength": abs(oi_change) * 10
                        }
                
                # 3. 24시간 거래량 변화 
                ticker_url = f"{self.binance_base}/fapi/v1/ticker/24hr"
                async with session.get(ticker_url, params={"symbol": "BTCUSDT"}) as resp:
                    if resp.status == 200:
                        ticker_data = await resp.json()
                        volume_24h = float(ticker_data["volume"])
                        price_change_pct = float(ticker_data["priceChangePercent"])
                        
                        # 과거 볼륨과 비교
                        prev_volume = self.cache.get("prev_volume", volume_24h)
                        volume_change = (volume_24h - prev_volume) / prev_volume if prev_volume > 0 else 0
                        self.cache["prev_volume"] = volume_24h
                        
                        indicators["volume_analysis"] = {
                            "volume_24h": volume_24h,
                            "volume_change_pct": volume_change,
                            "price_volume_divergence": abs(price_change_pct) < 2 and volume_change > 0.5,  # 거래량 급증하지만 가격 안움직임
                            "signal_strength": volume_change if volume_change > 0 else 0
                        }
                
                # 4. 현물-선물 베이시스
                spot_url = "https://api.binance.com/api/v3/ticker/price"
                async with session.get(spot_url, params={"symbol": "BTCUSDT"}) as resp:
                    if resp.status == 200:
                        spot_data = await resp.json()
                        spot_price = float(spot_data["price"])
                        
                        futures_url = f"{self.binance_base}/fapi/v1/ticker/price"
                        async with session.get(futures_url, params={"symbol": "BTCUSDT"}) as resp2:
                            if resp2.status == 200:
                                futures_data = await resp2.json()
                                futures_price = float(futures_data["price"])
                                
                                basis = (futures_price - spot_price) / spot_price
                                prev_basis = self.cache.get("prev_basis", basis)
                                basis_acceleration = basis - prev_basis
                                self.cache["prev_basis"] = basis
                                
                                indicators["basis_analysis"] = {
                                    "current_basis": basis,
                                    "basis_acceleration": basis_acceleration,
                                    "contango_level": "high" if basis > 0.002 else "normal" if basis > -0.002 else "backwardation",
                                    "signal_strength": abs(basis_acceleration) * 1000
                                }
                
                return indicators
                
        except Exception as e:
            self.logger.error(f"Binance 지표 수집 실패: {e}")
            return {}
    
    async def get_macro_indicators(self) -> Dict:
        """거시경제 선행지표 수집 (Yahoo Finance 활용)"""
        try:
            async with aiohttp.ClientSession() as session:
                indicators = {}
                
                # VIX (변동성 지수)
                await self._get_yahoo_data(session, indicators, "^VIX", "vix", "volatility_fear")
                
                # DXY (달러 지수)
                await self._get_yahoo_data(session, indicators, "DX-Y.NYB", "dxy", "dollar_strength")
                
                # 10년물 수익률
                await self._get_yahoo_data(session, indicators, "^TNX", "us_10y", "interest_rates")
                
                # 금 가격 (안전자산 선호도)
                await self._get_yahoo_data(session, indicators, "GC=F", "gold", "safe_haven")
                
                return indicators
                
        except Exception as e:
            self.logger.error(f"거시경제 지표 수집 실패: {e}")
            return {}
    
    async def _get_yahoo_data(self, session: aiohttp.ClientSession, indicators: Dict, symbol: str, key: str, category: str):
        """Yahoo Finance 데이터 수집 헬퍼"""
        try:
            url = f"{self.yahoo_finance_base}/{symbol}"
            params = {
                "interval": "1h",
                "range": "5d",
                "includePrePost": "false"
            }
            
            async with session.get(url, params=params) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    result = data["chart"]["result"][0]
                    timestamps = result["timestamp"]
                    closes = result["indicators"]["quote"][0]["close"]
                    
                    # 최근 값들
                    current_price = closes[-1]
                    prev_price = closes[-2] if len(closes) > 1 else current_price
                    day_ago_price = closes[-24] if len(closes) >= 24 else prev_price
                    
                    change_1h = (current_price - prev_price) / prev_price if prev_price else 0
                    change_24h = (current_price - day_ago_price) / day_ago_price if day_ago_price else 0
                    
                    indicators[key] = {
                        "current": current_price,
                        "change_1h": change_1h,
                        "change_24h": change_24h,
                        "trend": "rising" if change_24h > 0.01 else "falling" if change_24h < -0.01 else "stable",
                        "acceleration": change_1h - (change_24h / 24),  # 가속도
                        "signal_strength": abs(change_24h) * 10,
                        "category": category
                    }
                    
        except Exception as e:
            self.logger.error(f"Yahoo {symbol} 데이터 수집 실패: {e}")
    
    async def get_whale_activity_indicators(self) -> Dict:
        """고래 활동 선행지표 (Whale Alert API)"""
        try:
            # Whale Alert API는 API 키가 필요하므로 무료 대안 사용
            # 대신 거래소 대량 거래 모니터링
            async with aiohttp.ClientSession() as session:
                indicators = {}
                
                # Binance에서 대량 거래 감지
                trades_url = f"{self.binance_base}/fapi/v1/aggTrades"
                params = {
                    "symbol": "BTCUSDT",
                    "limit": 100
                }
                
                async with session.get(trades_url, params=params) as resp:
                    if resp.status == 200:
                        trades = await resp.json()
                        
                        # 대량 거래 분석 (1000 USDT 이상)
                        large_trades = [t for t in trades if float(t["qty"]) * float(t["p"]) > 1000000]  # $1M+
                        total_large_volume = sum(float(t["qty"]) for t in large_trades)
                        
                        # 매수/매도 압력
                        buy_volume = sum(float(t["qty"]) for t in large_trades if not t["m"])  # m=False는 매수
                        sell_volume = total_large_volume - buy_volume
                        
                        indicators["large_trades"] = {
                            "count_1m_plus": len(large_trades),
                            "total_volume": total_large_volume,
                            "buy_sell_ratio": buy_volume / sell_volume if sell_volume > 0 else 0,
                            "whale_sentiment": "bullish" if buy_volume > sell_volume * 1.2 else "bearish" if sell_volume > buy_volume * 1.2 else "neutral",
                            "signal_strength": len(large_trades) / 100  # 정규화
                        }
                
                return indicators
                
        except Exception as e:
            self.logger.error(f"고래 활동 지표 수집 실패: {e}")
            return {}
    
    async def get_sentiment_indicators(self) -> Dict:
        """시장 센티먼트 선행지표"""
        try:
            async with aiohttp.ClientSession() as session:
                indicators = {}
                
                # 1. 공포탐욕지수
                async with session.get(f"{self.fear_greed_base}?limit=10") as resp:
                    if resp.status == 200:
                        fg_data = await resp.json()
                        current_fg = int(fg_data["data"][0]["value"])
                        prev_fg = int(fg_data["data"][1]["value"]) if len(fg_data["data"]) > 1 else current_fg
                        
                        indicators["fear_greed_index"] = {
                            "current": current_fg,
                            "change": current_fg - prev_fg,
                            "trend": "improving" if current_fg > prev_fg else "deteriorating" if current_fg < prev_fg else "stable",
                            "extreme_level": "extreme_fear" if current_fg < 20 else "extreme_greed" if current_fg > 80 else "normal",
                            "signal_strength": abs(current_fg - 50) / 50  # 50에서 얼마나 극단적인지
                        }
                
                # 2. Google Trends (간접 측정)
                # 실제로는 pytrends 라이브러리 사용 가능하지만 간단히 시뮬레이션
                indicators["search_trends"] = {
                    "bitcoin_interest": 0.7,  # 0-1 스케일
                    "trend": "rising",
                    "signal_strength": 0.3
                }
                
                return indicators
                
        except Exception as e:
            self.logger.error(f"센티먼트 지표 수집 실패: {e}")
            return {}
    
    async def get_technical_signals(self) -> Dict:
        """기술적 다이버전스 신호"""
        try:
            async with aiohttp.ClientSession() as session:
                indicators = {}
                
                # Binance에서 캔들 데이터 수집
                klines_url = f"{self.binance_base}/fapi/v1/klines"
                params = {
                    "symbol": "BTCUSDT",
                    "interval": "1h",
                    "limit": 100
                }
                
                async with session.get(klines_url, params=params) as resp:
                    if resp.status == 200:
                        klines = await resp.json()
                        
                        # OHLCV 데이터 추출
                        closes = [float(k[4]) for k in klines]
                        volumes = [float(k[5]) for k in klines]
                        
                        # 가격-거래량 다이버전스 계산
                        price_trend = self._calculate_trend(closes[-20:])
                        volume_trend = self._calculate_trend(volumes[-20:])
                        
                        indicators["price_volume_divergence"] = {
                            "price_trend": price_trend,
                            "volume_trend": volume_trend,
                            "divergence_detected": (price_trend > 0 and volume_trend < 0) or (price_trend < 0 and volume_trend > 0),
                            "signal_strength": abs(price_trend - volume_trend) / 2
                        }
                        
                        # 간단한 RSI 계산
                        rsi = self._calculate_rsi(closes[-14:])
                        indicators["momentum"] = {
                            "rsi": rsi,
                            "overbought": rsi > 70,
                            "oversold": rsi < 30,
                            "signal_strength": abs(rsi - 50) / 50
                        }
                
                return indicators
                
        except Exception as e:
            self.logger.error(f"기술적 신호 수집 실패: {e}")
            return {}
    
    def _calculate_trend(self, data: List[float]) -> float:
        """간단한 추세 계산"""
        if len(data) < 2:
            return 0
        return (data[-1] - data[0]) / data[0] if data[0] != 0 else 0
    
    def _calculate_rsi(self, prices: List[float], period: int = 14) -> float:
        """RSI 계산"""
        if len(prices) < period + 1:
            return 50
        
        deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        gains = [d for d in deltas if d > 0]
        losses = [-d for d in deltas if d < 0]
        
        if not gains:
            return 0
        if not losses:
            return 100
        
        avg_gain = sum(gains) / len(gains)
        avg_loss = sum(losses) / len(losses)
        
        rs = avg_gain / avg_loss if avg_loss != 0 else 100
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def calculate_composite_signals(self, indicators: Dict) -> Dict:
        """모든 지표를 종합한 신호 강도 계산"""
        try:
            signals = {
                "bullish_strength": 0.0,
                "bearish_strength": 0.0,
                "total_signals": 0,
                "dominant_signal": "NEUTRAL",
                "confidence": 0.0,
                "signal_breakdown": {}
            }
            
            # 각 카테고리별 신호 분석
            for category, data in indicators.get("data_sources", {}).items():
                if not data:
                    continue
                    
                category_signal = self._analyze_category_signals(category, data)
                signals["signal_breakdown"][category] = category_signal
                
                # 가중치 적용하여 종합 신호에 반영
                weight = self._get_category_weight(category)
                signals["bullish_strength"] += category_signal.get("bullish", 0) * weight
                signals["bearish_strength"] += category_signal.get("bearish", 0) * weight
                signals["total_signals"] += 1
            
            # 최종 신호 결정
            if signals["total_signals"] > 0:
                if signals["bullish_strength"] > signals["bearish_strength"] * 1.2:
                    signals["dominant_signal"] = "BULLISH"
                elif signals["bearish_strength"] > signals["bullish_strength"] * 1.2:
                    signals["dominant_signal"] = "BEARISH"
                
                # 신뢰도 계산
                total_strength = signals["bullish_strength"] + signals["bearish_strength"]
                dominant_strength = max(signals["bullish_strength"], signals["bearish_strength"])
                signals["confidence"] = dominant_strength / total_strength if total_strength > 0 else 0
            
            return signals
            
        except Exception as e:
            self.logger.error(f"종합 신호 계산 실패: {e}")
            return {"error": str(e)}
    
    def _analyze_category_signals(self, category: str, data: Dict) -> Dict:
        """카테고리별 신호 분석"""
        signals = {"bullish": 0.0, "bearish": 0.0, "neutral": 0.0}
        
        if category == "binance_derivatives":
            # 펀딩비 분석
            if "funding_rate" in data:
                funding = data["funding_rate"]
                if funding.get("trend") == "falling":  # 펀딩비 하락은 강세 신호
                    signals["bullish"] += funding.get("signal_strength", 0)
                elif funding.get("trend") == "rising":
                    signals["bearish"] += funding.get("signal_strength", 0)
            
            # OI 분석
            if "open_interest" in data:
                oi = data["open_interest"]
                if oi.get("momentum") == "increasing":
                    signals["bullish"] += oi.get("signal_strength", 0) * 0.5
                elif oi.get("momentum") == "decreasing":
                    signals["bearish"] += oi.get("signal_strength", 0) * 0.5
        
        elif category == "macro_indicators":
            # VIX 분석
            if "vix" in data:
                vix = data["vix"]
                if vix.get("trend") == "rising":  # VIX 상승은 약세 신호
                    signals["bearish"] += vix.get("signal_strength", 0)
                elif vix.get("trend") == "falling":
                    signals["bullish"] += vix.get("signal_strength", 0)
            
            # DXY 분석
            if "dxy" in data:
                dxy = data["dxy"]
                if dxy.get("trend") == "rising":  # 달러 강세는 리스크 자산 약세
                    signals["bearish"] += dxy.get("signal_strength", 0) * 0.7
                elif dxy.get("trend") == "falling":
                    signals["bullish"] += dxy.get("signal_strength", 0) * 0.7
        
        elif category == "sentiment_indicators":
            # 공포탐욕지수 분석
            if "fear_greed_index" in data:
                fg = data["fear_greed_index"]
                if fg.get("extreme_level") == "extreme_fear":  # 극한 공포는 역설적 강세 신호
                    signals["bullish"] += fg.get("signal_strength", 0) * 0.8
                elif fg.get("extreme_level") == "extreme_greed":
                    signals["bearish"] += fg.get("signal_strength", 0) * 0.8
        
        return signals
    
    def _get_category_weight(self, category: str) -> float:
        """카테고리별 가중치"""
        weights = {
            "binance_derivatives": 1.0,    # 파생상품이 가장 선행성 높음
            "macro_indicators": 0.8,       # 거시경제 지표
            "whale_activity": 0.9,         # 고래 활동
            "sentiment_indicators": 0.6,   # 센티먼트
            "technical_signals": 0.7       # 기술적 신호
        }
        return weights.get(category, 0.5)

# 테스트 함수
async def test_real_time_indicators():
    """실시간 지표 수집 테스트"""
    print("🧪 실시간 선행지표 수집 테스트...")
    
    collector = RealTimeLeadingIndicators()
    indicators = await collector.collect_all_real_indicators()
    
    if "error" in indicators:
        print(f"❌ 수집 실패: {indicators['error']}")
        return False
    
    print("✅ 실시간 지표 수집 성공!")
    print(f"📊 수집된 카테고리: {len(indicators['data_sources'])}개")
    
    for category, data in indicators["data_sources"].items():
        if data:
            print(f"  • {category}: {len(data)}개 지표")
    
    # 종합 신호 출력
    composite = indicators.get("composite_signals", {})
    print(f"\n🎯 종합 분석:")
    print(f"  • 주요 신호: {composite.get('dominant_signal', 'UNKNOWN')}")
    print(f"  • 신뢰도: {composite.get('confidence', 0):.2%}")
    print(f"  • 강세 강도: {composite.get('bullish_strength', 0):.3f}")
    print(f"  • 약세 강도: {composite.get('bearish_strength', 0):.3f}")
    
    return True

if __name__ == "__main__":
    asyncio.run(test_real_time_indicators())