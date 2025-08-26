#!/usr/bin/env python3
"""
무료 API 기반 실시간 데이터 수집기
비용 최소화를 위해 무료 소스 우선 활용
"""

import asyncio
import aiohttp
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional
import time
from config import DATA_SOURCES, RISK_THRESHOLDS

class FreeDataCollector:
    def __init__(self):
        self.session = None
        self.logger = logging.getLogger(__name__)
        self.rate_limits = {}  # API별 속도 제한 추적
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def collect_all_data(self) -> Dict:
        """모든 무료 데이터 소스에서 데이터 수집"""
        try:
            data = {
                "timestamp": datetime.utcnow().isoformat(),
                "price_data": {},
                "volume_data": {},
                "derivatives_data": {},
                "macro_data": {},
                "sentiment_data": {},
                "onchain_data": {}
            }
            
            # 병렬로 데이터 수집
            tasks = [
                self.get_coingecko_data(),
                self.get_fear_greed_data(), 
                self.get_macro_data(),
                self.get_alternative_data(),
                self.get_derived_metrics()
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 결과 통합
            for i, result in enumerate(results):
                if not isinstance(result, Exception):
                    data = self.merge_data(data, result)
                else:
                    self.logger.error(f"데이터 수집 오류 {i}: {result}")
                    
            return data
            
        except Exception as e:
            self.logger.error(f"전체 데이터 수집 실패: {e}")
            return {"error": str(e), "timestamp": datetime.utcnow().isoformat()}

    async def get_coingecko_data(self) -> Dict:
        """CoinGecko 무료 API 데이터 수집"""
        try:
            base_url = DATA_SOURCES["coingecko"]["base_url"]
            
            # 현재 가격 및 기본 데이터
            price_url = f"{base_url}/simple/price"
            params = {
                "ids": "bitcoin",
                "vs_currencies": "usd",
                "include_24hr_vol": "true",
                "include_24hr_change": "true",
                "include_market_cap": "true",
                "include_last_updated_at": "true"
            }
            
            async with self.session.get(price_url, params=params) as resp:
                price_data = await resp.json()
                
            # 시장 차트 데이터 (최근 1일)
            chart_url = f"{base_url}/coins/bitcoin/market_chart"
            chart_params = {
                "vs_currency": "usd",
                "days": "1",
                "interval": "minutely"
            }
            
            async with self.session.get(chart_url, params=chart_params) as resp:
                chart_data = await resp.json()
            
            return {
                "source": "coingecko",
                "price_data": {
                    "current_price": price_data["bitcoin"]["usd"],
                    "market_cap": price_data["bitcoin"]["usd_market_cap"],
                    "volume_24h": price_data["bitcoin"]["usd_24h_vol"],
                    "change_24h": price_data["bitcoin"]["usd_24h_change"],
                    "last_updated": price_data["bitcoin"]["last_updated_at"]
                },
                "chart_data": chart_data,
                "collected_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"CoinGecko 데이터 수집 실패: {e}")
            return {"source": "coingecko", "error": str(e)}

    async def get_fear_greed_data(self) -> Dict:
        """공포탐욕지수 데이터 수집"""
        try:
            url = f"{DATA_SOURCES['alternative_me']['base_url']}/fng/"
            params = {"limit": "30", "format": "json"}
            
            async with self.session.get(url, params=params) as resp:
                data = await resp.json()
                
            return {
                "source": "fear_greed",
                "current_index": int(data["data"][0]["value"]),
                "classification": data["data"][0]["value_classification"],
                "timestamp": data["data"][0]["timestamp"],
                "historical": data["data"][:7],  # 최근 7일
                "collected_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"공포탐욕지수 수집 실패: {e}")
            return {"source": "fear_greed", "error": str(e)}

    async def get_macro_data(self) -> Dict:
        """거시경제 데이터 수집 (무료 소스)"""
        try:
            # Yahoo Finance 무료 API 대안 사용
            macro_data = {}
            
            # VIX 지수 (대체 소스)
            try:
                vix_url = "https://query1.finance.yahoo.com/v8/finance/chart/^VIX"
                async with self.session.get(vix_url) as resp:
                    vix_data = await resp.json()
                    if "chart" in vix_data and "result" in vix_data["chart"]:
                        result = vix_data["chart"]["result"][0]
                        macro_data["vix"] = {
                            "current": result["meta"]["regularMarketPrice"],
                            "change": result["meta"]["regularMarketPrice"] - result["meta"]["previousClose"],
                            "change_percent": ((result["meta"]["regularMarketPrice"] - result["meta"]["previousClose"]) / result["meta"]["previousClose"]) * 100
                        }
            except Exception as e:
                self.logger.warning(f"VIX 데이터 수집 실패: {e}")
                
            # DXY 지수
            try:
                dxy_url = "https://query1.finance.yahoo.com/v8/finance/chart/DX-Y.NYB"
                async with self.session.get(dxy_url) as resp:
                    dxy_data = await resp.json()
                    if "chart" in dxy_data and "result" in dxy_data["chart"]:
                        result = dxy_data["chart"]["result"][0]
                        macro_data["dxy"] = {
                            "current": result["meta"]["regularMarketPrice"],
                            "change": result["meta"]["regularMarketPrice"] - result["meta"]["previousClose"],
                            "change_percent": ((result["meta"]["regularMarketPrice"] - result["meta"]["previousClose"]) / result["meta"]["previousClose"]) * 100
                        }
            except Exception as e:
                self.logger.warning(f"DXY 데이터 수집 실패: {e}")
                
            return {
                "source": "macro",
                "data": macro_data,
                "collected_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"거시경제 데이터 수집 실패: {e}")
            return {"source": "macro", "error": str(e)}

    async def get_alternative_data(self) -> Dict:
        """대체 데이터 소스들"""
        try:
            alt_data = {}
            
            # Blockchain.info 무료 API
            try:
                blockchain_url = "https://api.blockchain.info/stats"
                async with self.session.get(blockchain_url) as resp:
                    blockchain_data = await resp.json()
                    alt_data["blockchain_info"] = {
                        "hash_rate": blockchain_data.get("hash_rate", 0),
                        "difficulty": blockchain_data.get("difficulty", 0),
                        "total_btc": blockchain_data.get("totalbc", 0) / 100000000,  # Satoshi to BTC
                        "market_price_usd": blockchain_data.get("market_price_usd", 0)
                    }
            except Exception as e:
                self.logger.warning(f"Blockchain.info 데이터 수집 실패: {e}")
                
            return {
                "source": "alternative",
                "data": alt_data,
                "collected_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"대체 데이터 수집 실패: {e}")
            return {"source": "alternative", "error": str(e)}

    async def get_derived_metrics(self) -> Dict:
        """기존 데이터로부터 파생 지표 계산"""
        try:
            # 여기서는 간단한 파생 지표만 계산
            # 실제로는 수집된 다른 데이터를 활용
            
            derived_data = {
                "timestamp": datetime.utcnow().isoformat(),
                "calculated_metrics": {
                    "data_collection_latency": 0,  # 나중에 계산
                    "api_success_rate": 1.0,       # 나중에 계산
                    "last_update_lag": 0           # 나중에 계산
                }
            }
            
            return {
                "source": "derived",
                "data": derived_data,
                "collected_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"파생 지표 계산 실패: {e}")
            return {"source": "derived", "error": str(e)}

    def merge_data(self, main_data: Dict, new_data: Dict) -> Dict:
        """수집된 데이터들을 통합"""
        try:
            source = new_data.get("source", "unknown")
            
            if source == "coingecko" and "price_data" in new_data:
                main_data["price_data"].update(new_data["price_data"])
                if "chart_data" in new_data:
                    main_data["chart_data"] = new_data["chart_data"]
                    
            elif source == "fear_greed":
                main_data["sentiment_data"]["fear_greed"] = new_data
                
            elif source == "macro":
                main_data["macro_data"].update(new_data.get("data", {}))
                
            elif source == "alternative":
                main_data["onchain_data"].update(new_data.get("data", {}))
                
            elif source == "derived":
                main_data["derived_metrics"] = new_data.get("data", {})
                
            return main_data
            
        except Exception as e:
            self.logger.error(f"데이터 병합 실패: {e}")
            return main_data

    def calculate_immediate_risk_indicators(self, data: Dict) -> Dict:
        """수집된 데이터로부터 즉시 위험 지표 계산"""
        try:
            risk_indicators = {
                "timestamp": datetime.utcnow().isoformat(),
                "price_risk": 0,
                "volume_risk": 0,
                "sentiment_risk": 0,
                "macro_risk": 0,
                "composite_risk": 0
            }
            
            # 가격 위험도
            if "price_data" in data and "change_24h" in data["price_data"]:
                change_24h = abs(data["price_data"]["change_24h"])
                if change_24h > 10:
                    risk_indicators["price_risk"] = 1.0
                elif change_24h > 5:
                    risk_indicators["price_risk"] = 0.7
                elif change_24h > 3:
                    risk_indicators["price_risk"] = 0.4
                    
            # 센티먼트 위험도
            if "sentiment_data" in data and "fear_greed" in data["sentiment_data"]:
                fg_index = data["sentiment_data"]["fear_greed"]["current_index"]
                if fg_index > 80 or fg_index < 20:
                    risk_indicators["sentiment_risk"] = 0.8
                elif fg_index > 70 or fg_index < 30:
                    risk_indicators["sentiment_risk"] = 0.5
                    
            # VIX 위험도
            if "macro_data" in data and "vix" in data["macro_data"]:
                vix_level = data["macro_data"]["vix"]["current"]
                if vix_level > 30:
                    risk_indicators["macro_risk"] = 0.9
                elif vix_level > 25:
                    risk_indicators["macro_risk"] = 0.6
                elif vix_level > 20:
                    risk_indicators["macro_risk"] = 0.3
                    
            # 종합 위험도 (가중평균)
            weights = {"price": 0.4, "volume": 0.2, "sentiment": 0.2, "macro": 0.2}
            risk_indicators["composite_risk"] = (
                risk_indicators["price_risk"] * weights["price"] +
                risk_indicators["volume_risk"] * weights["volume"] +
                risk_indicators["sentiment_risk"] * weights["sentiment"] +
                risk_indicators["macro_risk"] * weights["macro"]
            )
            
            return risk_indicators
            
        except Exception as e:
            self.logger.error(f"위험 지표 계산 실패: {e}")
            return {"error": str(e), "timestamp": datetime.utcnow().isoformat()}

# 테스트 함수
async def test_data_collection():
    """데이터 수집 테스트"""
    print("🔄 데이터 수집 테스트 시작...")
    
    async with FreeDataCollector() as collector:
        data = await collector.collect_all_data()
        risk_indicators = collector.calculate_immediate_risk_indicators(data)
        
        print("✅ 수집된 데이터:")
        for key, value in data.items():
            if key != "chart_data":  # 차트 데이터는 너무 길어서 제외
                print(f"  {key}: {type(value)}")
                
        print("\n✅ 위험 지표:")
        for key, value in risk_indicators.items():
            print(f"  {key}: {value}")
            
        return data, risk_indicators

if __name__ == "__main__":
    # 직접 실행 시 테스트
    asyncio.run(test_data_collection())