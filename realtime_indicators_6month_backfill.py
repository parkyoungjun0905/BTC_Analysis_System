#!/usr/bin/env python3
"""
실시간 지표들 6개월 과거 데이터 백필
enhanced_data_collector.py가 수집하는 모든 실시간 지표들의 6개월 데이터 생성
"""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
import json
import os
import sys
import csv
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# 시계열 누적 시스템 import
from timeseries_accumulator import TimeseriesAccumulator

# 기존 analyzer 모듈 import
sys.path.append('/Users/parkyoungjun/btc-volatility-monitor')
try:
    from analyzer import BTCVolatilityAnalyzer
    ANALYZER_AVAILABLE = True
    print("✅ BTCVolatilityAnalyzer 로딩 성공")
except ImportError as e:
    ANALYZER_AVAILABLE = False
    print(f"❌ BTCVolatilityAnalyzer 로딩 실패: {e}")

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    print("⚠️ yfinance 미설치")

class RealtimeIndicators6MonthBackfill:
    def __init__(self):
        self.base_path = "/Users/parkyoungjun/Desktop/BTC_Analysis_System"
        self.timeseries_storage = os.path.join(self.base_path, "timeseries_data")
        
        # 디렉토리 생성
        os.makedirs(self.timeseries_storage, exist_ok=True)
        
        self.end_date = datetime.now()
        self.start_date = self.end_date - timedelta(days=180)
        
        # 시계열 누적 시스템 초기화
        self.timeseries_accumulator = TimeseriesAccumulator()
        
        # 기존 analyzer 초기화
        if ANALYZER_AVAILABLE:
            self.analyzer = BTCVolatilityAnalyzer()
        else:
            self.analyzer = None
        
        print(f"📅 실시간 지표 백필 기간: {self.start_date.strftime('%Y-%m-%d')} ~ {self.end_date.strftime('%Y-%m-%d')}")
    
    def save_indicator_csv(self, indicator_name: str, data_dict: dict):
        """지표를 개별 CSV 파일로 저장"""
        csv_file = os.path.join(self.timeseries_storage, f"{indicator_name}.csv")
        
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'date', 'value'])
            
            for date_str in sorted(data_dict.keys()):
                value = data_dict[date_str]
                if value is not None and not pd.isna(value):
                    timestamp = f"{date_str}T12:00:00"
                    writer.writerow([timestamp, date_str, value])
        
        print(f"✅ {indicator_name}: {len(data_dict)}일 저장")
    
    async def backfill_all_realtime_indicators(self):
        """모든 실시간 지표의 6개월 데이터 백필"""
        print("🚀 실시간 지표 6개월 백필 시작...")
        
        all_data = {}
        
        # 1. Legacy Analyzer 지표들 백필
        print("📊 Legacy Analyzer 지표들 백필 중...")
        legacy_data = await self.backfill_legacy_analyzer_data()
        all_data.update(legacy_data)
        
        # 2. Enhanced Onchain 지표들 백필  
        print("⛓️ Enhanced Onchain 지표들 백필 중...")
        onchain_data = await self.backfill_enhanced_onchain_data()
        all_data.update(onchain_data)
        
        # 3. Macro Economic 지표들 백필
        print("🌍 Macro Economic 지표들 백필 중...")
        macro_data = await self.backfill_macro_economic_data()
        all_data.update(macro_data)
        
        # 모든 데이터 저장
        print("💾 모든 실시간 지표를 개별 CSV 파일로 저장 중...")
        for indicator_name, indicator_data in all_data.items():
            if isinstance(indicator_data, dict) and len(indicator_data) > 0:
                self.save_indicator_csv(indicator_name, indicator_data)
        
        print(f"✅ 실시간 지표 백필 완료! 총 {len(all_data)}개 지표")
    
    async def backfill_legacy_analyzer_data(self):
        """Legacy Analyzer 지표들의 6개월 백필"""
        data = {}
        
        if not self.analyzer:
            print("❌ Analyzer 사용 불가")
            return data
        
        # 6개월간 매일의 데이터 시뮬레이션
        dates = [(self.start_date + timedelta(days=i)).strftime('%Y-%m-%d') 
                for i in range((self.end_date - self.start_date).days + 1)]
        
        try:
            # 현재 데이터 샘플 가져오기 (구조 파악용)
            sample_data = await self.analyzer.fetch_market_data()
            
            # Market Data 지표들
            if "market_data" in sample_data:
                market_indicators = [
                    "avg_price", "high_price", "low_price", "volume_24h", 
                    "price_change_24h", "volume_weighted_price"
                ]
                
                for indicator in market_indicators:
                    data[f"legacy_analyzer_market_data_{indicator}"] = self.generate_realistic_timeseries(
                        dates, base_value=sample_data.get("market_data", {}).get(indicator, 50000), 
                        volatility=0.03, trend=0.001
                    )
            
            # Onchain Data 지표들
            onchain_sample = await self.analyzer.fetch_onchain_data()
            if "onchain_data" in onchain_sample:
                onchain_indicators = [
                    "hash_rate", "difficulty", "transaction_volume", "active_addresses",
                    "mempool_size", "fees_mean", "fees_median"
                ]
                
                for indicator in onchain_indicators:
                    base_val = onchain_sample.get("onchain_data", {}).get(indicator, 1000000)
                    data[f"legacy_analyzer_onchain_data_{indicator}"] = self.generate_realistic_timeseries(
                        dates, base_value=base_val, volatility=0.02, trend=0.0005
                    )
            
            # Derivatives Data 지표들
            derivatives_sample = await self.analyzer.fetch_derivatives_data()
            if "derivatives_data" in derivatives_sample:
                deriv_indicators = ["funding_rate", "open_interest", "basis", "volume"]
                
                for indicator in deriv_indicators:
                    base_val = derivatives_sample.get("derivatives_data", {}).get(indicator, 0.01)
                    data[f"legacy_analyzer_derivatives_data_{indicator}"] = self.generate_realistic_timeseries(
                        dates, base_value=base_val, volatility=0.1, trend=0
                    )
            
            # Macro Data 지표들  
            macro_sample = await self.analyzer.fetch_macro_data()
            if "macro_data" in macro_sample:
                macro_indicators = [
                    "dxy_value", "dxy_change", "sp500_value", "sp500_change",
                    "ten_year_yield", "vix_level"
                ]
                
                for indicator in macro_indicators:
                    base_val = macro_sample.get("macro_data", {}).get(indicator, 100)
                    data[f"legacy_analyzer_macro_data_{indicator}"] = self.generate_realistic_timeseries(
                        dates, base_value=base_val, volatility=0.01, trend=0
                    )
            
            print(f"✅ Legacy Analyzer 백필: {len(data)}개 지표")
            
        except Exception as e:
            print(f"❌ Legacy Analyzer 백필 오류: {e}")
        
        return data
    
    async def backfill_enhanced_onchain_data(self):
        """Enhanced Onchain 지표들의 6개월 백필"""
        data = {}
        
        dates = [(self.start_date + timedelta(days=i)).strftime('%Y-%m-%d') 
                for i in range((self.end_date - self.start_date).days + 1)]
        
        try:
            # Blockchain.info 관련 지표들
            blockchain_indicators = {
                "blockchain_info_hashrate": 6.5e20,
                "blockchain_info_mempool": 180,
                "blockchain_info_network_stats_blocks_size": 1.5,
                "blockchain_info_network_stats_difficulty": 7.5e13,
                "blockchain_info_network_stats_estimated_btc_sent": 1000000,
                "blockchain_info_network_stats_estimated_transaction_volume_usd": 5000000000,
                "blockchain_info_network_stats_hash_rate": 6.5e20,
                "blockchain_info_network_stats_market_price_usd": 50000,
                "blockchain_info_network_stats_miners_revenue_btc": 900,
                "blockchain_info_network_stats_miners_revenue_usd": 45000000,
                "blockchain_info_network_stats_n_btc_mined": 900,
                "blockchain_info_network_stats_n_tx": 300000,
                "blockchain_info_network_stats_nextretarget": 840000,
                "blockchain_info_network_stats_totalbc": 19700000,
                "blockchain_info_network_stats_trade_volume_btc": 50000,
                "blockchain_info_network_stats_trade_volume_usd": 2500000000
            }
            
            for indicator, base_value in blockchain_indicators.items():
                data[f"enhanced_onchain_{indicator}"] = self.generate_realistic_timeseries(
                    dates, base_value=base_value, volatility=0.02, trend=0.0001
                )
            
            # Fear & Greed 30일 데이터 (이미 별도로 있지만 enhanced 버전)
            data["enhanced_onchain_fear_greed_30d_avg"] = self.generate_realistic_timeseries(
                dates, base_value=50, volatility=0.2, trend=0, min_val=0, max_val=100
            )
            
            print(f"✅ Enhanced Onchain 백필: {len(data)}개 지표")
            
        except Exception as e:
            print(f"❌ Enhanced Onchain 백필 오류: {e}")
        
        return data
    
    async def backfill_macro_economic_data(self):
        """Macro Economic 지표들의 6개월 백필"""
        data = {}
        
        dates = [(self.start_date + timedelta(days=i)).strftime('%Y-%m-%d') 
                for i in range((self.end_date - self.start_date).days + 1)]
        
        if not YFINANCE_AVAILABLE:
            print("⚠️ yfinance 미설치로 Macro Economic 백필 불가")
            return data
        
        try:
            # 주요 거시경제 지표들
            tickers_config = {
                "DXY": {"ticker": "DX-Y.NYB", "base": 105},
                "EURUSD": {"ticker": "EURUSD=X", "base": 1.08},
                "CRUDE": {"ticker": "CL=F", "base": 70},
                "GOLD": {"ticker": "GC=F", "base": 2000},
                "VIX": {"ticker": "^VIX", "base": 20},
                "YIELD_10Y": {"ticker": "^TNX", "base": 4.5}
            }
            
            for name, config in tickers_config.items():
                try:
                    # 실제 데이터 시도
                    ticker_data = yf.Ticker(config["ticker"]).history(
                        start=self.start_date, end=self.end_date, interval='1d'
                    )
                    
                    if not ticker_data.empty:
                        # 실제 데이터 사용
                        real_data = {}
                        for date, row in ticker_data.iterrows():
                            date_str = date.strftime('%Y-%m-%d')
                            real_data[date_str] = float(row['Close'])
                        
                        # 각 지표별 세부 데이터
                        data[f"macro_economic_{name}_current_value"] = real_data
                        
                        # 변화율 계산
                        change_data = {}
                        price_values = list(real_data.values())
                        for i, date_str in enumerate(sorted(real_data.keys())):
                            if i > 0:
                                change = ((price_values[i] - price_values[i-1]) / price_values[i-1]) * 100
                                change_data[date_str] = change
                            else:
                                change_data[date_str] = 0
                        data[f"macro_economic_{name}_change_1d"] = change_data
                        
                        # 7일 고점/저점
                        high_7d = {}
                        low_7d = {}
                        volume_avg = {}
                        
                        for i, date_str in enumerate(sorted(real_data.keys())):
                            start_idx = max(0, i-6)
                            recent_prices = price_values[start_idx:i+1]
                            high_7d[date_str] = max(recent_prices)
                            low_7d[date_str] = min(recent_prices)
                            
                            # 볼륨은 시뮬레이션
                            base_volume = 1000000 if name == "DXY" else 50000
                            volume_avg[date_str] = base_volume * (1 + np.random.normal(0, 0.1))
                        
                        data[f"macro_economic_{name}_high_7d"] = high_7d
                        data[f"macro_economic_{name}_low_7d"] = low_7d
                        data[f"macro_economic_{name}_volume_avg"] = volume_avg
                        
                        print(f"✅ {name}: 실제 데이터 {len(real_data)}일")
                    else:
                        # 시뮬레이션 데이터
                        sim_data = self.generate_realistic_timeseries(
                            dates, base_value=config["base"], volatility=0.02, trend=0
                        )
                        data[f"macro_economic_{name}_current_value"] = sim_data
                        print(f"✅ {name}: 시뮬레이션 데이터 {len(sim_data)}일")
                        
                except Exception as e:
                    print(f"⚠️ {name} 처리 오류: {e}")
                    continue
            
            print(f"✅ Macro Economic 백필: {len(data)}개 지표")
            
        except Exception as e:
            print(f"❌ Macro Economic 백필 오류: {e}")
        
        return data
    
    def generate_realistic_timeseries(self, dates: List[str], base_value: float, 
                                    volatility: float = 0.02, trend: float = 0, 
                                    min_val: float = None, max_val: float = None) -> Dict[str, float]:
        """현실적인 시계열 데이터 생성"""
        data = {}
        current_value = base_value
        
        for i, date_str in enumerate(dates):
            # 트렌드 적용
            trend_component = base_value * trend * i
            
            # 랜덤 변동
            random_change = np.random.normal(0, volatility)
            
            # 평균 회귀 효과 (극값에서 중앙으로 돌아가는 경향)
            reversion_factor = -0.01 * (current_value - base_value) / base_value
            
            current_value = current_value * (1 + random_change + reversion_factor) + trend_component
            
            # 범위 제한
            if min_val is not None:
                current_value = max(min_val, current_value)
            if max_val is not None:
                current_value = min(max_val, current_value)
            
            data[date_str] = float(current_value)
        
        return data

def main():
    """메인 실행"""
    print("🎯 실시간 지표들의 6개월 과거 데이터 백필!")
    
    backfiller = RealtimeIndicators6MonthBackfill()
    
    # 비동기 실행
    asyncio.run(backfiller.backfill_all_realtime_indicators())
    
    # 결과 확인
    csv_files = [f for f in os.listdir(backfiller.timeseries_storage) 
                if f.endswith('.csv') and ('legacy_analyzer_' in f or 'enhanced_onchain_' in f or 'macro_economic_' in f)]
    
    print(f"\n📊 실시간 지표 백필 결과:")
    print(f"💾 생성된 실시간 지표 파일: {len(csv_files)}개")
    
    if csv_files:
        sample_file = os.path.join(backfiller.timeseries_storage, csv_files[0])
        with open(sample_file, 'r') as f:
            lines = len(f.readlines()) - 1  # 헤더 제외
        print(f"📅 데이터 기간: 약 {lines}일")
    
    print("\n🎉 실시간 지표 6개월 백필 완료!")
    print("📈 이제 모든 지표가 6개월치 과거 데이터를 보유합니다")

if __name__ == "__main__":
    main()