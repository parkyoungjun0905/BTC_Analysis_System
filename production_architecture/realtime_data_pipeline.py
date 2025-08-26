#!/usr/bin/env python3
"""
🌊 실시간 데이터 처리 파이프라인
- Apache Kafka 기반 스트리밍 데이터 처리
- 실시간 특성 엔지니어링
- 저지연 데이터 전달 (<5초)
- 자동 품질 검증 및 이상치 탐지
- 백프레셔 제어 및 오류 복구
"""

import asyncio
import json
import logging
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, asdict
import hashlib
from pathlib import Path
import queue
import threading
from concurrent.futures import ThreadPoolExecutor

# Apache Kafka
try:
    from kafka import KafkaProducer, KafkaConsumer
    from kafka.errors import KafkaError
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False
    print("⚠️ Kafka not available, using fallback message queue")

# Redis for caching and state management
import redis

# Data processing
from scipy import stats
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest, f_regression

# API clients
import requests
import websocket
import ccxt

# Monitoring
from prometheus_client import Counter, Histogram, Gauge

@dataclass
class MarketData:
    """시장 데이터 구조"""
    symbol: str
    timestamp: datetime
    price: float
    volume: float
    bid: Optional[float] = None
    ask: Optional[float] = None
    source: str = "unknown"
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class ProcessedFeatures:
    """처리된 특성 데이터"""
    symbol: str
    timestamp: datetime
    features: Dict[str, float]
    technical_indicators: Dict[str, float]
    onchain_metrics: Dict[str, float]
    macro_data: Dict[str, float]
    quality_score: float
    processing_latency_ms: int

class DataQualityValidator:
    """데이터 품질 검증기"""
    
    def __init__(self):
        self.price_bounds = {"min": 10000, "max": 200000}  # BTC 가격 범위
        self.volume_bounds = {"min": 0, "max": 1e9}  # 거래량 범위
        self.zscore_threshold = 3.0  # 이상치 탐지 임계값
        self.price_history = queue.deque(maxlen=1000)  # 최근 가격 이력
        
    def validate_market_data(self, data: MarketData) -> tuple[bool, List[str]]:
        """시장 데이터 검증"""
        errors = []
        
        # 1. 기본 범위 검증
        if not (self.price_bounds["min"] <= data.price <= self.price_bounds["max"]):
            errors.append(f"Price {data.price} out of bounds")
            
        if not (self.volume_bounds["min"] <= data.volume <= self.volume_bounds["max"]):
            errors.append(f"Volume {data.volume} out of bounds")
            
        # 2. 시간 검증 (너무 오래된 데이터)
        time_diff = datetime.now() - data.timestamp
        if time_diff.total_seconds() > 300:  # 5분 이상 지연
            errors.append(f"Data too old: {time_diff}")
            
        # 3. 가격 급변 감지
        if len(self.price_history) > 10:
            recent_prices = list(self.price_history)[-10:]
            price_changes = np.diff(recent_prices) / recent_prices[:-1]
            
            if abs(price_changes[-1]) > 0.1:  # 10% 이상 급변
                errors.append(f"Extreme price change: {price_changes[-1]:.2%}")
                
        # 4. Z-score 이상치 탐지
        if len(self.price_history) > 30:
            prices_array = np.array(list(self.price_history))
            zscore = abs(stats.zscore(np.append(prices_array, data.price))[-1])
            
            if zscore > self.zscore_threshold:
                errors.append(f"Price Z-score too high: {zscore:.2f}")
                
        # 가격 이력 업데이트
        self.price_history.append(data.price)
        
        return len(errors) == 0, errors

class RealTimeDataCollector:
    """실시간 데이터 수집기"""
    
    def __init__(self):
        self.setup_logging()
        self.setup_metrics()
        self.exchanges = {}
        self.websocket_connections = {}
        self.data_validator = DataQualityValidator()
        self.setup_exchanges()
        
    def setup_logging(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def setup_metrics(self):
        """메트릭 설정"""
        self.data_points_collected = Counter(
            'data_points_collected_total',
            'Total data points collected',
            ['source', 'symbol']
        )
        
        self.data_collection_latency = Histogram(
            'data_collection_latency_seconds',
            'Data collection latency',
            ['source']
        )
        
        self.data_quality_score = Gauge(
            'data_quality_score',
            'Data quality score',
            ['source']
        )
        
    def setup_exchanges(self):
        """거래소 API 설정"""
        try:
            # Binance
            self.exchanges['binance'] = ccxt.binance({
                'apiKey': '',  # 실제 환경에서는 환경변수에서 로드
                'secret': '',
                'sandbox': False,
                'enableRateLimit': True
            })
            
            # Coinbase Pro
            self.exchanges['coinbasepro'] = ccxt.coinbasepro({
                'enableRateLimit': True
            })
            
            self.logger.info(f"Exchanges initialized: {list(self.exchanges.keys())}")
            
        except Exception as e:
            self.logger.error(f"Exchange setup failed: {e}")
            
    async def collect_price_data(self) -> AsyncGenerator[MarketData, None]:
        """실시간 가격 데이터 수집"""
        while True:
            try:
                for exchange_name, exchange in self.exchanges.items():
                    start_time = time.time()
                    
                    try:
                        # BTC/USDT 가격 조회
                        ticker = exchange.fetch_ticker('BTC/USDT')
                        
                        market_data = MarketData(
                            symbol='BTC/USDT',
                            timestamp=datetime.fromtimestamp(ticker['timestamp'] / 1000),
                            price=ticker['last'],
                            volume=ticker['baseVolume'],
                            bid=ticker['bid'],
                            ask=ticker['ask'],
                            source=exchange_name,
                            metadata={
                                'high': ticker['high'],
                                'low': ticker['low'],
                                'open': ticker['open']
                            }
                        )
                        
                        # 데이터 검증
                        is_valid, errors = self.data_validator.validate_market_data(market_data)
                        
                        if is_valid:
                            # 메트릭 업데이트
                            self.data_points_collected.labels(
                                source=exchange_name,
                                symbol='BTC/USDT'
                            ).inc()
                            
                            self.data_collection_latency.labels(
                                source=exchange_name
                            ).observe(time.time() - start_time)
                            
                            yield market_data
                            
                        else:
                            self.logger.warning(
                                f"Invalid data from {exchange_name}: {errors}"
                            )
                            
                    except Exception as e:
                        self.logger.error(f"Error collecting from {exchange_name}: {e}")
                        
                await asyncio.sleep(1)  # 1초 간격
                
            except Exception as e:
                self.logger.error(f"Collection loop error: {e}")
                await asyncio.sleep(5)

class FeatureProcessor:
    """실시간 특성 처리기"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.price_buffer = queue.deque(maxlen=1000)  # 최근 1000개 가격
        self.volume_buffer = queue.deque(maxlen=1000)
        self.feature_cache = {}
        self.scaler = RobustScaler()
        
        # 기술적 지표 계산을 위한 파라미터
        self.ta_params = {
            'sma_periods': [5, 10, 20, 50, 200],
            'ema_periods': [12, 26, 50],
            'rsi_periods': [14, 21],
            'bb_periods': [20, 50]
        }
        
    def process_features(self, market_data: MarketData) -> ProcessedFeatures:
        """실시간 특성 처리"""
        start_time = time.time()
        
        # 버퍼 업데이트
        self.price_buffer.append(market_data.price)
        self.volume_buffer.append(market_data.volume)
        
        # 특성 계산
        features = {}
        technical_indicators = {}
        
        try:
            # 1. 기본 특성
            features.update(self._calculate_basic_features(market_data))
            
            # 2. 기술적 지표
            technical_indicators.update(self._calculate_technical_indicators())
            
            # 3. 거래량 지표
            features.update(self._calculate_volume_features())
            
            # 4. 가격 패턴
            features.update(self._calculate_price_patterns())
            
            # 5. 변동성 지표
            features.update(self._calculate_volatility_features())
            
            # 품질 점수 계산
            quality_score = self._calculate_quality_score(features, technical_indicators)
            
            processing_time_ms = int((time.time() - start_time) * 1000)
            
            return ProcessedFeatures(
                symbol=market_data.symbol,
                timestamp=market_data.timestamp,
                features=features,
                technical_indicators=technical_indicators,
                onchain_metrics={},  # 별도 처리
                macro_data={},  # 별도 처리
                quality_score=quality_score,
                processing_latency_ms=processing_time_ms
            )
            
        except Exception as e:
            self.logger.error(f"Feature processing failed: {e}")
            return self._create_fallback_features(market_data)
            
    def _calculate_basic_features(self, market_data: MarketData) -> Dict[str, float]:
        """기본 특성 계산"""
        features = {
            'price': market_data.price,
            'volume': market_data.volume,
            'timestamp_hour': market_data.timestamp.hour,
            'timestamp_day_of_week': market_data.timestamp.weekday(),
            'timestamp_minute': market_data.timestamp.minute
        }
        
        if market_data.bid and market_data.ask:
            features['spread'] = market_data.ask - market_data.bid
            features['spread_pct'] = features['spread'] / market_data.price
            features['mid_price'] = (market_data.bid + market_data.ask) / 2
            
        return features
        
    def _calculate_technical_indicators(self) -> Dict[str, float]:
        """기술적 지표 계산"""
        if len(self.price_buffer) < 50:
            return {}
            
        prices = np.array(list(self.price_buffer))
        indicators = {}
        
        try:
            # 이동평균
            for period in self.ta_params['sma_periods']:
                if len(prices) >= period:
                    indicators[f'sma_{period}'] = np.mean(prices[-period:])
                    
            # 지수이동평균
            for period in self.ta_params['ema_periods']:
                if len(prices) >= period:
                    alpha = 2 / (period + 1)
                    ema = prices[0]
                    for price in prices[1:]:
                        ema = alpha * price + (1 - alpha) * ema
                    indicators[f'ema_{period}'] = ema
                    
            # RSI
            for period in self.ta_params['rsi_periods']:
                if len(prices) >= period + 1:
                    rsi = self._calculate_rsi(prices, period)
                    indicators[f'rsi_{period}'] = rsi
                    
            # 볼린저 밴드
            for period in self.ta_params['bb_periods']:
                if len(prices) >= period:
                    sma = np.mean(prices[-period:])
                    std = np.std(prices[-period:])
                    indicators[f'bb_upper_{period}'] = sma + 2 * std
                    indicators[f'bb_lower_{period}'] = sma - 2 * std
                    indicators[f'bb_position_{period}'] = (prices[-1] - indicators[f'bb_lower_{period}']) / (indicators[f'bb_upper_{period}'] - indicators[f'bb_lower_{period}'])
                    
        except Exception as e:
            self.logger.error(f"Technical indicator calculation failed: {e}")
            
        return indicators
        
    def _calculate_rsi(self, prices: np.ndarray, period: int) -> float:
        """RSI 계산"""
        deltas = np.diff(prices)
        gains = deltas.copy()
        losses = deltas.copy()
        gains[gains < 0] = 0
        losses[losses > 0] = 0
        losses = -losses
        
        if len(gains) >= period:
            avg_gain = np.mean(gains[-period:])
            avg_loss = np.mean(losses[-period:])
            
            if avg_loss == 0:
                return 100
                
            rs = avg_gain / avg_loss
            return 100 - (100 / (1 + rs))
            
        return 50  # 중립값
        
    def _calculate_volume_features(self) -> Dict[str, float]:
        """거래량 특성 계산"""
        if len(self.volume_buffer) < 20:
            return {}
            
        volumes = np.array(list(self.volume_buffer))
        
        return {
            'volume_sma_20': np.mean(volumes[-20:]),
            'volume_ratio': volumes[-1] / np.mean(volumes[-20:]) if np.mean(volumes[-20:]) > 0 else 1,
            'volume_std_20': np.std(volumes[-20:]),
            'volume_trend': (volumes[-1] - volumes[-5]) / volumes[-5] if volumes[-5] > 0 else 0
        }
        
    def _calculate_price_patterns(self) -> Dict[str, float]:
        """가격 패턴 특성"""
        if len(self.price_buffer) < 10:
            return {}
            
        prices = np.array(list(self.price_buffer))
        
        return {
            'price_change_1': (prices[-1] - prices[-2]) / prices[-2] if len(prices) > 1 else 0,
            'price_change_5': (prices[-1] - prices[-6]) / prices[-6] if len(prices) > 5 else 0,
            'price_change_20': (prices[-1] - prices[-21]) / prices[-21] if len(prices) > 20 else 0,
            'price_momentum': np.corrcoef(range(min(10, len(prices))), prices[-min(10, len(prices)):]))[0, 1] if len(prices) > 2 else 0
        }
        
    def _calculate_volatility_features(self) -> Dict[str, float]:
        """변동성 특성"""
        if len(self.price_buffer) < 20:
            return {}
            
        prices = np.array(list(self.price_buffer))
        returns = np.diff(prices) / prices[:-1]
        
        return {
            'volatility_1h': np.std(returns[-60:]) if len(returns) >= 60 else np.std(returns),
            'volatility_4h': np.std(returns[-240:]) if len(returns) >= 240 else np.std(returns),
            'volatility_24h': np.std(returns) if len(returns) >= 1440 else np.std(returns),
            'volatility_rank': stats.percentileofscore(returns, returns[-1]) if len(returns) > 10 else 50
        }
        
    def _calculate_quality_score(self, features: Dict[str, float], indicators: Dict[str, float]) -> float:
        """데이터 품질 점수"""
        try:
            # 특성 완성도
            expected_features = 30
            actual_features = len(features) + len(indicators)
            completeness = min(1.0, actual_features / expected_features)
            
            # NaN 값 비율
            all_values = list(features.values()) + list(indicators.values())
            nan_ratio = sum(1 for v in all_values if np.isnan(v) or np.isinf(v)) / len(all_values) if all_values else 0
            
            # 최종 품질 점수
            quality = completeness * (1 - nan_ratio) * 0.9  # 최대 90%
            
            return max(0.0, min(1.0, quality))
            
        except Exception:
            return 0.5
            
    def _create_fallback_features(self, market_data: MarketData) -> ProcessedFeatures:
        """폴백 특성 생성"""
        return ProcessedFeatures(
            symbol=market_data.symbol,
            timestamp=market_data.timestamp,
            features={'price': market_data.price, 'volume': market_data.volume},
            technical_indicators={},
            onchain_metrics={},
            macro_data={},
            quality_score=0.3,
            processing_latency_ms=0
        )

class StreamProcessor:
    """스트림 데이터 처리 엔진"""
    
    def __init__(self, kafka_config: Dict[str, str] = None):
        self.kafka_config = kafka_config or {
            'bootstrap_servers': 'localhost:9092',
            'client_id': 'btc-stream-processor'
        }
        self.setup_logging()
        self.setup_kafka()
        self.setup_redis()
        
        self.data_collector = RealTimeDataCollector()
        self.feature_processor = FeatureProcessor()
        
        self.processing_stats = {
            'messages_processed': 0,
            'errors': 0,
            'start_time': time.time()
        }
        
    def setup_logging(self):
        self.logger = logging.getLogger(__name__)
        
    def setup_kafka(self):
        """Kafka 설정"""
        if not KAFKA_AVAILABLE:
            self.producer = None
            self.consumer = None
            self.logger.warning("Kafka not available, using fallback")
            return
            
        try:
            self.producer = KafkaProducer(
                **self.kafka_config,
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                compression_type='gzip',
                batch_size=16384,
                linger_ms=100  # 배치 최적화
            )
            
            self.consumer = KafkaConsumer(
                'btc-market-data',
                **self.kafka_config,
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                auto_offset_reset='latest',
                enable_auto_commit=True,
                group_id='btc-feature-processor'
            )
            
            self.logger.info("Kafka setup completed")
            
        except Exception as e:
            self.logger.error(f"Kafka setup failed: {e}")
            self.producer = None
            self.consumer = None
            
    def setup_redis(self):
        """Redis 설정"""
        try:
            self.redis_client = redis.Redis(
                host='localhost',
                port=6379,
                db=0,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5
            )
            
            # 연결 테스트
            self.redis_client.ping()
            self.logger.info("Redis setup completed")
            
        except Exception as e:
            self.logger.error(f"Redis setup failed: {e}")
            self.redis_client = None
            
    async def run_pipeline(self):
        """실시간 파이프라인 실행"""
        self.logger.info("Starting real-time data pipeline")
        
        try:
            # 동시 실행: 데이터 수집, 처리, 전송
            await asyncio.gather(
                self.data_collection_loop(),
                self.feature_processing_loop(),
                self.monitoring_loop(),
                return_exceptions=True
            )
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            raise
            
    async def data_collection_loop(self):
        """데이터 수집 루프"""
        self.logger.info("Starting data collection loop")
        
        async for market_data in self.data_collector.collect_price_data():
            try:
                # Kafka로 원시 데이터 전송
                if self.producer:
                    message = {
                        'type': 'market_data',
                        'data': asdict(market_data),
                        'timestamp': time.time()
                    }
                    
                    self.producer.send('btc-market-data', message)
                    
                # Redis 캐시에 최신 데이터 저장
                if self.redis_client:
                    cache_key = f"market_data:{market_data.symbol}"
                    cache_data = {
                        'price': market_data.price,
                        'volume': market_data.volume,
                        'timestamp': market_data.timestamp.isoformat(),
                        'source': market_data.source
                    }
                    
                    self.redis_client.setex(
                        cache_key, 
                        300,  # 5분 TTL
                        json.dumps(cache_data)
                    )
                    
            except Exception as e:
                self.logger.error(f"Data collection error: {e}")
                
    async def feature_processing_loop(self):
        """특성 처리 루프"""
        self.logger.info("Starting feature processing loop")
        
        if not self.consumer:
            self.logger.warning("No Kafka consumer, skipping feature processing")
            return
            
        try:
            for message in self.consumer:
                try:
                    # 메시지 파싱
                    data = message.value
                    
                    if data['type'] == 'market_data':
                        # MarketData 객체 재구성
                        market_data_dict = data['data']
                        market_data_dict['timestamp'] = datetime.fromisoformat(
                            market_data_dict['timestamp']
                        )
                        
                        market_data = MarketData(**market_data_dict)
                        
                        # 특성 처리
                        processed_features = self.feature_processor.process_features(market_data)
                        
                        # 처리된 특성을 Redis에 저장
                        await self.cache_processed_features(processed_features)
                        
                        # 다운스트림으로 전송
                        await self.send_processed_features(processed_features)
                        
                        self.processing_stats['messages_processed'] += 1
                        
                except Exception as e:
                    self.logger.error(f"Feature processing error: {e}")
                    self.processing_stats['errors'] += 1
                    
        except Exception as e:
            self.logger.error(f"Feature processing loop failed: {e}")
            
    async def cache_processed_features(self, features: ProcessedFeatures):
        """처리된 특성을 캐시에 저장"""
        if not self.redis_client:
            return
            
        try:
            cache_key = f"features:{features.symbol}"
            cache_data = {
                'timestamp': features.timestamp.isoformat(),
                'features': features.features,
                'technical_indicators': features.technical_indicators,
                'quality_score': features.quality_score,
                'processing_latency_ms': features.processing_latency_ms
            }
            
            # 최신 특성 저장 (1시간 TTL)
            self.redis_client.setex(
                cache_key,
                3600,
                json.dumps(cache_data, default=str)
            )
            
            # 시계열 특성 저장 (24시간 보관)
            timeseries_key = f"features_timeseries:{features.symbol}"
            self.redis_client.lpush(timeseries_key, json.dumps(cache_data, default=str))
            self.redis_client.ltrim(timeseries_key, 0, 1440)  # 최대 1440개 (24시간)
            self.redis_client.expire(timeseries_key, 86400)
            
        except Exception as e:
            self.logger.error(f"Feature caching failed: {e}")
            
    async def send_processed_features(self, features: ProcessedFeatures):
        """처리된 특성을 다운스트림으로 전송"""
        if self.producer:
            try:
                message = {
                    'type': 'processed_features',
                    'data': asdict(features),
                    'timestamp': time.time()
                }
                
                self.producer.send('btc-processed-features', message)
                
            except Exception as e:
                self.logger.error(f"Feature sending failed: {e}")
                
    async def monitoring_loop(self):
        """모니터링 루프"""
        while True:
            try:
                await asyncio.sleep(60)  # 1분마다 모니터링
                
                # 처리 통계
                runtime = time.time() - self.processing_stats['start_time']
                messages_per_second = self.processing_stats['messages_processed'] / runtime if runtime > 0 else 0
                error_rate = self.processing_stats['errors'] / self.processing_stats['messages_processed'] if self.processing_stats['messages_processed'] > 0 else 0
                
                self.logger.info(
                    f"Pipeline stats: {self.processing_stats['messages_processed']} messages, "
                    f"{messages_per_second:.2f} msg/s, {error_rate:.2%} error rate"
                )
                
                # Redis 연결 상태 확인
                if self.redis_client:
                    try:
                        self.redis_client.ping()
                    except Exception as e:
                        self.logger.error(f"Redis connection failed: {e}")
                        
                # Kafka 연결 상태 확인
                if self.producer:
                    try:
                        # 간단한 테스트 메시지
                        self.producer.send('btc-health-check', {'timestamp': time.time()})
                    except Exception as e:
                        self.logger.error(f"Kafka producer failed: {e}")
                        
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")

class DataPipelineManager:
    """데이터 파이프라인 매니저"""
    
    def __init__(self, config_path: str = None):
        self.config_path = config_path
        self.stream_processor = StreamProcessor()
        self.setup_logging()
        
    def setup_logging(self):
        self.logger = logging.getLogger(__name__)
        
    async def start_pipeline(self):
        """파이프라인 시작"""
        self.logger.info("🌊 Starting real-time data pipeline")
        
        try:
            await self.stream_processor.run_pipeline()
            
        except Exception as e:
            self.logger.error(f"Pipeline startup failed: {e}")
            raise
            
    def stop_pipeline(self):
        """파이프라인 중지"""
        self.logger.info("Stopping data pipeline")
        
        if hasattr(self.stream_processor, 'producer') and self.stream_processor.producer:
            self.stream_processor.producer.close()
            
        if hasattr(self.stream_processor, 'consumer') and self.stream_processor.consumer:
            self.stream_processor.consumer.close()

if __name__ == "__main__":
    # 실시간 데이터 파이프라인 실행
    async def main():
        manager = DataPipelineManager()
        
        try:
            await manager.start_pipeline()
        except KeyboardInterrupt:
            print("\n🛑 Pipeline stopped by user")
        except Exception as e:
            print(f"❌ Pipeline failed: {e}")
        finally:
            manager.stop_pipeline()
            
    asyncio.run(main())