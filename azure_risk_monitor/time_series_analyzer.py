"""
시계열 분석 시스템
과거 지수들의 변화 패턴으로 미래 예측
30분 간격이지만 1분 데이터로 시계열 분석
"""

import asyncio
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass
import logging
import json

logger = logging.getLogger(__name__)

@dataclass
class TimeSeriesData:
    """시계열 데이터 구조"""
    timestamp: datetime
    price: float
    volume: float
    indicators: Dict
    
class TimeSeriesAnalyzer:
    """시계열 분석기 - 과거 패턴으로 미래 예측"""
    
    def __init__(self, db_path: str = "timeseries.db"):
        self.db_path = db_path
        self.logger = logger
        self.init_database()
        
        # 시계열 분석 파라미터
        self.analysis_windows = {
            "short": 30,    # 30분 (1분 × 30)
            "medium": 180,  # 3시간 (1분 × 180)
            "long": 720     # 12시간 (1분 × 720)
        }
        
    def init_database(self):
        """시계열 데이터베이스 초기화"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 1분 간격 가격 데이터
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS price_series (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    price REAL NOT NULL,
                    volume REAL,
                    change_1m REAL,
                    change_5m REAL,
                    change_15m REAL,
                    rsi_14 REAL,
                    macd_signal TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # 지표별 시계열 데이터
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS indicator_series (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    indicator_name TEXT NOT NULL,
                    value REAL NOT NULL,
                    signal TEXT,
                    strength REAL,
                    trend TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # 패턴 매칭 결과
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS pattern_matches (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    pattern_type TEXT NOT NULL,
                    similarity REAL NOT NULL,
                    predicted_direction TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    actual_outcome TEXT,
                    accuracy_score REAL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # 인덱스 생성 (쿼리 최적화)
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_price_timestamp ON price_series(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_indicator_timestamp ON indicator_series(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_pattern_timestamp ON pattern_matches(timestamp)')
            
            conn.commit()
            conn.close()
            
            self.logger.info("✅ 시계열 데이터베이스 초기화 완료")
            
        except Exception as e:
            self.logger.error(f"시계열 DB 초기화 실패: {e}")
    
    async def store_realtime_data(self, current_data: Dict, indicators: Dict) -> bool:
        """실시간 데이터 저장 (매번 호출 시)"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            timestamp = datetime.utcnow().isoformat()
            price_data = current_data.get("price_data", {})
            
            # 1. 가격 데이터 저장
            current_price = price_data.get("current_price", 0)
            volume = price_data.get("volume_24h", 0)
            
            # 과거 가격들과 비교하여 변화율 계산
            changes = self._calculate_price_changes(current_price)
            
            cursor.execute('''
                INSERT INTO price_series 
                (timestamp, price, volume, change_1m, change_5m, change_15m, rsi_14, macd_signal)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                timestamp, current_price, volume,
                changes.get("1m", 0), changes.get("5m", 0), changes.get("15m", 0),
                0, "NEUTRAL"  # RSI, MACD는 별도 계산 필요
            ))
            
            # 2. 지표 데이터 저장
            for indicator_name, indicator_data in indicators.items():
                if isinstance(indicator_data, dict):
                    value = indicator_data.get("strength", 0)
                    signal = indicator_data.get("signal", "NEUTRAL")
                    trend = indicator_data.get("trend", "stable")
                    
                    cursor.execute('''
                        INSERT INTO indicator_series 
                        (timestamp, indicator_name, value, signal, strength, trend)
                        VALUES (?, ?, ?, ?, ?, ?)
                    ''', (timestamp, indicator_name, value, signal, value, trend))
            
            conn.commit()
            conn.close()
            
            return True
            
        except Exception as e:
            self.logger.error(f"실시간 데이터 저장 실패: {e}")
            return False
    
    def _calculate_price_changes(self, current_price: float) -> Dict:
        """과거 가격과 비교하여 변화율 계산"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            changes = {}
            
            # 1분, 5분, 15분 전 가격 가져오기
            time_frames = {
                "1m": 1,
                "5m": 5,
                "15m": 15
            }
            
            for tf_name, minutes_ago in time_frames.items():
                past_time = (datetime.utcnow() - timedelta(minutes=minutes_ago)).isoformat()
                
                cursor.execute('''
                    SELECT price FROM price_series 
                    WHERE timestamp <= ? 
                    ORDER BY timestamp DESC 
                    LIMIT 1
                ''', (past_time,))
                
                result = cursor.fetchone()
                if result:
                    past_price = result[0]
                    change_pct = ((current_price - past_price) / past_price) * 100 if past_price > 0 else 0
                    changes[tf_name] = change_pct
                else:
                    changes[tf_name] = 0
            
            conn.close()
            return changes
            
        except Exception as e:
            self.logger.error(f"가격 변화율 계산 실패: {e}")
            return {"1m": 0, "5m": 0, "15m": 0}
    
    async def analyze_time_series_patterns(self) -> Dict:
        """시계열 패턴 분석 (핵심 기능)"""
        try:
            self.logger.info("🔍 시계열 패턴 분석 시작...")
            
            # 1. 최근 데이터 가져오기
            recent_patterns = self._get_recent_patterns()
            
            if not recent_patterns:
                self.logger.warning("시계열 데이터 부족")
                return {"pattern_found": False, "confidence": 0}
            
            # 2. 과거 유사 패턴 검색
            similar_patterns = self._find_similar_patterns(recent_patterns)
            
            # 3. 예측 생성
            prediction = self._generate_pattern_prediction(similar_patterns)
            
            # 4. 패턴 매칭 결과 저장
            self._store_pattern_result(prediction)
            
            return prediction
            
        except Exception as e:
            self.logger.error(f"시계열 패턴 분석 실패: {e}")
            return {"pattern_found": False, "confidence": 0}
    
    def _get_recent_patterns(self, lookback_minutes: int = 60) -> List[Dict]:
        """최근 패턴 추출"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 최근 60분 데이터
            since_time = (datetime.utcnow() - timedelta(minutes=lookback_minutes)).isoformat()
            
            cursor.execute('''
                SELECT timestamp, price, change_1m, change_5m, change_15m
                FROM price_series 
                WHERE timestamp >= ? 
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (since_time, lookback_minutes))
            
            rows = cursor.fetchall()
            
            patterns = []
            for row in rows:
                patterns.append({
                    "timestamp": row[0],
                    "price": row[1],
                    "change_1m": row[2],
                    "change_5m": row[3], 
                    "change_15m": row[4]
                })
            
            conn.close()
            return patterns
            
        except Exception as e:
            self.logger.error(f"최근 패턴 추출 실패: {e}")
            return []
    
    def _find_similar_patterns(self, current_pattern: List[Dict]) -> List[Dict]:
        """과거 유사 패턴 검색"""
        try:
            if len(current_pattern) < 10:
                return []
                
            # 현재 패턴의 특징 벡터 생성
            current_features = self._extract_pattern_features(current_pattern)
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 과거 모든 데이터에서 유사 구간 검색
            cursor.execute('''
                SELECT timestamp, price, change_1m, change_5m, change_15m
                FROM price_series 
                WHERE timestamp < ? 
                ORDER BY timestamp DESC
                LIMIT 1440
            ''', ((datetime.utcnow() - timedelta(hours=2)).isoformat(),))
            
            historical_data = cursor.fetchall()
            
            similar_patterns = []
            
            # 슬라이딩 윈도우로 유사성 검사
            for i in range(len(historical_data) - len(current_pattern)):
                window = historical_data[i:i + len(current_pattern)]
                
                # 특징 벡터 추출
                window_features = self._extract_pattern_features([{
                    "price": row[1], "change_1m": row[2], 
                    "change_5m": row[3], "change_15m": row[4]
                } for row in window])
                
                # 유사도 계산
                similarity = self._calculate_similarity(current_features, window_features)
                
                if similarity > 0.7:  # 70% 이상 유사
                    # 해당 패턴 이후 결과 확인
                    future_outcome = self._get_pattern_outcome(historical_data[i]["timestamp"])
                    
                    similar_patterns.append({
                        "similarity": similarity,
                        "outcome": future_outcome,
                        "timestamp": historical_data[i][0]
                    })
            
            conn.close()
            
            # 유사도 순으로 정렬
            similar_patterns.sort(key=lambda x: x["similarity"], reverse=True)
            return similar_patterns[:5]  # 상위 5개
            
        except Exception as e:
            self.logger.error(f"유사 패턴 검색 실패: {e}")
            return []
    
    def _extract_pattern_features(self, pattern: List[Dict]) -> np.ndarray:
        """패턴에서 특징 벡터 추출"""
        features = []
        
        for data in pattern:
            features.extend([
                data.get("change_1m", 0),
                data.get("change_5m", 0), 
                data.get("change_15m", 0)
            ])
        
        return np.array(features)
    
    def _calculate_similarity(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """두 패턴 간 유사도 계산 (코사인 유사도)"""
        try:
            if len(features1) != len(features2):
                return 0.0
                
            dot_product = np.dot(features1, features2)
            norm_a = np.linalg.norm(features1)
            norm_b = np.linalg.norm(features2)
            
            if norm_a == 0 or norm_b == 0:
                return 0.0
                
            return dot_product / (norm_a * norm_b)
            
        except Exception:
            return 0.0
    
    def _get_pattern_outcome(self, pattern_timestamp: str) -> Dict:
        """패턴 이후 실제 결과 확인"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 30분 후 가격
            future_time = (datetime.fromisoformat(pattern_timestamp) + timedelta(minutes=30)).isoformat()
            
            cursor.execute('''
                SELECT price FROM price_series 
                WHERE timestamp >= ? 
                ORDER BY timestamp ASC 
                LIMIT 1
            ''', (future_time,))
            
            result = cursor.fetchone()
            
            if result:
                # 현재 가격과 30분 후 가격 비교
                cursor.execute('''
                    SELECT price FROM price_series 
                    WHERE timestamp <= ? 
                    ORDER BY timestamp DESC 
                    LIMIT 1
                ''', (pattern_timestamp,))
                
                current_result = cursor.fetchone()
                
                if current_result:
                    current_price = current_result[0]
                    future_price = result[0]
                    change_pct = ((future_price - current_price) / current_price) * 100
                    
                    if change_pct > 0.5:
                        direction = "BULLISH"
                    elif change_pct < -0.5:
                        direction = "BEARISH"
                    else:
                        direction = "NEUTRAL"
                        
                    return {
                        "direction": direction,
                        "change_percent": change_pct,
                        "confidence": min(abs(change_pct) * 20, 100)
                    }
            
            conn.close()
            return {"direction": "NEUTRAL", "change_percent": 0, "confidence": 0}
            
        except Exception as e:
            self.logger.error(f"패턴 결과 확인 실패: {e}")
            return {"direction": "NEUTRAL", "change_percent": 0, "confidence": 0}
    
    def _generate_pattern_prediction(self, similar_patterns: List[Dict]) -> Dict:
        """유사 패턴들로부터 예측 생성"""
        try:
            if not similar_patterns:
                return {
                    "pattern_found": False,
                    "prediction": "NEUTRAL",
                    "confidence": 0,
                    "reasoning": "유사 패턴 없음"
                }
            
            # 유사 패턴들의 결과 분석
            bullish_count = 0
            bearish_count = 0
            total_confidence = 0
            
            for pattern in similar_patterns:
                outcome = pattern.get("outcome", {})
                direction = outcome.get("direction", "NEUTRAL")
                similarity = pattern.get("similarity", 0)
                
                if direction == "BULLISH":
                    bullish_count += 1
                    total_confidence += outcome.get("confidence", 0) * similarity
                elif direction == "BEARISH":
                    bearish_count += 1
                    total_confidence += outcome.get("confidence", 0) * similarity
            
            # 예측 결정
            if bullish_count > bearish_count:
                prediction = "BULLISH"
                confidence = (total_confidence / len(similar_patterns)) * (bullish_count / len(similar_patterns))
            elif bearish_count > bullish_count:
                prediction = "BEARISH"  
                confidence = (total_confidence / len(similar_patterns)) * (bearish_count / len(similar_patterns))
            else:
                prediction = "NEUTRAL"
                confidence = 30
            
            return {
                "pattern_found": True,
                "prediction": prediction,
                "confidence": min(confidence, 95),
                "similar_patterns_count": len(similar_patterns),
                "bullish_patterns": bullish_count,
                "bearish_patterns": bearish_count,
                "reasoning": f"{len(similar_patterns)}개 유사 패턴 중 {bullish_count}개 강세, {bearish_count}개 약세"
            }
            
        except Exception as e:
            self.logger.error(f"패턴 예측 생성 실패: {e}")
            return {"pattern_found": False, "prediction": "NEUTRAL", "confidence": 0}
    
    def _store_pattern_result(self, prediction: Dict):
        """패턴 분석 결과 저장"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO pattern_matches 
                (timestamp, pattern_type, similarity, predicted_direction, confidence)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                datetime.utcnow().isoformat(),
                "time_series_pattern",
                prediction.get("similar_patterns_count", 0) / 10,
                prediction.get("prediction", "NEUTRAL"),
                prediction.get("confidence", 0)
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"패턴 결과 저장 실패: {e}")
    
    def get_time_series_summary(self) -> Dict:
        """시계열 분석 요약"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 데이터 현황
            cursor.execute('SELECT COUNT(*) FROM price_series')
            price_count = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM indicator_series')
            indicator_count = cursor.fetchone()[0]
            
            # 최근 패턴 정확도
            cursor.execute('''
                SELECT AVG(accuracy_score) FROM pattern_matches 
                WHERE accuracy_score IS NOT NULL 
                AND timestamp >= ?
            ''', ((datetime.utcnow() - timedelta(days=7)).isoformat(),))
            
            accuracy = cursor.fetchone()[0] or 0
            
            conn.close()
            
            return {
                "price_data_points": price_count,
                "indicator_data_points": indicator_count,
                "pattern_accuracy_7d": f"{accuracy:.1%}",
                "data_coverage": f"{price_count}분" if price_count < 1440 else f"{price_count/1440:.1f}일"
            }
            
        except Exception as e:
            self.logger.error(f"시계열 요약 생성 실패: {e}")
            return {}

async def test_time_series_analyzer():
    """시계열 분석기 테스트"""
    print("🧪 시계열 분석 시스템 테스트")
    print("="*50)
    
    analyzer = TimeSeriesAnalyzer()
    
    # 시뮬레이션 데이터 저장
    print("📊 시뮬레이션 데이터 생성 중...")
    
    for i in range(100):
        fake_data = {
            "price_data": {
                "current_price": 58000 + i * 10 + np.random.normal(0, 50),
                "volume_24h": 25000000000
            }
        }
        fake_indicators = {
            "test_indicator": {
                "strength": np.random.random(),
                "signal": np.random.choice(["BULLISH", "BEARISH", "NEUTRAL"]),
                "trend": np.random.choice(["rising", "falling", "stable"])
            }
        }
        
        await analyzer.store_realtime_data(fake_data, fake_indicators)
        await asyncio.sleep(0.01)  # 짧은 지연
    
    # 패턴 분석 실행
    print("🔍 시계열 패턴 분석 중...")
    result = await analyzer.analyze_time_series_patterns()
    
    if result.get("pattern_found"):
        print(f"✅ 패턴 발견!")
        print(f"  • 예측: {result.get('prediction')}")
        print(f"  • 신뢰도: {result.get('confidence'):.1f}%")
        print(f"  • 유사 패턴: {result.get('similar_patterns_count')}개")
        print(f"  • 이유: {result.get('reasoning')}")
    else:
        print("❌ 유사 패턴을 찾지 못했습니다")
    
    # 시계열 요약
    summary = analyzer.get_time_series_summary()
    print(f"\n📈 시계열 데이터 현황:")
    for key, value in summary.items():
        print(f"  • {key}: {value}")

if __name__ == "__main__":
    asyncio.run(test_time_series_analyzer())