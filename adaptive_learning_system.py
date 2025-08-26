"""
적응형 학습 시스템 v1.0
실시간으로 예측 결과를 학습하여 정확도를 지속적으로 개선
"""

import os
import json
import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import asyncio
import logging
from dataclasses import dataclass, asdict
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PredictionRecord:
    """예측 기록"""
    id: str
    timestamp: datetime
    current_price: float
    predicted_price: float
    predicted_direction: str
    confidence: float
    actual_price: Optional[float] = None
    actual_direction: Optional[str] = None
    accuracy_score: Optional[float] = None
    verified_at: Optional[datetime] = None
    model_version: str = "v1.0"

@dataclass
class ModelPerformance:
    """모델 성능 지표"""
    model_name: str
    accuracy: float
    directional_accuracy: float
    mae: float  # Mean Absolute Error
    mse: float  # Mean Squared Error
    sample_count: int
    last_updated: datetime

class AdaptiveLearningSystem:
    """적응형 학습 시스템"""
    
    def __init__(self):
        self.base_path = "/Users/parkyoungjun/Desktop/BTC_Analysis_System"
        self.db_path = os.path.join(self.base_path, "learning_database.db")
        self.model_weights_path = os.path.join(self.base_path, "adaptive_weights.pkl")
        
        # 기본 가중치 (시작점)
        self.base_weights = {
            "momentum_divergence": 0.20,
            "volume_price_analysis": 0.18,
            "whale_sentiment": 0.15,
            "funding_momentum": 0.12,
            "order_flow_imbalance": 0.10,
            "correlation_break": 0.08,
            "volatility_regime": 0.07,
            "social_momentum": 0.05,
            "institutional_flow": 0.05
        }
        
        # 현재 가중치 (학습으로 업데이트됨)
        self.current_weights = self.load_weights()
        
        # 성능 임계값
        self.performance_thresholds = {
            "min_accuracy": 0.6,      # 최소 60% 정확도
            "min_samples": 10,        # 최소 10개 샘플
            "confidence_threshold": 0.7,  # 70% 이상 신뢰도만 학습에 사용
            "learning_rate": 0.1      # 학습률
        }
        
        self.init_database()
    
    def init_database(self):
        """데이터베이스 초기화"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 예측 기록 테이블
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS predictions (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    current_price REAL NOT NULL,
                    predicted_price REAL NOT NULL,
                    predicted_direction TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    actual_price REAL,
                    actual_direction TEXT,
                    accuracy_score REAL,
                    verified_at TEXT,
                    model_version TEXT,
                    raw_data TEXT
                )
            ''')
            
            # 모델 성능 테이블
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS model_performance (
                    model_name TEXT PRIMARY KEY,
                    accuracy REAL NOT NULL,
                    directional_accuracy REAL NOT NULL,
                    mae REAL NOT NULL,
                    mse REAL NOT NULL,
                    sample_count INTEGER NOT NULL,
                    last_updated TEXT NOT NULL
                )
            ''')
            
            # 가중치 히스토리 테이블
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS weight_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    weights TEXT NOT NULL,
                    performance_score REAL,
                    reason TEXT
                )
            ''')
            
            conn.commit()
            conn.close()
            
            logger.info("✅ 학습 데이터베이스 초기화 완료")
            
        except Exception as e:
            logger.error(f"데이터베이스 초기화 실패: {e}")
    
    def save_prediction(self, prediction_data: Dict) -> str:
        """예측 결과 저장"""
        try:
            prediction_id = f"pred_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            record = PredictionRecord(
                id=prediction_id,
                timestamp=datetime.now(),
                current_price=prediction_data.get("current_price", 0),
                predicted_price=prediction_data.get("predicted_price", 0),
                predicted_direction=prediction_data.get("direction", "NEUTRAL"),
                confidence=prediction_data.get("confidence", 0.5),
                model_version=prediction_data.get("model_version", "v1.0")
            )
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO predictions 
                (id, timestamp, current_price, predicted_price, predicted_direction, 
                 confidence, model_version, raw_data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                record.id,
                record.timestamp.isoformat(),
                record.current_price,
                record.predicted_price,
                record.predicted_direction,
                record.confidence,
                record.model_version,
                json.dumps(prediction_data)
            ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"✅ 예측 저장: {prediction_id}")
            return prediction_id
            
        except Exception as e:
            logger.error(f"예측 저장 실패: {e}")
            return ""
    
    async def verify_and_learn(self) -> Dict:
        """예측 검증 및 학습"""
        try:
            logger.info("🔍 예측 결과 검증 시작...")
            
            # 검증 대상 예측들 조회 (1시간 이상 지난 것들)
            cutoff_time = datetime.now() - timedelta(hours=1)
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM predictions 
                WHERE verified_at IS NULL 
                AND timestamp < ? 
                ORDER BY timestamp DESC 
                LIMIT 50
            ''', (cutoff_time.isoformat(),))
            
            unverified = cursor.fetchall()
            conn.close()
            
            if not unverified:
                logger.info("검증할 예측이 없습니다")
                return {"verified": 0, "learned": 0}
            
            # 현재 가격 데이터 가져오기
            current_data = await self.get_current_market_data()
            current_price = current_data.get("price", 0) if current_data else 0
            
            verified_count = 0
            learned_count = 0
            
            for record in unverified:
                # 예측 검증
                verification_result = await self.verify_prediction(record, current_price)
                
                if verification_result["verified"]:
                    verified_count += 1
                    
                    # 높은 신뢰도 예측만 학습에 사용
                    if record[5] >= self.performance_thresholds["confidence_threshold"]:  # confidence
                        await self.learn_from_prediction(record, verification_result)
                        learned_count += 1
            
            # 모델 성능 업데이트
            await self.update_model_performance()
            
            # 가중치 최적화
            if learned_count > 0:
                await self.optimize_weights()
            
            logger.info(f"✅ 검증: {verified_count}개, 학습: {learned_count}개")
            
            return {
                "verified": verified_count,
                "learned": learned_count,
                "current_accuracy": await self.get_current_accuracy()
            }
            
        except Exception as e:
            logger.error(f"검증 및 학습 실패: {e}")
            return {"error": str(e)}
    
    async def verify_prediction(self, record: tuple, current_price: float) -> Dict:
        """개별 예측 검증"""
        try:
            # record 구조: id, timestamp, current_price, predicted_price, predicted_direction, confidence, ...
            pred_id = record[0]
            timestamp = datetime.fromisoformat(record[1])
            original_price = record[2]
            predicted_price = record[3]
            predicted_direction = record[4]
            confidence = record[5]
            
            # 실제 방향 계산
            price_change = current_price - original_price
            actual_direction = "BULLISH" if price_change > 0 else "BEARISH" if price_change < 0 else "NEUTRAL"
            
            # 정확도 점수 계산
            price_accuracy = 1 - abs(predicted_price - current_price) / original_price
            direction_correct = (predicted_direction == actual_direction)
            
            # 종합 정확도 (가격 50% + 방향 50%)
            accuracy_score = (price_accuracy * 0.5) + (1.0 if direction_correct else 0.0) * 0.5
            
            # DB 업데이트
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE predictions 
                SET actual_price = ?, actual_direction = ?, accuracy_score = ?, verified_at = ?
                WHERE id = ?
            ''', (
                current_price,
                actual_direction,
                accuracy_score,
                datetime.now().isoformat(),
                pred_id
            ))
            
            conn.commit()
            conn.close()
            
            return {
                "verified": True,
                "accuracy_score": accuracy_score,
                "direction_correct": direction_correct,
                "price_accuracy": price_accuracy,
                "actual_direction": actual_direction
            }
            
        except Exception as e:
            logger.error(f"예측 검증 실패: {e}")
            return {"verified": False, "error": str(e)}
    
    async def learn_from_prediction(self, record: tuple, verification: Dict):
        """예측 결과로부터 학습"""
        try:
            # 예측이 좋았다면 해당 신호들의 가중치 증가
            # 예측이 나빴다면 가중치 감소
            
            accuracy_score = verification["accuracy_score"]
            learning_rate = self.performance_thresholds["learning_rate"]
            
            # 가중치 조정 로직
            if accuracy_score > 0.7:  # 좋은 예측
                # 성공적인 예측에 기여한 요소들 강화
                await self.strengthen_successful_patterns(record, accuracy_score)
            elif accuracy_score < 0.3:  # 나쁜 예측
                # 실패한 예측에 기여한 요소들 약화
                await self.weaken_failed_patterns(record, accuracy_score)
            
            logger.debug(f"학습 완료: 정확도 {accuracy_score:.2f}")
            
        except Exception as e:
            logger.error(f"학습 실패: {e}")
    
    async def strengthen_successful_patterns(self, record: tuple, accuracy: float):
        """성공 패턴 강화"""
        try:
            # 성공도에 비례하여 주요 지표들의 가중치 증가
            boost_factor = (accuracy - 0.7) * 0.1  # 최대 3% 증가
            
            # 현재 높은 가중치를 가진 지표들을 더 강화
            high_weight_indicators = [
                "momentum_divergence",
                "volume_price_analysis", 
                "whale_sentiment"
            ]
            
            for indicator in high_weight_indicators:
                if indicator in self.current_weights:
                    self.current_weights[indicator] += boost_factor
                    
            self.normalize_weights()
            
        except Exception as e:
            logger.error(f"성공 패턴 강화 실패: {e}")
    
    async def weaken_failed_patterns(self, record: tuple, accuracy: float):
        """실패 패턴 약화"""
        try:
            # 실패도에 비례하여 가중치 감소
            penalty_factor = (0.3 - accuracy) * 0.05  # 최대 1.5% 감소
            
            # 모든 지표를 약간씩 감소
            for indicator in self.current_weights:
                self.current_weights[indicator] -= penalty_factor
                # 최소값 보장
                self.current_weights[indicator] = max(self.current_weights[indicator], 0.01)
                
            self.normalize_weights()
            
        except Exception as e:
            logger.error(f"실패 패턴 약화 실패: {e}")
    
    def normalize_weights(self):
        """가중치 정규화"""
        total_weight = sum(self.current_weights.values())
        if total_weight > 0:
            for key in self.current_weights:
                self.current_weights[key] /= total_weight
    
    async def update_model_performance(self):
        """모델 성능 업데이트"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 지난 7일간의 검증된 예측들 조회
            week_ago = (datetime.now() - timedelta(days=7)).isoformat()
            
            cursor.execute('''
                SELECT accuracy_score, predicted_direction, actual_direction,
                       predicted_price, actual_price, current_price
                FROM predictions 
                WHERE verified_at IS NOT NULL 
                AND timestamp > ?
            ''', (week_ago,))
            
            results = cursor.fetchall()
            
            if len(results) >= self.performance_thresholds["min_samples"]:
                accuracies = [r[0] for r in results if r[0] is not None]
                direction_correct = [r[1] == r[2] for r in results if r[1] and r[2]]
                
                # 성능 계산
                avg_accuracy = np.mean(accuracies) if accuracies else 0
                directional_accuracy = np.mean(direction_correct) if direction_correct else 0
                
                # MAE, MSE 계산
                price_errors = []
                for r in results:
                    if r[3] and r[4]:  # predicted_price, actual_price
                        error = abs(r[3] - r[4])
                        price_errors.append(error)
                
                mae = np.mean(price_errors) if price_errors else 0
                mse = np.mean([e**2 for e in price_errors]) if price_errors else 0
                
                # 성능 저장
                performance = ModelPerformance(
                    model_name="enhanced_v2.0",
                    accuracy=avg_accuracy,
                    directional_accuracy=directional_accuracy,
                    mae=mae,
                    mse=mse,
                    sample_count=len(results),
                    last_updated=datetime.now()
                )
                
                cursor.execute('''
                    INSERT OR REPLACE INTO model_performance
                    (model_name, accuracy, directional_accuracy, mae, mse, sample_count, last_updated)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    performance.model_name,
                    performance.accuracy,
                    performance.directional_accuracy,
                    performance.mae,
                    performance.mse,
                    performance.sample_count,
                    performance.last_updated.isoformat()
                ))
                
                conn.commit()
                logger.info(f"✅ 모델 성능 업데이트: 정확도 {avg_accuracy:.1%}, 방향성 {directional_accuracy:.1%}")
            
            conn.close()
            
        except Exception as e:
            logger.error(f"모델 성능 업데이트 실패: {e}")
    
    async def optimize_weights(self):
        """가중치 최적화"""
        try:
            # 현재 성능이 임계값보다 낮으면 가중치 재조정
            current_accuracy = await self.get_current_accuracy()
            
            if current_accuracy < self.performance_thresholds["min_accuracy"]:
                logger.info("성능이 낮아 가중치 재조정 실행")
                
                # 베이스 가중치로 일부 회귀
                regression_factor = 0.2
                for key in self.current_weights:
                    current_val = self.current_weights[key]
                    base_val = self.base_weights.get(key, 0.1)
                    self.current_weights[key] = current_val * (1 - regression_factor) + base_val * regression_factor
            
            # 가중치 저장
            self.save_weights()
            
            # 히스토리 저장
            await self.save_weight_history("optimization", current_accuracy)
            
        except Exception as e:
            logger.error(f"가중치 최적화 실패: {e}")
    
    def save_weights(self):
        """가중치 저장"""
        try:
            with open(self.model_weights_path, 'wb') as f:
                pickle.dump(self.current_weights, f)
            logger.debug("가중치 저장 완료")
        except Exception as e:
            logger.error(f"가중치 저장 실패: {e}")
    
    def load_weights(self) -> Dict:
        """가중치 로드"""
        try:
            if os.path.exists(self.model_weights_path):
                with open(self.model_weights_path, 'rb') as f:
                    weights = pickle.load(f)
                logger.info("✅ 기존 가중치 로드 완료")
                return weights
            else:
                logger.info("기본 가중치 사용")
                return self.base_weights.copy()
        except Exception as e:
            logger.error(f"가중치 로드 실패: {e}")
            return self.base_weights.copy()
    
    async def save_weight_history(self, reason: str, performance_score: float):
        """가중치 변경 히스토리 저장"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO weight_history (timestamp, weights, performance_score, reason)
                VALUES (?, ?, ?, ?)
            ''', (
                datetime.now().isoformat(),
                json.dumps(self.current_weights),
                performance_score,
                reason
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"가중치 히스토리 저장 실패: {e}")
    
    async def get_current_accuracy(self) -> float:
        """현재 정확도 조회"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT accuracy FROM model_performance 
                WHERE model_name = 'enhanced_v2.0'
                ORDER BY last_updated DESC LIMIT 1
            ''')
            
            result = cursor.fetchone()
            conn.close()
            
            return result[0] if result else 0.5
            
        except Exception as e:
            logger.error(f"현재 정확도 조회 실패: {e}")
            return 0.5
    
    async def get_current_market_data(self) -> Optional[Dict]:
        """현재 시장 데이터 조회"""
        try:
            # enhanced_data_collector에서 최신 데이터 가져오기
            historical_path = os.path.join(self.base_path, "historical_data")
            if not os.path.exists(historical_path):
                return None
                
            files = [f for f in os.listdir(historical_path) 
                     if f.startswith("btc_analysis_") and f.endswith(".json")]
            
            if not files:
                return None
                
            latest_file = sorted(files)[-1]
            file_path = os.path.join(historical_path, latest_file)
            
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # 가격 추출
            paths = [
                ["data_sources", "legacy_analyzer", "market_data", "avg_price"],
                ["summary", "current_btc_price"]
            ]
            
            for path in paths:
                try:
                    value = data
                    for key in path:
                        value = value[key]
                    if value and value > 0:
                        return {"price": float(value), "data": data}
                except:
                    continue
            
            return None
            
        except Exception as e:
            logger.error(f"시장 데이터 조회 실패: {e}")
            return None
    
    async def get_learning_report(self) -> Dict:
        """학습 리포트 생성"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # 최근 성능
            perf_df = pd.read_sql_query('''
                SELECT * FROM model_performance 
                ORDER BY last_updated DESC LIMIT 1
            ''', conn)
            
            # 최근 예측들
            pred_df = pd.read_sql_query('''
                SELECT * FROM predictions 
                WHERE verified_at IS NOT NULL 
                ORDER BY timestamp DESC LIMIT 20
            ''', conn)
            
            # 가중치 변화
            weight_df = pd.read_sql_query('''
                SELECT * FROM weight_history 
                ORDER BY timestamp DESC LIMIT 5
            ''', conn)
            
            conn.close()
            
            return {
                "current_performance": perf_df.to_dict('records')[0] if not perf_df.empty else {},
                "recent_predictions": pred_df.to_dict('records'),
                "weight_changes": weight_df.to_dict('records'),
                "current_weights": self.current_weights
            }
            
        except Exception as e:
            logger.error(f"학습 리포트 생성 실패: {e}")
            return {"error": str(e)}

async def run_learning_cycle():
    """학습 사이클 실행"""
    print("🤖 적응형 학습 시스템 시작")
    print("="*50)
    
    learning_system = AdaptiveLearningSystem()
    
    # 학습 실행
    result = await learning_system.verify_and_learn()
    
    if "error" in result:
        print(f"❌ 학습 실패: {result['error']}")
        return
    
    # 결과 출력
    print(f"✅ 검증: {result['verified']}개")
    print(f"📚 학습: {result['learned']}개") 
    print(f"🎯 현재 정확도: {result['current_accuracy']:.1%}")
    
    # 상세 리포트
    report = await learning_system.get_learning_report()
    
    if report.get("current_performance"):
        perf = report["current_performance"]
        print(f"\n📊 모델 성능:")
        print(f"  • 전체 정확도: {perf.get('accuracy', 0):.1%}")
        print(f"  • 방향성 정확도: {perf.get('directional_accuracy', 0):.1%}")
        print(f"  • 평균 오차: ${perf.get('mae', 0):.0f}")
        print(f"  • 샘플 수: {perf.get('sample_count', 0)}개")
    
    print(f"\n⚖️ 현재 가중치 (상위 5개):")
    weights = report.get("current_weights", {})
    sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)
    for name, weight in sorted_weights[:5]:
        print(f"  • {name}: {weight:.1%}")
    
    print("="*50)
    print("🎉 적응형 학습 완료!")

if __name__ == "__main__":
    asyncio.run(run_learning_cycle())