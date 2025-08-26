#!/usr/bin/env python3
"""
예측 정확도 추적 및 학습 시스템
Claude 예측의 성공/실패를 추적하여 시스템을 지속적으로 개선
"""

import json
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging
import numpy as np

class PredictionTracker:
    def __init__(self, db_path: str = "predictions.db"):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self.init_database()
        
    def init_database(self):
        """예측 추적 데이터베이스 초기화"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                current_price REAL NOT NULL,
                prediction_direction TEXT NOT NULL,
                predicted_price REAL NOT NULL,
                probability REAL NOT NULL,
                confidence TEXT NOT NULL,
                timeframe_hours INTEGER NOT NULL,
                leading_indicators TEXT NOT NULL,
                claude_reasoning TEXT NOT NULL,
                
                -- 결과 추적
                actual_price REAL,
                actual_direction TEXT,
                direction_correct BOOLEAN,
                price_accuracy REAL,
                outcome_timestamp TEXT,
                is_evaluated BOOLEAN DEFAULT FALSE,
                
                -- 메타데이터
                market_condition TEXT,
                volatility_regime TEXT,
                prediction_quality_score REAL
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS prediction_accuracy (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                period TEXT NOT NULL,  -- daily, weekly, monthly
                total_predictions INTEGER NOT NULL,
                correct_directions INTEGER NOT NULL,
                direction_accuracy REAL NOT NULL,
                avg_price_accuracy REAL NOT NULL,
                false_positive_rate REAL NOT NULL,
                confidence_calibration REAL NOT NULL,
                timestamp TEXT NOT NULL
            )
        ''')
        
        conn.commit()
        conn.close()
        
    def record_prediction(self, prediction: Dict, current_data: Dict, leading_indicators: Dict) -> int:
        """새로운 예측을 기록"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            current_price = current_data.get("price_data", {}).get("current_price", 0)
            pred_info = prediction.get("prediction", {})
            
            # 시간프레임을 시간 단위로 변환
            timeframe = pred_info.get("timeframe", "6-12시간")
            hours = self._parse_timeframe_to_hours(timeframe)
            
            cursor.execute('''
                INSERT INTO predictions (
                    timestamp, current_price, prediction_direction, predicted_price,
                    probability, confidence, timeframe_hours, leading_indicators,
                    claude_reasoning, market_condition, volatility_regime
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.utcnow().isoformat(),
                current_price,
                pred_info.get("direction", "NEUTRAL"),
                pred_info.get("target_price", current_price),
                pred_info.get("probability", 50),
                pred_info.get("confidence", "LOW"),
                hours,
                json.dumps(leading_indicators),
                prediction.get("analysis", {}).get("reasoning", ""),
                self._assess_market_condition(current_data),
                self._assess_volatility_regime(current_data)
            ))
            
            prediction_id = cursor.lastrowid
            conn.commit()
            conn.close()
            
            self.logger.info(f"예측 기록됨: ID {prediction_id}")
            return prediction_id
            
        except Exception as e:
            self.logger.error(f"예측 기록 실패: {e}")
            return -1
    
    def evaluate_predictions(self, current_data: Dict) -> Dict:
        """만료된 예측들을 평가"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            current_time = datetime.utcnow()
            current_price = current_data.get("price_data", {}).get("current_price", 0)
            
            # 평가할 예측들 찾기 (만료되었지만 아직 평가되지 않은 것들)
            cursor.execute('''
                SELECT id, timestamp, current_price, prediction_direction, 
                       predicted_price, probability, timeframe_hours
                FROM predictions 
                WHERE is_evaluated = FALSE 
                AND datetime(timestamp) <= datetime(?, '-' || timeframe_hours || ' hours')
            ''', (current_time.isoformat(),))
            
            predictions_to_evaluate = cursor.fetchall()
            evaluation_results = []
            
            for pred in predictions_to_evaluate:
                pred_id, timestamp, orig_price, direction, target_price, probability, timeframe = pred
                
                # 실제 결과 계산
                actual_direction = self._calculate_actual_direction(orig_price, current_price)
                direction_correct = (direction == actual_direction) or (direction == "NEUTRAL" and abs(current_price - orig_price) / orig_price < 0.02)
                
                # 가격 정확도 계산 (예측 대비 실제 결과)
                if direction == "NEUTRAL":
                    price_accuracy = 1.0 - min(abs(current_price - orig_price) / orig_price, 1.0)
                else:
                    price_accuracy = 1.0 - min(abs(current_price - target_price) / target_price, 1.0)
                
                # 결과 업데이트
                cursor.execute('''
                    UPDATE predictions SET 
                        actual_price = ?, actual_direction = ?, direction_correct = ?,
                        price_accuracy = ?, outcome_timestamp = ?, is_evaluated = TRUE,
                        prediction_quality_score = ?
                    WHERE id = ?
                ''', (
                    current_price, actual_direction, direction_correct,
                    price_accuracy, current_time.isoformat(),
                    self._calculate_quality_score(direction_correct, price_accuracy, probability),
                    pred_id
                ))
                
                evaluation_results.append({
                    "prediction_id": pred_id,
                    "direction_correct": direction_correct,
                    "price_accuracy": price_accuracy,
                    "timeframe": timeframe
                })
            
            conn.commit()
            conn.close()
            
            if evaluation_results:
                self.logger.info(f"{len(evaluation_results)}개 예측 평가 완료")
                
            return {
                "evaluated_count": len(evaluation_results),
                "results": evaluation_results
            }
            
        except Exception as e:
            self.logger.error(f"예측 평가 실패: {e}")
            return {"evaluated_count": 0, "results": []}
    
    def get_accuracy_metrics(self, days: int = 7) -> Dict:
        """최근 N일간의 정확도 메트릭스"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            since_date = (datetime.utcnow() - timedelta(days=days)).isoformat()
            
            cursor.execute('''
                SELECT direction_correct, price_accuracy, probability, confidence,
                       prediction_direction, volatility_regime
                FROM predictions 
                WHERE is_evaluated = TRUE 
                AND timestamp >= ?
            ''', (since_date,))
            
            results = cursor.fetchall()
            conn.close()
            
            if not results:
                return {"error": "평가된 예측이 없습니다"}
            
            # 메트릭스 계산
            total_predictions = len(results)
            correct_directions = sum(1 for r in results if r[0])  # direction_correct
            direction_accuracy = correct_directions / total_predictions
            
            price_accuracies = [r[1] for r in results if r[1] is not None]
            avg_price_accuracy = np.mean(price_accuracies) if price_accuracies else 0
            
            # 신뢰도별 정확도
            confidence_breakdown = {}
            for conf in ["HIGH", "MEDIUM", "LOW"]:
                conf_results = [r for r in results if r[3] == conf]
                if conf_results:
                    conf_accuracy = sum(1 for r in conf_results if r[0]) / len(conf_results)
                    confidence_breakdown[conf] = {
                        "count": len(conf_results),
                        "accuracy": conf_accuracy
                    }
            
            # 방향별 정확도
            direction_breakdown = {}
            for direction in ["BULLISH", "BEARISH", "NEUTRAL"]:
                dir_results = [r for r in results if r[4] == direction]
                if dir_results:
                    dir_accuracy = sum(1 for r in dir_results if r[0]) / len(dir_results)
                    direction_breakdown[direction] = {
                        "count": len(dir_results),
                        "accuracy": dir_accuracy
                    }
            
            # 거짓 양성률 (높은 확률로 예측했지만 틀린 경우)
            high_prob_predictions = [r for r in results if r[2] >= 70]  # probability >= 70%
            false_positives = sum(1 for r in high_prob_predictions if not r[0])
            false_positive_rate = false_positives / len(high_prob_predictions) if high_prob_predictions else 0
            
            return {
                "period_days": days,
                "total_predictions": total_predictions,
                "direction_accuracy": round(direction_accuracy, 3),
                "avg_price_accuracy": round(avg_price_accuracy, 3),
                "false_positive_rate": round(false_positive_rate, 3),
                "confidence_breakdown": confidence_breakdown,
                "direction_breakdown": direction_breakdown,
                "quality_score": round((direction_accuracy + avg_price_accuracy) / 2, 3)
            }
            
        except Exception as e:
            self.logger.error(f"정확도 메트릭스 계산 실패: {e}")
            return {"error": str(e)}
    
    def should_send_alert(self, prediction: Dict, accuracy_metrics: Dict) -> bool:
        """과거 성과를 고려한 알림 발송 여부 결정"""
        try:
            pred_info = prediction.get("prediction", {})
            direction = pred_info.get("direction", "NEUTRAL")
            probability = pred_info.get("probability", 50)
            confidence = pred_info.get("confidence", "LOW")
            
            # 기본 필터: NEUTRAL은 알림 안함
            if direction == "NEUTRAL":
                return False
            
            # 시스템 전체 성과가 너무 낮으면 알림 중단
            overall_accuracy = accuracy_metrics.get("direction_accuracy", 0)
            if overall_accuracy < 0.6:  # 60% 미만이면 신뢰도 부족
                self.logger.warning(f"전체 정확도 {overall_accuracy:.1%}로 낮아 알림 중단")
                return False
            
            # 신뢰도별 동적 임계값
            confidence_breakdown = accuracy_metrics.get("confidence_breakdown", {})
            conf_accuracy = confidence_breakdown.get(confidence, {}).get("accuracy", 0.5)
            
            if confidence == "HIGH":
                required_probability = 70 if conf_accuracy > 0.7 else 80
            elif confidence == "MEDIUM":
                required_probability = 80 if conf_accuracy > 0.7 else 85
            else:  # LOW
                return False  # 저신뢰도는 성과와 관계없이 알림 안함
            
            # 방향별 성과 고려
            direction_breakdown = accuracy_metrics.get("direction_breakdown", {})
            dir_accuracy = direction_breakdown.get(direction, {}).get("accuracy", 0.5)
            
            # 해당 방향 예측 성과가 나쁘면 임계값 상향
            if dir_accuracy < 0.6:
                required_probability += 10
            
            final_decision = probability >= required_probability
            
            self.logger.info(f"알림 결정: {direction} {probability}% {confidence} -> {'발송' if final_decision else '보류'} (성과기반 임계값: {required_probability}%)")
            
            return final_decision
            
        except Exception as e:
            self.logger.error(f"알림 결정 로직 오류: {e}")
            return False  # 오류 시 보수적 접근
    
    def _parse_timeframe_to_hours(self, timeframe: str) -> int:
        """시간프레임 문자열을 시간으로 변환"""
        if "시간" in timeframe:
            # "6-12시간" -> 12 (최대값 사용)
            hours = [int(x) for x in timeframe.replace("시간", "").split("-") if x.strip().isdigit()]
            return max(hours) if hours else 12
        elif "분" in timeframe:
            # "30분" -> 0.5시간
            minutes = [int(x) for x in timeframe.replace("분", "").split("-") if x.strip().isdigit()]
            return max(minutes) / 60 if minutes else 1
        else:
            return 12  # 기본값
    
    def _calculate_actual_direction(self, orig_price: float, current_price: float) -> str:
        """실제 가격 변화 방향 계산"""
        change_pct = (current_price - orig_price) / orig_price
        
        if change_pct > 0.02:  # 2% 이상 상승
            return "BULLISH"
        elif change_pct < -0.02:  # 2% 이상 하락
            return "BEARISH"
        else:
            return "NEUTRAL"
    
    def _calculate_quality_score(self, direction_correct: bool, price_accuracy: float, probability: float) -> float:
        """예측 품질 점수 계산"""
        direction_weight = 0.6
        price_weight = 0.4
        
        direction_score = 1.0 if direction_correct else 0.0
        
        # 확률 보정 (높은 확률로 예측했다면 더 엄격하게 평가)
        probability_factor = probability / 100.0
        confidence_penalty = 1.0 if direction_correct else (1.0 - probability_factor)
        
        return (direction_score * direction_weight + price_accuracy * price_weight) * confidence_penalty
    
    def _assess_market_condition(self, current_data: Dict) -> str:
        """시장 상황 평가"""
        # 간단한 버전 - 향후 고도화 가능
        if "macro_data" in current_data and "vix" in current_data["macro_data"]:
            vix = current_data["macro_data"]["vix"]["current"]
            if vix > 30:
                return "high_stress"
            elif vix > 20:
                return "moderate_stress"
            else:
                return "low_stress"
        return "unknown"
    
    def _assess_volatility_regime(self, current_data: Dict) -> str:
        """변동성 레짐 평가"""
        # 간단한 버전
        if "price_data" in current_data:
            change_24h = abs(current_data["price_data"].get("change_24h", 0))
            if change_24h > 10:
                return "high_vol"
            elif change_24h > 5:
                return "medium_vol"
            else:
                return "low_vol"
        return "unknown"

# 테스트 함수
def test_prediction_tracker():
    """예측 추적기 테스트"""
    print("🧪 예측 추적기 테스트...")
    
    tracker = PredictionTracker(":memory:")  # 메모리 DB 사용
    
    # 테스트 예측 기록
    test_prediction = {
        "prediction": {
            "direction": "BULLISH",
            "probability": 75,
            "target_price": 62000,
            "confidence": "HIGH",
            "timeframe": "6시간"
        },
        "analysis": {
            "reasoning": "테스트 예측"
        }
    }
    
    test_data = {
        "price_data": {"current_price": 60000, "change_24h": 2.3}
    }
    
    # 예측 기록
    pred_id = tracker.record_prediction(test_prediction, test_data, {})
    print(f"✅ 예측 기록됨: ID {pred_id}")
    
    # 정확도 메트릭스 (빈 결과)
    metrics = tracker.get_accuracy_metrics()
    print(f"✅ 메트릭스: {metrics}")
    
    return True

if __name__ == "__main__":
    test_prediction_tracker()