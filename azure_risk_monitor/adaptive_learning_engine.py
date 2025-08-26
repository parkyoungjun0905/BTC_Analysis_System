#!/usr/bin/env python3
"""
적응적 학습 엔진
실패 원인을 분석하고 시스템을 스스로 개선하는 AI 학습 시스템
"""

import json
import sqlite3
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from collections import defaultdict
import statistics

class AdaptiveLearningEngine:
    """스스로 학습하고 개선하는 예측 시스템"""
    
    def __init__(self, db_path: str = "predictions.db"):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self.init_learning_database()
        
        # 학습된 가중치들 (초기값)
        self.learned_weights = {
            "mempool_pressure": 1.4,
            "funding_rate": 1.5,
            "orderbook_imbalance": 1.2,
            "options_put_call": 1.3,
            "stablecoin_flows": 1.6,
            "fear_greed": 1.1,
            "social_volume": 0.9,
            "exchange_flows": 1.7,
            "whale_activity": 1.8,
            "miner_flows": 1.4,
            "price_momentum": 1.3,
            "volume_profile": 1.2,
            "liquidation_cascades": 1.5,
            "futures_basis": 1.4,
            "derivatives_oi": 1.3,
            "institutional_flows": 1.6,
            "correlation_breakdown": 1.1,
            "macro_indicators": 0.8,
            "regulatory_sentiment": 0.7
        }
        
        # 실패 패턴 분석 결과
        self.failure_patterns = {}
        
        # 동적 임계값들
        self.dynamic_thresholds = {
            "confidence_threshold": 70.0,
            "volatility_threshold": 0.03,
            "correlation_threshold": 0.7
        }
    
    def init_learning_database(self):
        """학습 데이터베이스 초기화"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 실패 분석 테이블
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS failure_analysis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_id INTEGER NOT NULL,
                failure_type TEXT NOT NULL,
                root_cause TEXT NOT NULL,
                failed_indicators TEXT NOT NULL,
                market_condition TEXT NOT NULL,
                severity_score REAL NOT NULL,
                corrective_action TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                FOREIGN KEY (prediction_id) REFERENCES predictions (id)
            )
        ''')
        
        # 가중치 진화 테이블
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS weight_evolution (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                indicator_name TEXT NOT NULL,
                old_weight REAL NOT NULL,
                new_weight REAL NOT NULL,
                performance_improvement REAL NOT NULL,
                market_condition TEXT NOT NULL,
                adjustment_reason TEXT NOT NULL,
                timestamp TEXT NOT NULL
            )
        ''')
        
        # 학습 성과 테이블
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS learning_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                period_start TEXT NOT NULL,
                period_end TEXT NOT NULL,
                accuracy_before REAL NOT NULL,
                accuracy_after REAL NOT NULL,
                improvement_score REAL NOT NULL,
                key_insights TEXT NOT NULL,
                timestamp TEXT NOT NULL
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def analyze_prediction_failures(self, days: int = 7) -> Dict:
        """최근 실패한 예측들을 분석하여 패턴과 원인 파악"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 최근 실패한 예측들 조회
            cursor.execute('''
                SELECT id, timestamp, prediction_direction, predicted_price, 
                       probability, confidence, leading_indicators, claude_reasoning,
                       actual_price, actual_direction, direction_correct, price_accuracy,
                       market_condition, volatility_regime
                FROM predictions 
                WHERE is_evaluated = TRUE 
                AND direction_correct = FALSE
                AND datetime(timestamp) >= datetime('now', '-' || ? || ' days')
                ORDER BY timestamp DESC
            ''', (days,))
            
            failed_predictions = cursor.fetchall()
            
            if not failed_predictions:
                return {"message": "분석할 실패 예측이 없습니다", "failures": []}
            
            analysis_results = []
            
            for failure in failed_predictions:
                (pred_id, timestamp, pred_dir, pred_price, probability, confidence, 
                 indicators_json, reasoning, actual_price, actual_dir, correct, 
                 price_acc, market_cond, volatility) = failure
                
                # 실패 원인 분석
                failure_analysis = self._analyze_single_failure(
                    pred_id, pred_dir, pred_price, probability, confidence,
                    indicators_json, reasoning, actual_price, actual_dir,
                    market_cond, volatility
                )
                
                analysis_results.append(failure_analysis)
                
                # 실패 분석을 데이터베이스에 저장
                self._save_failure_analysis(pred_id, failure_analysis)
            
            # 패턴 요약
            pattern_summary = self._summarize_failure_patterns(analysis_results)
            
            conn.close()
            
            return {
                "period_days": days,
                "total_failures": len(failed_predictions),
                "failure_analyses": analysis_results,
                "pattern_summary": pattern_summary,
                "recommended_actions": self._generate_corrective_actions(pattern_summary)
            }
            
        except Exception as e:
            self.logger.error(f"실패 분석 오류: {e}")
            return {"error": str(e)}
    
    def _analyze_single_failure(self, pred_id: int, pred_dir: str, pred_price: float, 
                               probability: float, confidence: str, indicators_json: str,
                               reasoning: str, actual_price: float, actual_dir: str,
                               market_cond: str, volatility: str) -> Dict:
        """개별 실패 사례 심층 분석"""
        try:
            indicators = json.loads(indicators_json) if indicators_json else {}
            
            # 실패 유형 분류
            failure_type = self._classify_failure_type(
                pred_dir, actual_dir, pred_price, actual_price, probability
            )
            
            # 근본 원인 분석
            root_cause = self._identify_root_cause(
                indicators, market_cond, volatility, failure_type
            )
            
            # 실패한 지표들 식별
            failed_indicators = self._identify_failed_indicators(
                indicators, pred_dir, actual_dir
            )
            
            # 심각도 점수 (0-10)
            severity_score = self._calculate_failure_severity(
                probability, confidence, pred_price, actual_price
            )
            
            return {
                "prediction_id": pred_id,
                "failure_type": failure_type,
                "root_cause": root_cause,
                "failed_indicators": failed_indicators,
                "severity_score": severity_score,
                "market_condition": market_cond,
                "volatility_regime": volatility,
                "prediction_confidence": confidence,
                "probability": probability,
                "price_deviation_pct": abs(actual_price - pred_price) / pred_price * 100
            }
            
        except Exception as e:
            self.logger.error(f"단일 실패 분석 오류: {e}")
            return {"error": str(e)}
    
    def _classify_failure_type(self, pred_dir: str, actual_dir: str, 
                              pred_price: float, actual_price: float, 
                              probability: float) -> str:
        """실패 유형 분류"""
        price_change_pct = abs(actual_price - pred_price) / pred_price * 100
        
        if pred_dir != actual_dir and probability >= 80:
            return "HIGH_CONFIDENCE_DIRECTION_ERROR"
        elif pred_dir != actual_dir and probability < 60:
            return "LOW_CONFIDENCE_DIRECTION_ERROR"
        elif pred_dir == actual_dir and price_change_pct > 10:
            return "CORRECT_DIRECTION_WRONG_MAGNITUDE"
        elif pred_dir == "NEUTRAL" and price_change_pct > 5:
            return "FALSE_NEUTRAL_PREDICTION"
        else:
            return "GENERAL_PREDICTION_ERROR"
    
    def _identify_root_cause(self, indicators: Dict, market_cond: str, 
                            volatility: str, failure_type: str) -> str:
        """근본 원인 식별"""
        causes = []
        
        # 지표 신뢰도 분석
        if indicators:
            low_confidence_indicators = [
                name for name, data in indicators.items() 
                if isinstance(data, dict) and data.get('confidence', 100) < 50
            ]
            if len(low_confidence_indicators) > 5:
                causes.append("INDICATOR_RELIABILITY_ISSUE")
        
        # 시장 상황 분석
        if market_cond == "HIGH_VOLATILITY" and failure_type.startswith("HIGH_CONFIDENCE"):
            causes.append("VOLATILITY_UNDERESTIMATION")
        
        if volatility == "REGIME_CHANGE":
            causes.append("MARKET_REGIME_SHIFT")
        
        # 외부 요인
        current_hour = datetime.now().hour
        if current_hour in [14, 15, 21, 22]:  # 미국/아시아 시장 겹치는 시간
            causes.append("CROSS_MARKET_INTERFERENCE")
        
        return "|".join(causes) if causes else "UNKNOWN_CAUSE"
    
    def _identify_failed_indicators(self, indicators: Dict, 
                                   pred_dir: str, actual_dir: str) -> List[str]:
        """실패에 기여한 지표들 식별"""
        failed = []
        
        for indicator_name, data in indicators.items():
            if isinstance(data, dict):
                signal = data.get('signal', 'NEUTRAL')
                confidence = data.get('confidence', 50)
                
                # 예측 방향과 반대 신호를 준 지표들
                if ((pred_dir == "BULLISH" and signal == "BEARISH") or
                    (pred_dir == "BEARISH" and signal == "BULLISH")):
                    if confidence > 70:  # 높은 확신으로 잘못된 신호
                        failed.append(indicator_name)
        
        return failed
    
    def _calculate_failure_severity(self, probability: float, confidence: str,
                                   pred_price: float, actual_price: float) -> float:
        """실패 심각도 계산 (0-10)"""
        severity = 0.0
        
        # 확률 기반 (높은 확률일수록 실패시 심각)
        severity += (probability / 100) * 4
        
        # 신뢰도 기반
        confidence_multiplier = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
        severity += confidence_multiplier.get(confidence, 1)
        
        # 가격 오차 기반
        price_error_pct = abs(actual_price - pred_price) / pred_price * 100
        severity += min(price_error_pct / 10, 3)  # 최대 3점
        
        return min(severity, 10.0)
    
    def adapt_indicator_weights(self, analysis_results: List[Dict]) -> Dict:
        """실패 분석 결과를 바탕으로 지표 가중치 적응적 조정"""
        try:
            adjustments = {}
            
            # 실패한 지표들 수집
            failed_indicator_counts = defaultdict(int)
            total_failures = len(analysis_results)
            
            for analysis in analysis_results:
                failed_indicators = analysis.get("failed_indicators", [])
                severity = analysis.get("severity_score", 0)
                
                for indicator in failed_indicators:
                    failed_indicator_counts[indicator] += severity
            
            # 가중치 조정 계산
            for indicator, failure_score in failed_indicator_counts.items():
                if indicator in self.learned_weights:
                    old_weight = self.learned_weights[indicator]
                    
                    # 실패 빈도와 심각도에 따른 가중치 감소
                    penalty_factor = min(failure_score / total_failures, 0.3)
                    new_weight = old_weight * (1 - penalty_factor)
                    
                    # 최소 가중치 보장
                    new_weight = max(new_weight, 0.3)
                    
                    adjustments[indicator] = {
                        "old_weight": old_weight,
                        "new_weight": new_weight,
                        "change_pct": ((new_weight - old_weight) / old_weight) * 100,
                        "failure_score": failure_score
                    }
                    
                    self.learned_weights[indicator] = new_weight
            
            # 성공적인 지표들 가중치 증가
            self._boost_successful_indicators(analysis_results)
            
            # 조정 결과 저장
            self._save_weight_adjustments(adjustments)
            
            return {
                "total_adjustments": len(adjustments),
                "weight_changes": adjustments,
                "adaptation_summary": self._summarize_adaptation(adjustments)
            }
            
        except Exception as e:
            self.logger.error(f"가중치 적응 오류: {e}")
            return {"error": str(e)}
    
    def _boost_successful_indicators(self, analysis_results: List[Dict]):
        """성공적인 지표들의 가중치 증가"""
        # 최근 성공한 예측들에서 기여도 높은 지표들 찾기
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT leading_indicators FROM predictions 
                WHERE is_evaluated = TRUE 
                AND direction_correct = TRUE
                AND datetime(timestamp) >= datetime('now', '-7 days')
            ''')
            
            successful_predictions = cursor.fetchall()
            
            successful_indicator_scores = defaultdict(float)
            
            for (indicators_json,) in successful_predictions:
                if indicators_json:
                    indicators = json.loads(indicators_json)
                    for name, data in indicators.items():
                        if isinstance(data, dict):
                            confidence = data.get('confidence', 50)
                            successful_indicator_scores[name] += confidence / 100
            
            # 성공 지표들 가중치 증가
            for indicator, success_score in successful_indicator_scores.items():
                if indicator in self.learned_weights and success_score > 3:
                    boost_factor = min(success_score / 10, 0.1)  # 최대 10% 증가
                    self.learned_weights[indicator] *= (1 + boost_factor)
                    self.learned_weights[indicator] = min(self.learned_weights[indicator], 2.5)
            
            conn.close()
            
        except Exception as e:
            self.logger.error(f"성공 지표 부스트 오류: {e}")
    
    def optimize_dynamic_thresholds(self) -> Dict:
        """동적 임계값 최적화"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 최근 예측 성과 분석
            cursor.execute('''
                SELECT probability, confidence, direction_correct, price_accuracy,
                       market_condition, volatility_regime
                FROM predictions 
                WHERE is_evaluated = TRUE 
                AND datetime(timestamp) >= datetime('now', '-14 days')
            ''')
            
            results = cursor.fetchall()
            
            if len(results) < 10:
                return {"message": "최적화를 위한 충분한 데이터가 없습니다"}
            
            # 확률별 정확도 분석
            probability_accuracies = defaultdict(list)
            for prob, conf, correct, price_acc, market, vol in results:
                prob_range = int(prob // 10) * 10  # 10% 단위로 그룹화
                probability_accuracies[prob_range].append(1.0 if correct else 0.0)
            
            # 최적 확률 임계값 찾기
            best_threshold = 70
            best_score = 0
            
            for threshold in range(60, 90, 5):
                high_prob_predictions = [r for r in results if r[0] >= threshold]
                if high_prob_predictions:
                    accuracy = sum(1 for r in high_prob_predictions if r[2]) / len(high_prob_predictions)
                    # 정확도와 예측 수의 균형
                    score = accuracy * min(len(high_prob_predictions) / len(results), 1)
                    if score > best_score:
                        best_score = score
                        best_threshold = threshold
            
            # 임계값 업데이트
            old_threshold = self.dynamic_thresholds["confidence_threshold"]
            self.dynamic_thresholds["confidence_threshold"] = best_threshold
            
            conn.close()
            
            return {
                "old_confidence_threshold": old_threshold,
                "new_confidence_threshold": best_threshold,
                "expected_accuracy_improvement": (best_score - 0.7) * 100,
                "optimization_date": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"임계값 최적화 오류: {e}")
            return {"error": str(e)}
    
    def generate_learning_insights(self) -> Dict:
        """학습 결과 인사이트 생성"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 최근 학습 성과
            cursor.execute('''
                SELECT * FROM learning_performance 
                ORDER BY timestamp DESC LIMIT 1
            ''')
            
            latest_performance = cursor.fetchone()
            
            # 가중치 진화 트렌드
            cursor.execute('''
                SELECT indicator_name, AVG(performance_improvement) as avg_improvement
                FROM weight_evolution 
                WHERE datetime(timestamp) >= datetime('now', '-30 days')
                GROUP BY indicator_name
                ORDER BY avg_improvement DESC
            ''')
            
            weight_trends = cursor.fetchall()
            
            # 실패 패턴 빈도
            cursor.execute('''
                SELECT failure_type, COUNT(*) as count, AVG(severity_score) as avg_severity
                FROM failure_analysis 
                WHERE datetime(timestamp) >= datetime('now', '-30 days')
                GROUP BY failure_type
                ORDER BY count DESC
            ''')
            
            failure_patterns = cursor.fetchall()
            
            conn.close()
            
            insights = {
                "learning_summary": {
                    "latest_performance": latest_performance,
                    "total_weight_adjustments": len(self.learned_weights),
                    "avg_weight": statistics.mean(self.learned_weights.values()),
                    "weight_std": statistics.stdev(self.learned_weights.values()) if len(self.learned_weights) > 1 else 0
                },
                "top_performing_indicators": [
                    {"indicator": name, "weight": weight} 
                    for name, weight in sorted(self.learned_weights.items(), 
                                             key=lambda x: x[1], reverse=True)[:5]
                ],
                "improvement_trends": [
                    {"indicator": name, "avg_improvement": improvement} 
                    for name, improvement in weight_trends[:5]
                ],
                "common_failure_patterns": [
                    {"type": ftype, "frequency": count, "avg_severity": severity}
                    for ftype, count, severity in failure_patterns
                ],
                "current_thresholds": self.dynamic_thresholds.copy(),
                "recommendations": self._generate_learning_recommendations()
            }
            
            return insights
            
        except Exception as e:
            self.logger.error(f"인사이트 생성 오류: {e}")
            return {"error": str(e)}
    
    def _generate_learning_recommendations(self) -> List[str]:
        """학습 기반 추천사항 생성"""
        recommendations = []
        
        # 가중치 분산도 체크
        weight_std = statistics.stdev(self.learned_weights.values()) if len(self.learned_weights) > 1 else 0
        if weight_std > 0.5:
            recommendations.append("지표간 가중치 편차가 큽니다. 균형 조정을 고려하세요.")
        
        # 임계값 체크
        if self.dynamic_thresholds["confidence_threshold"] > 85:
            recommendations.append("신뢰도 임계값이 너무 높아 예측 빈도가 줄어들 수 있습니다.")
        
        # 최고 성능 지표
        top_indicator = max(self.learned_weights.items(), key=lambda x: x[1])
        if top_indicator[1] > 2.0:
            recommendations.append(f"{top_indicator[0]} 지표의 성능이 매우 우수합니다. 관련 데이터 소스를 확장 고려하세요.")
        
        return recommendations
    
    def _save_failure_analysis(self, pred_id: int, analysis: Dict):
        """실패 분석 결과 저장"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO failure_analysis 
                (prediction_id, failure_type, root_cause, failed_indicators, 
                 market_condition, severity_score, corrective_action, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                pred_id,
                analysis.get("failure_type", ""),
                analysis.get("root_cause", ""),
                json.dumps(analysis.get("failed_indicators", [])),
                analysis.get("market_condition", ""),
                analysis.get("severity_score", 0),
                "WEIGHT_ADJUSTMENT",
                datetime.now().isoformat()
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"실패 분석 저장 오류: {e}")
    
    def _save_weight_adjustments(self, adjustments: Dict):
        """가중치 조정 기록 저장"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for indicator, adjustment in adjustments.items():
                cursor.execute('''
                    INSERT INTO weight_evolution 
                    (indicator_name, old_weight, new_weight, performance_improvement,
                     market_condition, adjustment_reason, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    indicator,
                    adjustment["old_weight"],
                    adjustment["new_weight"],
                    adjustment["change_pct"],
                    "MIXED",
                    f"Failure-based adjustment: {adjustment['failure_score']:.2f} severity",
                    datetime.now().isoformat()
                ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"가중치 조정 저장 오류: {e}")
    
    def _summarize_failure_patterns(self, analyses: List[Dict]) -> Dict:
        """실패 패턴 요약"""
        if not analyses:
            return {}
        
        failure_types = defaultdict(int)
        root_causes = defaultdict(int)
        total_severity = 0
        
        for analysis in analyses:
            failure_types[analysis.get("failure_type", "UNKNOWN")] += 1
            causes = analysis.get("root_cause", "").split("|")
            for cause in causes:
                if cause.strip():
                    root_causes[cause.strip()] += 1
            total_severity += analysis.get("severity_score", 0)
        
        return {
            "most_common_failure_type": max(failure_types.items(), key=lambda x: x[1])[0],
            "failure_type_distribution": dict(failure_types),
            "root_cause_distribution": dict(root_causes),
            "average_severity": total_severity / len(analyses),
            "critical_failures": sum(1 for a in analyses if a.get("severity_score", 0) >= 8)
        }
    
    def _generate_corrective_actions(self, pattern_summary: Dict) -> List[str]:
        """교정 조치 생성"""
        actions = []
        
        common_failure = pattern_summary.get("most_common_failure_type", "")
        if "HIGH_CONFIDENCE" in common_failure:
            actions.append("높은 확신 예측의 임계값을 상향 조정")
        
        if "VOLATILITY" in str(pattern_summary.get("root_cause_distribution", {})):
            actions.append("변동성 지표의 가중치 증가")
        
        if pattern_summary.get("average_severity", 0) > 7:
            actions.append("전체적인 예측 보수성 증가")
        
        return actions
    
    def _summarize_adaptation(self, adjustments: Dict) -> str:
        """적응 결과 요약"""
        if not adjustments:
            return "조정된 가중치가 없습니다"
        
        total_changes = len(adjustments)
        avg_change = statistics.mean([abs(adj["change_pct"]) for adj in adjustments.values()])
        
        return f"{total_changes}개 지표 가중치 조정 완료, 평균 변화율: {avg_change:.1f}%"

# 테스트 함수
async def test_learning_engine():
    """학습 엔진 테스트"""
    print("🧠 적응적 학습 엔진 테스트...")
    
    engine = AdaptiveLearningEngine()
    
    # 실패 분석
    failure_analysis = engine.analyze_prediction_failures(7)
    print(f"실패 분석: {failure_analysis}")
    
    # 가중치 적응
    if failure_analysis.get("failure_analyses"):
        adaptation_result = engine.adapt_indicator_weights(failure_analysis["failure_analyses"])
        print(f"가중치 적응: {adaptation_result}")
    
    # 임계값 최적화
    threshold_optimization = engine.optimize_dynamic_thresholds()
    print(f"임계값 최적화: {threshold_optimization}")
    
    # 학습 인사이트
    insights = engine.generate_learning_insights()
    print(f"학습 인사이트: {insights}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_learning_engine())