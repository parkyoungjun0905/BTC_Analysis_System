#!/usr/bin/env python3
"""
하이브리드 학습 최적화 시스템
AI와 로컬 학습의 최적 조합으로 정확도 극대화
"""

import json
import asyncio
import aiohttp
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from adaptive_learning_engine import AdaptiveLearningEngine

class HybridLearningOptimizer:
    """AI와 로컬 학습을 결합한 하이브리드 최적화 시스템"""
    
    def __init__(self, claude_api_key: str = None):
        self.claude_api_key = claude_api_key
        self.local_engine = AdaptiveLearningEngine()
        self.logger = logging.getLogger(__name__)
        
        # 하이브리드 학습 전략
        self.learning_strategy = {
            "ai_tasks": [
                "pattern_recognition",
                "root_cause_analysis", 
                "market_regime_detection",
                "explanation_generation"
            ],
            "local_tasks": [
                "statistical_optimization",
                "weight_adjustment",
                "threshold_tuning",
                "performance_tracking"
            ]
        }
        
        # 정확도 개선 메트릭
        self.accuracy_metrics = {
            "baseline_accuracy": 0.72,  # 현재 기준선
            "target_accuracy": 0.85,    # 목표 정확도
            "improvement_tracking": []
        }
        
        # 실시간 성능 가중치
        self.performance_weights = {
            "ai_confidence": 0.4,      # AI 예측 신뢰도
            "local_statistics": 0.3,   # 로컬 통계 분석
            "historical_accuracy": 0.2, # 과거 정확도
            "market_conditions": 0.1   # 시장 상황
        }
    
    async def run_hybrid_learning_cycle(self, prediction_data: Dict, 
                                       market_data: Dict) -> Dict:
        """하이브리드 학습 사이클 실행"""
        try:
            self.logger.info("🔄 하이브리드 학습 사이클 시작")
            results = {"timestamp": datetime.now().isoformat()}
            
            # 1단계: 로컬 통계 분석 (빠른 처리)
            self.logger.info("📊 1/5 - 로컬 통계 분석...")
            local_analysis = await self._run_local_analysis(prediction_data)
            results["local_analysis"] = local_analysis
            
            # 2단계: AI 패턴 인식 (조건부 실행)
            ai_analysis = {}
            if self._should_use_ai_learning(local_analysis):
                self.logger.info("🤖 2/5 - AI 패턴 분석...")
                ai_analysis = await self._run_ai_pattern_analysis(
                    prediction_data, market_data, local_analysis
                )
                results["ai_analysis"] = ai_analysis
            else:
                self.logger.info("💰 2/5 - AI 분석 건너뜀 (비용 절약)")
            
            # 3단계: 하이브리드 인사이트 통합
            self.logger.info("🔗 3/5 - 인사이트 통합...")
            integrated_insights = self._integrate_learning_insights(
                local_analysis, ai_analysis
            )
            results["integrated_insights"] = integrated_insights
            
            # 4단계: 실시간 최적화 적용
            self.logger.info("⚙️ 4/5 - 실시간 최적화 적용...")
            optimization_result = await self._apply_hybrid_optimization(
                integrated_insights
            )
            results["optimization"] = optimization_result
            
            # 5단계: 성능 검증 및 피드백
            self.logger.info("🎯 5/5 - 성능 검증...")
            performance_feedback = self._validate_learning_performance(
                results
            )
            results["performance_feedback"] = performance_feedback
            
            # 정확도 개선 추적
            self._track_accuracy_improvement(results)
            
            self.logger.info("✅ 하이브리드 학습 사이클 완료")
            return results
            
        except Exception as e:
            self.logger.error(f"하이브리드 학습 오류: {e}")
            return {"error": str(e), "timestamp": datetime.now().isoformat()}
    
    async def _run_local_analysis(self, prediction_data: Dict) -> Dict:
        """로컬 통계 분석 실행"""
        try:
            analysis_result = {}
            
            # 실패 패턴 통계 분석
            failure_stats = self.local_engine.analyze_prediction_failures(7)
            analysis_result["failure_statistics"] = failure_stats
            
            # 가중치 최적화 계산
            if failure_stats.get("failure_analyses"):
                weight_optimization = self.local_engine.adapt_indicator_weights(
                    failure_stats["failure_analyses"]
                )
                analysis_result["weight_optimization"] = weight_optimization
            
            # 임계값 통계 최적화
            threshold_optimization = self.local_engine.optimize_dynamic_thresholds()
            analysis_result["threshold_optimization"] = threshold_optimization
            
            # 성능 트렌드 분석
            performance_trend = self._analyze_performance_trend()
            analysis_result["performance_trend"] = performance_trend
            
            return analysis_result
            
        except Exception as e:
            self.logger.error(f"로컬 분석 오류: {e}")
            return {"error": str(e)}
    
    async def _run_ai_pattern_analysis(self, prediction_data: Dict, 
                                      market_data: Dict, 
                                      local_analysis: Dict) -> Dict:
        """AI를 활용한 고급 패턴 분석"""
        if not self.claude_api_key:
            return {"status": "skipped", "reason": "no_api_key"}
            
        try:
            # AI 분석 프롬프트 생성
            analysis_prompt = self._create_ai_learning_prompt(
                prediction_data, market_data, local_analysis
            )
            
            # Claude API 호출
            headers = {
                "Content-Type": "application/json",
                "x-api-key": self.claude_api_key,
                "anthropic-version": "2023-06-01"
            }
            
            data = {
                "model": "claude-3-sonnet-20240229",
                "max_tokens": 2000,
                "messages": [{"role": "user", "content": analysis_prompt}]
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://api.anthropic.com/v1/messages",
                    headers=headers,
                    json=data,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        ai_response = result.get("content", [{}])[0].get("text", "")
                        
                        # AI 응답 구조화
                        structured_response = self._parse_ai_learning_response(ai_response)
                        return structured_response
                    else:
                        error_text = await response.text()
                        self.logger.error(f"AI 분석 API 오류: {response.status} - {error_text}")
                        return {"error": f"API_ERROR_{response.status}"}
            
        except Exception as e:
            self.logger.error(f"AI 패턴 분석 오류: {e}")
            return {"error": str(e)}
    
    def _should_use_ai_learning(self, local_analysis: Dict) -> bool:
        """AI 학습 사용 여부 결정"""
        # 중요한 패턴 변화가 감지되었을 때만 AI 사용
        failure_count = local_analysis.get("failure_statistics", {}).get("total_failures", 0)
        performance_decline = local_analysis.get("performance_trend", {}).get("decline_detected", False)
        
        # 조건: 실패 건수 5건 이상 OR 성능 하락 감지
        return failure_count >= 5 or performance_decline
    
    def _create_ai_learning_prompt(self, prediction_data: Dict, 
                                  market_data: Dict, local_analysis: Dict) -> str:
        """AI 학습 분석 프롬프트 생성"""
        failure_stats = local_analysis.get("failure_statistics", {})
        
        prompt = f"""
당신은 암호화폐 시장 예측 시스템의 학습 최적화 전문가입니다.

현재 시스템 성능:
- 총 실패 예측: {failure_stats.get('total_failures', 0)}건
- 주요 실패 패턴: {failure_stats.get('pattern_summary', {}).get('most_common_failure_type', 'N/A')}
- 평균 심각도: {failure_stats.get('pattern_summary', {}).get('average_severity', 0):.1f}/10

시장 상황:
{json.dumps(market_data, indent=2) if market_data else "시장 데이터 없음"}

로컬 분석 결과:
{json.dumps(local_analysis, indent=2)}

다음 사항들을 분석해주세요:

1. PATTERN_RECOGNITION: 실패 패턴에서 발견되는 숨겨진 시장 신호
2. ROOT_CAUSE_ANALYSIS: 예측 실패의 근본적 원인 3가지
3. MARKET_REGIME_DETECTION: 현재 시장 레짐 변화 징후
4. OPTIMIZATION_RECOMMENDATIONS: 구체적인 개선 방안 5가지
5. RISK_FACTORS: 향후 예측에서 주의해야 할 리스크 요소
6. CONFIDENCE_CALIBRATION: 신뢰도 보정을 위한 제안

각 섹션을 명확히 구분하여 분석 결과를 제공해주세요.
"""
        return prompt
    
    def _parse_ai_learning_response(self, ai_response: str) -> Dict:
        """AI 응답을 구조화된 데이터로 파싱"""
        try:
            parsed_result = {
                "pattern_recognition": self._extract_section(ai_response, "PATTERN_RECOGNITION"),
                "root_cause_analysis": self._extract_section(ai_response, "ROOT_CAUSE_ANALYSIS"),
                "market_regime_detection": self._extract_section(ai_response, "MARKET_REGIME_DETECTION"),
                "optimization_recommendations": self._extract_section(ai_response, "OPTIMIZATION_RECOMMENDATIONS"),
                "risk_factors": self._extract_section(ai_response, "RISK_FACTORS"),
                "confidence_calibration": self._extract_section(ai_response, "CONFIDENCE_CALIBRATION"),
                "raw_response": ai_response
            }
            
            return parsed_result
            
        except Exception as e:
            self.logger.error(f"AI 응답 파싱 오류: {e}")
            return {"raw_response": ai_response, "parsing_error": str(e)}
    
    def _extract_section(self, text: str, section_name: str) -> str:
        """AI 응답에서 특정 섹션 추출"""
        try:
            section_start = text.find(section_name)
            if section_start == -1:
                return "섹션을 찾을 수 없음"
            
            # 다음 섹션 시작까지 또는 텍스트 끝까지
            next_section_patterns = ["PATTERN_RECOGNITION", "ROOT_CAUSE_ANALYSIS", 
                                   "MARKET_REGIME_DETECTION", "OPTIMIZATION_RECOMMENDATIONS",
                                   "RISK_FACTORS", "CONFIDENCE_CALIBRATION"]
            
            section_end = len(text)
            for pattern in next_section_patterns:
                if pattern != section_name:
                    pattern_pos = text.find(pattern, section_start + len(section_name))
                    if pattern_pos != -1 and pattern_pos < section_end:
                        section_end = pattern_pos
            
            section_content = text[section_start + len(section_name):section_end]
            return section_content.strip()
            
        except Exception:
            return "추출 실패"
    
    def _integrate_learning_insights(self, local_analysis: Dict, 
                                   ai_analysis: Dict) -> Dict:
        """로컬과 AI 분석 결과 통합"""
        try:
            integrated_insights = {
                "integration_timestamp": datetime.now().isoformat(),
                "data_sources": {
                    "local_available": bool(local_analysis),
                    "ai_available": bool(ai_analysis and "error" not in ai_analysis)
                }
            }
            
            # 가중치 조정 통합
            if local_analysis.get("weight_optimization"):
                integrated_insights["weight_adjustments"] = local_analysis["weight_optimization"]
            
            # AI 추천사항 통합
            if ai_analysis.get("optimization_recommendations"):
                integrated_insights["ai_recommendations"] = ai_analysis["optimization_recommendations"]
            
            # 리스크 요소 통합
            risk_factors = []
            if local_analysis.get("failure_statistics", {}).get("pattern_summary"):
                risk_factors.append("통계적 패턴 위험")
            if ai_analysis.get("risk_factors"):
                risk_factors.append("AI 식별 위험")
            
            integrated_insights["combined_risk_factors"] = risk_factors
            
            # 신뢰도 보정 통합
            confidence_adjustments = {}
            if local_analysis.get("threshold_optimization"):
                confidence_adjustments["statistical"] = local_analysis["threshold_optimization"]
            if ai_analysis.get("confidence_calibration"):
                confidence_adjustments["ai_calibration"] = ai_analysis["confidence_calibration"]
            
            integrated_insights["confidence_adjustments"] = confidence_adjustments
            
            return integrated_insights
            
        except Exception as e:
            self.logger.error(f"인사이트 통합 오류: {e}")
            return {"error": str(e)}
    
    async def _apply_hybrid_optimization(self, insights: Dict) -> Dict:
        """하이브리드 최적화 결과 적용"""
        try:
            optimization_results = []
            
            # 가중치 조정 적용
            if insights.get("weight_adjustments"):
                self.local_engine.learned_weights.update(
                    insights["weight_adjustments"].get("weight_changes", {})
                )
                optimization_results.append("가중치 업데이트 적용")
            
            # 신뢰도 임계값 조정
            if insights.get("confidence_adjustments", {}).get("statistical"):
                threshold_info = insights["confidence_adjustments"]["statistical"]
                if "new_confidence_threshold" in threshold_info:
                    self.local_engine.dynamic_thresholds["confidence_threshold"] = \
                        threshold_info["new_confidence_threshold"]
                    optimization_results.append("임계값 업데이트 적용")
            
            # 성능 가중치 동적 조정
            self._adjust_performance_weights(insights)
            optimization_results.append("성능 가중치 동적 조정")
            
            return {
                "applied_optimizations": optimization_results,
                "current_weights": dict(self.local_engine.learned_weights),
                "current_thresholds": dict(self.local_engine.dynamic_thresholds),
                "performance_weights": dict(self.performance_weights)
            }
            
        except Exception as e:
            self.logger.error(f"하이브리드 최적화 적용 오류: {e}")
            return {"error": str(e)}
    
    def _adjust_performance_weights(self, insights: Dict):
        """성능 가중치 동적 조정"""
        # AI 분석 품질에 따른 가중치 조정
        if insights.get("data_sources", {}).get("ai_available"):
            self.performance_weights["ai_confidence"] = min(0.5, self.performance_weights["ai_confidence"] * 1.1)
        else:
            self.performance_weights["local_statistics"] = min(0.4, self.performance_weights["local_statistics"] * 1.05)
        
        # 정규화
        total_weight = sum(self.performance_weights.values())
        for key in self.performance_weights:
            self.performance_weights[key] /= total_weight
    
    def _validate_learning_performance(self, results: Dict) -> Dict:
        """학습 성능 검증"""
        try:
            validation_score = 0
            validation_details = []
            
            # 로컬 분석 성공 여부
            if results.get("local_analysis") and "error" not in results["local_analysis"]:
                validation_score += 25
                validation_details.append("로컬 분석 성공")
            
            # AI 분석 성공 여부 (있는 경우)
            if results.get("ai_analysis"):
                if "error" not in results["ai_analysis"]:
                    validation_score += 35
                    validation_details.append("AI 분석 성공")
                else:
                    validation_details.append("AI 분석 실패")
            
            # 통합 인사이트 생성
            if results.get("integrated_insights") and "error" not in results["integrated_insights"]:
                validation_score += 25
                validation_details.append("인사이트 통합 성공")
            
            # 최적화 적용
            if results.get("optimization") and "error" not in results["optimization"]:
                validation_score += 15
                validation_details.append("최적화 적용 성공")
            
            performance_grade = (
                "EXCELLENT" if validation_score >= 90 else
                "GOOD" if validation_score >= 70 else
                "FAIR" if validation_score >= 50 else
                "POOR"
            )
            
            return {
                "validation_score": validation_score,
                "performance_grade": performance_grade,
                "validation_details": validation_details,
                "recommendation": self._get_performance_recommendation(validation_score)
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def _get_performance_recommendation(self, score: int) -> str:
        """성능 점수에 따른 추천사항"""
        if score >= 90:
            return "시스템이 최적 상태로 작동 중입니다."
        elif score >= 70:
            return "대체로 양호하나 일부 개선 가능한 영역이 있습니다."
        elif score >= 50:
            return "성능 개선이 필요합니다. AI 분석 빈도를 늘려보세요."
        else:
            return "시스템 점검이 필요합니다. 구성 요소들을 확인해주세요."
    
    def _analyze_performance_trend(self) -> Dict:
        """성능 트렌드 분석"""
        try:
            # 간단한 트렌드 분석
            recent_improvements = len(self.accuracy_metrics["improvement_tracking"])
            
            if recent_improvements < 3:
                trend = "STABLE"
                decline_detected = False
            elif recent_improvements > 5:
                trend = "IMPROVING"  
                decline_detected = False
            else:
                trend = "DECLINING"
                decline_detected = True
            
            return {
                "trend": trend,
                "decline_detected": decline_detected,
                "improvement_count": recent_improvements,
                "current_accuracy": self.accuracy_metrics.get("baseline_accuracy", 0.72)
            }
            
        except Exception:
            return {"trend": "UNKNOWN", "decline_detected": False}
    
    def _track_accuracy_improvement(self, results: Dict):
        """정확도 개선 추적"""
        try:
            improvement_record = {
                "timestamp": datetime.now().isoformat(),
                "local_success": "error" not in results.get("local_analysis", {}),
                "ai_success": results.get("ai_analysis") and "error" not in results.get("ai_analysis", {}),
                "optimization_applied": "error" not in results.get("optimization", {})
            }
            
            self.accuracy_metrics["improvement_tracking"].append(improvement_record)
            
            # 최근 10개 기록만 유지
            if len(self.accuracy_metrics["improvement_tracking"]) > 10:
                self.accuracy_metrics["improvement_tracking"] = \
                    self.accuracy_metrics["improvement_tracking"][-10:]
                    
        except Exception as e:
            self.logger.error(f"정확도 추적 오류: {e}")

# 테스트 함수
async def test_hybrid_optimizer():
    """하이브리드 학습 최적화 테스트"""
    print("🧠 하이브리드 학습 최적화 시스템 테스트")
    
    optimizer = HybridLearningOptimizer()
    
    # 테스트 데이터
    prediction_data = {"recent_predictions": 10, "failures": 3}
    market_data = {"volatility": 0.045, "trend": "SIDEWAYS"}
    
    result = await optimizer.run_hybrid_learning_cycle(prediction_data, market_data)
    print(f"테스트 결과: {result}")

if __name__ == "__main__":
    asyncio.run(test_hybrid_optimizer())