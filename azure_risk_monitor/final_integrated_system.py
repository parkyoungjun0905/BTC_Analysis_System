"""
최종 통합 리스크 모니터링 시스템
모든 개선사항 통합 + 24시간 실시간 감시
"""

import asyncio
import logging
from typing import Dict, Optional
from datetime import datetime, timedelta
import os
import json

# 핵심 모듈들
from enhanced_19_indicators import Enhanced19IndicatorSystem
from time_series_analyzer import TimeSeriesAnalyzer
from claude_predictor import ClaudePricePredictor
from prediction_tracker import PredictionTracker
from beginner_friendly_explainer import BeginnerFriendlyExplainer
from enhanced_telegram_notifier import EnhancedTelegramNotifier
from adaptive_learning_engine import AdaptiveLearningEngine
from hybrid_learning_optimizer import HybridLearningOptimizer
from accuracy_enhancement_roadmap import AccuracyEnhancementRoadmap
from custom_alert_system import CustomAlertSystem
from telegram_command_handler import TelegramCommandHandler

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FinalIntegratedRiskMonitor:
    """최종 통합 리스크 모니터링 시스템"""
    
    def __init__(self):
        # 핵심 컴포넌트들
        self.indicator_system = Enhanced19IndicatorSystem()
        self.time_series = TimeSeriesAnalyzer()
        self.claude_predictor = ClaudePricePredictor()
        self.prediction_tracker = PredictionTracker()
        self.explainer = BeginnerFriendlyExplainer()
        self.telegram = EnhancedTelegramNotifier()
        
        # 🧠 업그레이드된 학습 시스템
        self.learning_engine = AdaptiveLearningEngine()
        self.hybrid_optimizer = HybridLearningOptimizer(
            claude_api_key=os.environ.get('CLAUDE_API_KEY')
        )
        self.accuracy_roadmap = AccuracyEnhancementRoadmap()
        
        # 📱 맞춤형 알림 시스템
        self.custom_alerts = CustomAlertSystem()
        self.telegram_handler = TelegramCommandHandler(
            bot_token=os.environ.get('TELEGRAM_BOT_TOKEN', ''),
            chat_id=os.environ.get('TELEGRAM_CHAT_ID', '')
        )
        
        self.logger = logger
        
        # 시스템 상태
        self.monitoring_active = True
        self.last_alert_time = None
        self.alert_cooldown = 300  # 5분 쿨다운
        
        # 성능 통계
        self.daily_stats = {
            "total_predictions": 0,
            "correct_predictions": 0,
            "alerts_sent": 0,
            "high_priority_alerts": 0,
            "errors": 0
        }
    
    async def run_complete_analysis_cycle(self) -> Dict:
        """완전한 분석 사이클 실행"""
        try:
            self.logger.info("="*60)
            self.logger.info("🚀 통합 분석 사이클 시작")
            start_time = datetime.now()
            
            # 1단계: 19개 지표 수집
            self.logger.info("📊 1/6 - 19개 선행지표 수집 중...")
            indicators = await self._collect_all_indicators()
            
            if not indicators:
                raise Exception("지표 수집 실패")
            
            # 2단계: 시계열 분석
            self.logger.info("📈 2/6 - 시계열 패턴 분석 중...")
            time_series_result = await self._analyze_time_series(indicators)
            
            # 3단계: AI 예측 (조건부)
            self.logger.info("🤖 3/6 - AI 예측 분석 중...")
            ai_prediction = await self._get_ai_prediction(
                indicators, 
                time_series_result
            )
            
            # 4단계: 예측 통합 및 검증
            self.logger.info("🔍 4/6 - 예측 통합 및 검증 중...")
            final_prediction = self._integrate_predictions(
                indicators,
                time_series_result,
                ai_prediction
            )
            
            # 5단계: 업그레이드된 하이브리드 학습 (매 8번째 실행시)
            learning_result = None
            prediction_count = self.daily_stats["total_predictions"]
            
            if prediction_count % 8 == 0:
                self.logger.info("🧠 5/7 - 하이브리드 학습 엔진 실행 중...")
                learning_result = await self._run_hybrid_learning_cycle(
                    final_prediction, indicators, time_series_result
                )
            elif prediction_count % 20 == 0:
                self.logger.info("🎯 5/7 - 정확도 향상 로드맵 실행 중...")
                roadmap_result = await self._run_accuracy_roadmap()
                learning_result = {"roadmap_execution": roadmap_result}
            
            # 6단계: 알림 결정
            self.logger.info("🔔 6/7 - 알림 필요성 판단 중...")
            should_alert, alert_priority = self._determine_alert_need(final_prediction)
            
            # 7단계: 맞춤 알림 체크 (매번 실행)
            self.logger.info("📱 7/8 - 맞춤 알림 조건 체크 중...")
            custom_alerts_sent = await self._check_and_send_custom_alerts(indicators)
            
            # 8단계: 텔레그램 명령어 처리
            commands_processed = await self._process_telegram_commands()
            
            # 9단계: 시스템 알림 발송 (필요시)
            alert_sent = False
            if should_alert:
                self.logger.info(f"📨 9/9 - {alert_priority} 우선순위 알림 발송 중...")
                alert_sent = await self._send_alert(
                    final_prediction,
                    indicators,
                    time_series_result,
                    alert_priority
                )
            else:
                self.logger.info("📌 9/9 - 시스템 알림 기준 미달")
            
            # 분석 시간
            analysis_duration = (datetime.now() - start_time).total_seconds()
            
            # 결과 정리
            result = {
                "status": "success",
                "timestamp": datetime.now().isoformat(),
                "analysis_duration": f"{analysis_duration:.2f}초",
                "prediction": final_prediction,
                "alert_sent": alert_sent,
                "alert_priority": alert_priority if alert_sent else None,
                "custom_alerts_sent": custom_alerts_sent,
                "commands_processed": commands_processed,
                "indicators_collected": len(indicators.get("indicators", {})),
                "time_series_pattern": time_series_result.get("pattern_found", False),
                "learning_executed": learning_result is not None,
                "system_health": "healthy"
            }
            
            # 통계 업데이트
            self._update_statistics(final_prediction, alert_sent, alert_priority)
            
            # 예측 기록 저장
            self._record_prediction(final_prediction, indicators, time_series_result)
            
            self.logger.info(f"✅ 분석 완료 ({analysis_duration:.2f}초)")
            self.logger.info(f"📊 예측: {final_prediction.get('direction')} "
                           f"{final_prediction.get('probability')}% "
                           f"({final_prediction.get('confidence')})")
            self.logger.info("="*60)
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ 분석 사이클 오류: {e}")
            self.daily_stats["errors"] += 1
            
            # 오류 알림
            await self.telegram.send_error_alert(
                "Analysis Cycle Error",
                str(e)
            )
            
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def _collect_all_indicators(self) -> Dict:
        """모든 지표 수집"""
        try:
            # 19개 지표 시스템 실행
            indicators = await self.indicator_system.collect_enhanced_19_indicators()
            
            # 수집 상태 로깅
            if indicators:
                metadata = indicators.get("metadata", {})
                self.logger.info(f"✅ {metadata.get('total_indicators', 0)}개 지표 수집 완료")
                
                # 현재 가격 저장 (시계열용)
                current_price = metadata.get("current_price", 0)
                if current_price > 0:
                    await self.time_series.store_realtime_data(
                        {"price_data": {"current_price": current_price, "volume_24h": 0}},
                        indicators.get("indicators", {})
                    )
            
            return indicators
            
        except Exception as e:
            self.logger.error(f"지표 수집 오류: {e}")
            return {}
    
    async def _analyze_time_series(self, indicators: Dict) -> Dict:
        """시계열 패턴 분석"""
        try:
            # 시계열 분석 실행
            time_series_result = await self.time_series.analyze_time_series_patterns()
            
            if time_series_result.get("pattern_found"):
                self.logger.info(f"✅ 유사 패턴 발견: "
                               f"{time_series_result.get('similar_patterns_count', 0)}개 "
                               f"({time_series_result.get('confidence', 0):.0f}% 일치)")
            else:
                self.logger.info("📊 명확한 시계열 패턴 없음")
            
            return time_series_result
            
        except Exception as e:
            self.logger.error(f"시계열 분석 오류: {e}")
            return {"pattern_found": False, "confidence": 0}
    
    async def _get_ai_prediction(self, indicators: Dict, time_series: Dict) -> Dict:
        """AI 예측 (조건부)"""
        try:
            # 신뢰도 확인
            composite = indicators.get("composite_analysis", {})
            confidence = composite.get("confidence", 0)
            current_hour = datetime.now().hour
            
            # 시간대별 임계값
            should_use_claude = self._should_use_claude_api(confidence, current_hour)
            
            if should_use_claude:
                self.logger.info("🤖 Claude AI 분석 실행")
                
                # 현재 데이터 준비
                current_data = {
                    "price_data": {
                        "current_price": indicators.get("metadata", {}).get("current_price", 0),
                        "volume_24h": 25000000000
                    }
                }
                
                # Claude 예측 요청
                prediction = await self.claude_predictor.analyze_market_signals(
                    current_data,
                    []  # historical_data는 시계열에서 처리
                )
                
                return prediction
            else:
                self.logger.info("💰 로컬 분석만 사용 (비용 절약)")
                return self._generate_local_prediction(indicators, time_series)
                
        except Exception as e:
            self.logger.error(f"AI 예측 오류: {e}")
            return self._generate_local_prediction(indicators, time_series)
    
    def _should_use_claude_api(self, confidence: float, hour: int) -> bool:
        """Claude API 사용 여부 결정"""
        # 중요시간 (09-11, 15-17, 21-23시 한국시간)
        if hour in [9, 10, 15, 16, 21, 22]:
            return confidence >= 60
        # 일반시간
        elif hour in range(7, 23):
            return confidence >= 75
        # 한가시간
        else:
            return confidence >= 90
    
    def _generate_local_prediction(self, indicators: Dict, time_series: Dict) -> Dict:
        """로컬 예측 생성"""
        composite = indicators.get("composite_analysis", {})
        ts_prediction = time_series.get("prediction", "NEUTRAL")
        
        # 지표 기반 예측
        direction = composite.get("overall_signal", "NEUTRAL")
        if "BULLISH" in direction:
            direction = "BULLISH"
        elif "BEARISH" in direction:
            direction = "BEARISH"
        else:
            direction = "NEUTRAL"
        
        # 확률 계산
        confidence = composite.get("confidence", 50)
        if ts_prediction == direction:
            confidence = min(confidence * 1.2, 95)
        
        return {
            "prediction": {
                "direction": direction,
                "probability": confidence,
                "confidence": "HIGH" if confidence > 80 else "MEDIUM" if confidence > 60 else "LOW",
                "target_price": self._estimate_target_price(
                    indicators.get("metadata", {}).get("current_price", 0),
                    direction,
                    confidence
                ),
                "timeframe": "6-12시간",
                "source": "local_analysis"
            }
        }
    
    def _estimate_target_price(self, current_price: float, direction: str, confidence: float) -> float:
        """목표가 추정"""
        if current_price == 0:
            return 0
        
        # 신뢰도에 따른 변동폭
        change_percent = (confidence / 100) * 0.05  # 최대 5%
        
        if direction == "BULLISH":
            return current_price * (1 + change_percent)
        elif direction == "BEARISH":
            return current_price * (1 - change_percent)
        else:
            return current_price
    
    def _integrate_predictions(
        self, 
        indicators: Dict, 
        time_series: Dict,
        ai_prediction: Dict
    ) -> Dict:
        """모든 예측 통합"""
        
        # 각 소스에서 예측 추출
        indicator_pred = self._extract_indicator_prediction(indicators)
        ts_pred = time_series.get("prediction", "NEUTRAL")
        ai_pred = ai_prediction.get("prediction", {})
        
        # 앙상블 예측
        predictions = [indicator_pred, ts_pred, ai_pred.get("direction", "NEUTRAL")]
        
        # 다수결
        bullish_count = predictions.count("BULLISH")
        bearish_count = predictions.count("BEARISH")
        
        if bullish_count >= 2:
            final_direction = "BULLISH"
        elif bearish_count >= 2:
            final_direction = "BEARISH"
        else:
            final_direction = "NEUTRAL"
        
        # 신뢰도 계산
        agreement = max(bullish_count, bearish_count)
        base_confidence = 50 + (agreement - 1) * 25
        
        # AI 예측이 있으면 가중치 증가
        if ai_pred:
            final_probability = (base_confidence * 0.6 + ai_pred.get("probability", 50) * 0.4)
        else:
            final_probability = base_confidence
        
        return {
            "direction": final_direction,
            "probability": min(final_probability, 95),
            "confidence": "VERY_HIGH" if agreement == 3 else "HIGH" if agreement == 2 else "MEDIUM",
            "target_price": ai_pred.get("target_price", 0) if ai_pred else 0,
            "timeframe": ai_pred.get("timeframe", "6-12시간") if ai_pred else "6-12시간",
            "agreement_count": agreement,
            "sources": {
                "indicators": indicator_pred,
                "time_series": ts_pred,
                "ai": ai_pred.get("direction") if ai_pred else None
            }
        }
    
    def _extract_indicator_prediction(self, indicators: Dict) -> str:
        """지표에서 예측 추출"""
        composite = indicators.get("composite_analysis", {})
        signal = composite.get("overall_signal", "NEUTRAL")
        
        if "BULLISH" in signal:
            return "BULLISH"
        elif "BEARISH" in signal:
            return "BEARISH"
        else:
            return "NEUTRAL"
    
    def _determine_alert_need(self, prediction: Dict) -> tuple:
        """알림 필요성 판단"""
        # 기본 조건
        confidence = prediction.get("confidence", "LOW")
        probability = prediction.get("probability", 50)
        direction = prediction.get("direction", "NEUTRAL")
        
        # 중립은 알림 안함
        if direction == "NEUTRAL":
            return False, None
        
        # 쿨다운 체크
        if self.last_alert_time:
            time_since_last = (datetime.now() - self.last_alert_time).seconds
            if time_since_last < self.alert_cooldown:
                self.logger.info(f"⏰ 알림 쿨다운 중 ({self.alert_cooldown - time_since_last}초 남음)")
                return False, None
        
        # 과거 성과 확인
        accuracy_metrics = self.prediction_tracker.get_accuracy_metrics()
        should_send = self.prediction_tracker.should_send_alert(
            {"prediction": prediction},
            accuracy_metrics
        )
        
        if not should_send:
            return False, None
        
        # 우선순위 결정
        if confidence == "VERY_HIGH" and probability > 90:
            priority = "CRITICAL"
        elif confidence == "HIGH" and probability > 80:
            priority = "HIGH"
        elif confidence == "MEDIUM" and probability > 70:
            priority = "MEDIUM"
        else:
            priority = "LOW"
        
        # LOW 우선순위는 알림 안함
        if priority == "LOW":
            return False, None
        
        return True, priority
    
    async def _send_alert(
        self,
        prediction: Dict,
        indicators: Dict,
        time_series: Dict,
        priority: str
    ) -> bool:
        """알림 발송"""
        try:
            # 시스템 성과 가져오기
            system_performance = self.prediction_tracker.get_accuracy_metrics()
            
            # 초보자 친화적 설명 추가
            explained_prediction = self.explainer.explain_prediction(
                prediction,
                indicators
            )
            
            # 향상된 알림 발송
            success = await self.telegram.send_prediction_alert(
                prediction,
                indicators,
                time_series,
                system_performance
            )
            
            if success:
                self.last_alert_time = datetime.now()
                self.logger.info(f"✅ {priority} 알림 발송 성공")
                
                # 통계 업데이트
                self.daily_stats["alerts_sent"] += 1
                if priority in ["CRITICAL", "HIGH"]:
                    self.daily_stats["high_priority_alerts"] += 1
            
            return success
            
        except Exception as e:
            self.logger.error(f"알림 발송 실패: {e}")
            return False
    
    def _update_statistics(self, prediction: Dict, alert_sent: bool, priority: str):
        """통계 업데이트"""
        self.daily_stats["total_predictions"] += 1
        
        # 여기서는 실제 결과를 나중에 확인해야 함
        # 예측 정확도는 prediction_tracker가 처리
    
    def _record_prediction(self, prediction: Dict, indicators: Dict, time_series: Dict):
        """예측 기록"""
        try:
            # 현재 데이터
            current_data = {
                "price_data": {
                    "current_price": indicators.get("metadata", {}).get("current_price", 0),
                    "volume_24h": 0
                }
            }
            
            # 선행지표
            leading_indicators = {
                "indicators": indicators,
                "time_series": time_series
            }
            
            # 예측 기록 저장
            self.prediction_tracker.record_prediction(
                {"prediction": prediction},
                current_data,
                leading_indicators
            )
            
        except Exception as e:
            self.logger.error(f"예측 기록 실패: {e}")
    
    async def send_daily_summary(self):
        """일일 요약 발송"""
        try:
            # 현재 가격 가져오기
            indicators = await self.indicator_system.collect_enhanced_19_indicators()
            current_price = indicators.get("metadata", {}).get("current_price", 0)
            
            # 통계 준비
            stats = {
                **self.daily_stats,
                "close_price": current_price,
                "accuracy": (self.daily_stats["correct_predictions"] / 
                           max(self.daily_stats["total_predictions"], 1)) * 100
            }
            
            # 요약 발송
            await self.telegram.send_daily_summary(stats)
            
            # 통계 리셋
            self.daily_stats = {
                "total_predictions": 0,
                "correct_predictions": 0,
                "alerts_sent": 0,
                "high_priority_alerts": 0,
                "errors": 0
            }
            
        except Exception as e:
            self.logger.error(f"일일 요약 발송 실패: {e}")
    
    async def run_continuous_monitoring(self):
        """24시간 연속 모니터링"""
        self.logger.info("🚀 24시간 연속 모니터링 시작")
        
        while self.monitoring_active:
            try:
                # 현재 시간 확인
                current_hour = datetime.now().hour
                
                # 시간대별 실행 간격 결정
                if current_hour in [9, 10, 15, 16, 21, 22]:
                    # 중요시간: 5분마다
                    interval = 300
                    self.logger.info("⚡ 중요시간 - 5분 간격 모니터링")
                elif current_hour in range(7, 23):
                    # 일반시간: 30분마다
                    interval = 1800
                    self.logger.info("📊 일반시간 - 30분 간격 모니터링")
                else:
                    # 한가시간: 1시간마다
                    interval = 3600
                    self.logger.info("🌙 한가시간 - 1시간 간격 모니터링")
                
                # 분석 실행
                await self.run_complete_analysis_cycle()
                
                # 자정에 일일 요약
                if current_hour == 0 and datetime.now().minute < 5:
                    await self.send_daily_summary()
                
                # 대기
                self.logger.info(f"⏰ 다음 분석까지 {interval//60}분 대기...")
                await asyncio.sleep(interval)
                
            except KeyboardInterrupt:
                self.logger.info("사용자에 의해 중단됨")
                break
            except Exception as e:
                self.logger.error(f"모니터링 오류: {e}")
                await asyncio.sleep(60)  # 오류 시 1분 후 재시도
        
        self.logger.info("모니터링 종료")
    
    async def _run_hybrid_learning_cycle(self, prediction: Dict, 
                                       indicators: Dict, 
                                       time_series: Dict) -> Dict:
        """하이브리드 학습 사이클 실행"""
        try:
            self.logger.info("🔄 하이브리드 학습 시스템 시작")
            
            # 예측 데이터 준비
            prediction_data = {
                "recent_prediction": prediction,
                "prediction_count": self.daily_stats.get("total_predictions", 0),
                "recent_accuracy": self.daily_stats.get("accuracy_rate", 0.72)
            }
            
            # 시장 데이터 준비
            market_data = {
                "current_price": indicators.get("metadata", {}).get("current_price", 0),
                "volatility": self._calculate_current_volatility(indicators),
                "market_regime": self._detect_current_market_regime(indicators),
                "time_series_signals": time_series
            }
            
            # 하이브리드 학습 실행
            learning_result = await self.hybrid_optimizer.run_hybrid_learning_cycle(
                prediction_data, market_data
            )
            
            # 학습 결과 적용
            if "error" not in learning_result:
                await self._apply_hybrid_learning_results(learning_result)
                
                # 텔레그램으로 학습 결과 알림
                if learning_result.get("ai_analysis"):
                    await self._send_learning_update_alert(learning_result)
            
            self.logger.info("✅ 하이브리드 학습 완료")
            return learning_result
            
        except Exception as e:
            self.logger.error(f"하이브리드 학습 오류: {e}")
            return {"error": str(e)}
    
    async def _run_accuracy_roadmap(self) -> Dict:
        """정확도 향상 로드맵 실행"""
        try:
            self.logger.info("🎯 정확도 향상 로드맵 실행")
            
            # 현재 단계 실행
            roadmap_result = await self.accuracy_roadmap.execute_current_phase()
            
            # 단계 완료시 텔레그램 알림
            if roadmap_result.get("phase_completion", {}).get("phase_completed"):
                await self._send_phase_completion_alert(roadmap_result)
            
            self.logger.info("✅ 로드맵 실행 완료")
            return roadmap_result
            
        except Exception as e:
            self.logger.error(f"로드맵 실행 오류: {e}")
            return {"error": str(e)}
    
    def _apply_learned_weights(self):
        """학습된 가중치를 지표 시스템에 적용"""
        try:
            learned_weights = self.learning_engine.learned_weights
            
            # 19개 지표 시스템의 가중치 업데이트
            if hasattr(self.indicator_system, 'indicator_weights'):
                for indicator, weight in learned_weights.items():
                    if indicator in self.indicator_system.indicator_weights:
                        old_weight = self.indicator_system.indicator_weights[indicator]
                        self.indicator_system.indicator_weights[indicator] = weight
                        
                        if abs(weight - old_weight) > 0.1:  # 10% 이상 변화시 로그
                            self.logger.info(f"📊 {indicator} 가중치 조정: {old_weight:.2f} → {weight:.2f}")
                
                self.logger.info("🔄 학습된 가중치 적용 완료")
            
        except Exception as e:
            self.logger.error(f"가중치 적용 오류: {e}")
    
    def _calculate_current_volatility(self, indicators: Dict) -> float:
        """현재 변동성 계산"""
        try:
            # ATR 또는 기타 변동성 지표에서 계산
            atr_data = indicators.get("additional_free", {}).get("atr", {})
            if atr_data and "current_value" in atr_data:
                return float(atr_data["current_value"])
            
            # 기본값
            return 0.045
            
        except Exception:
            return 0.045
    
    def _detect_current_market_regime(self, indicators: Dict) -> str:
        """현재 시장 레짐 감지"""
        try:
            # 여러 지표로 시장 상태 판단
            composite = indicators.get("composite_analysis", {})
            overall_signal = composite.get("overall_signal", "NEUTRAL")
            confidence = composite.get("confidence", 50)
            
            # 트렌드 강도에 따른 레짐 분류
            if overall_signal == "BULLISH" and confidence > 75:
                return "BULL_MARKET"
            elif overall_signal == "BEARISH" and confidence > 75:
                return "BEAR_MARKET"
            else:
                return "SIDEWAYS_MARKET"
                
        except Exception:
            return "SIDEWAYS_MARKET"
    
    async def _apply_hybrid_learning_results(self, learning_result: Dict):
        """하이브리드 학습 결과 적용"""
        try:
            # 가중치 업데이트
            optimization = learning_result.get("optimization", {})
            if "current_weights" in optimization:
                current_weights = optimization["current_weights"]
                for indicator, weight in current_weights.items():
                    if hasattr(self.indicator_system, 'indicator_weights') and \
                       indicator in self.indicator_system.indicator_weights:
                        self.indicator_system.indicator_weights[indicator] = weight
            
            # 임계값 업데이트
            if "current_thresholds" in optimization:
                current_thresholds = optimization["current_thresholds"]
                self.hybrid_optimizer.local_engine.dynamic_thresholds.update(current_thresholds)
            
            self.logger.info("🔄 하이브리드 학습 결과 적용 완료")
            
        except Exception as e:
            self.logger.error(f"하이브리드 학습 결과 적용 오류: {e}")
    
    async def _send_learning_update_alert(self, learning_result: Dict):
        """학습 업데이트 알림"""
        try:
            ai_analysis = learning_result.get("ai_analysis", {})
            
            message = "🧠 **AI 학습 업데이트**\n\n"
            
            # 패턴 인식 결과
            pattern_recognition = ai_analysis.get("pattern_recognition", "")
            if pattern_recognition and len(pattern_recognition) > 50:
                message += f"🔍 **패턴 발견**: {pattern_recognition[:100]}...\n\n"
            
            # 최적화 추천
            optimization = ai_analysis.get("optimization_recommendations", "")
            if optimization and len(optimization) > 30:
                message += f"⚙️ **최적화 제안**: {optimization[:100]}...\n\n"
            
            # 리스크 요소
            risk_factors = ai_analysis.get("risk_factors", "")
            if risk_factors and len(risk_factors) > 30:
                message += f"⚠️ **주의사항**: {risk_factors[:100]}...\n\n"
            
            message += "💡 시스템이 AI를 통해 스스로 학습하고 있습니다!"
            
            await self.telegram.send_message(message)
            
        except Exception as e:
            self.logger.error(f"학습 알림 전송 오류: {e}")
    
    async def _send_phase_completion_alert(self, roadmap_result: Dict):
        """단계 완료 알림"""
        try:
            phase_completion = roadmap_result.get("phase_completion", {})
            
            current_accuracy = phase_completion.get("current_accuracy", 0)
            target_accuracy = phase_completion.get("target_accuracy", 0)
            
            message = "🎯 **정확도 향상 단계 완료!**\n\n"
            message += f"📊 **달성 정확도**: {current_accuracy:.1%}\n"
            message += f"🎯 **목표 정확도**: {target_accuracy:.1%}\n"
            
            next_phase = roadmap_result.get("next_phase", "")
            if next_phase and next_phase != "모든 단계 완료!":
                message += f"➡️ **다음 단계**: {next_phase}\n"
            else:
                message += "🏆 **모든 단계 완료! 시스템 마스터리 달성!**\n"
            
            message += "\n🚀 시스템이 체계적으로 발전하고 있습니다!"
            
            await self.telegram.send_message(message)
            
        except Exception as e:
            self.logger.error(f"단계 완료 알림 오류: {e}")
    
    async def _check_and_send_custom_alerts(self, indicators: Dict) -> int:
        """맞춤 알림 조건 체크 및 발송"""
        try:
            # 현재 사용자 ID (설정된 채팅 ID 사용)
            user_id = os.environ.get('TELEGRAM_CHAT_ID', 'default_user')
            
            # 맞춤 알림 조건 체크
            triggered_alerts = await self.custom_alerts.check_custom_alerts(indicators, user_id)
            
            alerts_sent = 0
            for alert in triggered_alerts:
                # 알림 메시지 포맷
                alert_message = self.custom_alerts.format_triggered_alert(alert)
                
                # 텔레그램으로 발송
                success = await self.telegram_handler.send_telegram_message(alert_message)
                if success:
                    alerts_sent += 1
                    self.logger.info(f"📱 맞춤 알림 발송: {alert['indicator']} {alert['operator']} {alert['threshold']}")
                
                # 알림 간 간격
                await asyncio.sleep(1)
            
            if alerts_sent > 0:
                self.logger.info(f"✅ 맞춤 알림 {alerts_sent}개 발송 완료")
            
            return alerts_sent
            
        except Exception as e:
            self.logger.error(f"맞춤 알림 체크 오류: {e}")
            return 0
    
    async def _process_telegram_commands(self) -> int:
        """텔레그램 명령어 처리"""
        try:
            # 새로운 명령어 처리
            commands_processed = await self.telegram_handler.process_and_respond()
            
            if commands_processed > 0:
                self.logger.info(f"📱 텔레그램 명령어 {commands_processed}개 처리됨")
            
            return commands_processed
            
        except Exception as e:
            self.logger.error(f"텔레그램 명령어 처리 오류: {e}")
            return 0

async def main():
    """메인 실행 함수"""
    print("\n" + "="*70)
    print("🚀 최종 통합 비트코인 리스크 모니터링 시스템")
    print("="*70)
    print("""
    ✅ 핵심 기능:
    • 19개 선행지표 실시간 분석
    • 시계열 패턴 매칭
    • Claude AI 예측 (조건부)
    • 초보자 친화적 설명
    • 24시간 연속 모니터링
    • 정확도 기반 필터링
    """)
    
    monitor = FinalIntegratedRiskMonitor()
    
    # 단일 분석 테스트
    print("\n📊 단일 분석 사이클 테스트...")
    result = await monitor.run_complete_analysis_cycle()
    
    if result.get("status") == "success":
        print(f"\n✅ 테스트 성공!")
        print(f"• 예측: {result['prediction']['direction']} {result['prediction']['probability']:.0f}%")
        print(f"• 신뢰도: {result['prediction']['confidence']}")
        print(f"• 알림 발송: {'예' if result['alert_sent'] else '아니오'}")
        
        # 연속 모니터링 시작 여부 확인
        # response = input("\n24시간 연속 모니터링을 시작하시겠습니까? (y/n): ")
        # if response.lower() == 'y':
        #     await monitor.run_continuous_monitoring()
    else:
        print(f"\n❌ 테스트 실패: {result.get('error')}")

if __name__ == "__main__":
    # 환경변수 체크
    required_env = {
        "CRYPTOQUANT_API_KEY": os.environ.get('CRYPTOQUANT_API_KEY'),
        "CLAUDE_API_KEY": os.environ.get('CLAUDE_API_KEY'),
        "TELEGRAM_BOT_TOKEN": os.environ.get('TELEGRAM_BOT_TOKEN'),
        "TELEGRAM_CHAT_ID": os.environ.get('TELEGRAM_CHAT_ID')
    }
    
    print("\n📋 환경변수 상태:")
    for name, value in required_env.items():
        status = "✅" if value else "❌"
        print(f"  • {name}: {status}")
    
    if not all(required_env.values()):
        print("\n⚠️ 일부 환경변수가 설정되지 않았습니다.")
        print("시뮬레이션 모드로 실행됩니다.")
    
    # 실행
    asyncio.run(main())