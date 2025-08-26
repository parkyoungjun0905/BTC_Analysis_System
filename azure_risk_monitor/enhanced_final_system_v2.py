"""
강화된 최종 통합 시스템 V2
예측적 가격 알림 시스템 포함
"""

import asyncio
import logging
from typing import Dict, Optional, List
from datetime import datetime, timedelta
import os
import json
import numpy as np

# 기존 핵심 모듈들
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

# 🎯 새로운 예측 시스템
from predictive_price_alert_system import (
    PredictivePriceAlertSystem, 
    PricePrediction,
    BacktestValidator
)

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EnhancedFinalSystemV2:
    """강화된 최종 통합 시스템 V2 - 예측적 가격 알림 포함"""
    
    def __init__(self):
        # 기존 핵심 컴포넌트들
        self.indicator_system = Enhanced19IndicatorSystem()
        self.time_series = TimeSeriesAnalyzer()
        self.claude_predictor = ClaudePricePredictor()
        self.prediction_tracker = PredictionTracker()
        self.explainer = BeginnerFriendlyExplainer()
        self.telegram = EnhancedTelegramNotifier()
        
        # 학습 시스템
        self.learning_engine = AdaptiveLearningEngine()
        self.hybrid_optimizer = HybridLearningOptimizer(
            claude_api_key=os.environ.get('CLAUDE_API_KEY')
        )
        self.accuracy_roadmap = AccuracyEnhancementRoadmap()
        
        # 맞춤형 알림 시스템
        self.custom_alerts = CustomAlertSystem()
        self.telegram_handler = TelegramCommandHandler(
            bot_token=os.environ.get('TELEGRAM_BOT_TOKEN', ''),
            chat_id=os.environ.get('TELEGRAM_CHAT_ID', '')
        )
        
        # 🎯 새로운 예측적 가격 알림 시스템
        self.price_predictor = PredictivePriceAlertSystem()
        self.backtest_validator = BacktestValidator()
        self.last_price_prediction = None
        self.prediction_history = []
        
        self.logger = logger
        
        # 시스템 상태
        self.monitoring_active = True
        self.last_alert_time = None
        self.alert_cooldown = 300  # 5분 쿨다운
        
        # 성능 통계 (예측 통계 추가)
        self.daily_stats = {
            "total_predictions": 0,
            "correct_predictions": 0,
            "alerts_sent": 0,
            "high_priority_alerts": 0,
            "price_predictions": 0,
            "prediction_accuracy": 0,
            "errors": 0
        }
    
    async def run_enhanced_analysis_cycle(self) -> Dict:
        """강화된 분석 사이클 - 예측적 가격 알림 포함"""
        try:
            self.logger.info("="*60)
            self.logger.info("🚀 강화된 분석 사이클 V2 시작")
            start_time = datetime.now()
            
            # 1단계: 19개 지표 수집
            self.logger.info("📊 1/8 - 19개 선행지표 수집 중...")
            indicators = await self._collect_all_indicators()
            
            if not indicators:
                raise Exception("지표 수집 실패")
            
            # 2단계: 시계열 분석
            self.logger.info("📈 2/8 - 시계열 패턴 분석 중...")
            time_series_result = await self._analyze_time_series(indicators)
            
            # 🎯 3단계: 예측적 가격 분석 (새로운 기능)
            self.logger.info("🔮 3/8 - 예측적 가격 변동 분석 중...")
            price_prediction = await self._predict_price_movement(indicators)
            
            # 4단계: AI 예측 (조건부)
            self.logger.info("🤖 4/8 - AI 예측 분석 중...")
            ai_prediction = await self._get_ai_prediction(
                indicators, 
                time_series_result,
                price_prediction
            )
            
            # 5단계: 예측 통합 및 검증
            self.logger.info("🔄 5/8 - 예측 통합 및 검증 중...")
            final_prediction = await self._integrate_predictions(
                indicators,
                time_series_result,
                ai_prediction,
                price_prediction
            )
            
            # 6단계: 리스크 평가 및 알림 결정
            self.logger.info("⚠️ 6/8 - 리스크 평가 중...")
            risk_assessment = await self._assess_risk_and_alert(
                final_prediction,
                price_prediction
            )
            
            # 7단계: 맞춤 알림 체크
            self.logger.info("🔔 7/8 - 맞춤 알림 확인 중...")
            custom_alerts_sent = await self._check_custom_alerts(indicators)
            
            # 8단계: 텔레그램 명령어 처리
            self.logger.info("📱 8/8 - 텔레그램 명령어 처리 중...")
            commands_processed = await self._process_telegram_commands()
            
            # 실행 시간 계산
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # 통계 업데이트
            self.daily_stats["total_predictions"] += 1
            if risk_assessment.get("alert_sent"):
                self.daily_stats["alerts_sent"] += 1
            if price_prediction and price_prediction.confidence > 70:
                self.daily_stats["price_predictions"] += 1
            
            # 결과 반환
            result = {
                "status": "success",
                "timestamp": datetime.now().isoformat(),
                "execution_time": execution_time,
                "indicators": indicators,
                "time_series": time_series_result,
                "price_prediction": self._serialize_price_prediction(price_prediction),
                "ai_prediction": ai_prediction,
                "final_prediction": final_prediction,
                "risk_assessment": risk_assessment,
                "custom_alerts_sent": custom_alerts_sent,
                "commands_processed": commands_processed,
                "daily_stats": self.daily_stats
            }
            
            self.logger.info(f"✅ 분석 사이클 완료 ({execution_time:.1f}초)")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ 분석 사이클 오류: {e}")
            self.daily_stats["errors"] += 1
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def _predict_price_movement(self, indicators: Dict) -> Optional[PricePrediction]:
        """예측적 가격 변동 분석"""
        try:
            # 예측을 위한 데이터 준비
            prediction_data = {
                'price': indicators.get('metadata', {}).get('current_price', 0),
                'funding_rate': self._extract_indicator_value(indicators, 'funding_rate'),
                'exchange_flows': self._extract_indicator_value(indicators, 'exchange_flows'),
                'whale_movements': self._extract_indicator_value(indicators, 'whale_activity'),
                'rsi': self._extract_indicator_value(indicators, 'rsi'),
                'volume_ratio': self._extract_indicator_value(indicators, 'volume_profile'),
                'fear_greed': self._extract_indicator_value(indicators, 'fear_greed'),
                'options_gamma': self._extract_indicator_value(indicators, 'options_put_call'),
                'price_range': self._calculate_price_range(indicators),
                'rsi_divergence': self._check_divergence(indicators)
            }
            
            # 가격 예측 생성
            prediction = self.price_predictor.predict_price_movement(prediction_data)
            
            # 이전 예측과 비교하여 업데이트 필요 여부 확인
            if self.last_price_prediction:
                if self.price_predictor.should_update_prediction(
                    self.last_price_prediction, 
                    prediction
                ):
                    # 예측 변경 알림
                    await self._send_prediction_update_alert(
                        self.last_price_prediction,
                        prediction
                    )
            
            # 예측 기록 저장
            self.last_price_prediction = prediction
            self.prediction_history.append({
                'timestamp': datetime.now(),
                'prediction': prediction
            })
            
            # 최근 100개만 유지
            if len(self.prediction_history) > 100:
                self.prediction_history = self.prediction_history[-100:]
            
            return prediction
            
        except Exception as e:
            self.logger.error(f"가격 예측 오류: {e}")
            return None
    
    def _extract_indicator_value(self, indicators: Dict, indicator_name: str) -> float:
        """지표값 추출 헬퍼"""
        try:
            # enhanced_19_system 구조에서 값 추출
            detailed = indicators.get('detailed_analysis', {})
            
            if indicator_name == 'funding_rate':
                return detailed.get('funding_rate', {}).get('current_value', 0)
            elif indicator_name == 'exchange_flows':
                flows = detailed.get('exchange_flows', {})
                inflows = flows.get('inflows', 0)
                outflows = flows.get('outflows', 0)
                avg = flows.get('average', 1)
                return (inflows - outflows) / avg if avg > 0 else 0
            elif indicator_name == 'whale_activity':
                return detailed.get('whale_activity', {}).get('current_value', 0)
            elif indicator_name == 'rsi':
                return detailed.get('price_momentum', {}).get('rsi_14', 50)
            elif indicator_name == 'volume_profile':
                return detailed.get('volume_profile', {}).get('ratio_to_average', 1.0)
            elif indicator_name == 'fear_greed':
                return detailed.get('fear_greed', {}).get('current_value', 50)
            elif indicator_name == 'options_put_call':
                return detailed.get('options_put_call', {}).get('ratio', 1.0)
            else:
                return 0
                
        except Exception:
            return 0
    
    def _calculate_price_range(self, indicators: Dict) -> float:
        """가격 범위 계산"""
        try:
            metadata = indicators.get('metadata', {})
            high = metadata.get('high_24h', 0)
            low = metadata.get('low_24h', 0)
            current = metadata.get('current_price', 1)
            
            if current > 0:
                return (high - low) / current
            return 0
        except:
            return 0
    
    def _check_divergence(self, indicators: Dict) -> float:
        """다이버전스 체크"""
        try:
            # RSI와 가격의 다이버전스 체크 (시뮬레이션)
            rsi = self._extract_indicator_value(indicators, 'rsi')
            
            if rsi > 70:
                return 1.0  # Bearish divergence
            elif rsi < 30:
                return -1.0  # Bullish divergence
            else:
                return 0
        except:
            return 0
    
    async def _send_prediction_update_alert(self, old_pred: PricePrediction, new_pred: PricePrediction):
        """예측 변경 알림"""
        try:
            # 방향 변경 체크
            direction_changed = (old_pred.predicted_change_percent > 0) != (new_pred.predicted_change_percent > 0)
            
            if direction_changed:
                emoji = "⚠️"
                update_type = "방향 전환"
            else:
                emoji = "📊"
                update_type = "예측 수정"
            
            message = f"""
{emoji} **가격 예측 {update_type}**

**이전 예측**: {abs(old_pred.predicted_change_percent)*100:.1f}% {"상승" if old_pred.predicted_change_percent > 0 else "하락"}
**새 예측**: {abs(new_pred.predicted_change_percent)*100:.1f}% {"상승" if new_pred.predicted_change_percent > 0 else "하락"}
**예상 시간**: {new_pred.timeframe_hours}시간 내
**신뢰도**: {new_pred.confidence:.0f}%

**변경 이유**:
"""
            # 상위 3개 트리거 표시
            for trigger in new_pred.trigger_indicators[:3]:
                message += f"• {trigger}\n"
            
            await self.telegram.send_message(message, priority="high")
            
        except Exception as e:
            self.logger.error(f"예측 업데이트 알림 오류: {e}")
    
    def _serialize_price_prediction(self, prediction: Optional[PricePrediction]) -> Optional[Dict]:
        """가격 예측 직렬화"""
        if not prediction:
            return None
            
        return {
            "current_price": prediction.current_price,
            "predicted_change_percent": prediction.predicted_change_percent,
            "predicted_price": prediction.predicted_price,
            "confidence": prediction.confidence,
            "timeframe_hours": prediction.timeframe_hours,
            "completion_time": prediction.completion_time.isoformat(),
            "trigger_indicators": prediction.trigger_indicators,
            "evidence": prediction.evidence
        }
    
    async def _integrate_predictions(self, indicators: Dict, time_series: Dict, 
                                   ai_prediction: Dict, price_prediction: Optional[PricePrediction]) -> Dict:
        """모든 예측 통합"""
        
        predictions = []
        weights = []
        
        # 1. 지표 기반 예측
        if indicators.get('composite_analysis'):
            predictions.append({
                'direction': indicators['composite_analysis'].get('overall_signal', 'NEUTRAL'),
                'confidence': indicators['composite_analysis'].get('confidence', 50)
            })
            weights.append(0.25)
        
        # 2. 시계열 예측
        if time_series:
            predictions.append({
                'direction': time_series.get('prediction', 'NEUTRAL'),
                'confidence': time_series.get('confidence', 50)
            })
            weights.append(0.20)
        
        # 3. AI 예측
        if ai_prediction and ai_prediction.get('prediction'):
            predictions.append({
                'direction': ai_prediction['prediction'].get('direction', 'NEUTRAL'),
                'confidence': ai_prediction['prediction'].get('probability', 50)
            })
            weights.append(0.25)
        
        # 4. 가격 예측 (새로운)
        if price_prediction and price_prediction.confidence > 60:
            direction = 'BULLISH' if price_prediction.predicted_change_percent > 0 else 'BEARISH'
            predictions.append({
                'direction': direction,
                'confidence': price_prediction.confidence,
                'magnitude': abs(price_prediction.predicted_change_percent),
                'timeframe': price_prediction.timeframe_hours
            })
            weights.append(0.30)
        
        # 앙상블 계산
        if not predictions:
            return {
                'final_direction': 'NEUTRAL',
                'confidence': 0,
                'magnitude': 0,
                'timeframe': 0
            }
        
        # 정규화
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w/total_weight for w in weights]
        
        # 가중 평균
        weighted_confidence = sum(p['confidence'] * w for p, w in zip(predictions, weights))
        
        # 방향 결정 (가중 투표)
        direction_scores = {'BULLISH': 0, 'BEARISH': 0, 'NEUTRAL': 0}
        for pred, weight in zip(predictions, weights):
            direction_scores[pred['direction']] += weight * pred['confidence']
        
        final_direction = max(direction_scores, key=direction_scores.get)
        
        # 크기와 시간대 (가격 예측에서 가져옴)
        magnitude = 0
        timeframe = 6  # 기본값
        
        if price_prediction and price_prediction.confidence > 60:
            magnitude = abs(price_prediction.predicted_change_percent)
            timeframe = price_prediction.timeframe_hours
        
        return {
            'final_direction': final_direction,
            'confidence': weighted_confidence,
            'magnitude': magnitude,
            'timeframe': timeframe,
            'components': predictions
        }
    
    async def _assess_risk_and_alert(self, final_prediction: Dict, price_prediction: Optional[PricePrediction]) -> Dict:
        """리스크 평가 및 알림 결정"""
        
        alert_sent = False
        priority = "low"
        message = ""
        
        # 가격 예측 기반 알림 조건
        if price_prediction and price_prediction.confidence > 70:
            magnitude = abs(price_prediction.predicted_change_percent)
            
            # 알림 우선순위 결정
            if magnitude > 0.05 and price_prediction.confidence > 80:
                priority = "critical"
            elif magnitude > 0.03 and price_prediction.confidence > 75:
                priority = "high"
            elif magnitude > 0.02 and price_prediction.confidence > 70:
                priority = "medium"
            else:
                priority = "low"
            
            # 쿨다운 체크
            if self._should_send_alert(priority):
                # 알림 메시지 생성
                direction = "📈 상승" if price_prediction.predicted_change_percent > 0 else "📉 하락"
                
                message = f"""
🎯 **가격 변동 예측 알림**

**예측**: {magnitude*100:.1f}% {direction}
**시간대**: {price_prediction.timeframe_hours}시간 내
**신뢰도**: {price_prediction.confidence:.0f}%
**우선순위**: {priority.upper()}

**현재 가격**: ${price_prediction.current_price:,.0f}
**예상 가격**: ${price_prediction.predicted_price:,.0f}

**주요 신호**:
"""
                # 트리거 지표 추가
                for trigger in price_prediction.trigger_indicators[:5]:
                    message += f"• {trigger}\n"
                
                # 추가 분석 정보
                if final_prediction['confidence'] > 60:
                    message += f"\n**종합 분석**: {final_prediction['final_direction']} (신뢰도 {final_prediction['confidence']:.0f}%)"
                
                # 알림 발송
                success = await self.telegram.send_message(message, priority=priority)
                
                if success:
                    alert_sent = True
                    self.last_alert_time = datetime.now()
                    
                    if priority in ["critical", "high"]:
                        self.daily_stats["high_priority_alerts"] += 1
        
        return {
            "alert_sent": alert_sent,
            "priority": priority,
            "message": message if alert_sent else None,
            "price_prediction": self._serialize_price_prediction(price_prediction),
            "cooldown_active": not self._should_send_alert("low")
        }
    
    def _should_send_alert(self, priority: str) -> bool:
        """알림 발송 여부 결정"""
        if not self.last_alert_time:
            return True
        
        # 우선순위별 쿨다운
        cooldowns = {
            "critical": 60,   # 1분
            "high": 180,      # 3분
            "medium": 300,    # 5분
            "low": 600        # 10분
        }
        
        cooldown = cooldowns.get(priority, 300)
        elapsed = (datetime.now() - self.last_alert_time).total_seconds()
        
        return elapsed > cooldown
    
    # 기존 메서드들 (수정 없이 유지)
    async def _collect_all_indicators(self) -> Dict:
        """19개 지표 수집"""
        try:
            return await self.indicator_system.collect_enhanced_19_indicators()
        except Exception as e:
            self.logger.error(f"지표 수집 오류: {e}")
            return {}
    
    async def _analyze_time_series(self, indicators: Dict) -> Dict:
        """시계열 분석"""
        try:
            # 시계열 데이터 저장
            await self.time_series.store_realtime_data(
                indicators.get('metadata', {}),
                indicators.get('indicators', {})
            )
            # 패턴 분석
            return await self.time_series.analyze_time_series_patterns()
        except Exception as e:
            self.logger.error(f"시계열 분석 오류: {e}")
            return {}
    
    async def _get_ai_prediction(self, indicators: Dict, time_series: Dict, 
                                price_prediction: Optional[PricePrediction]) -> Dict:
        """AI 예측 (Claude)"""
        try:
            # 비용 절감을 위해 조건부 실행
            composite = indicators.get('composite_analysis', {})
            confidence = composite.get('confidence', 0)
            
            # 가격 예측이 강한 신호일 때만 AI 호출
            should_call_ai = (
                confidence > 75 or 
                (price_prediction and price_prediction.confidence > 80 and 
                 abs(price_prediction.predicted_change_percent) > 0.04)
            )
            
            if should_call_ai:
                # 가격 예측 정보 추가
                enhanced_context = []
                if price_prediction:
                    enhanced_context.append(
                        f"가격 예측: {abs(price_prediction.predicted_change_percent)*100:.1f}% "
                        f"{'상승' if price_prediction.predicted_change_percent > 0 else '하락'} "
                        f"({price_prediction.timeframe_hours}시간 내, 신뢰도 {price_prediction.confidence:.0f}%)"
                    )
                
                return await self.claude_predictor.analyze_market_signals(
                    indicators.get('metadata', {}),
                    enhanced_context
                )
            
            return {"prediction": None, "skipped": True, "reason": "신뢰도 부족"}
            
        except Exception as e:
            self.logger.error(f"AI 예측 오류: {e}")
            return {"prediction": None, "error": str(e)}
    
    async def _check_custom_alerts(self, indicators: Dict) -> int:
        """맞춤 알림 체크"""
        try:
            # 모든 사용자의 맞춤 알림 체크
            triggered_count = 0
            
            # 실제로는 DB에서 모든 사용자 조회
            # 여기서는 현재 사용자만
            user_id = os.environ.get('TELEGRAM_CHAT_ID', '')
            
            if user_id:
                triggered = await self.custom_alerts.check_custom_alerts(
                    indicators, 
                    user_id
                )
                
                for alert in triggered:
                    message = f"""
🔔 **맞춤 알림 발생**

📊 **지표**: {alert['indicator_kr']}
🎯 **조건**: {alert['operator_kr']} {alert['threshold']}
📈 **현재값**: {alert.get('current_value', 'N/A')}
💬 **메시지**: {alert['message']}

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
                    success = await self.telegram.send_message(message, priority="medium")
                    if success:
                        triggered_count += 1
            
            return triggered_count
            
        except Exception as e:
            self.logger.error(f"맞춤 알림 체크 오류: {e}")
            return 0
    
    async def _process_telegram_commands(self) -> int:
        """텔레그램 명령어 처리"""
        try:
            return await self.telegram_handler.process_and_respond()
        except Exception as e:
            self.logger.error(f"텔레그램 명령어 처리 오류: {e}")
            return 0
    
    async def run_24h_monitoring(self):
        """24시간 모니터링 루프"""
        self.logger.info("🚀 24시간 모니터링 시작")
        
        while self.monitoring_active:
            try:
                # 현재 시간 체크
                current_hour = datetime.now().hour
                
                # 시간대별 실행 간격
                if current_hour in [9, 10, 15, 16, 21, 22]:  # 중요 시간
                    interval = 300  # 5분
                elif current_hour in range(7, 23):  # 일반 시간
                    interval = 1800  # 30분
                else:  # 한가한 시간
                    interval = 3600  # 1시간
                
                # 분석 실행
                result = await self.run_enhanced_analysis_cycle()
                
                # 일일 통계 출력 (자정)
                if current_hour == 0 and datetime.now().minute < 5:
                    await self._send_daily_report()
                
                # 대기
                await asyncio.sleep(interval)
                
            except Exception as e:
                self.logger.error(f"모니터링 루프 오류: {e}")
                await asyncio.sleep(60)  # 오류 시 1분 대기
    
    async def _send_daily_report(self):
        """일일 리포트 발송"""
        try:
            # 예측 정확도 계산
            if self.daily_stats["price_predictions"] > 0:
                accuracy = (self.daily_stats["correct_predictions"] / 
                          self.daily_stats["total_predictions"] * 100)
            else:
                accuracy = 0
            
            message = f"""
📊 **일일 시스템 리포트**

**예측 통계**:
• 총 예측: {self.daily_stats['total_predictions']}회
• 가격 예측: {self.daily_stats['price_predictions']}회
• 정확도: {accuracy:.1f}%

**알림 통계**:
• 총 알림: {self.daily_stats['alerts_sent']}회
• 중요 알림: {self.daily_stats['high_priority_alerts']}회

**시스템 상태**:
• 오류: {self.daily_stats['errors']}회
• 가동률: {(1 - self.daily_stats['errors']/max(self.daily_stats['total_predictions'], 1))*100:.1f}%

🎯 예측적 가격 알림 시스템 정상 작동 중
"""
            
            await self.telegram.send_message(message, priority="low")
            
            # 통계 리셋
            self.daily_stats = {
                "total_predictions": 0,
                "correct_predictions": 0,
                "alerts_sent": 0,
                "high_priority_alerts": 0,
                "price_predictions": 0,
                "prediction_accuracy": 0,
                "errors": 0
            }
            
        except Exception as e:
            self.logger.error(f"일일 리포트 발송 오류: {e}")

# Azure Functions 통합을 위한 메인 함수
async def main():
    """메인 실행 함수"""
    system = EnhancedFinalSystemV2()
    
    # 단일 실행 모드
    result = await system.run_enhanced_analysis_cycle()
    print(json.dumps(result, indent=2, default=str))
    
    # 또는 24시간 모니터링 모드
    # await system.run_24h_monitoring()

if __name__ == "__main__":
    asyncio.run(main())