#!/usr/bin/env python3
"""
메인 모니터링 시스템 - Azure Function 진입점
모든 컴포넌트를 통합하여 24시간 위험 감지 서비스 제공
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import traceback

# 로컬 모듈들
from data_collector import FreeDataCollector  
from risk_analyzer import TimeSeriesRiskAnalyzer
from telegram_notifier import TelegramNotifier
from claude_predictor import ClaudePricePredictor
from prediction_tracker import PredictionTracker
from advanced_data_sources import AdvancedDataCollector
from config import AZURE_CONFIG, RISK_THRESHOLDS, LOGGING_CONFIG

class BRCRiskMonitor:
    def __init__(self):
        self.logger = self.setup_logging()
        self.data_collector = None
        self.risk_analyzer = TimeSeriesRiskAnalyzer()
        self.claude_predictor = ClaudePricePredictor()
        self.notifier = TelegramNotifier()
        
        # 새로운 고급 예측 시스템
        self.prediction_tracker = PredictionTracker()
        self.advanced_data_collector = AdvancedDataCollector()
        
        # 상태 추적
        self.historical_data = []
        self.last_analysis = None
        self.last_prediction = None
        self.system_start_time = datetime.utcnow()
        
        self.logger.info("🚀 BTC 고급 예측 위험 모니터 초기화 완료")

    def setup_logging(self) -> logging.Logger:
        """로깅 설정"""
        logging.basicConfig(
            level=getattr(logging, LOGGING_CONFIG["level"]),
            format=LOGGING_CONFIG["format"]
        )
        return logging.getLogger(__name__)

    async def run_monitoring_cycle(self) -> Dict:
        """한 번의 모니터링 사이클 실행"""
        try:
            cycle_start = datetime.utcnow()
            self.logger.info(f"📊 모니터링 사이클 시작: {cycle_start.strftime('%H:%M:%S')}")
            
            # 1. 데이터 수집
            current_data = await self.collect_current_data()
            if "error" in current_data:
                raise Exception(f"데이터 수집 실패: {current_data['error']}")
                
            # 2. 히스토리컬 데이터에 추가
            self.historical_data.append(current_data)
            self.maintain_historical_data()  # 메모리 관리
            
            # 3. 기존 위험 분석  
            risk_analysis = self.analyze_current_risk(current_data)
            
            # 4. 이전 예측 평가 (학습 시스템)
            evaluation_result = self.prediction_tracker.evaluate_predictions(current_data)
            
            # 5. 정확도 메트릭스 가져오기 (성과 기반 필터링용)
            accuracy_metrics = self.prediction_tracker.get_accuracy_metrics(days=7)
            
            # 6. 고급 선행지표 수집
            leading_indicators = await self.advanced_data_collector.get_real_leading_indicators()
            
            # 7. 향상된 Claude 예측 분석 (핵심!)
            price_prediction = await self.claude_predictor.request_enhanced_claude_prediction(
                leading_indicators, current_data, accuracy_metrics
            )
            
            # 8. 새로운 예측 기록 (학습 데이터)
            if price_prediction.get("prediction", {}).get("direction", "NEUTRAL") != "NEUTRAL":
                prediction_id = self.prediction_tracker.record_prediction(
                    price_prediction, current_data, leading_indicators
                )
                self.logger.info(f"📝 예측 기록됨: ID {prediction_id}")
            
            # 9. 텔레그램 메시지 처리 (사용자 요청)
            await self.process_incoming_messages()
            
            # 10. 성과 기반 예측 알림 필터링 및 발송 (최우선!)
            prediction_alert_sent = await self.process_enhanced_prediction_alert(
                price_prediction, current_data, accuracy_metrics
            )
            
            # 11. 기존 위험 분석 알림 (보조)
            basic_alert_sent = await self.process_risk_alert(risk_analysis, current_data)
            
            # 12. 개인 요청 조건 체크 및 알림
            custom_alerts_sent = await self.process_custom_alerts(current_data)
            
            # 7. 결과 정리
            cycle_end = datetime.utcnow()
            cycle_duration = (cycle_end - cycle_start).total_seconds()
            
            result = {
                "success": True,
                "timestamp": cycle_end.isoformat(),
                "cycle_duration": cycle_duration,
                "data_collected": len(current_data) > 2,  # 최소한의 데이터 확인
                "risk_analysis": {
                    "risk_score": risk_analysis.get("composite_risk_score", 0),
                    "risk_level": risk_analysis.get("risk_level", "UNKNOWN"),
                    "confidence": risk_analysis.get("confidence", 0)
                },
                "prediction_alert_sent": prediction_alert_sent,
                "basic_alert_sent": basic_alert_sent,
                "custom_alerts_sent": custom_alerts_sent,
                "price_prediction": {
                    "direction": price_prediction.get("prediction", {}).get("direction", "NEUTRAL"),
                    "probability": price_prediction.get("prediction", {}).get("probability", 50),
                    "target_price": price_prediction.get("prediction", {}).get("target_price", 0)
                },
                "prediction_tracking": {
                    "evaluated_count": evaluation_result.get("evaluated_count", 0),
                    "accuracy_7d": accuracy_metrics.get("direction_accuracy", 0),
                    "quality_score_7d": accuracy_metrics.get("quality_score", 0)
                },
                "leading_indicators": {
                    "categories_collected": len(leading_indicators) - 1 if "timestamp" in leading_indicators else len(leading_indicators)
                },
                "historical_data_points": len(self.historical_data),
                "next_cycle": "1 minute",
                "system_version": "BRC v2.0 - Enhanced Prediction System"
            }
            
            self.last_analysis = risk_analysis
            self.last_prediction = price_prediction
            self.logger.info(f"✅ 모니터링 사이클 완료 ({cycle_duration:.2f}초)")
            
            return result
            
        except Exception as e:
            error_msg = f"모니터링 사이클 오류: {str(e)}"
            self.logger.error(error_msg)
            self.logger.error(traceback.format_exc())
            
            # 오류 알림 발송
            try:
                await self.notifier.send_error_notification(str(e))
            except:
                pass  # 알림 발송 실패는 무시
                
            return {
                "success": False,
                "error": error_msg,
                "timestamp": datetime.utcnow().isoformat()
            }

    async def collect_current_data(self) -> Dict:
        """현재 데이터 수집"""
        try:
            async with FreeDataCollector() as collector:
                data = await collector.collect_all_data()
                
                # 즉시 계산 가능한 위험 지표도 추가
                immediate_risk = collector.calculate_immediate_risk_indicators(data)
                data["immediate_risk"] = immediate_risk
                
                self.logger.info("✅ 데이터 수집 완료")
                return data
                
        except Exception as e:
            self.logger.error(f"데이터 수집 실패: {e}")
            return {"error": str(e), "timestamp": datetime.utcnow().isoformat()}

    def analyze_current_risk(self, current_data: Dict) -> Dict:
        """현재 위험도 분석"""
        try:
            # 히스토리컬 데이터가 충분하지 않으면 간단한 분석만
            if len(self.historical_data) < 10:
                return self.simple_risk_analysis(current_data)
                
            # 전체 시계열 분석
            risk_analysis = self.risk_analyzer.analyze_timeseries_risk(
                current_data, 
                self.historical_data[:-1]  # 현재 데이터 제외
            )
            
            self.logger.info(f"🧠 위험 분석 완료 - 점수: {risk_analysis.get('composite_risk_score', 0):.3f}")
            return risk_analysis
            
        except Exception as e:
            self.logger.error(f"위험 분석 실패: {e}")
            return self.fallback_risk_analysis(current_data)

    def simple_risk_analysis(self, current_data: Dict) -> Dict:
        """히스토리컬 데이터 부족 시 간단한 분석"""
        try:
            risk_score = 0
            risk_factors = []
            
            # 가격 변동성 체크
            if "price_data" in current_data and "change_24h" in current_data["price_data"]:
                change_24h = abs(current_data["price_data"]["change_24h"])
                if change_24h > 10:
                    risk_score += 0.5
                    risk_factors.append("높은 가격 변동성")
                elif change_24h > 5:
                    risk_score += 0.3
                    
            # VIX 레벨 체크
            if "macro_data" in current_data and "vix" in current_data["macro_data"]:
                vix_level = current_data["macro_data"]["vix"]["current"]
                if vix_level > 30:
                    risk_score += 0.4
                    risk_factors.append("높은 VIX 수준")
                elif vix_level > 25:
                    risk_score += 0.2
                    
            # 공포탐욕지수 체크
            if "sentiment_data" in current_data and "fear_greed" in current_data["sentiment_data"]:
                fg_index = current_data["sentiment_data"]["fear_greed"]["current_index"]
                if fg_index < 20 or fg_index > 80:
                    risk_score += 0.3
                    risk_factors.append("극한 시장 심리")
                    
            # 위험 레벨 결정
            if risk_score >= 0.7:
                risk_level = "WARNING"
            elif risk_score >= 0.4:
                risk_level = "INFO"  
            else:
                risk_level = "LOW"
                
            return {
                "composite_risk_score": min(risk_score, 1.0),
                "risk_level": risk_level,
                "confidence": 0.6,  # 간단한 분석이므로 낮은 신뢰도
                "risk_factors": risk_factors,
                "analysis_type": "simple",
                "timestamp": datetime.utcnow().isoformat(),
                "recommendations": ["히스토리컬 데이터 부족으로 간단 분석", "10회 이상 실행 후 정밀 분석 가능"]
            }
            
        except Exception as e:
            self.logger.error(f"간단 분석 실패: {e}")
            return self.fallback_risk_analysis(current_data)

    def fallback_risk_analysis(self, current_data: Dict) -> Dict:
        """최후 수단 분석 (모든 분석 실패 시)"""
        return {
            "composite_risk_score": 0.5,
            "risk_level": "WARNING",
            "confidence": 0.3,
            "analysis_type": "fallback",
            "timestamp": datetime.utcnow().isoformat(),
            "error": "분석 엔진 오류로 기본값 사용",
            "recommendations": ["시스템 점검 필요"]
        }

    async def process_risk_alert(self, risk_analysis: Dict, current_data: Dict) -> bool:
        """위험 분석 결과에 따른 알림 처리"""
        try:
            risk_level = risk_analysis.get("risk_level", "LOW")
            risk_score = risk_analysis.get("composite_risk_score", 0)
            
            # 알림 필요성 판단
            should_alert = self.should_send_alert(risk_level, risk_score)
            
            if should_alert:
                success = await self.notifier.send_risk_alert(risk_analysis, current_data)
                if success:
                    self.logger.info(f"📱 {risk_level} 알림 발송 완료")
                else:
                    self.logger.error(f"📱 {risk_level} 알림 발송 실패")
                return success
            else:
                self.logger.debug(f"알림 조건 미충족 - 레벨: {risk_level}, 점수: {risk_score:.3f}")
                return False
                
        except Exception as e:
            self.logger.error(f"알림 처리 오류: {e}")
            return False

    def should_send_alert(self, risk_level: str, risk_score: float) -> bool:
        """알림 발송 필요성 판단"""
        # 위험 레벨별 임계값
        thresholds = {
            "CRITICAL": 0.8,
            "WARNING": 0.6,
            "INFO": 0.4,
            "LOW": 1.1  # LOW는 기본적으로 알림 안 함
        }
        
        threshold = thresholds.get(risk_level, 0.5)
        return risk_score >= threshold

    def maintain_historical_data(self):
        """히스토리컬 데이터 메모리 관리"""
        max_history_points = 1440  # 24시간 (1분마다 실행 가정)
        
        if len(self.historical_data) > max_history_points:
            # 오래된 데이터 제거
            self.historical_data = self.historical_data[-max_history_points:]
            self.logger.debug(f"히스토리컬 데이터 정리: {len(self.historical_data)}개 유지")

    async def run_startup_sequence(self):
        """시스템 시작 시 초기화 작업"""
        try:
            self.logger.info("🚀 시스템 시작 시퀀스 실행")
            
            # 텔레그램 연결 테스트
            telegram_ok = await self.notifier.send_system_start_notification()
            
            # 첫 데이터 수집 테스트
            test_data = await self.collect_current_data()
            data_ok = "error" not in test_data
            
            # 분석 엔진 테스트
            if data_ok:
                test_analysis = self.simple_risk_analysis(test_data)
                analysis_ok = "error" not in test_analysis
            else:
                analysis_ok = False
                
            startup_status = {
                "telegram": "✅" if telegram_ok else "❌",
                "data_collection": "✅" if data_ok else "❌", 
                "risk_analysis": "✅" if analysis_ok else "❌"
            }
            
            self.logger.info(f"시작 상태: {startup_status}")
            
            return all([telegram_ok, data_ok, analysis_ok])
            
        except Exception as e:
            self.logger.error(f"시작 시퀀스 오류: {e}")
            return False

    def get_system_status(self) -> Dict:
        """시스템 상태 정보 반환"""
        uptime = datetime.utcnow() - self.system_start_time
        
        return {
            "status": "running",
            "uptime_seconds": uptime.total_seconds(),
            "uptime_formatted": str(uptime).split('.')[0],  # 소수점 제거
            "historical_data_points": len(self.historical_data),
            "last_analysis_time": self.last_analysis.get("timestamp") if self.last_analysis else None,
            "last_risk_score": self.last_analysis.get("composite_risk_score") if self.last_analysis else None,
            "last_risk_level": self.last_analysis.get("risk_level") if self.last_analysis else None,
            "system_start_time": self.system_start_time.isoformat()
        }

    async def process_incoming_messages(self):
        """텔레그램에서 들어온 사용자 메시지 처리"""
        try:
            # 새 메시지 확인
            messages = await self.notifier.check_incoming_messages()
            
            # 각 메시지 처리
            for message in messages:
                self.logger.info(f"사용자 메시지 수신: {message}")
                
                # 명령 처리
                response = await self.notifier.process_user_command(message)
                
                # 응답 발송
                if response:
                    await self.notifier.send_message(response)
                    self.logger.info(f"사용자 명령 응답 발송 완료")
                    
        except Exception as e:
            self.logger.error(f"메시지 처리 오류: {e}")

    async def process_custom_alerts(self, current_data: Dict) -> int:
        """개인 요청 조건들 체크 및 알림 발송"""
        try:
            # 개인 요청 조건들 체크
            alert_messages = await self.notifier.check_custom_alerts(current_data)
            
            # 알림 발송
            alerts_sent = 0
            for message in alert_messages:
                success = await self.notifier.send_message(message)
                if success:
                    alerts_sent += 1
                    
            if alerts_sent > 0:
                self.logger.info(f"개인 요청 알림 {alerts_sent}개 발송 완료")
                
            return alerts_sent
            
        except Exception as e:
            self.logger.error(f"개인 요청 알림 처리 오류: {e}")
            return 0

    async def process_prediction_alert(self, price_prediction: Dict, current_data: Dict) -> bool:
        """Claude 예측 기반 사전 경고 알림 처리"""
        try:
            prediction = price_prediction.get("prediction", {})
            direction = prediction.get("direction", "NEUTRAL")
            probability = prediction.get("probability", 0)
            confidence = prediction.get("confidence", "LOW")
            
            # 사전 경고 알림 조건
            should_alert = self.should_send_prediction_alert(direction, probability, confidence)
            
            if should_alert:
                # 예측 기반 경고 메시지 생성
                alert_message = self.generate_prediction_alert_message(price_prediction, current_data)
                
                # 알림 발송
                success = await self.notifier.send_message(alert_message)
                
                if success:
                    self.logger.info(f"🔮 예측 경고 알림 발송 완료: {direction} ({probability}%)")
                else:
                    self.logger.error(f"🔮 예측 경고 알림 발송 실패")
                    
                return success
            else:
                self.logger.debug(f"예측 알림 조건 미충족: {direction} {probability}% {confidence}")
                return False
                
        except Exception as e:
            self.logger.error(f"예측 알림 처리 오류: {e}")
            return False

    def should_send_prediction_alert(self, direction: str, probability: float, confidence: str) -> bool:
        """예측 기반 알림 발송 필요성 판단"""
        # 중성 예측은 알림하지 않음
        if direction == "NEUTRAL":
            return False
            
        # 고신뢰도 + 고확률 예측만 알림
        if confidence == "HIGH" and probability >= 70:
            return True
        elif confidence == "MEDIUM" and probability >= 80:
            return True
        elif confidence == "LOW":
            return False  # 저신뢰도는 알림 안 함
            
        return False

    def generate_prediction_alert_message(self, price_prediction: Dict, current_data: Dict) -> str:
        """예측 기반 경고 메시지 생성"""
        try:
            prediction = price_prediction.get("prediction", {})
            analysis = price_prediction.get("analysis", {})
            
            direction = prediction.get("direction", "NEUTRAL")
            probability = prediction.get("probability", 0)
            timeframe = prediction.get("timeframe", "6-12시간")
            target_price = prediction.get("target_price", 0)
            confidence = prediction.get("confidence", "LOW")
            
            current_price = current_data.get("price_data", {}).get("current_price", 0)
            
            # 방향별 이모지
            direction_emoji = {
                "BULLISH": "📈",
                "BEARISH": "📉", 
                "NEUTRAL": "➡️"
            }
            
            # 메시지 헤더
            header = f"{direction_emoji.get(direction, '🔮')} **가격 변동 예측 경고**"
            
            message = f"{header}\n\n"
            message += f"💰 현재가: ${current_price:,.0f}\n"
            message += f"🎯 예측 방향: **{direction}**\n"
            message += f"📊 확률: **{probability}%**\n"
            message += f"⏰ 예상 시간: {timeframe}\n"
            message += f"💵 목표가: ${target_price:,.0f}\n"
            message += f"🔍 신뢰도: {confidence}\n\n"
            
            # 주요 원인
            catalysts = analysis.get("catalysts", [])
            if catalysts:
                message += f"🔑 **주요 원인**:\n"
                for i, catalyst in enumerate(catalysts[:3], 1):
                    message += f"  {i}. {catalyst}\n"
                message += "\n"
            
            # 위험 요소
            risks = analysis.get("risks", [])
            if risks:
                message += f"⚠️ **위험 요소**:\n"
                for i, risk in enumerate(risks[:2], 1):
                    message += f"  {i}. {risk}\n"
                message += "\n"
            
            # 권장 조치
            recommended_action = price_prediction.get("recommended_action", "")
            if recommended_action:
                message += f"💡 **권장 조치**:\n{recommended_action}\n\n"
            
            # 분석 근거
            reasoning = analysis.get("reasoning", "")
            if reasoning:
                message += f"🧠 **분석 근거**:\n{reasoning}\n\n"
            
            # 주의사항
            message += f"⚠️ **주의**: 이는 예측이며 투자 조언이 아닙니다.\n"
            message += f"📅 {datetime.utcnow().strftime('%H:%M:%S')} UTC | Claude AI 분석"
            
            return message
            
        except Exception as e:
            self.logger.error(f"예측 메시지 생성 실패: {e}")
            return f"🔮 가격 변동 예측 알림\n\n❌ 메시지 생성 오류: {str(e)}\n📅 {datetime.utcnow().strftime('%H:%M:%S')}"
    
    async def process_enhanced_prediction_alert(self, price_prediction: Dict, current_data: Dict, accuracy_metrics: Dict) -> bool:
        """성과 기반 향상된 예측 알림 처리"""
        try:
            prediction = price_prediction.get("prediction", {})
            direction = prediction.get("direction", "NEUTRAL")
            probability = prediction.get("probability", 0)
            confidence = prediction.get("confidence", "LOW")
            
            # 성과 기반 알림 조건 검사
            should_alert = self.prediction_tracker.should_send_alert(price_prediction, accuracy_metrics)
            
            if should_alert:
                # 향상된 예측 기반 경고 메시지 생성
                alert_message = self.generate_enhanced_prediction_alert_message(
                    price_prediction, current_data, accuracy_metrics
                )
                
                # 알림 발송
                success = await self.notifier.send_message(alert_message)
                
                if success:
                    self.logger.info(f"🔮 성과기반 예측 알림 발송: {direction} {probability}% (전체정확도: {accuracy_metrics.get('direction_accuracy', 0):.1%})")
                else:
                    self.logger.error(f"🔮 예측 알림 발송 실패")
                    
                return success
            else:
                self.logger.debug(f"예측 알림 성과기반 필터링: {direction} {probability}% {confidence}")
                return False
                
        except Exception as e:
            self.logger.error(f"향상된 예측 알림 처리 오류: {e}")
            return False
    
    def generate_enhanced_prediction_alert_message(self, price_prediction: Dict, current_data: Dict, accuracy_metrics: Dict) -> str:
        """성과 기반 향상된 예측 경고 메시지 생성"""
        try:
            prediction = price_prediction.get("prediction", {})
            analysis = price_prediction.get("analysis", {})
            
            direction = prediction.get("direction", "NEUTRAL")
            probability = prediction.get("probability", 0)
            timeframe = prediction.get("timeframe", "6-12시간")
            target_price = prediction.get("target_price", 0)
            confidence = prediction.get("confidence", "LOW")
            
            current_price = current_data.get("price_data", {}).get("current_price", 0)
            
            # 성과 메트릭스
            overall_accuracy = accuracy_metrics.get("direction_accuracy", 0)
            confidence_breakdown = accuracy_metrics.get("confidence_breakdown", {})
            direction_breakdown = accuracy_metrics.get("direction_breakdown", {})
            
            # 방향별 이모지
            direction_emoji = {
                "BULLISH": "📈",
                "BEARISH": "📉", 
                "NEUTRAL": "➡️"
            }
            
            # 메시지 헤더 (성과 지표 포함)
            header = f"{direction_emoji.get(direction, '🔮')} **AI 예측 경고** (정확도: {overall_accuracy:.1%})"
            
            message = f"{header}\n\n"
            message += f"💰 현재가: **${current_price:,.0f}**\n"
            message += f"🎯 예측: **{direction}** ({probability}%)\n"
            message += f"⏰ 예상시간: {timeframe}\n"
            message += f"💵 목표가: ${target_price:,.0f}\n"
            message += f"🔍 신뢰도: {confidence}\n\n"
            
            # AI 시스템 성과 요약
            message += f"🤖 **AI 성과 요약**:\n"
            message += f"• 전체 정확도: {overall_accuracy:.1%}\n"
            
            # 신뢰도별 성과
            conf_accuracy = confidence_breakdown.get(confidence, {}).get("accuracy", 0)
            if conf_accuracy > 0:
                message += f"• {confidence} 신뢰도 정확도: {conf_accuracy:.1%}\n"
            
            # 방향별 성과
            dir_accuracy = direction_breakdown.get(direction, {}).get("accuracy", 0)
            if dir_accuracy > 0:
                message += f"• {direction} 예측 정확도: {dir_accuracy:.1%}\n"
            
            message += "\n"
            
            # 주요 선행지표 (새로운 기능)
            leading_indicators = analysis.get("leading_indicators_detected", [])
            if leading_indicators:
                message += f"🕰️ **감지된 선행지표**:\n"
                for indicator in leading_indicators[:3]:
                    message += f"• {indicator}\n"
                message += "\n"
            
            # 주요 원인
            catalysts = analysis.get("catalysts", [])
            if catalysts:
                message += f"🔑 **주요 원인**:\n"
                for i, catalyst in enumerate(catalysts[:2], 1):
                    message += f"{i}. {catalyst}\n"
                message += "\n"
            
            # 위험 요소
            risks = analysis.get("risks", [])
            if risks:
                message += f"⚠️ **위험 요소**:\n"
                for risk in risks[:2]:
                    message += f"• {risk}\n"
                message += "\n"
            
            # 권장 조치
            recommended_action = price_prediction.get("recommended_action", "")
            if recommended_action:
                message += f"💡 **권장사항**: {recommended_action}\n\n"
            
            # 주의사항 및 디스클레이머
            message += f"⚠️ **주의**: AI 예측이며 투자조언이 아님\n"
            message += f"🕰️ {datetime.utcnow().strftime('%H:%M:%S')} UTC | Claude AI v4.0"
            
            return message
            
        except Exception as e:
            self.logger.error(f"향상된 메시지 생성 실패: {e}")
            return f"🔮 AI 예측 알림\n\n❌ 데이터 처리 오류\n📅 {datetime.utcnow().strftime('%H:%M:%S')}"

# Azure Functions 진입점
async def main(req=None) -> Dict:
    """Azure Functions 메인 진입점"""
    try:
        # 환경 변수에서 설정 읽기 (Azure에서 실행 시)
        if os.environ.get("AZURE_FUNCTIONS_ENVIRONMENT"):
            # Azure 환경에서 실행 중
            pass
            
        # 모니터 인스턴스 생성 및 실행
        monitor = BRCRiskMonitor()
        
        # 첫 실행인 경우 시작 시퀀스 실행
        if not hasattr(main, '_initialized'):
            await monitor.run_startup_sequence()
            main._initialized = True
            
        # 모니터링 사이클 실행
        result = await monitor.run_monitoring_cycle()
        
        return result
        
    except Exception as e:
        error_result = {
            "success": False,
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # 로깅
        logger = logging.getLogger(__name__)
        logger.error(f"메인 함수 오류: {e}")
        logger.error(traceback.format_exc())
        
        return error_result

# 로컬 테스트용 함수
async def run_local_test():
    """로컬 환경에서 테스트 실행"""
    print("🧪 로컬 테스트 시작...")
    
    monitor = BRCRiskMonitor()
    
    # 시작 시퀀스
    print("1. 시작 시퀀스 실행...")
    startup_ok = await monitor.run_startup_sequence()
    print(f"   시작 시퀀스: {'✅ 성공' if startup_ok else '❌ 실패'}")
    
    if not startup_ok:
        print("❌ 시작 시퀀스 실패로 테스트 중단")
        return
        
    # 몇 차례 모니터링 사이클 실행
    for i in range(3):
        print(f"\n{i+1}. 모니터링 사이클 {i+1}/3 실행...")
        result = await monitor.run_monitoring_cycle()
        
        if result["success"]:
            print(f"   ✅ 성공 - 위험도: {result['risk_analysis']['risk_score']:.3f}")
            print(f"   레벨: {result['risk_analysis']['risk_level']}")
            print(f"   알림 발송: {'예' if result['alert_sent'] else '아니오'}")
        else:
            print(f"   ❌ 실패: {result.get('error', '알 수 없는 오류')}")
            
        # 다음 사이클을 위한 대기 (테스트에서는 10초)
        if i < 2:
            print("   10초 대기 중...")
            await asyncio.sleep(10)
            
    # 시스템 상태 출력
    print(f"\n📊 최종 시스템 상태:")
    status = monitor.get_system_status()
    for key, value in status.items():
        print(f"   {key}: {value}")
        
    print("\n✅ 로컬 테스트 완료!")

if __name__ == "__main__":
    # 직접 실행 시 로컬 테스트
    asyncio.run(run_local_test())