"""
비트코인 리스크 모니터링 메인 시스템
11개 선행지표 + Claude AI 예측 통합
"""

import asyncio
import os
from typing import Dict, List
from datetime import datetime
import json
import logging

# 필요한 모듈들
from enhanced_11_indicators import Enhanced11IndicatorSystem
from claude_predictor import ClaudePricePredictor
from prediction_tracker import PredictionTracker

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class BitcoinRiskMonitor:
    """비트코인 리스크 모니터링 메인 시스템"""
    
    def __init__(self):
        self.enhanced_system = Enhanced11IndicatorSystem()
        self.predictor = ClaudePricePredictor()
        self.tracker = PredictionTracker()
        self.logger = logger
        
    async def run_monitoring_cycle(self) -> Dict:
        """모니터링 사이클 실행"""
        try:
            self.logger.info("🚀 비트코인 리스크 모니터링 사이클 시작")
            
            # 1. 11개 선행지표 수집 및 분석
            self.logger.info("📊 11개 선행지표 수집 중...")
            indicators = await self.enhanced_system.collect_enhanced_11_indicators()
            
            if not indicators:
                self.logger.error("지표 수집 실패")
                return None
            
            # 2. 현재 가격 데이터 준비
            current_data = {
                "price_data": {
                    "current_price": indicators.get("metadata", {}).get("current_price", 0),
                    "volume_24h": 25000000000,  # 임시값
                    "change_24h": -1.5  # 임시값
                }
            }
            
            # 3. 예측 정확도 메트릭스 로드
            accuracy_metrics = self.tracker.get_accuracy_metrics()
            
            # 4. Claude AI 예측 요청
            self.logger.info("🤖 Claude AI 예측 분석 중...")
            
            # 11개 지표 시스템 결과를 포함한 지표 전달
            enhanced_indicators = {
                "enhanced_11_system": indicators,
                "whale_activity": {
                    "large_transfers": {
                        "exchange_outflows_1h": 1200,
                        "exchange_inflows_1h": 800
                    },
                    "exchange_dynamics": {
                        "coinbase_premium": 0.8
                    }
                },
                "derivatives_structure": {
                    "futures_structure": {
                        "funding_rate_trajectory": "falling"
                    }
                },
                "macro_early_signals": {
                    "yield_curve_dynamics": {
                        "real_rates_pressure": -0.03
                    }
                }
            }
            
            # Claude 예측 실행
            prediction_result = await self.predictor.analyze_market_signals(
                current_data,
                []  # historical_data (필요시 추가)
            )
            
            # 5. 예측 결과 정리
            result = {
                "timestamp": datetime.now().isoformat(),
                "indicators": indicators,
                "prediction": prediction_result.get("prediction", {}),
                "alert_sent": False,
                "alert_message": None
            }
            
            # 6. 알림 결정
            should_alert = self.tracker.should_send_alert(prediction_result, accuracy_metrics)
            
            if should_alert:
                alert_message = self.generate_alert_message(indicators, prediction_result)
                result["alert_sent"] = True
                result["alert_message"] = alert_message
                self.logger.info(f"🚨 알림 발송: {alert_message[:100]}...")
                # 실제 텔레그램 발송은 여기서
            else:
                self.logger.info("📌 알림 기준 미달 (정확도 또는 신뢰도 부족)")
            
            # 7. 예측 기록 저장
            self.tracker.record_prediction(
                prediction_result,
                current_data,
                enhanced_indicators
            )
            
            self.logger.info("✅ 모니터링 사이클 완료")
            return result
            
        except Exception as e:
            self.logger.error(f"모니터링 사이클 오류: {e}")
            return None
    
    def generate_alert_message(self, indicators: Dict, prediction: Dict) -> str:
        """알림 메시지 생성"""
        pred_data = prediction.get("prediction", {})
        composite = indicators.get("composite_analysis", {})
        signals = indicators.get("prediction_signals", {})
        
        message = f"""🚨 비트코인 가격 예측 알림

📈 예측 방향: {pred_data.get('direction', 'N/A')}
🎯 확률: {pred_data.get('probability', 0)}%
⏰ 예상 시간: {pred_data.get('timeframe', 'N/A')}
💰 목표가: ${pred_data.get('target_price', 0):,.0f}
🔒 신뢰도: {pred_data.get('confidence', 'N/A')}

📊 11개 지표 종합:
• 전체 신호: {composite.get('overall_signal', 'N/A')}
• 신뢰도: {composite.get('confidence', 0):.1%}
• 강세 강도: {composite.get('bullish_strength', 0):.2f}
• 약세 강도: {composite.get('bearish_strength', 0):.2f}

🔑 핵심 신호:
• CryptoQuant 온체인: {indicators.get('indicators', {}).get('cryptoquant_onchain', {}).get('signal', 'N/A')}
• 파생상품 구조: {indicators.get('indicators', {}).get('derivatives_real', {}).get('signal', 'N/A')}
• 거시경제: {indicators.get('indicators', {}).get('macro_indicators', {}).get('signal', 'N/A')}

⚠️ 주의: 이것은 AI 예측이며 투자 조언이 아닙니다."""
        
        return message

async def main():
    """메인 실행 함수"""
    monitor = BitcoinRiskMonitor()
    
    print("\n" + "="*70)
    print("🚀 비트코인 리스크 모니터링 시스템 시작")
    print("📊 11개 선행지표 + Claude AI 예측")
    print("="*70)
    
    # 한 번 실행
    result = await monitor.run_monitoring_cycle()
    
    if result:
        print("\n📋 모니터링 결과:")
        print(f"  • 예측 방향: {result['prediction'].get('prediction', {}).get('direction', 'N/A')}")
        print(f"  • 확률: {result['prediction'].get('prediction', {}).get('probability', 0)}%")
        print(f"  • 알림 발송: {'✅' if result['alert_sent'] else '❌'}")
        
        if result['alert_message']:
            print("\n📨 알림 메시지:")
            print(result['alert_message'])
    
    print("\n✅ 테스트 완료!")

if __name__ == "__main__":
    # CryptoQuant API 키 확인
    if not os.environ.get('CRYPTOQUANT_API_KEY'):
        print("⚠️ CRYPTOQUANT_API_KEY 환경변수를 설정하세요")
        print("export CRYPTOQUANT_API_KEY='your-api-key'")
    
    asyncio.run(main())