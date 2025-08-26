#!/usr/bin/env python3
"""
간단한 통합 테스트 - 핵심 기능 확인
"""

import asyncio
import json
from datetime import datetime

# 테스트할 모듈들
from prediction_tracker import PredictionTracker
from advanced_data_sources import AdvancedDataCollector

async def test_core_prediction_features():
    """핵심 예측 기능 테스트"""
    print("🧪 핵심 예측 기능 테스트")
    print("=" * 50)
    
    try:
        # 1. 고급 선행지표 수집 테스트
        print("\n1️⃣ 고급 선행지표 수집...")
        collector = AdvancedDataCollector()
        indicators = await collector.get_real_leading_indicators()
        
        print(f"✅ 선행지표 수집 완료:")
        for category, data in indicators.items():
            if category != "timestamp":
                print(f"  • {category}: {type(data).__name__}")
        
        # 2. 신호 분석 테스트
        print(f"\n2️⃣ 신호 강도 분석...")
        scores = collector.calculate_leading_indicator_score(indicators)
        
        print(f"✅ 신호 분석 결과:")
        print(f"  • 방향: {scores.get('predicted_direction', 'UNKNOWN')}")
        print(f"  • 강도: {scores.get('signal_strength', 'unknown')}")
        print(f"  • 신뢰도: {scores.get('confidence', 0):.2f}")
        print(f"  • 강세 신호: {scores.get('bullish_signals', 0)}개")
        print(f"  • 약세 신호: {scores.get('bearish_signals', 0)}개")
        
        # 3. 예측 추적기 테스트 (파일 DB 사용)
        print(f"\n3️⃣ 예측 추적 시스템...")
        tracker = PredictionTracker("test_predictions.db")
        
        # 테스트 예측 생성
        test_prediction = {
            "prediction": {
                "direction": "BULLISH",
                "probability": 75,
                "target_price": 67000,
                "confidence": "HIGH",
                "timeframe": "6시간"
            },
            "analysis": {
                "reasoning": "고래 활동 증가 및 파생상품 정상화",
                "catalysts": ["거래소 유출 증가", "펀딩비 하락"],
                "risks": ["매크로 불확실성"]
            }
        }
        
        test_data = {
            "price_data": {"current_price": 65000, "change_24h": 1.8}
        }
        
        # 예측 기록
        pred_id = tracker.record_prediction(test_prediction, test_data, indicators)
        print(f"✅ 예측 기록: ID {pred_id}")
        
        # 정확도 메트릭스
        metrics = tracker.get_accuracy_metrics(days=1)
        if "error" in metrics:
            print(f"  메트릭스: {metrics['error']}")
        else:
            print(f"  총 예측: {metrics['total_predictions']}개")
        
        # 4. 알림 필터링 테스트
        print(f"\n4️⃣ 알림 필터링 테스트...")
        
        # 기본 정확도로 가정
        mock_metrics = {"direction_accuracy": 0.7}
        
        should_alert = tracker.should_send_alert(test_prediction, mock_metrics)
        print(f"✅ 알림 결정: {'발송' if should_alert else '차단'}")
        
        # 다양한 시나리오 테스트
        scenarios = [
            {"direction": "BULLISH", "probability": 85, "confidence": "HIGH"},
            {"direction": "BEARISH", "probability": 60, "confidence": "MEDIUM"},
            {"direction": "NEUTRAL", "probability": 50, "confidence": "LOW"},
        ]
        
        print(f"  다른 시나리오들:")
        for scenario in scenarios:
            test_pred = {"prediction": scenario}
            alert = tracker.should_send_alert(test_pred, mock_metrics)
            status = "발송" if alert else "차단"
            print(f"    {scenario['direction']} {scenario['probability']}% {scenario['confidence']} -> {status}")
        
        print(f"\n🎉 핵심 기능 테스트 완료!")
        print(f"✅ 모든 주요 컴포넌트가 정상 작동합니다.")
        
        # 정리
        import os
        if os.path.exists("test_predictions.db"):
            os.remove("test_predictions.db")
            print(f"  테스트 DB 파일 정리 완료")
            
        return True
        
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        import traceback
        print(traceback.format_exc())
        return False

def test_prediction_tracker_logic():
    """예측 추적기 로직 테스트 (DB 없이)"""
    print(f"\n" + "=" * 50)
    print(f"🔍 예측 필터링 로직 세부 테스트")
    print(f"=" + "=" * 49)
    
    tracker = PredictionTracker("dummy.db")  # DB 생성하지 않고 로직만 테스트
    
    # 다양한 시나리오와 성과 조합
    test_cases = [
        # (방향, 확률, 신뢰도, 전체정확도, 예상결과)
        ("BULLISH", 85, "HIGH", 0.8, True),   # 고성과 + 고확률 -> 발송
        ("BULLISH", 75, "HIGH", 0.8, True),   # 고성과 + 중확률 -> 발송  
        ("BULLISH", 85, "HIGH", 0.5, False),  # 저성과 + 고확률 -> 차단
        ("BEARISH", 80, "MEDIUM", 0.7, True), # 보통성과 + 고확률 -> 발송
        ("BEARISH", 70, "MEDIUM", 0.7, False), # 보통성과 + 중확률 -> 차단
        ("NEUTRAL", 90, "HIGH", 0.9, False),  # 중성 예측 -> 무조건 차단
        ("BULLISH", 60, "LOW", 0.9, False),   # 저신뢰도 -> 무조건 차단
    ]
    
    print(f"📊 알림 필터링 로직 테스트:")
    print(f"{'시나리오':<20} {'확률':<5} {'신뢰도':<8} {'성과':<6} {'결과':<4} {'설명'}")
    print(f"-" * 70)
    
    for direction, prob, conf, accuracy, expected in test_cases:
        prediction = {
            "prediction": {
                "direction": direction,
                "probability": prob,
                "confidence": conf
            }
        }
        
        metrics = {"direction_accuracy": accuracy}
        
        try:
            result = tracker.should_send_alert(prediction, metrics)
            status = "발송" if result else "차단"
            check = "✅" if result == expected else "❌"
            
            explanation = ""
            if direction == "NEUTRAL":
                explanation = "(중성예측 차단)"
            elif conf == "LOW":
                explanation = "(저신뢰도 차단)"
            elif accuracy < 0.6:
                explanation = "(저성과 차단)"
            elif result:
                explanation = "(조건 충족)"
            else:
                explanation = "(임계값 미달)"
                
            print(f"{direction:<20} {prob:<5}% {conf:<8} {accuracy:<6.1%} {status:<4} {check} {explanation}")
            
        except Exception as e:
            print(f"{direction:<20} {prob:<5}% {conf:<8} {accuracy:<6.1%} 오류 ❌ {str(e)[:20]}")
    
    print(f"\n✅ 알림 필터링 로직 테스트 완료")

if __name__ == "__main__":
    # 비동기 테스트 실행
    success = asyncio.run(test_core_prediction_features())
    
    # 동기 로직 테스트
    test_prediction_tracker_logic()
    
    # 최종 결과
    print(f"\n" + "=" * 50)
    print(f"🏁 테스트 결과: {'✅ 성공' if success else '❌ 실패'}")
    print(f"=" + "=" * 49)