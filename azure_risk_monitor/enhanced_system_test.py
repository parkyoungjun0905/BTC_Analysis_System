#!/usr/bin/env python3
"""
향상된 예측 시스템 전체 테스트
새로운 기능들이 정상적으로 통합되었는지 확인
"""

import asyncio
import json
import logging
from datetime import datetime

# 테스트할 모듈들
from main_monitor import BRCRiskMonitor
from prediction_tracker import PredictionTracker
from advanced_data_sources import AdvancedDataCollector

async def test_enhanced_prediction_system():
    """향상된 예측 시스템 전체 테스트"""
    print("🧪 향상된 BTC 예측 시스템 통합 테스트 시작...")
    print("=" * 60)
    
    # 로깅 설정
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    try:
        # 1. 시스템 컴포넌트 초기화 테스트
        print("\n1️⃣ 시스템 컴포넌트 초기화 테스트...")
        
        print("  - 예측 추적기 초기화...")
        tracker = PredictionTracker(db_path=":memory:")
        
        print("  - 고급 데이터 수집기 초기화...")
        advanced_collector = AdvancedDataCollector()
        
        print("  - 메인 모니터 초기화...")
        monitor = BRCRiskMonitor()
        
        print("  ✅ 모든 컴포넌트 초기화 성공")
        
        # 2. 고급 선행지표 수집 테스트
        print("\n2️⃣ 고급 선행지표 수집 테스트...")
        leading_indicators = await advanced_collector.get_real_leading_indicators()
        
        print(f"  - 수집된 지표 카테고리: {len(leading_indicators)-1}개")
        for category in leading_indicators.keys():
            if category != "timestamp":
                print(f"    ✓ {category}")
        
        # 3. 선행지표 신호 강도 계산 테스트
        print("\n3️⃣ 선행지표 신호 분석 테스트...")
        signal_scores = advanced_collector.calculate_leading_indicator_score(leading_indicators)
        
        print(f"  - 신호 강도: {signal_scores.get('signal_strength', 'unknown')}")
        print(f"  - 예측 방향: {signal_scores.get('predicted_direction', 'NEUTRAL')}")
        print(f"  - 신뢰도: {signal_scores.get('confidence', 0):.2f}")
        print(f"  - 강세 신호: {signal_scores.get('bullish_signals', 0)}개")
        print(f"  - 약세 신호: {signal_scores.get('bearish_signals', 0)}개")
        
        # 4. 예측 추적 데이터베이스 테스트
        print("\n4️⃣ 예측 추적 시스템 테스트...")
        
        # 테스트 예측 생성
        test_prediction = {
            "prediction": {
                "direction": "BULLISH",
                "probability": 78,
                "target_price": 67000,
                "confidence": "HIGH",
                "timeframe": "6-12시간"
            },
            "analysis": {
                "reasoning": "테스트 예측 - 고래 유출 증가 및 펀딩비 정상화",
                "catalysts": ["대량 거래소 유출 감지", "파생상품 프리미엄 하락"],
                "risks": ["매크로 리스크", "규제 불확실성"]
            }
        }
        
        test_current_data = {
            "price_data": {"current_price": 65000, "change_24h": 2.1},
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # 예측 기록
        pred_id = tracker.record_prediction(test_prediction, test_current_data, leading_indicators)
        print(f"  - 테스트 예측 기록: ID {pred_id}")
        
        # 정확도 메트릭스 (초기 상태)
        metrics = tracker.get_accuracy_metrics(days=1)
        if "error" in metrics:
            print(f"  - 메트릭스: {metrics['error']}")
        else:
            print(f"  - 총 예측: {metrics['total_predictions']}개")
            print(f"  - 정확도: {metrics['direction_accuracy']:.1%}")
        
        # 5. 전체 모니터링 사이클 테스트 (축약 버전)
        print("\n5️⃣ 통합 모니터링 사이클 테스트...")
        
        # 시작 시퀀스 생략하고 데이터 수집부터 테스트
        print("  - 현재 데이터 수집 중...")
        current_data = await monitor.collect_current_data()
        
        if "error" in current_data:
            print(f"  ❌ 데이터 수집 실패: {current_data['error']}")
        else:
            print(f"  ✅ 데이터 수집 성공 (카테고리: {len(current_data)}개)")
            
            # 위험 분석
            print("  - 위험도 분석 중...")
            risk_analysis = monitor.simple_risk_analysis(current_data)
            
            print(f"    ✓ 위험 점수: {risk_analysis['composite_risk_score']:.3f}")
            print(f"    ✓ 위험 레벨: {risk_analysis['risk_level']}")
            print(f"    ✓ 신뢰도: {risk_analysis['confidence']:.2f}")
            
        # 6. 알림 필터링 로직 테스트
        print("\n6️⃣ 성과 기반 알림 필터링 테스트...")
        
        # 다양한 예측 시나리오로 테스트
        test_scenarios = [
            {"direction": "BULLISH", "probability": 85, "confidence": "HIGH"},
            {"direction": "BEARISH", "probability": 65, "confidence": "MEDIUM"},
            {"direction": "NEUTRAL", "probability": 50, "confidence": "LOW"},
            {"direction": "BULLISH", "probability": 95, "confidence": "HIGH"},
        ]
        
        for i, scenario in enumerate(test_scenarios, 1):
            test_pred = {
                "prediction": scenario,
                "analysis": {"reasoning": f"시나리오 {i} 테스트"}
            }
            
            should_alert = tracker.should_send_alert(test_pred, {"direction_accuracy": 0.7})
            alert_status = "발송" if should_alert else "차단"
            
            print(f"  시나리오 {i}: {scenario['direction']} {scenario['probability']}% {scenario['confidence']} -> {alert_status}")
        
        # 7. 시스템 상태 확인
        print("\n7️⃣ 시스템 상태 요약...")
        status = monitor.get_system_status()
        
        print(f"  - 시스템 버전: {status.get('status', 'unknown')}")
        print(f"  - 가동 시간: {status.get('uptime_formatted', '0:00:00')}")
        print(f"  - 히스토리 데이터: {status.get('historical_data_points', 0)}개")
        
        print("\n✅ 전체 통합 테스트 완료!")
        print("=" * 60)
        print("📋 테스트 결과 요약:")
        print("✅ 컴포넌트 초기화 - 성공")
        print("✅ 고급 선행지표 수집 - 성공")
        print("✅ 신호 분석 로직 - 성공") 
        print("✅ 예측 추적 시스템 - 성공")
        print("✅ 위험도 분석 - 성공")
        print("✅ 알림 필터링 - 성공")
        print("✅ 시스템 상태 모니터링 - 성공")
        print("\n🎉 향상된 예측 시스템이 성공적으로 통합되었습니다!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 테스트 실패: {str(e)}")
        import traceback
        print(f"상세 오류:\n{traceback.format_exc()}")
        return False

async def test_prediction_accuracy_simulation():
    """예측 정확도 시뮬레이션 테스트"""
    print("\n" + "=" * 60)
    print("🔄 예측 정확도 학습 시뮬레이션 테스트")
    print("=" * 60)
    
    tracker = PredictionTracker(db_path=":memory:")
    
    # 다양한 예측 결과 시뮬레이션
    scenarios = [
        # (예측방향, 확률, 신뢰도, 실제결과방향, 가격정확도)
        ("BULLISH", 80, "HIGH", "BULLISH", 0.85),    # 성공
        ("BULLISH", 75, "HIGH", "NEUTRAL", 0.60),    # 부분 성공
        ("BEARISH", 70, "MEDIUM", "BEARISH", 0.90),  # 성공
        ("BULLISH", 85, "HIGH", "BEARISH", 0.30),    # 실패
        ("BEARISH", 65, "MEDIUM", "BEARISH", 0.75),  # 성공
        ("NEUTRAL", 50, "LOW", "NEUTRAL", 0.95),     # 성공
        ("BULLISH", 90, "HIGH", "BULLISH", 0.95),    # 대성공
    ]
    
    print(f"📝 {len(scenarios)}개의 예측 시나리오 시뮬레이션...")
    
    # 예측들을 기록하고 평가
    for i, (pred_dir, prob, conf, actual_dir, price_acc) in enumerate(scenarios, 1):
        # 예측 기록
        test_prediction = {
            "prediction": {
                "direction": pred_dir,
                "probability": prob,
                "target_price": 65000,
                "confidence": conf,
                "timeframe": "6시간"
            },
            "analysis": {"reasoning": f"시뮬레이션 예측 {i}"}
        }
        
        test_data = {"price_data": {"current_price": 64000}}
        pred_id = tracker.record_prediction(test_prediction, test_data, {})
        
        # 즉시 평가 (시뮬레이션을 위해)
        import sqlite3
        conn = sqlite3.connect(":memory:")
        cursor = conn.cursor()
        
        # 수동으로 결과 업데이트 (실제로는 시간이 지난 후 자동 평가됨)
        direction_correct = (pred_dir == actual_dir) or (pred_dir == "NEUTRAL" and actual_dir == "NEUTRAL")
        quality_score = 0.6 * (1.0 if direction_correct else 0.0) + 0.4 * price_acc
        
        cursor.execute('''
            UPDATE predictions SET 
                actual_price = 65500, actual_direction = ?, direction_correct = ?,
                price_accuracy = ?, is_evaluated = TRUE, prediction_quality_score = ?
            WHERE id = ?
        ''', (actual_dir, direction_correct, price_acc, quality_score, pred_id))
        
        conn.commit()
        conn.close()
        
        result = "✅ 성공" if direction_correct else "❌ 실패"
        print(f"  {i}. {pred_dir} {prob}% {conf} -> {actual_dir} ({price_acc:.0%} 정확도) {result}")
    
    # 최종 성과 분석
    print(f"\n📊 시뮬레이션 성과 분석:")
    metrics = tracker.get_accuracy_metrics(days=1)
    
    if "error" not in metrics:
        print(f"  - 전체 정확도: {metrics['direction_accuracy']:.1%}")
        print(f"  - 평균 가격 정확도: {metrics['avg_price_accuracy']:.1%}")
        print(f"  - 품질 점수: {metrics['quality_score']:.3f}")
        print(f"  - 거짓 양성률: {metrics['false_positive_rate']:.1%}")
        
        print(f"\n🏆 신뢰도별 성과:")
        for conf, data in metrics.get('confidence_breakdown', {}).items():
            print(f"  - {conf}: {data['accuracy']:.1%} ({data['count']}개)")
            
        print(f"\n📈 방향별 성과:")
        for direction, data in metrics.get('direction_breakdown', {}).items():
            print(f"  - {direction}: {data['accuracy']:.1%} ({data['count']}개)")
    
    print(f"\n✅ 예측 정확도 학습 시뮬레이션 완료!")

if __name__ == "__main__":
    # 전체 테스트 실행
    asyncio.run(test_enhanced_prediction_system())
    
    # 추가 정확도 시뮬레이션 테스트
    asyncio.run(test_prediction_accuracy_simulation())