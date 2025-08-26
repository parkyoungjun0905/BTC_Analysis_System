#!/usr/bin/env python3
"""
적응적 학습 시스템 통합 테스트
"""

import asyncio
import json
import sqlite3
from datetime import datetime, timedelta
from adaptive_learning_engine import AdaptiveLearningEngine
from prediction_tracker import PredictionTracker

async def test_learning_system():
    """학습 시스템 전체 테스트"""
    print("\n" + "="*80)
    print("🧠 적응적 학습 시스템 통합 테스트")
    print("="*80)
    
    # 1. 컴포넌트 초기화
    print("\n📦 1/7 - 시스템 초기화...")
    learning_engine = AdaptiveLearningEngine()
    tracker = PredictionTracker()
    
    # 2. 테스트 예측 데이터 생성
    print("📝 2/7 - 테스트 예측 데이터 생성...")
    test_predictions = await generate_test_predictions(tracker)
    print(f"   ✅ {len(test_predictions)} 개 테스트 예측 생성")
    
    # 3. 실패 분석 테스트
    print("🔍 3/7 - 실패 분석 시스템 테스트...")
    failure_analysis = learning_engine.analyze_prediction_failures(7)
    
    if failure_analysis.get("total_failures", 0) > 0:
        print(f"   ✅ {failure_analysis['total_failures']}개 실패 사례 분석 완료")
        print(f"   📊 가장 흔한 실패 유형: {failure_analysis['pattern_summary']['most_common_failure_type']}")
    else:
        print("   ℹ️  분석할 실패 사례가 없습니다")
    
    # 4. 가중치 적응 테스트
    print("⚖️ 4/7 - 가중치 적응 시스템 테스트...")
    if failure_analysis.get("failure_analyses"):
        adaptation_result = learning_engine.adapt_indicator_weights(
            failure_analysis["failure_analyses"]
        )
        
        if adaptation_result.get("total_adjustments", 0) > 0:
            print(f"   ✅ {adaptation_result['total_adjustments']}개 지표 가중치 조정 완료")
            print(f"   📈 {adaptation_result['adaptation_summary']}")
        else:
            print("   ℹ️  조정된 가중치가 없습니다")
    else:
        print("   ⏭️  실패 데이터 없음, 가중치 조정 건너뜀")
    
    # 5. 임계값 최적화 테스트
    print("🎯 5/7 - 동적 임계값 최적화 테스트...")
    threshold_result = learning_engine.optimize_dynamic_thresholds()
    
    if "new_confidence_threshold" in threshold_result:
        old_th = threshold_result.get("old_confidence_threshold", 0)
        new_th = threshold_result.get("new_confidence_threshold", 0)
        print(f"   ✅ 임계값 최적화: {old_th}% → {new_th}%")
        
        improvement = threshold_result.get("expected_accuracy_improvement", 0)
        print(f"   📊 예상 정확도 향상: +{improvement:.1f}%")
    else:
        print("   ℹ️  임계값 최적화할 데이터 부족")
    
    # 6. 학습 인사이트 생성 테스트
    print("💡 6/7 - 학습 인사이트 생성 테스트...")
    insights = learning_engine.generate_learning_insights()
    
    if insights.get("top_performing_indicators"):
        top_indicators = insights["top_performing_indicators"][:3]
        print("   🏆 최고 성능 지표들:")
        for i, indicator in enumerate(top_indicators, 1):
            print(f"      {i}. {indicator['indicator']}: {indicator['weight']:.2f}")
    
    if insights.get("recommendations"):
        print(f"   💭 추천사항: {len(insights['recommendations'])}개")
        for rec in insights["recommendations"][:2]:
            print(f"      • {rec}")
    
    # 7. 전체 시스템 성능 평가
    print("📈 7/7 - 전체 시스템 성능 평가...")
    performance_stats = await evaluate_system_performance(tracker)
    
    print(f"   📊 전체 성능 통계:")
    print(f"      • 총 예측: {performance_stats['total_predictions']}개")
    print(f"      • 방향 정확도: {performance_stats['direction_accuracy']:.1f}%")
    print(f"      • 가격 정확도: {performance_stats['price_accuracy']:.1f}%")
    print(f"      • 거짓 양성률: {performance_stats['false_positive_rate']:.1f}%")
    
    # 학습 효과 확인
    learning_effectiveness = calculate_learning_effectiveness(learning_engine)
    print(f"      • 학습 효과: {learning_effectiveness['score']:.1f}/10")
    print(f"      • 적응 빈도: {learning_effectiveness['adaptation_frequency']}")
    
    print("\n" + "="*80)
    print("🎉 적응적 학습 시스템 테스트 완료!")
    print("="*80)
    
    # 최종 요약
    return {
        "test_status": "completed",
        "failure_analysis": failure_analysis,
        "adaptation_result": adaptation_result if 'adaptation_result' in locals() else None,
        "threshold_optimization": threshold_result,
        "insights": insights,
        "performance_stats": performance_stats,
        "learning_effectiveness": learning_effectiveness
    }

async def generate_test_predictions(tracker: PredictionTracker) -> list:
    """테스트용 예측 데이터 생성"""
    test_data = []
    
    # 다양한 시나리오의 테스트 예측 생성
    base_time = datetime.utcnow() - timedelta(days=10)
    base_price = 45000
    
    scenarios = [
        # 성공적인 예측들
        {"direction": "BULLISH", "prob": 85, "conf": "HIGH", "actual_dir": "BULLISH", "success": True},
        {"direction": "BEARISH", "prob": 78, "conf": "HIGH", "actual_dir": "BEARISH", "success": True},
        {"direction": "NEUTRAL", "prob": 65, "conf": "MEDIUM", "actual_dir": "NEUTRAL", "success": True},
        
        # 실패한 예측들
        {"direction": "BULLISH", "prob": 82, "conf": "HIGH", "actual_dir": "BEARISH", "success": False},
        {"direction": "BEARISH", "prob": 75, "conf": "MEDIUM", "actual_dir": "BULLISH", "success": False},
        {"direction": "NEUTRAL", "prob": 90, "conf": "HIGH", "actual_dir": "BULLISH", "success": False},
        
        # 낮은 신뢰도 예측들
        {"direction": "BULLISH", "prob": 55, "conf": "LOW", "actual_dir": "NEUTRAL", "success": False},
        {"direction": "BEARISH", "prob": 58, "conf": "LOW", "actual_dir": "BULLISH", "success": False},
    ]
    
    for i, scenario in enumerate(scenarios):
        timestamp = base_time + timedelta(hours=i*6)
        current_price = base_price + (i * 500)  # 가격 변동
        predicted_price = current_price * (1.03 if scenario["direction"] == "BULLISH" 
                                         else 0.97 if scenario["direction"] == "BEARISH" 
                                         else 1.0)
        
        # 실제 가격 (성공/실패에 따라)
        if scenario["success"]:
            actual_price = predicted_price
        else:
            actual_price = current_price * (0.98 if scenario["direction"] == "BULLISH" 
                                          else 1.02 if scenario["direction"] == "BEARISH" 
                                          else 1.05)
        
        # 테스트 지표 데이터
        indicators_data = {
            "mempool_pressure": {"value": 0.7, "signal": scenario["direction"]},
            "funding_rate": {"value": 0.002, "signal": scenario["direction"]},
            "orderbook_imbalance": {"value": 0.6, "signal": scenario["direction"]},
        }
        
        # 예측 기록
        pred_id = tracker.record_prediction(
            current_price=current_price,
            prediction_direction=scenario["direction"],
            predicted_price=predicted_price,
            probability=scenario["prob"],
            confidence=scenario["conf"],
            timeframe_hours=4,
            leading_indicators=indicators_data,
            claude_reasoning=f"Test prediction {i+1}",
            current_data={"price_data": {"current_price": current_price}}
        )
        
        # 즉시 평가를 위해 과거 시간으로 설정
        if pred_id > 0:
            conn = sqlite3.connect(tracker.db_path)
            cursor = conn.cursor()
            
            # 예측을 과거로 이동하여 평가 가능하게 만듦
            past_timestamp = (timestamp - timedelta(hours=5)).isoformat()
            cursor.execute('''
                UPDATE predictions 
                SET timestamp = ?
                WHERE id = ?
            ''', (past_timestamp, pred_id))
            
            conn.commit()
            conn.close()
            
            test_data.append({
                "id": pred_id,
                "scenario": scenario,
                "current_price": current_price,
                "predicted_price": predicted_price,
                "actual_price": actual_price
            })
    
    # 예측들을 평가
    current_data = {"price_data": {"current_price": base_price + 4000}}
    evaluation_result = tracker.evaluate_predictions(current_data)
    
    return test_data

async def evaluate_system_performance(tracker: PredictionTracker) -> dict:
    """시스템 전체 성능 평가"""
    try:
        # 최근 7일간 성능 메트릭
        metrics = tracker.calculate_accuracy_metrics(7)
        
        return {
            "total_predictions": metrics.get("total_predictions", 0),
            "direction_accuracy": metrics.get("direction_accuracy", 0) * 100,
            "price_accuracy": metrics.get("avg_price_accuracy", 0) * 100,
            "false_positive_rate": metrics.get("false_positive_rate", 0) * 100,
            "quality_score": metrics.get("quality_score", 0)
        }
        
    except Exception as e:
        print(f"   ⚠️  성능 평가 오류: {e}")
        return {
            "total_predictions": 0,
            "direction_accuracy": 0,
            "price_accuracy": 0,
            "false_positive_rate": 0,
            "quality_score": 0
        }

def calculate_learning_effectiveness(learning_engine: AdaptiveLearningEngine) -> dict:
    """학습 효과 계산"""
    try:
        # 가중치 분산도 (낮을수록 균형잡힘)
        weights = list(learning_engine.learned_weights.values())
        weight_variance = sum((w - sum(weights)/len(weights))**2 for w in weights) / len(weights)
        
        # 정규화된 효과 점수 (0-10)
        balance_score = max(0, 5 - weight_variance)  # 분산이 낮을수록 높은 점수
        adaptation_score = min(5, len(learning_engine.learned_weights) / 4)  # 적응된 지표 수
        
        total_score = balance_score + adaptation_score
        
        return {
            "score": total_score,
            "balance_score": balance_score,
            "adaptation_score": adaptation_score,
            "weight_variance": weight_variance,
            "adapted_indicators": len(learning_engine.learned_weights),
            "adaptation_frequency": "정상" if len(learning_engine.learned_weights) > 10 else "제한적"
        }
        
    except Exception as e:
        return {
            "score": 0,
            "error": str(e),
            "adaptation_frequency": "오류"
        }

if __name__ == "__main__":
    asyncio.run(test_learning_system())