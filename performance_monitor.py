#!/usr/bin/env python3
"""
🔍 BTC 학습 시스템 실시간 성능 모니터링
- 95% 정확도 달성 감지
- 지속적인 성능 추적
- 자동 알림 및 보고서
"""

import json
import time
import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass
import logging

@dataclass
class PerformanceMetrics:
    """성능 지표 데이터 클래스"""
    timestamp: str
    test_point: int
    direction_accuracy: float
    price_accuracy: float
    timing_accuracy: float
    combined_accuracy: float
    confidence: float
    predicted_price: float
    actual_price: float
    price_error_rate: float

class RealTimeMonitor:
    """실시간 성능 모니터링 시스템"""
    
    def __init__(self, target_accuracy: float = 0.95):
        self.target_accuracy = target_accuracy
        self.metrics_history: List[PerformanceMetrics] = []
        self.achievement_points: List[Dict] = []
        self.current_streak = 0
        self.best_accuracy = 0.0
        self.total_tests = 0
        
        # 로깅 설정
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('performance_monitor.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def add_test_result(self, test_point: int, direction_correct: bool, 
                       price_accuracy: float, timing_accuracy: float,
                       confidence: float, predicted_price: float, 
                       actual_price: float) -> None:
        """테스트 결과를 추가하고 분석"""
        
        # 종합 정확도 계산 (방향 50%, 가격 30%, 타이밍 20%)
        direction_score = 100.0 if direction_correct else 0.0
        combined_accuracy = (
            direction_score * 0.5 + 
            price_accuracy * 0.3 + 
            timing_accuracy * 0.2
        ) / 100.0
        
        # 가격 오차율 계산
        price_error_rate = abs(predicted_price - actual_price) / actual_price * 100
        
        # 메트릭 생성
        metrics = PerformanceMetrics(
            timestamp=datetime.datetime.now().isoformat(),
            test_point=test_point,
            direction_accuracy=direction_score,
            price_accuracy=price_accuracy,
            timing_accuracy=timing_accuracy,
            combined_accuracy=combined_accuracy * 100,  # 퍼센트로 변환
            confidence=confidence,
            predicted_price=predicted_price,
            actual_price=actual_price,
            price_error_rate=price_error_rate
        )
        
        self.metrics_history.append(metrics)
        self.total_tests += 1
        
        # 성능 분석
        self._analyze_performance(metrics)
        
        # 95% 달성 체크
        if combined_accuracy >= self.target_accuracy:
            self._handle_target_achievement(metrics)
            
    def _analyze_performance(self, metrics: PerformanceMetrics) -> None:
        """성능 분석 및 로깅"""
        
        # 최고 정확도 업데이트
        if metrics.combined_accuracy > self.best_accuracy:
            self.best_accuracy = metrics.combined_accuracy
            self.logger.info(f"🚀 신기록 달성: {self.best_accuracy:.1f}%")
            
        # 연속 성공 스트릭 추적
        if metrics.combined_accuracy >= 90.0:  # 90% 이상
            self.current_streak += 1
            if self.current_streak >= 3:
                self.logger.info(f"🔥 연속 고성능: {self.current_streak}회 연속 90%+")
        else:
            self.current_streak = 0
            
        # 실시간 상태 로깅
        status_emoji = "🎯" if metrics.combined_accuracy >= 95.0 else "📈" if metrics.combined_accuracy >= 90.0 else "🔄"
        
        self.logger.info(
            f"{status_emoji} 테스트 {metrics.test_point}: "
            f"종합 {metrics.combined_accuracy:.1f}% "
            f"(방향성: {'✅' if metrics.direction_accuracy > 0 else '❌'}, "
            f"가격: {metrics.price_accuracy:.1f}%, "
            f"타이밍: {metrics.timing_accuracy:.1f}%, "
            f"신뢰도: {metrics.confidence:.2f})"
        )
        
    def _handle_target_achievement(self, metrics: PerformanceMetrics) -> None:
        """95% 목표 달성 처리"""
        
        achievement = {
            "timestamp": metrics.timestamp,
            "test_point": metrics.test_point,
            "accuracy": metrics.combined_accuracy,
            "details": {
                "direction": metrics.direction_accuracy,
                "price": metrics.price_accuracy,
                "timing": metrics.timing_accuracy,
                "confidence": metrics.confidence,
                "price_error": metrics.price_error_rate
            }
        }
        
        self.achievement_points.append(achievement)
        
        self.logger.info("🎉" * 10)
        self.logger.info(f"🎯 95% 목표 달성! 테스트 {metrics.test_point}")
        self.logger.info(f"📊 종합 정확도: {metrics.combined_accuracy:.1f}%")
        self.logger.info(f"   - 방향성: {metrics.direction_accuracy:.1f}%")
        self.logger.info(f"   - 가격: {metrics.price_accuracy:.1f}%")
        self.logger.info(f"   - 타이밍: {metrics.timing_accuracy:.1f}%")
        self.logger.info(f"💰 가격 예측: ${metrics.predicted_price:,.2f} → ${metrics.actual_price:,.2f}")
        self.logger.info(f"📉 오차율: {metrics.price_error_rate:.2f}%")
        self.logger.info("🎉" * 10)
        
    def generate_performance_report(self) -> Dict:
        """성능 보고서 생성"""
        
        if not self.metrics_history:
            return {"error": "No data available"}
            
        # 통계 계산
        recent_10 = self.metrics_history[-10:] if len(self.metrics_history) >= 10 else self.metrics_history
        
        avg_accuracy = sum(m.combined_accuracy for m in recent_10) / len(recent_10)
        avg_price_error = sum(m.price_error_rate for m in recent_10) / len(recent_10)
        direction_success_rate = sum(1 for m in recent_10 if m.direction_accuracy > 0) / len(recent_10) * 100
        
        # 95% 달성 통계
        achievement_count = len(self.achievement_points)
        achievement_rate = (achievement_count / self.total_tests * 100) if self.total_tests > 0 else 0
        
        report = {
            "summary": {
                "총_테스트_수": self.total_tests,
                "95%_달성_횟수": achievement_count,
                "95%_달성률": f"{achievement_rate:.1f}%",
                "최고_정확도": f"{self.best_accuracy:.1f}%",
                "현재_연속_고성능": self.current_streak
            },
            "최근_10회_평균": {
                "종합_정확도": f"{avg_accuracy:.1f}%",
                "방향성_성공률": f"{direction_success_rate:.1f}%",
                "평균_가격_오차": f"{avg_price_error:.2f}%"
            },
            "달성_기록": self.achievement_points[-5:] if self.achievement_points else [],
            "성능_트렌드": self._calculate_trend(),
            "생성_시각": datetime.datetime.now().isoformat()
        }
        
        return report
        
    def _calculate_trend(self) -> str:
        """성능 트렌드 계산"""
        
        if len(self.metrics_history) < 10:
            return "데이터 부족"
            
        recent_5 = self.metrics_history[-5:]
        previous_5 = self.metrics_history[-10:-5]
        
        recent_avg = sum(m.combined_accuracy for m in recent_5) / 5
        previous_avg = sum(m.combined_accuracy for m in previous_5) / 5
        
        diff = recent_avg - previous_avg
        
        if diff > 5:
            return "🚀 급상승"
        elif diff > 2:
            return "📈 상승"
        elif diff > -2:
            return "➡️ 안정"
        elif diff > -5:
            return "📉 하락"
        else:
            return "⚠️ 급하락"
            
    def save_metrics_to_file(self, filename: str = "performance_metrics.json") -> None:
        """메트릭을 파일로 저장"""
        
        data = {
            "metrics_history": [
                {
                    "timestamp": m.timestamp,
                    "test_point": m.test_point,
                    "direction_accuracy": m.direction_accuracy,
                    "price_accuracy": m.price_accuracy,
                    "timing_accuracy": m.timing_accuracy,
                    "combined_accuracy": m.combined_accuracy,
                    "confidence": m.confidence,
                    "predicted_price": m.predicted_price,
                    "actual_price": m.actual_price,
                    "price_error_rate": m.price_error_rate
                }
                for m in self.metrics_history
            ],
            "achievement_points": self.achievement_points,
            "summary": self.generate_performance_report()
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            
        self.logger.info(f"📄 성능 데이터 저장: {filename}")
        
    def load_metrics_from_file(self, filename: str = "performance_metrics.json") -> bool:
        """파일에서 메트릭 로드"""
        
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # 메트릭 히스토리 복원
            self.metrics_history = []
            for m_data in data.get("metrics_history", []):
                metrics = PerformanceMetrics(
                    timestamp=m_data["timestamp"],
                    test_point=m_data["test_point"],
                    direction_accuracy=m_data["direction_accuracy"],
                    price_accuracy=m_data["price_accuracy"],
                    timing_accuracy=m_data["timing_accuracy"],
                    combined_accuracy=m_data["combined_accuracy"],
                    confidence=m_data["confidence"],
                    predicted_price=m_data["predicted_price"],
                    actual_price=m_data["actual_price"],
                    price_error_rate=m_data["price_error_rate"]
                )
                self.metrics_history.append(metrics)
                
            # 달성 기록 복원
            self.achievement_points = data.get("achievement_points", [])
            
            # 통계 업데이트
            self.total_tests = len(self.metrics_history)
            if self.metrics_history:
                self.best_accuracy = max(m.combined_accuracy for m in self.metrics_history)
                
            self.logger.info(f"📂 성능 데이터 로드: {len(self.metrics_history)}개 기록")
            return True
            
        except FileNotFoundError:
            self.logger.warning(f"📂 파일 없음: {filename}")
            return False
        except Exception as e:
            self.logger.error(f"❌ 데이터 로드 실패: {e}")
            return False

def main():
    """실시간 모니터링 시스템 테스트"""
    
    print("🔍 BTC 성능 모니터링 시스템 시작")
    
    monitor = RealTimeMonitor()
    
    # 기존 데이터 로드 시도
    monitor.load_metrics_from_file()
    
    # 샘플 데이터 추가 (실제로는 btc_learning_system.py에서 호출)
    sample_results = [
        (168, False, 86.8, 40.4, 0.67, 83015.12, 73320.95),  # 초기 저성능
        (172, True, 60.0, 60.0, 1.00, 80235.87, 73441.35),   # 첫 성공
        (214, True, 100.0, 79.0, 0.90, 73686.25, 74258.53), # 95% 달성!
    ]
    
    for test_point, direction_correct, price_acc, timing_acc, confidence, pred_price, actual_price in sample_results:
        monitor.add_test_result(
            test_point=test_point,
            direction_correct=direction_correct,
            price_accuracy=price_acc,
            timing_accuracy=timing_acc,
            confidence=confidence,
            predicted_price=pred_price,
            actual_price=actual_price
        )
        time.sleep(0.5)  # 시뮬레이션을 위한 딜레이
    
    # 성능 보고서 생성
    report = monitor.generate_performance_report()
    print("\n📊 성능 보고서:")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    
    # 파일로 저장
    monitor.save_metrics_to_file()

if __name__ == "__main__":
    main()