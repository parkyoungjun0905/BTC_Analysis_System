#!/usr/bin/env python3
"""
🧠 BTC Insight 핵심 시스템
- 백테스트 학습 시스템 총괄
- 예측 엔진 관리
"""

import os
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from pathlib import Path

from ..backtest.timetravel_learning import TimetravelLearningEngine
from ..analysis.timeseries_analyzer import TimeseriesAnalyzer
from ..utils.data_loader import DataLoader
from ..utils.logger import get_logger

class BTCInsightSystem:
    """BTC Insight 메인 시스템"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.base_path = Path(__file__).parent.parent.parent
        self.data_path = self.base_path / "ai_optimized_3month_data"
        
        # 핵심 컴포넌트 초기화
        self.data_loader = DataLoader(str(self.data_path))
        self.timetravel_engine = TimetravelLearningEngine()
        self.timeseries_analyzer = TimeseriesAnalyzer()
        
        # 학습 상태
        self.current_accuracy = 0.0
        self.target_accuracy = 95.0
        self.learning_history = []
        
        self.logger.info("BTC Insight 시스템 초기화 완료")
        print("🧠 BTC Insight 핵심 시스템 초기화")
        
    def run_backtest_learning(self, iterations: int = 100) -> Dict:
        """
        백테스트 학습 실행
        
        Args:
            iterations: 학습 반복 횟수
            
        Returns:
            학습 결과 딕셔너리
        """
        print(f"\n🔥 백테스트 학습 시작 (목표: {self.target_accuracy}% 정확도)")
        print("=" * 60)
        
        # 데이터 로드
        if not self.data_loader.load_data():
            self.logger.error("데이터 로드 실패")
            return None
            
        historical_data = self.data_loader.get_data()
        print(f"📊 학습 데이터: {historical_data.shape[0]}개 시점, {historical_data.shape[1]}개 지표")
        
        learning_results = []
        accuracy_progress = []
        
        for iteration in range(iterations):
            print(f"\n🔄 학습 반복 {iteration + 1:3d}/{iterations}")
            
            # 랜덤 시점 선택 (25년 7월 23일 같은 과거 시점)
            start_idx = np.random.randint(100, len(historical_data) - 200)  # 충분한 여유
            prediction_hours = np.random.choice([24, 48, 72])  # 1~3일 후 예측
            
            # 시간여행 백테스트 실행
            result = self.timetravel_engine.execute_backtest(
                historical_data, start_idx, prediction_hours
            )
            
            if result and result['success']:
                learning_results.append(result)
                accuracy = 100 - result['error_percentage']
                accuracy_progress.append(accuracy)
                
                print(f"   ✅ 정확도: {accuracy:.2f}% (에러: {result['error_percentage']:.2f}%)")
                
                # 실패 원인 분석 및 학습
                if result['error_percentage'] > 5.0:  # 5% 이상 에러시 원인 분석
                    self._analyze_failure_and_learn(result)
                    
            else:
                print(f"   ❌ 백테스트 실패")
                
            # 현재까지 평균 정확도 계산
            if accuracy_progress:
                current_avg_accuracy = np.mean(accuracy_progress[-10:])  # 최근 10회 평균
                self.current_accuracy = current_avg_accuracy
                
                if iteration % 10 == 9:  # 10회마다 상태 출력
                    print(f"📈 현재 평균 정확도: {current_avg_accuracy:.2f}%")
                    
                # 목표 달성 체크
                if current_avg_accuracy >= self.target_accuracy:
                    print(f"🎉 목표 달성! {current_avg_accuracy:.2f}% >= {self.target_accuracy}%")
                    break
        
        # 학습 결과 정리
        final_result = {
            'total_iterations': len(learning_results),
            'target_accuracy': self.target_accuracy,
            'achieved_accuracy': self.current_accuracy,
            'target_achieved': self.current_accuracy >= self.target_accuracy,
            'learning_results': learning_results,
            'accuracy_progress': accuracy_progress,
            'timestamp': datetime.now().isoformat()
        }
        
        # 결과 저장
        self._save_learning_results(final_result)
        
        print(f"\n🏆 학습 완료 결과:")
        print(f"   🎯 최종 정확도: {self.current_accuracy:.2f}%")
        print(f"   📊 성공한 학습: {len(learning_results)}회")
        print(f"   🏅 목표 달성: {'✅' if final_result['target_achieved'] else '❌'}")
        
        return final_result
        
    def _analyze_failure_and_learn(self, failure_result: Dict):
        """
        실패 원인 분석 및 학습
        
        Args:
            failure_result: 실패한 백테스트 결과
        """
        error_analysis = failure_result.get('error_analysis', {})
        
        # 급변동 이벤트 분석
        shock_events = error_analysis.get('shock_events', [])
        if shock_events:
            print(f"   💥 돌발변수 감지: {len(shock_events)}건")
            
        # 지표 기여도 분석
        indicator_changes = error_analysis.get('indicator_changes', [])
        if indicator_changes:
            top_indicators = sorted(indicator_changes, key=lambda x: x[1], reverse=True)[:3]
            print(f"   📊 주요 변화 지표: {[ind[0][:20] for ind in top_indicators]}")
            
        # 학습 히스토리에 추가
        self.learning_history.append({
            'timestamp': datetime.now().isoformat(),
            'error_percentage': failure_result['error_percentage'],
            'failure_reasons': error_analysis,
            'learned_patterns': self._extract_learning_patterns(error_analysis)
        })
        
    def _extract_learning_patterns(self, error_analysis: Dict) -> List[str]:
        """
        오류 분석에서 학습 패턴 추출
        
        Args:
            error_analysis: 오류 분석 결과
            
        Returns:
            학습된 패턴 리스트
        """
        patterns = []
        
        # 변동성 패턴
        if error_analysis.get('high_volatility', False):
            patterns.append("고변동성 구간에서 예측 어려움")
            
        # 돌발변수 패턴
        shock_events = error_analysis.get('shock_events', [])
        if len(shock_events) > 2:
            patterns.append("다수 돌발변수 발생시 예측 정확도 하락")
            
        # 지표 급변 패턴
        indicator_changes = error_analysis.get('indicator_changes', [])
        high_change_indicators = [ind for ind in indicator_changes if ind[1] > 20]
        if high_change_indicators:
            patterns.append(f"지표 급변시 주의: {[ind[0] for ind in high_change_indicators]}")
            
        return patterns
        
    def predict_future(self, hours_ahead: int = 72) -> Dict:
        """
        실시간 미래 예측
        
        Args:
            hours_ahead: 예측할 시간 (시간)
            
        Returns:
            예측 결과
        """
        print(f"\n🔮 {hours_ahead}시간 후 BTC 가격 예측")
        
        # 최신 데이터 로드
        if not self.data_loader.load_data():
            return None
            
        current_data = self.data_loader.get_latest_data()
        
        # 시계열 분석
        analysis_result = self.timeseries_analyzer.analyze(current_data)
        
        # 예측 실행
        prediction = self.timetravel_engine.predict_future(
            current_data, hours_ahead, analysis_result
        )
        
        if prediction:
            print(f"🎯 예측 가격: ${prediction['predicted_price']:.2f}")
            print(f"📊 신뢰도: {prediction['confidence']:.1f}%")
            print(f"⚠️ 예상 변동폭: ±{prediction['volatility_range']:.1f}%")
            
        return prediction
        
    def analyze_learning_results(self):
        """학습 결과 분석"""
        print("\n📈 학습 결과 분석")
        print("=" * 50)
        
        if not self.learning_history:
            print("📊 분석할 학습 데이터가 없습니다.")
            return
            
        # 정확도 개선 추이
        recent_accuracy = [item['accuracy'] for item in self.learning_history[-20:] 
                          if 'accuracy' in item]
        
        if recent_accuracy:
            improvement = recent_accuracy[-1] - recent_accuracy[0] if len(recent_accuracy) > 1 else 0
            print(f"📊 최근 정확도 개선: {improvement:+.2f}%")
            
        # 공통 실패 패턴
        common_patterns = {}
        for entry in self.learning_history:
            for pattern in entry.get('learned_patterns', []):
                common_patterns[pattern] = common_patterns.get(pattern, 0) + 1
                
        if common_patterns:
            print("\n🔍 발견된 공통 패턴:")
            for pattern, count in sorted(common_patterns.items(), 
                                       key=lambda x: x[1], reverse=True)[:5]:
                print(f"   • {pattern} ({count}회)")
                
    def _save_learning_results(self, results: Dict):
        """학습 결과 저장"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"btc_insight_learning_{timestamp}.json"
        
        logs_dir = Path(__file__).parent.parent / "logs"
        logs_dir.mkdir(exist_ok=True)
        
        filepath = logs_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
            
        print(f"💾 학습 결과 저장: {filename}")
        self.logger.info(f"학습 결과 저장: {filepath}")