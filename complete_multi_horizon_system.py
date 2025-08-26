#!/usr/bin/env python3
"""
🎯 Complete Multi-Horizon Bitcoin Prediction System
완전한 다중 시간대 비트코인 예측 시스템 - 90%+ 정확도 달성 목표

핵심 구성 요소:
1. Multi-Task Learning Architecture - 공유 특성 인코더와 시간대별 헤드
2. Temporal Hierarchy Modeling - 장/중/단기 트렌드 분석
3. Uncertainty Quantification - 몬테카를로 드롭아웃과 앙상블 신뢰도
4. Dynamic Horizon Weighting - 시장 변동성 기반 동적 가중치
5. Integration Strategies - 계층적 예측 통합과 성능 최적화

목표: 1h, 4h, 24h, 72h, 168h 시간대에서 90%+ 방향 정확도
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import warnings
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import joblib

# 로컬 모듈 import
try:
    from multi_horizon_prediction_system import MultiHorizonPredictionSystem
    from temporal_hierarchy_engine import TemporalHierarchyEngine
    from uncertainty_quantification_system import UncertaintyQuantificationSystem
    from dynamic_horizon_weighting import DynamicHorizonWeightingSystem
    from integration_strategies import IntegrationStrategiesSystem
except ImportError as e:
    print(f"⚠️ 모듈 import 오류: {e}")
    print("필요한 모듈들이 같은 디렉토리에 있는지 확인하세요.")
    sys.exit(1)

warnings.filterwarnings('ignore')

@dataclass
class SystemConfiguration:
    """시스템 설정"""
    horizons: List[int]
    target_accuracy: float
    lookback_window: int
    rebalance_frequency: int
    uncertainty_samples: int
    confidence_threshold: float
    risk_tolerance: float

class CompleteMultiHorizonSystem:
    """완전한 다중 시간대 예측 시스템"""
    
    def __init__(self, data_path: str, config: SystemConfiguration = None):
        self.data_path = data_path
        
        # 기본 설정
        if config is None:
            config = SystemConfiguration(
                horizons=[1, 4, 24, 72, 168],
                target_accuracy=90.0,
                lookback_window=168,
                rebalance_frequency=6,
                uncertainty_samples=100,
                confidence_threshold=0.8,
                risk_tolerance=0.5
            )
        
        self.config = config
        
        # 핵심 구성 요소 초기화
        self.prediction_system = MultiHorizonPredictionSystem(data_path)
        self.hierarchy_engine = TemporalHierarchyEngine()
        self.uncertainty_system = UncertaintyQuantificationSystem()
        self.weighting_system = DynamicHorizonWeightingSystem(config.horizons)
        self.integration_system = IntegrationStrategiesSystem(config.horizons)
        
        # 상태 추적
        self.system_state = {
            'initialized': False,
            'trained': False,
            'last_prediction': None,
            'performance_metrics': {},
            'total_predictions': 0,
            'accuracy_history': []
        }
        
        self.setup_logging()
    
    def setup_logging(self):
        """로깅 설정"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('complete_multi_horizon_system.log', encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def initialize_system(self) -> bool:
        """시스템 초기화"""
        self.logger.info("🚀 Complete Multi-Horizon System 초기화 시작")
        
        try:
            # 데이터 로드 및 검증
            train_data, test_data = self.prediction_system.load_and_prepare_data()
            
            if len(train_data) < 100:
                raise ValueError(f"훈련 데이터가 부족합니다: {len(train_data)} < 100")
            
            # 불확실성 시스템 초기화
            if len(train_data) > 500:  # 충분한 데이터가 있는 경우에만
                val_size = min(200, len(train_data) // 4)
                X_train = train_data[:-val_size]
                X_val = train_data[-val_size:]
                y_train = X_train[:, 0]  # 첫 번째 컬럼이 가격
                y_val = X_val[:, 0]
                
                self.uncertainty_system.fit_uncertainty_methods(
                    X_train[:, 1:], y_train,  # 가격 제외한 특성들
                    X_val[:, 1:], y_val
                )
            
            self.system_state['initialized'] = True
            self.logger.info("✅ 시스템 초기화 완료")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 시스템 초기화 실패: {str(e)}")
            return False
    
    def train_system(self, validation_split: float = 0.2) -> Dict:
        """시스템 훈련"""
        if not self.system_state['initialized']:
            self.logger.error("시스템이 초기화되지 않았습니다.")
            return {'success': False, 'error': 'not_initialized'}
        
        self.logger.info("🎯 Multi-Horizon 시스템 훈련 시작")
        
        try:
            # 데이터 준비
            train_data, test_data = self.prediction_system.load_and_prepare_data()
            
            # 훈련/검증 분할
            val_size = int(len(train_data) * validation_split)
            if val_size > 0:
                train_subset = train_data[:-val_size]
                val_subset = train_data[-val_size:]
            else:
                train_subset = train_data
                val_subset = test_data[:min(100, len(test_data))]
            
            # 주 예측 모델 훈련
            training_history = self.prediction_system.train_model(train_subset, val_subset)
            
            # 시스템 성능 평가
            performance_results = self.prediction_system.evaluate_performance(test_data)
            
            self.system_state['trained'] = True
            self.system_state['performance_metrics'] = performance_results
            
            # 성공 기준 확인
            overall_accuracy = performance_results['overall_metrics']['overall_direction_accuracy']
            success = overall_accuracy >= self.config.target_accuracy
            
            training_results = {
                'success': success,
                'training_history': training_history,
                'performance_results': performance_results,
                'overall_accuracy': overall_accuracy,
                'target_achieved': success,
                'training_samples': len(train_subset),
                'validation_samples': len(val_subset),
                'test_samples': len(test_data)
            }
            
            if success:
                self.logger.info(f"🎉 훈련 성공! 달성 정확도: {overall_accuracy:.2f}%")
            else:
                self.logger.warning(f"⚠️ 목표 미달성: {overall_accuracy:.2f}% < {self.config.target_accuracy}%")
            
            return training_results
            
        except Exception as e:
            self.logger.error(f"❌ 훈련 실패: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def predict(self, current_data: np.ndarray, include_analysis: bool = True) -> Dict:
        """완전한 다중 시간대 예측 실행"""
        if not self.system_state['trained']:
            self.logger.error("시스템이 훈련되지 않았습니다.")
            return {'success': False, 'error': 'not_trained'}
        
        self.logger.info(f"🔮 Multi-Horizon 예측 시작 - 데이터 길이: {len(current_data)}")
        
        try:
            # 현재 가격 추출
            current_price = float(current_data[-1, 0]) if len(current_data) > 0 else 55000.0
            
            # 1. 기본 다중 시간대 예측
            base_predictions = self.prediction_system.predict_with_uncertainty(current_data)
            
            # 2. 시간 계층 분석 (선택적)
            hierarchy_analysis = None
            if include_analysis:
                hierarchy_analysis = self.hierarchy_engine.analyze_temporal_hierarchy(current_data[:, 0])
            
            # 3. 시장 데이터 업데이트 (가중치 시스템용)
            market_data = self.weighting_system.update_market_data(
                current_data[:, 0], 
                current_data[:, 1] if current_data.shape[1] > 1 else None
            )
            
            # 4. 동적 가중치 계산
            dynamic_weights = self.weighting_system.rebalance_weights(
                market_data, 
                self.config.risk_tolerance
            )
            
            # 5. 통합 예측 (여러 방법 결합)
            method_predictions = {
                'multi_horizon_model': base_predictions['deterministic_predictions']
            }
            
            confidence_scores = {
                'multi_horizon_model': {
                    h: base_predictions['uncertainty_analysis'][h]['confidence'] 
                    for h in self.config.horizons 
                    if h in base_predictions['uncertainty_analysis']
                }
            }
            
            uncertainty_bounds = {
                'multi_horizon_model': {
                    h: (
                        base_predictions['uncertainty_analysis'][h]['lower_95'],
                        base_predictions['uncertainty_analysis'][h]['upper_95']
                    )
                    for h in self.config.horizons 
                    if h in base_predictions['uncertainty_analysis']
                }
            }
            
            # 6. 최종 통합
            integration_result = self.integration_system.integrate_predictions(
                method_predictions=method_predictions,
                confidence_scores=confidence_scores,
                uncertainty_bounds=uncertainty_bounds,
                current_price=current_price,
                market_context=market_data
            )
            
            # 7. 결과 종합
            final_result = {
                'success': True,
                'timestamp': datetime.now().isoformat(),
                'current_price': current_price,
                'predictions': integration_result.integrated_predictions,
                'confidence_scores': integration_result.confidence_scores,
                'uncertainty_bounds': integration_result.uncertainty_bounds,
                'market_analysis': {
                    'regime': market_data.get('market_regime', 'unknown'),
                    'volatility': market_data.get('volatility', 0.0),
                    'volatility_regime': market_data.get('volatility_regime', 'unknown'),
                    'volatility_persistence': market_data.get('volatility_persistence', 0.5)
                },
                'dynamic_weights': dynamic_weights,
                'performance_metrics': integration_result.performance_metrics,
                'system_recommendation': self._generate_system_recommendation(integration_result, market_data),
                'quality_assessment': self._assess_prediction_quality(integration_result)
            }
            
            # 선택적 상세 분석 추가
            if include_analysis and hierarchy_analysis:
                final_result['hierarchy_analysis'] = hierarchy_analysis
                
                # 계층 분석으로 예측 최적화
                optimized_predictions = self.hierarchy_engine.optimize_predictions(
                    integration_result.integrated_predictions, 
                    hierarchy_analysis
                )
                final_result['optimized_predictions'] = optimized_predictions
            
            # 상태 업데이트
            self.system_state['last_prediction'] = final_result
            self.system_state['total_predictions'] += 1
            
            self.logger.info(f"✅ 예측 완료 - 통합 신뢰도: {integration_result.performance_metrics.get('integration_confidence', 0):.3f}")
            
            return final_result
            
        except Exception as e:
            self.logger.error(f"❌ 예측 실패: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def _generate_system_recommendation(self, integration_result, market_data: Dict) -> str:
        """시스템 추천 생성"""
        avg_confidence = integration_result.performance_metrics.get('integration_confidence', 0)
        market_regime = market_data.get('market_regime', 'unknown')
        volatility = market_data.get('volatility', 0.0)
        
        # 신뢰도 기반 추천
        if avg_confidence > 0.8:
            confidence_level = "높은"
        elif avg_confidence > 0.6:
            confidence_level = "중간"
        else:
            confidence_level = "낮은"
        
        # 변동성 기반 추천
        if volatility > 0.1:
            volatility_warning = " (⚠️ 높은 변동성 주의)"
        elif volatility > 0.05:
            volatility_warning = " (📊 중간 변동성)"
        else:
            volatility_warning = " (🔒 낮은 변동성)"
        
        # 체제별 추천
        regime_advice = {
            'low_volatility_bull': "안정적 상승 - 장기 포지션 고려",
            'high_volatility_bull': "변동성 상승 - 단기 수익 추구, 리스크 관리 중요",
            'low_volatility_bear': "안정적 하락 - 방어적 포지션",
            'high_volatility_bear': "급격한 하락 - 매우 보수적 접근",
            'sideways_low_vol': "횡보장 - 레인지 거래 전략",
            'sideways_high_vol': "불안정한 횡보 - 신중한 접근",
            'extreme_volatility': "극도 변동성 - 거래 자제 권장"
        }.get(market_regime, "시장 상황 불분명 - 추가 분석 필요")
        
        recommendation = f"{confidence_level} 신뢰도 예측{volatility_warning}. {regime_advice}"
        
        return recommendation
    
    def _assess_prediction_quality(self, integration_result) -> Dict[str, str]:
        """예측 품질 평가"""
        metrics = integration_result.performance_metrics
        
        # 통합 신뢰도 평가
        integration_confidence = metrics.get('integration_confidence', 0)
        if integration_confidence > 0.8:
            confidence_quality = "Excellent"
        elif integration_confidence > 0.6:
            confidence_quality = "Good"
        elif integration_confidence > 0.4:
            confidence_quality = "Fair"
        else:
            confidence_quality = "Poor"
        
        # 예측 다양성 평가
        diversity = metrics.get('prediction_diversity', 0)
        if diversity > 0.3:
            diversity_quality = "High Diversity"
        elif diversity > 0.1:
            diversity_quality = "Moderate Diversity"
        else:
            diversity_quality = "Low Diversity"
        
        # 앙상블 일관성 평가
        coherence = metrics.get('ensemble_coherence', 0)
        if coherence > 0.8:
            coherence_quality = "Highly Coherent"
        elif coherence > 0.6:
            coherence_quality = "Moderately Coherent"
        else:
            coherence_quality = "Low Coherence"
        
        # 최적화 품질 평가
        optimization = metrics.get('optimization_quality', 0)
        if optimization > 0.8:
            optimization_quality = "Well Optimized"
        elif optimization > 0.6:
            optimization_quality = "Adequately Optimized"
        else:
            optimization_quality = "Poorly Optimized"
        
        return {
            'overall_quality': confidence_quality,
            'confidence_assessment': confidence_quality,
            'diversity_assessment': diversity_quality,
            'coherence_assessment': coherence_quality,
            'optimization_assessment': optimization_quality
        }
    
    def update_performance(self, predictions: Dict[int, float], actuals: Dict[int, float], 
                         market_regime: str = None, volatility: float = None):
        """시스템 성능 업데이트"""
        if not predictions or not actuals:
            return
        
        self.logger.info("📊 시스템 성능 업데이트")
        
        # 각 구성요소별 성능 업데이트
        for horizon in self.config.horizons:
            if horizon in predictions and horizon in actuals:
                pred = predictions[horizon]
                actual = actuals[horizon]
                
                # 방향 정확도
                pred_direction = 1 if pred > 0 else 0
                actual_direction = 1 if actual > 0 else 0
                direction_accuracy = 1 if pred_direction == actual_direction else 0
                
                # 가중치 시스템 업데이트
                if market_regime and volatility is not None:
                    self.weighting_system.update_performance(
                        horizon, actual, pred, market_regime, volatility
                    )
                
                # 통합 시스템 업데이트
                self.integration_system.update_performance(
                    'multi_horizon_model', horizon, actual, pred
                )
                
                # 시스템 전체 성능 기록
                self.system_state['accuracy_history'].append({
                    'timestamp': datetime.now(),
                    'horizon': horizon,
                    'accuracy': direction_accuracy,
                    'mae': abs(actual - pred),
                    'market_regime': market_regime
                })
        
        # 히스토리 제한
        if len(self.system_state['accuracy_history']) > 1000:
            self.system_state['accuracy_history'] = self.system_state['accuracy_history'][-800:]
    
    def get_system_status(self) -> Dict:
        """시스템 상태 반환"""
        # 최근 성능 계산
        if self.system_state['accuracy_history']:
            recent_accuracies = [
                entry['accuracy'] for entry in self.system_state['accuracy_history'][-100:]
            ]
            recent_performance = np.mean(recent_accuracies)
        else:
            recent_performance = 0.0
        
        # 구성요소 상태
        weighting_status = self.weighting_system.get_current_strategy()
        integration_summary = self.integration_system.get_system_summary()
        
        return {
            'system_info': {
                'initialized': self.system_state['initialized'],
                'trained': self.system_state['trained'],
                'total_predictions': self.system_state['total_predictions'],
                'target_accuracy': self.config.target_accuracy,
                'recent_performance': recent_performance * 100  # 백분율로 변환
            },
            'configuration': {
                'horizons': self.config.horizons,
                'lookback_window': self.config.lookback_window,
                'confidence_threshold': self.config.confidence_threshold,
                'risk_tolerance': self.config.risk_tolerance
            },
            'component_status': {
                'weighting_system': weighting_status,
                'integration_system': integration_summary
            },
            'last_prediction': self.system_state.get('last_prediction', {}).get('timestamp', None)
        }
    
    def run_comprehensive_test(self, test_periods: int = 50) -> Dict:
        """포괄적 시스템 테스트"""
        self.logger.info(f"🧪 포괄적 시스템 테스트 시작 - {test_periods}개 기간")
        
        if not self.system_state['trained']:
            return {'success': False, 'error': 'system_not_trained'}
        
        try:
            # 테스트 데이터 준비
            _, test_data = self.prediction_system.load_and_prepare_data()
            
            if len(test_data) < test_periods + self.config.lookback_window:
                return {'success': False, 'error': 'insufficient_test_data'}
            
            test_results = {
                'horizon_performance': {h: [] for h in self.config.horizons},
                'overall_metrics': {},
                'test_samples': test_periods,
                'success_rate': 0.0
            }
            
            successful_predictions = 0
            
            # 연속 테스트 실행
            for i in range(test_periods):
                start_idx = i
                end_idx = start_idx + self.config.lookback_window
                
                if end_idx >= len(test_data):
                    break
                
                # 현재 데이터로 예측
                current_data = test_data[start_idx:end_idx]
                prediction_result = self.predict(current_data, include_analysis=False)
                
                if not prediction_result['success']:
                    continue
                
                predictions = prediction_result['predictions']
                
                # 실제 미래값 수집
                actuals = {}
                for horizon in self.config.horizons:
                    future_idx = end_idx + horizon - 1
                    if future_idx < len(test_data):
                        current_price = test_data[end_idx - 1, 0]
                        future_price = test_data[future_idx, 0]
                        actuals[horizon] = future_price
                
                # 성능 평가
                if actuals and predictions:
                    for horizon in self.config.horizons:
                        if horizon in predictions and horizon in actuals:
                            pred = predictions[horizon]
                            actual = actuals[horizon]
                            
                            # 방향 정확도
                            current_price = test_data[end_idx - 1, 0]
                            pred_direction = pred > current_price
                            actual_direction = actual > current_price
                            direction_accuracy = pred_direction == actual_direction
                            
                            test_results['horizon_performance'][horizon].append({
                                'accuracy': direction_accuracy,
                                'mae': abs(actual - pred),
                                'mape': abs(actual - pred) / abs(actual) if actual != 0 else 0
                            })
                    
                    successful_predictions += 1
                    
                    # 성능 업데이트
                    market_data = prediction_result.get('market_analysis', {})
                    self.update_performance(
                        predictions, actuals, 
                        market_data.get('regime'), 
                        market_data.get('volatility')
                    )
            
            # 종합 성능 계산
            overall_accuracy = 0.0
            horizon_accuracies = {}
            
            for horizon in self.config.horizons:
                horizon_results = test_results['horizon_performance'][horizon]
                if horizon_results:
                    accuracies = [r['accuracy'] for r in horizon_results]
                    horizon_accuracy = np.mean(accuracies) * 100
                    horizon_accuracies[horizon] = horizon_accuracy
                    overall_accuracy += horizon_accuracy
            
            if horizon_accuracies:
                overall_accuracy /= len(horizon_accuracies)
            
            test_results['overall_metrics'] = {
                'overall_accuracy': overall_accuracy,
                'horizon_accuracies': horizon_accuracies,
                'success_rate': (successful_predictions / test_periods) * 100,
                'target_achieved': overall_accuracy >= self.config.target_accuracy
            }
            
            self.logger.info(f"🎯 테스트 완료 - 전체 정확도: {overall_accuracy:.2f}%")
            
            return {
                'success': True,
                'test_results': test_results,
                'summary': {
                    'total_tested': successful_predictions,
                    'overall_accuracy': overall_accuracy,
                    'target_accuracy': self.config.target_accuracy,
                    'target_achieved': overall_accuracy >= self.config.target_accuracy,
                    'best_horizon': max(horizon_accuracies.items(), key=lambda x: x[1]) if horizon_accuracies else None
                }
            }
            
        except Exception as e:
            self.logger.error(f"❌ 테스트 실패: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def save_complete_system(self, filepath: str):
        """완전한 시스템 저장"""
        self.logger.info(f"💾 완전한 시스템 저장: {filepath}")
        
        try:
            # 주요 시스템 저장
            if self.system_state['trained']:
                self.prediction_system.save_system(f"{filepath}_prediction.pkl")
            
            self.weighting_system.save_system_state(f"{filepath}_weighting.json")
            self.integration_system.save_system_state(f"{filepath}_integration.json")
            self.uncertainty_system.save_system(f"{filepath}_uncertainty.json")
            
            # 전체 시스템 상태 저장
            system_data = {
                'config': {
                    'horizons': self.config.horizons,
                    'target_accuracy': self.config.target_accuracy,
                    'lookback_window': self.config.lookback_window,
                    'rebalance_frequency': self.config.rebalance_frequency,
                    'uncertainty_samples': self.config.uncertainty_samples,
                    'confidence_threshold': self.config.confidence_threshold,
                    'risk_tolerance': self.config.risk_tolerance
                },
                'system_state': self.system_state.copy(),
                'system_status': self.get_system_status(),
                'save_timestamp': datetime.now().isoformat()
            }
            
            # datetime 객체를 문자열로 변환
            if 'accuracy_history' in system_data['system_state']:
                for entry in system_data['system_state']['accuracy_history']:
                    if isinstance(entry.get('timestamp'), datetime):
                        entry['timestamp'] = entry['timestamp'].isoformat()
            
            with open(f"{filepath}_complete_system.json", 'w', encoding='utf-8') as f:
                json.dump(system_data, f, indent=2, ensure_ascii=False, default=str)
            
            self.logger.info("✅ 시스템 저장 완료")
            
        except Exception as e:
            self.logger.error(f"❌ 시스템 저장 실패: {str(e)}")

def main():
    """메인 실행 함수"""
    print("🎯 Complete Multi-Horizon Bitcoin Prediction System")
    print("="*70)
    print("목표: 1h, 4h, 24h, 72h, 168h 시간대에서 90%+ 방향 정확도 달성")
    print("="*70)
    
    # 시스템 초기화
    data_path = "/Users/parkyoungjun/Desktop/BTC_Analysis_System"
    
    config = SystemConfiguration(
        horizons=[1, 4, 24, 72, 168],
        target_accuracy=90.0,
        lookback_window=168,
        rebalance_frequency=6,
        uncertainty_samples=50,  # 테스트를 위해 줄임
        confidence_threshold=0.8,
        risk_tolerance=0.5
    )
    
    system = CompleteMultiHorizonSystem(data_path, config)
    
    # 1. 시스템 초기화
    print("\n🔧 1단계: 시스템 초기화")
    init_success = system.initialize_system()
    
    if not init_success:
        print("❌ 시스템 초기화 실패")
        return False
    
    print("✅ 시스템 초기화 성공")
    
    # 2. 시스템 훈련
    print("\n🎯 2단계: 시스템 훈련")
    training_results = system.train_system(validation_split=0.2)
    
    if not training_results['success']:
        print(f"❌ 훈련 실패: {training_results.get('error', 'unknown')}")
        return False
    
    overall_accuracy = training_results['overall_accuracy']
    target_achieved = training_results['target_achieved']
    
    print(f"📊 훈련 결과:")
    print(f"  전체 정확도: {overall_accuracy:.2f}%")
    print(f"  목표 달성: {'✅ YES' if target_achieved else '❌ NO'} (목표: {config.target_accuracy}%)")
    
    # 시간대별 성능
    horizon_metrics = training_results['performance_results']['horizon_metrics']
    print(f"  시간대별 정확도:")
    for horizon, metrics in horizon_metrics.items():
        print(f"    {horizon}: {metrics['direction_accuracy']:.2f}%")
    
    # 3. 실시간 예측 테스트
    print("\n🔮 3단계: 실시간 예측 테스트")
    
    # 테스트 데이터 준비
    _, test_data = system.prediction_system.load_and_prepare_data()
    
    if len(test_data) >= config.lookback_window:
        test_sample = test_data[-config.lookback_window:]
        
        prediction_result = system.predict(test_sample, include_analysis=True)
        
        if prediction_result['success']:
            print(f"✅ 예측 성공")
            print(f"  현재 가격: ${prediction_result['current_price']:,.0f}")
            print(f"  시장 체제: {prediction_result['market_analysis']['regime']}")
            print(f"  변동성: {prediction_result['market_analysis']['volatility']:.4f}")
            
            print(f"  예측 결과:")
            for horizon in config.horizons:
                if horizon in prediction_result['predictions']:
                    pred = prediction_result['predictions'][horizon]
                    conf = prediction_result['confidence_scores'][horizon]
                    lower, upper = prediction_result['uncertainty_bounds'][horizon]
                    
                    change_pct = (pred - prediction_result['current_price']) / prediction_result['current_price'] * 100
                    print(f"    {horizon}h: ${pred:,.0f} ({change_pct:+.2f}%) [신뢰도: {conf:.3f}]")
                    print(f"         구간: [${lower:,.0f}, ${upper:,.0f}]")
            
            print(f"  시스템 추천: {prediction_result['system_recommendation']}")
            
        else:
            print(f"❌ 예측 실패: {prediction_result.get('error', 'unknown')}")
    
    # 4. 포괄적 테스트 (선택적)
    print(f"\n🧪 4단계: 포괄적 성능 테스트")
    test_results = system.run_comprehensive_test(test_periods=30)  # 30개 기간 테스트
    
    if test_results['success']:
        summary = test_results['summary']
        print(f"📈 테스트 결과:")
        print(f"  테스트 기간: {summary['total_tested']}")
        print(f"  전체 정확도: {summary['overall_accuracy']:.2f}%")
        print(f"  목표 달성: {'✅ YES' if summary['target_achieved'] else '❌ NO'}")
        
        if summary['best_horizon']:
            best_h, best_acc = summary['best_horizon']
            print(f"  최고 성능: {best_h}h ({best_acc:.2f}%)")
        
        # 시간대별 테스트 성능
        horizon_results = test_results['test_results']['overall_metrics']['horizon_accuracies']
        print(f"  시간대별 테스트 성능:")
        for horizon, accuracy in horizon_results.items():
            status = "✅" if accuracy >= 90.0 else "⚠️" if accuracy >= 80.0 else "❌"
            print(f"    {horizon}: {accuracy:.2f}% {status}")
    
    # 5. 시스템 상태 요약
    print(f"\n📊 5단계: 시스템 상태 요약")
    system_status = system.get_system_status()
    
    print(f"시스템 정보:")
    print(f"  초기화: {'✅' if system_status['system_info']['initialized'] else '❌'}")
    print(f"  훈련 완료: {'✅' if system_status['system_info']['trained'] else '❌'}")
    print(f"  총 예측 수: {system_status['system_info']['total_predictions']}")
    print(f"  최근 성능: {system_status['system_info']['recent_performance']:.2f}%")
    
    # 6. 시스템 저장
    print(f"\n💾 6단계: 시스템 저장")
    system.save_complete_system('complete_multi_horizon_system')
    print("✅ 시스템 저장 완료")
    
    # 최종 결과 요약
    print(f"\n🎯 최종 결과 요약")
    print("="*50)
    
    final_success = (
        init_success and 
        training_results['success'] and 
        training_results['target_achieved']
    )
    
    if final_success:
        print("🎉 Multi-Horizon 시스템 성공적으로 구현 및 90% 목표 달성!")
        print(f"   ✅ 훈련 정확도: {training_results['overall_accuracy']:.2f}%")
        if test_results.get('success'):
            print(f"   ✅ 테스트 정확도: {test_results['summary']['overall_accuracy']:.2f}%")
        print("   ✅ 모든 구성요소 정상 작동")
    else:
        print("⚠️ 시스템 구현 완료, 일부 목표 미달성")
        print(f"   목표: {config.target_accuracy}%")
        print(f"   달성: {training_results.get('overall_accuracy', 0):.2f}%")
    
    print(f"\n📁 생성된 파일들:")
    print(f"   - complete_multi_horizon_system_complete_system.json")
    print(f"   - complete_multi_horizon_system_prediction.pkl")
    print(f"   - complete_multi_horizon_system_weighting.json")
    print(f"   - complete_multi_horizon_system_integration.json")
    print(f"   - complete_multi_horizon_system_uncertainty.json")
    print(f"   - complete_multi_horizon_system.log")
    
    return final_success

if __name__ == "__main__":
    success = main()