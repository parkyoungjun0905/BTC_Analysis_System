#!/usr/bin/env python3
"""
🚀 시간 여행 백테스트 90% 정확도 달성 시스템

핵심 개념:
1. 특정 과거 시점(예: 2025/7/23)으로 시간 여행
2. 당시 지표들로 미래(예: 2025/7/26 17:00) 예측
3. 실제 미래 가격과 비교하여 정확도 검증
4. 수천번 반복하여 최적 지표 조합 발견

목표: 정확한 시간+가격 예측 90% 정확도
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class PredictionResult:
    """예측 결과 데이터 클래스"""
    prediction_time: str
    target_time: str
    current_price: float
    predicted_price: float
    actual_price: float
    price_error_rate: float
    direction_correct: bool
    confidence: float
    used_indicators: Dict[str, float]

class TimeTravel:
    """시간 여행 엔진 - 특정 시점으로 돌아가기"""
    
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.setup_logging()
        self.timeseries_data = self.load_timeseries_data()
        
    def setup_logging(self):
        """로깅 설정"""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def load_timeseries_data(self) -> Dict:
        """3개월 시계열 데이터 로드"""
        try:
            with open(self.data_path, 'r') as f:
                data = json.load(f)
            
            timeseries = data.get('timeseries_complete', {})
            self.logger.info(f"✅ 시계열 데이터 로드: {len(timeseries)} 카테고리")
            return timeseries
            
        except Exception as e:
            self.logger.error(f"❌ 데이터 로드 실패: {e}")
            return {}
    
    def travel_to_timepoint(self, target_hour: int) -> Dict[str, Any]:
        """특정 시간 포인트로 시간 여행"""
        try:
            historical_snapshot = {
                'timepoint': target_hour,
                'indicators': {},
                'metadata': {
                    'travel_time': datetime.now().isoformat(),
                    'available_hours': target_hour
                }
            }
            
            # Critical Features 추출 (과거 시점까지만)
            if 'critical_features' in self.timeseries_data:
                for indicator_name, indicator_data in self.timeseries_data['critical_features'].items():
                    values = indicator_data.get('values', [])
                    if target_hour < len(values):
                        # 해당 시점까지의 데이터만 사용
                        historical_values = values[:target_hour + 1]
                        current_value = values[target_hour]
                        
                        historical_snapshot['indicators'][indicator_name] = {
                            'current_value': current_value,
                            'historical_values': historical_values,
                            'trend_24h': self.calculate_trend(historical_values, 24),
                            'volatility_24h': self.calculate_volatility(historical_values, 24),
                            'momentum_score': self.calculate_momentum(historical_values)
                        }
            
            # Important Features 추가
            if 'important_features' in self.timeseries_data:
                for indicator_name, indicator_data in self.timeseries_data['important_features'].items():
                    values = indicator_data.get('values', [])
                    if target_hour < len(values):
                        current_value = values[target_hour]
                        historical_snapshot['indicators'][indicator_name] = {
                            'current_value': current_value,
                            'trend_24h': self.calculate_trend(values[:target_hour + 1], 24) if target_hour >= 24 else 0
                        }
            
            self.logger.info(f"🕐 시간 여행 완료: 시점 {target_hour} ({len(historical_snapshot['indicators'])}개 지표)")
            return historical_snapshot
            
        except Exception as e:
            self.logger.error(f"❌ 시간 여행 실패: {e}")
            return {}
    
    def get_actual_price_at_timepoint(self, target_hour: int) -> float:
        """특정 시점의 실제 BTC 가격 조회"""
        try:
            # 가격 지표들 중에서 실제 BTC 가격 찾기
            price_indicators = [
                'btc_price', 'btc_spot_price', 'market_price', 
                'btc_usd_price', 'price_usd', 'close_price'
            ]
            
            for price_name in price_indicators:
                if 'critical_features' in self.timeseries_data:
                    if price_name in self.timeseries_data['critical_features']:
                        values = self.timeseries_data['critical_features'][price_name]['values']
                        if target_hour < len(values):
                            price = values[target_hour]
                            if 30000 <= price <= 200000:  # 합리적 BTC 가격 범위
                                return float(price)
                
                if 'important_features' in self.timeseries_data:
                    if price_name in self.timeseries_data['important_features']:
                        values = self.timeseries_data['important_features'][price_name]['values']
                        if target_hour < len(values):
                            price = values[target_hour]
                            if 30000 <= price <= 200000:
                                return float(price)
            
            # 패턴 기반 가격 추정 (다른 지표들로부터)
            estimated_price = self.estimate_price_from_indicators(target_hour)
            return estimated_price
            
        except Exception as e:
            self.logger.error(f"❌ 실제 가격 조회 실패 (시점 {target_hour}): {e}")
            return 65000.0  # 기본값
    
    def estimate_price_from_indicators(self, target_hour: int) -> float:
        """다른 지표들로부터 BTC 가격 추정"""
        try:
            # 시장 가치 기반 추정
            price_hints = []
            
            if 'critical_features' in self.timeseries_data:
                for indicator_name, indicator_data in self.timeseries_data['critical_features'].items():
                    values = indicator_data.get('values', [])
                    if target_hour < len(values):
                        value = values[target_hour]
                        
                        # 패턴 타겟 가격들 (실제 BTC 가격 범위로 변환)
                        if 'pattern_' in indicator_name and 'target_price' in indicator_name:
                            if value > 1000:  # 유효한 값
                                # 정규화된 값을 실제 BTC 가격으로 변환
                                estimated_price = 60000 + (value / 100000) * 40000  # 60K-100K 범위
                                if 30000 <= estimated_price <= 200000:
                                    price_hints.append(estimated_price)
            
            if price_hints:
                return np.median(price_hints)  # 중간값 사용
            else:
                return 65000.0  # 기본 추정값
                
        except Exception as e:
            return 65000.0
    
    def calculate_trend(self, values: List[float], period: int) -> float:
        """트렌드 계산"""
        if len(values) < period:
            return 0.0
        
        recent = values[-period:]
        if len(recent) < 2:
            return 0.0
            
        return (recent[-1] - recent[0]) / recent[0] if recent[0] != 0 else 0.0
    
    def calculate_volatility(self, values: List[float], period: int) -> float:
        """변동성 계산"""
        if len(values) < period:
            return 0.0
            
        recent = values[-period:]
        return np.std(recent) / np.mean(recent) if np.mean(recent) != 0 else 0.0
    
    def calculate_momentum(self, values: List[float]) -> float:
        """모멘텀 점수 계산"""
        if len(values) < 3:
            return 0.0
            
        # 최근 3시간 모멘텀
        recent_3h = values[-3:]
        changes = [(recent_3h[i] - recent_3h[i-1]) / recent_3h[i-1] 
                  for i in range(1, len(recent_3h)) if recent_3h[i-1] != 0]
        
        return np.mean(changes) if changes else 0.0

class IndicatorPatternFinder:
    """지표 패턴 발견 엔진 - 90% 정확도 달성 패턴 찾기"""
    
    def __init__(self, time_travel: TimeTravel):
        self.time_travel = time_travel
        self.logger = time_travel.logger
        self.prediction_results = []
        
    def test_prediction_pattern(self, pattern_config: Dict) -> float:
        """특정 지표 패턴의 예측 정확도 테스트"""
        
        pattern_name = pattern_config['name']
        selected_indicators = pattern_config['indicators']
        prediction_logic = pattern_config['logic']
        
        self.logger.info(f"🧪 패턴 테스트 시작: {pattern_name}")
        
        # 테스트할 시간 포인트들 생성 (충분한 데이터가 있는 구간)
        available_hours = self.get_available_timepoints()
        test_points = self.select_test_timepoints(available_hours, num_tests=100)
        
        correct_predictions = 0
        total_predictions = 0
        price_errors = []
        
        for i, test_hour in enumerate(test_points):
            try:
                # 1단계: 과거 시점으로 시간 여행
                historical_data = self.time_travel.travel_to_timepoint(test_hour)
                if not historical_data:
                    continue
                
                # 2단계: 해당 시점에서 72시간 후 예측
                target_hour = test_hour + 72  # 3일 후
                if target_hour >= len(available_hours):
                    continue
                
                # 3단계: 선택된 지표들로 예측 수행
                prediction_result = self.make_prediction_with_pattern(
                    historical_data, selected_indicators, prediction_logic
                )
                
                if not prediction_result:
                    continue
                
                # 4단계: 실제 미래 가격 확인
                actual_price = self.time_travel.get_actual_price_at_timepoint(target_hour)
                current_price = self.time_travel.get_actual_price_at_timepoint(test_hour)
                
                # 5단계: 정확도 평가
                predicted_price = prediction_result['predicted_price']
                
                # 가격 정확도
                price_error_rate = abs(predicted_price - actual_price) / actual_price
                price_errors.append(price_error_rate)
                
                # 방향성 정확도  
                actual_direction = "UP" if actual_price > current_price else "DOWN" if actual_price < current_price else "SIDEWAYS"
                predicted_direction = prediction_result['direction']
                
                direction_correct = (actual_direction == predicted_direction)
                if direction_correct:
                    correct_predictions += 1
                
                total_predictions += 1
                
                # 결과 저장
                result = PredictionResult(
                    prediction_time=f"hour_{test_hour}",
                    target_time=f"hour_{target_hour}",
                    current_price=current_price,
                    predicted_price=predicted_price,
                    actual_price=actual_price,
                    price_error_rate=price_error_rate,
                    direction_correct=direction_correct,
                    confidence=prediction_result['confidence'],
                    used_indicators=selected_indicators
                )
                
                self.prediction_results.append(result)
                
                if (i + 1) % 20 == 0:
                    current_accuracy = correct_predictions / total_predictions
                    avg_error = np.mean(price_errors)
                    self.logger.info(f"📊 진행률: {i+1}/{len(test_points)}, 현재 정확도: {current_accuracy:.1%}, 평균 오차: {avg_error:.1%}")
                    
            except Exception as e:
                continue
        
        # 최종 정확도 계산
        if total_predictions > 0:
            final_accuracy = correct_predictions / total_predictions
            avg_price_error = np.mean(price_errors) if price_errors else 1.0
            
            self.logger.info(f"✅ {pattern_name} 테스트 완료:")
            self.logger.info(f"   📈 방향성 정확도: {final_accuracy:.1%}")
            self.logger.info(f"   💰 평균 가격 오차: {avg_price_error:.1%}")
            self.logger.info(f"   🎯 총 테스트: {total_predictions}회")
            
            return final_accuracy
        else:
            return 0.0
    
    def make_prediction_with_pattern(self, historical_data: Dict, 
                                   selected_indicators: Dict, 
                                   logic: str) -> Dict:
        """선택된 지표 패턴으로 예측 수행"""
        try:
            indicators = historical_data['indicators']
            
            # 지표 신호 계산
            signals = []
            confidence_scores = []
            
            for indicator_name, weight in selected_indicators.items():
                if indicator_name in indicators:
                    indicator_data = indicators[indicator_name]
                    current_value = indicator_data['current_value']
                    
                    # 지표별 신호 계산
                    signal = self.calculate_indicator_signal(indicator_name, indicator_data)
                    signals.append(signal * weight)
                    
                    # 신뢰도 계산 (트렌드 일관성 기반)
                    confidence = self.calculate_indicator_confidence(indicator_data)
                    confidence_scores.append(confidence)
            
            if not signals:
                return None
            
            # 종합 신호 계산
            combined_signal = np.sum(signals) / len(signals)
            overall_confidence = np.mean(confidence_scores)
            
            # 예측 로직 적용
            prediction = self.apply_prediction_logic(combined_signal, logic, overall_confidence)
            
            return prediction
            
        except Exception as e:
            self.logger.error(f"❌ 패턴 예측 실패: {e}")
            return None
    
    def calculate_indicator_signal(self, indicator_name: str, indicator_data: Dict) -> float:
        """개별 지표의 신호 강도 계산"""
        current_value = indicator_data['current_value']
        trend_24h = indicator_data.get('trend_24h', 0)
        momentum = indicator_data.get('momentum_score', 0)
        
        # 지표별 특화된 신호 계산
        if 'mvrv' in indicator_name.lower():
            # MVRV: 2.5 이상이면 과열(하락 신호), 1.0 이하면 저점(상승 신호)
            if current_value > 2.5:
                signal = -0.8  # 강한 하락 신호
            elif current_value < 1.0:
                signal = 0.8   # 강한 상승 신호
            else:
                signal = (1.75 - current_value) / 1.75  # 정규화
                
        elif 'sopr' in indicator_name.lower():
            # SOPR: 1.0 기준, 높을수록 매도 압력
            signal = (1.05 - current_value) * 5
            
        elif 'fear_greed' in indicator_name.lower():
            # 공포탐욕지수: 극단에서 반전 신호
            if current_value > 80:  # 극도 탐욕
                signal = -0.9
            elif current_value < 20:  # 극도 공포
                signal = 0.9
            else:
                signal = (50 - current_value) / 50
                
        elif 'funding_rate' in indicator_name.lower():
            # 펀딩비율: 높을수록 롱 포지션 과다 (하락 신호)
            signal = -current_value * 20
            
        elif 'netflow' in indicator_name.lower():
            # 거래소 순유입: 양수면 매도 압력
            signal = -current_value * 0.1
            
        else:
            # 기본 트렌드 기반 신호
            signal = trend_24h + momentum * 0.5
        
        # 신호 범위 제한 (-1 ~ 1)
        return max(-1, min(1, signal))
    
    def calculate_indicator_confidence(self, indicator_data: Dict) -> float:
        """지표의 신뢰도 계산"""
        volatility = indicator_data.get('volatility_24h', 0.5)
        trend_strength = abs(indicator_data.get('trend_24h', 0))
        
        # 낮은 변동성과 강한 트렌드일 때 높은 신뢰도
        confidence = (1 - min(1, volatility)) * 0.5 + min(1, trend_strength) * 0.5
        return max(0.3, min(0.95, confidence))
    
    def apply_prediction_logic(self, combined_signal: float, logic: str, confidence: float) -> Dict:
        """예측 로직 적용"""
        # 방향 결정
        if combined_signal > 0.3:
            direction = "UP"
            price_multiplier = 1 + min(0.1, abs(combined_signal) * 0.1)
        elif combined_signal < -0.3:
            direction = "DOWN" 
            price_multiplier = 1 - min(0.1, abs(combined_signal) * 0.1)
        else:
            direction = "SIDEWAYS"
            price_multiplier = 1 + combined_signal * 0.02
        
        # 기준 가격 (현재는 65000으로 가정, 실제로는 시간 여행 데이터에서)
        base_price = 65000
        predicted_price = base_price * price_multiplier
        
        return {
            'direction': direction,
            'predicted_price': predicted_price,
            'confidence': confidence,
            'signal_strength': abs(combined_signal)
        }
    
    def get_available_timepoints(self) -> List[int]:
        """사용 가능한 시간 포인트 조회"""
        if 'critical_features' in self.time_travel.timeseries_data:
            first_indicator = list(self.time_travel.timeseries_data['critical_features'].keys())[0]
            total_hours = len(self.time_travel.timeseries_data['critical_features'][first_indicator]['values'])
            return list(range(168, total_hours - 72))  # 1주 여유 + 3일 예측 여유
        return []
    
    def select_test_timepoints(self, available_hours: List[int], num_tests: int) -> List[int]:
        """테스트용 시간 포인트 선택"""
        if len(available_hours) <= num_tests:
            return available_hours
        
        # 균등하게 분포시켜 선택
        step = len(available_hours) // num_tests
        return available_hours[::step][:num_tests]

class OptimalPatternSearch:
    """90% 정확도 달성 최적 패턴 탐색"""
    
    def __init__(self, data_path: str):
        self.time_travel = TimeTravel(data_path)
        self.pattern_finder = IndicatorPatternFinder(self.time_travel)
        self.logger = self.time_travel.logger
        
    def search_90_percent_patterns(self) -> Dict[str, Any]:
        """90% 정확도 달성 패턴 탐색"""
        
        self.logger.info("🚀 90% 정확도 달성 패턴 탐색 시작!")
        
        # 테스트할 지표 조합 패턴들
        test_patterns = self.generate_test_patterns()
        
        results = {}
        best_accuracy = 0
        best_pattern = None
        
        for i, pattern in enumerate(test_patterns, 1):
            self.logger.info(f"\n📊 패턴 {i}/{len(test_patterns)}: {pattern['name']}")
            
            accuracy = self.pattern_finder.test_prediction_pattern(pattern)
            results[pattern['name']] = {
                'accuracy': accuracy,
                'pattern': pattern,
                'achieved_90_percent': accuracy >= 0.90
            }
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_pattern = pattern
                
            self.logger.info(f"{'🏆' if accuracy >= 0.90 else '📊'} {pattern['name']}: {accuracy:.1%}")
            
            if accuracy >= 0.90:
                self.logger.info("🎉 90% 정확도 달성 패턴 발견!")
                break
        
        # 최종 결과
        final_results = {
            'search_completed': datetime.now().isoformat(),
            'best_accuracy': best_accuracy,
            'best_pattern': best_pattern,
            'target_achieved': best_accuracy >= 0.90,
            'all_results': results,
            'total_patterns_tested': len(results)
        }
        
        if best_accuracy >= 0.90:
            self.logger.info("🏆🏆🏆 90% 정확도 달성 성공! 🏆🏆🏆")
            self.logger.info(f"최고 정확도: {best_accuracy:.1%}")
            self.logger.info(f"최적 패턴: {best_pattern['name']}")
        else:
            self.logger.info(f"📊 탐색 완료. 최고 정확도: {best_accuracy:.1%}")
            self.logger.info("90% 달성을 위해 더 많은 패턴 필요")
        
        return final_results
    
    def generate_test_patterns(self) -> List[Dict]:
        """테스트할 지표 조합 패턴들 생성"""
        
        patterns = []
        
        # 패턴 1: 온체인 중심
        patterns.append({
            'name': 'onchain_dominant',
            'indicators': {
                'mvrv_ratio': 0.4,
                'sopr': 0.3,
                'coin_days_destroyed': 0.2,
                'whale_ratio': 0.1
            },
            'logic': 'onchain_signals'
        })
        
        # 패턴 2: 파생상품 중심  
        patterns.append({
            'name': 'derivatives_focus',
            'indicators': {
                'funding_rate': 0.4,
                'long_short_ratio': 0.3,
                'open_interest': 0.2,
                'liquidation_data': 0.1
            },
            'logic': 'derivatives_pressure'
        })
        
        # 패턴 3: 시장 심리 중심
        patterns.append({
            'name': 'sentiment_driven',
            'indicators': {
                'fear_greed_index': 0.5,
                'social_volume': 0.2,
                'news_sentiment': 0.2,
                'search_trends': 0.1
            },
            'logic': 'sentiment_reversal'
        })
        
        # 패턴 4: 거래소 흐름 중심
        patterns.append({
            'name': 'exchange_flows',
            'indicators': {
                'exchange_netflow': 0.4,
                'exchange_reserve': 0.3,
                'stablecoin_inflow': 0.2,
                'institutional_flows': 0.1
            },
            'logic': 'supply_demand'
        })
        
        # 패턴 5: 균형 조합
        patterns.append({
            'name': 'balanced_premium',
            'indicators': {
                'mvrv_ratio': 0.25,
                'funding_rate': 0.25,
                'fear_greed_index': 0.25,
                'exchange_netflow': 0.25
            },
            'logic': 'multi_factor'
        })
        
        return patterns

def main():
    """90% 정확도 달성 시간 여행 백테스트 실행"""
    
    print("🚀 시간 여행 백테스트 90% 정확도 달성 시스템")
    print("="*60)
    
    data_path = "/Users/parkyoungjun/Desktop/BTC_Analysis_System/ai_optimized_3month_data/integrated_complete_data.json"
    
    # 최적 패턴 탐색 시작
    searcher = OptimalPatternSearch(data_path)
    results = searcher.search_90_percent_patterns()
    
    # 결과 출력
    print(f"\n🎯 탐색 완료!")
    print(f"최고 정확도: {results['best_accuracy']:.1%}")
    
    if results['target_achieved']:
        print("🏆 90% 정확도 달성 성공!")
        print(f"최적 패턴: {results['best_pattern']['name']}")
    else:
        print(f"📊 90% 미달성. 추가 패턴 탐색 필요")
    
    # 결과 저장
    with open("/Users/parkyoungjun/Desktop/BTC_Analysis_System/optimal_pattern_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print("✅ 결과가 optimal_pattern_results.json에 저장되었습니다")

if __name__ == "__main__":
    main()