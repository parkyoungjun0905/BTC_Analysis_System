#!/usr/bin/env python3
"""
🚀 실제 가격 기반 시간 여행 백테스트 시스템

핵심:
1. 실시간 BTC 가격 API 활용
2. 과거 특정 시점으로 시간 여행
3. 당시 지표로 미래 예측
4. 온라인 실제 가격과 비교
5. 90% 정확도 달성 목표
"""

import os
import json
import numpy as np
import pandas as pd
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import logging
import time
from dataclasses import dataclass

@dataclass
class PredictionTest:
    """예측 테스트 결과"""
    test_time: str
    prediction_time: str
    target_time: str
    current_price: float
    predicted_price: float
    actual_price: float
    price_error_pct: float
    direction_correct: bool
    confidence: float

class RealPriceBTCAPI:
    """실시간 BTC 가격 API"""
    
    def __init__(self):
        self.setup_logging()
        
    def setup_logging(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def get_current_btc_price(self) -> float:
        """현재 BTC 가격 조회 (여러 API 백업)"""
        apis = [
            {
                'name': 'CoinGecko',
                'url': 'https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd',
                'parser': lambda x: x['bitcoin']['usd']
            },
            {
                'name': 'CoinCap',
                'url': 'https://api.coincap.io/v2/assets/bitcoin',
                'parser': lambda x: float(x['data']['priceUsd'])
            },
            {
                'name': 'Binance',
                'url': 'https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT',
                'parser': lambda x: float(x['price'])
            }
        ]
        
        for api in apis:
            try:
                response = requests.get(api['url'], timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    price = api['parser'](data)
                    
                    if 30000 <= price <= 200000:  # 합리적 범위
                        self.logger.info(f"✅ {api['name']} BTC 가격: ${price:,.2f}")
                        return price
                        
            except Exception as e:
                self.logger.warning(f"⚠️ {api['name']} API 실패: {e}")
                continue
        
        # 모든 API 실패시 기본값
        self.logger.error("❌ 모든 BTC 가격 API 실패. 기본값 사용")
        return 65000.0
    
    def get_btc_historical_price(self, days_ago: int) -> float:
        """과거 특정 일의 BTC 가격 조회"""
        try:
            # CoinGecko 과거 가격 API
            target_date = (datetime.now() - timedelta(days=days_ago)).strftime('%d-%m-%Y')
            url = f'https://api.coingecko.com/api/v3/coins/bitcoin/history?date={target_date}'
            
            response = requests.get(url, timeout=15)
            if response.status_code == 200:
                data = response.json()
                price = data['market_data']['current_price']['usd']
                
                self.logger.info(f"📊 {days_ago}일 전 BTC 가격: ${price:,.2f}")
                return float(price)
                
        except Exception as e:
            self.logger.error(f"❌ 과거 가격 조회 실패 ({days_ago}일 전): {e}")
            
        # 실패시 현재 가격 기준 추정
        current = self.get_current_btc_price()
        # 간단한 랜덤워크 추정 (±20% 변동)
        variation = np.random.normal(0, 0.1)  # 10% 표준편차
        estimated = current * (1 + variation)
        
        self.logger.warning(f"⚠️ 추정 과거 가격 사용: ${estimated:,.2f}")
        return estimated

class EnhancedTimeTravel:
    """실제 가격 기반 시간 여행 시스템"""
    
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.price_api = RealPriceBTCAPI()
        self.setup_logging()
        self.load_data()
        
    def setup_logging(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def load_data(self):
        """지표 데이터 로드"""
        try:
            with open(self.data_path, 'r') as f:
                data = json.load(f)
            
            self.timeseries_data = data.get('timeseries_complete', {})
            self.logger.info(f"✅ 지표 데이터 로드: {len(self.timeseries_data)} 카테고리")
            
            # 실제 사용 가능한 지표명들 로깅
            if 'critical_features' in self.timeseries_data:
                indicators = list(self.timeseries_data['critical_features'].keys())[:10]
                self.logger.info(f"📊 주요 지표들: {indicators}")
            
        except Exception as e:
            self.logger.error(f"❌ 데이터 로드 실패: {e}")
            self.timeseries_data = {}
    
    def get_available_indicators(self) -> List[str]:
        """사용 가능한 실제 지표명 리스트"""
        indicators = []
        
        if 'critical_features' in self.timeseries_data:
            indicators.extend(list(self.timeseries_data['critical_features'].keys()))
            
        if 'important_features' in self.timeseries_data:
            indicators.extend(list(self.timeseries_data['important_features'].keys()))
            
        return indicators
    
    def find_price_related_indicators(self) -> List[str]:
        """가격 관련 지표들 찾기"""
        all_indicators = self.get_available_indicators()
        
        price_keywords = ['price', 'btc', 'usd', 'close', 'open', 'high', 'low', 'market']
        price_indicators = []
        
        for indicator in all_indicators:
            indicator_lower = indicator.lower()
            for keyword in price_keywords:
                if keyword in indicator_lower:
                    price_indicators.append(indicator)
                    break
        
        self.logger.info(f"💰 가격 관련 지표: {price_indicators[:5]}")
        return price_indicators
    
    def find_momentum_indicators(self) -> List[str]:
        """모멘텀 관련 지표들 찾기"""
        all_indicators = self.get_available_indicators()
        
        momentum_keywords = ['rsi', 'macd', 'momentum', 'trend', 'velocity', 'acceleration']
        momentum_indicators = []
        
        for indicator in all_indicators:
            indicator_lower = indicator.lower()
            for keyword in momentum_keywords:
                if keyword in indicator_lower:
                    momentum_indicators.append(indicator)
                    break
        
        self.logger.info(f"📈 모멘텀 지표: {momentum_indicators[:5]}")
        return momentum_indicators
    
    def find_volume_indicators(self) -> List[str]:
        """볼륨 관련 지표들 찾기"""
        all_indicators = self.get_available_indicators()
        
        volume_keywords = ['volume', 'flow', 'transaction', 'exchange', 'netflow']
        volume_indicators = []
        
        for indicator in all_indicators:
            indicator_lower = indicator.lower()
            for keyword in volume_keywords:
                if keyword in indicator_lower:
                    volume_indicators.append(indicator)
                    break
        
        self.logger.info(f"📊 볼륨 지표: {volume_indicators[:5]}")
        return volume_indicators
    
    def time_travel_to_hour(self, target_hour: int) -> Dict[str, Any]:
        """특정 시간으로 여행하여 당시 지표 상태 가져오기"""
        try:
            # 실제 가격 (온라인 API 기준)
            days_ago = target_hour // 24  # 시간을 일로 변환
            real_btc_price = self.price_api.get_btc_historical_price(days_ago)
            
            historical_snapshot = {
                'timepoint': target_hour,
                'real_btc_price': real_btc_price,
                'indicators': {},
                'metadata': {
                    'travel_time': datetime.now().isoformat(),
                    'days_ago': days_ago
                }
            }
            
            # 사용 가능한 지표들에서 값 추출
            categories = ['critical_features', 'important_features'] 
            
            for category in categories:
                if category not in self.timeseries_data:
                    continue
                    
                for indicator_name, indicator_data in self.timeseries_data[category].items():
                    values = indicator_data.get('values', [])
                    
                    if target_hour < len(values):
                        current_value = values[target_hour]
                        
                        # 유효한 숫자값인지 확인
                        if isinstance(current_value, (int, float)) and not np.isnan(current_value):
                            historical_snapshot['indicators'][indicator_name] = {
                                'current_value': current_value,
                                'category': category,
                                'trend': self.calculate_trend(values[:target_hour + 1])
                            }
            
            indicator_count = len(historical_snapshot['indicators'])
            self.logger.info(f"🕐 시간 여행 완료: 시점 {target_hour} ({indicator_count}개 지표, 실제가격: ${real_btc_price:,.0f})")
            
            return historical_snapshot
            
        except Exception as e:
            self.logger.error(f"❌ 시간 여행 실패 (시점 {target_hour}): {e}")
            return {}
    
    def calculate_trend(self, values: List[float]) -> float:
        """트렌드 계산 (최근 24시간 기준)"""
        if len(values) < 2:
            return 0.0
        
        try:
            # 최근 24시간 또는 사용 가능한 모든 값
            recent_values = values[-min(24, len(values)):]
            
            if len(recent_values) < 2:
                return 0.0
            
            # 선형 회귀로 트렌드 계산
            x = np.arange(len(recent_values))
            y = np.array(recent_values)
            
            # NaN 제거
            mask = ~np.isnan(y)
            if np.sum(mask) < 2:
                return 0.0
                
            x_clean = x[mask]
            y_clean = y[mask]
            
            # 기울기 계산
            slope = np.polyfit(x_clean, y_clean, 1)[0]
            
            # 정규화 (-1 ~ 1)
            avg_value = np.mean(y_clean)
            if avg_value != 0:
                normalized_slope = slope / avg_value * 100  # 백분율 변화
                return max(-1, min(1, normalized_slope))
            
            return 0.0
            
        except Exception:
            return 0.0

class SmartPatternMatcher:
    """지표 패턴 매칭 및 예측 엔진"""
    
    def __init__(self, time_travel: EnhancedTimeTravel):
        self.time_travel = time_travel
        self.logger = time_travel.logger
        
    def create_adaptive_patterns(self) -> List[Dict]:
        """실제 사용 가능한 지표들로 적응형 패턴 생성"""
        
        # 실제 지표들 분류
        all_indicators = self.time_travel.get_available_indicators()
        price_indicators = self.time_travel.find_price_related_indicators()
        momentum_indicators = self.time_travel.find_momentum_indicators()
        volume_indicators = self.time_travel.find_volume_indicators()
        
        patterns = []
        
        # 패턴 1: 가격 기반
        if price_indicators:
            patterns.append({
                'name': 'price_momentum',
                'indicators': price_indicators[:5],  # 상위 5개
                'weights': [0.3, 0.25, 0.2, 0.15, 0.1],
                'logic': 'price_trend_following'
            })
        
        # 패턴 2: 모멘텀 기반
        if momentum_indicators:
            patterns.append({
                'name': 'technical_momentum',
                'indicators': momentum_indicators[:4],
                'weights': [0.4, 0.3, 0.2, 0.1],
                'logic': 'momentum_reversal'
            })
        
        # 패턴 3: 볼륨 기반
        if volume_indicators:
            patterns.append({
                'name': 'volume_analysis',
                'indicators': volume_indicators[:4],
                'weights': [0.4, 0.3, 0.2, 0.1],
                'logic': 'volume_confirmation'
            })
        
        # 패턴 4: 혼합 패턴
        mixed_indicators = []
        if price_indicators: mixed_indicators.extend(price_indicators[:2])
        if momentum_indicators: mixed_indicators.extend(momentum_indicators[:2])  
        if volume_indicators: mixed_indicators.extend(volume_indicators[:2])
        
        if mixed_indicators:
            patterns.append({
                'name': 'balanced_mixed',
                'indicators': mixed_indicators,
                'weights': [1/len(mixed_indicators)] * len(mixed_indicators),
                'logic': 'ensemble_voting'
            })
        
        # 패턴 5: 상위 지표들
        top_indicators = all_indicators[:6]
        if top_indicators:
            patterns.append({
                'name': 'top_indicators',
                'indicators': top_indicators,
                'weights': [0.25, 0.2, 0.2, 0.15, 0.1, 0.1],
                'logic': 'weighted_ensemble'
            })
        
        self.logger.info(f"🧩 적응형 패턴 생성: {len(patterns)}개")
        return patterns
    
    def test_pattern_accuracy(self, pattern: Dict, num_tests: int = 50) -> float:
        """패턴 정확도 테스트"""
        
        pattern_name = pattern['name']
        indicators = pattern['indicators']
        weights = pattern['weights']
        logic = pattern['logic']
        
        self.logger.info(f"🧪 패턴 테스트: {pattern_name} ({len(indicators)}개 지표)")
        
        correct_predictions = 0
        total_tests = 0
        test_results = []
        
        # 테스트 시점들 (충분한 데이터가 있는 구간)
        test_hours = list(range(168, 1800, 30))  # 7일 후부터 30시간 간격
        selected_hours = test_hours[:num_tests]
        
        for i, test_hour in enumerate(selected_hours):
            try:
                # 1단계: 과거 시점으로 시간 여행
                historical_data = self.time_travel.time_travel_to_hour(test_hour)
                if not historical_data or len(historical_data['indicators']) < 3:
                    continue
                
                # 2단계: 72시간 후 예측
                target_hour = test_hour + 72
                prediction = self.make_prediction(historical_data, indicators, weights, logic)
                
                if not prediction:
                    continue
                
                # 3단계: 실제 미래 가격 확인 (온라인 API)
                future_days_ago = target_hour // 24
                actual_future_price = self.time_travel.price_api.get_btc_historical_price(future_days_ago)
                
                current_price = historical_data['real_btc_price']
                predicted_price = prediction['predicted_price']
                
                # 4단계: 정확도 평가
                price_error_pct = abs(predicted_price - actual_future_price) / actual_future_price
                
                # 방향성 평가
                actual_direction = "UP" if actual_future_price > current_price else "DOWN"
                predicted_direction = prediction['direction']
                direction_correct = (actual_direction == predicted_direction)
                
                if direction_correct:
                    correct_predictions += 1
                
                total_tests += 1
                
                # 결과 저장
                test_result = PredictionTest(
                    test_time=datetime.now().isoformat(),
                    prediction_time=f"hour_{test_hour}",
                    target_time=f"hour_{target_hour}",
                    current_price=current_price,
                    predicted_price=predicted_price,
                    actual_price=actual_future_price,
                    price_error_pct=price_error_pct * 100,
                    direction_correct=direction_correct,
                    confidence=prediction['confidence']
                )
                
                test_results.append(test_result)
                
                if (i + 1) % 10 == 0:
                    current_accuracy = correct_predictions / total_tests * 100
                    avg_error = np.mean([r.price_error_pct for r in test_results])
                    self.logger.info(f"📊 진행률: {i+1}/{len(selected_hours)} | 정확도: {current_accuracy:.1f}% | 평균오차: {avg_error:.1f}%")
                
                time.sleep(0.1)  # API 제한 방지
                
            except Exception as e:
                self.logger.warning(f"⚠️ 테스트 {i} 실패: {e}")
                continue
        
        # 최종 결과
        if total_tests > 0:
            accuracy = correct_predictions / total_tests
            avg_price_error = np.mean([r.price_error_pct for r in test_results])
            
            self.logger.info(f"✅ {pattern_name} 완료:")
            self.logger.info(f"   🎯 방향성 정확도: {accuracy:.1%}")
            self.logger.info(f"   💰 평균 가격 오차: {avg_price_error:.1f}%")
            self.logger.info(f"   📊 총 테스트: {total_tests}회")
            
            return accuracy
        else:
            self.logger.warning(f"❌ {pattern_name}: 유효한 테스트 없음")
            return 0.0
    
    def make_prediction(self, historical_data: Dict, indicators: List[str], 
                       weights: List[float], logic: str) -> Dict:
        """지표 기반 예측 수행"""
        try:
            available_indicators = historical_data['indicators']
            signals = []
            confidences = []
            
            # 선택된 지표들에서 신호 추출
            for i, indicator_name in enumerate(indicators):
                if indicator_name in available_indicators:
                    indicator_data = available_indicators[indicator_name]
                    
                    current_value = indicator_data['current_value']
                    trend = indicator_data['trend']
                    weight = weights[i] if i < len(weights) else 1/len(indicators)
                    
                    # 신호 계산
                    signal = self.calculate_indicator_signal(current_value, trend)
                    signals.append(signal * weight)
                    
                    # 신뢰도 계산
                    confidence = min(0.95, abs(trend) + 0.5)
                    confidences.append(confidence)
            
            if not signals:
                return None
            
            # 종합 신호 및 예측
            combined_signal = sum(signals)
            overall_confidence = np.mean(confidences)
            
            current_price = historical_data['real_btc_price']
            
            # 예측 로직 적용
            prediction = self.apply_prediction_logic(current_price, combined_signal, logic, overall_confidence)
            
            return prediction
            
        except Exception as e:
            self.logger.error(f"❌ 예측 실패: {e}")
            return None
    
    def calculate_indicator_signal(self, value: float, trend: float) -> float:
        """개별 지표 신호 계산"""
        try:
            # 기본 신호: 트렌드 기반
            base_signal = trend
            
            # 값 크기 고려 (정규화)
            if abs(value) > 1000:  # 큰 값들 (가격, 볼륨 등)
                value_signal = np.tanh(value / 100000)  # -1~1 범위로 정규화
            else:  # 작은 값들 (비율, 지수 등)
                value_signal = np.tanh(value)
            
            # 조합 신호
            combined = (base_signal * 0.7 + value_signal * 0.3)
            
            return max(-1, min(1, combined))
            
        except Exception:
            return 0.0
    
    def apply_prediction_logic(self, current_price: float, signal: float, 
                             logic: str, confidence: float) -> Dict:
        """예측 로직 적용"""
        
        # 신호 강도에 따른 가격 변화율
        if logic == 'price_trend_following':
            # 트렌드 추종
            price_change_pct = signal * 0.15  # 최대 ±15%
            
        elif logic == 'momentum_reversal':
            # 모멘텀 반전
            price_change_pct = -signal * 0.1  # 반대 방향, 최대 ±10%
            
        elif logic == 'volume_confirmation':
            # 볼륨 확인
            price_change_pct = signal * 0.08  # 보수적, 최대 ±8%
            
        else:  # ensemble_voting, weighted_ensemble
            # 앙상블
            price_change_pct = signal * 0.12  # 균형적, 최대 ±12%
        
        # 예측 가격 계산
        predicted_price = current_price * (1 + price_change_pct)
        
        # 방향 결정
        if price_change_pct > 0.02:  # 2% 이상
            direction = "UP"
        elif price_change_pct < -0.02:  # -2% 이하
            direction = "DOWN"
        else:
            direction = "SIDEWAYS"
        
        return {
            'predicted_price': predicted_price,
            'direction': direction,
            'confidence': confidence,
            'price_change_pct': price_change_pct * 100
        }

def main():
    """실제 가격 기반 90% 정확도 도전"""
    
    print("🚀 실제 가격 기반 BTC 시간 여행 백테스트")
    print("="*50)
    
    # 데이터 경로
    data_path = "/Users/parkyoungjun/Desktop/BTC_Analysis_System/ai_optimized_3month_data/integrated_complete_data.json"
    
    # 시스템 초기화
    time_travel = EnhancedTimeTravel(data_path)
    pattern_matcher = SmartPatternMatcher(time_travel)
    
    # 현재 BTC 가격 확인
    current_price = time_travel.price_api.get_current_btc_price()
    print(f"💰 현재 BTC 가격: ${current_price:,.2f}")
    
    # 적응형 패턴 생성
    patterns = pattern_matcher.create_adaptive_patterns()
    
    if not patterns:
        print("❌ 사용 가능한 패턴이 없습니다")
        return
    
    print(f"🧩 테스트할 패턴: {len(patterns)}개")
    
    # 각 패턴 테스트
    best_accuracy = 0
    best_pattern = None
    results = {}
    
    for i, pattern in enumerate(patterns, 1):
        print(f"\n🧪 패턴 {i}/{len(patterns)}: {pattern['name']}")
        
        accuracy = pattern_matcher.test_pattern_accuracy(pattern, num_tests=30)
        results[pattern['name']] = accuracy
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_pattern = pattern
        
        print(f"{'🏆' if accuracy >= 0.90 else '📊'} {pattern['name']}: {accuracy:.1%}")
        
        if accuracy >= 0.90:
            print("🎉 90% 정확도 달성!")
            break
    
    # 최종 결과
    print(f"\n🎯 최종 결과:")
    print(f"최고 정확도: {best_accuracy:.1%}")
    
    if best_accuracy >= 0.90:
        print("🏆🏆🏆 90% 목표 달성! 🏆🏆🏆")
        print(f"최적 패턴: {best_pattern['name']}")
    else:
        print(f"📊 목표 미달성. 추가 최적화 필요")
    
    # 결과 저장
    final_results = {
        'timestamp': datetime.now().isoformat(),
        'current_btc_price': current_price,
        'best_accuracy': best_accuracy,
        'best_pattern': best_pattern,
        'all_results': results,
        'target_achieved': best_accuracy >= 0.90
    }
    
    with open('/Users/parkyoungjun/Desktop/BTC_Analysis_System/real_price_backtest_results.json', 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print("✅ 결과 저장 완료: real_price_backtest_results.json")

if __name__ == "__main__":
    main()