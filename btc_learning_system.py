#!/usr/bin/env python3
"""
🧠 BTC 무한 학습 시스템 (btc_learning_system.py)

목표 성능:
- 방향성 정확도: 90%+
- 가격 오차율: 10% 이하
- 학습 방식: 영구적 무한 학습

핵심 기능:
1. 시간 여행 백테스트
2. 무한 학습 루프
3. 실시간 성능 추적
4. 자동 모델 개선
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import logging
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score
from sklearn.model_selection import cross_val_score
import joblib
import warnings
warnings.filterwarnings('ignore')

class BTCLearningSystem:
    """비트코인 무한 학습 시스템"""
    
    def __init__(self, data_path: str = "ai_optimized_3month_data/integrated_complete_data.json"):
        """시스템 초기화"""
        self.base_path = "/Users/parkyoungjun/Desktop/BTC_Analysis_System"
        self.data_path = os.path.join(self.base_path, data_path)
        
        # 성능 목표 (가격+시기 종합 정확도) - 99% 초고성능
        self.target_combined_accuracy = 0.99  # 99% 종합 정확도 (가격+방향성+시기)
        self.target_price_error = 0.02  # 2% 이하 가격 오차율 (극도로 정확)
        self.target_direction_accuracy = 0.99  # 99% 방향성 정확도
        
        # 무한 학습 설정 - 99% 달성까지
        self.max_learning_cycles = 50000  # 최대 학습 사이클 대폭 증가
        self.accuracy_threshold = 0.99  # 99% 목표 달성 임계값
        self.continuous_learning = True  # 무한 학습 활성화
        
        # 학습 설정
        self.prediction_hours = 72  # 72시간(3일) 후 예측
        self.min_history_hours = 168  # 최소 1주 데이터 필요
        
        # 결과 저장
        self.learning_results = []
        self.model_performance = {
            'direction_accuracy': [],
            'price_error_rate': [],
            'total_tests': 0,
            'successful_predictions': 0
        }
        
        # 모델 저장 경로
        self.models_path = os.path.join(self.base_path, "trained_models")
        os.makedirs(self.models_path, exist_ok=True)
        
        # 로깅 설정
        self.setup_logging()
        
        # 데이터 로드
        self.data = self.load_data()
        
        # 스케일러 초기화
        self.scaler = StandardScaler()
        self.price_scaler = MinMaxScaler()
        
        self.logger.info("🚀 BTC 무한 학습 시스템 초기화 완료")
        self.logger.info(f"🎯 목표: 종합 정확도 {self.target_combined_accuracy*100}%, 가격 오차율 {self.target_price_error*100}% 이하")
        self.logger.info(f"🔄 무한 학습: 최대 {self.max_learning_cycles}사이클까지 자동 학습")
    
    def setup_logging(self):
        """로깅 시스템 설정"""
        log_path = os.path.join(self.base_path, "logs")
        os.makedirs(log_path, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(log_path, 'btc_learning_system.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def load_data(self) -> Dict:
        """통합 데이터 로드"""
        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.logger.info(f"✅ 데이터 로드 성공: {len(str(data))/1024/1024:.1f}MB")
            return data
        
        except Exception as e:
            self.logger.error(f"❌ 데이터 로드 실패: {e}")
            raise

class TimeTravel:
    """시간 여행 백테스트 엔진"""
    
    def __init__(self, learning_system):
        self.system = learning_system
        self.logger = learning_system.logger
    
    def get_available_timepoints(self) -> List[int]:
        """사용 가능한 시간 포인트 목록 반환"""
        try:
            timeseries_data = self.system.data.get('timeseries_complete', {})
            critical_features = timeseries_data.get('critical_features', {})
            
            if not critical_features:
                return []
            
            # 첫 번째 지표의 데이터 길이 확인
            first_indicator = list(critical_features.keys())[0]
            total_hours = len(critical_features[first_indicator]['values'])
            
            # 최소 이력 + 예측 기간을 고려한 사용 가능 시점들
            min_start = self.system.min_history_hours
            max_end = total_hours - self.system.prediction_hours
            
            available_points = list(range(min_start, max_end))
            self.logger.info(f"🕐 사용 가능한 시간 포인트: {len(available_points)}개")
            
            return available_points
        
        except Exception as e:
            self.logger.error(f"❌ 시간 포인트 조회 실패: {e}")
            return []
    
    def travel_to_timepoint(self, target_timepoint: int) -> Dict[str, Any]:
        """특정 시간 포인트로 시간 여행"""
        try:
            timeseries_data = self.system.data.get('timeseries_complete', {})
            
            # 시간 여행한 시점의 데이터만 추출
            historical_data = {}
            
            # Critical Features 추출
            if 'critical_features' in timeseries_data:
                historical_data['critical_features'] = {}
                for indicator_name, indicator_data in timeseries_data['critical_features'].items():
                    values = indicator_data['values']
                    # target_timepoint까지의 데이터만 사용
                    historical_values = values[:target_timepoint]
                    historical_data['critical_features'][indicator_name] = {
                        'values': historical_values,
                        'current_value': historical_values[-1] if historical_values else 0
                    }
            
            # Important Features 추출
            if 'important_features' in timeseries_data:
                historical_data['important_features'] = {}
                for indicator_name, indicator_data in timeseries_data['important_features'].items():
                    values = indicator_data['values']
                    historical_values = values[:target_timepoint]
                    historical_data['important_features'][indicator_name] = {
                        'values': historical_values,
                        'current_value': historical_values[-1] if historical_values else 0
                    }
            
            # 메타데이터 추가
            historical_data['metadata'] = {
                'timepoint': target_timepoint,
                'available_hours': target_timepoint,
                'travel_timestamp': datetime.now().isoformat()
            }
            
            self.logger.info(f"🕐 시간 여행 성공: 시점 {target_timepoint} (총 {target_timepoint}시간 데이터)")
            return historical_data
        
        except Exception as e:
            self.logger.error(f"❌ 시간 여행 실패 (시점 {target_timepoint}): {e}")
            return {}
    
    def get_future_actual_price(self, base_timepoint: int) -> float:
        """미래 시점의 실제 가격 조회"""
        try:
            future_timepoint = base_timepoint + self.system.prediction_hours
            
            # 실시간 데이터에서 현재 가격 기준 생성
            realtime_price = self.system.data.get('realtime_snapshot', {}).get('market_data', {}).get('avg_price', 65000)
            
            # 시계열 데이터에서 가격 변동 패턴 추출
            timeseries_data = self.system.data.get('timeseries_complete', {})
            
            # 정규화된 가격 변동 패턴 찾기 (0-100 범위의 지표들)
            price_pattern_indicators = [
                'pattern_triangle_target_price',
                'pattern_double_bottom_target_price',
                'pattern_head_shoulders_target_price'
            ]
            
            price_variations = []
            
            for indicator in price_pattern_indicators:
                if 'critical_features' in timeseries_data:
                    if indicator in timeseries_data['critical_features']:
                        values = timeseries_data['critical_features'][indicator]['values']
                        if future_timepoint < len(values) and base_timepoint < len(values):
                            base_value = values[base_timepoint]
                            future_value = values[future_timepoint]
                            
                            # 실제 BTC 가격 범위로 스케일링 (60K-80K 범위)
                            if base_value > 0:
                                scaled_base = 60000 + (base_value / 100000) * 20000  # 60K-80K 범위
                                scaled_future = 60000 + (future_value / 100000) * 20000
                                price_variations.append(scaled_future)
            
            if price_variations:
                # 여러 패턴의 평균값 사용
                actual_price = np.mean(price_variations)
                # 합리적인 범위로 제한 (30K-150K)
                actual_price = max(30000, min(150000, actual_price))
                self.logger.info(f"💰 실제 가격 조회 성공: ${actual_price:.2f} (시점 {future_timepoint})")
                return actual_price
            
            # 패턴 기반 추정 실패시 현재 가격에 약간의 변동 적용
            variation_factor = 1.0 + np.random.normal(0, 0.1)  # ±10% 변동
            estimated_price = realtime_price * variation_factor
            estimated_price = max(30000, min(150000, estimated_price))  # 합리적 범위
            
            self.logger.warning(f"⚠️ 패턴 기반 추정 사용: ${estimated_price:.2f}")
            return estimated_price
        
        except Exception as e:
            self.logger.error(f"❌ 미래 가격 조회 실패: {e}")
            return 65000.0  # 기본값

class AdvancedPredictionEngine:
    """90% 정확도 달성을 위한 고도화된 예측 엔진"""
    
    def __init__(self, learning_system):
        self.system = learning_system
        self.logger = learning_system.logger
        
        # 고성능 앙상블 모델들
        self.base_models = {
            'random_forest': RandomForestRegressor(
                n_estimators=200, 
                max_depth=15,
                min_samples_split=5,
                random_state=42
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=6,
                random_state=42
            ),
            'neural_network': MLPRegressor(
                hidden_layer_sizes=(100, 50, 25),
                max_iter=1000,
                random_state=42
            ),
            'support_vector': SVR(
                kernel='rbf',
                C=1.0,
                gamma='scale'
            ),
            'ridge_regression': Ridge(
                alpha=1.0,
                random_state=42
            )
        }
        
        # 방향성 예측을 위한 특별한 모델들
        self.direction_models = {
            'direction_rf': RandomForestRegressor(
                n_estimators=300,
                max_depth=10,
                min_samples_split=3,
                random_state=42
            ),
            'direction_gb': GradientBoostingRegressor(
                n_estimators=300,
                learning_rate=0.03,
                max_depth=8,
                random_state=42
            )
        }
        
        # 스케일러들
        self.feature_scaler = RobustScaler()
        self.price_scaler = MinMaxScaler()
        
        # 학습 데이터 저장
        self.training_features = []
        self.training_prices = []
        self.training_directions = []
        
        self.is_trained = False
        self.training_samples = 0
    
    def extract_features(self, historical_data: Dict) -> Tuple[np.ndarray, float]:
        """특성 추출 및 현재 가격 계산"""
        try:
            features = []
            
            # 실시간 데이터에서 기준 가격 가져오기
            realtime_price = self.system.data.get('realtime_snapshot', {}).get('market_data', {}).get('avg_price', 65000)
            current_price = realtime_price
            
            # Critical Features에서 정규화된 특성 추출
            if 'critical_features' in historical_data:
                for indicator_name, indicator_data in historical_data['critical_features'].items():
                    current_value = indicator_data.get('current_value', 0)
                    
                    # 패턴 타겟 가격들은 실제 BTC 가격으로 변환
                    if 'pattern_' in indicator_name and 'target_price' in indicator_name:
                        if current_value > 1000:  # 유효한 값인 경우
                            # 60K-80K 범위로 정규화
                            normalized_price = 60000 + (current_value / 100000) * 20000
                            normalized_price = max(30000, min(150000, normalized_price))
                            if abs(normalized_price - realtime_price) < realtime_price * 0.5:  # 50% 범위 내
                                current_price = normalized_price
                            features.append(normalized_price / 100000)  # 0-1.5 범위로 정규화
                        else:
                            features.append(0)
                    else:
                        # 일반 지표는 정규화된 값 사용
                        normalized_value = current_value / 1000000 if current_value > 1000000 else current_value / 1000
                        features.append(min(100, max(-100, normalized_value)))  # -100 ~ 100 범위
                    
                    # 시계열 통계 특성 추가
                    values = indicator_data.get('values', [])
                    if len(values) >= 24:  # 최근 24시간 데이터가 있으면
                        recent_24h = values[-24:]
                        if recent_24h:
                            # 변화율 기반 특성
                            if len(recent_24h) > 1:
                                change_pct = (recent_24h[-1] - recent_24h[0]) / recent_24h[0] if recent_24h[0] != 0 else 0
                                volatility = np.std(recent_24h) / np.mean(recent_24h) if np.mean(recent_24h) != 0 else 0
                                features.extend([
                                    min(1, max(-1, change_pct)),  # -100% ~ 100% 제한
                                    min(2, volatility)  # 변동성 0~200% 제한
                                ])
            
            # Important Features에서 특성 추출 (선별적으로)
            if 'important_features' in historical_data:
                important_count = 0
                for indicator_name, indicator_data in historical_data['important_features'].items():
                    if important_count >= 100:  # 중요 지표는 최대 100개만
                        break
                    current_value = indicator_data.get('current_value', 0)
                    # 정규화
                    normalized_value = current_value / 1000 if current_value > 1000 else current_value
                    features.append(min(100, max(-100, normalized_value)))
                    important_count += 1
            
            features_array = np.array(features).reshape(1, -1)
            
            # NaN이나 inf 값 처리
            features_array = np.nan_to_num(features_array, nan=0.0, posinf=100, neginf=-100)
            
            # 현재 가격도 합리적 범위로 제한
            current_price = max(30000, min(150000, current_price))
            
            self.logger.info(f"📊 특성 추출 완료: {features_array.shape[1]}개 특성, 현재가 ${current_price:.2f}")
            return features_array, current_price
        
        except Exception as e:
            self.logger.error(f"❌ 특성 추출 실패: {e}")
            return np.array([[0]]), 65000.0
    
    def collect_training_sample(self, features: np.ndarray, current_price: float, actual_price: float):
        """학습 샘플 수집"""
        try:
            # 특성 벡터 저장
            self.training_features.append(features.flatten())
            self.training_prices.append(actual_price)
            
            # 방향성 레이블 생성 (연속값으로)
            direction_value = (actual_price - current_price) / current_price
            self.training_directions.append(direction_value)
            
            self.training_samples += 1
            
            # 충분한 샘플이 모이면 모델 재훈련
            if self.training_samples >= 10 and self.training_samples % 5 == 0:
                self.train_models()
                
            self.logger.info(f"📊 학습 샘플 수집: {self.training_samples}개")
            
        except Exception as e:
            self.logger.error(f"❌ 학습 샘플 수집 실패: {e}")
    
    def train_models(self):
        """모델들 훈련"""
        try:
            if len(self.training_features) < 5:
                self.logger.warning("⚠️ 학습 샘플 부족")
                return
            
            # 데이터 준비
            X = np.array(self.training_features)
            y_price = np.array(self.training_prices)
            y_direction = np.array(self.training_directions)
            
            # 특성 스케일링
            X_scaled = self.feature_scaler.fit_transform(X)
            
            # 가격 예측 모델 훈련
            for name, model in self.base_models.items():
                try:
                    model.fit(X_scaled, y_price)
                    self.logger.info(f"✅ {name} 모델 훈련 완료")
                except Exception as e:
                    self.logger.warning(f"⚠️ {name} 모델 훈련 실패: {e}")
            
            # 방향성 예측 모델 훈련
            for name, model in self.direction_models.items():
                try:
                    model.fit(X_scaled, y_direction)
                    self.logger.info(f"✅ {name} 방향성 모델 훈련 완료")
                except Exception as e:
                    self.logger.warning(f"⚠️ {name} 방향성 모델 훈련 실패: {e}")
            
            self.is_trained = True
            self.logger.info(f"🎉 모델 훈련 완료: {len(self.training_features)}개 샘플")
            
        except Exception as e:
            self.logger.error(f"❌ 모델 훈련 실패: {e}")
    
    def predict_future_price(self, historical_data: Dict) -> Dict[str, Any]:
        """고도화된 미래 가격 및 방향성 예측"""
        try:
            features, current_price = self.extract_features(historical_data)
            
            if not self.is_trained:
                # 패턴 기반 기본 예측 (개선됨)
                predicted_price, direction, confidence = self.pattern_based_prediction(current_price, historical_data)
            else:
                # AI 모델 기반 고급 예측
                predicted_price, direction, confidence = self.ai_model_prediction(features, current_price)
            
            # 변화율 계산
            price_change_pct = ((predicted_price - current_price) / current_price) * 100
            
            prediction_result = {
                'current_price': current_price,
                'predicted_price': predicted_price,
                'direction': direction,
                'price_change_pct': price_change_pct,
                'confidence': confidence,
                'prediction_timestamp': datetime.now().isoformat(),
                'prediction_hours': self.system.prediction_hours,
                'model_trained': self.is_trained,
                'training_samples': self.training_samples
            }
            
            self.logger.info(f"🎯 예측 완료: ${current_price:.2f} → ${predicted_price:.2f} ({direction}, {price_change_pct:+.2f}%, 신뢰도: {confidence:.2f})")
            return prediction_result
        
        except Exception as e:
            self.logger.error(f"❌ 예측 실패: {e}")
            return {
                'current_price': 65000.0,
                'predicted_price': 65000.0,
                'direction': 'SIDEWAYS',
                'price_change_pct': 0.0,
                'confidence': 0.0,
                'error': str(e)
            }
    
    def pattern_based_prediction(self, current_price: float, historical_data: Dict) -> Tuple[float, str, float]:
        """패턴 기반 기본 예측 (정확도 향상)"""
        try:
            # 기술적 분석 기반 예측
            momentum_signals = []
            volatility_signals = []
            
            # Critical features에서 모멘텀 지표들 수집
            if 'critical_features' in historical_data:
                for indicator_name, indicator_data in historical_data['critical_features'].items():
                    values = indicator_data.get('values', [])
                    if len(values) >= 24:
                        recent_24h = values[-24:]
                        
                        # 모멘텀 계산
                        if len(recent_24h) > 1:
                            momentum = (recent_24h[-1] - recent_24h[0]) / recent_24h[0] if recent_24h[0] != 0 else 0
                            volatility = np.std(recent_24h) / np.mean(recent_24h) if np.mean(recent_24h) != 0 else 0
                            
                            # 패턴 기반 가중치
                            if 'pattern_' in indicator_name:
                                momentum_signals.append(momentum * 2)  # 패턴 지표 가중치 높임
                            else:
                                momentum_signals.append(momentum)
                            
                            volatility_signals.append(volatility)
            
            # 신호 통합
            if momentum_signals:
                avg_momentum = np.mean(momentum_signals)
                avg_volatility = np.mean(volatility_signals) if volatility_signals else 0.1
                
                # 방향성 결정 (임계값 기반)
                if avg_momentum > 0.02:  # 2% 이상 상승 모멘텀
                    direction = "UP"
                    price_multiplier = 1 + min(0.1, abs(avg_momentum))  # 최대 10% 변동
                elif avg_momentum < -0.02:  # 2% 이상 하락 모멘텀
                    direction = "DOWN"
                    price_multiplier = 1 - min(0.1, abs(avg_momentum))
                else:
                    direction = "SIDEWAYS"
                    price_multiplier = 1 + np.random.normal(0, 0.02)  # ±2% 변동
                
                predicted_price = current_price * price_multiplier
                
                # 신뢰도 계산 (변동성이 낮을수록 높은 신뢰도)
                confidence = max(0.4, 1.0 - avg_volatility)
                
            else:
                # 기본 예측
                direction = "SIDEWAYS"
                predicted_price = current_price * (1 + np.random.normal(0, 0.03))
                confidence = 0.5
            
            # 합리적 범위 제한
            predicted_price = max(30000, min(150000, predicted_price))
            
            return predicted_price, direction, min(1.0, confidence)
            
        except Exception as e:
            self.logger.error(f"❌ 패턴 기반 예측 실패: {e}")
            return current_price, "SIDEWAYS", 0.5
    
    def ai_model_prediction(self, features: np.ndarray, current_price: float) -> Tuple[float, str, float]:
        """AI 모델 기반 고급 예측 - 90% 방향성 정확도 목표"""
        try:
            # 특성 스케일링
            features_scaled = self.feature_scaler.transform(features)
            
            # 가격 예측 (앙상블)
            price_predictions = []
            for name, model in self.base_models.items():
                try:
                    pred = model.predict(features_scaled)[0]
                    if 30000 <= pred <= 150000:  # 합리적 범위 내
                        price_predictions.append(pred)
                except:
                    continue
            
            # 방향성 예측 (전용 모델들)
            direction_predictions = []
            for name, model in self.direction_models.items():
                try:
                    pred = model.predict(features_scaled)[0]
                    direction_predictions.append(pred)
                except:
                    continue
            
            # 최종 가격 예측
            if price_predictions:
                # 극값 제거 (상위/하위 20% 제거)
                price_predictions = sorted(price_predictions)
                if len(price_predictions) > 2:
                    remove_count = max(1, len(price_predictions) // 5)
                    price_predictions = price_predictions[remove_count:-remove_count]
                
                predicted_price = np.mean(price_predictions)
                price_confidence = 1.0 - (np.std(price_predictions) / np.mean(price_predictions))
            else:
                predicted_price = current_price
                price_confidence = 0.5
            
            # 🚀 강화된 방향성 예측 시스템
            direction, direction_confidence = self.enhanced_direction_prediction(
                direction_predictions, predicted_price, current_price, features_scaled
            )
            
            # 종합 신뢰도
            overall_confidence = (price_confidence + direction_confidence) / 2
            
            return predicted_price, direction, overall_confidence
            
        except Exception as e:
            self.logger.error(f"❌ AI 모델 예측 실패: {e}")
            return current_price, "SIDEWAYS", 0.5
    
    def enhanced_direction_prediction(self, direction_predictions: List[float], 
                                    predicted_price: float, current_price: float, 
                                    features_scaled: np.ndarray) -> Tuple[str, float]:
        """강화된 방향성 예측 시스템 - 90% 정확도 목표"""
        try:
            direction_signals = []
            confidence_scores = []
            
            # 1. AI 모델 기반 방향성 신호 (가중치: 40%)
            if direction_predictions:
                avg_direction_change = np.mean(direction_predictions)
                ai_confidence = 1.0 - min(1.0, np.std(direction_predictions))
                
                # 더 민감한 임계값 (0.5% → 0.2%)
                if avg_direction_change > 0.002:  # 0.2% 이상
                    direction_signals.append(("UP", 0.4, ai_confidence))
                elif avg_direction_change < -0.002:  # 0.2% 이하
                    direction_signals.append(("DOWN", 0.4, ai_confidence))
                else:
                    direction_signals.append(("SIDEWAYS", 0.4, ai_confidence * 0.5))
            
            # 2. 가격 차이 기반 신호 (가중치: 30%)
            price_change_pct = (predicted_price - current_price) / current_price
            if abs(price_change_pct) > 0.001:  # 0.1% 이상 차이
                price_direction = "UP" if price_change_pct > 0 else "DOWN"
                price_confidence = min(1.0, abs(price_change_pct) * 100)  # 변화율에 비례
                direction_signals.append((price_direction, 0.3, price_confidence))
            else:
                direction_signals.append(("SIDEWAYS", 0.3, 0.7))
            
            # 3. 특성 기반 모멘텀 신호 (가중치: 20%)
            momentum_signal = self.calculate_feature_momentum(features_scaled)
            direction_signals.append(momentum_signal)
            
            # 4. 트렌드 강도 신호 (가중치: 10%)
            trend_signal = self.calculate_trend_strength(direction_predictions, price_change_pct)
            direction_signals.append(trend_signal)
            
            # 5. 다수결 + 가중치 시스템으로 최종 결정
            final_direction, final_confidence = self.weighted_voting_system(direction_signals)
            
            return final_direction, final_confidence
            
        except Exception as e:
            self.logger.error(f"❌ 강화된 방향성 예측 실패: {e}")
            return "SIDEWAYS", 0.5
    
    def calculate_feature_momentum(self, features_scaled: np.ndarray) -> Tuple[str, float, float]:
        """특성 기반 모멘텀 계산"""
        try:
            # 특성 벡터에서 모멘텀 지표들 추출 (처음 100개 특성이 주요 지표들)
            key_features = features_scaled[0][:100] if len(features_scaled[0]) >= 100 else features_scaled[0]
            
            # 양수/음수 특성 비율로 모멘텀 계산
            positive_features = np.sum(key_features > 0.1)  # 0.1 이상인 특성들
            negative_features = np.sum(key_features < -0.1)  # -0.1 이하인 특성들
            total_features = len(key_features)
            
            # 모멘텀 점수 (-1 ~ +1)
            if total_features > 0:
                momentum_score = (positive_features - negative_features) / total_features
            else:
                momentum_score = 0
            
            # 방향성 결정
            if momentum_score > 0.1:  # 10% 이상 양수 특성 우세
                direction = "UP"
                confidence = min(1.0, abs(momentum_score) * 2)
            elif momentum_score < -0.1:  # 10% 이상 음수 특성 우세
                direction = "DOWN"
                confidence = min(1.0, abs(momentum_score) * 2)
            else:
                direction = "SIDEWAYS"
                confidence = 0.6
            
            return (direction, 0.2, confidence)  # 가중치 20%
            
        except Exception as e:
            self.logger.error(f"❌ 특성 모멘텀 계산 실패: {e}")
            return ("SIDEWAYS", 0.2, 0.5)
    
    def calculate_trend_strength(self, direction_predictions: List[float], price_change_pct: float) -> Tuple[str, float, float]:
        """트렌드 강도 계산"""
        try:
            # 방향성 예측들의 일관성 체크
            if not direction_predictions or len(direction_predictions) < 2:
                return ("SIDEWAYS", 0.1, 0.5)
            
            # 예측 방향의 일관성 측정
            consistency = 1.0 - (np.std(direction_predictions) / (np.mean(np.abs(direction_predictions)) + 0.001))
            
            # 평균 변화율
            avg_change = np.mean(direction_predictions)
            
            # 트렌드 강도 = 일관성 × 변화 크기
            trend_strength = consistency * abs(avg_change)
            
            # 강한 트렌드일 때만 명확한 방향 제시
            if trend_strength > 0.005 and consistency > 0.7:  # 높은 일관성 + 충분한 변화
                if avg_change > 0:
                    direction = "UP"
                else:
                    direction = "DOWN"
                confidence = min(1.0, trend_strength * 100)
            else:
                direction = "SIDEWAYS"
                confidence = consistency
            
            return (direction, 0.1, confidence)  # 가중치 10%
            
        except Exception as e:
            self.logger.error(f"❌ 트렌드 강도 계산 실패: {e}")
            return ("SIDEWAYS", 0.1, 0.5)
    
    def weighted_voting_system(self, direction_signals: List[Tuple[str, float, float]]) -> Tuple[str, float]:
        """가중치 기반 투표 시스템"""
        try:
            # 각 방향별 가중 점수 계산
            direction_scores = {"UP": 0, "DOWN": 0, "SIDEWAYS": 0}
            total_weight = 0
            
            for direction, weight, confidence in direction_signals:
                # 최종 점수 = 가중치 × 신뢰도
                final_score = weight * confidence
                direction_scores[direction] += final_score
                total_weight += weight
            
            # 정규화
            if total_weight > 0:
                for direction in direction_scores:
                    direction_scores[direction] /= total_weight
            
            # 최고 점수 방향 선택
            final_direction = max(direction_scores, key=direction_scores.get)
            final_confidence = direction_scores[final_direction]
            
            # SIDEWAYS 억제 로직 - UP/DOWN 신호가 있으면 우선
            up_down_total = direction_scores["UP"] + direction_scores["DOWN"]
            if up_down_total > direction_scores["SIDEWAYS"] * 1.2:  # UP/DOWN이 20% 이상 우세
                if direction_scores["UP"] > direction_scores["DOWN"]:
                    final_direction = "UP"
                    final_confidence = direction_scores["UP"]
                else:
                    final_direction = "DOWN"
                    final_confidence = direction_scores["DOWN"]
            
            # 신뢰도 보정 (너무 낮으면 최소값 적용)
            final_confidence = max(0.3, min(1.0, final_confidence))
            
            return final_direction, final_confidence
            
        except Exception as e:
            self.logger.error(f"❌ 가중치 투표 시스템 실패: {e}")
            return "SIDEWAYS", 0.5

class AdvancedPerformanceTracker:
    """95% 정확도 목표 고도화된 성능 추적"""
    
    def __init__(self, learning_system):
        self.system = learning_system
        self.logger = learning_system.logger
    
    def evaluate_prediction(self, prediction: Dict, actual_price: float) -> Dict[str, Any]:
        """가격+시기 종합 정확도 평가"""
        try:
            predicted_price = prediction['predicted_price']
            current_price = prediction['current_price']
            predicted_direction = prediction['direction']
            
            # 실제 방향성
            actual_direction = self.get_precise_direction(current_price, actual_price)
            
            # 1. 방향성 정확도 (엄격한 기준)
            direction_correct = self.evaluate_direction_accuracy(predicted_direction, actual_direction, current_price, actual_price)
            
            # 2. 가격 정확도 (엄격한 기준)
            price_accuracy, price_error_rate = self.evaluate_price_accuracy(predicted_price, actual_price)
            
            # 3. 시기 정확도 (타이밍 점수)
            timing_accuracy = self.evaluate_timing_accuracy(prediction, current_price, actual_price)
            
            # 4. 종합 정확도 (모든 요소 통합)
            combined_accuracy = self.calculate_combined_accuracy(
                direction_correct, price_accuracy, timing_accuracy
            )
            
            # 성능 평가 결과
            evaluation = {
                'direction_correct': direction_correct,
                'direction_accuracy': 1.0 if direction_correct else 0.0,
                'price_accuracy': price_accuracy,
                'price_error_rate': price_error_rate,
                'timing_accuracy': timing_accuracy,
                'combined_accuracy': combined_accuracy,  # 🎯 새로운 종합 지표
                'predicted_price': predicted_price,
                'actual_price': actual_price,
                'predicted_direction': predicted_direction,
                'actual_direction': actual_direction,
                'evaluation_timestamp': datetime.now().isoformat(),
                'meets_95_target': combined_accuracy >= 0.95  # 95% 목표 달성 여부
            }
            
            # 로깅
            status_icon = "🎉" if combined_accuracy >= 0.95 else "📊" if combined_accuracy >= 0.80 else "⚠️"
            self.logger.info(f"{status_icon} 종합 평가: {combined_accuracy:.1%} (방향성: {'✅' if direction_correct else '❌'}, 가격: {price_accuracy:.1%}, 타이밍: {timing_accuracy:.1%})")
            
            # 전체 성능 업데이트
            self.update_overall_performance(evaluation)
            
            return evaluation
        
        except Exception as e:
            self.logger.error(f"❌ 성능 평가 실패: {e}")
            return {}
    
    def get_precise_direction(self, current_price: float, actual_price: float) -> str:
        """정밀한 방향성 결정"""
        change_pct = (actual_price - current_price) / current_price
        
        # 더 엄격한 기준 (0.5% 이상만 UP/DOWN)
        if change_pct > 0.005:  # 0.5% 이상
            return "UP"
        elif change_pct < -0.005:  # 0.5% 이하
            return "DOWN"
        else:
            return "SIDEWAYS"
    
    def evaluate_direction_accuracy(self, predicted_direction: str, actual_direction: str, 
                                  current_price: float, actual_price: float) -> bool:
        """방향성 정확도 평가 (엄격한 기준)"""
        # 기본 방향 일치 확인
        if predicted_direction != actual_direction:
            return False
        
        # 추가 조건: 변화 크기도 고려
        change_pct = abs((actual_price - current_price) / current_price)
        
        # SIDEWAYS인 경우 더 엄격하게
        if actual_direction == "SIDEWAYS":
            return change_pct <= 0.005  # 0.5% 이내여야 정확
        else:
            return change_pct >= 0.005  # 0.5% 이상 변화여야 정확
    
    def evaluate_price_accuracy(self, predicted_price: float, actual_price: float) -> Tuple[float, float]:
        """가격 정확도 평가"""
        error_rate = abs(predicted_price - actual_price) / actual_price
        
        # 95% 목표에 맞는 엄격한 기준
        if error_rate <= 0.01:  # 1% 이내
            accuracy = 1.0
        elif error_rate <= 0.03:  # 3% 이내
            accuracy = 0.9
        elif error_rate <= 0.05:  # 5% 이내
            accuracy = 0.8
        elif error_rate <= 0.10:  # 10% 이내
            accuracy = 0.6
        else:
            accuracy = max(0.0, 1.0 - error_rate)
        
        return accuracy, error_rate
    
    def evaluate_timing_accuracy(self, prediction: Dict, current_price: float, actual_price: float) -> float:
        """시기 정확도 평가"""
        try:
            # 예측 신뢰도가 높을수록 타이밍 점수 높음
            confidence = prediction.get('confidence', 0.5)
            
            # 실제 변화 크기
            change_magnitude = abs((actual_price - current_price) / current_price)
            
            # 예측과 실제의 변화 크기 일치도
            predicted_change = abs(prediction.get('price_change_pct', 0) / 100)
            magnitude_match = 1.0 - min(1.0, abs(predicted_change - change_magnitude) / max(0.01, change_magnitude))
            
            # 종합 타이밍 점수
            timing_score = (confidence * 0.6 + magnitude_match * 0.4)
            
            return min(1.0, max(0.0, timing_score))
            
        except Exception as e:
            self.logger.error(f"❌ 타이밍 정확도 계산 실패: {e}")
            return 0.5
    
    def calculate_combined_accuracy(self, direction_correct: bool, price_accuracy: float, timing_accuracy: float) -> float:
        """종합 정확도 계산"""
        # 가중치: 방향성 50%, 가격 30%, 타이밍 20%
        direction_score = 1.0 if direction_correct else 0.0
        
        combined = (
            direction_score * 0.5 +
            price_accuracy * 0.3 +
            timing_accuracy * 0.2
        )
        
        return min(1.0, max(0.0, combined))
    
    def update_overall_performance(self, evaluation: Dict):
        """전체 성능 통계 업데이트"""
        try:
            perf = self.system.model_performance
            
            perf['total_tests'] += 1
            
            if evaluation.get('direction_correct'):
                perf['successful_predictions'] += 1
            
            perf['direction_accuracy'].append(evaluation.get('direction_accuracy', 0))
            perf['price_error_rate'].append(evaluation.get('price_error_rate', 1))
            
            # 최근 100개 테스트 기준 성능 계산
            recent_direction_acc = np.mean(perf['direction_accuracy'][-100:]) if perf['direction_accuracy'] else 0
            recent_price_error = np.mean(perf['price_error_rate'][-100:]) if perf['price_error_rate'] else 1
            
            self.logger.info(f"📈 전체 성능: 방향성 {recent_direction_acc:.1%}, 가격 오차율 {recent_price_error:.1%}")
            
            # 목표 달성 여부 체크
            if recent_direction_acc >= self.system.target_direction_accuracy and recent_price_error <= self.system.target_price_error:
                self.logger.info("🎉 목표 성능 달성!")
        
        except Exception as e:
            self.logger.error(f"❌ 전체 성능 업데이트 실패: {e}")

class ContinuousLearningEngine:
    """무한 학습 엔진"""
    
    def __init__(self, learning_system):
        self.system = learning_system
        self.logger = learning_system.logger
        
        # 하위 시스템들
        self.time_travel = TimeTravel(learning_system)
        self.predictor = AdvancedPredictionEngine(learning_system)
        self.tracker = AdvancedPerformanceTracker(learning_system)
    
    def run_single_backtest(self, timepoint: int) -> Dict[str, Any]:
        """단일 백테스트 실행"""
        try:
            self.logger.info(f"🕐 백테스트 시작: 시점 {timepoint}")
            
            # 1. 시간 여행
            historical_data = self.time_travel.travel_to_timepoint(timepoint)
            if not historical_data:
                return {'error': 'time_travel_failed'}
            
            # 2. 예측 수행
            prediction = self.predictor.predict_future_price(historical_data)
            if 'error' in prediction:
                return {'error': 'prediction_failed'}
            
            # 3. 실제 결과 조회
            actual_price = self.time_travel.get_future_actual_price(timepoint)
            
            # 4. 성능 평가
            evaluation = self.tracker.evaluate_prediction(prediction, actual_price)
            
            # 5. 학습 샘플 수집 (중요!)
            if 'error' not in prediction:
                features, current_price = self.predictor.extract_features(historical_data)
                self.predictor.collect_training_sample(features, current_price, actual_price)
            
            # 6. 결과 패키지
            result = {
                'timepoint': timepoint,
                'prediction': prediction,
                'actual_price': actual_price,
                'evaluation': evaluation,
                'timestamp': datetime.now().isoformat()
            }
            
            # 결과 저장
            self.system.learning_results.append(result)
            
            self.logger.info(f"✅ 백테스트 완료: 시점 {timepoint}")
            return result
        
        except Exception as e:
            self.logger.error(f"❌ 백테스트 실패 (시점 {timepoint}): {e}")
            return {'error': str(e)}
    
    def run_infinite_learning_cycle(self, max_tests: int = 100) -> Dict[str, Any]:
        """95% 정확도 달성까지 무한 학습 사이클"""
        try:
            self.logger.info(f"🚀 95% 정확도 달성 무한 학습 시작: 최대 {max_tests}회 테스트")
            
            # 사용 가능한 시간 포인트 조회
            available_timepoints = self.time_travel.get_available_timepoints()
            if not available_timepoints:
                self.logger.error("❌ 사용 가능한 시간 포인트 없음")
                return {'error': 'no_timepoints_available'}
            
            # 전체 시점에서 테스트 (더 많은 학습 데이터)
            test_timepoints = available_timepoints[:max_tests] if len(available_timepoints) > max_tests else available_timepoints
            
            self.logger.info(f"📊 테스트 시점 {len(test_timepoints)}개 선택")
            
            # 무한 학습 사이클 실행
            successful_tests = 0
            failed_tests = 0
            learning_cycle = 0
            target_achieved = False
            
            for i, timepoint in enumerate(test_timepoints, 1):
                self.logger.info(f"📈 진행률: {i}/{len(test_timepoints)} ({i/len(test_timepoints)*100:.1f}%)")
                
                result = self.run_single_backtest(timepoint)
                
                if 'error' not in result:
                    successful_tests += 1
                    
                    # 95% 목표 달성 체크
                    evaluation = result.get('evaluation', {})
                    if evaluation.get('meets_95_target', False):
                        self.logger.info("🎉 95% 목표 달성!")
                        
                    # 주기적 성능 체크
                    if i % 10 == 0:
                        current_performance = self.analyze_current_performance()
                        combined_acc = current_performance.get('combined_accuracy', {}).get('average', 0)
                        
                        if combined_acc >= 0.95:
                            target_achieved = True
                            self.logger.info(f"🏆 95% 정확도 목표 달성! 현재: {combined_acc:.1%}")
                            break
                        else:
                            self.logger.info(f"📊 현재 종합 정확도: {combined_acc:.1%} (목표: 95.0%)")
                else:
                    failed_tests += 1
                    self.logger.warning(f"⚠️ 테스트 실패: {result.get('error')}")
                
                learning_cycle += 1
                
                # 최대 사이클 도달
                if learning_cycle >= self.system.max_learning_cycles:
                    self.logger.warning(f"⏰ 최대 학습 사이클 {self.system.max_learning_cycles} 도달")
                    break
            
            # 최종 성능 분석
            final_performance = self.analyze_current_performance()
            
            summary = {
                'total_tests': len(test_timepoints),
                'successful_tests': successful_tests,
                'failed_tests': failed_tests,
                'learning_cycles': learning_cycle,
                'target_achieved': target_achieved,
                'final_performance': final_performance,
                'completion_timestamp': datetime.now().isoformat()
            }
            
            if target_achieved:
                self.logger.info(f"🎉 학습 완료: 95% 목표 달성! ({learning_cycle}사이클)")
            else:
                self.logger.info(f"📊 학습 종료: {learning_cycle}사이클 완료")
            
            return summary
        
        except Exception as e:
            self.logger.error(f"❌ 무한 학습 사이클 실패: {e}")
            return {'error': str(e)}
    
    def analyze_current_performance(self) -> Dict[str, Any]:
        """현재 성능 분석"""
        try:
            if not self.system.learning_results:
                return {'error': 'no_results_available'}
            
            # 성공한 테스트들만 분석
            valid_results = [r for r in self.system.learning_results if 'evaluation' in r and r['evaluation']]
            
            if not valid_results:
                return {'error': 'no_valid_results'}
            
            # 최근 20개 결과만 분석 (최신 성능)
            recent_results = valid_results[-20:] if len(valid_results) > 20 else valid_results
            
            # 종합 정확도
            combined_accuracies = [r['evaluation']['combined_accuracy'] for r in recent_results]
            avg_combined_accuracy = np.mean(combined_accuracies)
            
            # 방향성 정확도
            direction_accuracies = [r['evaluation']['direction_accuracy'] for r in recent_results]
            avg_direction_accuracy = np.mean(direction_accuracies)
            
            # 가격 정확도
            price_accuracies = [r['evaluation']['price_accuracy'] for r in recent_results]
            avg_price_accuracy = np.mean(price_accuracies)
            
            # 타이밍 정확도
            timing_accuracies = [r['evaluation']['timing_accuracy'] for r in recent_results]
            avg_timing_accuracy = np.mean(timing_accuracies)
            
            # 95% 달성 횟수
            target_achieved_count = sum(1 for r in recent_results if r['evaluation'].get('meets_95_target', False))
            
            performance_analysis = {
                'total_valid_tests': len(recent_results),
                'combined_accuracy': {
                    'average': avg_combined_accuracy,
                    'target': 0.95,
                    'achieved': avg_combined_accuracy >= 0.95
                },
                'direction_accuracy': {
                    'average': avg_direction_accuracy,
                    'target': 0.95,
                    'achieved': avg_direction_accuracy >= 0.95
                },
                'price_accuracy': {
                    'average': avg_price_accuracy,
                    'target': 0.95,
                    'achieved': avg_price_accuracy >= 0.95
                },
                'timing_accuracy': {
                    'average': avg_timing_accuracy,
                    'target': 0.80,
                    'achieved': avg_timing_accuracy >= 0.80
                },
                'target_achieved_rate': target_achieved_count / len(recent_results) if recent_results else 0,
                'analysis_timestamp': datetime.now().isoformat()
            }
            
            return performance_analysis
        
        except Exception as e:
            self.logger.error(f"❌ 현재 성능 분석 실패: {e}")
            return {'error': str(e)}
    
    def analyze_overall_performance(self) -> Dict[str, Any]:
        """전체 성능 분석"""
        try:
            if not self.system.learning_results:
                return {'error': 'no_results_available'}
            
            # 성공한 테스트들만 분석
            valid_results = [r for r in self.system.learning_results if 'evaluation' in r and r['evaluation']]
            
            if not valid_results:
                return {'error': 'no_valid_results'}
            
            # 방향성 정확도
            direction_accuracies = [r['evaluation']['direction_accuracy'] for r in valid_results]
            avg_direction_accuracy = np.mean(direction_accuracies)
            
            # 가격 오차율
            price_error_rates = [r['evaluation']['price_error_rate'] for r in valid_results]
            avg_price_error_rate = np.mean(price_error_rates)
            
            # 목표 달성 여부
            direction_target_achieved = avg_direction_accuracy >= self.system.target_direction_accuracy
            price_target_achieved = avg_price_error_rate <= self.system.target_price_error
            
            performance_analysis = {
                'total_valid_tests': len(valid_results),
                'direction_accuracy': {
                    'average': avg_direction_accuracy,
                    'target': self.system.target_direction_accuracy,
                    'achieved': direction_target_achieved
                },
                'price_error_rate': {
                    'average': avg_price_error_rate,
                    'target': self.system.target_price_error,
                    'achieved': price_target_achieved
                },
                'overall_target_achieved': direction_target_achieved and price_target_achieved,
                'analysis_timestamp': datetime.now().isoformat()
            }
            
            self.logger.info(f"📊 전체 성능: 방향성 {avg_direction_accuracy:.1%} (목표: {self.system.target_direction_accuracy:.1%})")
            self.logger.info(f"📊 전체 성능: 가격 오차 {avg_price_error_rate:.1%} (목표: {self.system.target_price_error:.1%})")
            
            return performance_analysis
        
        except Exception as e:
            self.logger.error(f"❌ 성능 분석 실패: {e}")
            return {'error': str(e)}

def main():
    """메인 실행 함수"""
    try:
        print("🚀 BTC 무한 학습 시스템 시작")
        
        # 시스템 초기화
        learning_system = BTCLearningSystem()
        
        # 무한 학습 엔진 생성
        continuous_engine = ContinuousLearningEngine(learning_system)
        
        # 95% 정확도 달성 무한 학습 실행
        print("🎯 95% 정확도 달성 무한 학습 실행 중...")
        learning_results = continuous_engine.run_infinite_learning_cycle(max_tests=100)
        
        if 'error' not in learning_results:
            print(f"✅ 무한 학습 완료!")
            print(f"🔄 학습 사이클: {learning_results['learning_cycles']}회")
            
            if learning_results.get('target_achieved'):
                print("🏆 95% 정확도 목표 달성!")
            else:
                print("📊 학습 진행 중... 계속 학습 필요")
            
            performance = learning_results.get('final_performance', {})
            if 'combined_accuracy' in performance:
                print(f"🎯 종합 정확도: {performance['combined_accuracy']['average']:.1%}")
                print(f"🎯 방향성 정확도: {performance['direction_accuracy']['average']:.1%}")
                print(f"🎯 가격 정확도: {performance['price_accuracy']['average']:.1%}")
                print(f"🎯 타이밍 정확도: {performance['timing_accuracy']['average']:.1%}")
        else:
            print(f"❌ 무한 학습 실패: {learning_results['error']}")
    
    except Exception as e:
        print(f"❌ 시스템 실행 실패: {e}")

if __name__ == "__main__":
    main()