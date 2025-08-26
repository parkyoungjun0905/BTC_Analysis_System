#!/usr/bin/env python3
"""
🕰️ 시간여행 백테스트 학습 엔진
- 과거 시점으로 돌아가서 예측 → 실제값 비교 → 학습
- 사용자 요구사항: 25년 7월 23일 → 7월 26일 17시 예측 → 검증 → 학습
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error

from ..utils.logger import get_logger

class TimetravelLearningEngine:
    """시간여행 백테스트 학습 엔진"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.learned_patterns = {}
        self.failure_analysis = []
        
        print("🕰️ 시간여행 백테스트 학습 엔진 초기화")
        
    def execute_backtest(self, historical_data: pd.DataFrame, 
                        start_idx: int, prediction_hours: int) -> Dict:
        """
        시간여행 백테스트 실행
        
        사용자 예시 과정:
        1. 25년 7월 23일 시점으로 돌아감 (start_idx)
        2. 해당 시점의 지표들로 7월 26일 17시 예측 (prediction_hours)
        3. 실제 7월 26일 17시 값과 비교
        4. 틀린 원인 분석 및 학습
        
        Args:
            historical_data: 3개월치 1시간 단위 통합 데이터
            start_idx: 시작 시점 인덱스 (예: 25년 7월 23일)
            prediction_hours: 예측 시간 (예: 72시간 = 3일 후)
            
        Returns:
            백테스트 결과 딕셔너리
        """
        print(f"🕰️ 시간여행: {start_idx}번째 시점 → {prediction_hours}시간 후 예측")
        
        try:
            # 1단계: 과거 시점으로 "시간여행"
            historical_point = historical_data.iloc[:start_idx].copy()
            
            if len(historical_point) < 100:
                return {'success': False, 'error': '학습 데이터 부족 (100개 미만)'}
            
            # 2단계: BTC 가격 컬럼 식별
            btc_price_col = self._identify_btc_price_column(historical_data)
            if not btc_price_col:
                return {'success': False, 'error': 'BTC 가격 컬럼을 찾을 수 없음'}
            
            # 3단계: 예측 타겟 시점 계산
            target_idx = start_idx + prediction_hours
            if target_idx >= len(historical_data):
                return {'success': False, 'error': '예측 타겟이 데이터 범위 초과'}
            
            # 4단계: 시계열 특성 피처 생성 (필수 요구사항)
            X_features, y_target = self._prepare_timeseries_features(
                historical_point, btc_price_col, prediction_hours
            )
            
            if len(X_features) < 50:
                return {'success': False, 'error': '시계열 학습 데이터 부족'}
            
            # 5단계: 앙상블 모델 학습 (과거 데이터만 사용)
            prediction_result = self._train_and_predict(
                X_features, y_target, historical_point, btc_price_col
            )
            
            if not prediction_result:
                return {'success': False, 'error': '모델 학습/예측 실패'}
            
            # 6단계: 실제값과 비교 ("7월 26일 17시 실제 BTC 값" 확인)
            actual_future_price = historical_data.iloc[target_idx][btc_price_col]
            current_price = historical_data.iloc[start_idx][btc_price_col]
            predicted_price = prediction_result['prediction']
            
            # 7단계: 예측 오차 계산
            absolute_error = abs(actual_future_price - predicted_price)
            percentage_error = (absolute_error / actual_future_price) * 100
            accuracy = max(0, 100 - percentage_error)
            
            # 8단계: 실패 원인 분석 (틀렸을 때 왜 틀렸는지)
            error_analysis = self._analyze_prediction_failure(
                historical_data, start_idx, target_idx, 
                predicted_price, actual_future_price, btc_price_col
            )
            
            # 9단계: 학습 패턴 업데이트
            self._update_learned_patterns(error_analysis, percentage_error)
            
            result = {
                'success': True,
                'start_idx': start_idx,
                'target_idx': target_idx,
                'prediction_hours': prediction_hours,
                'current_price': current_price,
                'predicted_price': predicted_price,
                'actual_price': actual_future_price,
                'absolute_error': absolute_error,
                'error_percentage': percentage_error,
                'accuracy': accuracy,
                'error_analysis': error_analysis,
                'model_details': prediction_result['model_info']
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"백테스트 실행 오류: {e}")
            return {'success': False, 'error': str(e)}
    
    def _identify_btc_price_column(self, data: pd.DataFrame) -> str:
        """BTC 가격 컬럼 식별"""
        candidates = [
            'onchain_blockchain_info_network_stats_market_price_usd',
            'btc_price', 'price', 'close', 'market_price_usd'
        ]
        
        for candidate in candidates:
            if candidate in data.columns:
                return candidate
        
        # 가격으로 추정되는 컬럼 찾기 (큰 숫자값)
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if data[col].mean() > 1000:  # BTC 가격은 보통 수만 달러
                return col
                
        return numeric_cols[0] if len(numeric_cols) > 0 else None
    
    def _prepare_timeseries_features(self, historical_data: pd.DataFrame, 
                                   btc_col: str, prediction_hours: int) -> Tuple[pd.DataFrame, pd.Series]:
        """
        시계열 분석을 위한 피처 준비 (사용자 요구사항)
        
        Args:
            historical_data: 과거 데이터
            btc_col: BTC 가격 컬럼명
            prediction_hours: 예측 시간
            
        Returns:
            (피처 DataFrame, 타겟 Series)
        """
        # 원본 지표들
        feature_columns = [col for col in historical_data.columns if col != btc_col]
        X_base = historical_data[feature_columns].copy()
        
        # BTC 가격 시계열 피처 추가 (시계열 분석 필수)
        btc_prices = historical_data[btc_col]
        
        # 1. 가격 지연 피처 (Lag Features)
        X_base['btc_lag_1h'] = btc_prices.shift(1)
        X_base['btc_lag_6h'] = btc_prices.shift(6)
        X_base['btc_lag_12h'] = btc_prices.shift(12)
        X_base['btc_lag_24h'] = btc_prices.shift(24)
        
        # 2. 가격 변화율 피처
        X_base['btc_change_1h'] = btc_prices.pct_change(1) * 100
        X_base['btc_change_6h'] = btc_prices.pct_change(6) * 100
        X_base['btc_change_24h'] = btc_prices.pct_change(24) * 100
        
        # 3. 이동평균 피처
        X_base['btc_sma_6h'] = btc_prices.rolling(6).mean()
        X_base['btc_sma_12h'] = btc_prices.rolling(12).mean()
        X_base['btc_sma_24h'] = btc_prices.rolling(24).mean()
        
        # 4. 변동성 피처
        X_base['btc_std_6h'] = btc_prices.rolling(6).std()
        X_base['btc_std_24h'] = btc_prices.rolling(24).std()
        
        # 5. 모멘텀 피처
        X_base['btc_rsi_14'] = self._calculate_rsi(btc_prices, 14)
        X_base['btc_momentum_12h'] = btc_prices - btc_prices.shift(12)
        
        # 타겟: prediction_hours 시간 후 가격
        y_target = btc_prices.shift(-prediction_hours)
        
        # 결측치 제거
        valid_idx = ~(X_base.isnull().any(axis=1) | y_target.isnull())
        X_features = X_base[valid_idx].iloc[:-prediction_hours]  # 미래 데이터 제외
        y_target = y_target[valid_idx].iloc[:-prediction_hours]
        
        return X_features, y_target
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """RSI 계산"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _train_and_predict(self, X_features: pd.DataFrame, y_target: pd.Series,
                          historical_data: pd.DataFrame, btc_col: str) -> Dict:
        """
        모델 학습 및 예측
        
        Args:
            X_features: 시계열 특성 피처
            y_target: 예측 타겟
            historical_data: 원본 데이터
            btc_col: BTC 가격 컬럼
            
        Returns:
            예측 결과 딕셔너리
        """
        try:
            # 데이터 스케일링
            scaler = RobustScaler()
            X_scaled = scaler.fit_transform(X_features)
            
            # 앙상블 모델 구성
            models = {
                'random_forest': RandomForestRegressor(
                    n_estimators=100, 
                    max_depth=10,
                    random_state=42
                ),
                'gradient_boosting': GradientBoostingRegressor(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42
                )
            }
            
            # 모델 학습 및 예측
            predictions = {}
            model_scores = {}
            
            for name, model in models.items():
                # 학습
                model.fit(X_scaled, y_target)
                
                # 현재 시점 예측 (마지막 데이터 포인트)
                current_features = X_features.iloc[-1:].values
                current_scaled = scaler.transform(current_features)
                pred = model.predict(current_scaled)[0]
                
                predictions[name] = pred
                
                # 검증 점수 (마지막 20% 데이터로)
                split_idx = int(len(X_scaled) * 0.8)
                val_pred = model.predict(X_scaled[split_idx:])
                val_actual = y_target.iloc[split_idx:]
                score = mean_absolute_error(val_actual, val_pred)
                model_scores[name] = score
            
            # 가중 앙상블 예측
            weights = {}
            total_score = sum(model_scores.values())
            for name, score in model_scores.items():
                weights[name] = (total_score - score) / total_score  # 낮은 에러일수록 높은 가중치
            
            # 정규화
            total_weight = sum(weights.values())
            weights = {k: v/total_weight for k, v in weights.items()}
            
            # 최종 예측값
            final_prediction = sum(predictions[name] * weights[name] 
                                 for name in predictions.keys())
            
            return {
                'prediction': final_prediction,
                'model_info': {
                    'individual_predictions': predictions,
                    'model_scores': model_scores,
                    'ensemble_weights': weights
                }
            }
            
        except Exception as e:
            self.logger.error(f"모델 학습/예측 오류: {e}")
            return None
    
    def _analyze_prediction_failure(self, historical_data: pd.DataFrame,
                                  start_idx: int, target_idx: int,
                                  predicted: float, actual: float, 
                                  btc_col: str) -> Dict:
        """
        예측 실패 원인 분석 (사용자 요구사항: 틀린 원인 찾고 학습)
        
        Args:
            historical_data: 전체 데이터
            start_idx: 시작 인덱스
            target_idx: 타겟 인덱스  
            predicted: 예측값
            actual: 실제값
            btc_col: BTC 가격 컬럼
            
        Returns:
            오류 분석 결과
        """
        # 예측 기간 데이터 추출
        period_data = historical_data.iloc[start_idx:target_idx+1].copy()
        btc_prices = period_data[btc_col]
        
        analysis = {
            'prediction_error': abs(actual - predicted),
            'error_direction': 'overestimate' if predicted > actual else 'underestimate',
            'price_volatility': btc_prices.std(),
            'max_price_swing': btc_prices.max() - btc_prices.min(),
            'shock_events': [],
            'indicator_changes': [],
            'market_regime': self._detect_market_regime(btc_prices),
            'high_volatility': btc_prices.std() > btc_prices.mean() * 0.05  # 5% 이상 변동성
        }
        
        # 돌발변수 감지 (급격한 가격 변화)
        price_changes = btc_prices.pct_change().abs()
        shock_threshold = 0.05  # 5% 이상 1시간 변화
        
        shock_points = price_changes[price_changes > shock_threshold]
        for idx in shock_points.index:
            if idx > 0:
                analysis['shock_events'].append({
                    'timestamp_idx': idx,
                    'price_change_pct': price_changes.loc[idx] * 100,
                    'price_before': btc_prices.iloc[idx-1] if idx > 0 else None,
                    'price_after': btc_prices.iloc[idx]
                })
        
        # 주요 지표 변화량 분석
        numeric_cols = period_data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col != btc_col and not period_data[col].isnull().all():
                start_val = period_data[col].iloc[0]
                end_val = period_data[col].iloc[-1]
                if start_val != 0:
                    change_pct = abs((end_val - start_val) / start_val) * 100
                    if change_pct > 10:  # 10% 이상 변화
                        analysis['indicator_changes'].append((col, change_pct))
        
        # 변화량 순으로 정렬
        analysis['indicator_changes'].sort(key=lambda x: x[1], reverse=True)
        
        return analysis
    
    def _detect_market_regime(self, btc_prices: pd.Series) -> str:
        """시장 상황 감지"""
        if len(btc_prices) < 24:
            return 'insufficient_data'
            
        # 24시간 트렌드 분석
        recent_trend = (btc_prices.iloc[-1] - btc_prices.iloc[-24]) / btc_prices.iloc[-24] * 100
        
        # 변동성 분석
        volatility = btc_prices.pct_change().std() * 100
        
        if recent_trend > 5:
            return 'bull_market'
        elif recent_trend < -5:
            return 'bear_market'
        elif volatility > 3:
            return 'high_volatility'
        else:
            return 'sideways'
    
    def _update_learned_patterns(self, error_analysis: Dict, error_percentage: float):
        """
        학습 패턴 업데이트 (실패에서 학습)
        
        Args:
            error_analysis: 오류 분석 결과
            error_percentage: 오류 퍼센트
        """
        # 실패 케이스 저장
        self.failure_analysis.append({
            'timestamp': datetime.now(),
            'error_percentage': error_percentage,
            'analysis': error_analysis
        })
        
        # 패턴 학습
        market_regime = error_analysis.get('market_regime', 'unknown')
        shock_count = len(error_analysis.get('shock_events', []))
        
        pattern_key = f"{market_regime}_{shock_count}_shocks"
        
        if pattern_key not in self.learned_patterns:
            self.learned_patterns[pattern_key] = {
                'occurrences': 0,
                'avg_error': 0,
                'characteristics': []
            }
        
        pattern = self.learned_patterns[pattern_key]
        pattern['occurrences'] += 1
        pattern['avg_error'] = ((pattern['avg_error'] * (pattern['occurrences'] - 1)) + 
                               error_percentage) / pattern['occurrences']
        
        # 특성 패턴 기록
        if error_analysis.get('high_volatility'):
            pattern['characteristics'].append('high_volatility')
        if shock_count > 2:
            pattern['characteristics'].append('multiple_shocks')
            
    def predict_future(self, current_data: pd.DataFrame, 
                      hours_ahead: int, analysis_context: Dict = None) -> Dict:
        """
        실시간 미래 예측 (학습된 패턴 활용)
        
        Args:
            current_data: 현재까지의 데이터
            hours_ahead: 예측할 시간
            analysis_context: 시계열 분석 컨텍스트
            
        Returns:
            예측 결과
        """
        print(f"🔮 {hours_ahead}시간 후 예측 실행")
        
        try:
            btc_col = self._identify_btc_price_column(current_data)
            if not btc_col:
                return None
            
            # 시계열 피처 준비
            X_features, _ = self._prepare_timeseries_features(
                current_data, btc_col, hours_ahead
            )
            
            # 예측 실행
            prediction_result = self._train_and_predict(
                X_features.iloc[:-hours_ahead], 
                current_data[btc_col].iloc[hours_ahead:len(X_features)-hours_ahead],
                current_data, btc_col
            )
            
            if not prediction_result:
                return None
            
            # 신뢰도 계산 (학습된 패턴 기반)
            current_regime = self._detect_market_regime(current_data[btc_col])
            confidence = self._calculate_prediction_confidence(current_regime)
            
            # 변동성 범위 추정
            recent_volatility = current_data[btc_col].pct_change().tail(24).std() * 100
            
            return {
                'predicted_price': prediction_result['prediction'],
                'confidence': confidence,
                'volatility_range': recent_volatility * 2,  # 2σ 범위
                'market_regime': current_regime,
                'prediction_timestamp': datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"실시간 예측 오류: {e}")
            return None
    
    def _calculate_prediction_confidence(self, market_regime: str) -> float:
        """
        시장 상황별 예측 신뢰도 계산 (학습 경험 기반)
        
        Args:
            market_regime: 시장 상황
            
        Returns:
            신뢰도 (0-100)
        """
        base_confidence = 75.0  # 기본 신뢰도
        
        # 학습된 패턴에서 해당 시장 상황의 평균 정확도 확인
        regime_patterns = [pattern for pattern in self.learned_patterns.keys() 
                          if pattern.startswith(market_regime)]
        
        if regime_patterns:
            regime_errors = [self.learned_patterns[pattern]['avg_error'] 
                           for pattern in regime_patterns]
            avg_regime_error = sum(regime_errors) / len(regime_errors)
            confidence = max(50, 100 - avg_regime_error)
        else:
            # 시장 상황별 기본 신뢰도
            regime_confidence = {
                'sideways': 85,
                'bull_market': 80,
                'bear_market': 78,
                'high_volatility': 65
            }
            confidence = regime_confidence.get(market_regime, base_confidence)
        
        return min(95, confidence)  # 최대 95%로 제한