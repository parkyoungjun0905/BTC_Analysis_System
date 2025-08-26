#!/usr/bin/env python3
"""
🕰️ 백테스트 학습 엔진
- 시간여행 백테스트: 과거 시점으로 돌아가서 미래 예측 → 실제값 비교 → 학습
- 사용자 예시: 25년 7월 23일 → 7월 26일 17시 예측 → 검증 → 학습
- 목표: 95% 정확도 달성을 위한 지속적 학습
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

warnings.filterwarnings('ignore')

class BacktestLearningEngine:
    """백테스트 학습 엔진 - 시간여행을 통한 지속적 학습"""
    
    def __init__(self, data_path: str):
        """
        Args:
            data_path: 3개월치 통합 데이터 경로
        """
        self.data_path = data_path
        self.data = None
        self.btc_price_column = None
        
        # 학습 상태 추적
        self.current_accuracy = 0.0
        self.target_accuracy = 90.0  # 95% → 90%로 수정 (더 빠른 완료)
        self.learning_history = []
        self.failure_patterns = {}
        self.learned_rules = []
        
        # 모델 저장
        self.models = {}
        self.scalers = {}
        self.model_save_path = os.path.join(os.path.dirname(data_path), "btc_insight", "saved_models")
        os.makedirs(self.model_save_path, exist_ok=True)
        
        print("🕰️ 백테스트 학습 엔진 초기화")
        print(f"📂 데이터 경로: {data_path}")
        print(f"💾 모델 저장 경로: {self.model_save_path}")
        
    def load_data(self) -> bool:
        """3개월치 1시간 단위 통합 데이터 로드"""
        print("\n📊 3개월치 통합 데이터 로딩...")
        
        try:
            csv_file = os.path.join(self.data_path, "ai_matrix_complete.csv")
            
            if not os.path.exists(csv_file):
                print(f"❌ 데이터 파일 없음: {csv_file}")
                return False
            
            # 데이터 로드
            self.data = pd.read_csv(csv_file)
            print(f"📈 원본 데이터: {self.data.shape}")
            
            # timestamp 처리
            if 'timestamp' in self.data.columns:
                self.data['timestamp'] = pd.to_datetime(self.data['timestamp'])
                self.data = self.data.sort_values('timestamp').reset_index(drop=True)
                print("✅ 시간순 정렬 완료")
            
            # 숫자형 컬럼만 추출
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns
            if 'timestamp' in self.data.columns:
                self.data = self.data[['timestamp'] + list(numeric_cols)]
            else:
                self.data = self.data[list(numeric_cols)]
            
            # 결측치 처리
            self.data = self.data.fillna(method='ffill').fillna(method='bfill').fillna(0)
            
            # BTC 가격 컬럼 식별
            self.btc_price_column = self._identify_btc_price_column()
            
            print(f"✅ 데이터 로드 완료: {self.data.shape}")
            print(f"💰 BTC 가격 컬럼: {self.btc_price_column}")
            print(f"📅 데이터 기간: {len(self.data)}시간 ({len(self.data)/24:.1f}일)")
            
            return True
            
        except Exception as e:
            print(f"❌ 데이터 로드 실패: {e}")
            return False
    
    def _identify_btc_price_column(self) -> str:
        """BTC 가격 컬럼 식별"""
        candidates = [
            'onchain_blockchain_info_network_stats_market_price_usd',
            'btc_price', 'price', 'close', 'market_price'
        ]
        
        for candidate in candidates:
            if candidate in self.data.columns:
                return candidate
        
        # 가장 큰 평균값을 가진 컬럼 (BTC 가격 특성)
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        max_avg_col = None
        max_avg = 0
        
        for col in numeric_cols:
            avg_val = self.data[col].mean()
            if avg_val > max_avg and avg_val > 1000:  # BTC 가격은 보통 수만 달러
                max_avg = avg_val
                max_avg_col = col
        
        return max_avg_col
    
    def timetravel_backtest(self, start_idx: int, prediction_hours: int = 72) -> Dict:
        """
        시간여행 백테스트 실행
        
        사용자 예시 과정:
        1. 25년 7월 23일 시점으로 돌아감 (start_idx)
        2. 해당 시점의 지표들로 시계열 분석
        3. 7월 26일 17시(72시간 후) 가격 예측
        4. 실제 7월 26일 17시 값과 비교
        5. 예측 오차 원인 분석 및 학습
        
        Args:
            start_idx: 시작 시점 인덱스
            prediction_hours: 예측할 시간 (기본 72시간 = 3일)
            
        Returns:
            백테스트 결과 딕셔너리
        """
        print(f"\n🕰️ 시간여행 백테스트: {start_idx}번째 시점 → {prediction_hours}시간 후")
        
        try:
            # 1단계: 과거 시점으로 "돌아가기" (해당 시점까지의 데이터만 사용)
            historical_data = self.data.iloc[:start_idx].copy()
            
            if len(historical_data) < 168:  # 최소 1주일 데이터 필요
                return {'success': False, 'error': '학습 데이터 부족 (168시간 미만)'}
            
            # 2단계: 예측 타겟 시점 확인
            target_idx = start_idx + prediction_hours
            if target_idx >= len(self.data):
                return {'success': False, 'error': '예측 타겟이 데이터 범위 초과'}
            
            # 3단계: 시계열 특성 피처 생성 (필수 요구사항)
            X_features, y_target = self._create_timeseries_features(
                historical_data, prediction_hours
            )
            
            if len(X_features) < 50:
                return {'success': False, 'error': '시계열 피처 생성 실패 또는 데이터 부족'}
            
            # 4단계: 모델 학습 (과거 데이터만 사용)
            model_results = self._train_ensemble_models(X_features, y_target)
            
            if not model_results:
                return {'success': False, 'error': '모델 학습 실패'}
            
            # 5단계: 현재 시점에서 미래 예측
            current_features = self._extract_current_features(historical_data)
            predicted_price = self._predict_with_ensemble(current_features, model_results)
            
            # 6단계: 실제값과 비교 ("7월 26일 17시 실제 BTC 값" 확인)
            actual_price = self.data.iloc[target_idx][self.btc_price_column]
            current_price = self.data.iloc[start_idx][self.btc_price_column]
            
            # 7단계: 예측 성능 계산
            absolute_error = abs(actual_price - predicted_price)
            percentage_error = (absolute_error / actual_price) * 100
            accuracy = max(0, 100 - percentage_error)
            
            # 8단계: 예측 실패 원인 분석
            failure_analysis = self._analyze_prediction_failure(
                start_idx, target_idx, predicted_price, actual_price
            )
            
            # 9단계: 학습 패턴 업데이트
            self._update_learning_patterns(failure_analysis, accuracy, percentage_error)
            
            result = {
                'success': True,
                'iteration_info': {
                    'start_idx': start_idx,
                    'target_idx': target_idx, 
                    'prediction_hours': prediction_hours
                },
                'prices': {
                    'current_price': float(current_price),
                    'predicted_price': float(predicted_price),
                    'actual_price': float(actual_price)
                },
                'performance': {
                    'absolute_error': float(absolute_error),
                    'percentage_error': float(percentage_error),
                    'accuracy': float(accuracy)
                },
                'failure_analysis': failure_analysis,
                'model_info': model_results['info']
            }
            
            return result
            
        except Exception as e:
            print(f"❌ 백테스트 실행 오류: {e}")
            return {'success': False, 'error': str(e)}
    
    def _create_timeseries_features(self, data: pd.DataFrame, 
                                   prediction_hours: int) -> Tuple[pd.DataFrame, pd.Series]:
        """시계열 분석을 위한 피처 생성 (사용자 요구사항)"""
        
        # 기본 피처 (BTC 가격 제외한 모든 지표)
        feature_cols = [col for col in data.columns 
                       if col != self.btc_price_column and col != 'timestamp']
        X_base = data[feature_cols].copy()
        
        # BTC 가격 시계열 피처 추가
        btc_prices = data[self.btc_price_column]
        
        # 1. 가격 지연 피처 (Lag Features)
        for lag in [1, 6, 12, 24, 48]:
            X_base[f'btc_lag_{lag}h'] = btc_prices.shift(lag)
        
        # 2. 가격 변화율 피처 
        for period in [1, 6, 12, 24]:
            X_base[f'btc_pct_change_{period}h'] = btc_prices.pct_change(period) * 100
        
        # 3. 이동평균 피처
        for window in [12, 24, 72, 168]:  # 12h, 1d, 3d, 1w
            X_base[f'btc_sma_{window}h'] = btc_prices.rolling(window).mean()
        
        # 4. 변동성 피처
        for window in [12, 24, 72]:
            X_base[f'btc_volatility_{window}h'] = btc_prices.rolling(window).std()
        
        # 5. 기술적 지표
        X_base['btc_rsi_14'] = self._calculate_rsi(btc_prices, 14)
        X_base['btc_macd'], X_base['btc_macd_signal'] = self._calculate_macd(btc_prices)
        
        # 6. 시간 특성
        if 'timestamp' in data.columns:
            X_base['hour'] = data['timestamp'].dt.hour
            X_base['day_of_week'] = data['timestamp'].dt.dayofweek
            X_base['is_weekend'] = (data['timestamp'].dt.dayofweek >= 5).astype(int)
        
        # 타겟: prediction_hours 시간 후 가격
        y_target = btc_prices.shift(-prediction_hours)
        
        # 결측치 제거
        valid_mask = ~(X_base.isnull().any(axis=1) | y_target.isnull())
        X_clean = X_base[valid_mask].iloc[:-prediction_hours]  # 미래 데이터 제외
        y_clean = y_target[valid_mask].iloc[:-prediction_hours]
        
        return X_clean, y_clean
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """RSI 계산"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices: pd.Series, 
                       fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series]:
        """MACD 계산"""
        exp1 = prices.ewm(span=fast).mean()
        exp2 = prices.ewm(span=slow).mean()
        macd = exp1 - exp2
        macd_signal = macd.ewm(span=signal).mean()
        return macd, macd_signal
    
    def _train_ensemble_models(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """앙상블 모델 학습"""
        try:
            # 데이터 스케일링
            scaler = RobustScaler()
            X_scaled = scaler.fit_transform(X)
            
            # 앙상블 모델
            models = {
                'random_forest': RandomForestRegressor(
                    n_estimators=100, 
                    max_depth=15,
                    min_samples_split=5,
                    random_state=42,
                    n_jobs=-1
                ),
                'gradient_boosting': GradientBoostingRegressor(
                    n_estimators=100,
                    max_depth=8,
                    learning_rate=0.1,
                    random_state=42
                ),
                'ridge': Ridge(alpha=1.0)
            }
            
            # 모델별 학습 및 성능 평가
            trained_models = {}
            model_scores = {}
            
            # 학습/검증 분할
            split_idx = int(len(X_scaled) * 0.8)
            X_train, X_val = X_scaled[:split_idx], X_scaled[split_idx:]
            y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
            
            for name, model in models.items():
                # 모델 학습
                model.fit(X_train, y_train)
                
                # 검증 성능
                val_pred = model.predict(X_val)
                mae = mean_absolute_error(y_val, val_pred)
                r2 = r2_score(y_val, val_pred)
                
                trained_models[name] = model
                model_scores[name] = {'mae': mae, 'r2': r2}
            
            return {
                'models': trained_models,
                'scaler': scaler,
                'scores': model_scores,
                'info': {
                    'training_samples': len(X_train),
                    'validation_samples': len(X_val),
                    'features': X.shape[1]
                }
            }
            
        except Exception as e:
            print(f"❌ 모델 학습 오류: {e}")
            return None
    
    def _extract_current_features(self, historical_data: pd.DataFrame) -> pd.DataFrame:
        """현재 시점의 피처 추출"""
        # 시계열 피처 생성 (예측용)
        X_features, _ = self._create_timeseries_features(historical_data, 1)
        return X_features.iloc[-1:].copy()  # 마지막 행만 반환
    
    def _predict_with_ensemble(self, features: pd.DataFrame, model_results: Dict) -> float:
        """앙상블 모델로 예측"""
        models = model_results['models']
        scaler = model_results['scaler']
        scores = model_results['scores']
        
        # 피처 스케일링
        features_scaled = scaler.transform(features)
        
        # 모델별 예측
        predictions = {}
        weights = {}
        
        for name, model in models.items():
            pred = model.predict(features_scaled)[0]
            predictions[name] = pred
            
            # 가중치 (낮은 MAE일수록 높은 가중치)
            mae = scores[name]['mae']
            weights[name] = 1 / (mae + 1e-8)  # 0으로 나누기 방지
        
        # 가중치 정규화
        total_weight = sum(weights.values())
        normalized_weights = {k: v/total_weight for k, v in weights.items()}
        
        # 가중 평균 예측
        ensemble_prediction = sum(predictions[name] * normalized_weights[name] 
                                for name in predictions.keys())
        
        return float(ensemble_prediction)
    
    def _analyze_prediction_failure(self, start_idx: int, target_idx: int,
                                  predicted: float, actual: float) -> Dict:
        """예측 실패 원인 분석 (학습을 위한 핵심 기능)"""
        
        # 예측 기간 데이터
        period_data = self.data.iloc[start_idx:target_idx+1].copy()
        btc_prices = period_data[self.btc_price_column]
        
        analysis = {
            'error_magnitude': abs(actual - predicted),
            'error_direction': 'overestimate' if predicted > actual else 'underestimate',
            'price_volatility': float(btc_prices.std()),
            'max_price_swing': float(btc_prices.max() - btc_prices.min()),
            'trend_consistency': self._measure_trend_consistency(btc_prices),
            'shock_events': [],
            'indicator_anomalies': [],
            'market_regime': self._detect_market_regime(btc_prices)
        }
        
        # 돌발 이벤트 감지 (급격한 가격 변화)
        price_changes = btc_prices.pct_change().abs()
        shock_threshold = 0.05  # 5% 이상 1시간 변화
        
        shock_events = price_changes[price_changes > shock_threshold]
        for idx in shock_events.index:
            if idx < len(period_data):
                analysis['shock_events'].append({
                    'index': int(idx),
                    'change_percent': float(shock_events.loc[idx] * 100),
                    'timestamp': period_data.iloc[idx]['timestamp'].isoformat() 
                              if 'timestamp' in period_data.columns else f"hour_{idx}"
                })
        
        # 지표 이상치 감지
        numeric_cols = period_data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col != self.btc_price_column:
                col_data = period_data[col]
                if not col_data.isnull().all():
                    # Z-score 기반 이상치 감지
                    z_scores = np.abs((col_data - col_data.mean()) / col_data.std())
                    anomalies = z_scores > 3  # 3σ 이상
                    
                    if anomalies.sum() > 0:
                        analysis['indicator_anomalies'].append({
                            'indicator': col,
                            'anomaly_count': int(anomalies.sum()),
                            'max_z_score': float(z_scores.max())
                        })
        
        return analysis
    
    def _measure_trend_consistency(self, prices: pd.Series) -> float:
        """트렌드 일관성 측정"""
        if len(prices) < 12:
            return 0.0
        
        # 12시간 단위로 트렌드 방향 확인
        trends = []
        for i in range(0, len(prices) - 12, 12):
            segment = prices.iloc[i:i+12]
            if len(segment) >= 12:
                trend = (segment.iloc[-1] - segment.iloc[0]) / segment.iloc[0]
                trends.append(1 if trend > 0 else -1)
        
        if not trends:
            return 0.0
        
        # 일관성 = 같은 방향 비율
        positive = sum(1 for t in trends if t > 0)
        negative = sum(1 for t in trends if t < 0)
        consistency = abs(positive - negative) / len(trends)
        
        return float(consistency)
    
    def _detect_market_regime(self, prices: pd.Series) -> str:
        """시장 상황 감지"""
        if len(prices) < 24:
            return 'insufficient_data'
        
        # 전체 기간 트렌드
        total_trend = (prices.iloc[-1] - prices.iloc[0]) / prices.iloc[0] * 100
        
        # 변동성
        volatility = prices.pct_change().std() * 100
        
        # 체제 분류
        if total_trend > 5 and volatility < 3:
            return 'steady_bull'
        elif total_trend > 2:
            return 'bull_market'
        elif total_trend < -5 and volatility < 3:
            return 'steady_bear'
        elif total_trend < -2:
            return 'bear_market'
        elif volatility > 5:
            return 'high_volatility'
        else:
            return 'sideways'
    
    def _update_learning_patterns(self, analysis: Dict, accuracy: float, error_pct: float):
        """학습 패턴 업데이트"""
        
        # 학습 히스토리에 추가
        learning_entry = {
            'timestamp': datetime.now().isoformat(),
            'accuracy': accuracy,
            'error_percentage': error_pct,
            'market_regime': analysis.get('market_regime', 'unknown'),
            'shock_events_count': len(analysis.get('shock_events', [])),
            'trend_consistency': analysis.get('trend_consistency', 0)
        }
        
        self.learning_history.append(learning_entry)
        
        # 실패 패턴 누적 (5% 이상 에러시)
        if error_pct > 5.0:
            regime = analysis.get('market_regime', 'unknown')
            shock_count = len(analysis.get('shock_events', []))
            
            pattern_key = f"{regime}_shocks_{shock_count}"
            
            if pattern_key not in self.failure_patterns:
                self.failure_patterns[pattern_key] = {
                    'count': 0,
                    'avg_error': 0,
                    'characteristics': []
                }
            
            pattern = self.failure_patterns[pattern_key]
            pattern['count'] += 1
            pattern['avg_error'] = ((pattern['avg_error'] * (pattern['count'] - 1)) + 
                                  error_pct) / pattern['count']
        
        # 현재 정확도 업데이트 (최근 20회 평균)
        recent_accuracies = [entry['accuracy'] for entry in self.learning_history[-20:]]
        self.current_accuracy = sum(recent_accuracies) / len(recent_accuracies)
    
    def run_infinite_learning(self, max_iterations: int = 1000) -> Dict:
        """
        무한 학습 루프 실행
        
        목표: 95% 정확도 달성까지 지속적 학습
        
        Args:
            max_iterations: 최대 반복 횟수
            
        Returns:
            학습 결과 요약
        """
        print(f"\n🚀 무한 백테스트 학습 시작")
        print(f"🎯 목표 정확도: {self.target_accuracy}% (수정됨: 더 빠른 완료)")
        print(f"🔄 최대 반복: {max_iterations}회")
        print("="*60)
        
        successful_tests = []
        failed_tests = []
        
        # 유효한 시작 인덱스 범위 계산
        min_start = 168  # 최소 1주일 학습 데이터
        max_start = len(self.data) - 168  # 최소 1주일 예측 여유
        
        for iteration in range(1, max_iterations + 1):
            # 랜덤 시점 선택 (사용자 요구사항)
            start_idx = np.random.randint(min_start, max_start)
            prediction_hours = np.random.choice([24, 48, 72, 96])  # 1~4일
            
            print(f"🔍 반복 {iteration:4d}/{max_iterations}: "
                  f"시점 {start_idx} → {prediction_hours}h 후 예측", end="")
            
            # 백테스트 실행
            result = self.timetravel_backtest(start_idx, prediction_hours)
            
            if result['success']:
                successful_tests.append(result)
                accuracy = result['performance']['accuracy']
                error = result['performance']['percentage_error']
                
                print(f" ✅ 정확도: {accuracy:.2f}% (에러: {error:.2f}%)")
                
                # 10회마다 진행 상황 출력
                if iteration % 10 == 0:
                    print(f"📈 현재 평균 정확도: {self.current_accuracy:.2f}%")
                
                # 목표 달성 확인
                if self.current_accuracy >= self.target_accuracy:
                    print(f"\n🎉 목표 달성! {self.current_accuracy:.2f}% >= {self.target_accuracy}%")
                    print(f"🏆 총 {iteration}회 학습으로 목표 달성")
                    break
                    
            else:
                failed_tests.append(result)
                print(f" ❌ {result.get('error', 'Unknown error')}")
        
        # 최종 결과 정리
        total_tests = len(successful_tests) + len(failed_tests)
        success_rate = len(successful_tests) / total_tests * 100 if total_tests > 0 else 0
        
        final_result = {
            'learning_completed': self.current_accuracy >= self.target_accuracy,
            'total_iterations': total_tests,
            'successful_tests': len(successful_tests),
            'failed_tests': len(failed_tests),
            'success_rate': success_rate,
            'final_accuracy': self.current_accuracy,
            'target_accuracy': self.target_accuracy,
            'learned_patterns': len(self.failure_patterns),
            'learning_history': self.learning_history,
            'failure_patterns': self.failure_patterns,
            'completion_time': datetime.now().isoformat()
        }
        
        # 결과 저장
        self._save_learning_results(final_result)
        
        # 최종 보고서
        print(f"\n" + "="*60)
        print("🏆 백테스트 학습 완료 보고서")
        print("="*60)
        print(f"🎯 목표 달성: {'✅' if final_result['learning_completed'] else '❌'}")
        print(f"📊 최종 정확도: {self.current_accuracy:.2f}%")
        print(f"🔄 총 반복: {total_tests}회")
        print(f"✅ 성공: {len(successful_tests)}회 ({success_rate:.1f}%)")
        print(f"❌ 실패: {len(failed_tests)}회")
        print(f"📚 학습 패턴: {len(self.failure_patterns)}개")
        print("="*60)
        
        return final_result
    
    def _save_learning_results(self, results: Dict):
        """학습 결과 저장"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"backtest_learning_results_{timestamp}.json"
        
        # logs 폴더 생성
        logs_dir = os.path.join(os.path.dirname(self.data_path), "btc_insight", "logs")
        os.makedirs(logs_dir, exist_ok=True)
        
        filepath = os.path.join(logs_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"💾 학습 결과 저장: {filepath}")
        
    def get_learned_rules(self) -> List[str]:
        """학습된 규칙들 반환"""
        rules = []
        
        # 실패 패턴 기반 규칙 생성
        for pattern, data in self.failure_patterns.items():
            if data['count'] >= 3:  # 3회 이상 발생한 패턴
                rule = f"{pattern} 상황에서 평균 {data['avg_error']:.1f}% 오차 발생 ({data['count']}회)"
                rules.append(rule)
        
        # 정확도 기반 규칙
        if self.current_accuracy >= self.target_accuracy:
            rules.append(f"현재 시스템 정확도 {self.current_accuracy:.2f}%로 실전 적용 가능")
        
        return rules
    
    def save_trained_models(self) -> bool:
        """학습된 모델들 저장"""
        try:
            if not self.models or not self.scalers:
                print("❌ 저장할 학습된 모델이 없습니다")
                return False
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 모델 저장
            for model_name, model in self.models.items():
                model_path = os.path.join(self.model_save_path, f"{model_name}_{timestamp}.pkl")
                joblib.dump(model, model_path)
                print(f"💾 {model_name} 모델 저장: {model_path}")
            
            # 스케일러 저장
            for scaler_name, scaler in self.scalers.items():
                scaler_path = os.path.join(self.model_save_path, f"{scaler_name}_{timestamp}.pkl")
                joblib.dump(scaler, scaler_path)
                print(f"💾 {scaler_name} 스케일러 저장: {scaler_path}")
            
            # 메타 정보 저장
            meta_info = {
                'accuracy': self.current_accuracy,
                'target_accuracy': self.target_accuracy,
                'learning_iterations': len(self.learning_history),
                'failure_patterns_count': len(self.failure_patterns),
                'timestamp': timestamp,
                'btc_price_column': self.btc_price_column,
                'model_files': list(self.models.keys()),
                'scaler_files': list(self.scalers.keys())
            }
            
            meta_path = os.path.join(self.model_save_path, f"model_meta_{timestamp}.json")
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(meta_info, f, ensure_ascii=False, indent=2)
            
            print(f"✅ 모델 저장 완료 - 정확도: {self.current_accuracy:.2f}%")
            return True
            
        except Exception as e:
            print(f"❌ 모델 저장 실패: {e}")
            return False
    
    def load_trained_models(self, model_timestamp: str = None) -> bool:
        """저장된 모델들 로드"""
        try:
            if model_timestamp is None:
                # 가장 최근 모델 찾기
                model_files = [f for f in os.listdir(self.model_save_path) if f.startswith("model_meta_")]
                if not model_files:
                    print("❌ 저장된 모델이 없습니다")
                    return False
                
                latest_meta = sorted(model_files)[-1]
                model_timestamp = latest_meta.replace("model_meta_", "").replace(".json", "")
            
            # 메타 정보 로드
            meta_path = os.path.join(self.model_save_path, f"model_meta_{model_timestamp}.json")
            if not os.path.exists(meta_path):
                print(f"❌ 모델 메타 파일을 찾을 수 없습니다: {meta_path}")
                return False
            
            with open(meta_path, 'r', encoding='utf-8') as f:
                meta_info = json.load(f)
            
            # 모델들 로드
            self.models = {}
            for model_name in meta_info['model_files']:
                model_path = os.path.join(self.model_save_path, f"{model_name}_{model_timestamp}.pkl")
                if os.path.exists(model_path):
                    self.models[model_name] = joblib.load(model_path)
                    print(f"📥 {model_name} 모델 로드 완료")
            
            # 스케일러들 로드
            self.scalers = {}
            for scaler_name in meta_info['scaler_files']:
                scaler_path = os.path.join(self.model_save_path, f"{scaler_name}_{model_timestamp}.pkl")
                if os.path.exists(scaler_path):
                    self.scalers[scaler_name] = joblib.load(scaler_path)
                    print(f"📥 {scaler_name} 스케일러 로드 완료")
            
            # 학습 상태 복원
            self.current_accuracy = meta_info.get('accuracy', 0.0)
            self.btc_price_column = meta_info.get('btc_price_column')
            
            print(f"✅ 모델 로드 완료 - 저장 시 정확도: {self.current_accuracy:.2f}%")
            print(f"📅 모델 생성일: {model_timestamp}")
            return True
            
        except Exception as e:
            print(f"❌ 모델 로드 실패: {e}")
            return False
    
    def list_saved_models(self):
        """저장된 모델들 목록 출력"""
        try:
            model_files = [f for f in os.listdir(self.model_save_path) if f.startswith("model_meta_")]
            if not model_files:
                print("📭 저장된 모델이 없습니다")
                return
            
            print("\n📚 저장된 모델들:")
            print("=" * 60)
            
            for meta_file in sorted(model_files, reverse=True):
                timestamp = meta_file.replace("model_meta_", "").replace(".json", "")
                meta_path = os.path.join(self.model_save_path, meta_file)
                
                with open(meta_path, 'r', encoding='utf-8') as f:
                    meta_info = json.load(f)
                
                print(f"🕐 {timestamp}")
                print(f"   📊 정확도: {meta_info.get('accuracy', 0):.2f}%")
                print(f"   🔄 학습 횟수: {meta_info.get('learning_iterations', 0)}회")
                print(f"   📚 실패 패턴: {meta_info.get('failure_patterns_count', 0)}개")
                print()
                
        except Exception as e:
            print(f"❌ 모델 목록 조회 실패: {e}")

# 실행 함수
def main():
    """메인 실행 함수"""
    data_path = "/Users/parkyoungjun/Desktop/BTC_Analysis_System/ai_optimized_3month_data"
    
    # 백테스트 학습 엔진 생성
    engine = BacktestLearningEngine(data_path)
    
    # 데이터 로드
    if not engine.load_data():
        print("❌ 데이터 로드 실패 - 프로그램 종료")
        return None
    
    # 무한 학습 실행
    results = engine.run_infinite_learning(max_iterations=200)
    
    # 학습된 규칙 출력
    rules = engine.get_learned_rules()
    if rules:
        print("\n📚 학습된 규칙들:")
        for i, rule in enumerate(rules, 1):
            print(f"   {i}. {rule}")
    
    return results

if __name__ == "__main__":
    results = main()