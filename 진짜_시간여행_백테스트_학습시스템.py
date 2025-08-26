#!/usr/bin/env python3
"""
🕰️ 진짜 시간여행 백테스트 학습 시스템
- 과거 시점으로 돌아가서 예측 → 실제값 비교 → 실패 원인 분석 → 학습
- 돌발변수 영향도 분석 및 실시간 감시 리스트 생성
- 100%에 가까운 예측 정확도 달성을 위한 지속적 학습
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

class TimeravelBacktestLearningSystem:
    """시간여행 백테스트 학습 시스템"""
    
    def __init__(self):
        self.data_path = "/Users/parkyoungjun/Desktop/BTC_Analysis_System"
        self.historical_data = None
        self.prediction_errors = []
        self.error_analysis = {}
        self.shock_variables = {}
        self.learning_progress = {}
        self.current_accuracy = 0.0
        self.target_accuracy = 99.0  # 100%에 가까운 목표
        
        # 돌발변수 카테고리
        self.shock_categories = {
            'regulatory': ['SEC결정', '각국규제', '법적이슈'],
            'institutional': ['기관매수', '기관매도', 'ETF소식'],
            'technical': ['해킹', '네트워크장애', '업그레이드'], 
            'macro': ['금리변화', '달러강세', '경제위기'],
            'social': ['일론머스크', '소셜미디어', '언론보도']
        }
        
        print("🕰️ 시간여행 백테스트 학습 시스템 초기화")
        
    def load_historical_data(self) -> bool:
        """3개월치 통합 데이터 로드"""
        print("\n📂 3개월치 통합 데이터 로딩...")
        
        try:
            # AI 최적화된 3개월 데이터
            csv_path = os.path.join(self.data_path, "ai_optimized_3month_data", "ai_matrix_complete.csv")
            df = pd.read_csv(csv_path)
            
            # timestamp 컬럼 찾기 및 정렬
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.sort_values('timestamp').reset_index(drop=True)
                print("✅ timestamp로 시계열 정렬 완료")
            else:
                print("⚠️ timestamp 없음 - 원본 순서 사용")
            
            # 숫자형 컬럼만 추출
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            self.historical_data = df[['timestamp'] + list(numeric_cols) if 'timestamp' in df.columns else list(numeric_cols)].copy()
            
            # 결측치 처리
            self.historical_data = self.historical_data.fillna(method='ffill').fillna(method='bfill').fillna(0)
            
            print(f"✅ 데이터 로드 완료: {self.historical_data.shape}")
            print(f"📅 기간: {len(self.historical_data)}개 시점")
            
            return True
            
        except Exception as e:
            print(f"❌ 데이터 로드 실패: {e}")
            return False
    
    def identify_btc_price_column(self) -> str:
        """BTC 가격 컬럼 식별"""
        price_candidates = [
            'onchain_blockchain_info_network_stats_market_price_usd',
            'btc_price', 'price', 'close', 'market_price'
        ]
        
        for candidate in price_candidates:
            if candidate in self.historical_data.columns:
                return candidate
        
        # 첫 번째 숫자 컬럼을 가격으로 사용
        numeric_cols = self.historical_data.select_dtypes(include=[np.number]).columns
        return numeric_cols[0]
    
    def timetravel_backtest(self, start_date_idx: int, prediction_hours: int = 72) -> Dict:
        """
        시간여행 백테스트 실행
        
        Args:
            start_date_idx: 시작 시점 인덱스
            prediction_hours: 예측할 시간 (시간 단위)
        """
        print(f"\n🕰️ 시간여행 백테스트 실행")
        print(f"   📅 시작점: {start_date_idx}번째 데이터")
        print(f"   🎯 예측: {prediction_hours}시간 후")
        
        try:
            # 1단계: 과거 시점으로 이동
            historical_point = self.historical_data.iloc[:start_date_idx].copy()
            
            if len(historical_point) < 100:  # 충분한 학습 데이터 필요
                return {'success': False, 'error': '학습 데이터 부족'}
            
            # 2단계: 해당 시점의 지표들로 미래 예측
            btc_col = self.identify_btc_price_column()
            
            # 피처 준비
            X_historical = historical_point.drop(columns=['timestamp'] if 'timestamp' in historical_point.columns else []).drop(columns=[btc_col])
            y_historical = historical_point[btc_col]
            
            # 시계열 피처 추가
            X_enhanced = self.add_timeseries_features(X_historical, y_historical)
            
            # 타겟: prediction_hours 시간 후 가격
            target_idx = start_date_idx + prediction_hours
            
            if target_idx >= len(self.historical_data):
                return {'success': False, 'error': '예측 타겟 시점이 데이터 범위 초과'}
            
            actual_future_price = self.historical_data.iloc[target_idx][btc_col]
            current_price = self.historical_data.iloc[start_date_idx][btc_col]
            
            # 3단계: 모델 학습 (과거 데이터만 사용)
            y_target = y_historical.shift(-prediction_hours).dropna()
            X_target = X_enhanced.iloc[:-prediction_hours]
            
            if len(X_target) < 50:
                return {'success': False, 'error': '타겟 학습 데이터 부족'}
            
            # 스케일링
            scaler = RobustScaler()
            X_scaled = scaler.fit_transform(X_target)
            
            # 앙상블 모델 학습
            models = {
                'rf': RandomForestRegressor(n_estimators=50, random_state=42),
                'gb': GradientBoostingRegressor(n_estimators=50, random_state=42)
            }
            
            predictions = {}
            for name, model in models.items():
                model.fit(X_scaled, y_target)
                # 현재 시점 데이터로 예측
                current_features = X_enhanced.iloc[-1:].values
                current_scaled = scaler.transform(current_features)
                pred = model.predict(current_scaled)[0]
                predictions[name] = pred
            
            # 앙상블 예측
            final_prediction = (predictions['rf'] + predictions['gb']) / 2
            
            # 4단계: 실제값과 비교
            prediction_error = abs(actual_future_price - final_prediction)
            prediction_error_pct = (prediction_error / actual_future_price) * 100
            
            # 5단계: 에러 원인 분석
            error_analysis = self.analyze_prediction_error(
                start_date_idx, target_idx, final_prediction, actual_future_price
            )
            
            result = {
                'success': True,
                'start_idx': start_date_idx,
                'target_idx': target_idx,
                'current_price': current_price,
                'predicted_price': final_prediction,
                'actual_price': actual_future_price,
                'error_absolute': prediction_error,
                'error_percentage': prediction_error_pct,
                'error_analysis': error_analysis,
                'model_predictions': predictions
            }
            
            # 에러 기록 저장
            self.prediction_errors.append(result)
            
            return result
            
        except Exception as e:
            print(f"❌ 백테스트 실행 오류: {e}")
            return {'success': False, 'error': str(e)}
    
    def add_timeseries_features(self, X: pd.DataFrame, price_series: pd.Series) -> pd.DataFrame:
        """시계열 특성 피처 추가"""
        X_enhanced = X.copy()
        
        # 가격 기반 피처들
        X_enhanced['price_lag1'] = price_series.shift(1)
        X_enhanced['price_lag6'] = price_series.shift(6)
        X_enhanced['price_lag24'] = price_series.shift(24)
        
        X_enhanced['price_change_1h'] = price_series.pct_change(1) * 100
        X_enhanced['price_change_6h'] = price_series.pct_change(6) * 100
        X_enhanced['price_change_24h'] = price_series.pct_change(24) * 100
        
        X_enhanced['price_sma_12'] = price_series.rolling(12).mean()
        X_enhanced['price_sma_24'] = price_series.rolling(24).mean()
        X_enhanced['price_std_24'] = price_series.rolling(24).std()
        
        # NaN 처리
        X_enhanced = X_enhanced.fillna(method='bfill').fillna(0)
        
        return X_enhanced
    
    def analyze_prediction_error(self, start_idx: int, target_idx: int, 
                                prediction: float, actual: float) -> Dict:
        """예측 오류 원인 분석"""
        
        # 해당 기간 데이터 추출
        period_data = self.historical_data.iloc[start_idx:target_idx+1].copy()
        btc_col = self.identify_btc_price_column()
        
        analysis = {
            'error_magnitude': abs(actual - prediction),
            'error_direction': 'overpredict' if prediction > actual else 'underpredict',
            'price_volatility': period_data[btc_col].std(),
            'max_price_change': period_data[btc_col].max() - period_data[btc_col].min(),
            'potential_shock_events': []
        }
        
        # 급격한 변화 감지 (잠재적 돌발변수)
        price_changes = period_data[btc_col].pct_change().abs()
        shock_threshold = 0.05  # 5% 이상 변화
        
        shock_points = price_changes[price_changes > shock_threshold]
        if len(shock_points) > 0:
            analysis['potential_shock_events'] = [
                {
                    'timestamp': period_data.iloc[idx]['timestamp'] if 'timestamp' in period_data.columns else f"index_{idx}",
                    'change_pct': change * 100,
                    'price_before': period_data.iloc[idx-1][btc_col] if idx > 0 else None,
                    'price_after': period_data.iloc[idx][btc_col]
                }
                for idx, change in shock_points.items()
            ]
        
        # 지표별 기여도 분석 (상위 변화량 지표들)
        numeric_cols = period_data.select_dtypes(include=[np.number]).columns
        indicator_changes = {}
        
        for col in numeric_cols:
            if col != btc_col and len(period_data[col]) > 1:
                start_val = period_data[col].iloc[0]
                end_val = period_data[col].iloc[-1]
                if start_val != 0:
                    change_pct = abs((end_val - start_val) / start_val) * 100
                    indicator_changes[col] = change_pct
        
        # 상위 10개 변화량 지표
        top_changed_indicators = sorted(indicator_changes.items(), 
                                      key=lambda x: x[1], reverse=True)[:10]
        analysis['top_changed_indicators'] = top_changed_indicators
        
        return analysis
    
    def run_massive_backtest(self, num_tests: int = 100) -> Dict:
        """대규모 시간여행 백테스트 실행"""
        print(f"\n🚀 대규모 시간여행 백테스트 시작")
        print(f"   🎯 테스트 횟수: {num_tests}회")
        print("="*60)
        
        successful_tests = []
        failed_tests = []
        
        # 데이터 범위에서 랜덤하게 시작점 선택
        data_length = len(self.historical_data)
        prediction_hours = 72  # 3일 후 예측
        
        valid_start_range = data_length - prediction_hours - 50  # 충분한 여유
        
        for i in range(num_tests):
            start_idx = np.random.randint(100, valid_start_range)  # 최소 100개 학습 데이터
            
            print(f"🔍 테스트 {i+1:3d}/{num_tests}: 시점 {start_idx}", end="")
            
            result = self.timetravel_backtest(start_idx, prediction_hours)
            
            if result['success']:
                successful_tests.append(result)
                print(f" ✅ 에러 {result['error_percentage']:.2f}%")
            else:
                failed_tests.append(result)
                print(f" ❌ {result.get('error', 'Unknown')}")
        
        # 결과 분석
        if successful_tests:
            errors = [test['error_percentage'] for test in successful_tests]
            avg_error = np.mean(errors)
            median_error = np.median(errors)
            std_error = np.std(errors)
            
            # 정확도 계산 (에러가 작을수록 정확도 높음)
            accuracy = max(0, 100 - avg_error)
            
            print(f"\n📊 백테스트 결과 분석")
            print("="*50)
            print(f"✅ 성공한 테스트: {len(successful_tests)}/{num_tests}")
            print(f"📈 평균 에러율: {avg_error:.2f}%")
            print(f"📈 중간 에러율: {median_error:.2f}%") 
            print(f"📊 에러 표준편차: {std_error:.2f}%")
            print(f"🎯 추정 정확도: {accuracy:.2f}%")
            
            self.current_accuracy = accuracy
            
            # 에러 패턴 분석
            self.analyze_error_patterns(successful_tests)
            
            # 돌발변수 영향도 분석
            self.analyze_shock_variables(successful_tests)
            
        else:
            print("❌ 모든 백테스트 실패")
            accuracy = 0
        
        summary = {
            'total_tests': num_tests,
            'successful_tests': len(successful_tests),
            'failed_tests': len(failed_tests),
            'accuracy': accuracy,
            'avg_error': avg_error if successful_tests else 100,
            'error_analysis': self.error_analysis,
            'shock_variables': self.shock_variables
        }
        
        # 결과 저장
        self.save_learning_results(summary)
        
        return summary
    
    def analyze_error_patterns(self, test_results: List[Dict]):
        """에러 패턴 분석 - 왜 틀렸는가?"""
        print(f"\n🔍 에러 패턴 분석")
        print("-"*40)
        
        # 에러 크기별 분류
        small_errors = [t for t in test_results if t['error_percentage'] < 1]
        medium_errors = [t for t in test_results if 1 <= t['error_percentage'] < 5]
        large_errors = [t for t in test_results if t['error_percentage'] >= 5]
        
        print(f"📊 소에러 (<1%):   {len(small_errors)}건")
        print(f"📊 중에러 (1-5%):  {len(medium_errors)}건") 
        print(f"📊 대에러 (≥5%):   {len(large_errors)}건")
        
        # 대에러 사례 분석
        if large_errors:
            print(f"\n🚨 대에러 사례 분석:")
            for i, error in enumerate(large_errors[:3]):  # 상위 3개만
                print(f"   {i+1}. 에러 {error['error_percentage']:.1f}% - "
                      f"돌발이벤트 {len(error['error_analysis']['potential_shock_events'])}건")
        
        # 공통 실패 지표 찾기
        all_changed_indicators = {}
        for test in large_errors:
            for indicator, change in test['error_analysis']['top_changed_indicators']:
                if indicator in all_changed_indicators:
                    all_changed_indicators[indicator] += change
                else:
                    all_changed_indicators[indicator] = change
        
        # 실패와 가장 연관된 지표들
        problem_indicators = sorted(all_changed_indicators.items(), 
                                  key=lambda x: x[1], reverse=True)[:10]
        
        self.error_analysis = {
            'error_distribution': {
                'small': len(small_errors),
                'medium': len(medium_errors), 
                'large': len(large_errors)
            },
            'problem_indicators': problem_indicators,
            'avg_shock_events_per_large_error': np.mean([
                len(t['error_analysis']['potential_shock_events']) 
                for t in large_errors
            ]) if large_errors else 0
        }
        
        print(f"📈 문제 지표 TOP 5:")
        for i, (indicator, impact) in enumerate(problem_indicators[:5]):
            print(f"   {i+1}. {indicator[:50]}... (영향도: {impact:.1f})")
    
    def analyze_shock_variables(self, test_results: List[Dict]):
        """돌발변수 영향도 분석"""
        print(f"\n💥 돌발변수 영향도 분석")
        print("-"*40)
        
        # 돌발이벤트가 있었던 테스트들 분석
        shock_tests = [t for t in test_results 
                      if len(t['error_analysis']['potential_shock_events']) > 0]
        
        no_shock_tests = [t for t in test_results 
                         if len(t['error_analysis']['potential_shock_events']) == 0]
        
        if shock_tests and no_shock_tests:
            shock_avg_error = np.mean([t['error_percentage'] for t in shock_tests])
            normal_avg_error = np.mean([t['error_percentage'] for t in no_shock_tests])
            
            print(f"📊 돌발이벤트 有: 평균 에러 {shock_avg_error:.2f}% ({len(shock_tests)}건)")
            print(f"📊 돌발이벤트 無: 평균 에러 {normal_avg_error:.2f}% ({len(no_shock_tests)}건)")
            print(f"💥 돌발변수 영향: +{shock_avg_error - normal_avg_error:.2f}% 에러 증가")
            
            # 돌발변수 위험도 분류
            shock_impact = shock_avg_error - normal_avg_error
            if shock_impact > 5:
                risk_level = "🔴 고위험"
            elif shock_impact > 2:
                risk_level = "🟡 중위험"
            else:
                risk_level = "🟢 저위험"
            
            self.shock_variables = {
                'shock_impact_pct': shock_impact,
                'risk_level': risk_level,
                'shock_frequency': len(shock_tests) / len(test_results) * 100,
                'monitoring_priority': 'high' if shock_impact > 2 else 'medium'
            }
            
            print(f"🎯 돌발변수 위험도: {risk_level}")
            print(f"📊 돌발이벤트 빈도: {self.shock_variables['shock_frequency']:.1f}%")
            
        else:
            print("📊 돌발변수 영향도 분석 불가 (데이터 부족)")
            self.shock_variables = {'analysis': 'insufficient_data'}
    
    def generate_monitoring_recommendations(self) -> Dict:
        """실시간 감시 권장사항 생성"""
        print(f"\n👀 실시간 감시 권장사항 생성")
        print("-"*40)
        
        recommendations = {
            'critical_indicators': [],
            'shock_variables_to_monitor': [],
            'monitoring_frequency': {},
            'alert_thresholds': {}
        }
        
        # 1. 중요 지표 (에러 분석 기반)
        if hasattr(self, 'error_analysis') and 'problem_indicators' in self.error_analysis:
            top_indicators = self.error_analysis['problem_indicators'][:10]
            recommendations['critical_indicators'] = [
                {
                    'name': indicator,
                    'impact_score': impact,
                    'monitoring_priority': 'high' if impact > 50 else 'medium'
                }
                for indicator, impact in top_indicators
            ]
        
        # 2. 돌발변수 감시 리스트
        if self.shock_variables.get('monitoring_priority') == 'high':
            recommendations['shock_variables_to_monitor'] = [
                {
                    'category': '가격 급변동',
                    'threshold': '5% 이상 1시간 변동',
                    'action': '즉시 알림'
                },
                {
                    'category': '거래량 급증',
                    'threshold': '평균 대비 3배 이상',
                    'action': '주의 관찰'
                },
                {
                    'category': '뉴스/소셜미디어',
                    'threshold': '비트코인 언급량 급증',
                    'action': '수동 확인'
                }
            ]
        
        # 3. 감시 빈도
        recommendations['monitoring_frequency'] = {
            'price_data': '1분마다',
            'technical_indicators': '5분마다',
            'onchain_data': '1시간마다',
            'news_sentiment': '30분마다'
        }
        
        # 4. 알림 임계값
        recommendations['alert_thresholds'] = {
            'prediction_confidence_drop': '95% 미만',
            'unusual_market_activity': '정상 패턴에서 2σ 이상 이탈',
            'shock_variable_trigger': '주요 지표 10% 이상 급변'
        }
        
        print("✅ 감시 권장사항 생성 완료")
        print(f"   🎯 핵심 지표: {len(recommendations['critical_indicators'])}개")
        print(f"   💥 돌발변수: {len(recommendations['shock_variables_to_monitor'])}개")
        
        return recommendations
    
    def save_learning_results(self, summary: Dict):
        """학습 결과 저장"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        results = {
            'timestamp': timestamp,
            'summary': summary,
            'current_accuracy': self.current_accuracy,
            'target_accuracy': self.target_accuracy,
            'error_analysis': self.error_analysis,
            'shock_variables': self.shock_variables,
            'monitoring_recommendations': self.generate_monitoring_recommendations()
        }
        
        filename = f"timetravel_backtest_results_{timestamp}.json"
        filepath = os.path.join(self.data_path, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"\n💾 학습 결과 저장: {filename}")
        return filepath
    
    def run_complete_learning_cycle(self):
        """완전한 학습 사이클 실행"""
        print("🚀 시간여행 백테스트 학습 시스템 시작")
        print("="*70)
        
        # 1단계: 데이터 로드
        if not self.load_historical_data():
            print("❌ 데이터 로드 실패 - 시스템 종료")
            return None
        
        # 2단계: 대규모 백테스트
        summary = self.run_massive_backtest(num_tests=50)  # 50회 테스트
        
        # 3단계: 결과 출력
        print(f"\n" + "="*70)
        print("🏆 최종 학습 결과")
        print("="*70)
        print(f"🎯 현재 정확도:     {self.current_accuracy:.2f}%")
        print(f"🎯 목표 정확도:     {self.target_accuracy:.2f}%")
        
        if self.current_accuracy >= self.target_accuracy:
            print("🎉 목표 달성!")
        else:
            needed = self.target_accuracy - self.current_accuracy
            print(f"⚠️ 추가 필요:      +{needed:.2f}%")
        
        print(f"💥 돌발변수 영향:   +{self.shock_variables.get('shock_impact_pct', 0):.2f}% 에러")
        print(f"📊 백테스트 성공:   {summary['successful_tests']}/{summary['total_tests']}회")
        
        print("="*70)
        
        return summary

# 실행 함수
def main():
    system = TimeravelBacktestLearningSystem()
    return system.run_complete_learning_cycle()

if __name__ == "__main__":
    results = main()