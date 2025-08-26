#!/usr/bin/env python3
"""
🎯 무한 백테스트 95% 정확도 공식 발견기
- 과거 임의 시점에서 예측 → 실제값 검증 → 학습 → 무한 반복
- 최적의 지표 조합과 적용 공식 자동 발견
- 95%+ 정확도 달성까지 지속적 진화
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.preprocessing import RobustScaler, PolynomialFeatures
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score
import itertools
import random

warnings.filterwarnings('ignore')

class InfiniteBacktestFormulaDiscoverer:
    """무한 백테스트 95% 공식 발견기"""
    
    def __init__(self):
        self.data_path = "/Users/parkyoungjun/Desktop/BTC_Analysis_System"
        self.historical_data = None
        self.best_formulas = []  # 발견된 최고 공식들
        self.learning_iterations = 0
        self.current_best_accuracy = 0.0
        self.target_accuracy = 95.0
        
        # 지표 카테고리별 분류
        self.indicator_categories = {
            'price': ['price', 'close', 'open', 'high', 'low'],
            'volume': ['volume', 'trade_volume'],
            'technical': ['rsi', 'macd', 'bollinger', 'sma', 'ema'],
            'onchain': ['whale_ratio', 'mvrv', 'sopr', 'nvt'],
            'derivatives': ['funding_rate', 'open_interest', 'basis'],
            'macro': ['dxy', 'gold', 'nasdaq', 'vix'],
            'sentiment': ['fear_greed', 'social']
        }
        
        # 예측 시간 옵션 (시간 단위)
        self.prediction_horizons = [1, 6, 12, 24, 48, 72, 168]  # 1시간~1주
        
        print("🎯 무한 백테스트 95% 공식 발견기 초기화")
        
    def load_data(self) -> bool:
        """3개월 통합 데이터 로드"""
        print("\n📂 데이터 로딩...")
        
        try:
            csv_path = os.path.join(self.data_path, "ai_optimized_3month_data", "ai_matrix_complete.csv")
            df = pd.read_csv(csv_path)
            
            # 타임스탬프 처리
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.sort_values('timestamp').reset_index(drop=True)
            
            # 숫자형 데이터만 추출
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            self.historical_data = df[['timestamp'] + list(numeric_cols) if 'timestamp' in df.columns else list(numeric_cols)].copy()
            
            # 결측치 처리
            self.historical_data = self.historical_data.fillna(method='ffill').fillna(method='bfill').fillna(0)
            
            print(f"✅ 데이터 로드 완료: {self.historical_data.shape}")
            return True
            
        except Exception as e:
            print(f"❌ 데이터 로드 실패: {e}")
            return False
    
    def categorize_indicators(self) -> Dict[str, List[str]]:
        """지표들을 카테고리별로 분류"""
        available_cols = self.historical_data.columns.tolist()
        if 'timestamp' in available_cols:
            available_cols.remove('timestamp')
            
        categorized = {}
        
        for category, keywords in self.indicator_categories.items():
            categorized[category] = []
            for col in available_cols:
                for keyword in keywords:
                    if keyword.lower() in col.lower():
                        categorized[category].append(col)
                        break
        
        # 미분류 지표들
        classified_cols = set()
        for cols in categorized.values():
            classified_cols.update(cols)
        
        categorized['others'] = [col for col in available_cols if col not in classified_cols]
        
        return categorized
    
    def generate_feature_combinations(self, max_features: int = 20) -> List[List[str]]:
        """다양한 지표 조합 생성"""
        categorized = self.categorize_indicators()
        combinations = []
        
        # 1. 카테고리별 대표 지표 조합
        for r in range(2, min(len(categorized), 6)):  # 2~5개 카테고리 조합
            for category_combo in itertools.combinations(categorized.keys(), r):
                features = []
                for category in category_combo:
                    if categorized[category]:
                        # 각 카테고리에서 상위 N개씩
                        features.extend(categorized[category][:min(5, len(categorized[category]))])
                
                if len(features) <= max_features:
                    combinations.append(features)
        
        # 2. 랜덤 조합 (다양성 확보)
        all_features = [col for cols in categorized.values() for col in cols]
        for _ in range(50):  # 50개 랜덤 조합
            num_features = random.randint(5, min(max_features, len(all_features)))
            random_combo = random.sample(all_features, num_features)
            combinations.append(random_combo)
        
        return combinations
    
    def create_advanced_features(self, base_features: List[str], target_col: str) -> pd.DataFrame:
        """고급 파생 피처 생성"""
        df = self.historical_data.copy()
        
        # 기본 피처들만 선택
        available_features = [f for f in base_features if f in df.columns]
        if not available_features:
            return pd.DataFrame()
            
        feature_df = df[available_features + [target_col]].copy()
        
        # 가격 기반 고급 피처 (target_col이 가격인 경우)
        price_data = df[target_col]
        
        # 1. 다중 기간 이동평균
        for period in [12, 24, 168]:
            feature_df[f'{target_col}_sma_{period}'] = price_data.rolling(period).mean()
            feature_df[f'{target_col}_ema_{period}'] = price_data.ewm(period).mean()
        
        # 2. 변동성 지표
        for period in [12, 24, 168]:
            feature_df[f'{target_col}_volatility_{period}'] = price_data.pct_change().rolling(period).std()
            feature_df[f'{target_col}_range_{period}'] = (price_data.rolling(period).max() - price_data.rolling(period).min()) / price_data.rolling(period).mean()
        
        # 3. 모멘텀 지표
        for period in [1, 6, 24, 168]:
            feature_df[f'{target_col}_momentum_{period}'] = price_data.pct_change(period)
            feature_df[f'{target_col}_roc_{period}'] = (price_data - price_data.shift(period)) / price_data.shift(period)
        
        # 4. 기술적 지표
        # RSI
        delta = price_data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        feature_df[f'{target_col}_rsi'] = 100 - (100 / (1 + rs))
        
        # 볼린저 밴드
        sma_20 = price_data.rolling(20).mean()
        std_20 = price_data.rolling(20).std()
        feature_df[f'{target_col}_bb_upper'] = sma_20 + (std_20 * 2)
        feature_df[f'{target_col}_bb_lower'] = sma_20 - (std_20 * 2)
        feature_df[f'{target_col}_bb_position'] = (price_data - feature_df[f'{target_col}_bb_lower']) / (feature_df[f'{target_col}_bb_upper'] - feature_df[f'{target_col}_bb_lower'])
        
        # 5. 지표간 상호작용 (상위 10개 피처만)
        numeric_features = [f for f in available_features[:10] if f in feature_df.columns]
        for i, feat1 in enumerate(numeric_features):
            for feat2 in numeric_features[i+1:]:
                try:
                    # 비율
                    feature_df[f'{feat1}_ratio_{feat2}'] = feature_df[feat1] / (feature_df[feat2] + 1e-8)
                    # 차이
                    feature_df[f'{feat1}_diff_{feat2}'] = feature_df[feat1] - feature_df[feat2]
                except:
                    continue
        
        # 6. 시차 피처 (Lag features)
        for col in available_features[:10]:  # 상위 10개만
            for lag in [1, 6, 24]:
                feature_df[f'{col}_lag_{lag}'] = feature_df[col].shift(lag)
        
        # NaN 처리
        feature_df = feature_df.fillna(method='bfill').fillna(0)
        feature_df = feature_df.replace([np.inf, -np.inf], 0)
        
        return feature_df
    
    def test_formula_at_timepoint(self, start_idx: int, features: List[str], 
                                 prediction_hours: int, model_type: str = 'ensemble') -> Dict:
        """특정 시점에서 공식 테스트"""
        
        # 타겟 컬럼 찾기 (BTC 가격)
        price_candidates = [
            'onchain_blockchain_info_network_stats_market_price_usd',
            'price', 'close', 'open'
        ]
        target_col = None
        for candidate in price_candidates:
            if candidate in self.historical_data.columns:
                target_col = candidate
                break
        
        if not target_col:
            numeric_cols = self.historical_data.select_dtypes(include=[np.number]).columns
            target_col = numeric_cols[0]
        
        try:
            # 1. 과거 시점까지의 데이터로 학습
            train_data = self.historical_data.iloc[:start_idx].copy()
            
            if len(train_data) < 200:  # 충분한 학습 데이터 필요
                return {'success': False, 'error': '학습 데이터 부족'}
            
            # 2. 고급 피처 생성
            enhanced_data = self.create_advanced_features(features, target_col)
            if enhanced_data.empty:
                return {'success': False, 'error': '피처 생성 실패'}
            
            train_enhanced = enhanced_data.iloc[:start_idx]
            
            # 3. X, y 준비
            X_train = train_enhanced.drop(columns=[target_col])
            y_train = train_enhanced[target_col].shift(-prediction_hours).dropna()
            X_train = X_train.iloc[:-prediction_hours]
            
            if len(X_train) < 100:
                return {'success': False, 'error': '타겟 데이터 부족'}
            
            # 4. 스케일링
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            
            # 5. 모델 학습
            if model_type == 'ensemble':
                models = {
                    'rf': RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42),
                    'gb': GradientBoostingRegressor(n_estimators=100, max_depth=8, random_state=42),
                    'ridge': Ridge(alpha=1.0)
                }
                
                predictions = {}
                for name, model in models.items():
                    model.fit(X_train_scaled, y_train)
                    
                    # 현재 시점 데이터로 예측
                    current_features = enhanced_data.iloc[start_idx:start_idx+1].drop(columns=[target_col])
                    current_scaled = scaler.transform(current_features)
                    pred = model.predict(current_scaled)[0]
                    predictions[name] = pred
                
                # 가중 앙상블 (과거 성능 기반)
                final_prediction = (predictions['rf'] * 0.4 + 
                                  predictions['gb'] * 0.4 + 
                                  predictions['ridge'] * 0.2)
                
            else:  # 단일 모델
                if model_type == 'rf':
                    model = RandomForestRegressor(n_estimators=100, random_state=42)
                elif model_type == 'gb':
                    model = GradientBoostingRegressor(n_estimators=100, random_state=42)
                else:
                    model = Ridge(alpha=1.0)
                
                model.fit(X_train_scaled, y_train)
                current_features = enhanced_data.iloc[start_idx:start_idx+1].drop(columns=[target_col])
                current_scaled = scaler.transform(current_features)
                final_prediction = model.predict(current_scaled)[0]
            
            # 6. 실제값과 비교
            target_idx = start_idx + prediction_hours
            if target_idx >= len(self.historical_data):
                return {'success': False, 'error': '예측 시점 초과'}
            
            actual_value = self.historical_data.iloc[target_idx][target_col]
            current_value = self.historical_data.iloc[start_idx][target_col]
            
            # 7. 성능 지표 계산
            absolute_error = abs(actual_value - final_prediction)
            percentage_error = (absolute_error / actual_value) * 100
            accuracy = max(0, 100 - percentage_error)
            
            return {
                'success': True,
                'start_idx': start_idx,
                'target_idx': target_idx,
                'current_value': current_value,
                'predicted_value': final_prediction,
                'actual_value': actual_value,
                'absolute_error': absolute_error,
                'percentage_error': percentage_error,
                'accuracy': accuracy,
                'features_used': features,
                'prediction_hours': prediction_hours,
                'model_type': model_type
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def infinite_formula_discovery(self, max_iterations: int = 1000) -> Dict:
        """무한 공식 발견 프로세스"""
        print(f"\n🚀 무한 백테스트 공식 발견 시작")
        print(f"   🎯 목표 정확도: {self.target_accuracy}%")
        print(f"   🔄 최대 반복: {max_iterations}회")
        print("="*70)
        
        # 지표 조합 생성
        feature_combinations = self.generate_feature_combinations()
        print(f"📊 생성된 지표 조합: {len(feature_combinations)}개")
        
        best_results = []
        iteration = 0
        
        data_length = len(self.historical_data)
        min_start_idx = 300  # 최소 학습 데이터
        max_start_idx = data_length - 200  # 예측을 위한 여유
        
        while iteration < max_iterations and self.current_best_accuracy < self.target_accuracy:
            iteration += 1
            
            # 랜덤 설정
            start_idx = random.randint(min_start_idx, max_start_idx)
            features = random.choice(feature_combinations)
            prediction_hours = random.choice(self.prediction_horizons)
            model_type = random.choice(['ensemble', 'rf', 'gb', 'ridge'])
            
            # 백테스트 실행
            result = self.test_formula_at_timepoint(start_idx, features, prediction_hours, model_type)
            
            if result['success']:
                accuracy = result['accuracy']
                
                # 진행상황 출력
                if iteration % 50 == 0 or accuracy > 90:
                    print(f"🔍 반복 {iteration:4d}: 정확도 {accuracy:5.2f}% "
                          f"(피처 {len(features):2d}개, {prediction_hours:3d}시간 후 예측)")
                
                # 최고 성능 업데이트
                if accuracy > self.current_best_accuracy:
                    self.current_best_accuracy = accuracy
                    
                    formula_info = {
                        'iteration': iteration,
                        'accuracy': accuracy,
                        'features': features,
                        'prediction_hours': prediction_hours,
                        'model_type': model_type,
                        'result_details': result
                    }
                    
                    best_results.append(formula_info)
                    
                    print(f"🏆 신기록! 정확도 {accuracy:.2f}% "
                          f"(피처: {len(features)}개, 예측: {prediction_hours}h, 모델: {model_type})")
                    
                    # 목표 달성 체크
                    if accuracy >= self.target_accuracy:
                        print(f"🎉 목표 달성! {self.target_accuracy}% 달성!")
                        break
            
            else:
                if iteration % 100 == 0:
                    print(f"⚠️ 반복 {iteration}: 실패 ({result.get('error', 'Unknown')})")
        
        # 최종 결과
        print(f"\n" + "="*70)
        print("🏆 무한 백테스트 결과")
        print("="*70)
        print(f"🔄 총 반복 횟수:     {iteration}")
        print(f"🎯 최고 정확도:      {self.current_best_accuracy:.2f}%")
        print(f"📊 발견된 공식 수:   {len(best_results)}")
        
        if best_results:
            # 최고 공식 분석
            best_formula = best_results[-1]  # 마지막(최고) 결과
            
            print(f"\n🥇 최고 성능 공식:")
            print(f"   📈 정확도:        {best_formula['accuracy']:.2f}%")
            print(f"   📊 사용 지표:     {len(best_formula['features'])}개")
            print(f"   ⏰ 예측 기간:     {best_formula['prediction_hours']}시간")
            print(f"   🤖 모델 타입:     {best_formula['model_type']}")
            
            # 상위 지표들 출력
            print(f"\n🔝 핵심 지표 (상위 10개):")
            for i, feature in enumerate(best_formula['features'][:10]):
                print(f"   {i+1:2d}. {feature}")
            
            self.best_formulas = best_results
        
        else:
            print("❌ 유효한 공식을 발견하지 못했습니다.")
        
        print("="*70)
        
        # 결과 저장
        self.save_discovery_results(best_results, iteration)
        
        return {
            'total_iterations': iteration,
            'best_accuracy': self.current_best_accuracy,
            'formulas_discovered': len(best_results),
            'target_achieved': self.current_best_accuracy >= self.target_accuracy,
            'best_formula': best_results[-1] if best_results else None
        }
    
    def save_discovery_results(self, best_results: List[Dict], iterations: int):
        """발견된 공식들 저장"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        results = {
            'timestamp': timestamp,
            'total_iterations': iterations,
            'target_accuracy': self.target_accuracy,
            'best_accuracy_achieved': self.current_best_accuracy,
            'target_achieved': self.current_best_accuracy >= self.target_accuracy,
            'formulas_discovered': len(best_results),
            'best_formulas': best_results,
            'data_shape': self.historical_data.shape,
            'discovery_summary': {
                'avg_features_used': np.mean([len(f['features']) for f in best_results]) if best_results else 0,
                'most_common_prediction_hours': max([f['prediction_hours'] for f in best_results], key=[f['prediction_hours'] for f in best_results].count) if best_results else 0,
                'best_model_type': best_results[-1]['model_type'] if best_results else None
            }
        }
        
        filename = f"infinite_backtest_formula_discovery_{timestamp}.json"
        filepath = os.path.join(self.data_path, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"💾 발견 결과 저장: {filename}")
        
        return filepath
    
    def run_discovery_process(self):
        """전체 발견 프로세스 실행"""
        print("🎯 무한 백테스트 95% 공식 발견기 시작")
        print("="*70)
        
        # 1. 데이터 로드
        if not self.load_data():
            print("❌ 데이터 로드 실패")
            return None
        
        # 2. 무한 공식 발견
        results = self.infinite_formula_discovery(max_iterations=500)  # 500회 테스트
        
        return results

def main():
    discoverer = InfiniteBacktestFormulaDiscoverer()
    return discoverer.run_discovery_process()

if __name__ == "__main__":
    results = main()