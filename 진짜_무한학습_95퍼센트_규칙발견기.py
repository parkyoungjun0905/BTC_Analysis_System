#!/usr/bin/env python3
"""
🔥 진짜 무한학습 95% 규칙 발견기
- 다방면 예측 시도 → 실패 분석 → 학습 → 무한 반복
- 공통 규칙/패턴 자동 발견 및 안내
- 95% 성공률 달성까지 진화하는 AI 학습 시스템
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge, ElasticNet, Lasso, BayesianRidge
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score
import itertools
import random
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

class InfiniteLearningRuleDiscoverer:
    """진짜 무한학습 규칙 발견기"""
    
    def __init__(self):
        self.data_path = "/Users/parkyoungjun/Desktop/BTC_Analysis_System"
        self.historical_data = None
        
        # 학습 기록 저장소
        self.all_predictions = []  # 모든 예측 시도 기록
        self.success_patterns = []  # 성공한 패턴들
        self.failure_patterns = []  # 실패한 패턴들
        self.discovered_rules = []  # 발견된 공통 규칙들
        
        # 성능 추적
        self.total_attempts = 0
        self.success_count = 0
        self.current_success_rate = 0.0
        self.target_success_rate = 95.0
        
        # 동적 학습 파라미터
        self.learning_weights = defaultdict(float)  # 지표별 학습된 가중치
        self.pattern_memory = defaultdict(list)    # 패턴별 성공/실패 기록
        self.market_regime_rules = {}              # 시장 상황별 규칙
        
        # 다방면 예측 전략들
        self.prediction_strategies = {
            'technical_focus': {'weight': 1.0, 'success': 0, 'attempts': 0},
            'onchain_focus': {'weight': 1.0, 'success': 0, 'attempts': 0},
            'macro_focus': {'weight': 1.0, 'success': 0, 'attempts': 0},
            'sentiment_focus': {'weight': 1.0, 'success': 0, 'attempts': 0},
            'hybrid_ensemble': {'weight': 1.0, 'success': 0, 'attempts': 0},
            'momentum_based': {'weight': 1.0, 'success': 0, 'attempts': 0},
            'mean_reversion': {'weight': 1.0, 'success': 0, 'attempts': 0}
        }
        
        print("🔥 진짜 무한학습 95% 규칙 발견기 초기화")
        print(f"🎯 목표: {self.target_success_rate}% 성공률 달성")
        
    def load_data(self) -> bool:
        """3개월 통합 데이터 로드"""
        print("\n📂 3개월 통합 데이터 로딩...")
        
        try:
            csv_path = os.path.join(self.data_path, "ai_optimized_3month_data", "ai_matrix_complete.csv")
            df = pd.read_csv(csv_path)
            
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.sort_values('timestamp').reset_index(drop=True)
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            self.historical_data = df[['timestamp'] + list(numeric_cols) if 'timestamp' in df.columns else list(numeric_cols)].copy()
            self.historical_data = self.historical_data.fillna(method='ffill').fillna(method='bfill').fillna(0)
            
            print(f"✅ 데이터 로드 완료: {self.historical_data.shape}")
            print(f"📊 사용 가능 지표: {len(numeric_cols)}개")
            
            return True
            
        except Exception as e:
            print(f"❌ 데이터 로드 실패: {e}")
            return False
    
    def categorize_all_indicators(self) -> Dict[str, List[str]]:
        """1,369개 지표를 카테고리별 완전 분류"""
        all_cols = [col for col in self.historical_data.columns if col != 'timestamp']
        
        categories = {
            'price_basic': [],
            'volume': [], 
            'technical': [],
            'onchain_whale': [],
            'onchain_hodl': [],
            'onchain_network': [],
            'derivatives': [],
            'macro': [],
            'sentiment': [],
            'correlation': [],
            'volatility': [],
            'pattern': [],
            'support_resistance': [],
            'market_structure': [],
            'liquidity': [],
            'others': []
        }
        
        # 키워드 기반 분류
        keywords = {
            'price_basic': ['price', 'open', 'high', 'low', 'close'],
            'volume': ['volume', 'trade_volume', 'transaction_volume'],
            'technical': ['rsi', 'macd', 'bollinger', 'sma', 'ema', 'stoch', 'williams'],
            'onchain_whale': ['whale', 'large_tx', 'top100'],
            'onchain_hodl': ['hodl', 'lth', 'sth', 'age'],
            'onchain_network': ['hash', 'difficulty', 'miner', 'blockchain_info'],
            'derivatives': ['funding', 'basis', 'futures', 'options', 'perpetual'],
            'macro': ['dxy', 'gold', 'nasdaq', 'spx', 'vix', 'crude', 'bonds'],
            'sentiment': ['fear_greed', 'social', 'sentiment'],
            'correlation': ['correlation', 'corr'],
            'volatility': ['volatility', 'vol', 'atr'],
            'pattern': ['pattern', 'flag', 'triangle', 'head_shoulders'],
            'support_resistance': ['support', 'resistance', 'level'],
            'market_structure': ['market_structure', 'cross', 'lead_lag'],
            'liquidity': ['liquidity', 'depth', 'spread', 'slippage']
        }
        
        # 분류 실행
        for col in all_cols:
            categorized = False
            for category, words in keywords.items():
                if any(word.lower() in col.lower() for word in words):
                    categories[category].append(col)
                    categorized = True
                    break
            
            if not categorized:
                categories['others'].append(col)
        
        # 분류 결과 출력
        print(f"\n📊 지표 카테고리별 분류:")
        for category, indicators in categories.items():
            if indicators:
                print(f"  {category:20s}: {len(indicators):4d}개")
        
        return categories
    
    def generate_diverse_prediction_strategy(self) -> Dict:
        """다방면 예측 전략 생성"""
        categories = self.categorize_all_indicators()
        
        strategy_types = [
            'technical_focus', 'onchain_focus', 'macro_focus', 'sentiment_focus',
            'hybrid_ensemble', 'momentum_based', 'mean_reversion'
        ]
        
        selected_strategy = random.choice(strategy_types)
        
        # 전략별 지표 선택
        if selected_strategy == 'technical_focus':
            selected_indicators = (
                categories['technical'][:10] + 
                categories['price_basic'][:5] + 
                categories['volume'][:3]
            )
            model_type = 'rf'
            
        elif selected_strategy == 'onchain_focus':
            selected_indicators = (
                categories['onchain_whale'][:8] + 
                categories['onchain_hodl'][:8] + 
                categories['onchain_network'][:4]
            )
            model_type = 'gb'
            
        elif selected_strategy == 'macro_focus':
            selected_indicators = (
                categories['macro'][:8] + 
                categories['correlation'][:8] + 
                categories['price_basic'][:4]
            )
            model_type = 'ridge'
            
        elif selected_strategy == 'sentiment_focus':
            selected_indicators = (
                categories['sentiment'][:5] + 
                categories['volume'][:5] + 
                categories['social'] if 'social' in categories else categories['others'][:5]
            )
            model_type = 'elastic'
            
        elif selected_strategy == 'hybrid_ensemble':
            # 모든 카테고리에서 균형있게
            selected_indicators = []
            for cat_name, indicators in categories.items():
                if indicators and cat_name != 'others':
                    selected_indicators.extend(indicators[:3])
            model_type = 'ensemble'
            
        elif selected_strategy == 'momentum_based':
            selected_indicators = (
                categories['technical'][:6] + 
                categories['volatility'][:6] + 
                categories['pattern'][:4] + 
                categories['price_basic'][:4]
            )
            model_type = 'gb'
            
        else:  # mean_reversion
            selected_indicators = (
                categories['support_resistance'][:8] + 
                categories['technical'][:6] + 
                categories['onchain_hodl'][:6]
            )
            model_type = 'ridge'
        
        # 실제 존재하는 지표만 필터링
        available_indicators = [ind for ind in selected_indicators if ind in self.historical_data.columns]
        
        if not available_indicators:
            available_indicators = random.sample(
                [col for col in self.historical_data.columns if col != 'timestamp'], 
                min(15, len(self.historical_data.columns)-1)
            )
        
        return {
            'strategy_type': selected_strategy,
            'indicators': available_indicators[:20],  # 최대 20개
            'model_type': model_type,
            'prediction_hours': random.choice([1, 6, 12, 24, 48, 72]),
            'market_regime': self.detect_market_regime(),
            'preprocessing': random.choice(['robust', 'standard', 'minmax'])
        }
    
    def detect_market_regime(self) -> str:
        """현재 시장 상황 감지"""
        # 간단한 시장 상황 분류
        regimes = ['bull_trend', 'bear_trend', 'sideways', 'high_volatility', 'low_volatility']
        return random.choice(regimes)  # 임시로 랜덤 (실제로는 가격 분석 기반)
    
    def execute_prediction_attempt(self, start_idx: int, strategy: Dict) -> Dict:
        """단일 예측 시도 실행"""
        try:
            # 타겟 컬럼 찾기
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
            
            # 학습 데이터 준비
            train_data = self.historical_data.iloc[:start_idx]
            if len(train_data) < 100:
                return {'success': False, 'error': '학습 데이터 부족'}
            
            # 피처 준비
            X_train = train_data[strategy['indicators']]
            prediction_hours = strategy['prediction_hours']
            y_train = train_data[target_col].shift(-prediction_hours).dropna()
            X_train = X_train.iloc[:-prediction_hours]
            
            if len(X_train) < 50:
                return {'success': False, 'error': '타겟 데이터 부족'}
            
            # 전처리
            if strategy['preprocessing'] == 'robust':
                scaler = RobustScaler()
            elif strategy['preprocessing'] == 'standard':
                scaler = StandardScaler()
            else:
                scaler = MinMaxScaler()
            
            X_train_scaled = scaler.fit_transform(X_train)
            
            # 모델 학습
            if strategy['model_type'] == 'rf':
                model = RandomForestRegressor(n_estimators=50, random_state=42)
            elif strategy['model_type'] == 'gb':
                model = GradientBoostingRegressor(n_estimators=50, random_state=42)
            elif strategy['model_type'] == 'ridge':
                model = Ridge(alpha=1.0)
            elif strategy['model_type'] == 'elastic':
                model = ElasticNet(alpha=1.0)
            elif strategy['model_type'] == 'ensemble':
                # 간단한 앙상블
                models = [
                    RandomForestRegressor(n_estimators=30, random_state=42),
                    GradientBoostingRegressor(n_estimators=30, random_state=42)
                ]
                predictions = []
                for m in models:
                    m.fit(X_train_scaled, y_train)
                    current_features = self.historical_data.iloc[start_idx:start_idx+1][strategy['indicators']]
                    current_scaled = scaler.transform(current_features)
                    pred = m.predict(current_scaled)[0]
                    predictions.append(pred)
                
                final_prediction = np.mean(predictions)
                
                # 실제값과 비교
                target_idx = start_idx + prediction_hours
                if target_idx >= len(self.historical_data):
                    return {'success': False, 'error': '예측 시점 초과'}
                
                actual_value = self.historical_data.iloc[target_idx][target_col]
                error_pct = abs(actual_value - final_prediction) / actual_value * 100
                accuracy = max(0, 100 - error_pct)
                
                return {
                    'success': True,
                    'accuracy': accuracy,
                    'predicted': final_prediction,
                    'actual': actual_value,
                    'error_pct': error_pct,
                    'strategy': strategy
                }
            
            # 단일 모델의 경우
            model.fit(X_train_scaled, y_train)
            
            # 예측
            current_features = self.historical_data.iloc[start_idx:start_idx+1][strategy['indicators']]
            current_scaled = scaler.transform(current_features)
            prediction = model.predict(current_scaled)[0]
            
            # 검증
            target_idx = start_idx + prediction_hours
            if target_idx >= len(self.historical_data):
                return {'success': False, 'error': '예측 시점 초과'}
            
            actual_value = self.historical_data.iloc[target_idx][target_col]
            error_pct = abs(actual_value - prediction) / actual_value * 100
            accuracy = max(0, 100 - error_pct)
            
            return {
                'success': True,
                'accuracy': accuracy,
                'predicted': prediction,
                'actual': actual_value,
                'error_pct': error_pct,
                'strategy': strategy
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def analyze_prediction_result(self, result: Dict, attempt_num: int):
        """예측 결과 분석 및 학습"""
        self.total_attempts += 1
        
        if result['success']:
            accuracy = result['accuracy']
            strategy = result['strategy']
            
            # 성공/실패 기준 (80% 이상을 성공으로 간주)
            is_success = accuracy >= 80.0
            
            if is_success:
                self.success_count += 1
                self.success_patterns.append(result)
                
                # 성공한 전략의 가중치 증가
                strategy_type = strategy['strategy_type']
                self.prediction_strategies[strategy_type]['success'] += 1
                
                # 성공한 지표들의 가중치 증가
                for indicator in strategy['indicators']:
                    self.learning_weights[indicator] += 0.1
                
                print(f"✅ 시도 {attempt_num}: 성공! 정확도 {accuracy:.1f}% (전략: {strategy_type})")
                
            else:
                self.failure_patterns.append(result)
                
                # 실패한 지표들의 가중치 감소
                for indicator in strategy['indicators']:
                    self.learning_weights[indicator] -= 0.05
                
                if attempt_num % 100 == 0:
                    print(f"❌ 시도 {attempt_num}: 실패 정확도 {accuracy:.1f}% (전략: {strategy['strategy_type']})")
            
            # 전략별 시도 횟수 증가
            self.prediction_strategies[strategy['strategy_type']]['attempts'] += 1
            
        else:
            if attempt_num % 100 == 0:
                print(f"⚠️ 시도 {attempt_num}: 오류 - {result.get('error', 'Unknown')}")
        
        # 성공률 업데이트
        if self.total_attempts > 0:
            self.current_success_rate = (self.success_count / self.total_attempts) * 100
        
        # 주기적으로 규칙 발견 시도
        if attempt_num % 500 == 0:
            self.discover_common_rules()
            self.print_progress_report(attempt_num)
    
    def discover_common_rules(self):
        """공통 규칙 발견"""
        if len(self.success_patterns) < 10:
            return
        
        print(f"\n🔍 공통 규칙 분석 중... (성공 사례 {len(self.success_patterns)}개 기반)")
        
        # 규칙 1: 가장 효과적인 지표들
        indicator_success_count = defaultdict(int)
        for pattern in self.success_patterns:
            for indicator in pattern['strategy']['indicators']:
                indicator_success_count[indicator] += 1
        
        top_indicators = sorted(indicator_success_count.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # 규칙 2: 가장 효과적인 전략 타입
        strategy_success_rate = {}
        for strategy_type, stats in self.prediction_strategies.items():
            if stats['attempts'] > 0:
                success_rate = (stats['success'] / stats['attempts']) * 100
                strategy_success_rate[strategy_type] = success_rate
        
        best_strategies = sorted(strategy_success_rate.items(), key=lambda x: x[1], reverse=True)
        
        # 규칙 3: 최적 예측 시간
        prediction_hours_success = defaultdict(list)
        for pattern in self.success_patterns:
            hours = pattern['strategy']['prediction_hours']
            accuracy = pattern['accuracy']
            prediction_hours_success[hours].append(accuracy)
        
        optimal_hours = {}
        for hours, accuracies in prediction_hours_success.items():
            optimal_hours[hours] = np.mean(accuracies)
        
        best_hours = sorted(optimal_hours.items(), key=lambda x: x[1], reverse=True)
        
        # 발견된 규칙 저장
        new_rule = {
            'discovery_time': datetime.now(),
            'success_cases': len(self.success_patterns),
            'total_attempts': self.total_attempts,
            'top_indicators': top_indicators,
            'best_strategies': best_strategies,
            'optimal_prediction_hours': best_hours,
            'current_success_rate': self.current_success_rate
        }
        
        self.discovered_rules.append(new_rule)
        
        # 사용자에게 규칙 안내
        self.report_discovered_rules(new_rule)
    
    def report_discovered_rules(self, rule: Dict):
        """발견된 규칙을 사용자에게 안내"""
        print(f"\n🎯 발견된 공통 규칙 (성공률 {rule['current_success_rate']:.1f}%)")
        print("="*60)
        
        print(f"📊 최고 성과 지표 TOP 5:")
        for i, (indicator, count) in enumerate(rule['top_indicators'][:5]):
            print(f"  {i+1}. {indicator[:50]}... (성공 {count}회)")
        
        print(f"\n🎯 최고 성과 전략 TOP 3:")
        for i, (strategy, rate) in enumerate(rule['best_strategies'][:3]):
            print(f"  {i+1}. {strategy:20s} 성공률 {rate:.1f}%")
        
        print(f"\n⏰ 최적 예측 시간 TOP 3:")
        for i, (hours, avg_acc) in enumerate(rule['optimal_prediction_hours'][:3]):
            print(f"  {i+1}. {hours:2d}시간 후 예측: 평균 정확도 {avg_acc:.1f}%")
        
        print("="*60)
    
    def print_progress_report(self, attempt_num: int):
        """진행 상황 보고"""
        print(f"\n📈 진행 상황 보고 (시도 {attempt_num}회)")
        print(f"🎯 현재 성공률: {self.current_success_rate:.2f}% (목표: {self.target_success_rate}%)")
        print(f"✅ 성공: {self.success_count}회, ❌ 실패: {self.total_attempts - self.success_count}회")
        
        if self.current_success_rate >= self.target_success_rate:
            print(f"🎉 목표 달성! {self.target_success_rate}% 성공률 달성!")
            return True
        else:
            remaining = self.target_success_rate - self.current_success_rate
            print(f"⚠️ 목표까지 +{remaining:.2f}% 더 필요")
            return False
    
    def run_infinite_learning(self, max_attempts: int = 10000):
        """무한 학습 실행"""
        print(f"\n🚀 무한 학습 시작 (최대 {max_attempts}회)")
        print(f"🎯 목표: {self.target_success_rate}% 성공률 달성")
        print("="*70)
        
        data_length = len(self.historical_data)
        min_start = 200
        max_start = data_length - 100
        
        for attempt in range(1, max_attempts + 1):
            # 랜덤 시점 선택
            start_idx = random.randint(min_start, max_start)
            
            # 다방면 예측 전략 생성
            strategy = self.generate_diverse_prediction_strategy()
            
            # 예측 시도
            result = self.execute_prediction_attempt(start_idx, strategy)
            
            # 결과 분석 및 학습
            self.analyze_prediction_result(result, attempt)
            
            # 목표 달성 체크
            if attempt % 1000 == 0:
                if self.print_progress_report(attempt):
                    break
        
        # 최종 결과
        self.print_final_results()
        self.save_learning_results()
    
    def print_final_results(self):
        """최종 결과 출력"""
        print(f"\n" + "="*70)
        print("🏆 무한 학습 최종 결과")
        print("="*70)
        print(f"🔄 총 시도 횟수:     {self.total_attempts:,}")
        print(f"✅ 성공 횟수:       {self.success_count:,}")
        print(f"🎯 최종 성공률:     {self.current_success_rate:.2f}%")
        print(f"📊 발견된 규칙 수:   {len(self.discovered_rules)}")
        
        if self.current_success_rate >= self.target_success_rate:
            print(f"🎉 목표 달성! ({self.target_success_rate}% 이상)")
        else:
            print(f"⚠️ 목표 미달성 (목표: {self.target_success_rate}%)")
        
        # 최종 발견된 규칙들 요약
        if self.discovered_rules:
            final_rule = self.discovered_rules[-1]
            print(f"\n🎯 최종 발견 규칙:")
            print(f"📈 최고 지표: {final_rule['top_indicators'][0][0] if final_rule['top_indicators'] else 'N/A'}")
            print(f"🏆 최고 전략: {final_rule['best_strategies'][0][0] if final_rule['best_strategies'] else 'N/A'}")
            print(f"⏰ 최적 시간: {final_rule['optimal_prediction_hours'][0][0] if final_rule['optimal_prediction_hours'] else 'N/A'}시간")
        
        print("="*70)
    
    def save_learning_results(self):
        """학습 결과 저장"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        results = {
            'timestamp': timestamp,
            'total_attempts': self.total_attempts,
            'success_count': self.success_count,
            'final_success_rate': self.current_success_rate,
            'target_achieved': self.current_success_rate >= self.target_success_rate,
            'discovered_rules': self.discovered_rules,
            'learning_weights': dict(self.learning_weights),
            'strategy_performance': self.prediction_strategies,
            'success_patterns_count': len(self.success_patterns),
            'failure_patterns_count': len(self.failure_patterns)
        }
        
        filename = f"infinite_learning_rules_{timestamp}.json"
        filepath = os.path.join(self.data_path, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"💾 학습 결과 저장: {filename}")
        return filepath
    
    def run_complete_system(self):
        """전체 시스템 실행"""
        print("🔥 진짜 무한학습 95% 규칙 발견기 시작")
        print("="*70)
        
        if not self.load_data():
            return None
        
        # 무한 학습 실행
        self.run_infinite_learning(max_attempts=5000)  # 5000회 시도
        
        return {
            'success_rate': self.current_success_rate,
            'total_attempts': self.total_attempts,
            'rules_discovered': len(self.discovered_rules),
            'target_achieved': self.current_success_rate >= self.target_success_rate
        }

def main():
    learner = InfiniteLearningRuleDiscoverer()
    return learner.run_complete_system()

if __name__ == "__main__":
    results = main()