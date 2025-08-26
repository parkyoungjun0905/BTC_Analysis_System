#!/usr/bin/env python3
"""
🧠 스마트 BTC 학습 시스템 (smart_learning_system.py)

구조:
1. 한 번 학습 → 결과 저장
2. 저장된 모델로 실시간 예측
3. 새로운 데이터로 점진적 학습 업데이트
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import logging
import sqlite3
from pathlib import Path

class SmartBTCLearningSystem:
    """효율적인 BTC 학습 시스템"""
    
    def __init__(self, data_path: str = "ai_optimized_3month_data/integrated_complete_data.json"):
        self.base_path = "/Users/parkyoungjun/Desktop/BTC_Analysis_System"
        self.data_path = os.path.join(self.base_path, data_path)
        
        # 학습 결과 저장 경로
        self.models_path = os.path.join(self.base_path, "trained_models")
        self.learning_db_path = os.path.join(self.base_path, "learning_database.db")
        os.makedirs(self.models_path, exist_ok=True)
        
        # 최고 성능 모델 경로
        self.best_model_path = os.path.join(self.models_path, "best_prediction_model.pkl")
        self.best_patterns_path = os.path.join(self.models_path, "best_analysis_patterns.json")
        
        self.setup_logging()
        self.setup_database()
        
        # 학습된 모델이 있는지 확인
        self.has_trained_model = os.path.exists(self.best_model_path)
        self.best_patterns = self.load_best_patterns()
        
        self.logger.info(f"🚀 스마트 학습 시스템 초기화 완료")
        if self.has_trained_model:
            self.logger.info("✅ 기존 학습 모델 발견 - 즉시 예측 가능")
        else:
            self.logger.info("🔄 새로운 학습 필요")
    
    def setup_logging(self):
        """로깅 설정"""
        log_path = os.path.join(self.base_path, "logs")
        os.makedirs(log_path, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(log_path, 'smart_learning_system.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def setup_database(self):
        """학습 결과 DB 설정"""
        conn = sqlite3.connect(self.learning_db_path)
        cursor = conn.cursor()
        
        # 학습 결과 테이블
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS learning_results (
                id INTEGER PRIMARY KEY,
                timestamp TEXT,
                pattern_name TEXT,
                accuracy REAL,
                precision_score REAL,
                pattern_config TEXT,
                performance_metrics TEXT
            )
        ''')
        
        # 최고 성능 패턴 테이블
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS best_patterns (
                pattern_name TEXT PRIMARY KEY,
                accuracy REAL,
                config TEXT,
                last_updated TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def load_best_patterns(self) -> Dict:
        """저장된 최고 성능 패턴들 로드"""
        if os.path.exists(self.best_patterns_path):
            try:
                with open(self.best_patterns_path, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {}
    
    def save_best_patterns(self, patterns: Dict):
        """최고 성능 패턴들 저장"""
        with open(self.best_patterns_path, 'w') as f:
            json.dump(patterns, f, indent=2)
    
    def run_learning_phase(self, max_tests: int = 50) -> Dict:
        """1단계: 학습 페이즈 (한 번만 실행)"""
        if self.has_trained_model and not self.should_retrain():
            self.logger.info("✅ 기존 학습 모델 사용 - 학습 단계 스킵")
            return {"status": "skipped", "reason": "model_exists"}
        
        self.logger.info(f"🎯 학습 페이즈 시작: 최적 분석 패턴 탐색")
        
        # 데이터 로드
        data = self.load_data()
        if not data:
            return {"error": "data_load_failed"}
        
        # 다양한 분석 패턴 테스트
        patterns_to_test = self.generate_analysis_patterns()
        best_results = {}
        
        for i, pattern in enumerate(patterns_to_test, 1):
            self.logger.info(f"📊 패턴 테스트 {i}/{len(patterns_to_test)}: {pattern['name']}")
            
            # 패턴 성능 평가
            performance = self.evaluate_pattern_performance(pattern, data, max_tests)
            
            if performance['accuracy'] > 0.7:  # 70% 이상만 저장
                best_results[pattern['name']] = {
                    'pattern': pattern,
                    'performance': performance,
                    'timestamp': datetime.now().isoformat()
                }
                
                self.logger.info(f"✅ 우수 패턴 발견: {pattern['name']} (정확도: {performance['accuracy']:.1%})")
        
        # 최고 성능 패턴 저장
        if best_results:
            self.save_learning_results(best_results)
            self.save_best_model(best_results)
            self.has_trained_model = True
            
            best_accuracy = max(r['performance']['accuracy'] for r in best_results.values())
            self.logger.info(f"🏆 학습 완료! 최고 정확도: {best_accuracy:.1%}")
            
            return {"status": "completed", "best_accuracy": best_accuracy, "patterns_found": len(best_results)}
        else:
            self.logger.warning("⚠️ 유효한 패턴을 찾지 못했습니다")
            return {"status": "failed", "reason": "no_valid_patterns"}
    
    def generate_analysis_patterns(self) -> List[Dict]:
        """다양한 분석 패턴 생성"""
        patterns = []
        
        # 패턴 1: 모멘텀 중심 분석
        patterns.append({
            'name': 'momentum_focus',
            'feature_weights': {
                'momentum_indicators': 0.6,
                'volume_indicators': 0.3,
                'price_patterns': 0.1
            },
            'prediction_logic': 'momentum_based',
            'threshold': 0.02  # 2% 변화 임계값
        })
        
        # 패턴 2: 볼륨 중심 분석
        patterns.append({
            'name': 'volume_focus',
            'feature_weights': {
                'volume_indicators': 0.5,
                'momentum_indicators': 0.3,
                'price_patterns': 0.2
            },
            'prediction_logic': 'volume_based',
            'threshold': 0.03
        })
        
        # 패턴 3: 패턴 중심 분석
        patterns.append({
            'name': 'pattern_focus',
            'feature_weights': {
                'price_patterns': 0.5,
                'momentum_indicators': 0.3,
                'volume_indicators': 0.2
            },
            'prediction_logic': 'pattern_based',
            'threshold': 0.015
        })
        
        # 패턴 4: 균형 분석
        patterns.append({
            'name': 'balanced_analysis',
            'feature_weights': {
                'momentum_indicators': 0.4,
                'volume_indicators': 0.3,
                'price_patterns': 0.3
            },
            'prediction_logic': 'ensemble',
            'threshold': 0.025
        })
        
        # 패턴 5: 고민감도 분석
        patterns.append({
            'name': 'high_sensitivity',
            'feature_weights': {
                'momentum_indicators': 0.7,
                'volume_indicators': 0.2,
                'price_patterns': 0.1
            },
            'prediction_logic': 'sensitive',
            'threshold': 0.005  # 0.5% 민감도
        })
        
        return patterns
    
    def evaluate_pattern_performance(self, pattern: Dict, data: Dict, max_tests: int) -> Dict:
        """패턴 성능 평가"""
        try:
            correct_predictions = 0
            total_predictions = 0
            price_errors = []
            
            # 시뮬레이션 실행
            for test_num in range(min(max_tests, 30)):  # 효율성을 위해 30회로 제한
                # 가상 시나리오 생성
                scenario = self.create_test_scenario(data, test_num)
                if not scenario:
                    continue
                
                # 패턴으로 예측
                prediction = self.predict_with_pattern(pattern, scenario['current_data'])
                if not prediction:
                    continue
                
                # 실제 결과와 비교
                actual_direction = scenario['actual_direction']
                predicted_direction = prediction['direction']
                
                total_predictions += 1
                
                # 방향성 정확도
                if predicted_direction == actual_direction:
                    correct_predictions += 1
                
                # 가격 오차
                price_error = abs(prediction['price'] - scenario['actual_price']) / scenario['actual_price']
                price_errors.append(price_error)
            
            if total_predictions == 0:
                return {'accuracy': 0, 'precision': 0, 'avg_price_error': 1}
            
            accuracy = correct_predictions / total_predictions
            avg_price_error = np.mean(price_errors) if price_errors else 1
            
            return {
                'accuracy': accuracy,
                'precision': accuracy,  # 간단화
                'avg_price_error': avg_price_error,
                'total_tests': total_predictions
            }
        
        except Exception as e:
            self.logger.error(f"❌ 패턴 평가 실패: {e}")
            return {'accuracy': 0, 'precision': 0, 'avg_price_error': 1}
    
    def create_test_scenario(self, data: Dict, test_num: int) -> Dict:
        """테스트 시나리오 생성"""
        try:
            # 간단한 시나리오 생성 (실제로는 시계열 데이터 사용)
            base_price = 65000 + (test_num * 1000)  # 가변 기준 가격
            
            # 가상의 현재 데이터
            current_data = {
                'price': base_price,
                'momentum_score': np.random.normal(0, 1),
                'volume_ratio': np.random.uniform(0.8, 1.5),
                'pattern_strength': np.random.uniform(0, 1)
            }
            
            # 가상의 미래 결과
            price_change = np.random.normal(0, 0.05)  # ±5% 변동
            actual_price = base_price * (1 + price_change)
            
            actual_direction = "UP" if price_change > 0.01 else "DOWN" if price_change < -0.01 else "SIDEWAYS"
            
            return {
                'current_data': current_data,
                'actual_price': actual_price,
                'actual_direction': actual_direction
            }
        
        except Exception as e:
            self.logger.error(f"❌ 시나리오 생성 실패: {e}")
            return None
    
    def predict_with_pattern(self, pattern: Dict, current_data: Dict) -> Dict:
        """패턴으로 예측 수행"""
        try:
            logic = pattern['prediction_logic']
            threshold = pattern['threshold']
            weights = pattern['feature_weights']
            
            # 각 로직별 예측
            if logic == 'momentum_based':
                signal = current_data['momentum_score'] * weights['momentum_indicators']
            elif logic == 'volume_based':
                signal = (current_data['volume_ratio'] - 1) * weights['volume_indicators']
            elif logic == 'pattern_based':
                signal = (current_data['pattern_strength'] - 0.5) * weights['price_patterns']
            else:  # ensemble
                signal = (current_data['momentum_score'] * weights['momentum_indicators'] + 
                         (current_data['volume_ratio'] - 1) * weights['volume_indicators'] +
                         (current_data['pattern_strength'] - 0.5) * weights['price_patterns'])
            
            # 방향 결정
            if signal > threshold:
                direction = "UP"
                price_multiplier = 1 + min(0.1, abs(signal))
            elif signal < -threshold:
                direction = "DOWN"
                price_multiplier = 1 - min(0.1, abs(signal))
            else:
                direction = "SIDEWAYS"
                price_multiplier = 1
            
            predicted_price = current_data['price'] * price_multiplier
            
            return {
                'direction': direction,
                'price': predicted_price,
                'confidence': min(1.0, abs(signal) * 2)
            }
        
        except Exception as e:
            self.logger.error(f"❌ 패턴 예측 실패: {e}")
            return None
    
    def predict_future_price(self, current_market_data: Dict = None) -> Dict:
        """2단계: 실시간 예측 (저장된 모델 사용)"""
        if not self.has_trained_model:
            return {"error": "no_trained_model", "message": "학습을 먼저 실행해주세요"}
        
        try:
            # 최고 성능 패턴 로드
            best_pattern = self.get_best_pattern()
            if not best_pattern:
                return {"error": "no_best_pattern"}
            
            # 현재 시장 데이터 준비
            if not current_market_data:
                current_market_data = self.get_current_market_data()
            
            # 최고 패턴으로 예측
            prediction = self.predict_with_pattern(best_pattern['pattern'], current_market_data)
            
            if prediction:
                prediction['pattern_used'] = best_pattern['name']
                prediction['pattern_accuracy'] = best_pattern['performance']['accuracy']
                prediction['prediction_timestamp'] = datetime.now().isoformat()
                
                self.logger.info(f"🎯 예측 완료: ${current_market_data.get('price', 65000):.0f} → ${prediction['price']:.0f} ({prediction['direction']})")
                
                return prediction
            else:
                return {"error": "prediction_failed"}
        
        except Exception as e:
            self.logger.error(f"❌ 실시간 예측 실패: {e}")
            return {"error": str(e)}
    
    def get_best_pattern(self) -> Dict:
        """최고 성능 패턴 조회"""
        patterns = self.load_best_patterns()
        if not patterns:
            return None
        
        # 정확도가 가장 높은 패턴 선택
        best_name = max(patterns.keys(), key=lambda k: patterns[k]['performance']['accuracy'])
        best_pattern = patterns[best_name]
        best_pattern['name'] = best_name
        
        return best_pattern
    
    def get_current_market_data(self) -> Dict:
        """현재 시장 데이터 가져오기"""
        # 실제로는 실시간 API에서 데이터를 가져와야 함
        return {
            'price': 65000,  # 현재 가격
            'momentum_score': 0.1,  # 모멘텀 점수
            'volume_ratio': 1.2,  # 볼륨 비율
            'pattern_strength': 0.7  # 패턴 강도
        }
    
    def should_retrain(self) -> bool:
        """재학습 필요 여부 판단"""
        if not os.path.exists(self.best_patterns_path):
            return True
        
        # 파일이 7일 이상 오래되면 재학습
        file_age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(self.best_patterns_path))
        return file_age.days > 7
    
    def save_learning_results(self, results: Dict):
        """학습 결과 DB에 저장"""
        conn = sqlite3.connect(self.learning_db_path)
        cursor = conn.cursor()
        
        for name, result in results.items():
            cursor.execute('''
                INSERT OR REPLACE INTO best_patterns 
                (pattern_name, accuracy, config, last_updated)
                VALUES (?, ?, ?, ?)
            ''', (
                name,
                result['performance']['accuracy'],
                json.dumps(result['pattern']),
                result['timestamp']
            ))
        
        conn.commit()
        conn.close()
        
        # JSON 파일로도 저장
        self.save_best_patterns(results)
    
    def save_best_model(self, results: Dict):
        """최고 모델 저장"""
        try:
            with open(self.best_model_path, 'wb') as f:
                pickle.dump(results, f)
            self.logger.info(f"✅ 최고 모델 저장 완료: {self.best_model_path}")
        except Exception as e:
            self.logger.error(f"❌ 모델 저장 실패: {e}")
    
    def load_data(self) -> Dict:
        """데이터 로드"""
        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"❌ 데이터 로드 실패: {e}")
            return {}

def main():
    """메인 실행 함수"""
    print("🧠 스마트 BTC 학습 시스템")
    print("="*50)
    
    system = SmartBTCLearningSystem()
    
    # 1단계: 학습 (필요시에만)
    print("1️⃣ 학습 페이즈 실행...")
    learning_result = system.run_learning_phase()
    
    if learning_result.get('status') == 'completed':
        print(f"✅ 학습 완료! 최고 정확도: {learning_result['best_accuracy']:.1%}")
    elif learning_result.get('status') == 'skipped':
        print("✅ 기존 모델 사용")
    else:
        print(f"❌ 학습 실패: {learning_result}")
        return
    
    # 2단계: 실시간 예측
    print("\n2️⃣ 실시간 예측 실행...")
    prediction = system.predict_future_price()
    
    if 'error' not in prediction:
        print(f"🎯 예측 결과:")
        print(f"   방향: {prediction['direction']}")
        print(f"   가격: ${prediction['price']:.2f}")
        print(f"   신뢰도: {prediction['confidence']:.1%}")
        print(f"   사용 패턴: {prediction['pattern_used']}")
    else:
        print(f"❌ 예측 실패: {prediction}")

if __name__ == "__main__":
    main()