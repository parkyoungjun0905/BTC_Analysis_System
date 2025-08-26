#!/usr/bin/env python3
"""
🎯 고급 모델 선택 및 앙상블 최적화 시스템
유전 알고리즘, 강화학습, 다목적 최적화를 활용한 동적 모델 선택

핵심 기능:
- 유전 알고리즘 기반 가중치 최적화
- 강화학습을 통한 동적 모델 선택
- 다목적 최적화 (정확도 vs 다양성)
- 온라인 적응형 앙상블
- 성능 일관성 모니터링
"""

import numpy as np
import pandas as pd
import json
import pickle
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
import sqlite3
from concurrent.futures import ThreadPoolExecutor
import logging

# 최적화 라이브러리
try:
    from deap import base, creator, tools, algorithms
    from scipy.optimize import differential_evolution, minimize
    OPTIMIZATION_AVAILABLE = True
except ImportError:
    OPTIMIZATION_AVAILABLE = False
    print("⚠️ 최적화 라이브러리 미설치 (deap, scipy)")

# 강화학습 라이브러리 (선택적)
try:
    import gym
    from gym import spaces
    REINFORCEMENT_AVAILABLE = True
except ImportError:
    REINFORCEMENT_AVAILABLE = False
    print("⚠️ OpenAI Gym 미설치 - 강화학습 비활성화")

# 머신러닝 라이브러리
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

class GeneticEnsembleOptimizer:
    """
    🧬 유전 알고리즘 기반 앙상블 최적화
    """
    
    def __init__(self, population_size: int = 50, generations: int = 100):
        self.population_size = population_size
        self.generations = generations
        
        if OPTIMIZATION_AVAILABLE:
            # DEAP 설정
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
            creator.create("Individual", list, fitness=creator.FitnessMax)
            
            self.toolbox = base.Toolbox()
        
        self.logger = logging.getLogger(__name__)

    def create_genetic_operators(self, n_models: int):
        """유전 연산자 생성"""
        if not OPTIMIZATION_AVAILABLE:
            return None
        
        # 가중치 범위: 0.0 ~ 1.0
        self.toolbox.register("attr_weight", np.random.random)
        self.toolbox.register("individual", tools.initRepeat, creator.Individual,
                             self.toolbox.attr_weight, n_models)
        self.toolbox.register("population", tools.initRepeat, list,
                             self.toolbox.individual)
        
        # 평가 함수 등록
        self.toolbox.register("evaluate", self.evaluate_ensemble)
        self.toolbox.register("mate", tools.cxBlend, alpha=0.5)
        self.toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.1)
        self.toolbox.register("select", tools.selTournament, tournsize=3)

    def normalize_weights(self, weights: List[float]) -> List[float]:
        """가중치 정규화"""
        weights = np.array(weights)
        weights = np.maximum(weights, 0)  # 음수 제거
        total = np.sum(weights)
        return (weights / total).tolist() if total > 0 else [1.0/len(weights)] * len(weights)

    def evaluate_ensemble(self, individual: List[float]) -> Tuple[float]:
        """개체 평가 함수"""
        weights = self.normalize_weights(individual)
        
        # 앙상블 예측 계산
        ensemble_pred = np.zeros(len(self.y_val))
        
        for i, (model_name, pred) in enumerate(self.model_predictions.items()):
            ensemble_pred += weights[i] * pred
        
        # 평가 메트릭 계산
        # 1. 방향성 정확도 (주요 목표)
        direction_actual = np.diff(self.y_val) > 0
        direction_pred = np.diff(ensemble_pred) > 0
        direction_accuracy = np.mean(direction_actual == direction_pred)
        
        # 2. R² 점수
        r2 = r2_score(self.y_val, ensemble_pred)
        r2_normalized = max(0, min(1, r2))
        
        # 3. 다양성 점수 (가중치 분산)
        diversity = 1 - np.var(weights)  # 균등할수록 높은 점수
        
        # 종합 점수 계산
        fitness = (
            0.6 * direction_accuracy +  # 방향성이 가장 중요
            0.3 * r2_normalized +       # 설명력
            0.1 * diversity            # 다양성
        )
        
        return (fitness,)

    def optimize_ensemble_weights(self, model_predictions: Dict[str, np.ndarray], 
                                y_val: np.ndarray) -> Dict[str, float]:
        """
        유전 알고리즘으로 앙상블 가중치 최적화
        
        Args:
            model_predictions: 모델별 예측값
            y_val: 검증 타겟
            
        Returns:
            Dict[str, float]: 최적화된 가중치
        """
        if not OPTIMIZATION_AVAILABLE:
            # 기본 균등 가중치 반환
            n_models = len(model_predictions)
            return {name: 1.0/n_models for name in model_predictions.keys()}
        
        self.model_predictions = model_predictions
        self.y_val = y_val
        
        n_models = len(model_predictions)
        self.create_genetic_operators(n_models)
        
        # 초기 개체군 생성
        population = self.toolbox.population(n=self.population_size)
        
        # 통계 설정
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)
        
        # 유전 알고리즘 실행
        population, logbook = algorithms.eaSimple(
            population, self.toolbox, 
            cxpb=0.7, mutpb=0.3, ngen=self.generations,
            stats=stats, verbose=False
        )
        
        # 최적 개체 선택
        best_individual = tools.selBest(population, 1)[0]
        best_weights = self.normalize_weights(best_individual)
        
        # 모델명과 매칭
        model_names = list(model_predictions.keys())
        optimal_weights = {}
        
        for i, name in enumerate(model_names):
            optimal_weights[name] = best_weights[i]
        
        # 성능 평가
        best_fitness = best_individual.fitness.values[0]
        
        self.logger.info(f"🧬 유전 알고리즘 최적화 완료 - 최적 적합도: {best_fitness:.4f}")
        
        return optimal_weights

class ReinforcementEnsembleAgent:
    """
    🎮 강화학습 기반 동적 앙상블 에이전트
    """
    
    def __init__(self, n_models: int, learning_rate: float = 0.01):
        self.n_models = n_models
        self.learning_rate = learning_rate
        
        # Q-테이블 (상태 x 액션)
        self.q_table = {}
        self.epsilon = 0.1  # 탐험 비율
        self.gamma = 0.95   # 할인 팩터
        
        # 성능 기록
        self.performance_history = []
        
        self.logger = logging.getLogger(__name__)

    def get_market_state(self, recent_prices: np.ndarray, 
                        model_performances: Dict[str, float]) -> str:
        """
        시장 상태 인코딩
        
        Args:
            recent_prices: 최근 가격 데이터
            model_performances: 모델별 최근 성능
            
        Returns:
            str: 인코딩된 상태
        """
        # 1. 시장 트렌드 (상승/하락/횡보)
        price_change = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
        if price_change > 0.02:
            trend = "bull"
        elif price_change < -0.02:
            trend = "bear"
        else:
            trend = "sideways"
        
        # 2. 변동성 (높음/보통/낮음)
        volatility = np.std(np.diff(recent_prices) / recent_prices[:-1])
        if volatility > 0.05:
            vol_state = "high"
        elif volatility > 0.02:
            vol_state = "medium"
        else:
            vol_state = "low"
        
        # 3. 모델 성능 상태 (좋음/보통/나쁨)
        avg_performance = np.mean(list(model_performances.values()))
        if avg_performance > 0.7:
            perf_state = "good"
        elif avg_performance > 0.5:
            perf_state = "medium"
        else:
            perf_state = "poor"
        
        return f"{trend}_{vol_state}_{perf_state}"

    def select_models(self, state: str, available_models: List[str]) -> List[str]:
        """
        현재 상태에서 모델 선택
        
        Args:
            state: 현재 시장 상태
            available_models: 사용 가능한 모델 목록
            
        Returns:
            List[str]: 선택된 모델들
        """
        # 상태가 처음 보는 경우 초기화
        if state not in self.q_table:
            self.q_table[state] = np.random.random(len(available_models))
        
        # ε-탐욕 정책
        if np.random.random() < self.epsilon:
            # 탐험: 랜덤 선택
            n_select = max(1, int(len(available_models) * 0.6))  # 60% 선택
            selected_indices = np.random.choice(
                len(available_models), size=n_select, replace=False
            )
        else:
            # 활용: Q값 기반 선택
            q_values = self.q_table[state]
            n_select = max(1, int(len(available_models) * 0.6))
            selected_indices = np.argsort(q_values)[-n_select:]  # 상위 모델들
        
        selected_models = [available_models[i] for i in selected_indices]
        return selected_models

    def update_q_values(self, prev_state: str, action_indices: List[int], 
                       reward: float, current_state: str):
        """Q값 업데이트"""
        if prev_state not in self.q_table:
            self.q_table[prev_state] = np.random.random(self.n_models)
        
        if current_state not in self.q_table:
            self.q_table[current_state] = np.random.random(self.n_models)
        
        # Q-learning 업데이트
        for action_idx in action_indices:
            current_q = self.q_table[prev_state][action_idx]
            max_future_q = np.max(self.q_table[current_state])
            
            new_q = current_q + self.learning_rate * (
                reward + self.gamma * max_future_q - current_q
            )
            
            self.q_table[prev_state][action_idx] = new_q
        
        # 탐험 비율 감소
        self.epsilon = max(0.01, self.epsilon * 0.995)

    def calculate_reward(self, ensemble_performance: float, 
                        individual_performances: List[float]) -> float:
        """보상 계산"""
        # 1. 앙상블 성능 보상
        performance_reward = ensemble_performance
        
        # 2. 개별 모델 대비 개선 보상
        max_individual = max(individual_performances) if individual_performances else 0
        improvement_reward = max(0, ensemble_performance - max_individual)
        
        # 3. 다양성 보상 (성능 분산이 클수록 좋음)
        diversity_reward = np.std(individual_performances) if len(individual_performances) > 1 else 0
        
        total_reward = (
            0.6 * performance_reward +
            0.3 * improvement_reward +
            0.1 * diversity_reward
        )
        
        return total_reward

class MultiObjectiveEnsembleOptimizer:
    """
    🎯 다목적 앙상블 최적화 (정확도 vs 다양성)
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def calculate_diversity_metrics(self, predictions: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        다양성 메트릭 계산
        
        Args:
            predictions: 모델별 예측값
            
        Returns:
            Dict[str, float]: 다양성 지표들
        """
        pred_matrix = np.array(list(predictions.values()))
        
        # 1. 상관관계 기반 다양성
        correlations = np.corrcoef(pred_matrix)
        avg_correlation = np.mean(correlations[np.triu_indices_from(correlations, k=1)])
        correlation_diversity = 1 - abs(avg_correlation)
        
        # 2. 분산 기반 다양성
        pred_variance = np.var(pred_matrix, axis=0)
        variance_diversity = np.mean(pred_variance)
        
        # 3. 거리 기반 다양성
        distances = []
        model_names = list(predictions.keys())
        for i in range(len(model_names)):
            for j in range(i+1, len(model_names)):
                dist = np.mean(np.abs(pred_matrix[i] - pred_matrix[j]))
                distances.append(dist)
        
        distance_diversity = np.mean(distances) if distances else 0
        
        return {
            'correlation_diversity': correlation_diversity,
            'variance_diversity': variance_diversity,
            'distance_diversity': distance_diversity,
            'combined_diversity': np.mean([correlation_diversity, 
                                         variance_diversity/np.max(pred_variance) if np.max(pred_variance) > 0 else 0,
                                         distance_diversity])
        }

    def pareto_optimal_selection(self, model_performances: Dict[str, Dict],
                               diversity_threshold: float = 0.3) -> List[str]:
        """
        파레토 최적 모델 선택
        
        Args:
            model_performances: 모델별 성능 지표
            diversity_threshold: 다양성 임계값
            
        Returns:
            List[str]: 파레토 최적 모델들
        """
        models = []
        accuracies = []
        diversities = []
        
        for name, perf in model_performances.items():
            if 'direction_accuracy' in perf:
                models.append(name)
                accuracies.append(perf['direction_accuracy'])
                
                # 다양성 점수 (임시로 r2 점수의 역수 사용)
                diversity_score = 1.0 - abs(perf.get('r2', 0))
                diversities.append(diversity_score)
        
        if len(models) < 2:
            return models
        
        # 파레토 프론트 찾기
        pareto_models = []
        
        for i, (model, acc, div) in enumerate(zip(models, accuracies, diversities)):
            is_dominated = False
            
            for j, (other_acc, other_div) in enumerate(zip(accuracies, diversities)):
                if i != j:
                    # 다른 모델이 이 모델을 지배하는지 확인
                    if (other_acc >= acc and other_div >= div and 
                        (other_acc > acc or other_div > div)):
                        is_dominated = True
                        break
            
            if not is_dominated:
                pareto_models.append(model)
        
        return pareto_models

class OnlineEnsembleAdapter:
    """
    🔄 온라인 적응형 앙상블
    """
    
    def __init__(self, adaptation_window: int = 100):
        self.adaptation_window = adaptation_window
        self.recent_performances = {}
        self.weight_history = []
        
        self.logger = logging.getLogger(__name__)

    def update_model_performance(self, model_name: str, accuracy: float, 
                               timestamp: datetime):
        """모델 성능 업데이트"""
        if model_name not in self.recent_performances:
            self.recent_performances[model_name] = []
        
        self.recent_performances[model_name].append({
            'accuracy': accuracy,
            'timestamp': timestamp
        })
        
        # 윈도우 크기 유지
        if len(self.recent_performances[model_name]) > self.adaptation_window:
            self.recent_performances[model_name] = \
                self.recent_performances[model_name][-self.adaptation_window:]

    def adaptive_weight_adjustment(self, current_weights: Dict[str, float]) -> Dict[str, float]:
        """
        적응형 가중치 조정
        
        Args:
            current_weights: 현재 가중치
            
        Returns:
            Dict[str, float]: 조정된 가중치
        """
        adjusted_weights = current_weights.copy()
        
        # 최근 성능 기반 조정
        for model_name in current_weights.keys():
            if (model_name in self.recent_performances and 
                len(self.recent_performances[model_name]) >= 10):
                
                recent_scores = [p['accuracy'] for p in 
                               self.recent_performances[model_name][-10:]]
                
                # 성능 트렌드 계산
                if len(recent_scores) >= 5:
                    recent_avg = np.mean(recent_scores[-5:])
                    older_avg = np.mean(recent_scores[:-5])
                    trend = (recent_avg - older_avg) / older_avg if older_avg > 0 else 0
                    
                    # 가중치 조정 (±20% 범위)
                    adjustment = min(0.2, max(-0.2, trend))
                    adjusted_weights[model_name] *= (1 + adjustment)
        
        # 정규화
        total_weight = sum(adjusted_weights.values())
        if total_weight > 0:
            adjusted_weights = {name: weight/total_weight 
                              for name, weight in adjusted_weights.items()}
        
        # 가중치 변화 기록
        self.weight_history.append({
            'timestamp': datetime.now(),
            'weights': adjusted_weights.copy()
        })
        
        return adjusted_weights

    def detect_concept_drift(self, recent_accuracies: List[float], 
                           window_size: int = 20) -> bool:
        """
        개념 이동 감지
        
        Args:
            recent_accuracies: 최근 정확도들
            window_size: 비교 윈도우 크기
            
        Returns:
            bool: 개념 이동 감지 여부
        """
        if len(recent_accuracies) < 2 * window_size:
            return False
        
        # 최근 윈도우와 이전 윈도우 비교
        recent_window = recent_accuracies[-window_size:]
        older_window = recent_accuracies[-2*window_size:-window_size]
        
        # t-검정 (간단한 버전)
        recent_mean = np.mean(recent_window)
        older_mean = np.mean(older_window)
        
        # 성능이 10% 이상 저하되면 개념 이동으로 간주
        performance_drop = (older_mean - recent_mean) / older_mean if older_mean > 0 else 0
        
        if performance_drop > 0.1:
            self.logger.warning(f"🚨 개념 이동 감지 - 성능 저하: {performance_drop:.2%}")
            return True
        
        return False

class AdvancedModelSelectionSystem:
    """
    🎯 고급 모델 선택 및 최적화 통합 시스템
    """
    
    def __init__(self):
        self.genetic_optimizer = GeneticEnsembleOptimizer()
        self.rl_agent = None
        self.multi_objective_optimizer = MultiObjectiveEnsembleOptimizer()
        self.online_adapter = OnlineEnsembleAdapter()
        
        # 성능 추적
        self.optimization_history = []
        
        # 로깅 설정
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('model_selection_optimizer.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def comprehensive_model_optimization(self, 
                                       model_predictions: Dict[str, np.ndarray],
                                       y_val: np.ndarray,
                                       model_performances: Dict[str, Dict]) -> Dict:
        """
        종합적인 모델 최적화
        
        Args:
            model_predictions: 모델별 예측값
            y_val: 검증 타겟
            model_performances: 모델별 성능 지표
            
        Returns:
            Dict: 최적화 결과
        """
        print("🎯 고급 모델 선택 및 최적화 시작...")
        
        start_time = datetime.now()
        
        # 1. 유전 알고리즘 최적화
        print("🧬 유전 알고리즘 가중치 최적화...")
        genetic_weights = self.genetic_optimizer.optimize_ensemble_weights(
            model_predictions, y_val
        )
        
        # 2. 다목적 최적화 - 파레토 최적 모델 선택
        print("🎯 파레토 최적 모델 선택...")
        pareto_models = self.multi_objective_optimizer.pareto_optimal_selection(
            model_performances
        )
        
        # 3. 다양성 메트릭 계산
        print("📊 앙상블 다양성 분석...")
        diversity_metrics = self.multi_objective_optimizer.calculate_diversity_metrics(
            model_predictions
        )
        
        # 4. 강화학습 에이전트 초기화 (모델 수가 충분한 경우)
        if len(model_predictions) >= 3 and REINFORCEMENT_AVAILABLE:
            print("🎮 강화학습 모델 선택 에이전트 초기화...")
            self.rl_agent = ReinforcementEnsembleAgent(len(model_predictions))
            
            # 시장 상태 계산 (임시)
            recent_prices = y_val[-20:] if len(y_val) >= 20 else y_val
            model_perf_dict = {name: perf.get('direction_accuracy', 0) 
                              for name, perf in model_performances.items()}
            
            market_state = self.rl_agent.get_market_state(recent_prices, model_perf_dict)
            rl_selected_models = self.rl_agent.select_models(
                market_state, list(model_predictions.keys())
            )
        else:
            rl_selected_models = list(model_predictions.keys())
        
        # 5. 여러 방법의 결과 통합
        print("🔄 최적화 결과 통합...")
        
        # 유전 알고리즘 앙상블 평가
        genetic_ensemble = np.zeros(len(y_val))
        for model_name, pred in model_predictions.items():
            genetic_ensemble += genetic_weights.get(model_name, 0) * pred
        
        genetic_accuracy = self.calculate_direction_accuracy(y_val, genetic_ensemble)
        
        # 파레토 최적 앙상블 (균등 가중치)
        pareto_ensemble = np.zeros(len(y_val))
        pareto_weight = 1.0 / len(pareto_models) if pareto_models else 0
        
        for model_name in pareto_models:
            if model_name in model_predictions:
                pareto_ensemble += pareto_weight * model_predictions[model_name]
        
        pareto_accuracy = self.calculate_direction_accuracy(y_val, pareto_ensemble) if pareto_models else 0
        
        # 강화학습 선택 앙상블 (균등 가중치)
        rl_ensemble = np.zeros(len(y_val))
        rl_weight = 1.0 / len(rl_selected_models) if rl_selected_models else 0
        
        for model_name in rl_selected_models:
            if model_name in model_predictions:
                rl_ensemble += rl_weight * model_predictions[model_name]
        
        rl_accuracy = self.calculate_direction_accuracy(y_val, rl_ensemble) if rl_selected_models else 0
        
        # 6. 최적 방법 선택
        optimization_results = [
            {
                'method': 'genetic_algorithm',
                'weights': genetic_weights,
                'accuracy': genetic_accuracy,
                'selected_models': list(genetic_weights.keys())
            },
            {
                'method': 'pareto_optimal',
                'weights': {name: pareto_weight for name in pareto_models},
                'accuracy': pareto_accuracy,
                'selected_models': pareto_models
            },
            {
                'method': 'reinforcement_learning',
                'weights': {name: rl_weight for name in rl_selected_models},
                'accuracy': rl_accuracy,
                'selected_models': rl_selected_models
            }
        ]
        
        # 최고 성능 방법 선택
        best_method = max(optimization_results, key=lambda x: x['accuracy'])
        
        # 7. 결과 정리
        end_time = datetime.now()
        optimization_time = (end_time - start_time).total_seconds()
        
        result = {
            'optimization_completed': end_time.isoformat(),
            'optimization_time_seconds': optimization_time,
            'best_method': best_method,
            'all_methods': optimization_results,
            'diversity_metrics': diversity_metrics,
            'genetic_weights': genetic_weights,
            'pareto_models': pareto_models,
            'rl_selected_models': rl_selected_models,
            'total_models_evaluated': len(model_predictions)
        }
        
        # 최적화 기록 저장
        self.optimization_history.append(result)
        
        # 결과 출력
        print("\n" + "="*50)
        print("🎯 모델 선택 최적화 완료!")
        print("="*50)
        print(f"⏱️  최적화 시간: {optimization_time:.1f}초")
        print(f"🏆 최적 방법: {best_method['method']}")
        print(f"📈 최적 정확도: {best_method['accuracy']:.3f} ({best_method['accuracy']*100:.1f}%)")
        print(f"🤖 선택된 모델 수: {len(best_method['selected_models'])}개")
        print(f"📊 앙상블 다양성: {diversity_metrics['combined_diversity']:.3f}")
        
        if best_method['accuracy'] >= 0.90:
            print("🎉 90% 목표 달성!")
        
        return result

    def calculate_direction_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """방향성 정확도 계산"""
        if len(y_true) <= 1:
            return 0.0
        
        direction_actual = np.diff(y_true) > 0
        direction_pred = np.diff(y_pred) > 0
        return np.mean(direction_actual == direction_pred)

    def save_optimization_results(self, results: Dict, 
                                 file_path: str = None) -> str:
        """최적화 결과 저장"""
        if file_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = f"/Users/parkyoungjun/Desktop/BTC_Analysis_System/model_optimization_results_{timestamp}.json"
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        self.logger.info(f"💾 최적화 결과 저장: {file_path}")
        return file_path

def main():
    """메인 실행 함수"""
    print("🎯 고급 모델 선택 및 앙상블 최적화 시스템")
    
    # 시스템 초기화
    optimizer = AdvancedModelSelectionSystem()
    
    print("✅ 고급 모델 선택 최적화 시스템 초기화 완료")
    print("📋 사용 가능한 기능:")
    print("  🧬 유전 알고리즘 가중치 최적화")
    print("  🎯 파레토 최적 모델 선택")
    print("  📊 다목적 최적화 (정확도 vs 다양성)")
    if REINFORCEMENT_AVAILABLE:
        print("  🎮 강화학습 기반 동적 모델 선택")
    print("  🔄 온라인 적응형 앙상블")
    
    return optimizer

if __name__ == "__main__":
    optimizer = main()