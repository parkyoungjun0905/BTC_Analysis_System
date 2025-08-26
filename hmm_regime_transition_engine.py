#!/usr/bin/env python3
"""
Hidden Markov Model 기반 체제 전환 감지 엔진
Bitcoin 시장의 숨겨진 상태(체제) 전환을 통계적으로 모델링하고 예측

핵심 기능:
1. 시장 데이터의 숨겨진 체제 상태 추론
2. 체제 전환 확률 및 타이밍 예측  
3. 체제별 관측값 분포 학습
4. 실시간 체제 전환 신호 감지
"""

import os
import json
import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
import logging
from dataclasses import dataclass, asdict
import asyncio
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings("ignore")

# HMM 라이브러리 대신 수치적 구현
from scipy.stats import multivariate_normal
from scipy.special import logsumexp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class HMMRegimeState:
    """HMM 체제 상태 정의"""
    state_id: int
    state_name: str
    mean_features: np.ndarray
    covariance_matrix: np.ndarray
    steady_state_probability: float
    avg_duration_days: float
    
@dataclass 
class RegimeTransition:
    """체제 전환 정보"""
    from_state: str
    to_state: str
    transition_probability: float
    predicted_timing: datetime
    confidence: float
    leading_indicators: List[str]
    market_catalysts: List[str]

class HMMRegimeTransitionEngine:
    """Hidden Markov Model 기반 체제 전환 엔진"""
    
    def __init__(self, base_path: str = "/Users/parkyoungjun/Desktop/BTC_Analysis_System"):
        self.base_path = base_path
        self.db_path = os.path.join(base_path, "hmm_regime_db.db")
        self.models_path = os.path.join(base_path, "hmm_models")
        os.makedirs(self.models_path, exist_ok=True)
        
        # HMM 파라미터
        self.n_states = 5  # 5개 체제
        self.n_features = 8  # 관측 특징 수
        
        # 상태 정의 (0부터 시작)
        self.state_names = {
            0: "LOW_VOLATILITY_ACCUMULATION",
            1: "BULL_MARKET", 
            2: "SIDEWAYS",
            3: "BEAR_MARKET",
            4: "HIGH_VOLATILITY_SHOCK"
        }
        
        # HMM 모델 파라미터들
        self.transition_matrix = None  # A[i,j] = P(state_j | state_i)
        self.emission_means = None     # 각 상태별 관측값 평균
        self.emission_covariances = None # 각 상태별 관측값 공분산
        self.initial_probs = None      # 초기 상태 확률
        
        # 학습된 모델 상태
        self.is_trained = False
        self.scaler = StandardScaler()
        
        # 현재 추정 상태
        self.current_state_probs = np.ones(self.n_states) / self.n_states
        self.state_history = []
        
        # 체제 전환 감지 파라미터
        self.transition_threshold = 0.3  # 전환 감지 임계값
        self.confidence_threshold = 0.7   # 신뢰도 임계값
        self.lookback_window = 20        # 관측 윈도우 (일)
        
        self.init_database()
        self.load_hmm_model()
    
    def init_database(self):
        """데이터베이스 초기화"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # HMM 상태 기록
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS hmm_states (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    predicted_state INTEGER NOT NULL,
                    state_name TEXT NOT NULL,
                    state_probabilities TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    observation_features TEXT NOT NULL,
                    forward_probability REAL,
                    backward_probability REAL
                )
            ''')
            
            # 체제 전환 예측
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS regime_transitions_hmm (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    from_state INTEGER NOT NULL,
                    to_state INTEGER NOT NULL,
                    from_state_name TEXT NOT NULL,
                    to_state_name TEXT NOT NULL,
                    transition_probability REAL NOT NULL,
                    predicted_timing TEXT,
                    confidence REAL NOT NULL,
                    leading_indicators TEXT NOT NULL,
                    actual_transition BOOLEAN,
                    verification_timestamp TEXT
                )
            ''')
            
            # HMM 모델 파라미터
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS hmm_model_params (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_version TEXT NOT NULL,
                    transition_matrix TEXT NOT NULL,
                    emission_means TEXT NOT NULL,
                    emission_covariances TEXT NOT NULL,
                    initial_probabilities TEXT NOT NULL,
                    training_date TEXT NOT NULL,
                    training_samples INTEGER NOT NULL,
                    log_likelihood REAL,
                    is_active BOOLEAN DEFAULT TRUE
                )
            ''')
            
            # 학습 데이터
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS hmm_training_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    features TEXT NOT NULL,
                    true_state INTEGER,
                    true_state_name TEXT,
                    is_validated BOOLEAN DEFAULT FALSE
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("✅ HMM 데이터베이스 초기화 완료")
            
        except Exception as e:
            logger.error(f"HMM 데이터베이스 초기화 실패: {e}")
    
    def load_hmm_model(self):
        """학습된 HMM 모델 로드"""
        try:
            model_file = os.path.join(self.models_path, "hmm_regime_model.pkl")
            if os.path.exists(model_file):
                with open(model_file, 'rb') as f:
                    model_data = pickle.load(f)
                    
                    self.transition_matrix = model_data.get('transition_matrix')
                    self.emission_means = model_data.get('emission_means')
                    self.emission_covariances = model_data.get('emission_covariances') 
                    self.initial_probs = model_data.get('initial_probs')
                    self.scaler = model_data.get('scaler', StandardScaler())
                    
                    if all(x is not None for x in [self.transition_matrix, self.emission_means, 
                                                  self.emission_covariances, self.initial_probs]):
                        self.is_trained = True
                        logger.info("✅ HMM 모델 로드 완료")
                    else:
                        logger.warning("⚠️ HMM 모델 파라미터 불완전")
            else:
                logger.info("HMM 모델 파일이 없습니다. 새로운 학습이 필요합니다.")
                self.initialize_default_parameters()
                
        except Exception as e:
            logger.error(f"HMM 모델 로드 실패: {e}")
            self.initialize_default_parameters()
    
    def initialize_default_parameters(self):
        """기본 HMM 파라미터 초기화"""
        try:
            # 전환 확률 행렬 (경험적 추정값)
            self.transition_matrix = np.array([
                # LOW_VOL, BULL, SIDEWAYS, BEAR, HIGH_VOL
                [0.7,     0.2,  0.05,     0.03,  0.02],   # LOW_VOLATILITY_ACCUMULATION
                [0.05,    0.75, 0.15,     0.04,  0.01],   # BULL_MARKET
                [0.1,     0.25, 0.5,      0.1,   0.05],   # SIDEWAYS
                [0.02,    0.03, 0.15,     0.75,  0.05],   # BEAR_MARKET
                [0.15,    0.2,  0.3,      0.25,  0.1]     # HIGH_VOLATILITY_SHOCK
            ])
            
            # 관측값 평균 (각 상태별)
            self.emission_means = np.array([
                [0.01,  0.02,  0.3,   45,   0.5,  0.0,   60,   0.1],   # LOW_VOL
                [0.05,  0.03,  0.8,   65,   0.7,  0.05,  75,   0.6],   # BULL
                [0.0,   0.025, 0.5,   50,   0.5,  0.0,   50,   0.3],   # SIDEWAYS
                [-0.05, 0.04,  0.2,   35,   0.3, -0.05,  25,   0.2],   # BEAR
                [0.0,   0.08,  0.4,   50,   0.5,  0.0,   40,   0.4]    # HIGH_VOL
            ])
            
            # 공분산 행렬 (단순화된 대각 행렬)
            self.emission_covariances = np.array([
                np.diag([0.01, 0.005, 0.2, 100, 0.1, 0.02, 200, 0.1]) for _ in range(self.n_states)
            ])
            
            # 초기 상태 확률
            self.initial_probs = np.array([0.2, 0.2, 0.4, 0.1, 0.1])
            
            logger.info("✅ 기본 HMM 파라미터 초기화 완료")
            
        except Exception as e:
            logger.error(f"기본 파라미터 초기화 실패: {e}")
    
    async def train_hmm_model(self, training_data: List[Dict], max_iterations: int = 100) -> Dict:
        """Baum-Welch 알고리즘으로 HMM 모델 학습"""
        try:
            if not training_data:
                return {"error": "학습 데이터가 없습니다"}
            
            logger.info(f"🧠 HMM 모델 학습 시작 (데이터: {len(training_data)}개)")
            
            # 특징값 추출 및 전처리
            observations = []
            true_states = []
            
            for data in training_data:
                features = data.get('features')
                true_state = data.get('true_state')
                
                if features and len(features) == self.n_features:
                    observations.append(features)
                    if true_state is not None:
                        true_states.append(true_state)
            
            if len(observations) < 10:
                return {"error": "충분한 학습 데이터가 없습니다"}
            
            observations = np.array(observations)
            observations = self.scaler.fit_transform(observations)
            T = len(observations)
            
            # Baum-Welch 알고리즘 실행
            log_likelihood_history = []
            
            for iteration in range(max_iterations):
                # Forward-Backward 알고리즘
                alpha, c = self.forward_algorithm(observations)
                beta = self.backward_algorithm(observations, c)
                
                # 현재 log-likelihood 계산
                log_likelihood = -np.sum(np.log(c))
                log_likelihood_history.append(log_likelihood)
                
                if iteration > 0:
                    improvement = log_likelihood - log_likelihood_history[-2]
                    if abs(improvement) < 1e-6:
                        logger.info(f"수렴 완료 (반복: {iteration})")
                        break
                
                # Expectation step
                gamma, xi = self.expectation_step(alpha, beta, observations)
                
                # Maximization step
                self.maximization_step(observations, gamma, xi)
                
                if iteration % 10 == 0:
                    logger.info(f"반복 {iteration}: Log-likelihood = {log_likelihood:.2f}")
            
            # 학습 완료된 모델 저장
            await self.save_hmm_model(log_likelihood_history[-1], len(training_data))
            
            self.is_trained = True
            
            return {
                "training_completed": True,
                "iterations": len(log_likelihood_history),
                "final_log_likelihood": log_likelihood_history[-1],
                "training_samples": len(observations),
                "convergence_improvement": log_likelihood_history[-1] - log_likelihood_history[0] if len(log_likelihood_history) > 1 else 0
            }
            
        except Exception as e:
            logger.error(f"HMM 모델 학습 실패: {e}")
            return {"error": str(e)}
    
    def forward_algorithm(self, observations: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Forward 알고리즘"""
        T = len(observations)
        alpha = np.zeros((T, self.n_states))
        c = np.zeros(T)  # scaling factor
        
        # 초기화 (t=0)
        for i in range(self.n_states):
            alpha[0, i] = self.initial_probs[i] * self.emission_probability(observations[0], i)
        
        c[0] = np.sum(alpha[0])
        if c[0] > 0:
            alpha[0] /= c[0]
        
        # 순방향 계산 (t=1 to T-1)
        for t in range(1, T):
            for j in range(self.n_states):
                alpha[t, j] = np.sum(alpha[t-1] * self.transition_matrix[:, j]) * \
                             self.emission_probability(observations[t], j)
            
            c[t] = np.sum(alpha[t])
            if c[t] > 0:
                alpha[t] /= c[t]
        
        return alpha, c
    
    def backward_algorithm(self, observations: np.ndarray, c: np.ndarray) -> np.ndarray:
        """Backward 알고리즘"""
        T = len(observations)
        beta = np.zeros((T, self.n_states))
        
        # 초기화 (t=T-1)
        beta[T-1] = 1.0
        
        # 역방향 계산 (t=T-2 to 0)
        for t in range(T-2, -1, -1):
            for i in range(self.n_states):
                beta[t, i] = np.sum(
                    self.transition_matrix[i] * 
                    np.array([self.emission_probability(observations[t+1], j) for j in range(self.n_states)]) * 
                    beta[t+1]
                )
            
            if c[t+1] > 0:
                beta[t] /= c[t+1]
        
        return beta
    
    def expectation_step(self, alpha: np.ndarray, beta: np.ndarray, 
                        observations: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Expectation step (gamma, xi 계산)"""
        T = len(observations)
        
        # gamma[t][i] = P(state=i at time t | observations)
        gamma = alpha * beta
        gamma_sum = np.sum(gamma, axis=1, keepdims=True)
        gamma = gamma / np.maximum(gamma_sum, 1e-10)
        
        # xi[t][i][j] = P(state=i at t, state=j at t+1 | observations)
        xi = np.zeros((T-1, self.n_states, self.n_states))
        
        for t in range(T-1):
            for i in range(self.n_states):
                for j in range(self.n_states):
                    xi[t, i, j] = (alpha[t, i] * self.transition_matrix[i, j] * 
                                  self.emission_probability(observations[t+1], j) * 
                                  beta[t+1, j])
            
            xi_sum = np.sum(xi[t])
            if xi_sum > 0:
                xi[t] /= xi_sum
        
        return gamma, xi
    
    def maximization_step(self, observations: np.ndarray, gamma: np.ndarray, xi: np.ndarray):
        """Maximization step (파라미터 업데이트)"""
        T = len(observations)
        
        # 초기 상태 확률 업데이트
        self.initial_probs = gamma[0]
        
        # 전환 확률 업데이트
        for i in range(self.n_states):
            for j in range(self.n_states):
                numerator = np.sum(xi[:, i, j])
                denominator = np.sum(gamma[:-1, i])
                self.transition_matrix[i, j] = numerator / max(denominator, 1e-10)
        
        # 관측 확률 파라미터 업데이트 (가우시안 가정)
        for i in range(self.n_states):
            gamma_sum = np.sum(gamma[:, i])
            
            if gamma_sum > 0:
                # 평균 업데이트
                self.emission_means[i] = np.sum(observations * gamma[:, i:i+1], axis=0) / gamma_sum
                
                # 공분산 업데이트
                diff = observations - self.emission_means[i]
                self.emission_covariances[i] = np.sum(
                    gamma[:, i:i+1] * diff[:, :, np.newaxis] * diff[:, np.newaxis, :], axis=0
                ) / gamma_sum
                
                # 수치적 안정성을 위한 정규화
                self.emission_covariances[i] += np.eye(self.n_features) * 1e-6
    
    def emission_probability(self, observation: np.ndarray, state: int) -> float:
        """관측값에 대한 방출 확률 (다변량 가우시안)"""
        try:
            mean = self.emission_means[state]
            cov = self.emission_covariances[state]
            
            # 수치적 안정성 체크
            if np.any(np.isnan(observation)) or np.any(np.isnan(mean)):
                return 1e-10
            
            # 공분산 행렬의 특이값 처리
            try:
                prob = multivariate_normal.pdf(observation, mean, cov)
                return max(prob, 1e-10)  # 최소값 보장
            except:
                return 1e-10
                
        except Exception as e:
            logger.error(f"방출 확률 계산 실패: {e}")
            return 1e-10
    
    async def predict_regime_state(self, market_features: np.ndarray) -> Dict:
        """현재 시장 특징으로부터 체제 상태 예측"""
        try:
            if not self.is_trained:
                return {"error": "HMM 모델이 학습되지 않았습니다"}
            
            # 특징값 전처리
            features_scaled = self.scaler.transform(market_features.reshape(1, -1))[0]
            
            # 각 상태별 관측 확률 계산
            state_likelihoods = np.array([
                self.emission_probability(features_scaled, i) for i in range(self.n_states)
            ])
            
            # 이전 상태 확률과 전환 확률 고려
            prior_probs = self.current_state_probs @ self.transition_matrix
            
            # 베이즈 업데이트
            posterior_probs = prior_probs * state_likelihoods
            posterior_probs /= np.sum(posterior_probs)
            
            # 상태 업데이트
            self.current_state_probs = posterior_probs
            
            # 가장 가능성 높은 상태
            predicted_state = np.argmax(posterior_probs)
            confidence = posterior_probs[predicted_state]
            
            # 상태 히스토리 업데이트
            self.state_history.append({
                "timestamp": datetime.now(),
                "predicted_state": predicted_state,
                "confidence": confidence,
                "all_probabilities": posterior_probs.tolist()
            })
            
            # 체제 전환 감지
            transition_info = self.detect_regime_transition()
            
            result = {
                "predicted_state": predicted_state,
                "state_name": self.state_names[predicted_state],
                "confidence": confidence,
                "state_probabilities": {
                    self.state_names[i]: prob for i, prob in enumerate(posterior_probs)
                },
                "transition_detected": transition_info.get("transition_detected", False),
                "transition_info": transition_info,
                "timestamp": datetime.now().isoformat()
            }
            
            # 결과 저장
            await self.save_state_prediction(result, features_scaled)
            
            return result
            
        except Exception as e:
            logger.error(f"체제 상태 예측 실패: {e}")
            return {"error": str(e)}
    
    def detect_regime_transition(self) -> Dict:
        """체제 전환 감지"""
        try:
            if len(self.state_history) < 2:
                return {"transition_detected": False}
            
            current_probs = self.state_history[-1]["all_probabilities"]
            previous_probs = self.state_history[-2]["all_probabilities"]
            
            # 상태 확률 변화량 계산
            prob_changes = np.array(current_probs) - np.array(previous_probs)
            max_increase = np.max(prob_changes)
            max_decrease = np.min(prob_changes)
            
            # 전환 조건 확인
            transition_detected = False
            from_state = None
            to_state = None
            
            if max_increase > self.transition_threshold and abs(max_decrease) > self.transition_threshold:
                from_state = int(np.argmin(prob_changes))
                to_state = int(np.argmax(prob_changes))
                
                # 신뢰도 확인
                if current_probs[to_state] > self.confidence_threshold:
                    transition_detected = True
            
            # 전환 타이밍 예측
            predicted_timing = None
            if transition_detected:
                predicted_timing = self.predict_transition_timing(from_state, to_state)
            
            return {
                "transition_detected": transition_detected,
                "from_state": from_state,
                "to_state": to_state,
                "from_state_name": self.state_names.get(from_state) if from_state is not None else None,
                "to_state_name": self.state_names.get(to_state) if to_state is not None else None,
                "probability_change": max_increase,
                "confidence": current_probs[to_state] if to_state is not None else 0,
                "predicted_timing": predicted_timing,
                "leading_indicators": self.identify_leading_indicators(from_state, to_state)
            }
            
        except Exception as e:
            logger.error(f"체제 전환 감지 실패: {e}")
            return {"transition_detected": False, "error": str(e)}
    
    def predict_transition_timing(self, from_state: int, to_state: int) -> Optional[str]:
        """체제 전환 타이밍 예측"""
        try:
            # 전환 확률 기반 예상 대기 시간 계산
            transition_prob = self.transition_matrix[from_state, to_state]
            
            if transition_prob > 0:
                # 기하 분포 기반 예상 대기 시간 (일)
                expected_days = 1 / transition_prob
                predicted_date = datetime.now() + timedelta(days=expected_days)
                return predicted_date.isoformat()
            
            return None
            
        except Exception as e:
            logger.error(f"전환 타이밍 예측 실패: {e}")
            return None
    
    def identify_leading_indicators(self, from_state: Optional[int], to_state: Optional[int]) -> List[str]:
        """체제 전환 선행 지표 식별"""
        indicators = []
        
        try:
            if from_state is None or to_state is None:
                return indicators
            
            # 상태별 주요 특징 인덱스
            feature_names = [
                "price_trend", "volatility", "volume_trend", "rsi", 
                "whale_activity", "funding_rate", "fear_greed", "correlation"
            ]
            
            # 두 상태간 특징값 차이 분석
            from_mean = self.emission_means[from_state]
            to_mean = self.emission_means[to_state]
            
            differences = np.abs(to_mean - from_mean)
            significant_indices = np.where(differences > np.std(differences))[0]
            
            for idx in significant_indices:
                if idx < len(feature_names):
                    indicators.append(feature_names[idx])
            
            # 상태별 특화 지표 추가
            state_specific_indicators = {
                0: ["hodler_behavior", "whale_accumulation", "low_volume"],    # LOW_VOL
                1: ["momentum_indicators", "volume_surge", "positive_sentiment"], # BULL
                2: ["range_indicators", "consolidation_patterns"],              # SIDEWAYS
                3: ["fear_indicators", "selling_pressure", "exchange_inflows"], # BEAR
                4: ["volatility_spike", "liquidations", "news_events"]         # HIGH_VOL
            }
            
            if to_state in state_specific_indicators:
                indicators.extend(state_specific_indicators[to_state])
            
        except Exception as e:
            logger.error(f"선행 지표 식별 실패: {e}")
        
        return indicators[:5]  # 상위 5개 지표만 반환
    
    async def save_hmm_model(self, log_likelihood: float, training_samples: int):
        """HMM 모델 저장"""
        try:
            # 파일 저장
            model_data = {
                'transition_matrix': self.transition_matrix,
                'emission_means': self.emission_means,
                'emission_covariances': self.emission_covariances,
                'initial_probs': self.initial_probs,
                'scaler': self.scaler
            }
            
            model_file = os.path.join(self.models_path, "hmm_regime_model.pkl")
            with open(model_file, 'wb') as f:
                pickle.dump(model_data, f)
            
            # 데이터베이스 저장
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO hmm_model_params 
                (model_version, transition_matrix, emission_means, emission_covariances,
                 initial_probabilities, training_date, training_samples, log_likelihood, is_active)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                json.dumps(self.transition_matrix.tolist()),
                json.dumps(self.emission_means.tolist()),
                json.dumps([cov.tolist() for cov in self.emission_covariances]),
                json.dumps(self.initial_probs.tolist()),
                datetime.now().isoformat(),
                training_samples,
                log_likelihood,
                True
            ))
            
            # 이전 모델들을 비활성화
            cursor.execute('UPDATE hmm_model_params SET is_active = FALSE WHERE id != last_insert_rowid()')
            
            conn.commit()
            conn.close()
            
            logger.info(f"✅ HMM 모델 저장 완료 (Log-likelihood: {log_likelihood:.2f})")
            
        except Exception as e:
            logger.error(f"HMM 모델 저장 실패: {e}")
    
    async def save_state_prediction(self, prediction: Dict, features: np.ndarray):
        """상태 예측 결과 저장"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO hmm_states 
                (timestamp, predicted_state, state_name, state_probabilities, 
                 confidence, observation_features)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                prediction["timestamp"],
                prediction["predicted_state"],
                prediction["state_name"],
                json.dumps(prediction["state_probabilities"]),
                prediction["confidence"],
                json.dumps(features.tolist())
            ))
            
            # 체제 전환이 감지되면 별도 저장
            if prediction["transition_detected"]:
                transition_info = prediction["transition_info"]
                cursor.execute('''
                    INSERT INTO regime_transitions_hmm
                    (timestamp, from_state, to_state, from_state_name, to_state_name,
                     transition_probability, predicted_timing, confidence, leading_indicators)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    prediction["timestamp"],
                    transition_info.get("from_state"),
                    transition_info.get("to_state"),
                    transition_info.get("from_state_name"),
                    transition_info.get("to_state_name"),
                    transition_info.get("probability_change"),
                    transition_info.get("predicted_timing"),
                    transition_info.get("confidence"),
                    json.dumps(transition_info.get("leading_indicators", []))
                ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"상태 예측 저장 실패: {e}")
    
    async def generate_transition_forecast(self, horizon_days: int = 7) -> Dict:
        """체제 전환 예측 (지정된 기간 내)"""
        try:
            if not self.is_trained:
                return {"error": "HMM 모델이 학습되지 않았습니다"}
            
            forecasts = []
            current_probs = self.current_state_probs.copy()
            
            for day in range(1, horizon_days + 1):
                # 다음 날 상태 확률 계산
                next_probs = current_probs @ self.transition_matrix
                
                # 가장 가능성 높은 상태
                predicted_state = np.argmax(next_probs)
                confidence = next_probs[predicted_state]
                
                # 전환 확률 (현재 상태에서 다른 상태로)
                current_state = np.argmax(current_probs)
                transition_probs = {}
                
                for i in range(self.n_states):
                    if i != current_state:
                        transition_probs[self.state_names[i]] = self.transition_matrix[current_state, i]
                
                forecasts.append({
                    "day": day,
                    "date": (datetime.now() + timedelta(days=day)).isoformat(),
                    "predicted_state": predicted_state,
                    "state_name": self.state_names[predicted_state],
                    "confidence": confidence,
                    "state_probabilities": {
                        self.state_names[i]: prob for i, prob in enumerate(next_probs)
                    },
                    "transition_probabilities": transition_probs
                })
                
                current_probs = next_probs
            
            # 주요 전환 시점 식별
            major_transitions = []
            for i, forecast in enumerate(forecasts[1:], 1):
                prev_state = forecasts[i-1]["predicted_state"]
                curr_state = forecast["predicted_state"]
                
                if prev_state != curr_state and forecast["confidence"] > 0.6:
                    major_transitions.append({
                        "day": forecast["day"],
                        "date": forecast["date"],
                        "from_state": self.state_names[prev_state],
                        "to_state": self.state_names[curr_state],
                        "confidence": forecast["confidence"]
                    })
            
            return {
                "forecast_horizon_days": horizon_days,
                "daily_forecasts": forecasts,
                "major_transitions": major_transitions,
                "current_state": self.state_names[np.argmax(self.current_state_probs)],
                "current_confidence": np.max(self.current_state_probs),
                "forecast_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"전환 예측 실패: {e}")
            return {"error": str(e)}
    
    async def get_model_diagnostics(self) -> Dict:
        """HMM 모델 진단 정보"""
        try:
            diagnostics = {
                "model_status": "trained" if self.is_trained else "untrained",
                "n_states": self.n_states,
                "n_features": self.n_features
            }
            
            if self.is_trained:
                # 전환 행렬 분석
                diagnostics["transition_matrix"] = {
                    "matrix": self.transition_matrix.tolist(),
                    "most_stable_state": self.state_names[np.argmax(np.diag(self.transition_matrix))],
                    "most_volatile_state": self.state_names[np.argmin(np.diag(self.transition_matrix))]
                }
                
                # 정상 상태 확률 계산
                eigenvals, eigenvecs = np.linalg.eig(self.transition_matrix.T)
                steady_state_idx = np.argmax(eigenvals.real)
                steady_state = np.abs(eigenvecs[:, steady_state_idx].real)
                steady_state /= np.sum(steady_state)
                
                diagnostics["steady_state_probabilities"] = {
                    self.state_names[i]: prob for i, prob in enumerate(steady_state)
                }
                
                # 현재 상태 정보
                diagnostics["current_state"] = {
                    "probabilities": {self.state_names[i]: prob for i, prob in enumerate(self.current_state_probs)},
                    "most_likely": self.state_names[np.argmax(self.current_state_probs)],
                    "confidence": np.max(self.current_state_probs)
                }
                
                # 최근 상태 히스토리
                if self.state_history:
                    recent_states = self.state_history[-10:]  # 최근 10개
                    diagnostics["recent_history"] = [
                        {
                            "timestamp": h["timestamp"].isoformat() if isinstance(h["timestamp"], datetime) else h["timestamp"],
                            "state": self.state_names[h["predicted_state"]], 
                            "confidence": h["confidence"]
                        } for h in recent_states
                    ]
            
            return diagnostics
            
        except Exception as e:
            logger.error(f"모델 진단 실패: {e}")
            return {"error": str(e)}

# 테스트 및 실행 함수
async def test_hmm_regime_engine():
    """HMM 체제 전환 엔진 테스트"""
    print("🔬 HMM 체제 전환 엔진 테스트")
    print("=" * 50)
    
    engine = HMMRegimeTransitionEngine()
    
    # 모델 진단
    diagnostics = await engine.get_model_diagnostics()
    print(f"📊 모델 상태: {diagnostics.get('model_status')}")
    
    if diagnostics.get('model_status') == 'trained':
        # 테스트 특징값
        test_features = np.array([
            0.02,   # price_trend  
            0.04,   # volatility
            0.5,    # volume_trend
            55,     # rsi
            0.6,    # whale_activity
            0.01,   # funding_rate
            45,     # fear_greed
            0.2     # correlation
        ])
        
        # 상태 예측
        prediction = await engine.predict_regime_state(test_features)
        
        if not prediction.get("error"):
            print(f"🎯 예측 상태: {prediction['state_name']}")
            print(f"🔥 신뢰도: {prediction['confidence']:.1%}")
            print(f"📈 상태 확률:")
            for state, prob in prediction['state_probabilities'].items():
                print(f"   • {state}: {prob:.1%}")
            
            if prediction.get('transition_detected'):
                trans = prediction['transition_info']
                print(f"⚡ 체제 전환 감지!")
                print(f"   • {trans['from_state_name']} → {trans['to_state_name']}")
                print(f"   • 신뢰도: {trans['confidence']:.1%}")
                if trans.get('predicted_timing'):
                    print(f"   • 예상 시점: {trans['predicted_timing']}")
        
        # 미래 예측
        forecast = await engine.generate_transition_forecast(7)
        if not forecast.get("error"):
            print(f"\n📅 7일 예측:")
            for transition in forecast.get('major_transitions', []):
                print(f"   • Day {transition['day']}: {transition['from_state']} → {transition['to_state']} ({transition['confidence']:.1%})")
    else:
        print("⚠️ 모델이 학습되지 않았습니다. 기본 파라미터 사용")
    
    print("\n" + "=" * 50)
    print("🎉 HMM 엔진 테스트 완료!")

if __name__ == "__main__":
    asyncio.run(test_hmm_regime_engine())