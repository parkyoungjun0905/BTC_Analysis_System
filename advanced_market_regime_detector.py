#!/usr/bin/env python3
"""
고도화된 시장 체제 감지 시스템
Bitcoin 시장의 5가지 주요 체제를 실시간 감지하고 체제별 맞춤형 예측 모델 적용

체제 분류:
1. BULL_MARKET: 지속적 상승 추세 (트렌드 기반 전략 유리)
2. BEAR_MARKET: 지속적 하락 추세 (리버설 전략 유리)  
3. SIDEWAYS: 횡보/통합 구간 (레인지 트레이딩 유리)
4. HIGH_VOLATILITY_SHOCK: 급변동/충격 구간 (위험 관리 우선)
5. LOW_VOLATILITY_ACCUMULATION: 저변동성 축적 구간 (브레이크아웃 대기)
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
from collections import deque, defaultdict
import statistics
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from hmmlearn import hmm
import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RegimeFeatures:
    """체제 분류를 위한 특징값들"""
    timestamp: datetime
    
    # 가격 트렌드 특징
    price_trend_1d: float
    price_trend_7d: float
    price_trend_30d: float
    trend_consistency: float
    
    # 변동성 특징
    volatility_1d: float
    volatility_7d: float
    volatility_30d: float
    volatility_regime_change: float
    
    # 거래량 특징
    volume_trend: float
    volume_volatility: float
    volume_price_correlation: float
    
    # 기술적 지표
    rsi_14: float
    macd_signal: float
    bollinger_position: float
    
    # 온체인 특징
    whale_activity: float
    exchange_flow: float
    hodler_behavior: float
    
    # 시장 구조 특징
    futures_basis: float
    funding_rate: float
    put_call_ratio: float
    fear_greed_index: float
    
    # 거시경제
    correlation_gold: float
    correlation_stocks: float
    dxy_impact: float

@dataclass
class MarketRegime:
    """시장 체제 정의"""
    regime_type: str
    confidence: float
    duration_days: int
    key_characteristics: List[str]
    expected_duration_days: int
    transition_probability: Dict[str, float]
    optimal_strategies: List[str]
    risk_level: str
    
class AdvancedMarketRegimeDetector:
    """고도화된 시장 체제 감지 시스템"""
    
    def __init__(self):
        self.base_path = "/Users/parkyoungjun/Desktop/BTC_Analysis_System"
        self.db_path = os.path.join(self.base_path, "market_regime_db.db")
        self.models_path = os.path.join(self.base_path, "regime_models")
        os.makedirs(self.models_path, exist_ok=True)
        
        # 체제별 특성 정의
        self.regime_definitions = {
            "BULL_MARKET": {
                "description": "지속적 상승 추세",
                "min_duration": 7,
                "key_indicators": ["price_trend_7d", "volume_trend", "rsi_14"],
                "expected_duration": 30,
                "risk_level": "LOW_TO_MEDIUM"
            },
            "BEAR_MARKET": {
                "description": "지속적 하락 추세", 
                "min_duration": 7,
                "key_indicators": ["price_trend_7d", "fear_greed_index", "exchange_flow"],
                "expected_duration": 45,
                "risk_level": "MEDIUM_TO_HIGH"
            },
            "SIDEWAYS": {
                "description": "횡보/통합 구간",
                "min_duration": 5,
                "key_indicators": ["volatility_1d", "bollinger_position", "volume_volatility"],
                "expected_duration": 20,
                "risk_level": "LOW"
            },
            "HIGH_VOLATILITY_SHOCK": {
                "description": "급변동/충격 구간",
                "min_duration": 1,
                "key_indicators": ["volatility_1d", "volume_volatility", "futures_basis"],
                "expected_duration": 7,
                "risk_level": "VERY_HIGH"
            },
            "LOW_VOLATILITY_ACCUMULATION": {
                "description": "저변동성 축적 구간",
                "min_duration": 3,
                "key_indicators": ["volatility_7d", "hodler_behavior", "whale_activity"],
                "expected_duration": 15,
                "risk_level": "LOW"
            }
        }
        
        # 체제 전환 확률 매트릭스 (기본값)
        self.transition_matrix = {
            "BULL_MARKET": {
                "BULL_MARKET": 0.7, "BEAR_MARKET": 0.1, "SIDEWAYS": 0.15,
                "HIGH_VOLATILITY_SHOCK": 0.04, "LOW_VOLATILITY_ACCUMULATION": 0.01
            },
            "BEAR_MARKET": {
                "BULL_MARKET": 0.05, "BEAR_MARKET": 0.75, "SIDEWAYS": 0.15,
                "HIGH_VOLATILITY_SHOCK": 0.04, "LOW_VOLATILITY_ACCUMULATION": 0.01
            },
            "SIDEWAYS": {
                "BULL_MARKET": 0.25, "BEAR_MARKET": 0.25, "SIDEWAYS": 0.35,
                "HIGH_VOLATILITY_SHOCK": 0.1, "LOW_VOLATILITY_ACCUMULATION": 0.05
            },
            "HIGH_VOLATILITY_SHOCK": {
                "BULL_MARKET": 0.2, "BEAR_MARKET": 0.3, "SIDEWAYS": 0.3,
                "HIGH_VOLATILITY_SHOCK": 0.15, "LOW_VOLATILITY_ACCUMULATION": 0.05
            },
            "LOW_VOLATILITY_ACCUMULATION": {
                "BULL_MARKET": 0.4, "BEAR_MARKET": 0.15, "SIDEWAYS": 0.2,
                "HIGH_VOLATILITY_SHOCK": 0.15, "LOW_VOLATILITY_ACCUMULATION": 0.1
            }
        }
        
        # 모델들
        self.regime_classifier = None
        self.hmm_model = None
        self.scaler = StandardScaler()
        
        # 체제별 예측 모델들
        self.regime_specific_models = {}
        
        # 현재 체제 추적
        self.current_regime = None
        self.regime_history = deque(maxlen=100)
        self.regime_start_time = None
        
        self.init_database()
        self.load_models()
        
    def init_database(self):
        """데이터베이스 초기화"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 체제 감지 기록
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS regime_detections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    regime_type TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    duration_days INTEGER NOT NULL,
                    key_features TEXT NOT NULL,
                    transition_from TEXT,
                    detection_method TEXT NOT NULL,
                    market_conditions TEXT NOT NULL
                )
            ''')
            
            # 체제별 성능 기록
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS regime_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    regime_type TEXT NOT NULL,
                    start_time TEXT NOT NULL,
                    end_time TEXT,
                    duration_actual INTEGER,
                    duration_expected INTEGER,
                    accuracy_score REAL,
                    prediction_count INTEGER,
                    avg_confidence REAL,
                    best_indicators TEXT,
                    performance_metrics TEXT
                )
            ''')
            
            # 특징값 히스토리
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS feature_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    features_json TEXT NOT NULL,
                    regime_label TEXT,
                    is_training_data BOOLEAN DEFAULT FALSE
                )
            ''')
            
            # 체제 전환 예측
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS regime_transitions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    from_regime TEXT NOT NULL,
                    to_regime TEXT NOT NULL,
                    predicted_probability REAL NOT NULL,
                    actual_transition BOOLEAN,
                    leading_indicators TEXT NOT NULL,
                    market_catalysts TEXT
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("✅ 체제 감지 데이터베이스 초기화 완료")
            
        except Exception as e:
            logger.error(f"데이터베이스 초기화 실패: {e}")
    
    def load_models(self):
        """학습된 모델들 로드"""
        try:
            model_files = {
                "classifier": "regime_classifier.pkl",
                "hmm": "hmm_model.pkl", 
                "scaler": "feature_scaler.pkl"
            }
            
            for model_name, filename in model_files.items():
                filepath = os.path.join(self.models_path, filename)
                if os.path.exists(filepath):
                    with open(filepath, 'rb') as f:
                        if model_name == "classifier":
                            self.regime_classifier = pickle.load(f)
                        elif model_name == "hmm":
                            self.hmm_model = pickle.load(f)
                        elif model_name == "scaler":
                            self.scaler = pickle.load(f)
            
            # 체제별 특화 모델 로드
            for regime in self.regime_definitions.keys():
                model_path = os.path.join(self.models_path, f"{regime.lower()}_model.pkl")
                if os.path.exists(model_path):
                    with open(model_path, 'rb') as f:
                        self.regime_specific_models[regime] = pickle.load(f)
            
            logger.info(f"✅ 모델 로드 완료: 분류기={self.regime_classifier is not None}, HMM={self.hmm_model is not None}")
            
        except Exception as e:
            logger.error(f"모델 로드 실패: {e}")
    
    async def extract_regime_features(self, market_data: Dict) -> Optional[RegimeFeatures]:
        """시장 데이터에서 체제 분류용 특징값 추출"""
        try:
            if not market_data:
                return None
            
            # 가격 데이터 추출 및 검증
            current_price = await self.extract_price(market_data)
            if not current_price or current_price <= 0:
                logger.warning("유효한 가격 데이터를 찾을 수 없습니다")
                return None
            
            # 과거 데이터 조회
            historical_data = await self.get_historical_features(days=30)
            if not historical_data:
                logger.warning("과거 데이터 부족으로 기본값 사용")
                return self.create_default_features(current_price)
            
            # 특징값 계산
            features = RegimeFeatures(
                timestamp=datetime.now(),
                
                # 가격 트렌드 (7일, 30일 변화율)
                price_trend_1d=self.calculate_trend(historical_data, 1),
                price_trend_7d=self.calculate_trend(historical_data, 7),
                price_trend_30d=self.calculate_trend(historical_data, 30),
                trend_consistency=self.calculate_trend_consistency(historical_data),
                
                # 변동성 (표준편차 기반)
                volatility_1d=self.calculate_volatility(historical_data, 1),
                volatility_7d=self.calculate_volatility(historical_data, 7),
                volatility_30d=self.calculate_volatility(historical_data, 30),
                volatility_regime_change=self.detect_volatility_regime_change(historical_data),
                
                # 거래량 특징
                volume_trend=self.calculate_volume_trend(market_data),
                volume_volatility=self.calculate_volume_volatility(market_data),
                volume_price_correlation=self.calculate_volume_price_correlation(historical_data),
                
                # 기술적 지표
                rsi_14=self.calculate_rsi(historical_data, 14),
                macd_signal=self.calculate_macd_signal(historical_data),
                bollinger_position=self.calculate_bollinger_position(historical_data, current_price),
                
                # 온체인 지표
                whale_activity=self.extract_whale_activity(market_data),
                exchange_flow=self.extract_exchange_flow(market_data),
                hodler_behavior=self.extract_hodler_behavior(market_data),
                
                # 시장 구조
                futures_basis=self.extract_futures_basis(market_data),
                funding_rate=self.extract_funding_rate(market_data),
                put_call_ratio=self.extract_put_call_ratio(market_data),
                fear_greed_index=self.extract_fear_greed(market_data),
                
                # 거시경제
                correlation_gold=self.extract_correlation(market_data, "gold"),
                correlation_stocks=self.extract_correlation(market_data, "stocks"),
                dxy_impact=self.extract_dxy_impact(market_data)
            )
            
            return features
            
        except Exception as e:
            logger.error(f"특징값 추출 실패: {e}")
            return None
    
    def create_default_features(self, current_price: float) -> RegimeFeatures:
        """기본 특징값 생성 (데이터 부족시)"""
        return RegimeFeatures(
            timestamp=datetime.now(),
            price_trend_1d=0.0, price_trend_7d=0.0, price_trend_30d=0.0,
            trend_consistency=0.5, volatility_1d=0.02, volatility_7d=0.03,
            volatility_30d=0.04, volatility_regime_change=0.0,
            volume_trend=0.0, volume_volatility=0.02, volume_price_correlation=0.3,
            rsi_14=50.0, macd_signal=0.0, bollinger_position=0.5,
            whale_activity=0.5, exchange_flow=0.0, hodler_behavior=0.5,
            futures_basis=0.0, funding_rate=0.0, put_call_ratio=0.5,
            fear_greed_index=50.0, correlation_gold=0.1, correlation_stocks=0.2,
            dxy_impact=0.0
        )
    
    async def detect_current_regime(self, market_data: Dict) -> Optional[MarketRegime]:
        """현재 시장 체제 감지"""
        try:
            # 특징값 추출
            features = await self.extract_regime_features(market_data)
            if not features:
                return None
            
            # 다중 방법론으로 체제 감지
            detections = await self.multi_method_regime_detection(features)
            
            # 결과 통합 및 신뢰도 계산
            regime_result = self.integrate_detection_results(detections)
            
            # 체제 지속성 및 전환 확률 고려
            final_regime = await self.apply_regime_stability_rules(regime_result, features)
            
            # 결과 저장
            await self.save_regime_detection(final_regime, features, detections)
            
            # 현재 상태 업데이트
            self.update_current_regime(final_regime)
            
            return final_regime
            
        except Exception as e:
            logger.error(f"체제 감지 실패: {e}")
            return None
    
    async def multi_method_regime_detection(self, features: RegimeFeatures) -> Dict:
        """다중 방법론 체제 감지"""
        detections = {}
        
        try:
            # 1. 규칙 기반 감지
            detections["rule_based"] = self.rule_based_detection(features)
            
            # 2. 머신러닝 분류기 감지 
            if self.regime_classifier:
                detections["ml_classifier"] = self.ml_classifier_detection(features)
            
            # 3. Hidden Markov Model 감지
            if self.hmm_model:
                detections["hmm"] = self.hmm_detection(features)
            
            # 4. 통계적 체제 감지
            detections["statistical"] = self.statistical_regime_detection(features)
            
            # 5. 변동성 체제 감지
            detections["volatility"] = self.volatility_regime_detection(features)
            
            return detections
            
        except Exception as e:
            logger.error(f"다중 방법론 감지 실패: {e}")
            return {"error": str(e)}
    
    def rule_based_detection(self, features: RegimeFeatures) -> Dict:
        """규칙 기반 체제 감지"""
        try:
            scores = {}
            
            # BULL_MARKET 조건
            bull_score = 0
            if features.price_trend_7d > 0.05:  # 7일간 5% 이상 상승
                bull_score += 3
            if features.price_trend_30d > 0.1:   # 30일간 10% 이상 상승
                bull_score += 3
            if features.rsi_14 > 55 and features.rsi_14 < 80:  # RSI 건강한 상승
                bull_score += 2
            if features.volume_trend > 0.1:     # 거래량 증가
                bull_score += 2
            scores["BULL_MARKET"] = bull_score / 10.0
            
            # BEAR_MARKET 조건
            bear_score = 0
            if features.price_trend_7d < -0.05:  # 7일간 5% 이상 하락
                bear_score += 3
            if features.price_trend_30d < -0.1:  # 30일간 10% 이상 하락
                bear_score += 3
            if features.fear_greed_index < 30:   # 공포 지수
                bear_score += 2
            if features.exchange_flow > 0.1:     # 거래소 유입 증가
                bear_score += 2
            scores["BEAR_MARKET"] = bear_score / 10.0
            
            # SIDEWAYS 조건
            sideways_score = 0
            if abs(features.price_trend_7d) < 0.02:  # 7일간 2% 이내 변동
                sideways_score += 3
            if features.volatility_1d < 0.03:       # 낮은 변동성
                sideways_score += 2
            if features.bollinger_position > 0.3 and features.bollinger_position < 0.7:
                sideways_score += 3
            if features.volume_volatility < 0.02:   # 안정적 거래량
                sideways_score += 2
            scores["SIDEWAYS"] = sideways_score / 10.0
            
            # HIGH_VOLATILITY_SHOCK 조건
            shock_score = 0
            if features.volatility_1d > 0.08:       # 일간 8% 이상 변동성
                shock_score += 4
            if features.volume_volatility > 0.05:   # 거래량 급변
                shock_score += 3
            if abs(features.futures_basis) > 0.02:  # 선물 베이시스 확대
                shock_score += 2
            if features.volatility_regime_change > 0.5:  # 변동성 체제 변화
                shock_score += 1
            scores["HIGH_VOLATILITY_SHOCK"] = shock_score / 10.0
            
            # LOW_VOLATILITY_ACCUMULATION 조건
            accumulation_score = 0
            if features.volatility_7d < 0.02:       # 7일간 낮은 변동성
                accumulation_score += 3
            if features.hodler_behavior > 0.6:      # 홀더 행동 증가
                accumulation_score += 2
            if features.whale_activity > 0.6:       # 고래 누적
                accumulation_score += 3
            if features.volume_trend < 0:           # 거래량 감소
                accumulation_score += 2
            scores["LOW_VOLATILITY_ACCUMULATION"] = accumulation_score / 10.0
            
            # 최고 점수 체제 선택
            best_regime = max(scores.items(), key=lambda x: x[1])
            
            return {
                "predicted_regime": best_regime[0],
                "confidence": best_regime[1],
                "all_scores": scores,
                "method": "rule_based"
            }
            
        except Exception as e:
            logger.error(f"규칙 기반 감지 실패: {e}")
            return {"error": str(e)}
    
    def ml_classifier_detection(self, features: RegimeFeatures) -> Dict:
        """머신러닝 분류기 체제 감지"""
        try:
            # 특징값을 배열로 변환
            feature_array = np.array([
                features.price_trend_1d, features.price_trend_7d, features.price_trend_30d,
                features.trend_consistency, features.volatility_1d, features.volatility_7d,
                features.volatility_30d, features.volatility_regime_change,
                features.volume_trend, features.volume_volatility, features.volume_price_correlation,
                features.rsi_14, features.macd_signal, features.bollinger_position,
                features.whale_activity, features.exchange_flow, features.hodler_behavior,
                features.futures_basis, features.funding_rate, features.put_call_ratio,
                features.fear_greed_index, features.correlation_gold, features.correlation_stocks,
                features.dxy_impact
            ]).reshape(1, -1)
            
            # 스케일링 적용
            feature_scaled = self.scaler.transform(feature_array)
            
            # 예측 수행
            predicted_regime = self.regime_classifier.predict(feature_scaled)[0]
            prediction_proba = self.regime_classifier.predict_proba(feature_scaled)[0]
            
            # 클래스별 확률
            regime_classes = self.regime_classifier.classes_
            class_probabilities = dict(zip(regime_classes, prediction_proba))
            
            return {
                "predicted_regime": predicted_regime,
                "confidence": max(prediction_proba),
                "class_probabilities": class_probabilities,
                "method": "ml_classifier"
            }
            
        except Exception as e:
            logger.error(f"ML 분류기 감지 실패: {e}")
            return {"error": str(e)}
    
    def hmm_detection(self, features: RegimeFeatures) -> Dict:
        """Hidden Markov Model 체제 감지"""
        try:
            # HMM 입력 특징 (연속 시계열이 필요)
            recent_features = self.get_recent_features_sequence(features, length=10)
            if len(recent_features) < 3:
                return {"error": "HMM을 위한 충분한 시계열 데이터가 없습니다"}
            
            # HMM 예측
            hidden_states = self.hmm_model.predict(recent_features)
            current_state = hidden_states[-1]
            
            # 상태를 체제로 매핑
            state_to_regime = {
                0: "LOW_VOLATILITY_ACCUMULATION",
                1: "BULL_MARKET", 
                2: "SIDEWAYS",
                3: "BEAR_MARKET",
                4: "HIGH_VOLATILITY_SHOCK"
            }
            
            predicted_regime = state_to_regime.get(current_state, "SIDEWAYS")
            
            # 상태 확률 계산
            state_proba = self.hmm_model.predict_proba(recent_features)
            confidence = np.max(state_proba[-1])
            
            return {
                "predicted_regime": predicted_regime,
                "confidence": confidence,
                "hidden_state": current_state,
                "state_sequence": hidden_states.tolist(),
                "method": "hmm"
            }
            
        except Exception as e:
            logger.error(f"HMM 감지 실패: {e}")
            return {"error": str(e)}
    
    def statistical_regime_detection(self, features: RegimeFeatures) -> Dict:
        """통계적 체제 감지"""
        try:
            # Z-score 기반 이상치 감지
            volatility_zscore = self.calculate_zscore(features.volatility_1d, "volatility_1d")
            trend_zscore = self.calculate_zscore(features.price_trend_7d, "price_trend_7d")
            volume_zscore = self.calculate_zscore(features.volume_trend, "volume_trend")
            
            # 체제 분류 로직
            regime_scores = {}
            
            # 고변동성 체제 (Z-score > 2)
            if volatility_zscore > 2:
                regime_scores["HIGH_VOLATILITY_SHOCK"] = min(volatility_zscore / 3, 1.0)
            
            # 저변동성 체제 (Z-score < -1.5)
            elif volatility_zscore < -1.5:
                regime_scores["LOW_VOLATILITY_ACCUMULATION"] = min(abs(volatility_zscore) / 2, 1.0)
            
            # 트렌드 기반 체제
            if trend_zscore > 1.5:
                regime_scores["BULL_MARKET"] = min(trend_zscore / 2, 1.0)
            elif trend_zscore < -1.5:
                regime_scores["BEAR_MARKET"] = min(abs(trend_zscore) / 2, 1.0)
            else:
                regime_scores["SIDEWAYS"] = 1 - abs(trend_zscore) / 2
            
            if not regime_scores:
                regime_scores["SIDEWAYS"] = 0.5
            
            best_regime = max(regime_scores.items(), key=lambda x: x[1])
            
            return {
                "predicted_regime": best_regime[0],
                "confidence": best_regime[1],
                "z_scores": {
                    "volatility": volatility_zscore,
                    "trend": trend_zscore,
                    "volume": volume_zscore
                },
                "method": "statistical"
            }
            
        except Exception as e:
            logger.error(f"통계적 감지 실패: {e}")
            return {"error": str(e)}
    
    def volatility_regime_detection(self, features: RegimeFeatures) -> Dict:
        """변동성 기반 체제 감지"""
        try:
            # 변동성 클러스터링 감지
            vol_1d = features.volatility_1d
            vol_7d = features.volatility_7d
            vol_30d = features.volatility_30d
            
            # 변동성 체제 임계값
            high_vol_threshold = 0.06
            low_vol_threshold = 0.02
            
            regime_scores = {}
            
            # 고변동성 체제
            if vol_1d > high_vol_threshold:
                regime_scores["HIGH_VOLATILITY_SHOCK"] = min(vol_1d / high_vol_threshold, 1.0) * 0.8
            
            # 저변동성 체제  
            if vol_7d < low_vol_threshold:
                regime_scores["LOW_VOLATILITY_ACCUMULATION"] = (low_vol_threshold - vol_7d) / low_vol_threshold * 0.8
            
            # 변동성 증가/감소 트렌드
            vol_trend = (vol_1d - vol_30d) / vol_30d if vol_30d > 0 else 0
            
            if vol_trend > 0.5:  # 변동성 급증
                regime_scores["HIGH_VOLATILITY_SHOCK"] = regime_scores.get("HIGH_VOLATILITY_SHOCK", 0) + 0.3
            elif vol_trend < -0.5:  # 변동성 진정
                regime_scores["LOW_VOLATILITY_ACCUMULATION"] = regime_scores.get("LOW_VOLATILITY_ACCUMULATION", 0) + 0.2
            
            # 중간 변동성은 다른 지표로 판단
            if not regime_scores:
                if features.price_trend_7d > 0.03:
                    regime_scores["BULL_MARKET"] = 0.6
                elif features.price_trend_7d < -0.03:
                    regime_scores["BEAR_MARKET"] = 0.6
                else:
                    regime_scores["SIDEWAYS"] = 0.5
            
            best_regime = max(regime_scores.items(), key=lambda x: x[1])
            
            return {
                "predicted_regime": best_regime[0],
                "confidence": best_regime[1],
                "volatility_metrics": {
                    "vol_1d": vol_1d,
                    "vol_7d": vol_7d, 
                    "vol_30d": vol_30d,
                    "vol_trend": vol_trend
                },
                "method": "volatility"
            }
            
        except Exception as e:
            logger.error(f"변동성 기반 감지 실패: {e}")
            return {"error": str(e)}
    
    def integrate_detection_results(self, detections: Dict) -> Dict:
        """다중 감지 결과 통합"""
        try:
            if not detections or all(d.get("error") for d in detections.values()):
                return {"error": "모든 감지 방법이 실패했습니다"}
            
            # 방법별 가중치 (성능 기반)
            method_weights = {
                "rule_based": 0.25,
                "ml_classifier": 0.30,
                "hmm": 0.20,
                "statistical": 0.15,
                "volatility": 0.10
            }
            
            # 체제별 점수 집계
            regime_scores = defaultdict(list)
            total_weight = 0
            
            for method, result in detections.items():
                if result.get("error"):
                    continue
                    
                regime = result.get("predicted_regime")
                confidence = result.get("confidence", 0)
                weight = method_weights.get(method, 0.1)
                
                if regime and confidence > 0:
                    regime_scores[regime].append(confidence * weight)
                    total_weight += weight
            
            # 최종 점수 계산
            final_scores = {}
            for regime, scores in regime_scores.items():
                final_scores[regime] = sum(scores) / total_weight if total_weight > 0 else 0
            
            if not final_scores:
                return {"error": "통합 결과 계산 실패"}
            
            # 최고 점수 체제 선택
            best_regime = max(final_scores.items(), key=lambda x: x[1])
            
            return {
                "predicted_regime": best_regime[0],
                "confidence": best_regime[1],
                "regime_scores": final_scores,
                "contributing_methods": len([d for d in detections.values() if not d.get("error")]),
                "method": "integrated"
            }
            
        except Exception as e:
            logger.error(f"결과 통합 실패: {e}")
            return {"error": str(e)}
    
    async def apply_regime_stability_rules(self, regime_result: Dict, features: RegimeFeatures) -> MarketRegime:
        """체제 안정성 규칙 적용"""
        try:
            predicted_regime = regime_result.get("predicted_regime", "SIDEWAYS")
            confidence = regime_result.get("confidence", 0.5)
            
            # 현재 체제가 있고 지속 기간이 최소 기간보다 짧으면 유지
            if (self.current_regime and self.regime_start_time):
                current_duration = (datetime.now() - self.regime_start_time).days
                min_duration = self.regime_definitions[self.current_regime.regime_type]["min_duration"]
                
                if (current_duration < min_duration and 
                    predicted_regime != self.current_regime.regime_type and
                    confidence < 0.8):  # 높은 확신이 아니면 유지
                    
                    logger.info(f"체제 안정성 규칙: {self.current_regime.regime_type} 유지 (지속기간: {current_duration}일)")
                    return self.current_regime
            
            # 전환 확률 고려
            if self.current_regime:
                current_type = self.current_regime.regime_type
                transition_prob = self.transition_matrix.get(current_type, {}).get(predicted_regime, 0.1)
                
                # 낮은 전환 확률이면 높은 확신이 필요
                if transition_prob < 0.2 and confidence < 0.7:
                    logger.info(f"전환 확률 낮음: {current_type} -> {predicted_regime} (확률: {transition_prob})")
                    return self.current_regime
            
            # 새로운 체제 생성
            regime_def = self.regime_definitions[predicted_regime]
            
            # 체제 지속 기간 예측
            duration_days = self.calculate_current_duration() if self.current_regime else 0
            
            # 핵심 특성 추출
            key_characteristics = self.extract_key_characteristics(features, predicted_regime)
            
            # 전환 확률 계산
            transition_probs = self.transition_matrix.get(predicted_regime, {})
            
            # 최적 전략 추천
            optimal_strategies = self.recommend_strategies(predicted_regime, features)
            
            new_regime = MarketRegime(
                regime_type=predicted_regime,
                confidence=confidence,
                duration_days=duration_days,
                key_characteristics=key_characteristics,
                expected_duration_days=regime_def["expected_duration"],
                transition_probability=transition_probs,
                optimal_strategies=optimal_strategies,
                risk_level=regime_def["risk_level"]
            )
            
            return new_regime
            
        except Exception as e:
            logger.error(f"안정성 규칙 적용 실패: {e}")
            # 실패시 기본 체제 반환
            return MarketRegime(
                regime_type="SIDEWAYS",
                confidence=0.5,
                duration_days=0,
                key_characteristics=["데이터 부족"],
                expected_duration_days=20,
                transition_probability={"BULL_MARKET": 0.3, "BEAR_MARKET": 0.3, "SIDEWAYS": 0.4},
                optimal_strategies=["관망"],
                risk_level="LOW"
            )
    
    def extract_key_characteristics(self, features: RegimeFeatures, regime_type: str) -> List[str]:
        """체제별 핵심 특성 추출"""
        characteristics = []
        
        try:
            if regime_type == "BULL_MARKET":
                if features.price_trend_7d > 0.05:
                    characteristics.append(f"강한 상승 추세 (+{features.price_trend_7d:.1%})")
                if features.rsi_14 > 60:
                    characteristics.append(f"상승 모멘텀 강화 (RSI: {features.rsi_14:.1f})")
                if features.volume_trend > 0:
                    characteristics.append("거래량 증가 확인")
                    
            elif regime_type == "BEAR_MARKET":
                if features.price_trend_7d < -0.05:
                    characteristics.append(f"강한 하락 추세 ({features.price_trend_7d:.1%})")
                if features.fear_greed_index < 30:
                    characteristics.append(f"공포 심리 확산 ({features.fear_greed_index:.0f})")
                if features.exchange_flow > 0:
                    characteristics.append("거래소 유입 증가")
                    
            elif regime_type == "SIDEWAYS":
                if abs(features.price_trend_7d) < 0.02:
                    characteristics.append("가격 횡보 구간")
                if features.volatility_1d < 0.03:
                    characteristics.append("낮은 변동성")
                if 0.3 < features.bollinger_position < 0.7:
                    characteristics.append("볼린저밴드 중간 구간")
                    
            elif regime_type == "HIGH_VOLATILITY_SHOCK":
                if features.volatility_1d > 0.08:
                    characteristics.append(f"극도 높은 변동성 ({features.volatility_1d:.1%})")
                if features.volume_volatility > 0.05:
                    characteristics.append("거래량 급변")
                if abs(features.futures_basis) > 0.02:
                    characteristics.append("선물 베이시스 확대")
                    
            elif regime_type == "LOW_VOLATILITY_ACCUMULATION":
                if features.volatility_7d < 0.02:
                    characteristics.append(f"낮은 변동성 ({features.volatility_7d:.1%})")
                if features.hodler_behavior > 0.6:
                    characteristics.append("홀더 누적 증가")
                if features.whale_activity > 0.6:
                    characteristics.append("고래 활동 활발")
            
            if not characteristics:
                characteristics.append("일반적인 시장 특성")
                
        except Exception as e:
            logger.error(f"특성 추출 실패: {e}")
            characteristics = ["특성 분석 실패"]
        
        return characteristics
    
    def recommend_strategies(self, regime_type: str, features: RegimeFeatures) -> List[str]:
        """체제별 최적 전략 추천"""
        strategies = []
        
        try:
            if regime_type == "BULL_MARKET":
                strategies.extend([
                    "트렌드 팔로잉 전략",
                    "매수 후 보유 (HODL)",
                    "점진적 매수 (DCA)"
                ])
                if features.rsi_14 < 70:
                    strategies.append("추가 매수 기회")
                    
            elif regime_type == "BEAR_MARKET":
                strategies.extend([
                    "매도 신호 주시",
                    "현금 비중 확대",
                    "반등 대기"
                ])
                if features.rsi_14 < 30:
                    strategies.append("역추세 매수 기회 탐색")
                    
            elif regime_type == "SIDEWAYS":
                strategies.extend([
                    "레인지 트레이딩",
                    "지지/저항선 활용",
                    "변동성 매매"
                ])
                
            elif regime_type == "HIGH_VOLATILITY_SHOCK":
                strategies.extend([
                    "위험 관리 최우선",
                    "포지션 축소",
                    "관망 전략"
                ])
                
            elif regime_type == "LOW_VOLATILITY_ACCUMULATION":
                strategies.extend([
                    "브레이크아웃 대기",
                    "점진적 매수",
                    "장기 포지션 구축"
                ])
            
            if not strategies:
                strategies = ["신중한 관망"]
                
        except Exception as e:
            logger.error(f"전략 추천 실패: {e}")
            strategies = ["위험 관리 우선"]
        
        return strategies
    
    # ===== 헬퍼 메서드들 =====
    
    async def extract_price(self, market_data: Dict) -> Optional[float]:
        """시장 데이터에서 현재 가격 추출"""
        try:
            # 다양한 경로에서 가격 찾기
            price_paths = [
                ["current_price"],
                ["market_data", "price"],
                ["price"],
                ["data_sources", "legacy_analyzer", "market_data", "avg_price"],
                ["summary", "current_btc_price"]
            ]
            
            for path in price_paths:
                try:
                    value = market_data
                    for key in path:
                        value = value[key]
                    if value and isinstance(value, (int, float)) and value > 0:
                        return float(value)
                except (KeyError, TypeError):
                    continue
            
            return None
            
        except Exception as e:
            logger.error(f"가격 추출 실패: {e}")
            return None
    
    async def get_historical_features(self, days: int = 30) -> List[Dict]:
        """과거 특징값 데이터 조회"""
        try:
            # 실제 구현에서는 historical_data 폴더나 데이터베이스에서 조회
            historical_data = []
            
            # 과거 분석 데이터 파일들 조회
            historical_path = os.path.join(self.base_path, "historical_data")
            if os.path.exists(historical_path):
                files = [f for f in os.listdir(historical_path) 
                        if f.startswith("btc_analysis_") and f.endswith(".json")]
                
                # 최근 파일들 선택
                recent_files = sorted(files)[-days:]
                
                for file in recent_files:
                    try:
                        with open(os.path.join(historical_path, file), 'r') as f:
                            data = json.load(f)
                            historical_data.append(data)
                    except:
                        continue
            
            return historical_data[-days:] if historical_data else []
            
        except Exception as e:
            logger.error(f"과거 데이터 조회 실패: {e}")
            return []
    
    def calculate_trend(self, historical_data: List[Dict], days: int) -> float:
        """가격 트렌드 계산"""
        try:
            if len(historical_data) < days:
                return 0.0
            
            prices = []
            for data in historical_data[-days:]:
                price = self.extract_price_from_historical(data)
                if price:
                    prices.append(price)
            
            if len(prices) < 2:
                return 0.0
            
            # 선형 회귀 기울기로 트렌드 계산
            x = np.arange(len(prices))
            slope = np.polyfit(x, prices, 1)[0]
            
            # 일평균 변화율로 정규화
            avg_price = np.mean(prices)
            return (slope / avg_price) if avg_price > 0 else 0.0
            
        except Exception as e:
            logger.error(f"트렌드 계산 실패: {e}")
            return 0.0
    
    def calculate_volatility(self, historical_data: List[Dict], days: int) -> float:
        """변동성 계산"""
        try:
            if len(historical_data) < days:
                return 0.02  # 기본값
            
            prices = []
            for data in historical_data[-days:]:
                price = self.extract_price_from_historical(data)
                if price:
                    prices.append(price)
            
            if len(prices) < 2:
                return 0.02
            
            # 일수익률 계산
            returns = []
            for i in range(1, len(prices)):
                return_val = (prices[i] - prices[i-1]) / prices[i-1]
                returns.append(return_val)
            
            # 표준편차 (변동성)
            return np.std(returns) if returns else 0.02
            
        except Exception as e:
            logger.error(f"변동성 계산 실패: {e}")
            return 0.02
    
    def calculate_trend_consistency(self, historical_data: List[Dict]) -> float:
        """트렌드 일관성 계산"""
        try:
            if len(historical_data) < 7:
                return 0.5
            
            # 최근 7일간 일별 트렌드 방향 계산
            directions = []
            prices = [self.extract_price_from_historical(d) for d in historical_data[-7:]]
            prices = [p for p in prices if p]
            
            for i in range(1, len(prices)):
                direction = 1 if prices[i] > prices[i-1] else -1
                directions.append(direction)
            
            if not directions:
                return 0.5
            
            # 방향 일관성 (같은 방향 비율)
            positive_days = sum(1 for d in directions if d > 0)
            negative_days = sum(1 for d in directions if d < 0)
            
            consistency = max(positive_days, negative_days) / len(directions)
            return consistency
            
        except Exception as e:
            logger.error(f"트렌드 일관성 계산 실패: {e}")
            return 0.5
    
    def extract_price_from_historical(self, data: Dict) -> Optional[float]:
        """과거 데이터에서 가격 추출"""
        try:
            paths = [
                ["data_sources", "legacy_analyzer", "market_data", "avg_price"],
                ["summary", "current_btc_price"],
                ["current_price"],
                ["price"]
            ]
            
            for path in paths:
                try:
                    value = data
                    for key in path:
                        value = value[key]
                    if value and value > 0:
                        return float(value)
                except:
                    continue
            return None
        except:
            return None
    
    # 기본값 반환하는 헬퍼 메서드들 (실제 구현에서는 실제 계산 로직 필요)
    def detect_volatility_regime_change(self, historical_data: List[Dict]) -> float:
        """변동성 체제 변화 감지"""
        return 0.0  # 기본값
    
    def calculate_volume_trend(self, market_data: Dict) -> float:
        """거래량 트렌드"""
        return 0.0  # 기본값
    
    def calculate_volume_volatility(self, market_data: Dict) -> float:
        """거래량 변동성"""
        return 0.02  # 기본값
    
    def calculate_volume_price_correlation(self, historical_data: List[Dict]) -> float:
        """거래량-가격 상관관계"""
        return 0.3  # 기본값
    
    def calculate_rsi(self, historical_data: List[Dict], period: int) -> float:
        """RSI 계산"""
        return 50.0  # 기본값 (중립)
    
    def calculate_macd_signal(self, historical_data: List[Dict]) -> float:
        """MACD 신호"""
        return 0.0  # 기본값
    
    def calculate_bollinger_position(self, historical_data: List[Dict], current_price: float) -> float:
        """볼린저밴드 내 위치"""
        return 0.5  # 중간값
    
    def extract_whale_activity(self, market_data: Dict) -> float:
        """고래 활동 추출"""
        return 0.5  # 기본값
    
    def extract_exchange_flow(self, market_data: Dict) -> float:
        """거래소 플로우"""
        return 0.0  # 기본값
    
    def extract_hodler_behavior(self, market_data: Dict) -> float:
        """홀더 행동"""
        return 0.5  # 기본값
    
    def extract_futures_basis(self, market_data: Dict) -> float:
        """선물 베이시스"""
        return 0.0  # 기본값
    
    def extract_funding_rate(self, market_data: Dict) -> float:
        """펀딩레이트"""
        return 0.0  # 기본값
    
    def extract_put_call_ratio(self, market_data: Dict) -> float:
        """풋콜비율"""
        return 0.5  # 기본값
    
    def extract_fear_greed(self, market_data: Dict) -> float:
        """공포탐욕지수"""
        return 50.0  # 중립값
    
    def extract_correlation(self, market_data: Dict, asset: str) -> float:
        """자산별 상관관계"""
        return 0.1  # 기본값
    
    def extract_dxy_impact(self, market_data: Dict) -> float:
        """달러지수 영향"""
        return 0.0  # 기본값
    
    def calculate_zscore(self, value: float, feature_name: str) -> float:
        """Z-점수 계산"""
        # 실제 구현에서는 과거 데이터의 평균과 표준편차 사용
        historical_mean = 0.0  # 실제로는 DB에서 조회
        historical_std = 1.0   # 실제로는 DB에서 조회
        
        return (value - historical_mean) / historical_std if historical_std > 0 else 0.0
    
    def get_recent_features_sequence(self, current_features: RegimeFeatures, length: int) -> np.ndarray:
        """최근 특징값 시퀀스 조회 (HMM용)"""
        # 실제 구현에서는 DB에서 시계열 데이터 조회
        # 임시로 현재 특징값을 복제해서 시퀀스 생성
        feature_array = np.array([
            current_features.volatility_1d,
            current_features.price_trend_7d,
            current_features.volume_trend,
            current_features.rsi_14,
            current_features.fear_greed_index
        ])
        
        # length만큼 시퀀스 생성 (실제로는 과거 데이터 사용)
        sequence = np.tile(feature_array, (length, 1))
        return sequence
    
    def calculate_current_duration(self) -> int:
        """현재 체제 지속 기간 계산"""
        if self.regime_start_time:
            return (datetime.now() - self.regime_start_time).days
        return 0
    
    def update_current_regime(self, new_regime: MarketRegime):
        """현재 체제 상태 업데이트"""
        try:
            if (not self.current_regime or 
                self.current_regime.regime_type != new_regime.regime_type):
                
                # 체제 변경시 시작 시간 리셋
                self.regime_start_time = datetime.now()
                logger.info(f"체제 변경: {self.current_regime.regime_type if self.current_regime else 'None'} -> {new_regime.regime_type}")
            
            self.current_regime = new_regime
            self.regime_history.append({
                "timestamp": datetime.now(),
                "regime": new_regime.regime_type,
                "confidence": new_regime.confidence
            })
            
        except Exception as e:
            logger.error(f"현재 체제 업데이트 실패: {e}")
    
    async def save_regime_detection(self, regime: MarketRegime, features: RegimeFeatures, detections: Dict):
        """체제 감지 결과 저장"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 주요 감지 결과 저장
            cursor.execute('''
                INSERT INTO regime_detections 
                (timestamp, regime_type, confidence, duration_days, key_features,
                 transition_from, detection_method, market_conditions)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                regime.timestamp if hasattr(regime, 'timestamp') else datetime.now().isoformat(),
                regime.regime_type,
                regime.confidence,
                regime.duration_days,
                json.dumps(regime.key_characteristics),
                self.current_regime.regime_type if self.current_regime else None,
                "multi_method",
                json.dumps({
                    "volatility_1d": features.volatility_1d,
                    "price_trend_7d": features.price_trend_7d,
                    "fear_greed": features.fear_greed_index
                })
            ))
            
            # 특징값 히스토리 저장
            cursor.execute('''
                INSERT INTO feature_history (timestamp, features_json, regime_label)
                VALUES (?, ?, ?)
            ''', (
                features.timestamp.isoformat(),
                json.dumps(asdict(features), default=str),
                regime.regime_type
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"체제 감지 저장 실패: {e}")
    
    async def get_regime_specific_prediction(self, market_data: Dict) -> Optional[Dict]:
        """체제별 특화 예측"""
        try:
            if not self.current_regime:
                return None
            
            regime_type = self.current_regime.regime_type
            
            # 체제별 특화 모델이 있으면 사용
            if regime_type in self.regime_specific_models:
                model = self.regime_specific_models[regime_type]
                
                # 특징값 추출 및 예측
                features = await self.extract_regime_features(market_data)
                if features:
                    prediction = await self.predict_with_regime_model(model, features, regime_type)
                    return prediction
            
            # 체제별 기본 예측 로직
            return self.get_default_regime_prediction(regime_type, market_data)
            
        except Exception as e:
            logger.error(f"체제별 예측 실패: {e}")
            return None
    
    def get_default_regime_prediction(self, regime_type: str, market_data: Dict) -> Dict:
        """체제별 기본 예측"""
        try:
            current_price = asyncio.run(self.extract_price(market_data))
            if not current_price:
                return {"error": "가격 데이터 없음"}
            
            predictions = {}
            
            if regime_type == "BULL_MARKET":
                # 상승장: 긍정적 예측
                predictions = {
                    "direction": "BULLISH",
                    "confidence": 0.75,
                    "price_target_24h": current_price * 1.03,
                    "probability": 75,
                    "reasoning": "강세장 체제에서 상승 추세 지속 예상"
                }
                
            elif regime_type == "BEAR_MARKET":
                # 하락장: 부정적 예측
                predictions = {
                    "direction": "BEARISH", 
                    "confidence": 0.70,
                    "price_target_24h": current_price * 0.97,
                    "probability": 70,
                    "reasoning": "약세장 체제에서 하락 추세 지속 예상"
                }
                
            elif regime_type == "SIDEWAYS":
                # 횡보장: 중립적 예측
                predictions = {
                    "direction": "NEUTRAL",
                    "confidence": 0.60,
                    "price_target_24h": current_price * (0.999 + np.random.normal(0, 0.005)),
                    "probability": 60,
                    "reasoning": "횡보 체제에서 제한적 움직임 예상"
                }
                
            elif regime_type == "HIGH_VOLATILITY_SHOCK":
                # 고변동성: 불확실성 높음
                predictions = {
                    "direction": "UNCERTAIN",
                    "confidence": 0.40,
                    "price_target_24h": current_price * (0.95 + np.random.normal(0, 0.1)),
                    "probability": 40,
                    "reasoning": "고변동성 충격 구간으로 예측 불확실성 높음"
                }
                
            elif regime_type == "LOW_VOLATILITY_ACCUMULATION":
                # 저변동성: 대기 모드
                predictions = {
                    "direction": "ACCUMULATION",
                    "confidence": 0.65,
                    "price_target_24h": current_price * 1.01,
                    "probability": 65,
                    "reasoning": "저변동성 축적 구간으로 브레이크아웃 대기"
                }
            
            predictions.update({
                "regime_type": regime_type,
                "regime_confidence": self.current_regime.confidence,
                "current_price": current_price,
                "timestamp": datetime.now().isoformat()
            })
            
            return predictions
            
        except Exception as e:
            logger.error(f"기본 체제 예측 실패: {e}")
            return {"error": str(e)}
    
    async def predict_with_regime_model(self, model, features: RegimeFeatures, regime_type: str) -> Dict:
        """체제별 특화 모델로 예측"""
        try:
            # 특징값을 모델 입력 형태로 변환
            feature_array = np.array([
                features.price_trend_1d, features.price_trend_7d, features.volatility_1d,
                features.volatility_7d, features.rsi_14, features.volume_trend,
                features.fear_greed_index, features.whale_activity
            ]).reshape(1, -1)
            
            # 예측 수행
            if hasattr(model, 'predict_proba'):
                prediction_proba = model.predict_proba(feature_array)[0]
                predicted_class = model.predict(feature_array)[0]
                confidence = max(prediction_proba)
            else:
                predicted_class = model.predict(feature_array)[0]
                confidence = 0.7  # 기본값
            
            return {
                "predicted_class": predicted_class,
                "confidence": confidence,
                "regime_type": regime_type,
                "model_type": "regime_specific",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"체제별 모델 예측 실패: {e}")
            return {"error": str(e)}

# 메인 실행 함수
async def run_regime_detection_test():
    """체제 감지 시스템 테스트"""
    print("🔍 고도화된 시장 체제 감지 시스템 테스트")
    print("=" * 60)
    
    detector = AdvancedMarketRegimeDetector()
    
    # 테스트용 시장 데이터
    test_market_data = {
        "current_price": 65000,
        "data_sources": {
            "legacy_analyzer": {
                "market_data": {
                    "avg_price": 65000
                }
            }
        },
        "summary": {
            "current_btc_price": 65000
        }
    }
    
    # 체제 감지 실행
    regime = await detector.detect_current_regime(test_market_data)
    
    if regime:
        print(f"🎯 감지된 체제: {regime.regime_type}")
        print(f"🔥 신뢰도: {regime.confidence:.1%}")
        print(f"⏱️ 지속 기간: {regime.duration_days}일")
        print(f"📈 위험 수준: {regime.risk_level}")
        print(f"🎨 핵심 특성:")
        for char in regime.key_characteristics:
            print(f"   • {char}")
        print(f"💡 추천 전략:")
        for strategy in regime.optimal_strategies:
            print(f"   • {strategy}")
        
        # 체제별 예측
        prediction = await detector.get_regime_specific_prediction(test_market_data)
        if prediction and not prediction.get("error"):
            print(f"\n📊 체제별 예측:")
            print(f"   • 방향: {prediction.get('direction', 'N/A')}")
            print(f"   • 확률: {prediction.get('probability', 0)}%")
            print(f"   • 목표가: ${prediction.get('price_target_24h', 0):,.0f}")
            print(f"   • 근거: {prediction.get('reasoning', 'N/A')}")
    else:
        print("❌ 체제 감지 실패")
    
    print("\n" + "=" * 60)
    print("🎉 시장 체제 감지 시스템 테스트 완료!")

if __name__ == "__main__":
    asyncio.run(run_regime_detection_test())