"""
시장 조건 적응형 전략 엔진 v1.0
- 실시간 시장 상황 분석
- 조건별 전략 자동 전환
- 위험 관리 통합
- 성과 추적 및 최적화
"""

import os
import json
import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import asyncio
import logging
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TradingStrategy:
    """거래 전략 정의"""
    name: str
    description: str
    market_conditions: List[str]  # 적합한 시장 조건들
    risk_level: str  # 'low', 'medium', 'high'
    parameters: Dict[str, Any]
    historical_performance: Dict[str, float]
    last_used: Optional[datetime] = None
    usage_count: int = 0
    success_rate: float = 0.0

@dataclass 
class MarketCondition:
    """시장 조건 정의"""
    name: str
    indicators: Dict[str, Tuple[float, float]]  # indicator: (min, max)
    volatility_range: Tuple[float, float]
    trend_strength: Tuple[float, float]
    volume_pattern: str  # 'low', 'normal', 'high'
    duration_hours: int
    
@dataclass
class StrategyPerformance:
    """전략 성과 기록"""
    strategy_name: str
    market_condition: str
    timestamp: datetime
    entry_price: float
    exit_price: Optional[float] = None
    duration_minutes: Optional[int] = None
    pnl_percent: Optional[float] = None
    max_drawdown: Optional[float] = None
    win: Optional[bool] = None
    confidence: float = 0.0

class MarketConditionClassifier:
    """시장 조건 분류기"""
    
    def __init__(self):
        self.conditions = self._define_market_conditions()
        self.classifier = None
        self.scaler = StandardScaler()
        self.feature_history = deque(maxlen=100)
        
    def _define_market_conditions(self) -> Dict[str, MarketCondition]:
        """시장 조건 정의"""
        return {
            'bull_strong': MarketCondition(
                name='강한 강세장',
                indicators={
                    'rsi': (60, 85),
                    'macd': (0, 100),
                    'price_sma20_ratio': (1.02, 1.20),
                    'volume_sma': (1.2, 3.0)
                },
                volatility_range=(0.01, 0.03),
                trend_strength=(0.7, 1.0),
                volume_pattern='high',
                duration_hours=6
            ),
            
            'bull_weak': MarketCondition(
                name='약한 강세장',
                indicators={
                    'rsi': (50, 70),
                    'macd': (-10, 50),
                    'price_sma20_ratio': (1.00, 1.05),
                    'volume_sma': (0.8, 1.5)
                },
                volatility_range=(0.015, 0.04),
                trend_strength=(0.4, 0.7),
                volume_pattern='normal',
                duration_hours=4
            ),
            
            'bear_strong': MarketCondition(
                name='강한 약세장',
                indicators={
                    'rsi': (15, 40),
                    'macd': (-100, 0),
                    'price_sma20_ratio': (0.80, 0.98),
                    'volume_sma': (1.2, 3.0)
                },
                volatility_range=(0.02, 0.06),
                trend_strength=(0.7, 1.0),
                volume_pattern='high',
                duration_hours=6
            ),
            
            'bear_weak': MarketCondition(
                name='약한 약세장',
                indicators={
                    'rsi': (30, 50),
                    'macd': (-50, 10),
                    'price_sma20_ratio': (0.95, 1.00),
                    'volume_sma': (0.8, 1.5)
                },
                volatility_range=(0.015, 0.04),
                trend_strength=(0.4, 0.7),
                volume_pattern='normal',
                duration_hours=4
            ),
            
            'sideways_stable': MarketCondition(
                name='안정적 횡보',
                indicators={
                    'rsi': (40, 60),
                    'macd': (-20, 20),
                    'price_sma20_ratio': (0.98, 1.02),
                    'volume_sma': (0.7, 1.2)
                },
                volatility_range=(0.01, 0.025),
                trend_strength=(0.0, 0.3),
                volume_pattern='low',
                duration_hours=8
            ),
            
            'sideways_volatile': MarketCondition(
                name='변동성 횡보',
                indicators={
                    'rsi': (35, 65),
                    'macd': (-30, 30),
                    'price_sma20_ratio': (0.95, 1.05),
                    'volume_sma': (1.0, 2.0)
                },
                volatility_range=(0.03, 0.08),
                trend_strength=(0.0, 0.4),
                volume_pattern='high',
                duration_hours=3
            ),
            
            'breakout': MarketCondition(
                name='돌파 상황',
                indicators={
                    'rsi': (50, 80),
                    'macd': (0, 100),
                    'price_sma20_ratio': (1.05, 1.15),
                    'volume_sma': (2.0, 5.0)
                },
                volatility_range=(0.04, 0.10),
                trend_strength=(0.8, 1.0),
                volume_pattern='high',
                duration_hours=2
            ),
            
            'breakdown': MarketCondition(
                name='붕괴 상황',
                indicators={
                    'rsi': (20, 50),
                    'macd': (-100, 0),
                    'price_sma20_ratio': (0.85, 0.95),
                    'volume_sma': (2.0, 5.0)
                },
                volatility_range=(0.04, 0.12),
                trend_strength=(0.8, 1.0),
                volume_pattern='high',
                duration_hours=2
            ),
            
            'uncertainty': MarketCondition(
                name='불확실성',
                indicators={
                    'rsi': (30, 70),
                    'macd': (-50, 50),
                    'price_sma20_ratio': (0.90, 1.10),
                    'volume_sma': (0.5, 2.0)
                },
                volatility_range=(0.05, 0.15),
                trend_strength=(0.0, 0.6),
                volume_pattern='normal',
                duration_hours=1
            )
        }
    
    def extract_features(self, market_data: Dict) -> np.ndarray:
        """시장 데이터에서 특성 추출"""
        features = []
        
        # 기본 지표들
        features.extend([
            market_data.get('rsi', 50),
            market_data.get('macd', 0),
            market_data.get('price', 50000) / market_data.get('sma_20', 50000),
            market_data.get('volume', 1000000) / market_data.get('volume_sma', 1000000),
            market_data.get('volatility', 0.02),
            market_data.get('bollinger_width', 0.05),
            market_data.get('atr', 1000),
            market_data.get('fear_greed_index', 50)
        ])
        
        # 추가 기술적 지표들
        if 'indicators' in market_data:
            indicators = market_data['indicators']
            features.extend([
                indicators.get('stoch_k', 50),
                indicators.get('stoch_d', 50),
                indicators.get('williams_r', -50),
                indicators.get('cci', 0),
                indicators.get('momentum', 0),
                indicators.get('roc', 0)
            ])
        else:
            features.extend([50, 50, -50, 0, 0, 0])  # 기본값
        
        return np.array(features)
    
    def classify_condition(self, market_data: Dict) -> Tuple[str, float]:
        """시장 조건 분류"""
        features = self.extract_features(market_data)
        self.feature_history.append(features)
        
        # 각 조건에 대한 점수 계산
        condition_scores = {}
        
        for condition_name, condition in self.conditions.items():
            score = self._calculate_condition_score(features, condition)
            condition_scores[condition_name] = score
        
        # 가장 높은 점수의 조건 선택
        best_condition = max(condition_scores.items(), key=lambda x: x[1])
        
        return best_condition[0], best_condition[1]
    
    def _calculate_condition_score(self, features: np.ndarray, condition: MarketCondition) -> float:
        """조건별 적합도 점수 계산"""
        score = 0.0
        total_weight = 0.0
        
        # 지표 기반 점수
        feature_map = {
            'rsi': 0,
            'macd': 1, 
            'price_sma20_ratio': 2,
            'volume_sma': 3
        }
        
        for indicator, (min_val, max_val) in condition.indicators.items():
            if indicator in feature_map:
                feature_idx = feature_map[indicator]
                feature_val = features[feature_idx]
                
                # 범위 내에 있으면 점수 부여
                if min_val <= feature_val <= max_val:
                    # 중앙값에 가까울수록 높은 점수
                    center = (min_val + max_val) / 2
                    range_size = max_val - min_val
                    distance_from_center = abs(feature_val - center)
                    indicator_score = 1.0 - (distance_from_center / (range_size / 2))
                    score += indicator_score
                else:
                    # 범위를 벗어나면 거리에 따라 감점
                    if feature_val < min_val:
                        penalty = (min_val - feature_val) / min_val
                    else:
                        penalty = (feature_val - max_val) / max_val
                    score -= min(penalty, 1.0)
                
                total_weight += 1.0
        
        # 변동성 점수
        volatility = features[4] if len(features) > 4 else 0.02
        vol_min, vol_max = condition.volatility_range
        if vol_min <= volatility <= vol_max:
            vol_score = 1.0 - abs(volatility - (vol_min + vol_max) / 2) / ((vol_max - vol_min) / 2)
            score += vol_score * 0.5
        total_weight += 0.5
        
        # 정규화
        final_score = score / total_weight if total_weight > 0 else 0.0
        return max(0.0, min(1.0, final_score))

class StrategyManager:
    """전략 관리자"""
    
    def __init__(self):
        self.strategies = self._define_strategies()
        self.performance_history = defaultdict(list)
        self.current_strategy = None
        self.strategy_switch_cooldown = timedelta(minutes=30)
        self.last_switch_time = None
        
    def _define_strategies(self) -> Dict[str, TradingStrategy]:
        """거래 전략 정의"""
        return {
            'momentum_bull': TradingStrategy(
                name='강세 모멘텀',
                description='강한 상승 추세에서 모멘텀 추종',
                market_conditions=['bull_strong', 'breakout'],
                risk_level='high',
                parameters={
                    'entry_threshold': 0.8,
                    'exit_threshold': 0.3,
                    'stop_loss': 0.05,
                    'take_profit': 0.10,
                    'position_size': 0.5,
                    'confirmation_periods': 2
                },
                historical_performance={'avg_return': 0.08, 'win_rate': 0.65, 'max_dd': 0.15}
            ),
            
            'trend_following': TradingStrategy(
                name='추세 추종',
                description='중장기 추세 방향 추종',
                market_conditions=['bull_weak', 'bear_weak'],
                risk_level='medium',
                parameters={
                    'entry_threshold': 0.6,
                    'exit_threshold': 0.4,
                    'stop_loss': 0.04,
                    'take_profit': 0.08,
                    'position_size': 0.3,
                    'confirmation_periods': 3
                },
                historical_performance={'avg_return': 0.05, 'win_rate': 0.58, 'max_dd': 0.08}
            ),
            
            'mean_reversion': TradingStrategy(
                name='평균 회귀',
                description='횡보장에서 평균 회귀 전략',
                market_conditions=['sideways_stable', 'sideways_volatile'],
                risk_level='medium',
                parameters={
                    'entry_threshold': 0.7,
                    'exit_threshold': 0.5,
                    'stop_loss': 0.03,
                    'take_profit': 0.06,
                    'position_size': 0.4,
                    'confirmation_periods': 1
                },
                historical_performance={'avg_return': 0.04, 'win_rate': 0.72, 'max_dd': 0.06}
            ),
            
            'contrarian_bear': TradingStrategy(
                name='역추세 매매',
                description='강한 약세에서 반등 포착',
                market_conditions=['bear_strong', 'breakdown'],
                risk_level='high',
                parameters={
                    'entry_threshold': 0.9,
                    'exit_threshold': 0.2,
                    'stop_loss': 0.06,
                    'take_profit': 0.12,
                    'position_size': 0.3,
                    'confirmation_periods': 4
                },
                historical_performance={'avg_return': 0.10, 'win_rate': 0.45, 'max_dd': 0.20}
            ),
            
            'breakout_momentum': TradingStrategy(
                name='돌파 모멘텀',
                description='주요 저항/지지선 돌파시 진입',
                market_conditions=['breakout', 'breakdown'],
                risk_level='high',
                parameters={
                    'entry_threshold': 0.85,
                    'exit_threshold': 0.25,
                    'stop_loss': 0.04,
                    'take_profit': 0.15,
                    'position_size': 0.6,
                    'confirmation_periods': 1
                },
                historical_performance={'avg_return': 0.12, 'win_rate': 0.55, 'max_dd': 0.18}
            ),
            
            'volatility_scalping': TradingStrategy(
                name='변동성 스캘핑',
                description='고변동성 구간에서 단기 매매',
                market_conditions=['sideways_volatile', 'uncertainty'],
                risk_level='high',
                parameters={
                    'entry_threshold': 0.75,
                    'exit_threshold': 0.6,
                    'stop_loss': 0.02,
                    'take_profit': 0.04,
                    'position_size': 0.8,
                    'confirmation_periods': 1
                },
                historical_performance={'avg_return': 0.06, 'win_rate': 0.68, 'max_dd': 0.12}
            ),
            
            'conservative_hold': TradingStrategy(
                name='보수적 홀드',
                description='불확실한 상황에서 보수적 대기',
                market_conditions=['uncertainty'],
                risk_level='low',
                parameters={
                    'entry_threshold': 0.95,
                    'exit_threshold': 0.1,
                    'stop_loss': 0.02,
                    'take_profit': 0.03,
                    'position_size': 0.1,
                    'confirmation_periods': 5
                },
                historical_performance={'avg_return': 0.02, 'win_rate': 0.80, 'max_dd': 0.03}
            )
        }
    
    def select_strategy(self, market_condition: str, confidence: float) -> Optional[TradingStrategy]:
        """시장 조건에 따른 최적 전략 선택"""
        
        # 쿨다운 체크
        if (self.last_switch_time and 
            datetime.now() - self.last_switch_time < self.strategy_switch_cooldown):
            return self.current_strategy
        
        # 해당 시장 조건에 적합한 전략들 필터링
        suitable_strategies = [
            strategy for strategy in self.strategies.values()
            if market_condition in strategy.market_conditions
        ]
        
        if not suitable_strategies:
            # 적합한 전략이 없으면 보수적 전략 선택
            return self.strategies['conservative_hold']
        
        # 성능 기반 전략 점수 계산
        strategy_scores = {}
        
        for strategy in suitable_strategies:
            # 기본 성능 점수
            base_score = (
                strategy.historical_performance['win_rate'] * 0.4 +
                strategy.historical_performance['avg_return'] * 0.4 -
                strategy.historical_performance['max_dd'] * 0.2
            )
            
            # 최근 성능 보정
            recent_performance = self._get_recent_performance(strategy.name)
            if recent_performance:
                recent_score = np.mean(recent_performance)
                # 최근 성능이 좋으면 가산점, 나쁘면 감점
                performance_adjustment = (recent_score - 0.5) * 0.2
                base_score += performance_adjustment
            
            # 신뢰도 기반 보정
            confidence_adjustment = confidence * 0.1
            final_score = base_score + confidence_adjustment
            
            strategy_scores[strategy.name] = final_score
        
        # 최고 점수 전략 선택
        best_strategy_name = max(strategy_scores.items(), key=lambda x: x[1])[0]
        selected_strategy = self.strategies[best_strategy_name]
        
        # 전략이 변경되었으면 기록
        if self.current_strategy != selected_strategy:
            logger.info(f"전략 변경: {self.current_strategy.name if self.current_strategy else 'None'} → {selected_strategy.name}")
            self.current_strategy = selected_strategy
            self.last_switch_time = datetime.now()
            
            # 사용 횟수 증가
            selected_strategy.usage_count += 1
            selected_strategy.last_used = datetime.now()
        
        return selected_strategy
    
    def _get_recent_performance(self, strategy_name: str, days: int = 7) -> List[float]:
        """최근 성과 조회"""
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_records = [
            record for record in self.performance_history[strategy_name]
            if record.timestamp > cutoff_date and record.win is not None
        ]
        
        return [1.0 if record.win else 0.0 for record in recent_records]
    
    def record_performance(self, performance: StrategyPerformance):
        """전략 성과 기록"""
        self.performance_history[performance.strategy_name].append(performance)
        
        # 전략의 성공률 업데이트
        if performance.strategy_name in self.strategies:
            strategy = self.strategies[performance.strategy_name]
            recent_performance = self._get_recent_performance(performance.strategy_name, 30)
            if recent_performance:
                strategy.success_rate = np.mean(recent_performance)
    
    def get_strategy_stats(self) -> Dict[str, Dict]:
        """전략별 통계 조회"""
        stats = {}
        
        for name, strategy in self.strategies.items():
            recent_perf = self._get_recent_performance(name, 30)
            
            stats[name] = {
                'usage_count': strategy.usage_count,
                'success_rate': strategy.success_rate,
                'recent_trades': len(recent_perf),
                'recent_win_rate': np.mean(recent_perf) if recent_perf else 0.0,
                'last_used': strategy.last_used.isoformat() if strategy.last_used else None,
                'risk_level': strategy.risk_level,
                'suitable_conditions': strategy.market_conditions
            }
        
        return stats

class RiskManager:
    """위험 관리자"""
    
    def __init__(self):
        self.max_daily_loss = 0.05  # 일일 최대 손실 5%
        self.max_position_size = 0.8  # 최대 포지션 크기
        self.max_concurrent_positions = 3
        self.daily_pnl = 0.0
        self.current_positions = []
        self.risk_metrics = {
            'var_95': 0.03,  # 95% VaR
            'max_drawdown': 0.10,
            'sharpe_threshold': 0.5
        }
    
    def assess_risk(self, strategy: TradingStrategy, market_condition: str, confidence: float) -> Dict[str, Any]:
        """위험 평가"""
        risk_assessment = {
            'approved': True,
            'suggested_position_size': strategy.parameters['position_size'],
            'risk_warnings': [],
            'stop_loss': strategy.parameters['stop_loss'],
            'take_profit': strategy.parameters['take_profit']
        }
        
        # 일일 손실 한도 체크
        if abs(self.daily_pnl) > self.max_daily_loss:
            risk_assessment['approved'] = False
            risk_assessment['risk_warnings'].append('일일 손실 한도 초과')
        
        # 포지션 수 체크
        if len(self.current_positions) >= self.max_concurrent_positions:
            risk_assessment['approved'] = False
            risk_assessment['risk_warnings'].append('최대 동시 포지션 수 초과')
        
        # 전략 위험도에 따른 포지션 크기 조정
        risk_multiplier = {
            'low': 1.0,
            'medium': 0.8, 
            'high': 0.6
        }.get(strategy.risk_level, 0.5)
        
        # 신뢰도에 따른 추가 조정
        confidence_multiplier = min(1.0, confidence / 0.7)  # 신뢰도 70% 기준
        
        # 최종 포지션 크기
        adjusted_size = (strategy.parameters['position_size'] * 
                        risk_multiplier * 
                        confidence_multiplier)
        
        risk_assessment['suggested_position_size'] = min(adjusted_size, self.max_position_size)
        
        # 추가 위험 경고
        if confidence < 0.6:
            risk_assessment['risk_warnings'].append('낮은 신뢰도')
        
        if strategy.historical_performance['max_dd'] > 0.15:
            risk_assessment['risk_warnings'].append('높은 역사적 최대낙폭')
        
        return risk_assessment
    
    def update_daily_pnl(self, pnl: float):
        """일일 손익 업데이트"""
        self.daily_pnl += pnl
    
    def add_position(self, position_id: str, strategy_name: str, size: float):
        """포지션 추가"""
        self.current_positions.append({
            'id': position_id,
            'strategy': strategy_name,
            'size': size,
            'start_time': datetime.now()
        })
    
    def remove_position(self, position_id: str):
        """포지션 제거"""
        self.current_positions = [pos for pos in self.current_positions if pos['id'] != position_id]

class MarketAdaptiveStrategyEngine:
    """시장 적응형 전략 엔진"""
    
    def __init__(self):
        self.base_path = "/Users/parkyoungjun/Desktop/BTC_Analysis_System"
        self.db_path = os.path.join(self.base_path, "strategy_engine.db")
        
        self.condition_classifier = MarketConditionClassifier()
        self.strategy_manager = StrategyManager()
        self.risk_manager = RiskManager()
        
        self.current_market_condition = None
        self.current_strategy = None
        self.condition_history = deque(maxlen=100)
        self.strategy_history = deque(maxlen=50)
        
        self.init_database()
    
    def init_database(self):
        """데이터베이스 초기화"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 시장 조건 기록 테이블
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS market_conditions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    condition_name TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    features TEXT NOT NULL,
                    duration_minutes INTEGER
                )
            ''')
            
            # 전략 실행 기록 테이블
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS strategy_executions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    strategy_name TEXT NOT NULL,
                    market_condition TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    position_size REAL NOT NULL,
                    entry_price REAL,
                    exit_price REAL,
                    pnl_percent REAL,
                    duration_minutes INTEGER,
                    win INTEGER,
                    stop_loss REAL,
                    take_profit REAL
                )
            ''')
            
            # 위험 관리 기록 테이블
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS risk_management (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    action TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    daily_pnl REAL,
                    position_count INTEGER,
                    risk_level TEXT
                )
            ''')
            
            conn.commit()
            conn.close()
            
            logger.info("✅ 전략 엔진 데이터베이스 초기화 완료")
            
        except Exception as e:
            logger.error(f"데이터베이스 초기화 실패: {e}")
    
    async def analyze_and_adapt(self, market_data: Dict) -> Dict[str, Any]:
        """시장 분석 및 전략 적응"""
        try:
            result = {
                'timestamp': datetime.now().isoformat(),
                'market_analysis': {},
                'strategy_decision': {},
                'risk_assessment': {},
                'execution_plan': {}
            }
            
            # 1. 시장 조건 분류
            condition, confidence = self.condition_classifier.classify_condition(market_data)
            
            result['market_analysis'] = {
                'condition': condition,
                'confidence': confidence,
                'condition_description': self.condition_classifier.conditions[condition].description
            }
            
            # 조건 변경 감지
            condition_changed = (self.current_market_condition != condition)
            if condition_changed:
                logger.info(f"시장 조건 변경: {self.current_market_condition} → {condition} (신뢰도: {confidence:.2f})")
                self.current_market_condition = condition
                
                # 데이터베이스 기록
                await self.record_market_condition(condition, confidence, market_data)
            
            self.condition_history.append((condition, confidence, datetime.now()))
            
            # 2. 전략 선택
            selected_strategy = self.strategy_manager.select_strategy(condition, confidence)
            
            if selected_strategy:
                result['strategy_decision'] = {
                    'strategy_name': selected_strategy.name,
                    'strategy_description': selected_strategy.description,
                    'risk_level': selected_strategy.risk_level,
                    'parameters': selected_strategy.parameters,
                    'success_rate': selected_strategy.success_rate,
                    'strategy_changed': (self.current_strategy != selected_strategy)
                }
                
                self.current_strategy = selected_strategy
                self.strategy_history.append((selected_strategy.name, datetime.now()))
            
            # 3. 위험 평가
            if selected_strategy:
                risk_assessment = self.risk_manager.assess_risk(selected_strategy, condition, confidence)
                result['risk_assessment'] = risk_assessment
                
                # 4. 실행 계획 생성
                if risk_assessment['approved']:
                    execution_plan = await self.generate_execution_plan(
                        selected_strategy, condition, confidence, market_data
                    )
                    result['execution_plan'] = execution_plan
                else:
                    result['execution_plan'] = {
                        'action': 'hold',
                        'reason': 'risk_management_block',
                        'warnings': risk_assessment['risk_warnings']
                    }
            
            # 5. 성과 추적 업데이트
            await self.update_performance_tracking()
            
            return result
            
        except Exception as e:
            logger.error(f"시장 분석 및 적응 실패: {e}")
            return {'error': str(e)}
    
    async def generate_execution_plan(self, strategy: TradingStrategy, condition: str, 
                                    confidence: float, market_data: Dict) -> Dict[str, Any]:
        """실행 계획 생성"""
        try:
            current_price = market_data.get('price', 0)
            
            # 신호 강도 계산
            signal_strength = confidence * strategy.parameters.get('entry_threshold', 0.7)
            
            # 진입 조건 확인
            should_enter = signal_strength >= strategy.parameters.get('entry_threshold', 0.7)
            
            plan = {
                'action': 'hold',
                'signal_strength': signal_strength,
                'entry_threshold': strategy.parameters.get('entry_threshold', 0.7),
                'current_price': current_price,
                'timestamp': datetime.now().isoformat()
            }
            
            if should_enter:
                # 리스크 관리된 포지션 크기
                position_size = self.risk_manager.assess_risk(strategy, condition, confidence)['suggested_position_size']
                
                # 스탑로스/테이크프로핏 계산
                stop_loss_price = current_price * (1 - strategy.parameters['stop_loss'])
                take_profit_price = current_price * (1 + strategy.parameters['take_profit'])
                
                plan.update({
                    'action': 'buy' if 'bull' in condition or 'breakout' in condition else 'sell',
                    'position_size': position_size,
                    'entry_price': current_price,
                    'stop_loss': stop_loss_price,
                    'take_profit': take_profit_price,
                    'max_holding_time': strategy.parameters.get('max_holding_hours', 24) * 60,  # 분 단위
                    'rationale': f"{strategy.description} - {condition} 상황에서 신호 강도 {signal_strength:.2f}"
                })
                
                # 실행 기록
                await self.record_strategy_execution(plan, strategy.name, condition, confidence)
            
            return plan
            
        except Exception as e:
            logger.error(f"실행 계획 생성 실패: {e}")
            return {'action': 'hold', 'error': str(e)}
    
    async def record_market_condition(self, condition: str, confidence: float, market_data: Dict):
        """시장 조건 기록"""
        try:
            features = self.condition_classifier.extract_features(market_data)
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO market_conditions 
                (timestamp, condition_name, confidence, features)
                VALUES (?, ?, ?, ?)
            ''', (
                datetime.now().isoformat(),
                condition,
                confidence,
                json.dumps(features.tolist())
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"시장 조건 기록 실패: {e}")
    
    async def record_strategy_execution(self, plan: Dict, strategy_name: str, condition: str, confidence: float):
        """전략 실행 기록"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO strategy_executions 
                (timestamp, strategy_name, market_condition, confidence, position_size, entry_price, stop_loss, take_profit)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                plan['timestamp'],
                strategy_name,
                condition,
                confidence,
                plan.get('position_size', 0),
                plan.get('entry_price', 0),
                plan.get('stop_loss', 0),
                plan.get('take_profit', 0)
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"전략 실행 기록 실패: {e}")
    
    async def update_performance_tracking(self):
        """성과 추적 업데이트"""
        try:
            # 최근 실행된 전략들의 성과 확인 및 업데이트
            # 여기서는 간단한 로깅만 수행
            
            if len(self.strategy_history) > 0:
                recent_strategies = list(self.strategy_history)[-5:]
                strategy_counts = {}
                for strategy_name, timestamp in recent_strategies:
                    strategy_counts[strategy_name] = strategy_counts.get(strategy_name, 0) + 1
                
                logger.debug(f"최근 전략 사용: {strategy_counts}")
            
        except Exception as e:
            logger.error(f"성과 추적 업데이트 실패: {e}")
    
    async def get_system_status(self) -> Dict[str, Any]:
        """시스템 상태 조회"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # 최근 시장 조건들
            conditions_df = pd.read_sql_query('''
                SELECT * FROM market_conditions 
                ORDER BY timestamp DESC LIMIT 10
            ''', conn)
            
            # 최근 전략 실행들
            executions_df = pd.read_sql_query('''
                SELECT * FROM strategy_executions 
                ORDER BY timestamp DESC LIMIT 10
            ''', conn)
            
            conn.close()
            
            # 전략 통계
            strategy_stats = self.strategy_manager.get_strategy_stats()
            
            return {
                'current_market_condition': self.current_market_condition,
                'current_strategy': self.current_strategy.name if self.current_strategy else None,
                'recent_conditions': conditions_df.to_dict('records'),
                'recent_executions': executions_df.to_dict('records'),
                'strategy_statistics': strategy_stats,
                'risk_metrics': {
                    'daily_pnl': self.risk_manager.daily_pnl,
                    'current_positions': len(self.risk_manager.current_positions),
                    'max_positions': self.risk_manager.max_concurrent_positions
                },
                'condition_history_length': len(self.condition_history),
                'strategy_history_length': len(self.strategy_history)
            }
            
        except Exception as e:
            logger.error(f"시스템 상태 조회 실패: {e}")
            return {'error': str(e)}

async def run_strategy_engine_demo():
    """전략 엔진 데모 실행"""
    print("🎯 시장 적응형 전략 엔진 시작")
    print("="*60)
    
    engine = MarketAdaptiveStrategyEngine()
    
    # 시뮬레이션 시나리오들
    scenarios = [
        # 강한 상승 시나리오
        {
            'name': '강세 돌파',
            'price': 52000,
            'rsi': 75,
            'macd': 50,
            'sma_20': 50000,
            'volume': 2000000,
            'volume_sma': 1000000,
            'volatility': 0.025,
            'fear_greed_index': 80
        },
        # 횡보 시나리오
        {
            'name': '횡보 안정',
            'price': 50500,
            'rsi': 50,
            'macd': 5,
            'sma_20': 50000,
            'volume': 800000,
            'volume_sma': 1000000,
            'volatility': 0.015,
            'fear_greed_index': 50
        },
        # 약세 시나리오
        {
            'name': '약세 하락',
            'price': 47000,
            'rsi': 30,
            'macd': -30,
            'sma_20': 50000,
            'volume': 1500000,
            'volume_sma': 1000000,
            'volatility': 0.045,
            'fear_greed_index': 25
        },
        # 고변동성 시나리오
        {
            'name': '변동성 급증',
            'price': 49000,
            'rsi': 60,
            'macd': 10,
            'sma_20': 50000,
            'volume': 3000000,
            'volume_sma': 1000000,
            'volatility': 0.08,
            'fear_greed_index': 45
        }
    ]
    
    print("📊 다양한 시장 시나리오 테스트:")
    print("-" * 40)
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n🎬 시나리오 {i}: {scenario['name']}")
        
        # 시장 분석 및 전략 적응
        result = await engine.analyze_and_adapt(scenario)
        
        if 'error' in result:
            print(f"❌ 오류: {result['error']}")
            continue
        
        # 결과 출력
        market_analysis = result.get('market_analysis', {})
        strategy_decision = result.get('strategy_decision', {})
        risk_assessment = result.get('risk_assessment', {})
        execution_plan = result.get('execution_plan', {})
        
        print(f"  📈 시장 조건: {market_analysis.get('condition', 'N/A')} (신뢰도: {market_analysis.get('confidence', 0):.2f})")
        
        if strategy_decision:
            print(f"  🎯 선택 전략: {strategy_decision.get('strategy_name', 'N/A')}")
            print(f"  ⚡ 위험 수준: {strategy_decision.get('risk_level', 'N/A')}")
            print(f"  📊 성공률: {strategy_decision.get('success_rate', 0):.1%}")
        
        if execution_plan.get('action') != 'hold':
            print(f"  🚀 실행 계획: {execution_plan.get('action', 'N/A')}")
            print(f"  💰 포지션 크기: {execution_plan.get('position_size', 0):.1%}")
            print(f"  🛑 스탑로스: ${execution_plan.get('stop_loss', 0):,.0f}")
            print(f"  🎯 목표가: ${execution_plan.get('take_profit', 0):,.0f}")
        else:
            print(f"  ⏸️ 액션: {execution_plan.get('action', 'hold').upper()}")
            if execution_plan.get('reason'):
                print(f"  📝 사유: {execution_plan.get('reason', 'N/A')}")
        
        # 위험 경고가 있으면 출력
        if risk_assessment.get('risk_warnings'):
            print(f"  ⚠️ 위험 경고: {', '.join(risk_assessment['risk_warnings'])}")
        
        await asyncio.sleep(1)  # 시각적 효과
    
    # 최종 시스템 상태
    print("\n" + "="*60)
    print("📊 최종 시스템 상태")
    
    status = await engine.get_system_status()
    
    print(f"🌐 현재 시장 조건: {status.get('current_market_condition', 'N/A')}")
    print(f"🎯 현재 전략: {status.get('current_strategy', 'N/A')}")
    print(f"📈 조건 히스토리: {status.get('condition_history_length', 0)}개")
    print(f"🔄 전략 히스토리: {status.get('strategy_history_length', 0)}개")
    
    # 전략 통계
    strategy_stats = status.get('strategy_statistics', {})
    if strategy_stats:
        print(f"\n📊 전략별 사용 통계:")
        for name, stats in strategy_stats.items():
            print(f"  • {name}: 사용 {stats['usage_count']}회, 성공률 {stats['recent_win_rate']:.1%}")
    
    print("\n" + "="*60)
    print("🎉 시장 적응형 전략 엔진 데모 완료!")

if __name__ == "__main__":
    asyncio.run(run_strategy_engine_demo())