#!/usr/bin/env python3
"""
📈 시계열 분석 엔진
- 1시간 단위 BTC 데이터 시계열 분석
- 트렌드, 계절성, 변동성 분석
- 예측 모델을 위한 시계열 특성 추출
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from scipy import stats
import warnings

from ..utils.logger import get_logger

warnings.filterwarnings('ignore')

class TimeseriesAnalyzer:
    """시계열 분석 엔진"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        print("📈 시계열 분석 엔진 초기화")
        
    def analyze(self, data: pd.DataFrame) -> Dict:
        """
        종합 시계열 분석
        
        Args:
            data: 1시간 단위 시계열 데이터
            
        Returns:
            분석 결과 딕셔너리
        """
        print("📊 시계열 종합 분석 시작")
        
        btc_col = self._find_btc_price_column(data)
        if not btc_col:
            self.logger.error("BTC 가격 컬럼을 찾을 수 없음")
            return {}
            
        btc_prices = data[btc_col].dropna()
        
        analysis_result = {
            'basic_stats': self._calculate_basic_stats(btc_prices),
            'trend_analysis': self._analyze_trend(btc_prices),
            'volatility_analysis': self._analyze_volatility(btc_prices),
            'seasonality_analysis': self._analyze_seasonality(btc_prices),
            'momentum_analysis': self._analyze_momentum(btc_prices),
            'technical_indicators': self._calculate_technical_indicators(btc_prices),
            'market_regime': self._detect_market_regime(btc_prices),
            'forecast_features': self._extract_forecast_features(btc_prices)
        }
        
        print("✅ 시계열 분석 완료")
        return analysis_result
    
    def _find_btc_price_column(self, data: pd.DataFrame) -> str:
        """BTC 가격 컬럼 찾기"""
        candidates = [
            'onchain_blockchain_info_network_stats_market_price_usd',
            'btc_price', 'price', 'close', 'market_price_usd'
        ]
        
        for candidate in candidates:
            if candidate in data.columns:
                return candidate
        
        # 큰 수치를 가진 컬럼 찾기 (BTC 가격 추정)
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if data[col].mean() > 1000:
                return col
                
        return numeric_cols[0] if len(numeric_cols) > 0 else None
    
    def _calculate_basic_stats(self, prices: pd.Series) -> Dict:
        """기본 통계 계산"""
        return {
            'count': len(prices),
            'mean': float(prices.mean()),
            'std': float(prices.std()),
            'min': float(prices.min()),
            'max': float(prices.max()),
            'current_price': float(prices.iloc[-1]),
            'price_range_pct': float((prices.max() - prices.min()) / prices.mean() * 100)
        }
    
    def _analyze_trend(self, prices: pd.Series) -> Dict:
        """트렌드 분석"""
        # 선형 회귀를 이용한 트렌드 계산
        x = np.arange(len(prices))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, prices)
        
        # 단기/중기/장기 트렌드
        short_trend = self._calculate_trend(prices.tail(24))    # 1일
        medium_trend = self._calculate_trend(prices.tail(168))  # 1주일
        long_trend = self._calculate_trend(prices.tail(720))   # 1개월
        
        return {
            'overall_slope': float(slope),
            'trend_strength': float(abs(r_value)),
            'trend_direction': 'up' if slope > 0 else 'down',
            'short_term_trend': short_trend,
            'medium_term_trend': medium_trend,
            'long_term_trend': long_trend,
            'trend_consistency': self._measure_trend_consistency(prices)
        }
    
    def _calculate_trend(self, prices: pd.Series) -> Dict:
        """특정 기간의 트렌드 계산"""
        if len(prices) < 2:
            return {'direction': 'neutral', 'strength': 0}
            
        start_price = prices.iloc[0]
        end_price = prices.iloc[-1]
        change_pct = (end_price - start_price) / start_price * 100
        
        direction = 'up' if change_pct > 1 else 'down' if change_pct < -1 else 'neutral'
        strength = min(abs(change_pct), 100)  # 최대 100%로 제한
        
        return {
            'direction': direction,
            'strength': float(strength),
            'change_pct': float(change_pct)
        }
    
    def _measure_trend_consistency(self, prices: pd.Series) -> float:
        """트렌드 일관성 측정"""
        if len(prices) < 24:
            return 0.0
            
        # 24시간 단위로 나누어 각 구간의 트렌드 방향 확인
        segments = []
        for i in range(0, len(prices) - 24, 24):
            segment = prices.iloc[i:i+24]
            if len(segment) >= 24:
                trend = (segment.iloc[-1] - segment.iloc[0]) / segment.iloc[0]
                segments.append(1 if trend > 0 else -1)
        
        if not segments:
            return 0.0
        
        # 같은 방향 트렌드의 비율
        positive_segments = sum(1 for s in segments if s > 0)
        negative_segments = sum(1 for s in segments if s < 0)
        
        consistency = abs(positive_segments - negative_segments) / len(segments)
        return float(consistency)
    
    def _analyze_volatility(self, prices: pd.Series) -> Dict:
        """변동성 분석"""
        returns = prices.pct_change().dropna()
        
        # 다양한 시간대 변동성
        hourly_vol = returns.std() * 100
        daily_vol = returns.rolling(24).std().mean() * 100 if len(returns) >= 24 else 0
        weekly_vol = returns.rolling(168).std().mean() * 100 if len(returns) >= 168 else 0
        
        # 변동성 클러스터링 (GARCH 효과)
        volatility_clustering = self._measure_volatility_clustering(returns)
        
        # 극단값 빈도
        extreme_moves = returns[abs(returns) > returns.std() * 2]
        extreme_frequency = len(extreme_moves) / len(returns) * 100
        
        return {
            'hourly_volatility': float(hourly_vol),
            'daily_volatility': float(daily_vol),
            'weekly_volatility': float(weekly_vol),
            'volatility_clustering': float(volatility_clustering),
            'extreme_move_frequency': float(extreme_frequency),
            'volatility_regime': self._classify_volatility_regime(hourly_vol)
        }
    
    def _measure_volatility_clustering(self, returns: pd.Series) -> float:
        """변동성 클러스터링 측정"""
        if len(returns) < 48:
            return 0.0
            
        # 절댓값 수익률의 자기상관
        abs_returns = abs(returns)
        correlation = abs_returns.autocorr(lag=1)
        return correlation if not np.isnan(correlation) else 0.0
    
    def _classify_volatility_regime(self, volatility: float) -> str:
        """변동성 체제 분류"""
        if volatility > 5:
            return 'high'
        elif volatility > 2:
            return 'medium'
        else:
            return 'low'
    
    def _analyze_seasonality(self, prices: pd.Series) -> Dict:
        """계절성 분석 (시간대별 패턴)"""
        if len(prices) < 168:  # 최소 1주일 데이터 필요
            return {'pattern': 'insufficient_data'}
        
        # 시간대별 평균 수익률 (0-23시)
        hourly_returns = {}
        price_changes = prices.pct_change().dropna()
        
        for hour in range(24):
            hour_indices = [i for i in range(len(price_changes)) if i % 24 == hour]
            if hour_indices:
                hourly_data = price_changes.iloc[hour_indices]
                hourly_returns[hour] = {
                    'mean_return': float(hourly_data.mean() * 100),
                    'volatility': float(hourly_data.std() * 100),
                    'count': len(hourly_data)
                }
        
        # 요일별 패턴 (7일 주기)
        daily_returns = {}
        for day in range(7):
            day_indices = [i for i in range(len(price_changes)) if (i // 24) % 7 == day]
            if day_indices:
                daily_data = price_changes.iloc[day_indices]
                daily_returns[day] = {
                    'mean_return': float(daily_data.mean() * 100),
                    'volatility': float(daily_data.std() * 100),
                    'count': len(daily_data)
                }
        
        # 최고/최저 활성 시간대
        best_hour = max(hourly_returns.items(), key=lambda x: x[1]['mean_return'])[0] if hourly_returns else 0
        worst_hour = min(hourly_returns.items(), key=lambda x: x[1]['mean_return'])[0] if hourly_returns else 0
        
        return {
            'hourly_patterns': hourly_returns,
            'daily_patterns': daily_returns,
            'best_trading_hour': int(best_hour),
            'worst_trading_hour': int(worst_hour),
            'seasonality_strength': self._calculate_seasonality_strength(hourly_returns)
        }
    
    def _calculate_seasonality_strength(self, hourly_data: Dict) -> float:
        """계절성 강도 계산"""
        if not hourly_data:
            return 0.0
            
        returns = [data['mean_return'] for data in hourly_data.values()]
        return float(np.std(returns)) if returns else 0.0
    
    def _analyze_momentum(self, prices: pd.Series) -> Dict:
        """모멘텀 분석"""
        # 다양한 기간의 모멘텀
        momentum_1h = prices.iloc[-1] - prices.iloc[-2] if len(prices) >= 2 else 0
        momentum_6h = prices.iloc[-1] - prices.iloc[-7] if len(prices) >= 7 else 0
        momentum_24h = prices.iloc[-1] - prices.iloc[-25] if len(prices) >= 25 else 0
        momentum_168h = prices.iloc[-1] - prices.iloc[-169] if len(prices) >= 169 else 0  # 1주일
        
        # 모멘텀 퍼센트 변화
        current_price = prices.iloc[-1]
        momentum_1h_pct = (momentum_1h / current_price * 100) if current_price > 0 else 0
        momentum_6h_pct = (momentum_6h / current_price * 100) if current_price > 0 else 0
        momentum_24h_pct = (momentum_24h / current_price * 100) if current_price > 0 else 0
        momentum_168h_pct = (momentum_168h / current_price * 100) if current_price > 0 else 0
        
        # 모멘텀 방향 일치성
        momentum_values = [momentum_1h_pct, momentum_6h_pct, momentum_24h_pct, momentum_168h_pct]
        positive_momentum = sum(1 for m in momentum_values if m > 0)
        momentum_alignment = positive_momentum / len(momentum_values)
        
        return {
            'momentum_1h': float(momentum_1h_pct),
            'momentum_6h': float(momentum_6h_pct),
            'momentum_24h': float(momentum_24h_pct),
            'momentum_168h': float(momentum_168h_pct),
            'momentum_alignment': float(momentum_alignment),
            'momentum_strength': float(np.mean([abs(m) for m in momentum_values])),
            'momentum_direction': 'bullish' if momentum_alignment > 0.6 else 'bearish' if momentum_alignment < 0.4 else 'mixed'
        }
    
    def _calculate_technical_indicators(self, prices: pd.Series) -> Dict:
        """기술적 지표 계산"""
        indicators = {}
        
        # RSI
        indicators['rsi'] = self._calculate_rsi(prices)
        
        # 이동평균
        indicators['sma_24'] = float(prices.rolling(24).mean().iloc[-1]) if len(prices) >= 24 else float(prices.mean())
        indicators['sma_168'] = float(prices.rolling(168).mean().iloc[-1]) if len(prices) >= 168 else float(prices.mean())
        
        # 볼린저 밴드
        if len(prices) >= 20:
            bb_data = self._calculate_bollinger_bands(prices, 20)
            indicators.update(bb_data)
        
        # MACD
        if len(prices) >= 26:
            macd_data = self._calculate_macd(prices)
            indicators.update(macd_data)
        
        # 현재 가격 대비 위치
        current_price = float(prices.iloc[-1])
        indicators['price_vs_sma24'] = ((current_price - indicators['sma_24']) / indicators['sma_24'] * 100) if indicators['sma_24'] > 0 else 0
        indicators['price_vs_sma168'] = ((current_price - indicators['sma_168']) / indicators['sma_168'] * 100) if indicators['sma_168'] > 0 else 0
        
        return indicators
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """RSI 계산"""
        if len(prices) < period + 1:
            return 50.0
            
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return float(rsi.iloc[-1]) if not np.isnan(rsi.iloc[-1]) else 50.0
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20) -> Dict:
        """볼린저 밴드 계산"""
        sma = prices.rolling(period).mean()
        std = prices.rolling(period).std()
        
        upper_band = sma + (std * 2)
        lower_band = sma - (std * 2)
        
        current_price = prices.iloc[-1]
        current_upper = upper_band.iloc[-1]
        current_lower = lower_band.iloc[-1]
        current_middle = sma.iloc[-1]
        
        # %B (밴드 내 위치)
        bb_percent = ((current_price - current_lower) / (current_upper - current_lower)) if (current_upper - current_lower) > 0 else 0.5
        
        return {
            'bb_upper': float(current_upper),
            'bb_middle': float(current_middle),
            'bb_lower': float(current_lower),
            'bb_percent': float(bb_percent),
            'bb_width': float((current_upper - current_lower) / current_middle * 100) if current_middle > 0 else 0
        }
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict:
        """MACD 계산"""
        exp1 = prices.ewm(span=fast).mean()
        exp2 = prices.ewm(span=slow).mean()
        
        macd_line = exp1 - exp2
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        
        return {
            'macd': float(macd_line.iloc[-1]),
            'macd_signal': float(signal_line.iloc[-1]),
            'macd_histogram': float(histogram.iloc[-1])
        }
    
    def _detect_market_regime(self, prices: pd.Series) -> Dict:
        """시장 체제 감지"""
        if len(prices) < 24:
            return {'regime': 'insufficient_data', 'confidence': 0}
        
        # 최근 24시간, 168시간(1주일) 트렌드
        recent_24h = (prices.iloc[-1] - prices.iloc[-24]) / prices.iloc[-24] * 100
        recent_168h = (prices.iloc[-1] - prices.iloc[-168]) / prices.iloc[-168] * 100 if len(prices) >= 168 else recent_24h
        
        # 변동성
        volatility = prices.pct_change().tail(24).std() * 100
        
        # 체제 분류
        if recent_24h > 3 and recent_168h > 5:
            regime = 'strong_bull'
        elif recent_24h > 1 and recent_168h > 0:
            regime = 'bull'
        elif recent_24h < -3 and recent_168h < -5:
            regime = 'strong_bear'
        elif recent_24h < -1 and recent_168h < 0:
            regime = 'bear'
        elif volatility > 4:
            regime = 'high_volatility'
        else:
            regime = 'sideways'
        
        # 신뢰도 계산
        confidence = min(95, abs(recent_24h) * 10 + abs(recent_168h) * 5)
        
        return {
            'regime': regime,
            'confidence': float(confidence),
            'trend_24h': float(recent_24h),
            'trend_168h': float(recent_168h),
            'volatility': float(volatility)
        }
    
    def _extract_forecast_features(self, prices: pd.Series) -> Dict:
        """예측을 위한 핵심 특성 추출"""
        features = {}
        
        # 가격 레벨 특성
        current_price = prices.iloc[-1]
        features['price_level'] = float(current_price)
        features['price_zscore'] = float((current_price - prices.mean()) / prices.std()) if prices.std() > 0 else 0
        
        # 지연 특성
        for lag in [1, 6, 12, 24, 168]:
            if len(prices) > lag:
                features[f'price_lag_{lag}'] = float(prices.iloc[-lag-1])
                features[f'return_lag_{lag}'] = float((prices.iloc[-1] - prices.iloc[-lag-1]) / prices.iloc[-lag-1] * 100)
        
        # 롤링 통계
        for window in [6, 12, 24, 168]:
            if len(prices) >= window:
                rolling_data = prices.rolling(window)
                features[f'sma_{window}'] = float(rolling_data.mean().iloc[-1])
                features[f'std_{window}'] = float(rolling_data.std().iloc[-1])
                features[f'min_{window}'] = float(rolling_data.min().iloc[-1])
                features[f'max_{window}'] = float(rolling_data.max().iloc[-1])
        
        # 수익률 특성
        returns = prices.pct_change()
        for window in [6, 24]:
            if len(returns) >= window:
                recent_returns = returns.tail(window)
                features[f'return_mean_{window}'] = float(recent_returns.mean() * 100)
                features[f'return_std_{window}'] = float(recent_returns.std() * 100)
                features[f'return_skew_{window}'] = float(recent_returns.skew()) if not np.isnan(recent_returns.skew()) else 0
        
        return features