#!/usr/bin/env python3
"""
📈 시계열 분석 엔진
- 1시간 단위 BTC 데이터 전문 시계열 분석
- 백테스트 학습을 위한 핵심 시계열 특성 추출
- 사용자 요구사항: "시계열 분석이 꼭 필요"
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
from scipy import stats
from scipy.signal import find_peaks
import warnings

warnings.filterwarnings('ignore')

class TimeSeriesEngine:
    """시계열 분석 전문 엔진"""
    
    def __init__(self):
        """시계열 분석 엔진 초기화"""
        print("📈 시계열 분석 엔진 초기화")
        self.analysis_cache = {}
        
    def comprehensive_timeseries_analysis(self, data: pd.DataFrame, 
                                        price_column: str) -> Dict:
        """
        종합적인 시계열 분석
        
        Args:
            data: 시계열 데이터 (1시간 단위)
            price_column: BTC 가격 컬럼명
            
        Returns:
            시계열 분석 결과 딕셔너리
        """
        print(f"📊 종합 시계열 분석 시작: {len(data)}시간 데이터")
        
        if price_column not in data.columns:
            raise ValueError(f"가격 컬럼 '{price_column}' 없음")
        
        prices = data[price_column].dropna()
        
        analysis_result = {
            'basic_statistics': self._basic_statistics(prices),
            'trend_analysis': self._trend_analysis(prices),
            'seasonality_analysis': self._seasonality_analysis(prices, data),
            'volatility_analysis': self._volatility_analysis(prices),
            'cycle_analysis': self._cycle_analysis(prices),
            'momentum_analysis': self._momentum_analysis(prices),
            'pattern_recognition': self._pattern_recognition(prices),
            'support_resistance': self._support_resistance_levels(prices),
            'forecast_indicators': self._forecast_indicators(prices),
            'market_regime_detection': self._market_regime_detection(prices),
            'anomaly_detection': self._anomaly_detection(prices)
        }
        
        print("✅ 종합 시계열 분석 완료")
        return analysis_result
    
    def _basic_statistics(self, prices: pd.Series) -> Dict:
        """기본 통계 분석"""
        returns = prices.pct_change().dropna()
        
        return {
            'count': len(prices),
            'mean_price': float(prices.mean()),
            'std_price': float(prices.std()),
            'min_price': float(prices.min()),
            'max_price': float(prices.max()),
            'current_price': float(prices.iloc[-1]),
            'price_range_pct': float((prices.max() - prices.min()) / prices.mean() * 100),
            'mean_return': float(returns.mean() * 100),
            'std_return': float(returns.std() * 100),
            'skewness': float(returns.skew()),
            'kurtosis': float(returns.kurtosis()),
            'sharpe_ratio': float(returns.mean() / returns.std()) if returns.std() > 0 else 0
        }
    
    def _trend_analysis(self, prices: pd.Series) -> Dict:
        """트렌드 분석 (선형회귀, 다항회귀)"""
        x = np.arange(len(prices))
        
        # 선형 트렌드
        linear_slope, linear_intercept, linear_r, _, _ = stats.linregress(x, prices)
        
        # 다항 트렌드 (2차)
        poly_coeffs = np.polyfit(x, prices, 2)
        poly_trend = np.polyval(poly_coeffs, x)
        
        # 단기/중기/장기 트렌드
        short_trend = self._calculate_period_trend(prices.tail(24))   # 1일
        medium_trend = self._calculate_period_trend(prices.tail(168)) # 1주일  
        long_trend = self._calculate_period_trend(prices.tail(720))   # 1개월
        
        # 트렌드 강도 및 일관성
        trend_strength = abs(linear_r)
        trend_consistency = self._calculate_trend_consistency(prices)
        
        return {
            'linear_slope': float(linear_slope),
            'linear_r_squared': float(linear_r ** 2),
            'trend_direction': 'bullish' if linear_slope > 0 else 'bearish',
            'trend_strength': float(trend_strength),
            'trend_consistency': float(trend_consistency),
            'short_term_trend': short_trend,
            'medium_term_trend': medium_trend,
            'long_term_trend': long_trend,
            'polynomial_coefficients': [float(c) for c in poly_coeffs],
            'detrended_volatility': float(np.std(prices - poly_trend))
        }
    
    def _calculate_period_trend(self, prices: pd.Series) -> Dict:
        """특정 기간 트렌드 계산"""
        if len(prices) < 2:
            return {'direction': 'neutral', 'strength': 0, 'change_pct': 0}
        
        start_price = prices.iloc[0]
        end_price = prices.iloc[-1]
        change_pct = (end_price - start_price) / start_price * 100
        
        # 방향 결정
        if change_pct > 1:
            direction = 'bullish'
        elif change_pct < -1:
            direction = 'bearish'
        else:
            direction = 'neutral'
        
        # 강도 계산 (변화율과 일관성)
        strength = min(abs(change_pct), 50)  # 최대 50%로 제한
        
        return {
            'direction': direction,
            'strength': float(strength),
            'change_pct': float(change_pct)
        }
    
    def _calculate_trend_consistency(self, prices: pd.Series) -> float:
        """트렌드 일관성 측정"""
        if len(prices) < 24:
            return 0.0
        
        # 12시간 단위 세그먼트별 트렌드 방향
        segment_trends = []
        for i in range(0, len(prices) - 12, 12):
            segment = prices.iloc[i:i+12]
            if len(segment) >= 12:
                trend = (segment.iloc[-1] - segment.iloc[0]) / segment.iloc[0]
                segment_trends.append(1 if trend > 0 else -1)
        
        if not segment_trends:
            return 0.0
        
        # 같은 방향 트렌드 비율
        positive_count = sum(1 for t in segment_trends if t > 0)
        negative_count = sum(1 for t in segment_trends if t < 0)
        
        consistency = abs(positive_count - negative_count) / len(segment_trends)
        return consistency
    
    def _seasonality_analysis(self, prices: pd.Series, full_data: pd.DataFrame) -> Dict:
        """계절성/주기성 분석"""
        
        # 시간 정보 추출
        seasonality_patterns = {}
        
        if 'timestamp' in full_data.columns:
            # 시간대별 패턴 (0-23시)
            hourly_patterns = self._analyze_hourly_patterns(prices, full_data['timestamp'])
            seasonality_patterns['hourly'] = hourly_patterns
            
            # 요일별 패턴
            daily_patterns = self._analyze_daily_patterns(prices, full_data['timestamp'])
            seasonality_patterns['daily'] = daily_patterns
            
            # 월별 패턴 (데이터가 충분할 경우)
            if len(full_data) > 720:  # 1개월 이상
                monthly_patterns = self._analyze_monthly_patterns(prices, full_data['timestamp'])
                seasonality_patterns['monthly'] = monthly_patterns
        
        # 주기 감지 (FFT 기반)
        dominant_cycles = self._detect_dominant_cycles(prices)
        
        return {
            'seasonal_patterns': seasonality_patterns,
            'dominant_cycles': dominant_cycles,
            'seasonality_strength': self._calculate_seasonality_strength(seasonality_patterns),
            'best_trading_hours': self._find_best_trading_hours(seasonality_patterns),
            'worst_trading_hours': self._find_worst_trading_hours(seasonality_patterns)
        }
    
    def _analyze_hourly_patterns(self, prices: pd.Series, timestamps: pd.Series) -> Dict:
        """시간대별 패턴 분석"""
        hourly_data = pd.DataFrame({
            'price': prices,
            'hour': timestamps.dt.hour,
            'return': prices.pct_change()
        }).dropna()
        
        hourly_stats = {}
        for hour in range(24):
            hour_data = hourly_data[hourly_data['hour'] == hour]
            if len(hour_data) > 0:
                hourly_stats[hour] = {
                    'mean_return': float(hour_data['return'].mean() * 100),
                    'std_return': float(hour_data['return'].std() * 100),
                    'mean_price': float(hour_data['price'].mean()),
                    'count': len(hour_data)
                }
        
        return hourly_stats
    
    def _analyze_daily_patterns(self, prices: pd.Series, timestamps: pd.Series) -> Dict:
        """요일별 패턴 분석"""
        daily_data = pd.DataFrame({
            'price': prices,
            'day_of_week': timestamps.dt.dayofweek,
            'return': prices.pct_change()
        }).dropna()
        
        daily_stats = {}
        day_names = ['월', '화', '수', '목', '금', '토', '일']
        
        for day in range(7):
            day_data = daily_data[daily_data['day_of_week'] == day]
            if len(day_data) > 0:
                daily_stats[day_names[day]] = {
                    'mean_return': float(day_data['return'].mean() * 100),
                    'std_return': float(day_data['return'].std() * 100),
                    'volatility': float(day_data['return'].std() * 100),
                    'count': len(day_data)
                }
        
        return daily_stats
    
    def _analyze_monthly_patterns(self, prices: pd.Series, timestamps: pd.Series) -> Dict:
        """월별 패턴 분석"""
        monthly_data = pd.DataFrame({
            'price': prices,
            'month': timestamps.dt.month,
            'return': prices.pct_change()
        }).dropna()
        
        monthly_stats = {}
        month_names = ['1월', '2월', '3월', '4월', '5월', '6월',
                      '7월', '8월', '9월', '10월', '11월', '12월']
        
        for month in range(1, 13):
            month_data = monthly_data[monthly_data['month'] == month]
            if len(month_data) > 0:
                monthly_stats[month_names[month-1]] = {
                    'mean_return': float(month_data['return'].mean() * 100),
                    'volatility': float(month_data['return'].std() * 100),
                    'count': len(month_data)
                }
        
        return monthly_stats
    
    def _detect_dominant_cycles(self, prices: pd.Series) -> List[Dict]:
        """FFT를 이용한 주기 감지"""
        if len(prices) < 48:
            return []
        
        # 트렌드 제거
        detrended = prices - prices.rolling(24, center=True).mean()
        detrended = detrended.dropna()
        
        # FFT 실행
        fft = np.fft.fft(detrended)
        frequencies = np.fft.fftfreq(len(detrended))
        
        # 주파수별 파워 계산
        power_spectrum = np.abs(fft) ** 2
        
        # 상위 5개 주기 추출
        top_indices = np.argsort(power_spectrum)[-6:-1]  # DC 성분 제외
        
        dominant_cycles = []
        for idx in top_indices:
            if frequencies[idx] != 0:
                period_hours = 1 / abs(frequencies[idx])
                if 2 <= period_hours <= len(prices) / 2:  # 유의미한 주기만
                    dominant_cycles.append({
                        'period_hours': float(period_hours),
                        'period_days': float(period_hours / 24),
                        'strength': float(power_spectrum[idx]),
                        'frequency': float(frequencies[idx])
                    })
        
        return sorted(dominant_cycles, key=lambda x: x['strength'], reverse=True)
    
    def _calculate_seasonality_strength(self, patterns: Dict) -> float:
        """계절성 강도 계산"""
        if not patterns:
            return 0.0
        
        strength_scores = []
        
        # 시간대별 패턴 강도
        if 'hourly' in patterns:
            hourly_returns = [data['mean_return'] for data in patterns['hourly'].values()]
            if hourly_returns:
                strength_scores.append(np.std(hourly_returns))
        
        # 요일별 패턴 강도
        if 'daily' in patterns:
            daily_returns = [data['mean_return'] for data in patterns['daily'].values()]
            if daily_returns:
                strength_scores.append(np.std(daily_returns))
        
        return float(np.mean(strength_scores)) if strength_scores else 0.0
    
    def _find_best_trading_hours(self, patterns: Dict) -> List[int]:
        """최적 거래 시간대 찾기"""
        if 'hourly' not in patterns:
            return []
        
        hourly_returns = [(hour, data['mean_return']) 
                         for hour, data in patterns['hourly'].items()]
        
        # 상위 3개 시간대
        best_hours = sorted(hourly_returns, key=lambda x: x[1], reverse=True)[:3]
        return [hour for hour, _ in best_hours]
    
    def _find_worst_trading_hours(self, patterns: Dict) -> List[int]:
        """최악 거래 시간대 찾기"""
        if 'hourly' not in patterns:
            return []
        
        hourly_returns = [(hour, data['mean_return']) 
                         for hour, data in patterns['hourly'].items()]
        
        # 하위 3개 시간대
        worst_hours = sorted(hourly_returns, key=lambda x: x[1])[:3]
        return [hour for hour, _ in worst_hours]
    
    def _volatility_analysis(self, prices: pd.Series) -> Dict:
        """변동성 분석"""
        returns = prices.pct_change().dropna()
        
        # 다양한 시간대 변동성
        volatility_1h = float(returns.std() * 100)
        volatility_24h = float(returns.rolling(24).std().mean() * 100) if len(returns) >= 24 else 0
        volatility_168h = float(returns.rolling(168).std().mean() * 100) if len(returns) >= 168 else 0
        
        # 변동성 클러스터링 (GARCH 효과)
        volatility_clustering = self._measure_volatility_clustering(returns)
        
        # 변동성 체제 변화 감지
        volatility_regimes = self._detect_volatility_regimes(returns)
        
        # 극단 움직임 빈도
        extreme_threshold = returns.std() * 2
        extreme_moves = returns[abs(returns) > extreme_threshold]
        extreme_frequency = len(extreme_moves) / len(returns) * 100
        
        return {
            'hourly_volatility': volatility_1h,
            'daily_volatility': volatility_24h,
            'weekly_volatility': volatility_168h,
            'volatility_clustering': float(volatility_clustering),
            'volatility_regimes': volatility_regimes,
            'extreme_move_frequency': float(extreme_frequency),
            'current_volatility_regime': self._classify_current_volatility(volatility_1h),
            'volatility_trend': self._analyze_volatility_trend(returns)
        }
    
    def _measure_volatility_clustering(self, returns: pd.Series) -> float:
        """변동성 클러스터링 측정 (자기상관)"""
        if len(returns) < 24:
            return 0.0
        
        abs_returns = abs(returns)
        autocorr = abs_returns.autocorr(lag=1)
        return autocorr if not np.isnan(autocorr) else 0.0
    
    def _detect_volatility_regimes(self, returns: pd.Series) -> List[Dict]:
        """변동성 체제 변화 감지"""
        if len(returns) < 48:
            return []
        
        # 24시간 롤링 변동성
        rolling_vol = returns.rolling(24).std() * 100
        rolling_vol = rolling_vol.dropna()
        
        # 변동성 임계값
        vol_mean = rolling_vol.mean()
        vol_std = rolling_vol.std()
        
        high_vol_threshold = vol_mean + vol_std
        low_vol_threshold = vol_mean - vol_std
        
        regimes = []
        current_regime = None
        regime_start = 0
        
        for i, vol in enumerate(rolling_vol):
            if vol > high_vol_threshold:
                regime = 'high_volatility'
            elif vol < low_vol_threshold:
                regime = 'low_volatility'
            else:
                regime = 'normal_volatility'
            
            if regime != current_regime:
                if current_regime is not None:
                    regimes.append({
                        'regime': current_regime,
                        'start_index': regime_start,
                        'end_index': i,
                        'duration': i - regime_start
                    })
                current_regime = regime
                regime_start = i
        
        # 마지막 체제 추가
        if current_regime is not None:
            regimes.append({
                'regime': current_regime,
                'start_index': regime_start,
                'end_index': len(rolling_vol),
                'duration': len(rolling_vol) - regime_start
            })
        
        return regimes
    
    def _classify_current_volatility(self, current_vol: float) -> str:
        """현재 변동성 체제 분류"""
        if current_vol > 5:
            return 'high'
        elif current_vol > 2:
            return 'medium'
        else:
            return 'low'
    
    def _analyze_volatility_trend(self, returns: pd.Series) -> Dict:
        """변동성 트렌드 분석"""
        if len(returns) < 48:
            return {'trend': 'insufficient_data'}
        
        # 12시간 롤링 변동성
        rolling_vol = returns.rolling(12).std() * 100
        rolling_vol = rolling_vol.dropna()
        
        # 변동성 트렌드
        x = np.arange(len(rolling_vol))
        slope, _, r_value, _, _ = stats.linregress(x, rolling_vol)
        
        trend_direction = 'increasing' if slope > 0 else 'decreasing'
        trend_strength = abs(r_value)
        
        return {
            'trend': trend_direction,
            'strength': float(trend_strength),
            'slope': float(slope)
        }
    
    def _cycle_analysis(self, prices: pd.Series) -> Dict:
        """주기 분석 (더 상세)"""
        
        # 가격 주기
        price_cycles = self._find_price_cycles(prices)
        
        # 변동성 주기
        returns = prices.pct_change().dropna()
        volatility_cycles = self._find_volatility_cycles(returns)
        
        return {
            'price_cycles': price_cycles,
            'volatility_cycles': volatility_cycles,
            'cycle_consistency': self._measure_cycle_consistency(price_cycles)
        }
    
    def _find_price_cycles(self, prices: pd.Series) -> List[Dict]:
        """가격 주기 찾기"""
        if len(prices) < 48:
            return []
        
        # 최고점과 최저점 찾기
        peaks, _ = find_peaks(prices, distance=12)  # 최소 12시간 간격
        troughs, _ = find_peaks(-prices, distance=12)
        
        cycles = []
        
        # 최고점 간 주기
        if len(peaks) > 1:
            peak_intervals = np.diff(peaks)
            avg_peak_cycle = float(np.mean(peak_intervals))
            cycles.append({
                'type': 'peak_to_peak',
                'average_hours': avg_peak_cycle,
                'count': len(peaks),
                'std_deviation': float(np.std(peak_intervals))
            })
        
        # 최저점 간 주기
        if len(troughs) > 1:
            trough_intervals = np.diff(troughs)
            avg_trough_cycle = float(np.mean(trough_intervals))
            cycles.append({
                'type': 'trough_to_trough',
                'average_hours': avg_trough_cycle,
                'count': len(troughs),
                'std_deviation': float(np.std(trough_intervals))
            })
        
        return cycles
    
    def _find_volatility_cycles(self, returns: pd.Series) -> List[Dict]:
        """변동성 주기 찾기"""
        if len(returns) < 48:
            return []
        
        # 절댓값 수익률 (변동성 프록시)
        abs_returns = abs(returns)
        
        # 변동성 피크 찾기
        vol_peaks, _ = find_peaks(abs_returns.rolling(6).mean(), distance=12)
        
        cycles = []
        if len(vol_peaks) > 1:
            vol_intervals = np.diff(vol_peaks)
            avg_vol_cycle = float(np.mean(vol_intervals))
            cycles.append({
                'type': 'volatility_peaks',
                'average_hours': avg_vol_cycle,
                'count': len(vol_peaks),
                'std_deviation': float(np.std(vol_intervals))
            })
        
        return cycles
    
    def _measure_cycle_consistency(self, price_cycles: List[Dict]) -> float:
        """주기 일관성 측정"""
        if not price_cycles:
            return 0.0
        
        consistency_scores = []
        for cycle in price_cycles:
            if cycle['average_hours'] > 0:
                # 변동계수의 역수 (낮은 변동계수 = 높은 일관성)
                cv = cycle['std_deviation'] / cycle['average_hours']
                consistency = 1 / (1 + cv)
                consistency_scores.append(consistency)
        
        return float(np.mean(consistency_scores)) if consistency_scores else 0.0
    
    def _momentum_analysis(self, prices: pd.Series) -> Dict:
        """모멘텀 분석"""
        
        # 다양한 시간대 모멘텀
        momentum_indicators = {}
        
        for period in [1, 6, 12, 24, 72, 168]:  # 1h, 6h, 12h, 1d, 3d, 1w
            if len(prices) > period:
                momentum = prices.iloc[-1] - prices.iloc[-period-1]
                momentum_pct = momentum / prices.iloc[-period-1] * 100
                momentum_indicators[f'{period}h'] = {
                    'absolute': float(momentum),
                    'percentage': float(momentum_pct)
                }
        
        # 모멘텀 방향 일치성
        momentum_values = [ind['percentage'] for ind in momentum_indicators.values()]
        positive_momentum = sum(1 for m in momentum_values if m > 0)
        momentum_alignment = positive_momentum / len(momentum_values) if momentum_values else 0
        
        # 모멘텀 강도
        momentum_strength = np.mean([abs(m) for m in momentum_values]) if momentum_values else 0
        
        # 모멘텀 가속도 (2차 도함수)
        momentum_acceleration = self._calculate_momentum_acceleration(prices)
        
        return {
            'momentum_indicators': momentum_indicators,
            'momentum_alignment': float(momentum_alignment),
            'momentum_strength': float(momentum_strength),
            'momentum_acceleration': momentum_acceleration,
            'dominant_momentum_direction': 'bullish' if momentum_alignment > 0.6 else 'bearish' if momentum_alignment < 0.4 else 'mixed'
        }
    
    def _calculate_momentum_acceleration(self, prices: pd.Series) -> Dict:
        """모멘텀 가속도 계산"""
        if len(prices) < 24:
            return {'status': 'insufficient_data'}
        
        # 12시간 모멘텀
        momentum_12h = prices.diff(12).dropna()
        
        # 모멘텀의 변화 (가속도)
        momentum_change = momentum_12h.diff().dropna()
        
        current_acceleration = momentum_change.iloc[-1] if len(momentum_change) > 0 else 0
        
        return {
            'current_acceleration': float(current_acceleration),
            'acceleration_trend': 'accelerating' if current_acceleration > 0 else 'decelerating',
            'acceleration_magnitude': float(abs(current_acceleration))
        }
    
    def _pattern_recognition(self, prices: pd.Series) -> Dict:
        """패턴 인식"""
        
        patterns = {
            'support_resistance': self._find_support_resistance_patterns(prices),
            'breakouts': self._detect_breakout_patterns(prices),
            'reversals': self._detect_reversal_patterns(prices),
            'consolidation': self._detect_consolidation_patterns(prices)
        }
        
        return patterns
    
    def _find_support_resistance_patterns(self, prices: pd.Series) -> Dict:
        """지지선/저항선 패턴"""
        if len(prices) < 48:
            return {'status': 'insufficient_data'}
        
        # 최고점과 최저점
        peaks, _ = find_peaks(prices, distance=12)
        troughs, _ = find_peaks(-prices, distance=12)
        
        # 저항선 (최고점들)
        resistance_levels = []
        if len(peaks) >= 2:
            peak_prices = prices.iloc[peaks]
            for level in peak_prices:
                nearby_peaks = peak_prices[abs(peak_prices - level) < level * 0.02]  # 2% 범위
                if len(nearby_peaks) >= 2:
                    resistance_levels.append({
                        'level': float(level),
                        'strength': len(nearby_peaks),
                        'type': 'resistance'
                    })
        
        # 지지선 (최저점들)
        support_levels = []
        if len(troughs) >= 2:
            trough_prices = prices.iloc[troughs]
            for level in trough_prices:
                nearby_troughs = trough_prices[abs(trough_prices - level) < level * 0.02]
                if len(nearby_troughs) >= 2:
                    support_levels.append({
                        'level': float(level),
                        'strength': len(nearby_troughs),
                        'type': 'support'
                    })
        
        return {
            'resistance_levels': resistance_levels,
            'support_levels': support_levels,
            'current_position': self._analyze_current_position(prices.iloc[-1], resistance_levels, support_levels)
        }
    
    def _analyze_current_position(self, current_price: float, 
                                resistance_levels: List[Dict], 
                                support_levels: List[Dict]) -> Dict:
        """현재 가격 위치 분석"""
        
        # 가장 가까운 저항선
        nearest_resistance = None
        min_resistance_distance = float('inf')
        
        for level in resistance_levels:
            distance = level['level'] - current_price
            if distance > 0 and distance < min_resistance_distance:
                min_resistance_distance = distance
                nearest_resistance = level
        
        # 가장 가까운 지지선
        nearest_support = None
        min_support_distance = float('inf')
        
        for level in support_levels:
            distance = current_price - level['level']
            if distance > 0 and distance < min_support_distance:
                min_support_distance = distance
                nearest_support = level
        
        return {
            'nearest_resistance': nearest_resistance,
            'nearest_support': nearest_support,
            'resistance_distance_pct': (min_resistance_distance / current_price * 100) if nearest_resistance else None,
            'support_distance_pct': (min_support_distance / current_price * 100) if nearest_support else None
        }
    
    def _detect_breakout_patterns(self, prices: pd.Series) -> List[Dict]:
        """브레이크아웃 패턴 감지"""
        if len(prices) < 72:
            return []
        
        breakouts = []
        
        # 72시간 롤링 최고/최저가
        rolling_high = prices.rolling(72).max()
        rolling_low = prices.rolling(72).min()
        
        for i in range(72, len(prices)):
            current_price = prices.iloc[i]
            prev_high = rolling_high.iloc[i-1]
            prev_low = rolling_low.iloc[i-1]
            
            # 상향 브레이크아웃
            if current_price > prev_high:
                breakout_strength = (current_price - prev_high) / prev_high * 100
                if breakout_strength > 2:  # 2% 이상
                    breakouts.append({
                        'type': 'upward_breakout',
                        'index': i,
                        'price': float(current_price),
                        'strength': float(breakout_strength),
                        'previous_resistance': float(prev_high)
                    })
            
            # 하향 브레이크아웃
            elif current_price < prev_low:
                breakout_strength = (prev_low - current_price) / prev_low * 100
                if breakout_strength > 2:
                    breakouts.append({
                        'type': 'downward_breakout',
                        'index': i,
                        'price': float(current_price),
                        'strength': float(breakout_strength),
                        'previous_support': float(prev_low)
                    })
        
        return breakouts[-10:]  # 최근 10개만 반환
    
    def _detect_reversal_patterns(self, prices: pd.Series) -> List[Dict]:
        """반전 패턴 감지"""
        if len(prices) < 48:
            return []
        
        reversals = []
        
        # 24시간 롤링 트렌드
        rolling_trend = prices.diff(24).rolling(12).mean()
        
        for i in range(36, len(rolling_trend)):
            current_trend = rolling_trend.iloc[i]
            prev_trend = rolling_trend.iloc[i-12]
            
            # 트렌드 반전 감지
            if (prev_trend > 0 and current_trend < 0) or (prev_trend < 0 and current_trend > 0):
                reversal_strength = abs(current_trend - prev_trend)
                
                reversals.append({
                    'type': 'trend_reversal',
                    'index': i,
                    'price': float(prices.iloc[i]),
                    'from_trend': 'bullish' if prev_trend > 0 else 'bearish',
                    'to_trend': 'bearish' if prev_trend > 0 else 'bullish',
                    'strength': float(reversal_strength)
                })
        
        return reversals[-5:]  # 최근 5개만 반환
    
    def _detect_consolidation_patterns(self, prices: pd.Series) -> List[Dict]:
        """횡보/통합 패턴 감지"""
        if len(prices) < 48:
            return []
        
        consolidations = []
        
        # 24시간 롤링 레인지
        rolling_range = (prices.rolling(24).max() - prices.rolling(24).min()) / prices.rolling(24).mean() * 100
        
        # 낮은 변동성 구간 찾기 (2% 미만)
        low_volatility_threshold = 2
        
        in_consolidation = False
        consolidation_start = None
        
        for i, vol in enumerate(rolling_range):
            if pd.isna(vol):
                continue
                
            if vol < low_volatility_threshold and not in_consolidation:
                # 횡보 시작
                in_consolidation = True
                consolidation_start = i
                
            elif vol >= low_volatility_threshold and in_consolidation:
                # 횡보 종료
                if consolidation_start is not None and i - consolidation_start >= 24:  # 최소 24시간
                    consolidation_period = prices.iloc[consolidation_start:i+1]
                    consolidations.append({
                        'start_index': consolidation_start,
                        'end_index': i,
                        'duration_hours': i - consolidation_start,
                        'price_range': {
                            'high': float(consolidation_period.max()),
                            'low': float(consolidation_period.min()),
                            'range_pct': float((consolidation_period.max() - consolidation_period.min()) / consolidation_period.mean() * 100)
                        }
                    })
                
                in_consolidation = False
                consolidation_start = None
        
        return consolidations[-3:]  # 최근 3개만 반환
    
    def _support_resistance_levels(self, prices: pd.Series) -> Dict:
        """지지/저항 레벨 상세 분석"""
        
        # 기본 지지/저항 찾기
        sr_patterns = self._find_support_resistance_patterns(prices)
        
        # 동적 지지/저항 (이동평균 기반)
        dynamic_levels = self._calculate_dynamic_support_resistance(prices)
        
        # 피벗 포인트
        pivot_points = self._calculate_pivot_points(prices)
        
        return {
            'static_levels': sr_patterns,
            'dynamic_levels': dynamic_levels,
            'pivot_points': pivot_points,
            'key_levels': self._identify_key_levels(sr_patterns, dynamic_levels)
        }
    
    def _calculate_dynamic_support_resistance(self, prices: pd.Series) -> Dict:
        """동적 지지/저항 계산"""
        
        dynamic_levels = {}
        
        # 이동평균 기반 동적 레벨
        for period in [20, 50, 100, 200]:
            if len(prices) >= period:
                ma = prices.rolling(period).mean().iloc[-1]
                dynamic_levels[f'MA_{period}'] = {
                    'level': float(ma),
                    'type': 'dynamic_resistance' if prices.iloc[-1] < ma else 'dynamic_support',
                    'period': period
                }
        
        # 볼린저 밴드
        if len(prices) >= 20:
            bb_middle = prices.rolling(20).mean()
            bb_std = prices.rolling(20).std()
            bb_upper = bb_middle + (bb_std * 2)
            bb_lower = bb_middle - (bb_std * 2)
            
            dynamic_levels['BB_Upper'] = {
                'level': float(bb_upper.iloc[-1]),
                'type': 'dynamic_resistance',
                'indicator': 'bollinger_bands'
            }
            dynamic_levels['BB_Lower'] = {
                'level': float(bb_lower.iloc[-1]),
                'type': 'dynamic_support',
                'indicator': 'bollinger_bands'
            }
        
        return dynamic_levels
    
    def _calculate_pivot_points(self, prices: pd.Series) -> Dict:
        """피벗 포인트 계산"""
        if len(prices) < 24:
            return {}
        
        # 최근 24시간 데이터
        recent_24h = prices.tail(24)
        high = recent_24h.max()
        low = recent_24h.min()
        close = recent_24h.iloc[-1]
        
        # 표준 피벗 포인트
        pivot = (high + low + close) / 3
        
        # 지지/저항 레벨
        r1 = 2 * pivot - low
        s1 = 2 * pivot - high
        r2 = pivot + (high - low)
        s2 = pivot - (high - low)
        r3 = high + 2 * (pivot - low)
        s3 = low - 2 * (high - pivot)
        
        return {
            'pivot': float(pivot),
            'resistance_levels': {
                'R1': float(r1),
                'R2': float(r2),
                'R3': float(r3)
            },
            'support_levels': {
                'S1': float(s1),
                'S2': float(s2),
                'S3': float(s3)
            }
        }
    
    def _identify_key_levels(self, static_levels: Dict, dynamic_levels: Dict) -> List[Dict]:
        """핵심 레벨 식별"""
        key_levels = []
        
        # 정적 레벨에서 강력한 것들
        if 'resistance_levels' in static_levels:
            for level in static_levels['resistance_levels']:
                if level['strength'] >= 3:  # 3번 이상 테스트된 레벨
                    key_levels.append({
                        'level': level['level'],
                        'type': 'key_resistance',
                        'strength': level['strength'],
                        'source': 'static'
                    })
        
        if 'support_levels' in static_levels:
            for level in static_levels['support_levels']:
                if level['strength'] >= 3:
                    key_levels.append({
                        'level': level['level'],
                        'type': 'key_support',
                        'strength': level['strength'],
                        'source': 'static'
                    })
        
        # 중요한 동적 레벨들
        important_dynamic = ['MA_50', 'MA_200', 'BB_Upper', 'BB_Lower']
        for level_name in important_dynamic:
            if level_name in dynamic_levels:
                level_data = dynamic_levels[level_name]
                key_levels.append({
                    'level': level_data['level'],
                    'type': f"key_{level_data['type'].split('_')[1]}",  # support 또는 resistance
                    'strength': 2,  # 동적 레벨은 중간 강도
                    'source': 'dynamic',
                    'indicator': level_name
                })
        
        return sorted(key_levels, key=lambda x: x['strength'], reverse=True)
    
    def _forecast_indicators(self, prices: pd.Series) -> Dict:
        """예측을 위한 핵심 지표들"""
        
        indicators = {}
        
        # 기술적 지표
        indicators.update(self._calculate_technical_indicators(prices))
        
        # 시계열 특성
        indicators.update(self._extract_timeseries_features(prices))
        
        # 예측 신호
        indicators['forecast_signals'] = self._generate_forecast_signals(prices)
        
        return indicators
    
    def _calculate_technical_indicators(self, prices: pd.Series) -> Dict:
        """기술적 지표 계산"""
        indicators = {}
        
        # RSI
        if len(prices) >= 14:
            rsi = self._calculate_rsi(prices, 14)
            indicators['RSI_14'] = float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50
        
        # MACD
        if len(prices) >= 26:
            macd_line, signal_line = self._calculate_macd(prices)
            histogram = macd_line - signal_line
            indicators['MACD'] = float(macd_line.iloc[-1]) if not pd.isna(macd_line.iloc[-1]) else 0
            indicators['MACD_Signal'] = float(signal_line.iloc[-1]) if not pd.isna(signal_line.iloc[-1]) else 0
            indicators['MACD_Histogram'] = float(histogram.iloc[-1]) if not pd.isna(histogram.iloc[-1]) else 0
        
        # 스토캐스틱
        if len(prices) >= 14:
            stoch_k, stoch_d = self._calculate_stochastic(prices, 14)
            indicators['Stoch_K'] = float(stoch_k.iloc[-1]) if not pd.isna(stoch_k.iloc[-1]) else 50
            indicators['Stoch_D'] = float(stoch_d.iloc[-1]) if not pd.isna(stoch_d.iloc[-1]) else 50
        
        # 윌리엄스 %R
        if len(prices) >= 14:
            williams_r = self._calculate_williams_r(prices, 14)
            indicators['Williams_R'] = float(williams_r.iloc[-1]) if not pd.isna(williams_r.iloc[-1]) else -50
        
        return indicators
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """RSI 계산"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series]:
        """MACD 계산"""
        exp1 = prices.ewm(span=fast).mean()
        exp2 = prices.ewm(span=slow).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=signal).mean()
        return macd, signal
    
    def _calculate_stochastic(self, prices: pd.Series, k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        """스토캐스틱 계산"""
        low_min = prices.rolling(k_period).min()
        high_max = prices.rolling(k_period).max()
        k_percent = 100 * ((prices - low_min) / (high_max - low_min))
        d_percent = k_percent.rolling(d_period).mean()
        return k_percent, d_percent
    
    def _calculate_williams_r(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """윌리엄스 %R 계산"""
        high_max = prices.rolling(period).max()
        low_min = prices.rolling(period).min()
        williams_r = -100 * ((high_max - prices) / (high_max - low_min))
        return williams_r
    
    def _extract_timeseries_features(self, prices: pd.Series) -> Dict:
        """시계열 특성 추출"""
        features = {}
        
        # 현재 가격 정보
        current_price = prices.iloc[-1]
        features['current_price'] = float(current_price)
        
        # 가격 레벨 (Z-score)
        mean_price = prices.mean()
        std_price = prices.std()
        features['price_zscore'] = float((current_price - mean_price) / std_price) if std_price > 0 else 0
        
        # 지연 특성
        for lag in [1, 6, 12, 24, 72]:
            if len(prices) > lag:
                lag_price = prices.iloc[-lag-1]
                features[f'price_lag_{lag}'] = float(lag_price)
                features[f'return_lag_{lag}'] = float((current_price - lag_price) / lag_price * 100)
        
        # 롤링 통계
        for window in [12, 24, 72, 168]:
            if len(prices) >= window:
                rolling_data = prices.rolling(window)
                features[f'sma_{window}'] = float(rolling_data.mean().iloc[-1])
                features[f'std_{window}'] = float(rolling_data.std().iloc[-1])
                features[f'min_{window}'] = float(rolling_data.min().iloc[-1])
                features[f'max_{window}'] = float(rolling_data.max().iloc[-1])
                
                # 현재 가격의 롤링 구간 내 위치
                features[f'price_position_{window}'] = float((current_price - rolling_data.min().iloc[-1]) / 
                                                           (rolling_data.max().iloc[-1] - rolling_data.min().iloc[-1]))
        
        return features
    
    def _generate_forecast_signals(self, prices: pd.Series) -> Dict:
        """예측 신호 생성"""
        signals = {}
        
        # 트렌드 신호
        if len(prices) >= 24:
            short_ma = prices.rolling(12).mean().iloc[-1]
            long_ma = prices.rolling(24).mean().iloc[-1]
            signals['trend_signal'] = 'bullish' if short_ma > long_ma else 'bearish'
            signals['trend_strength'] = float(abs(short_ma - long_ma) / long_ma * 100)
        
        # 모멘텀 신호
        if len(prices) >= 12:
            momentum = prices.iloc[-1] - prices.iloc[-12]
            signals['momentum_signal'] = 'positive' if momentum > 0 else 'negative'
            signals['momentum_strength'] = float(abs(momentum / prices.iloc[-12] * 100))
        
        # 변동성 신호
        if len(prices) >= 24:
            recent_vol = prices.pct_change().tail(24).std() * 100
            signals['volatility_signal'] = 'high' if recent_vol > 3 else 'low'
            signals['volatility_level'] = float(recent_vol)
        
        return signals
    
    def _market_regime_detection(self, prices: pd.Series) -> Dict:
        """시장 체제 감지"""
        
        if len(prices) < 168:  # 최소 1주일 데이터 필요
            return {'regime': 'insufficient_data', 'confidence': 0}
        
        # 다양한 시간대 트렌드
        trends = {}
        for period in [24, 72, 168]:  # 1일, 3일, 1주일
            if len(prices) >= period:
                trend_pct = (prices.iloc[-1] - prices.iloc[-period]) / prices.iloc[-period] * 100
                trends[f'{period}h'] = trend_pct
        
        # 변동성
        volatility = prices.pct_change().tail(168).std() * 100
        
        # 체제 분류 로직
        long_trend = trends.get('168h', 0)
        medium_trend = trends.get('72h', 0)
        short_trend = trends.get('24h', 0)
        
        # 강세장 조건
        if long_trend > 5 and medium_trend > 2 and volatility < 4:
            regime = 'strong_bull'
            confidence = 90
        elif long_trend > 2 and medium_trend > 0:
            regime = 'bull_market'
            confidence = 80
        
        # 약세장 조건
        elif long_trend < -5 and medium_trend < -2 and volatility < 4:
            regime = 'strong_bear'
            confidence = 90
        elif long_trend < -2 and medium_trend < 0:
            regime = 'bear_market'
            confidence = 80
        
        # 고변동성 조건
        elif volatility > 6:
            regime = 'high_volatility'
            confidence = 85
        
        # 횡보장 조건
        elif abs(long_trend) < 2 and volatility < 3:
            regime = 'sideways'
            confidence = 75
        
        else:
            regime = 'transitional'
            confidence = 60
        
        return {
            'regime': regime,
            'confidence': confidence,
            'trends': trends,
            'volatility': float(volatility),
            'regime_characteristics': self._get_regime_characteristics(regime)
        }
    
    def _get_regime_characteristics(self, regime: str) -> Dict:
        """체제별 특성"""
        characteristics = {
            'strong_bull': {
                'expected_accuracy': 85,
                'risk_level': 'medium',
                'trading_strategy': 'trend_following'
            },
            'bull_market': {
                'expected_accuracy': 80,
                'risk_level': 'medium',
                'trading_strategy': 'trend_following'
            },
            'strong_bear': {
                'expected_accuracy': 85,
                'risk_level': 'high',
                'trading_strategy': 'short_selling'
            },
            'bear_market': {
                'expected_accuracy': 75,
                'risk_level': 'high',
                'trading_strategy': 'defensive'
            },
            'high_volatility': {
                'expected_accuracy': 60,
                'risk_level': 'very_high',
                'trading_strategy': 'range_trading'
            },
            'sideways': {
                'expected_accuracy': 70,
                'risk_level': 'low',
                'trading_strategy': 'range_trading'
            },
            'transitional': {
                'expected_accuracy': 65,
                'risk_level': 'medium',
                'trading_strategy': 'wait_and_see'
            }
        }
        
        return characteristics.get(regime, {
            'expected_accuracy': 60,
            'risk_level': 'unknown',
            'trading_strategy': 'cautious'
        })
    
    def _anomaly_detection(self, prices: pd.Series) -> Dict:
        """이상치 감지"""
        
        anomalies = {
            'price_anomalies': self._detect_price_anomalies(prices),
            'volatility_anomalies': self._detect_volatility_anomalies(prices),
            'volume_anomalies': []  # 볼륨 데이터가 있을 경우 구현
        }
        
        return anomalies
    
    def _detect_price_anomalies(self, prices: pd.Series) -> List[Dict]:
        """가격 이상치 감지"""
        if len(prices) < 48:
            return []
        
        # Z-score 기반 이상치
        z_scores = np.abs((prices - prices.mean()) / prices.std())
        anomaly_threshold = 3  # 3σ
        
        price_anomalies = []
        for i, z_score in enumerate(z_scores):
            if z_score > anomaly_threshold:
                price_anomalies.append({
                    'index': i,
                    'price': float(prices.iloc[i]),
                    'z_score': float(z_score),
                    'anomaly_type': 'extreme_price'
                })
        
        return price_anomalies[-10:]  # 최근 10개만
    
    def _detect_volatility_anomalies(self, prices: pd.Series) -> List[Dict]:
        """변동성 이상치 감지"""
        if len(prices) < 48:
            return []
        
        returns = prices.pct_change().dropna()
        rolling_vol = returns.rolling(24).std() * 100
        
        # 변동성 이상치
        vol_mean = rolling_vol.mean()
        vol_std = rolling_vol.std()
        
        volatility_anomalies = []
        for i, vol in enumerate(rolling_vol):
            if not pd.isna(vol):
                z_score = abs(vol - vol_mean) / vol_std if vol_std > 0 else 0
                if z_score > 2:  # 2σ 이상
                    volatility_anomalies.append({
                        'index': i,
                        'volatility': float(vol),
                        'z_score': float(z_score),
                        'anomaly_type': 'extreme_volatility'
                    })
        
        return volatility_anomalies[-5:]  # 최근 5개만

# 사용 예제
def main():
    """시계열 분석 엔진 테스트"""
    
    # 샘플 데이터 생성 (실제로는 CSV에서 로드)
    import os
    
    data_path = "/Users/parkyoungjun/Desktop/BTC_Analysis_System/ai_optimized_3month_data/ai_matrix_complete.csv"
    
    if os.path.exists(data_path):
        data = pd.read_csv(data_path)
        
        # 시계열 분석 엔진 생성
        engine = TimeSeriesEngine()
        
        # BTC 가격 컬럼 찾기
        price_candidates = [
            'onchain_blockchain_info_network_stats_market_price_usd',
            'btc_price', 'price', 'close', 'market_price'
        ]
        
        price_column = None
        for candidate in price_candidates:
            if candidate in data.columns:
                price_column = candidate
                break
        
        if price_column:
            print(f"📊 BTC 가격 컬럼: {price_column}")
            
            # 종합 분석 실행
            analysis = engine.comprehensive_timeseries_analysis(data, price_column)
            
            print("✅ 시계열 분석 완료!")
            print(f"🎯 현재 시장 체제: {analysis['market_regime_detection']['regime']}")
            print(f"📈 트렌드 방향: {analysis['trend_analysis']['trend_direction']}")
            print(f"💹 변동성 수준: {analysis['volatility_analysis']['current_volatility_regime']}")
            
            return analysis
        else:
            print("❌ BTC 가격 컬럼을 찾을 수 없음")
    else:
        print(f"❌ 데이터 파일 없음: {data_path}")
    
    return None

if __name__ == "__main__":
    results = main()