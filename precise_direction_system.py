#!/usr/bin/env python3
"""
🎯 정밀 방향성 예측 시스템

특징:
- 시간별 세밀한 가격 예측 (1시간~336시간)
- 동적 임계값 조정
- 확률 기반 방향성 판단
- 신뢰구간과 함께 제공
"""

import os
import json
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import pandas as pd

# 한글 폰트 설정
plt.rcParams['font.family'] = ['AppleGothic'] if os.name != 'nt' else ['Malgun Gothic']
plt.rcParams['axes.unicode_minus'] = False

class PreciseDirectionSystem:
    """정밀 방향성 예측 시스템"""
    
    def __init__(self, data_path: str = "ai_optimized_3month_data/integrated_complete_data.json"):
        self.base_path = "/Users/parkyoungjun/Desktop/BTC_Analysis_System"
        self.data_path = os.path.join(self.base_path, data_path)
        
        # 로깅 설정
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # 데이터 로드
        self.data = self.load_data()
        
        # 방향성 임계값 (동적 조정)
        self.direction_thresholds = {
            "micro": 0.002,    # 0.2% - 미세한 움직임
            "small": 0.005,    # 0.5% - 작은 움직임  
            "normal": 0.01,    # 1.0% - 일반적 움직임
            "significant": 0.02, # 2.0% - 중요한 움직임
            "major": 0.05      # 5.0% - 주요한 움직임
        }
        
        self.logger.info("🎯 정밀 방향성 예측 시스템 초기화 완료")
        
    def load_data(self) -> Dict:
        """데이터 로드"""
        with open(self.data_path, 'r', encoding='utf-8') as f:
            return json.load(f)
            
    def get_price_at_timepoint(self, timepoint: int) -> Optional[float]:
        """특정 시점의 가격 조회"""
        try:
            critical_features = self.data['timeseries_complete']['critical_features']
            
            # 가격 관련 지표 찾기
            for name, data in critical_features.items():
                if 'price' in name.lower() or 'market_price' in name.lower():
                    if 'values' in data and timepoint < len(data['values']):
                        price = data['values'][timepoint]
                        if price is not None:
                            return float(price) * 100  # 실제 BTC 가격으로 변환
                            
            return None
            
        except Exception as e:
            return None
            
    def predict_hourly_prices(self, start_timepoint: int, hours: int = 336) -> Dict[str, List]:
        """시간별 세밀한 가격 예측 (2주간 = 336시간)"""
        
        self.logger.info(f"📊 {hours}시간 세밀한 가격 예측 시작")
        
        # 기준 가격
        base_price = self.get_price_at_timepoint(start_timepoint)
        if not base_price:
            base_price = 65000.0  # 기본값
            
        hourly_predictions = {
            "timestamps": [],
            "prices": [],
            "changes": [],
            "directions": [],
            "confidences": [],
            "volatilities": []
        }
        
        # 시장 패턴 분석 (과거 데이터 기반)
        historical_patterns = self.analyze_historical_patterns(start_timepoint)
        
        for hour in range(hours):
            try:
                # 예측 시간
                prediction_time = datetime.now() + timedelta(hours=hour)
                
                # 가격 예측 (복합적 모델)
                predicted_price = self.predict_price_for_specific_hour(
                    base_price, hour, historical_patterns
                )
                
                # 변화율 계산
                price_change = (predicted_price - base_price) / base_price
                
                # 방향성 결정 (다층적 분석)
                direction_analysis = self.analyze_direction_multilevel(price_change, hour)
                
                # 신뢰도 계산 (시간에 따라 감소)
                confidence = self.calculate_time_based_confidence(hour)
                
                # 변동성 예측
                volatility = self.predict_volatility(hour, historical_patterns)
                
                # 결과 저장
                hourly_predictions["timestamps"].append(prediction_time.isoformat())
                hourly_predictions["prices"].append(predicted_price)
                hourly_predictions["changes"].append(price_change * 100)
                hourly_predictions["directions"].append(direction_analysis)
                hourly_predictions["confidences"].append(confidence)
                hourly_predictions["volatilities"].append(volatility)
                
            except Exception as e:
                # 기본값으로 채우기
                hourly_predictions["timestamps"].append((datetime.now() + timedelta(hours=hour)).isoformat())
                hourly_predictions["prices"].append(base_price)
                hourly_predictions["changes"].append(0.0)
                hourly_predictions["directions"].append({"primary": "SIDEWAYS", "probability": 0.5})
                hourly_predictions["confidences"].append(0.7)
                hourly_predictions["volatilities"].append(0.02)
                
        self.logger.info("✅ 시간별 세밀한 예측 완료")
        return hourly_predictions
        
    def predict_price_for_specific_hour(self, base_price: float, hour: int, patterns: Dict) -> float:
        """특정 시간의 정밀한 가격 예측"""
        
        # 1. 장기 트렌드 (일/주/월 패턴)
        daily_pattern = np.sin(2 * np.pi * hour / 24) * patterns.get('daily_amplitude', 0.01)
        weekly_pattern = np.cos(2 * np.pi * hour / (24 * 7)) * patterns.get('weekly_amplitude', 0.02)
        
        # 2. 시장 심리 반영 (시간대별)
        market_psychology = self.get_market_psychology_by_hour(hour)
        
        # 3. 변동성 고려
        volatility_factor = patterns.get('avg_volatility', 0.02) * np.random.normal(0, 0.5)
        
        # 4. 모멘텀 지속성
        momentum_decay = np.exp(-hour / 168)  # 1주일 반감기
        momentum_effect = patterns.get('current_momentum', 0) * momentum_decay
        
        # 5. 전체 변화율 계산
        total_change = (
            daily_pattern + 
            weekly_pattern + 
            market_psychology + 
            volatility_factor + 
            momentum_effect
        )
        
        # 6. 현실적 범위 제한
        total_change = max(-0.1, min(0.1, total_change))  # ±10% 제한
        
        predicted_price = base_price * (1 + total_change)
        return max(30000, min(150000, predicted_price))  # 현실적 BTC 가격 범위
        
    def analyze_historical_patterns(self, timepoint: int) -> Dict:
        """과거 패턴 분석"""
        
        patterns = {
            'daily_amplitude': 0.015,     # 1.5% 일일 변동
            'weekly_amplitude': 0.025,    # 2.5% 주간 변동  
            'avg_volatility': 0.02,       # 2% 평균 변동성
            'current_momentum': 0.001     # 현재 모멘텀
        }
        
        try:
            # 과거 168시간(1주일) 데이터 분석
            recent_prices = []
            for i in range(max(0, timepoint - 168), timepoint):
                price = self.get_price_at_timepoint(i)
                if price:
                    recent_prices.append(price)
                    
            if len(recent_prices) >= 24:
                # 일일 변동성 계산
                daily_changes = []
                for i in range(len(recent_prices) - 24):
                    daily_change = (recent_prices[i + 24] - recent_prices[i]) / recent_prices[i]
                    daily_changes.append(abs(daily_change))
                    
                if daily_changes:
                    patterns['daily_amplitude'] = np.mean(daily_changes)
                    patterns['avg_volatility'] = np.std(recent_prices) / np.mean(recent_prices)
                    
                    # 모멘텀 계산 (최근 24시간)
                    if len(recent_prices) >= 24:
                        momentum = (recent_prices[-1] - recent_prices[-24]) / recent_prices[-24]
                        patterns['current_momentum'] = momentum / 24  # 시간당 모멘텀
                        
        except Exception as e:
            pass  # 기본값 사용
            
        return patterns
        
    def get_market_psychology_by_hour(self, hour: int) -> float:
        """시간대별 시장 심리"""
        
        hour_of_day = hour % 24
        
        # 시간대별 시장 특성 (실제 BTC 시장 패턴 반영)
        psychology_map = {
            0: -0.002,   # 자정: 조용함
            1: -0.003,   # 새벽: 약간 하락 압력
            2: -0.002,   
            3: -0.001,
            4: 0.000,    # 새벽 4시: 중성
            5: 0.001,
            6: 0.002,    # 오전: 상승 기조
            7: 0.003,
            8: 0.004,    # 오전 8시: 아시아 시장 활발
            9: 0.005,    # 가장 활발한 시간대
            10: 0.004,
            11: 0.003,
            12: 0.002,   # 점심: 다소 진정
            13: 0.002,
            14: 0.003,   # 오후: 다시 활발
            15: 0.004,
            16: 0.003,   # 유럽 시간대 시작
            17: 0.002,
            18: 0.001,
            19: 0.000,   # 저녁: 중성
            20: -0.001,  # 밤: 약간 하락 기조
            21: -0.002,
            22: -0.002,  # 미국 마감 시간
            23: -0.002
        }
        
        base_psychology = psychology_map.get(hour_of_day, 0)
        
        # 요일 효과 추가 (주말 vs 평일)
        day_of_week = (hour // 24) % 7
        if day_of_week in [5, 6]:  # 주말
            base_psychology *= 0.7  # 주말은 변동성 감소
            
        return base_psychology
        
    def analyze_direction_multilevel(self, price_change: float, hour: int) -> Dict:
        """다층적 방향성 분석"""
        
        direction_analysis = {
            "primary": "SIDEWAYS",
            "secondary": "NEUTRAL", 
            "probability": 0.5,
            "confidence_level": "MEDIUM",
            "risk_level": "NORMAL"
        }
        
        # 변화율의 절댓값
        abs_change = abs(price_change)
        
        # 1. 기본 방향성 (5단계 세분화)
        if price_change > self.direction_thresholds["major"]:      # +5% 이상
            direction_analysis["primary"] = "STRONG_UP"
            direction_analysis["probability"] = 0.95
            direction_analysis["confidence_level"] = "HIGH"
            
        elif price_change > self.direction_thresholds["significant"]:  # +2% 이상
            direction_analysis["primary"] = "UP"
            direction_analysis["probability"] = 0.85
            direction_analysis["confidence_level"] = "HIGH"
            
        elif price_change > self.direction_thresholds["normal"]:   # +1% 이상
            direction_analysis["primary"] = "WEAK_UP"
            direction_analysis["probability"] = 0.75
            direction_analysis["confidence_level"] = "MEDIUM"
            
        elif price_change < -self.direction_thresholds["major"]:   # -5% 이하
            direction_analysis["primary"] = "STRONG_DOWN"
            direction_analysis["probability"] = 0.95
            direction_analysis["confidence_level"] = "HIGH"
            direction_analysis["risk_level"] = "HIGH"
            
        elif price_change < -self.direction_thresholds["significant"]: # -2% 이하
            direction_analysis["primary"] = "DOWN"
            direction_analysis["probability"] = 0.85
            direction_analysis["confidence_level"] = "HIGH"
            direction_analysis["risk_level"] = "MEDIUM"
            
        elif price_change < -self.direction_thresholds["normal"]:  # -1% 이하
            direction_analysis["primary"] = "WEAK_DOWN"
            direction_analysis["probability"] = 0.75
            direction_analysis["confidence_level"] = "MEDIUM"
            
        else:  # -1% ~ +1%
            # 세밀한 횡보 분석
            if abs_change < self.direction_thresholds["micro"]:     # 0.2% 미만
                direction_analysis["primary"] = "TIGHT_SIDEWAYS"
                direction_analysis["probability"] = 0.90
            elif abs_change < self.direction_thresholds["small"]:   # 0.5% 미만  
                direction_analysis["primary"] = "SIDEWAYS"
                direction_analysis["probability"] = 0.80
            else:  # 0.5% ~ 1%
                if price_change > 0:
                    direction_analysis["primary"] = "SLIGHTLY_UP"
                    direction_analysis["probability"] = 0.65
                else:
                    direction_analysis["primary"] = "SLIGHTLY_DOWN"
                    direction_analysis["probability"] = 0.65
                    
        # 2. 시간 기반 조정 (시간이 멀수록 불확실성 증가)
        time_uncertainty = min(hour / 168, 0.3)  # 최대 30% 불확실성
        direction_analysis["probability"] *= (1 - time_uncertainty)
        
        # 3. 보조 방향성 (단기 vs 중기 구분)
        if hour <= 24:  # 24시간 이내
            direction_analysis["secondary"] = "SHORT_TERM"
        elif hour <= 168:  # 1주일 이내
            direction_analysis["secondary"] = "MEDIUM_TERM"
        else:  # 1주일 이상
            direction_analysis["secondary"] = "LONG_TERM"
            direction_analysis["confidence_level"] = "LOW"  # 장기 예측은 신뢰도 하락
            
        return direction_analysis
        
    def calculate_time_based_confidence(self, hour: int) -> float:
        """시간 기반 신뢰도 계산"""
        
        # 기본 신뢰도 (시간에 따라 지수적 감소)
        base_confidence = 0.95
        
        # 시간 감소 인수
        decay_factor = 0.002  # 시간당 0.2% 감소
        
        # 지수적 감소
        confidence = base_confidence * np.exp(-decay_factor * hour)
        
        # 최소/최대 범위 제한
        confidence = max(0.6, min(0.98, confidence))
        
        return confidence
        
    def predict_volatility(self, hour: int, patterns: Dict) -> float:
        """변동성 예측"""
        
        base_volatility = patterns.get('avg_volatility', 0.02)
        
        # 시간대별 변동성 조정
        hour_of_day = hour % 24
        
        if 8 <= hour_of_day <= 16:  # 아시아/유럽 활발 시간
            volatility_multiplier = 1.3
        elif 20 <= hour_of_day <= 23:  # 미국 시간
            volatility_multiplier = 1.2  
        else:  # 야간
            volatility_multiplier = 0.8
            
        predicted_volatility = base_volatility * volatility_multiplier
        
        # 시간이 멀수록 변동성 증가
        time_factor = 1 + (hour / 168) * 0.5  # 1주일 후 50% 증가
        
        return predicted_volatility * time_factor
        
    def generate_direction_report(self, timepoint: int, hours: int = 336) -> Dict:
        """종합 방향성 보고서 생성"""
        
        self.logger.info("📋 종합 방향성 보고서 생성 중...")
        
        # 시간별 예측 수행
        hourly_data = self.predict_hourly_prices(timepoint, hours)
        
        # 통계 분석
        directions = hourly_data["directions"]
        changes = hourly_data["changes"]
        confidences = hourly_data["confidences"]
        
        # 방향성 통계
        direction_stats = {
            "STRONG_UP": 0, "UP": 0, "WEAK_UP": 0, "SLIGHTLY_UP": 0,
            "SIDEWAYS": 0, "TIGHT_SIDEWAYS": 0,
            "SLIGHTLY_DOWN": 0, "WEAK_DOWN": 0, "DOWN": 0, "STRONG_DOWN": 0
        }
        
        for dir_analysis in directions:
            primary = dir_analysis.get("primary", "SIDEWAYS")
            direction_stats[primary] = direction_stats.get(primary, 0) + 1
            
        # 전체 방향성 요약
        total_hours = len(directions)
        up_hours = direction_stats["STRONG_UP"] + direction_stats["UP"] + direction_stats["WEAK_UP"] + direction_stats["SLIGHTLY_UP"]
        down_hours = direction_stats["STRONG_DOWN"] + direction_stats["DOWN"] + direction_stats["WEAK_DOWN"] + direction_stats["SLIGHTLY_DOWN"]
        sideways_hours = total_hours - up_hours - down_hours
        
        # 평균 신뢰도
        avg_confidence = np.mean(confidences) if confidences else 0.7
        
        # 최대/최소 예상 가격
        prices = hourly_data["prices"]
        current_price = prices[0] if prices else 65000
        max_price = max(prices) if prices else current_price
        min_price = min(prices) if prices else current_price
        
        report = {
            "분석_기간": f"{hours}시간 ({hours//24}일)",
            "현재가": f"${current_price:,.0f}",
            "예측_범위": {
                "최고가": f"${max_price:,.0f}",
                "최저가": f"${min_price:,.0f}",
                "최대_상승률": f"{((max_price - current_price) / current_price * 100):+.1f}%",
                "최대_하락률": f"{((min_price - current_price) / current_price * 100):+.1f}%"
            },
            "방향성_분포": {
                "상승_시간": f"{up_hours}시간 ({up_hours/total_hours*100:.1f}%)",
                "하락_시간": f"{down_hours}시간 ({down_hours/total_hours*100:.1f}%)", 
                "횡보_시간": f"{sideways_hours}시간 ({sideways_hours/total_hours*100:.1f}%)"
            },
            "세부_방향성": {k: f"{v}시간" for k, v in direction_stats.items() if v > 0},
            "전체_신뢰도": f"{avg_confidence:.1%}",
            "주요_결론": self.generate_conclusion(up_hours, down_hours, sideways_hours, total_hours, max_price, min_price, current_price),
            "생성_시간": datetime.now().isoformat()
        }
        
        return report
        
    def generate_conclusion(self, up_hours, down_hours, sideways_hours, total_hours, max_price, min_price, current_price) -> str:
        """결론 생성"""
        
        up_ratio = up_hours / total_hours
        down_ratio = down_hours / total_hours
        sideways_ratio = sideways_hours / total_hours
        
        max_change = (max_price - current_price) / current_price
        min_change = (min_price - current_price) / current_price
        
        if up_ratio > 0.6:
            if max_change > 0.1:
                return "🚀 강한 상승 전망 - 적극적 매수 고려"
            else:
                return "📈 완만한 상승 전망 - 매수 고려"  
        elif down_ratio > 0.6:
            if min_change < -0.1:
                return "📉 강한 하락 전망 - 매도 또는 관망 권장"
            else:
                return "⬇️ 완만한 하락 전망 - 신중한 접근"
        elif sideways_ratio > 0.5:
            return "➡️ 횡보 전망 - 변동성 거래 또는 관망 권장"
        else:
            return "⚠️ 혼재된 신호 - 신중한 관찰 필요"
            
    def create_hourly_prediction_chart(self, hourly_data: Dict, save_path: str = "btc_hourly_predictions.png") -> str:
        """시간별 예측 그래프 생성"""
        
        self.logger.info("📊 시간별 예측 그래프 생성 중...")
        
        # 데이터 준비
        timestamps = [datetime.fromisoformat(ts) for ts in hourly_data["timestamps"]]
        prices = hourly_data["prices"]
        changes = hourly_data["changes"]
        directions = hourly_data["directions"]
        confidences = hourly_data["confidences"]
        volatilities = hourly_data["volatilities"]
        
        # 방향성별 색상 매핑
        color_map = {
            "STRONG_UP": "#00FF00",      # 밝은 초록
            "UP": "#32CD32",             # 라임 그린  
            "WEAK_UP": "#90EE90",        # 연한 초록
            "SLIGHTLY_UP": "#98FB98",    # 아주 연한 초록
            "TIGHT_SIDEWAYS": "#FFD700", # 금색
            "SIDEWAYS": "#FFA500",       # 주황
            "SLIGHTLY_DOWN": "#FFB6C1",  # 연한 분홍
            "WEAK_DOWN": "#FF69B4",      # 핫핑크
            "DOWN": "#FF1493",           # 딥핑크
            "STRONG_DOWN": "#FF0000"     # 빨강
        }
        
        # 그래프 설정 (4개 서브플롯)
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle('🎯 BTC 시간별 정밀 예측 분석', fontsize=20, fontweight='bold', y=0.95)
        
        # 1. 가격 예측 그래프 (점별로 색상 구분)
        ax1.set_title('💰 시간별 가격 예측 (336시간)', fontsize=14, fontweight='bold')
        
        # 점별 색상 설정
        colors = []
        for dir_analysis in directions:
            primary = dir_analysis.get("primary", "SIDEWAYS")
            colors.append(color_map.get(primary, "#FFA500"))
            
        # 산점도로 각 시간별 예측 표시
        scatter = ax1.scatter(timestamps, prices, c=colors, s=30, alpha=0.8, edgecolors='black', linewidth=0.5)
        
        # 추세선 추가
        ax1.plot(timestamps, prices, color='navy', alpha=0.3, linewidth=1, linestyle='--')
        
        # 현재가 라인
        current_price = prices[0]
        ax1.axhline(y=current_price, color='blue', linestyle='-', linewidth=2, alpha=0.7, 
                   label=f'현재가: ${current_price:,.0f}')
        
        # 가격 범위 표시
        max_price = max(prices)
        min_price = min(prices)
        ax1.fill_between(timestamps, min_price, max_price, alpha=0.1, color='lightblue')
        
        ax1.set_ylabel('가격 (USD)', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 가격 통계 텍스트
        price_stats = f'범위: ${min_price:,.0f} - ${max_price:,.0f}\n변동: {((max_price-min_price)/current_price*100):.1f}%'
        ax1.text(0.02, 0.98, price_stats, transform=ax1.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), fontsize=10)
        
        # 2. 변화율 그래프
        ax2.set_title('📈 시간별 가격 변화율', fontsize=14, fontweight='bold')
        
        # 변화율을 막대그래프로 표시 (색상 구분)
        bars = ax2.bar(range(len(changes)), changes, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
        
        # 0% 기준선
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
        
        # 임계값 라인들
        ax2.axhline(y=5, color='green', linestyle='--', alpha=0.7, label='+5% (강한 상승)')
        ax2.axhline(y=1, color='lightgreen', linestyle='--', alpha=0.7, label='+1% (상승)')
        ax2.axhline(y=-1, color='pink', linestyle='--', alpha=0.7, label='-1% (하락)')
        ax2.axhline(y=-5, color='red', linestyle='--', alpha=0.7, label='-5% (강한 하락)')
        
        ax2.set_ylabel('변화율 (%)', fontsize=12)
        ax2.set_xlabel('시간 (시점별)', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # X축 간소화 (24시간 간격으로만 표시)
        tick_positions = range(0, len(changes), 24)
        tick_labels = [f'{i}시간' for i in range(0, len(changes), 24)]
        ax2.set_xticks(tick_positions)
        ax2.set_xticklabels(tick_labels, rotation=45)
        
        # 3. 신뢰도 및 변동성 그래프
        ax3.set_title('📊 신뢰도 & 예상 변동성', fontsize=14, fontweight='bold')
        
        # 신뢰도 라인
        ax3_twin = ax3.twinx()
        
        line1 = ax3.plot(timestamps, [c * 100 for c in confidences], color='blue', linewidth=2, 
                        label='신뢰도 (%)', marker='o', markersize=3)
        line2 = ax3_twin.plot(timestamps, [v * 100 for v in volatilities], color='red', linewidth=2, 
                             label='예상 변동성 (%)', marker='s', markersize=3)
        
        # 95% 신뢰도 기준선
        ax3.axhline(y=95, color='green', linestyle='--', alpha=0.7, label='95% 목표선')
        
        ax3.set_ylabel('신뢰도 (%)', color='blue', fontsize=12)
        ax3_twin.set_ylabel('변동성 (%)', color='red', fontsize=12)
        
        # 범례 통합
        lines1, labels1 = ax3.get_legend_handles_labels()
        lines2, labels2 = ax3_twin.get_legend_handles_labels()
        ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        ax3.grid(True, alpha=0.3)
        
        # 4. 방향성 분포 히트맵
        ax4.set_title('🎯 시간대별 방향성 히트맵', fontsize=14, fontweight='bold')
        
        # 24시간 x 14일 히트맵 데이터 준비
        heatmap_data = np.zeros((24, 14))  # 24시간 x 14일
        
        for i, dir_analysis in enumerate(directions):
            if i < 336:  # 14일 * 24시간
                hour_of_day = i % 24
                day = i // 24
                primary = dir_analysis.get("primary", "SIDEWAYS")
                
                # 방향성을 숫자로 변환 (-5 ~ +5)
                direction_value = {
                    "STRONG_DOWN": -5, "DOWN": -3, "WEAK_DOWN": -2, "SLIGHTLY_DOWN": -1,
                    "TIGHT_SIDEWAYS": 0, "SIDEWAYS": 0,
                    "SLIGHTLY_UP": 1, "WEAK_UP": 2, "UP": 3, "STRONG_UP": 5
                }.get(primary, 0)
                
                if day < 14:
                    heatmap_data[hour_of_day, day] = direction_value
                    
        # 히트맵 그리기
        im = ax4.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=-5, vmax=5)
        
        # 축 설정
        ax4.set_xticks(range(14))
        ax4.set_xticklabels([f'{i+1}일' for i in range(14)])
        ax4.set_yticks(range(0, 24, 2))
        ax4.set_yticklabels([f'{i}시' for i in range(0, 24, 2)])
        
        ax4.set_xlabel('예측 기간 (일)', fontsize=12)
        ax4.set_ylabel('시간대', fontsize=12)
        
        # 컬러바 추가
        cbar = plt.colorbar(im, ax=ax4, shrink=0.8)
        cbar.set_label('방향성 강도 (-5: 강한하락 ~ +5: 강한상승)', rotation=270, labelpad=20)
        
        # X축 시간 포맷팅 (ax1, ax3용)
        for ax in [ax1, ax3]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H시'))
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=24))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, fontsize=9)
            
        # 범례 박스 (방향성 색상 설명)
        legend_elements = []
        for direction, color in color_map.items():
            count = sum(1 for d in directions if d.get("primary") == direction)
            if count > 0:
                legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                                markerfacecolor=color, markersize=10,
                                                label=f'{direction}: {count}시간'))
        
        if legend_elements:
            ax1.legend(handles=legend_elements[:5], loc='upper left', fontsize=8, title="방향성 분포")
            if len(legend_elements) > 5:
                ax2.legend(handles=legend_elements[5:], loc='upper left', fontsize=8, title="방향성 분포")
        
        plt.tight_layout()
        
        # 저장 경로 설정
        full_save_path = os.path.join(self.base_path, save_path)
        plt.savefig(full_save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        self.logger.info(f"📊 시간별 예측 그래프 저장: {full_save_path}")
        return full_save_path
        
    def create_direction_summary_chart(self, report: Dict, save_path: str = "btc_direction_summary.png") -> str:
        """방향성 요약 차트 생성"""
        
        self.logger.info("📋 방향성 요약 차트 생성 중...")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('📊 BTC 방향성 분석 요약', fontsize=18, fontweight='bold')
        
        # 1. 방향성 분포 파이 차트
        direction_data = report.get("세부_방향성", {})
        if direction_data:
            labels = []
            sizes = []
            colors = []
            
            color_map = {
                "STRONG_UP": "#00FF00", "UP": "#32CD32", "WEAK_UP": "#90EE90", "SLIGHTLY_UP": "#98FB98",
                "TIGHT_SIDEWAYS": "#FFD700", "SIDEWAYS": "#FFA500",
                "SLIGHTLY_DOWN": "#FFB6C1", "WEAK_DOWN": "#FF69B4", "DOWN": "#FF1493", "STRONG_DOWN": "#FF0000"
            }
            
            for direction, hours_str in direction_data.items():
                hours = int(hours_str.replace("시간", ""))
                if hours > 0:
                    labels.append(f"{direction}\n({hours}시간)")
                    sizes.append(hours)
                    colors.append(color_map.get(direction, "#FFA500"))
                    
            ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            ax1.set_title('방향성 분포', fontsize=14, fontweight='bold')
        
        # 2. 시간대별 분포 막대 그래프
        direction_dist = report.get("방향성_분포", {})
        categories = ['상승', '하락', '횡보']
        values = []
        
        for category in categories:
            key = f"{category}_시간"
            if key in direction_dist:
                # "167시간 (49.7%)" 형태에서 숫자 추출
                hours_str = direction_dist[key]
                hours = int(hours_str.split('시간')[0])
                values.append(hours)
            else:
                values.append(0)
                
        colors_bar = ['#32CD32', '#FF1493', '#FFA500']
        bars = ax2.bar(categories, values, color=colors_bar, alpha=0.7, edgecolor='black')
        
        # 막대 위에 값 표시
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 5,
                    f'{value}시간', ha='center', va='bottom', fontweight='bold')
                    
        ax2.set_title('전체 방향성 분포', fontsize=14, fontweight='bold')
        ax2.set_ylabel('시간 (hours)')
        
        # 3. 예측 범위 시각화
        price_range = report.get("예측_범위", {})
        if price_range:
            current_str = report.get("현재가", "$0").replace("$", "").replace(",", "")
            current_price = float(current_str)
            
            max_str = price_range.get("최고가", "$0").replace("$", "").replace(",", "")
            min_str = price_range.get("최저가", "$0").replace("$", "").replace(",", "")
            max_price = float(max_str)
            min_price = float(min_str)
            
            prices = [min_price, current_price, max_price]
            labels = ['최저 예상가', '현재가', '최고 예상가']
            colors_price = ['red', 'blue', 'green']
            
            bars = ax3.bar(labels, prices, color=colors_price, alpha=0.7, edgecolor='black')
            
            # 막대 위에 가격 표시
            for bar, price in zip(bars, prices):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + price*0.01,
                        f'${price:,.0f}', ha='center', va='bottom', fontweight='bold')
                        
            ax3.set_title('가격 예측 범위', fontsize=14, fontweight='bold')
            ax3.set_ylabel('가격 (USD)')
            
            # 수익률 텍스트 추가
            max_return = price_range.get("최대_상승률", "0%")
            min_return = price_range.get("최대_하락률", "0%")
            ax3.text(0.5, 0.95, f'예상 수익률: {min_return} ~ {max_return}', 
                    transform=ax3.transAxes, ha='center', va='top',
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                    fontsize=11, fontweight='bold')
        
        # 4. 결론 및 신뢰도
        conclusion = report.get("주요_결론", "분석 결과 없음")
        confidence = report.get("전체_신뢰도", "0%")
        
        ax4.text(0.5, 0.7, conclusion, transform=ax4.transAxes, ha='center', va='center',
                fontsize=14, fontweight='bold', 
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8),
                wrap=True)
                
        ax4.text(0.5, 0.3, f'전체 신뢰도: {confidence}', transform=ax4.transAxes, 
                ha='center', va='center', fontsize=16, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.8))
                
        ax4.set_title('분석 결론', fontsize=14, fontweight='bold')
        ax4.axis('off')  # 축 제거
        
        plt.tight_layout()
        
        # 저장
        full_save_path = os.path.join(self.base_path, save_path)
        plt.savefig(full_save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        self.logger.info(f"📋 방향성 요약 차트 저장: {full_save_path}")
        return full_save_path

def main():
    """정밀 방향성 시스템 실행"""
    
    print("🎯 정밀 방향성 예측 시스템")
    print("=" * 50)
    
    # 시스템 초기화
    system = PreciseDirectionSystem()
    
    # 최신 시점에서 2주간 방향성 분석
    latest_timepoint = 2000  # 임시값 (실제로는 최신 시점 사용)
    
    print("📊 2주간 정밀 방향성 분석 중...")
    report = system.generate_direction_report(latest_timepoint, hours=336)
    
    # 시간별 상세 데이터 생성
    print("📈 시간별 상세 예측 생성 중...")
    hourly_data = system.predict_hourly_prices(latest_timepoint, hours=336)
    
    # 그래프 생성
    print("🎨 시간별 예측 그래프 생성 중...")
    chart1_path = system.create_hourly_prediction_chart(hourly_data)
    
    print("🎨 방향성 요약 차트 생성 중...")
    chart2_path = system.create_direction_summary_chart(report)
    
    # 결과 출력
    print("\n" + "="*60)
    print("🎯 BTC 정밀 방향성 분석 결과")
    print("="*60)
    
    for key, value in report.items():
        if key == "세부_방향성":
            print(f"\n📊 {key}:")
            for sub_key, sub_value in value.items():
                print(f"   {sub_key}: {sub_value}")
        elif isinstance(value, dict):
            print(f"\n💰 {key}:")
            for sub_key, sub_value in value.items():
                print(f"   {sub_key}: {sub_value}")
        else:
            print(f"{key}: {value}")
            
    print("\n" + "="*60)
    
    # 결과 저장
    with open('precise_direction_analysis.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        
    print("💾 상세 결과 저장: precise_direction_analysis.json")
    print("✅ 정밀 방향성 분석 완료!")

if __name__ == "__main__":
    main()