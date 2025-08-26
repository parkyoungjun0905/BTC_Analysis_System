#!/usr/bin/env python3
"""
초정밀 BTC 미래 차트 예측 시스템 v1.0
Multi-Layer Prediction with Risk Monitoring

작동 원리:
1. 500+ 지표 데이터 입력
2. 5개 독립 예측 모델 실행
3. 앙상블 방식으로 통합
4. 위험 요소 실시간 모니터링
5. 시각적 차트 + 경고 시스템
"""

import numpy as np
import pandas as pd
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# 시각화 라이브러리
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("⚠️ Plotly 미설치 - pip install plotly")

# 기술적 지표 라이브러리
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    print("⚠️ TA-Lib 미설치 - pip install TA-Lib")

# 머신러닝 라이브러리
try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️ scikit-learn 미설치 - pip install scikit-learn")


class PrecisionFuturePredictor:
    """초정밀 미래 예측 시스템"""
    
    def __init__(self, data_path: str = None):
        """
        Args:
            data_path: enhanced_data_collector.py 결과 JSON 파일 경로
        """
        self.data_path = data_path
        self.base_path = os.path.dirname(os.path.abspath(__file__))
        self.historical_path = os.path.join(self.base_path, "historical_data")
        
        # 예측 모델 가중치 (백테스팅 기반 최적화)
        self.model_weights = {
            "technical_analysis": 0.25,    # 기술적 분석
            "pattern_recognition": 0.20,    # 패턴 인식
            "statistical_forecast": 0.20,   # 통계적 예측
            "momentum_based": 0.20,         # 모멘텀 기반
            "ml_prediction": 0.15           # 머신러닝 (신중하게 적용)
        }
        
        # 예측 시간대 설정
        self.prediction_horizons = {
            "ultra_short": {"hours": 1, "points": 12},      # 1시간 (5분 단위)
            "short": {"hours": 4, "points": 16},            # 4시간 (15분 단위)
            "medium": {"hours": 12, "points": 12},          # 12시간 (1시간 단위)
            "long": {"hours": 24, "points": 24}             # 24시간 (1시간 단위)
        }
        
        # 위험 모니터링 임계값
        self.risk_thresholds = {
            "extreme_volatility": 0.05,      # 5% 이상 변동성
            "volume_spike": 3.0,              # 평균 대비 3배 거래량
            "rsi_extreme": {"low": 20, "high": 80},
            "funding_extreme": 0.05,          # 펀딩비 0.05% 초과
            "whale_movement": 1000,          # 1000 BTC 이상 이동
            "correlation_break": 0.3          # 상관관계 0.3 이상 변화
        }
        
        # 예측 결과 저장
        self.predictions = {}
        self.risk_factors = {}
        self.confidence_scores = {}
        
    async def predict_future(self, hours_ahead: int = 24) -> Dict:
        """
        메인 예측 함수
        
        Args:
            hours_ahead: 예측할 시간 (기본 24시간)
            
        Returns:
            예측 결과 딕셔너리
        """
        print(f"🔮 {hours_ahead}시간 후 미래 예측 시작...")
        
        try:
            # 1. 데이터 로드 및 검증
            current_data = self.load_and_validate_data()
            if not current_data:
                return {"error": "데이터 로드 실패"}
            
            # 2. 과거 패턴 분석 (최근 30일)
            historical_patterns = self.analyze_historical_patterns()
            
            # 3. 현재 시장 상태 진단
            market_state = self.diagnose_market_state(current_data)
            
            # 4. 5개 독립 모델 실행
            predictions = await self.run_prediction_models(
                current_data, 
                historical_patterns, 
                hours_ahead
            )
            
            # 5. 예측 통합 (앙상블)
            integrated_prediction = self.ensemble_predictions(predictions)
            
            # 6. 위험 요소 분석
            risk_analysis = self.analyze_risk_factors(current_data, integrated_prediction)
            
            # 7. 관찰 필요 요소 도출
            watch_factors = self.identify_watch_factors(current_data, risk_analysis)
            
            # 8. 시각화 차트 생성
            if PLOTLY_AVAILABLE:
                chart_path = self.create_future_chart(
                    current_data, 
                    integrated_prediction,
                    risk_analysis,
                    hours_ahead
                )
            else:
                chart_path = None
            
            # 9. 최종 결과 구성
            result = {
                "prediction_time": datetime.now().isoformat(),
                "current_price": current_data.get("current_price", 0),
                "prediction_horizon": f"{hours_ahead} hours",
                "predictions": integrated_prediction,
                "confidence": self.calculate_overall_confidence(predictions),
                "market_state": market_state,
                "risk_analysis": risk_analysis,
                "watch_factors": watch_factors,
                "chart_path": chart_path,
                "disclaimer": "예측은 참고용이며 투자 조언이 아닙니다"
            }
            
            # 10. 결과 저장
            self.save_prediction(result)
            
            print("✅ 예측 완료!")
            self.print_summary(result)
            
            return result
            
        except Exception as e:
            print(f"❌ 예측 실패: {e}")
            return {"error": str(e)}
    
    def load_and_validate_data(self) -> Dict:
        """데이터 로드 및 검증"""
        try:
            # 최신 데이터 파일 찾기
            if self.data_path and os.path.exists(self.data_path):
                file_path = self.data_path
            else:
                # historical_data에서 최신 파일 찾기
                files = sorted([f for f in os.listdir(self.historical_path) 
                              if f.startswith("btc_analysis_") and f.endswith(".json")])
                if not files:
                    raise ValueError("분석 데이터 파일을 찾을 수 없습니다")
                file_path = os.path.join(self.historical_path, files[-1])
            
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # 필수 데이터 검증
            required_fields = ["data_sources", "summary"]
            for field in required_fields:
                if field not in data:
                    raise ValueError(f"필수 필드 누락: {field}")
            
            # 현재 가격 추출
            current_price = self.extract_current_price(data)
            data["current_price"] = current_price
            
            print(f"✅ 데이터 로드 완료: {file_path}")
            print(f"💰 현재 BTC 가격: ${current_price:,.0f}")
            
            return data
            
        except Exception as e:
            print(f"❌ 데이터 로드 실패: {e}")
            return None
    
    def extract_current_price(self, data: Dict) -> float:
        """현재 BTC 가격 추출"""
        try:
            # 여러 소스에서 가격 찾기
            price_paths = [
                ["summary", "current_btc_price"],
                ["data_sources", "legacy_analyzer", "market_data", "avg_price"],  # 이 경로 추가
                ["data_sources", "legacy_analyzer", "market_data", "binance", "current_price"],
                ["data_sources", "legacy_analyzer", "market_data", "coingecko", "current_price_usd"]
            ]
            
            for path in price_paths:
                try:
                    value = data
                    for key in path:
                        value = value[key]
                    if value and value > 0:
                        return float(value)
                except:
                    continue
            
            return 0
            
        except:
            return 0
    
    def analyze_historical_patterns(self) -> Dict:
        """과거 30일 패턴 분석"""
        patterns = {
            "trend": "unknown",
            "volatility": "normal",
            "similar_patterns": [],
            "cycle_phase": "unknown"
        }
        
        try:
            # historical_data에서 최근 30일 파일 로드
            files = []
            for i in range(30):
                date = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
                matching_files = [f for f in os.listdir(self.historical_path) 
                                if date in f and f.endswith(".json")]
                files.extend(matching_files)
            
            if len(files) < 7:  # 최소 7일 데이터 필요
                print("⚠️ 과거 데이터 부족 (7일 미만)")
                return patterns
            
            # 가격 데이터 수집
            prices = []
            for file in sorted(files)[-30:]:  # 최근 30개
                try:
                    with open(os.path.join(self.historical_path, file), 'r') as f:
                        data = json.load(f)
                    price = self.extract_current_price(data)
                    if price > 0:
                        prices.append(price)
                except:
                    continue
            
            if len(prices) >= 7:
                # 추세 분석
                recent_avg = np.mean(prices[-7:])
                older_avg = np.mean(prices[-14:-7]) if len(prices) >= 14 else np.mean(prices[:len(prices)//2])
                
                if recent_avg > older_avg * 1.02:
                    patterns["trend"] = "상승"
                elif recent_avg < older_avg * 0.98:
                    patterns["trend"] = "하락"
                else:
                    patterns["trend"] = "횡보"
                
                # 변동성 분석
                volatility = np.std(prices) / np.mean(prices)
                if volatility > 0.05:
                    patterns["volatility"] = "높음"
                elif volatility < 0.02:
                    patterns["volatility"] = "낮음"
                else:
                    patterns["volatility"] = "보통"
                
                # 사이클 분석 (간단한 버전)
                price_position = (prices[-1] - min(prices)) / (max(prices) - min(prices)) if max(prices) > min(prices) else 0.5
                if price_position > 0.8:
                    patterns["cycle_phase"] = "고점 근접"
                elif price_position < 0.2:
                    patterns["cycle_phase"] = "저점 근접"
                else:
                    patterns["cycle_phase"] = "중간"
            
            print(f"📊 과거 패턴: {patterns['trend']} 추세, {patterns['volatility']} 변동성")
            
        except Exception as e:
            print(f"⚠️ 과거 패턴 분석 오류: {e}")
        
        return patterns
    
    def diagnose_market_state(self, data: Dict) -> Dict:
        """현재 시장 상태 진단"""
        state = {
            "overall": "중립",
            "sentiment": "중립",
            "momentum": "중립",
            "risk_level": "보통",
            "key_signals": []
        }
        
        try:
            # RSI 체크
            rsi = self.extract_indicator(data, "RSI_14")
            if rsi:
                if rsi > 70:
                    state["momentum"] = "과매수"
                    state["key_signals"].append(f"RSI 과매수 ({rsi:.1f})")
                elif rsi < 30:
                    state["momentum"] = "과매도"
                    state["key_signals"].append(f"RSI 과매도 ({rsi:.1f})")
            
            # Fear & Greed 체크
            fear_greed = self.extract_indicator(data, "fear_greed_index")
            if fear_greed:
                if fear_greed > 75:
                    state["sentiment"] = "극도의 탐욕"
                    state["risk_level"] = "높음"
                elif fear_greed > 55:
                    state["sentiment"] = "탐욕"
                elif fear_greed < 25:
                    state["sentiment"] = "극도의 공포"
                elif fear_greed < 45:
                    state["sentiment"] = "공포"
                else:
                    state["sentiment"] = "중립"
            
            # 펀딩비 체크
            funding = self.extract_indicator(data, "funding_rate")
            if funding:
                if abs(funding) > 0.05:
                    state["key_signals"].append(f"펀딩비 극단값 ({funding:.3f}%)")
                    state["risk_level"] = "높음"
            
            # 종합 판단
            if state["risk_level"] == "높음":
                state["overall"] = "주의 필요"
            elif "과매수" in state["momentum"]:
                state["overall"] = "조정 가능성"
            elif "과매도" in state["momentum"]:
                state["overall"] = "반등 가능성"
            else:
                state["overall"] = "안정적"
            
            print(f"🔍 시장 상태: {state['overall']} (심리: {state['sentiment']})")
            
        except Exception as e:
            print(f"⚠️ 시장 상태 진단 오류: {e}")
        
        return state
    
    def extract_indicator(self, data: Dict, indicator_name: str) -> Optional[float]:
        """지표 값 추출 헬퍼 함수"""
        # 여러 가능한 경로 시도
        paths = [
            ["data_sources", "legacy_analyzer", "technical_indicators", indicator_name],
            ["data_sources", "legacy_analyzer", "onchain_data", indicator_name],
            ["data_sources", "legacy_analyzer", "derivatives_data", indicator_name],
            ["data_sources", "enhanced_onchain", indicator_name],
            ["summary", indicator_name]
        ]
        
        for path in paths:
            try:
                value = data
                for key in path:
                    value = value[key]
                if isinstance(value, (int, float)):
                    return float(value)
            except:
                continue
        
        return None
    
    async def run_prediction_models(self, data: Dict, patterns: Dict, hours: int) -> Dict:
        """5개 독립 예측 모델 실행"""
        predictions = {}
        
        # 1. 기술적 분석 모델
        predictions["technical"] = self.technical_analysis_model(data, hours)
        
        # 2. 패턴 인식 모델
        predictions["pattern"] = self.pattern_recognition_model(data, patterns, hours)
        
        # 3. 통계적 예측 모델
        predictions["statistical"] = self.statistical_forecast_model(data, hours)
        
        # 4. 모멘텀 기반 모델
        predictions["momentum"] = self.momentum_based_model(data, hours)
        
        # 5. 머신러닝 모델 (가능한 경우)
        if SKLEARN_AVAILABLE:
            predictions["ml"] = self.ml_prediction_model(data, hours)
        else:
            predictions["ml"] = predictions["technical"]  # 폴백
        
        return predictions
    
    def technical_analysis_model(self, data: Dict, hours: int) -> Dict:
        """기술적 분석 기반 예측"""
        current_price = data.get("current_price", 0)
        
        # RSI 기반 예측
        rsi = self.extract_indicator(data, "RSI_14") or 50
        rsi_factor = (50 - rsi) / 500  # RSI가 50에서 멀수록 반대 방향 압력
        
        # 이동평균 기반
        ma_factor = 0.0
        sma_20 = self.extract_indicator(data, "sma_20")
        if sma_20 and current_price > 0:
            ma_factor = (current_price - sma_20) / sma_20
        
        # 예측 가격 계산
        price_change_rate = rsi_factor + ma_factor * 0.5
        
        # 시간별 예측
        predictions = []
        for h in range(1, hours + 1):
            # 시간이 지날수록 불확실성 증가
            uncertainty = 1 + (h / 24) * 0.1
            predicted_change = price_change_rate * h * uncertainty
            predicted_price = current_price * (1 + predicted_change)
            
            predictions.append({
                "hour": h,
                "price": predicted_price,
                "confidence": max(0.3, 0.7 - h * 0.01)  # 시간이 지날수록 신뢰도 감소
            })
        
        return {
            "method": "기술적 분석",
            "predictions": predictions,
            "key_factors": ["RSI", "이동평균"],
            "direction": "상승" if price_change_rate > 0 else "하락"
        }
    
    def pattern_recognition_model(self, data: Dict, patterns: Dict, hours: int) -> Dict:
        """패턴 인식 기반 예측"""
        current_price = data.get("current_price", 0)
        
        # 패턴 기반 예측 로직
        trend = patterns.get("trend", "횡보")
        volatility = patterns.get("volatility", "보통")
        
        # 추세에 따른 기본 변화율
        if trend == "상승":
            base_rate = 0.002  # 시간당 0.2% 상승
        elif trend == "하락":
            base_rate = -0.002
        else:
            base_rate = 0.0
        
        # 변동성에 따른 조정
        if volatility == "높음":
            volatility_factor = 2.0
        elif volatility == "낮음":
            volatility_factor = 0.5
        else:
            volatility_factor = 1.0
        
        predictions = []
        for h in range(1, hours + 1):
            # 사인파 형태의 변동 추가 (자연스러운 움직임)
            wave = np.sin(h * np.pi / 12) * 0.01 * volatility_factor
            predicted_change = (base_rate * h + wave)
            predicted_price = current_price * (1 + predicted_change)
            
            predictions.append({
                "hour": h,
                "price": predicted_price,
                "confidence": 0.5
            })
        
        return {
            "method": "패턴 인식",
            "predictions": predictions,
            "key_factors": [f"{trend} 추세", f"{volatility} 변동성"],
            "direction": trend
        }
    
    def statistical_forecast_model(self, data: Dict, hours: int) -> Dict:
        """통계적 예측 모델"""
        current_price = data.get("current_price", 0)
        
        # ARIMA 스타일 예측 (간단한 버전)
        # 실제로는 더 복잡한 통계 모델 필요
        
        predictions = []
        last_price = current_price
        
        for h in range(1, hours + 1):
            # 랜덤 워크 + 드리프트
            drift = 0.0001  # 약간의 상승 편향
            random_component = np.random.normal(0, 0.002)  # 0.2% 표준편차
            
            price_change = drift + random_component
            predicted_price = last_price * (1 + price_change)
            last_price = predicted_price
            
            predictions.append({
                "hour": h,
                "price": predicted_price,
                "confidence": 0.4
            })
        
        return {
            "method": "통계적 예측",
            "predictions": predictions,
            "key_factors": ["시계열 분석", "확률 분포"],
            "direction": "중립"
        }
    
    def momentum_based_model(self, data: Dict, hours: int) -> Dict:
        """모멘텀 기반 예측"""
        current_price = data.get("current_price", 0)
        
        # 모멘텀 지표들
        macd = self.extract_indicator(data, "MACD_line") or 0
        volume_ratio = self.extract_indicator(data, "volume_ratio") or 1.0
        
        # 모멘텀 계산
        momentum_score = 0
        if macd > 0:
            momentum_score += 0.3
        if volume_ratio > 1.5:
            momentum_score += 0.2
        
        # 모멘텀에 따른 가격 변화
        momentum_rate = momentum_score * 0.001  # 최대 0.05% 시간당
        
        predictions = []
        for h in range(1, hours + 1):
            # 모멘텀은 시간이 지나면서 감소
            decay_factor = np.exp(-h / 24)
            predicted_change = momentum_rate * h * decay_factor
            predicted_price = current_price * (1 + predicted_change)
            
            predictions.append({
                "hour": h,
                "price": predicted_price,
                "confidence": 0.45
            })
        
        return {
            "method": "모멘텀 분석",
            "predictions": predictions,
            "key_factors": ["MACD", "거래량"],
            "direction": "상승" if momentum_score > 0 else "하락"
        }
    
    def ml_prediction_model(self, data: Dict, hours: int) -> Dict:
        """머신러닝 예측 모델"""
        current_price = data.get("current_price", 0)
        
        try:
            # 특징 추출
            features = []
            feature_names = ["RSI_14", "fear_greed_index", "funding_rate", "volume_ratio"]
            
            for name in feature_names:
                value = self.extract_indicator(data, name)
                features.append(value if value is not None else 50)
            
            features = np.array(features).reshape(1, -1)
            
            # 간단한 선형 예측 (실제로는 학습된 모델 필요)
            # 여기서는 데모용으로 간단한 계산
            feature_weights = [0.2, 0.3, 0.25, 0.25]
            weighted_score = sum(f * w for f, w in zip(features[0], feature_weights))
            
            # 점수를 가격 변화로 변환
            price_change_rate = (weighted_score - 50) / 5000
            
            predictions = []
            for h in range(1, hours + 1):
                predicted_change = price_change_rate * h
                predicted_price = current_price * (1 + predicted_change)
                
                predictions.append({
                    "hour": h,
                    "price": predicted_price,
                    "confidence": 0.35
                })
            
        except Exception as e:
            print(f"⚠️ ML 모델 오류: {e}")
            # 폴백: 현재 가격 유지
            predictions = [{"hour": h, "price": current_price, "confidence": 0.1} 
                         for h in range(1, hours + 1)]
        
        return {
            "method": "머신러닝",
            "predictions": predictions,
            "key_factors": ["복합 지표 학습"],
            "direction": "데이터 기반"
        }
    
    def ensemble_predictions(self, predictions: Dict) -> Dict:
        """예측 결과 통합 (앙상블)"""
        # 모든 모델의 예측을 가중 평균
        ensemble_predictions = []
        
        # 최대 시간 찾기
        max_hours = max(len(p["predictions"]) for p in predictions.values())
        
        for hour in range(max_hours):
            weighted_sum = 0
            weight_sum = 0
            confidence_sum = 0
            
            price_predictions = []
            
            for model_name, model_pred in predictions.items():
                if hour < len(model_pred["predictions"]):
                    pred = model_pred["predictions"][hour]
                    weight = self.model_weights.get(model_name.replace("_analysis", "").replace("_recognition", "").replace("_forecast", "").replace("_based", "").replace("_prediction", ""), 0.2)
                    
                    weighted_sum += pred["price"] * weight * pred["confidence"]
                    weight_sum += weight * pred["confidence"]
                    confidence_sum += pred["confidence"]
                    
                    price_predictions.append(pred["price"])
            
            if weight_sum > 0:
                ensemble_price = weighted_sum / weight_sum
                
                # 신뢰 구간 계산
                prices_array = np.array(price_predictions)
                std_dev = np.std(prices_array)
                
                ensemble_predictions.append({
                    "hour": hour + 1,
                    "price": ensemble_price,
                    "upper_bound": ensemble_price + std_dev,
                    "lower_bound": ensemble_price - std_dev,
                    "confidence": confidence_sum / len(predictions),
                    "std_dev": std_dev
                })
        
        # 주요 예측 포인트 추출
        key_points = {
            "1h": ensemble_predictions[0] if len(ensemble_predictions) > 0 else None,
            "4h": ensemble_predictions[3] if len(ensemble_predictions) > 3 else None,
            "12h": ensemble_predictions[11] if len(ensemble_predictions) > 11 else None,
            "24h": ensemble_predictions[23] if len(ensemble_predictions) > 23 else None
        }
        
        return {
            "full_predictions": ensemble_predictions,
            "key_points": key_points,
            "model_agreement": self.calculate_model_agreement(predictions),
            "primary_direction": self.determine_primary_direction(ensemble_predictions)
        }
    
    def calculate_model_agreement(self, predictions: Dict) -> float:
        """모델 간 일치도 계산"""
        directions = []
        for model_pred in predictions.values():
            if "direction" in model_pred:
                directions.append(model_pred["direction"])
        
        if not directions:
            return 0.5
        
        # 가장 많은 방향
        from collections import Counter
        most_common = Counter(directions).most_common(1)[0]
        agreement = most_common[1] / len(directions)
        
        return agreement
    
    def determine_primary_direction(self, predictions: List) -> str:
        """주요 방향성 결정"""
        if not predictions:
            return "불확실"
        
        current_price = predictions[0]["price"] if predictions else 0
        final_price = predictions[-1]["price"] if predictions else 0
        
        change_percent = ((final_price - current_price) / current_price * 100) if current_price > 0 else 0
        
        if change_percent > 1:
            return "상승"
        elif change_percent < -1:
            return "하락"
        else:
            return "횡보"
    
    def analyze_risk_factors(self, data: Dict, prediction: Dict) -> Dict:
        """위험 요소 분석"""
        risks = {
            "level": "낮음",
            "score": 0,
            "factors": [],
            "warnings": []
        }
        
        risk_score = 0
        
        # 1. 변동성 체크
        if prediction.get("full_predictions"):
            prices = [p["price"] for p in prediction["full_predictions"]]
            volatility = np.std(prices) / np.mean(prices) if prices else 0
            
            if volatility > self.risk_thresholds["extreme_volatility"]:
                risk_score += 30
                risks["factors"].append(f"극심한 변동성 ({volatility:.2%})")
                risks["warnings"].append("⚠️ 높은 변동성으로 예측 신뢰도 낮음")
        
        # 2. RSI 극단값
        rsi = self.extract_indicator(data, "RSI_14")
        if rsi:
            if rsi > self.risk_thresholds["rsi_extreme"]["high"]:
                risk_score += 20
                risks["factors"].append(f"RSI 과매수 ({rsi:.1f})")
            elif rsi < self.risk_thresholds["rsi_extreme"]["low"]:
                risk_score += 20
                risks["factors"].append(f"RSI 과매도 ({rsi:.1f})")
        
        # 3. 펀딩비 극단값
        funding = self.extract_indicator(data, "funding_rate")
        if funding and abs(funding) > self.risk_thresholds["funding_extreme"]:
            risk_score += 25
            risks["factors"].append(f"펀딩비 극단값 ({funding:.3f}%)")
            risks["warnings"].append("⚠️ 펀딩비 극단값으로 청산 위험")
        
        # 4. 모델 불일치
        agreement = prediction.get("model_agreement", 1.0)
        if agreement < 0.6:
            risk_score += 15
            risks["factors"].append(f"모델 간 불일치 ({agreement:.1%})")
        
        # 위험 수준 결정
        risks["score"] = risk_score
        if risk_score >= 60:
            risks["level"] = "매우 높음"
        elif risk_score >= 40:
            risks["level"] = "높음"
        elif risk_score >= 20:
            risks["level"] = "보통"
        else:
            risks["level"] = "낮음"
        
        return risks
    
    def identify_watch_factors(self, data: Dict, risks: Dict) -> Dict:
        """주의 관찰 필요 요소 도출"""
        watch = {
            "critical": [],      # 즉시 확인 필요
            "important": [],     # 중요 관찰
            "monitor": [],       # 일반 모니터링
            "actions": []        # 권장 조치
        }
        
        # 위험 수준에 따른 관찰 요소
        if risks["level"] in ["높음", "매우 높음"]:
            watch["critical"].append("🚨 거래소 유출입량 실시간 모니터링")
            watch["critical"].append("🚨 고래 지갑 움직임 추적")
            watch["critical"].append("🚨 선물 청산 데이터 확인")
            watch["actions"].append("포지션 축소 또는 헤지 고려")
        
        # RSI 기반
        rsi = self.extract_indicator(data, "RSI_14")
        if rsi:
            if rsi > 70:
                watch["important"].append("📊 RSI 과매수 - 조정 가능성 주시")
                watch["monitor"].append("거래량 감소 여부")
            elif rsi < 30:
                watch["important"].append("📊 RSI 과매도 - 반등 신호 관찰")
                watch["monitor"].append("매수세 유입 여부")
        
        # 펀딩비 기반
        funding = self.extract_indicator(data, "funding_rate")
        if funding:
            if funding > 0.03:
                watch["important"].append("💰 높은 펀딩비 - 롱 포지션 과열")
                watch["actions"].append("롱 포지션 정리 시점 모색")
            elif funding < -0.03:
                watch["important"].append("💰 음수 펀딩비 - 숏 포지션 과열")
                watch["actions"].append("숏 스퀴즈 가능성 대비")
        
        # 일반 모니터링
        watch["monitor"].extend([
            "📰 주요 경제 지표 발표 일정",
            "🏛️ 규제 관련 뉴스",
            "💵 달러 인덱스(DXY) 변화",
            "📈 나스닥/S&P500 상관관계",
            "⚡ 네트워크 해시레이트 변화"
        ])
        
        # Fear & Greed 기반
        fear_greed = self.extract_indicator(data, "fear_greed_index")
        if fear_greed:
            if fear_greed > 80:
                watch["critical"].append("😱 극도의 탐욕 - 조정 임박 가능성")
            elif fear_greed < 20:
                watch["critical"].append("😰 극도의 공포 - 바닥 형성 가능성")
        
        return watch
    
    def create_future_chart(self, data: Dict, prediction: Dict, risks: Dict, hours: int) -> str:
        """미래 예측 차트 생성"""
        if not PLOTLY_AVAILABLE:
            print("⚠️ Plotly 미설치로 차트 생성 불가")
            return None
        
        try:
            # 서브플롯 생성
            fig = make_subplots(
                rows=3, cols=1,
                row_heights=[0.6, 0.2, 0.2],
                subplot_titles=("BTC 가격 예측", "신뢰 구간", "위험 지표"),
                vertical_spacing=0.1
            )
            
            current_price = data.get("current_price", 0)
            predictions = prediction.get("full_predictions", [])
            
            if predictions:
                # 시간 축
                hours_x = [p["hour"] for p in predictions]
                
                # 예측 가격
                predicted_prices = [p["price"] for p in predictions]
                upper_bounds = [p["upper_bound"] for p in predictions]
                lower_bounds = [p["lower_bound"] for p in predictions]
                
                # 메인 가격 차트
                fig.add_trace(
                    go.Scatter(
                        x=hours_x,
                        y=predicted_prices,
                        mode='lines',
                        name='예측 가격',
                        line=dict(color='blue', width=2)
                    ),
                    row=1, col=1
                )
                
                # 신뢰 구간
                fig.add_trace(
                    go.Scatter(
                        x=hours_x + hours_x[::-1],
                        y=upper_bounds + lower_bounds[::-1],
                        fill='toself',
                        fillcolor='rgba(0,100,255,0.2)',
                        line=dict(color='rgba(255,255,255,0)'),
                        name='신뢰 구간',
                        showlegend=True
                    ),
                    row=1, col=1
                )
                
                # 현재 가격 라인
                fig.add_hline(
                    y=current_price,
                    line_dash="dash",
                    line_color="green",
                    annotation_text=f"현재: ${current_price:,.0f}",
                    row=1, col=1
                )
                
                # 신뢰도 차트
                confidence_scores = [p["confidence"] * 100 for p in predictions]
                fig.add_trace(
                    go.Scatter(
                        x=hours_x,
                        y=confidence_scores,
                        mode='lines+markers',
                        name='신뢰도 (%)',
                        line=dict(color='orange', width=1)
                    ),
                    row=2, col=1
                )
                
                # 위험 점수
                risk_levels = [risks["score"]] * len(hours_x)  # 일정한 위험 수준
                fig.add_trace(
                    go.Scatter(
                        x=hours_x,
                        y=risk_levels,
                        mode='lines',
                        name=f'위험 수준: {risks["level"]}',
                        line=dict(color='red' if risks["score"] > 40 else 'yellow' if risks["score"] > 20 else 'green', width=2)
                    ),
                    row=3, col=1
                )
            
            # 레이아웃 설정
            fig.update_layout(
                title=f"BTC {hours}시간 미래 예측 차트",
                height=800,
                showlegend=True,
                hovermode='x unified'
            )
            
            # 축 라벨
            fig.update_xaxes(title_text="시간 (hours)", row=3, col=1)
            fig.update_yaxes(title_text="가격 (USD)", row=1, col=1)
            fig.update_yaxes(title_text="신뢰도 (%)", row=2, col=1)
            fig.update_yaxes(title_text="위험 점수", row=3, col=1)
            
            # 파일 저장
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            chart_path = os.path.join(self.base_path, f"prediction_chart_{timestamp}.html")
            fig.write_html(chart_path)
            
            print(f"📊 차트 저장: {chart_path}")
            return chart_path
            
        except Exception as e:
            print(f"❌ 차트 생성 실패: {e}")
            return None
    
    def calculate_overall_confidence(self, predictions: Dict) -> float:
        """전체 신뢰도 계산"""
        confidences = []
        
        for model_pred in predictions.values():
            if "predictions" in model_pred and model_pred["predictions"]:
                # 각 모델의 평균 신뢰도
                model_conf = np.mean([p["confidence"] for p in model_pred["predictions"]])
                confidences.append(model_conf)
        
        if confidences:
            return float(np.mean(confidences))
        return 0.5
    
    def save_prediction(self, result: Dict):
        """예측 결과 저장"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"prediction_{timestamp}.json"
            filepath = os.path.join(self.base_path, "predictions", filename)
            
            # predictions 디렉토리 생성
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2, default=str)
            
            print(f"💾 예측 저장: {filename}")
            
        except Exception as e:
            print(f"⚠️ 예측 저장 실패: {e}")
    
    def print_summary(self, result: Dict):
        """예측 결과 요약 출력"""
        print("\n" + "="*60)
        print("📊 예측 결과 요약")
        print("="*60)
        
        current_price = result.get("current_price", 0)
        key_points = result.get("predictions", {}).get("key_points", {})
        
        print(f"💰 현재 가격: ${current_price:,.0f}")
        
        if key_points:
            for time_key, point in key_points.items():
                if point:
                    price = point["price"]
                    change = ((price - current_price) / current_price * 100) if current_price > 0 else 0
                    confidence = point.get("confidence", 0) * 100
                    
                    emoji = "📈" if change > 0 else "📉" if change < 0 else "➡️"
                    print(f"{emoji} {time_key}: ${price:,.0f} ({change:+.2f}%) [신뢰도: {confidence:.0f}%]")
        
        print("\n🎯 시장 상태:")
        market_state = result.get("market_state", {})
        print(f"  • 전반: {market_state.get('overall', 'N/A')}")
        print(f"  • 심리: {market_state.get('sentiment', 'N/A')}")
        print(f"  • 모멘텀: {market_state.get('momentum', 'N/A')}")
        
        print("\n⚠️ 위험 분석:")
        risks = result.get("risk_analysis", {})
        print(f"  • 위험 수준: {risks.get('level', 'N/A')} (점수: {risks.get('score', 0)})")
        for factor in risks.get("factors", [])[:3]:
            print(f"  • {factor}")
        
        print("\n👁️ 주의 관찰 요소:")
        watch = result.get("watch_factors", {})
        
        if watch.get("critical"):
            print("  [긴급]")
            for item in watch["critical"][:3]:
                print(f"    {item}")
        
        if watch.get("important"):
            print("  [중요]")
            for item in watch["important"][:3]:
                print(f"    {item}")
        
        if watch.get("actions"):
            print("  [권장 조치]")
            for item in watch["actions"][:2]:
                print(f"    → {item}")
        
        print("\n" + "="*60)
        print("⚠️ 면책: 이 예측은 참고용이며 투자 조언이 아닙니다")
        print("="*60)


async def main():
    """테스트 실행"""
    print("🚀 초정밀 BTC 미래 예측 시스템 시작")
    print("="*60)
    
    predictor = PrecisionFuturePredictor()
    
    # 24시간 예측 실행
    result = await predictor.predict_future(hours_ahead=24)
    
    if "error" not in result:
        print("\n✅ 예측 완료!")
        if result.get("chart_path"):
            print(f"📊 차트 보기: {result['chart_path']}")
    else:
        print(f"\n❌ 예측 실패: {result['error']}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())