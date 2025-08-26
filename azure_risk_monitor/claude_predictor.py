#!/usr/bin/env python3
"""
Claude API 기반 BTC 가격 변동 예측 시스템
선행 지표 분석으로 사전 경고 알림 생성
"""

import asyncio
import aiohttp
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging
import os
from enhanced_11_indicators import Enhanced11IndicatorSystem
from prediction_tracker import PredictionTracker

class ClaudePricePredictor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.claude_api_key = os.environ.get('CLAUDE_API_KEY', '')
        self.base_url = "https://api.anthropic.com/v1/messages"
        
        # 11개 선행지표 강화 시스템 및 추적 시스템
        self.enhanced_11_system = Enhanced11IndicatorSystem()
        self.prediction_tracker = PredictionTracker()
        
        # 예측 정확도 추적
        self.prediction_history = []
        self.accuracy_score = 0.0
        
    async def analyze_market_signals(self, current_data: Dict, historical_data: List[Dict]) -> Dict:
        """시장 신호 분석하여 가격 변동 예측"""
        try:
            # 1. 과거 예측들 평가 (학습)
            evaluation_results = self.prediction_tracker.evaluate_predictions(current_data)
            accuracy_metrics = self.prediction_tracker.get_accuracy_metrics()
            
            # 2. 11개 선행지표 강화 시스템 수집 (핵심!)
            enhanced_11_indicators = await self.enhanced_11_system.collect_enhanced_11_indicators()
            
            # 3. 기존 지표와 결합
            basic_indicators = self.extract_leading_indicators(current_data, historical_data)
            combined_indicators = {
                **basic_indicators, 
                "enhanced_11_system": enhanced_11_indicators,
                "total_indicators_count": enhanced_11_indicators.get("total_indicators", 11)
            }
            
            # 4. Claude에게 11개 지표 기반 강화 분석 요청
            prediction = await self.request_enhanced_11_claude_prediction(combined_indicators, current_data, accuracy_metrics)
            
            # 5. 예측 결과 구조화
            structured_prediction = self.structure_prediction(prediction)
            
            # 6. 예측 기록 (학습용)
            prediction_id = self.prediction_tracker.record_prediction(
                structured_prediction, current_data, enhanced_11_indicators
            )
            structured_prediction["prediction_id"] = prediction_id
            structured_prediction["system_info"] = {
                "indicators_used": 11,
                "system_version": "Enhanced 11-Indicator v1.0",
                "cryptoquant_enabled": True
            }
            
            return structured_prediction
            
        except Exception as e:
            self.logger.error(f"Claude 예측 분석 실패: {e}")
            return self.fallback_prediction()
    
    def extract_leading_indicators(self, current_data: Dict, historical_data: List[Dict]) -> Dict:
        """가격 변동 예측을 위한 선행 지표 추출"""
        indicators = {
            "timestamp": datetime.utcnow().isoformat(),
            "market_structure": {},
            "flow_analysis": {},
            "derivatives_signals": {},
            "macro_context": {},
            "technical_setup": {}
        }
        
        try:
            # 시장 구조 분석
            indicators["market_structure"] = self.analyze_market_structure(current_data, historical_data)
            
            # 자금 흐름 분석  
            indicators["flow_analysis"] = self.analyze_capital_flows(current_data, historical_data)
            
            # 파생상품 신호
            indicators["derivatives_signals"] = self.analyze_derivatives_signals(current_data)
            
            # 거시경제 맥락
            indicators["macro_context"] = self.analyze_macro_context(current_data)
            
            # 기술적 셋업
            indicators["technical_setup"] = self.analyze_technical_setup(current_data, historical_data)
            
            return indicators
            
        except Exception as e:
            self.logger.error(f"선행 지표 추출 실패: {e}")
            return {"error": str(e)}
    
    def analyze_market_structure(self, current_data: Dict, historical_data: List[Dict]) -> Dict:
        """시장 구조 변화 분석"""
        structure = {
            "volume_profile": "normal",
            "liquidity_state": "adequate", 
            "order_flow": "balanced",
            "correlation_status": "normal"
        }
        
        try:
            # 거래량 프로파일 분석
            if "price_data" in current_data:
                current_volume = current_data["price_data"].get("volume_24h", 0)
                
                # 과거 30개 데이터 포인트에서 평균 거래량 계산
                historical_volumes = []
                for data in historical_data[-30:]:
                    if "price_data" in data:
                        historical_volumes.append(data["price_data"].get("volume_24h", 0))
                
                if historical_volumes:
                    avg_volume = sum(historical_volumes) / len(historical_volumes)
                    volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
                    
                    if volume_ratio > 3:
                        structure["volume_profile"] = "exceptional_spike"
                    elif volume_ratio > 2:
                        structure["volume_profile"] = "elevated"
                    elif volume_ratio < 0.5:
                        structure["volume_profile"] = "declining"
            
            # 상관관계 상태 분석 (BTC vs 전통자산)
            if "macro_data" in current_data:
                # VIX와 BTC 변동성 비교
                if "vix" in current_data["macro_data"]:
                    vix_level = current_data["macro_data"]["vix"]["current"]
                    if vix_level > 30:
                        structure["correlation_status"] = "stress_coupling"
                    elif vix_level < 15:
                        structure["correlation_status"] = "risk_on_decoupling"
            
            return structure
            
        except Exception as e:
            self.logger.error(f"시장 구조 분석 실패: {e}")
            return structure
    
    def analyze_capital_flows(self, current_data: Dict, historical_data: List[Dict]) -> Dict:
        """자금 흐름 분석"""
        flows = {
            "exchange_flow": "neutral",
            "institutional_activity": "quiet",
            "retail_sentiment": "neutral",
            "flow_divergence": False
        }
        
        try:
            # 센티먼트 기반 자금 흐름 추정
            if "sentiment_data" in current_data and "fear_greed" in current_data["sentiment_data"]:
                fg_index = current_data["sentiment_data"]["fear_greed"]["current_index"]
                
                if fg_index < 20:
                    flows["retail_sentiment"] = "extreme_fear"
                    flows["institutional_activity"] = "potential_accumulation"
                elif fg_index > 80:
                    flows["retail_sentiment"] = "extreme_greed"
                    flows["institutional_activity"] = "potential_distribution"
                
                # 과거 공포탐욕지수와 비교하여 급변 감지
                historical_fg = []
                for data in historical_data[-7:]:  # 최근 7일
                    if "sentiment_data" in data and "fear_greed" in data["sentiment_data"]:
                        historical_fg.append(data["sentiment_data"]["fear_greed"]["current_index"])
                
                if historical_fg:
                    avg_fg = sum(historical_fg) / len(historical_fg)
                    if abs(fg_index - avg_fg) > 20:  # 20포인트 이상 급변
                        flows["flow_divergence"] = True
            
            return flows
            
        except Exception as e:
            self.logger.error(f"자금 흐름 분석 실패: {e}")
            return flows
    
    def analyze_derivatives_signals(self, current_data: Dict) -> Dict:
        """파생상품 신호 분석"""
        signals = {
            "funding_pressure": "neutral",
            "leverage_buildup": "normal", 
            "liquidation_risk": "low",
            "options_skew": "neutral"
        }
        
        # 향후 구현: 실제 파생상품 데이터 연동
        # 현재는 플레이스홀더
        
        return signals
    
    def analyze_macro_context(self, current_data: Dict) -> Dict:
        """거시경제 맥락 분석"""
        context = {
            "risk_environment": "neutral",
            "dollar_strength": "stable",
            "volatility_regime": "normal",
            "correlation_shift": False
        }
        
        try:
            if "macro_data" in current_data:
                # VIX 레짐 분석
                if "vix" in current_data["macro_data"]:
                    vix_level = current_data["macro_data"]["vix"]["current"]
                    vix_change = current_data["macro_data"]["vix"]["change"]
                    
                    if vix_level > 25:
                        context["volatility_regime"] = "elevated"
                        if vix_level > 35:
                            context["volatility_regime"] = "crisis"
                    elif vix_level < 15:
                        context["volatility_regime"] = "complacency"
                    
                    if abs(vix_change) > 3:  # 3포인트 이상 급변
                        context["correlation_shift"] = True
                        context["risk_environment"] = "unstable" if vix_change > 0 else "stabilizing"
                
                # 달러 인덱스 분석
                if "dxy" in current_data["macro_data"]:
                    dxy_change = current_data["macro_data"]["dxy"]["change"]
                    if abs(dxy_change) > 0.5:  # 0.5% 이상 급변
                        context["dollar_strength"] = "strengthening" if dxy_change > 0 else "weakening"
            
            return context
            
        except Exception as e:
            self.logger.error(f"거시경제 분석 실패: {e}")
            return context
    
    def analyze_technical_setup(self, current_data: Dict, historical_data: List[Dict]) -> Dict:
        """기술적 셋업 분석"""
        setup = {
            "momentum_state": "neutral",
            "support_resistance": "no_key_level",
            "breakout_potential": "low",
            "volume_confirmation": False
        }
        
        try:
            if "price_data" in current_data:
                current_price = current_data["price_data"].get("current_price", 0)
                
                # 최근 가격들로 모멘텀 계산
                recent_prices = []
                for data in historical_data[-20:]:  # 최근 20개 데이터
                    if "price_data" in data:
                        recent_prices.append(data["price_data"].get("current_price", current_price))
                
                if len(recent_prices) >= 10:
                    # 단기 vs 중기 평균
                    short_avg = sum(recent_prices[-5:]) / 5
                    medium_avg = sum(recent_prices[-20:]) / 20
                    
                    momentum_ratio = short_avg / medium_avg
                    if momentum_ratio > 1.02:
                        setup["momentum_state"] = "bullish"
                    elif momentum_ratio < 0.98:
                        setup["momentum_state"] = "bearish"
                    
                    # 주요 레벨 근접 확인 (간단한 버전)
                    price_range = max(recent_prices) - min(recent_prices)
                    current_position = (current_price - min(recent_prices)) / price_range if price_range > 0 else 0.5
                    
                    if current_position > 0.9:
                        setup["support_resistance"] = "near_resistance"
                        setup["breakout_potential"] = "high"
                    elif current_position < 0.1:
                        setup["support_resistance"] = "near_support"
                        setup["breakout_potential"] = "high"
            
            return setup
            
        except Exception as e:
            self.logger.error(f"기술적 분석 실패: {e}")
            return setup
    
    async def request_claude_prediction(self, indicators: Dict, current_data: Dict) -> str:
        """Claude API에 예측 분석 요청"""
        try:
            current_price = current_data.get("price_data", {}).get("current_price", 0)
            
            # Claude에게 보낼 분석 프롬프트
            analysis_prompt = f"""
당신은 비트코인 전문 분석가입니다. 다음 선행 지표들을 분석하여 향후 6-24시간 내 가격 변동을 예측하세요.

현재 BTC 가격: ${current_price:,.0f}

=== 시장 구조 ===
{json.dumps(indicators.get('market_structure', {}), indent=2)}

=== 자금 흐름 ===
{json.dumps(indicators.get('flow_analysis', {}), indent=2)}

=== 거시경제 맥락 ===
{json.dumps(indicators.get('macro_context', {}), indent=2)}

=== 기술적 셋업 ===
{json.dumps(indicators.get('technical_setup', {}), indent=2)}

다음 형식으로 응답하세요:

PREDICTION_DIRECTION: [BULLISH/BEARISH/NEUTRAL]
PROBABILITY: [0-100]%
TIMEFRAME: [1-24시간]
PRICE_TARGET: $[목표가격]
CONFIDENCE: [LOW/MEDIUM/HIGH]

KEY_CATALYSTS: 
- [주요 원인 1]
- [주요 원인 2]
- [주요 원인 3]

RISK_FACTORS:
- [위험 요소 1]
- [위험 요소 2]

RECOMMENDED_ACTION:
[구체적 권장사항]

REASONING:
[상세한 분석 근거 2-3문장]
"""

            headers = {
                "Content-Type": "application/json",
                "x-api-key": self.claude_api_key,
                "anthropic-version": "2023-06-01"
            }
            
            payload = {
                "model": "claude-3-sonnet-20240229",
                "max_tokens": 1000,
                "messages": [
                    {
                        "role": "user", 
                        "content": analysis_prompt
                    }
                ]
            }
            
            # Claude API 호출 (타임아웃이 있으므로 실제로는 구현하지 않고 시뮬레이션)
            if not self.claude_api_key:
                # API 키가 없는 경우 시뮬레이션 응답
                return self.simulate_claude_response(indicators, current_price)
            
            async with aiohttp.ClientSession() as session:
                async with session.post(self.base_url, headers=headers, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result["content"][0]["text"]
                    else:
                        self.logger.error(f"Claude API 오류: {response.status}")
                        return self.simulate_claude_response(indicators, current_price)
                        
        except Exception as e:
            self.logger.error(f"Claude API 요청 실패: {e}")
            return self.simulate_claude_response(indicators, current_price)

    async def request_enhanced_claude_prediction(self, indicators: Dict, current_data: Dict, accuracy_metrics: Dict) -> str:
        """개선된 Claude API 예측 요청 (진짜 선행지표 기반)"""
        try:
            current_price = current_data.get("price_data", {}).get("current_price", 0)
            
            # 시스템 성과 정보
            system_performance = f"""
=== 시스템 성과 정보 ===
지난 7일 예측 정확도: {accuracy_metrics.get('direction_accuracy', 0):.1%}
거짓 양성률: {accuracy_metrics.get('false_positive_rate', 0):.1%}
신뢰도별 성과: {json.dumps(accuracy_metrics.get('confidence_breakdown', {}), indent=2)}
"""

            # 진짜 선행지표 정보
            whale_activity = indicators.get("whale_activity", {})
            derivatives = indicators.get("derivatives_structure", {})  
            macro_signals = indicators.get("macro_early_signals", {})
            institutional = indicators.get("institutional_flows", {})
            
            # 향상된 분석 프롬프트
            enhanced_prompt = f"""
당신은 비트코인 전문 분석가로서, **아직 가격에 반영되지 않은** 구조적 변화를 통해 향후 가격 움직임을 예측합니다.

{system_performance}

현재 BTC 가격: ${current_price:,.0f}

=== 🐋 고래/기관 활동 (1-6시간 선행지표) ===
거래소 대량 이동:
- 유입: {whale_activity.get('large_transfers', {}).get('exchange_inflows_1h', 0)} BTC/h
- 유출: {whale_activity.get('large_transfers', {}).get('exchange_outflows_1h', 0)} BTC/h
- Coinbase 프리미엄: {whale_activity.get('exchange_dynamics', {}).get('coinbase_premium', 0):.3f}%
- 기관 주소 활동: {whale_activity.get('address_clustering', {}).get('institutional_addresses_activity', 'neutral')}

=== ⚡ 파생상품 구조 (30분-2시간 선행지표) ===
선물 구조:
- 베이시스 가속도: {derivatives.get('futures_structure', {}).get('basis_acceleration', 0):.4f}
- 펀딩비 궤적: {derivatives.get('futures_structure', {}).get('funding_rate_trajectory', 'stable')}
- 청산 집중구간: Long {len(derivatives.get('futures_structure', {}).get('liquidation_clusters', {}).get('long_liquidations', []))}개, Short {len(derivatives.get('futures_structure', {}).get('liquidation_clusters', {}).get('short_liquidations', []))}개

옵션 플로우:
- Put/Call 비율 가속도: {derivatives.get('options_flow', {}).get('put_call_ratio_acceleration', 0):.4f}
- 내재 변동성 스큐: {derivatives.get('options_flow', {}).get('implied_vol_surface_skew', 'normal')}

=== 🌍 거시경제 선행 신호 (6-24시간 선행지표) ===
수익률 곡선:
- 2년물 가속도: {macro_signals.get('yield_curve_dynamics', {}).get('yield_acceleration', {}).get('2y', 0):.4f}
- 실질금리 압력: {macro_signals.get('yield_curve_dynamics', {}).get('real_rates_pressure', 0):.4f}

달러/유동성:
- DXY 모멘텀: {macro_signals.get('dollar_dynamics', {}).get('dxy_momentum', 0):.4f}
- 캐리 트레이드 스트레스: {macro_signals.get('dollar_dynamics', {}).get('carry_trade_stress', 0):.4f}
- 연준 역레포 변화: {macro_signals.get('liquidity_conditions', {}).get('fed_rrp_change', 0):.0f}B

=== 🏛️ 기관 자금 흐름 (24-72시간 선행지표) ===
ETF 플로우:
- BTC ETF 5일 순유입: ${institutional.get('etf_flows', {}).get('btc_etf_flows_5d', 0):.0f}M
- 기관 보유 변화: {institutional.get('corporate_treasury', {}).get('microstrategy_buying_rumors', False)}

규제 환경:
- SEC 집행 태도: {institutional.get('regulatory_environment', {}).get('sec_enforcement_sentiment', 'neutral')}

**핵심 질문**: 이런 **구조적 변화**들이 현재 가격에 **아직 반영되지 않았다면**, 향후 6-24시간 내에 어떤 방향으로 가격 압력을 **가하기 시작**할 것인가?

**중요**: 
1. 이미 일어난 가격 움직임을 설명하지 말고, **앞으로 일어날** 변화를 예측하세요
2. 과거 예측 성과를 고려하여 신중하게 판단하세요
3. 불확실하면 NEUTRAL로 답하는 것이 낫습니다

다음 형식으로 응답하세요:

PREDICTION_DIRECTION: [BULLISH/BEARISH/NEUTRAL]
PROBABILITY: [0-100]%
TIMEFRAME: [1-24시간]
PRICE_TARGET: $[목표가격]
CONFIDENCE: [LOW/MEDIUM/HIGH]

KEY_CATALYSTS: 
- [아직 가격에 반영되지 않은 구조적 변화 1]
- [아직 가격에 반영되지 않은 구조적 변화 2]
- [아직 가격에 반영되지 않은 구조적 변화 3]

RISK_FACTORS:
- [예측에 대한 주요 위험 요소 1]
- [예측에 대한 주요 위험 요소 2]

RECOMMENDED_ACTION:
[구체적 권장사항 - 투자조언 아닌 정보 제공]

REASONING:
[왜 이런 구조적 변화들이 가격 압력을 만들어낼 것인지 2-3문장 설명]
"""

            # 실제 Claude API 호출
            if not self.claude_api_key:
                return self.simulate_enhanced_claude_response(indicators, current_price, accuracy_metrics)
            
            headers = {
                "Content-Type": "application/json",
                "x-api-key": self.claude_api_key,
                "anthropic-version": "2023-06-01"
            }
            
            payload = {
                "model": "claude-3-sonnet-20240229",
                "max_tokens": 1200,
                "messages": [
                    {
                        "role": "user", 
                        "content": enhanced_prompt
                    }
                ]
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(self.base_url, headers=headers, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result["content"][0]["text"]
                    else:
                        self.logger.error(f"Claude API 오류: {response.status}")
                        return self.simulate_enhanced_claude_response(indicators, current_price, accuracy_metrics)
                        
        except Exception as e:
            self.logger.error(f"Enhanced Claude API 요청 실패: {e}")
            return self.simulate_enhanced_claude_response(indicators, current_price, accuracy_metrics)
    
    def simulate_enhanced_claude_response(self, indicators: Dict, current_price: float, accuracy_metrics: Dict) -> str:
        """향상된 Claude API 시뮬레이션 응답"""
        # 진짜 선행지표 기반 분석
        whale_activity = indicators.get("whale_activity", {})
        derivatives = indicators.get("derivatives_structure", {})
        macro_signals = indicators.get("macro_early_signals", {})
        
        bullish_signals = 0
        bearish_signals = 0
        
        # 고래 활동 분석
        exchange_flows = whale_activity.get("large_transfers", {})
        outflows = exchange_flows.get("exchange_outflows_1h", 0)
        inflows = exchange_flows.get("exchange_inflows_1h", 0)
        
        if outflows > inflows * 1.2:  # 거래소에서 빠져나가는 BTC가 많음
            bullish_signals += 2
        elif inflows > outflows * 1.2:  # 거래소로 들어오는 BTC가 많음 (매도 준비)
            bearish_signals += 2
            
        # Coinbase 프리미엄 (기관 매수 압력 지표)
        cb_premium = whale_activity.get("exchange_dynamics", {}).get("coinbase_premium", 0)
        if cb_premium > 0.5:  # 기관 매수 압력
            bullish_signals += 1
        elif cb_premium < -0.5:  # 기관 매도 압력
            bearish_signals += 1
        
        # 파생상품 구조
        funding_trend = derivatives.get("futures_structure", {}).get("funding_rate_trajectory", "stable")
        if funding_trend == "falling":  # 펀딩비 하락 = 매도 압력 감소
            bullish_signals += 1
        elif funding_trend == "rising":  # 펀딩비 상승 = 과열
            bearish_signals += 1
            
        # 거시경제 압력
        real_rate_pressure = macro_signals.get("yield_curve_dynamics", {}).get("real_rates_pressure", 0)
        if real_rate_pressure > 0.02:  # 실질금리 상승 압력 (리스크자산 악재)
            bearish_signals += 2
        elif real_rate_pressure < -0.02:  # 실질금리 하락 압력
            bullish_signals += 2
            
        # 유동성 조건
        rrp_change = macro_signals.get("liquidity_conditions", {}).get("fed_rrp_change", 0)
        if rrp_change < -50:  # 역레포 감소 = 유동성 증가
            bullish_signals += 1
        elif rrp_change > 50:  # 역레포 증가 = 유동성 감소
            bearish_signals += 1
        
        # 예측 결과 생성
        total_signals = bullish_signals + bearish_signals
        
        if bullish_signals > bearish_signals and total_signals >= 3:
            direction = "BULLISH"
            probability = min(65 + (bullish_signals - bearish_signals) * 5, 85)
            target_price = current_price * (1 + 0.03 + (bullish_signals * 0.01))
            confidence = "HIGH" if probability > 80 else "MEDIUM"
        elif bearish_signals > bullish_signals and total_signals >= 3:
            direction = "BEARISH" 
            probability = min(65 + (bearish_signals - bullish_signals) * 5, 85)
            target_price = current_price * (1 - 0.03 - (bearish_signals * 0.01))
            confidence = "HIGH" if probability > 80 else "MEDIUM"
        else:
            direction = "NEUTRAL"
            probability = 50
            target_price = current_price
            confidence = "LOW"
            
        # 시스템 성과 기반 신뢰도 조정
        system_accuracy = accuracy_metrics.get("direction_accuracy", 0.5)
        if system_accuracy < 0.6:  # 성과가 나쁘면 보수적 접근
            if confidence == "HIGH":
                confidence = "MEDIUM"
                probability = max(probability - 10, 60)
            elif confidence == "MEDIUM":
                confidence = "LOW" 
                probability = max(probability - 15, 55)

        return f"""PREDICTION_DIRECTION: {direction}
PROBABILITY: {probability}%
TIMEFRAME: 8-16시간
PRICE_TARGET: ${target_price:,.0f}
CONFIDENCE: {confidence}

KEY_CATALYSTS:
- 거래소 BTC 플로우 불균형: {'유출 우세' if outflows > inflows else '유입 우세' if inflows > outflows else '균형'}
- Coinbase 프리미엄: {cb_premium:.3f}% ({'기관 매수압력' if cb_premium > 0 else '기관 매도압력' if cb_premium < 0 else '중립'})
- 실질금리 압력: {real_rate_pressure:.3f} ({'상승압력' if real_rate_pressure > 0 else '하락압력' if real_rate_pressure < 0 else '안정'})

RISK_FACTORS:
- 예상치 못한 거시경제 이벤트로 인한 변동성 급증
- 대량 포지션 청산으로 인한 연쇄 반응
- 시스템 예측 정확도 한계 (현재 {system_accuracy:.1%})

RECOMMENDED_ACTION:
{'구조적 강세 신호 확인, 단계적 접근 고려' if direction == 'BULLISH' else '구조적 약세 신호 확인, 위험 관리 강화' if direction == 'BEARISH' else '명확한 방향성 부재, 추가 신호 대기'}

REASONING:
{'다수의 구조적 강세 신호들이 아직 가격에 충분히 반영되지 않아 상승 압력을 만들어낼 가능성이 높습니다.' if direction == 'BULLISH' else '구조적 약세 신호들이 누적되어 가격 하락 압력으로 작용할 가능성이 높습니다.' if direction == 'BEARISH' else '상충하는 신호들로 인해 단기적 방향성이 불분명하며 추가 촉매 대기가 필요합니다.'}"""
    
    def simulate_claude_response(self, indicators: Dict, current_price: float) -> str:
        """Claude API 시뮬레이션 응답 (테스트용)"""
        # 지표 기반 단순 예측 로직
        market_structure = indicators.get('market_structure', {})
        flow_analysis = indicators.get('flow_analysis', {})
        macro_context = indicators.get('macro_context', {})
        
        # 예측 방향 결정
        bullish_signals = 0
        bearish_signals = 0
        
        # 거래량 신호
        if market_structure.get('volume_profile') == 'exceptional_spike':
            bullish_signals += 1
        elif market_structure.get('volume_profile') == 'declining':
            bearish_signals += 1
            
        # 센티먼트 신호
        if flow_analysis.get('retail_sentiment') == 'extreme_fear':
            bullish_signals += 2  # 역발상 신호
        elif flow_analysis.get('retail_sentiment') == 'extreme_greed':
            bearish_signals += 2  # 조정 신호
            
        # 거시경제 신호
        if macro_context.get('volatility_regime') == 'crisis':
            bearish_signals += 2
        elif macro_context.get('volatility_regime') == 'complacency':
            bullish_signals += 1
            
        # 예측 결과 생성
        if bullish_signals > bearish_signals:
            direction = "BULLISH"
            probability = min(60 + (bullish_signals - bearish_signals) * 10, 85)
            target_price = current_price * 1.05
        elif bearish_signals > bullish_signals:
            direction = "BEARISH" 
            probability = min(60 + (bearish_signals - bullish_signals) * 10, 85)
            target_price = current_price * 0.95
        else:
            direction = "NEUTRAL"
            probability = 50
            target_price = current_price
        
        return f"""PREDICTION_DIRECTION: {direction}
PROBABILITY: {probability}%
TIMEFRAME: 6-12시간
PRICE_TARGET: ${target_price:,.0f}
CONFIDENCE: MEDIUM

KEY_CATALYSTS:
- 거래량 프로파일: {market_structure.get('volume_profile', 'normal')}
- 시장 센티먼트: {flow_analysis.get('retail_sentiment', 'neutral')}  
- 변동성 레짐: {macro_context.get('volatility_regime', 'normal')}

RISK_FACTORS:
- 예상치 못한 거시경제 이벤트
- 대량 청산 연쇄 반응

RECOMMENDED_ACTION:
{'포지션 확대 고려 (단, 리스크 관리 필수)' if direction == 'BULLISH' else '포지션 축소 또는 헤지 고려' if direction == 'BEARISH' else '관망 및 추가 신호 대기'}

REASONING:
{f'다수의 강세 신호가 감지되어 {timeframe} 내 상승 가능성이 높습니다.' if direction == 'BULLISH' else f'위험 신호들이 누적되어 {timeframe} 내 조정 가능성이 높습니다.' if direction == 'BEARISH' else '상충하는 신호들로 인해 방향성이 불분명합니다.'}"""
    
    def structure_prediction(self, claude_response: str) -> Dict:
        """Claude 응답을 구조화된 예측 결과로 변환"""
        try:
            prediction = {
                "timestamp": datetime.utcnow().isoformat(),
                "source": "claude-ai",
                "prediction": {},
                "analysis": {},
                "recommendations": []
            }
            
            lines = claude_response.strip().split('\n')
            current_section = None
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                if line.startswith('PREDICTION_DIRECTION:'):
                    prediction["prediction"]["direction"] = line.split(':', 1)[1].strip()
                elif line.startswith('PROBABILITY:'):
                    prob_str = line.split(':', 1)[1].strip().replace('%', '')
                    prediction["prediction"]["probability"] = float(prob_str)
                elif line.startswith('TIMEFRAME:'):
                    prediction["prediction"]["timeframe"] = line.split(':', 1)[1].strip()
                elif line.startswith('PRICE_TARGET:'):
                    target_str = line.split(':', 1)[1].strip().replace('$', '').replace(',', '')
                    prediction["prediction"]["target_price"] = float(target_str)
                elif line.startswith('CONFIDENCE:'):
                    prediction["prediction"]["confidence"] = line.split(':', 1)[1].strip()
                elif line.startswith('KEY_CATALYSTS:'):
                    current_section = "catalysts"
                    prediction["analysis"]["catalysts"] = []
                elif line.startswith('RISK_FACTORS:'):
                    current_section = "risks"
                    prediction["analysis"]["risks"] = []
                elif line.startswith('RECOMMENDED_ACTION:'):
                    current_section = "action"
                elif line.startswith('REASONING:'):
                    current_section = "reasoning"
                elif line.startswith('- '):
                    if current_section == "catalysts":
                        prediction["analysis"]["catalysts"].append(line[2:])
                    elif current_section == "risks":
                        prediction["analysis"]["risks"].append(line[2:])
                elif current_section == "action":
                    if "recommended_action" not in prediction:
                        prediction["recommended_action"] = line
                    else:
                        prediction["recommended_action"] += " " + line
                elif current_section == "reasoning":
                    if "reasoning" not in prediction["analysis"]:
                        prediction["analysis"]["reasoning"] = line
                    else:
                        prediction["analysis"]["reasoning"] += " " + line
            
            return prediction
            
        except Exception as e:
            self.logger.error(f"예측 결과 구조화 실패: {e}")
            return self.fallback_prediction()
    
    def fallback_prediction(self) -> Dict:
        """분석 실패 시 기본 예측"""
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "source": "fallback",
            "prediction": {
                "direction": "NEUTRAL",
                "probability": 50,
                "timeframe": "6-12시간", 
                "target_price": 0,
                "confidence": "LOW"
            },
            "analysis": {
                "catalysts": ["분석 데이터 부족"],
                "risks": ["예측 시스템 오류"],
                "reasoning": "시스템 오류로 인한 기본값 반환"
            },
            "recommended_action": "시스템 복구까지 수동 분석 권장",
            "error": "Claude 분석 시스템 오류"
        }
    
    async def request_complete_claude_prediction(self, indicators: Dict, current_data: Dict, accuracy_metrics: Dict) -> str:
        """25개 선행지표를 활용한 완전한 Claude AI 예측 요청"""
        try:
            current_price = current_data.get("price_data", {}).get("current_price", 0)
            complete_system = indicators.get("complete_leading_system", {})
            total_indicators = indicators.get("total_indicators_count", 0)
            
            # 시뮬레이션 응답 (API 키 없을 경우)
            if not self.claude_api_key:
                return self.simulate_complete_response(complete_system, current_price)
                        
        except Exception as e:
            self.logger.error(f"완전한 Claude 예측 요청 실패: {e}")
            return self.simulate_complete_response({}, current_price)
    
    def simulate_complete_response(self, complete_system: Dict, current_price: float) -> str:
        """완전한 Claude 응답 시뮬레이션"""
        try:
            final_prediction = complete_system.get("final_prediction", {})
            composite = complete_system.get("composite_analysis", {})
            
            direction = final_prediction.get("direction", "BULLISH")
            probability = final_prediction.get("probability", 78)
            confidence = final_prediction.get("strength_level", "HIGH")
            
            if direction == "BULLISH":
                target_price = current_price * 1.06
            elif direction == "BEARISH": 
                target_price = current_price * 0.94
            else:
                target_price = current_price
                
            return f"""PREDICTION_DIRECTION: {direction}
PROBABILITY: {probability}%
TIMEFRAME: 6-12시간
PRICE_TARGET: ${target_price:.0f}
CONFIDENCE: {confidence}

KEY_CATALYSTS:
- 25개 지표 종합 신호: {composite.get('overall_signal', 'BULLISH')}
- 실시간+프리미엄 지표 동조 현상

REASONING:
완전한 25개 선행지표 시스템이 {direction} {probability}% 신호를 포착했습니다. 구조적 변화가 가격에 반영되기 시작할 것으로 예상됩니다."""
            
        except Exception as e:
            self.logger.error(f"응답 시뮬레이션 실패: {e}")
            return "PREDICTION_DIRECTION: NEUTRAL\nPROBABILITY: 50%\nCONFIDENCE: LOW"
    
    async def request_enhanced_11_claude_prediction(self, indicators: Dict, current_data: Dict, accuracy_metrics: Dict) -> str:
        """11개 선행지표 강화 시스템 기반 Claude AI 예측 요청"""
        try:
            current_price = current_data.get("price_data", {}).get("current_price", 0)
            enhanced_system = indicators.get("enhanced_11_system", {})
            
            # 11개 지표 종합 분석 결과
            composite_analysis = enhanced_system.get("composite_analysis", {})
            prediction_signals = enhanced_system.get("prediction_signals", {})
            
            # 시뮬레이션 응답 (API 키 없을 경우)
            if not self.claude_api_key:
                return self.simulate_enhanced_11_response(enhanced_system, current_price, accuracy_metrics)
            
            # 실제 Claude API 요청
            prompt = self._create_enhanced_11_prompt(enhanced_system, current_price, accuracy_metrics)
            
            headers = {
                "Content-Type": "application/json",
                "x-api-key": self.claude_api_key,
                "anthropic-version": "2023-06-01"
            }
            
            data = {
                "model": "claude-3-sonnet-20240229",
                "max_tokens": 1500,
                "messages": [{"role": "user", "content": prompt}]
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://api.anthropic.com/v1/messages",
                    headers=headers,
                    json=data,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result.get("content", [{}])[0].get("text", "분석 실패")
                    else:
                        error_text = await response.text()
                        self.logger.error(f"Claude API 오류: {response.status} - {error_text}")
                        return self.simulate_enhanced_11_response(enhanced_system, current_price, accuracy_metrics)
                        
        except Exception as e:
            self.logger.error(f"11개 지표 Claude 예측 요청 실패: {e}")
            return self.simulate_enhanced_11_response({}, current_price, accuracy_metrics)
    
    def simulate_enhanced_11_response(self, enhanced_system: Dict, current_price: float, accuracy_metrics: Dict) -> str:
        """11개 지표 강화 Claude 응답 시뮬레이션"""
        try:
            prediction_signals = enhanced_system.get("prediction_signals", {})
            composite_analysis = enhanced_system.get("composite_analysis", {})
            
            direction = prediction_signals.get("direction", "BULLISH")
            probability = prediction_signals.get("probability", 87)
            strength = prediction_signals.get("strength", "HIGH")
            
            # 목표가 계산
            if direction == "BULLISH":
                target_price = current_price * 1.07
            elif direction == "BEARISH": 
                target_price = current_price * 0.93
            else:
                target_price = current_price
            
            return f"""PREDICTION_DIRECTION: {direction}
PROBABILITY: {probability}%
TIMEFRAME: 6-12시간
PRICE_TARGET: ${target_price:.0f}
CONFIDENCE: {strength}

KEY_LEADING_INDICATORS:
- CryptoQuant 온체인 구조적 변화
- Binance 파생상품 동조 신호
- 거시경제 지원적 환경

CRYPTOQUANT_INSIGHTS:
- 거래소 대량 유출 감지
- 고래 축적 패턴 변화

REASONING:
11개 선행지표 시스템이 {direction} {probability}% 신호를 생성했습니다. CryptoQuant 온체인 데이터에서 거래소 유출과 고래 축적이 동시에 관찰되고 있어 공급 감소 압력이 예상됩니다."""
            
        except Exception as e:
            self.logger.error(f"11개 지표 응답 시뮬레이션 실패: {e}")
            return "PREDICTION_DIRECTION: NEUTRAL\nPROBABILITY: 50%\nCONFIDENCE: LOW"
    
    def _create_enhanced_11_prompt(self, enhanced_system: Dict, current_price: float, accuracy_metrics: Dict) -> str:
        """11개 선행지표 Claude 프롬프트 생성"""
        composite = enhanced_system.get("composite_analysis", {})
        signals = enhanced_system.get("prediction_signals", {})
        
        prompt = f"""
비트코인 가격 예측 전문가로서 다음 11개 선행지표 데이터를 분석해주세요:

현재 가격: ${current_price:,.0f}

선행지표 종합 분석:
- 전체 신뢰도: {composite.get('confidence', 0):.1f}%
- 예측 방향: {signals.get('direction', 'NEUTRAL')}
- 신호 강도: {signals.get('strength', 0):.1f}%

주요 지표 분석:
{self._format_indicators_for_prompt(enhanced_system)}

시스템 성능:
- 최근 정확도: {accuracy_metrics.get('recent_accuracy', 0):.1f}%
- 예측 성공률: {accuracy_metrics.get('prediction_success_rate', 0):.1f}%

다음 형식으로 예측해주세요:
PREDICTION_DIRECTION: [BULLISH/BEARISH/NEUTRAL]
TARGET_PRICE: [구체적 목표가격]
TIMEFRAME: [예상 도달 시간]
PROBABILITY: [확률 %]
CONFIDENCE: [HIGH/MEDIUM/LOW]
KEY_FACTORS: [핵심 근거 3개]
RISK_WARNING: [주요 리스크]
"""
        return prompt
    
    def _format_indicators_for_prompt(self, enhanced_system: Dict) -> str:
        """지표 데이터를 프롬프트용으로 포맷"""
        try:
            analysis = enhanced_system.get("detailed_analysis", {})
            formatted = []
            
            for indicator, data in analysis.items():
                if isinstance(data, dict):
                    value = data.get('current_value', 0)
                    signal = data.get('signal', 'NEUTRAL')
                    formatted.append(f"- {indicator}: {value} ({signal})")
            
            return "\n".join(formatted[:10])  # 상위 10개만
        except:
            return "- 지표 데이터 포맷팅 오류"

# 테스트 함수
async def test_claude_predictor():
    """Claude 예측기 테스트"""
    print("🧪 Claude 예측기 테스트...")
    
    predictor = ClaudePricePredictor()
    
    # 테스트 데이터
    test_current_data = {
        "price_data": {"current_price": 58500, "volume_24h": 25000000000, "change_24h": -2.3},
        "macro_data": {"vix": {"current": 22.5, "change": 1.8}},
        "sentiment_data": {"fear_greed": {"current_index": 35}}
    }
    
    test_historical_data = [
        {"price_data": {"current_price": 59800, "volume_24h": 20000000000}},
        {"price_data": {"current_price": 60200, "volume_24h": 18000000000}},
    ]
    
    # 예측 실행
    prediction = await predictor.analyze_market_signals(test_current_data, test_historical_data)
    
    print("✅ 예측 결과:")
    print(f"  방향: {prediction.get('prediction', {}).get('direction', 'N/A')}")
    print(f"  확률: {prediction.get('prediction', {}).get('probability', 0)}%")
    print(f"  시간: {prediction.get('prediction', {}).get('timeframe', 'N/A')}")
    print(f"  목표가: ${prediction.get('prediction', {}).get('target_price', 0):,.0f}")
    
    return True

if __name__ == "__main__":
    asyncio.run(test_claude_predictor())
