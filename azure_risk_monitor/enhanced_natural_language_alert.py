#!/usr/bin/env python3
"""
자연어 명령 처리 + 확장된 지표 알림 시스템
사용자가 자연스럽게 명령하면 자동으로 파싱하여 알림 설정
"""

import re
import json
from typing import Dict, List, Optional, Tuple
from custom_alert_system import CustomAlertSystem

class EnhancedNaturalLanguageAlert(CustomAlertSystem):
    """자연어 처리가 가능한 확장 알림 시스템"""
    
    def __init__(self):
        super().__init__()
        
        # 🎯 대폭 확장된 지표 목록 (100+ 지표)
        self.extended_indicators = {
            # === 기본 가격 지표 ===
            "btc_price": "비트코인 가격",
            "price_change_1h": "1시간 가격변동률",
            "price_change_24h": "24시간 가격변동률", 
            "price_change_7d": "7일 가격변동률",
            
            # === 기술적 지표 (확장) ===
            "rsi": "RSI",
            "rsi_divergence": "RSI 다이버전스",
            "macd": "MACD",
            "macd_signal": "MACD 시그널",
            "macd_histogram": "MACD 히스토그램",
            "bollinger_upper": "볼린저밴드 상단",
            "bollinger_lower": "볼린저밴드 하단",
            "bollinger_width": "볼린저밴드 폭",
            "sma_20": "20일 이동평균",
            "sma_50": "50일 이동평균", 
            "sma_200": "200일 이동평균",
            "ema_12": "12일 지수이동평균",
            "ema_26": "26일 지수이동평균",
            "atr": "ATR 변동성",
            "stochastic": "스토캐스틱",
            "williams_r": "윌리엄스 %R",
            "cci": "상품채널지수",
            "momentum": "모멘텀",
            "roc": "변화율",
            
            # === 거래량 지표 ===
            "volume_24h": "24시간 거래량",
            "volume_sma": "거래량 이평",
            "volume_ratio": "거래량 비율",
            "obv": "거래량균형지표",
            "volume_weighted_price": "거래량가중평균가",
            "accumulation_distribution": "누적분배선",
            
            # === 온체인 지표 (대폭 확장) ===
            "fear_greed": "공포탐욕지수",
            "funding_rate": "펀딩비",
            "open_interest": "미결제약정",
            "long_short_ratio": "롱숏비율",
            "whale_activity": "고래 활동",
            "exchange_flows": "거래소 유출입",
            "exchange_inflows": "거래소 유입",
            "exchange_outflows": "거래소 유출",
            "stablecoin_flows": "스테이블코인 유출입",
            "miner_flows": "마이너 유출입",
            "dormant_coins": "휴면코인 움직임",
            "coin_days_destroyed": "코인데이즈 파괴",
            "network_value_to_transactions": "NVT 비율",
            "realized_cap": "실현 시가총액",
            "market_cap_to_realized_cap": "MVRV",
            "supply_shock": "공급 충격",
            "illiquid_supply": "비유동 공급량",
            "active_addresses": "활성 주소 수",
            "new_addresses": "신규 주소 수",
            "transaction_count": "트랜잭션 수",
            "mempool_size": "멤풀 크기",
            "mempool_pressure": "멤풀 압력",
            "hash_rate": "해시레이트",
            "mining_difficulty": "채굴 난이도",
            "block_time": "블록생성시간",
            
            # === 파생상품 지표 ===
            "futures_premium": "선물 프리미엄",
            "basis_spread": "베이시스 스프레드", 
            "options_put_call": "옵션 PUT/CALL 비율",
            "options_skew": "옵션 스큐",
            "implied_volatility": "내재변동성",
            "realized_volatility": "실현변동성",
            "volatility_smile": "변동성 스마일",
            "term_structure": "기간구조",
            
            # === 감정 지표 ===
            "social_volume": "소셜 볼륨",
            "social_sentiment": "소셜 감정",
            "news_sentiment": "뉴스 감정",
            "reddit_sentiment": "레딧 감정",
            "twitter_sentiment": "트위터 감정",
            "google_trends": "구글 트렌드",
            "search_volume": "검색량",
            
            # === 거시경제 지표 ===
            "dollar_index": "달러지수",
            "gold_price": "금 가격",
            "sp500": "S&P500",
            "nasdaq": "나스닥",
            "vix": "VIX 공포지수",
            "bond_yield_10y": "10년물 국채수익률",
            "inflation_rate": "인플레이션",
            
            # === 유동성 지표 ===
            "bid_ask_spread": "호가 스프레드",
            "market_depth": "시장 깊이",
            "slippage": "슬리피지",
            "orderbook_imbalance": "호가창 불균형",
            "market_impact": "시장 충격",
            
            # === 시장구조 지표 ===
            "dominance_btc": "비트코인 점유율",
            "altcoin_season": "알트코인 시즌",
            "correlation_traditional": "전통자산 상관관계",
            "decoupling_score": "탈동조화 점수",
            
            # === 고급 지표 ===
            "gamma_exposure": "감마 익스포저",
            "delta_neutral": "델타 중립",
            "funding_arbitrage": "펀딩 차익거래",
            "basis_momentum": "베이시스 모멘텀",
            "volatility_surface": "변동성 표면"
        }
        
        # 🗣️ 자연어 패턴 매칭
        self.natural_patterns = {
            # 조건 표현
            "상승": [">", "초과", "넘으면", "오르면", "높아지면"],
            "하락": ["<", "미만", "떨어지면", "내려가면", "낮아지면"], 
            "같음": ["=", "==", "같으면", "도달하면"],
            
            # 지표 별명
            "공포지수": "fear_greed",
            "공포탐욕지수": "fear_greed",
            "펀딩비": "funding_rate", 
            "펀딩요율": "funding_rate",
            "고래활동": "whale_activity",
            "대형거래": "whale_activity",
            "거래량": "volume_24h",
            "소셜볼륨": "social_volume",
            "소셜감정": "social_sentiment",
            "비트코인가격": "btc_price",
            "BTC가격": "btc_price",
            "볼밴상단": "bollinger_upper",
            "볼밴하단": "bollinger_lower",
            "이동평균": "sma_20",
            "20일이평": "sma_20",
            "200일이평": "sma_200"
        }
        
        # 📝 메시지 템플릿
        self.message_templates = {
            "default": "{indicator_kr} {condition} 감지!",
            "fear_greed": "시장 심리 변화 감지!",
            "whale_activity": "대형 거래 포착!",
            "funding_rate": "펀딩비 이상 징후!",
            "volume": "거래량 급변 감지!"
        }

    def parse_natural_command(self, natural_text: str) -> Optional[Dict]:
        """자연어 명령을 파싱하여 알림 조건으로 변환"""
        try:
            # 1. 기본 정리
            text = natural_text.lower().strip()
            
            # 2. 지표 식별
            indicator = self._extract_indicator(text)
            if not indicator:
                return {"error": "지표를 식별할 수 없습니다"}
            
            # 3. 조건 식별 (>, <, =)
            operator = self._extract_operator(text)
            if not operator:
                return {"error": "조건을 식별할 수 없습니다"}
            
            # 4. 임계값 추출
            threshold = self._extract_threshold(text)
            if threshold is None:
                return {"error": "기준값을 찾을 수 없습니다"}
            
            # 5. 메시지 생성
            message = self._generate_auto_message(indicator, operator, threshold, text)
            
            return {
                "indicator": indicator,
                "operator": operator, 
                "threshold": threshold,
                "message": message,
                "valid": True,
                "original_text": natural_text
            }
            
        except Exception as e:
            return {"error": f"파싱 오류: {str(e)}"}
    
    def _extract_indicator(self, text: str) -> Optional[str]:
        """텍스트에서 지표 추출"""
        # 직접 매칭
        for alias, indicator in self.natural_patterns.items():
            if alias in text:
                if indicator in self.extended_indicators:
                    return indicator
        
        # 지표명 직접 검색
        for indicator, kr_name in self.extended_indicators.items():
            if kr_name in text or indicator.lower() in text:
                return indicator
                
        # 키워드 기반 추론
        if any(word in text for word in ["공포", "탐욕", "심리"]):
            return "fear_greed"
        elif any(word in text for word in ["펀딩", "funding"]):
            return "funding_rate"
        elif any(word in text for word in ["고래", "대형", "whale"]):
            return "whale_activity"
        elif any(word in text for word in ["rsi", "과매수", "과매도"]):
            return "rsi"
        elif any(word in text for word in ["거래량", "volume"]):
            return "volume_24h"
        elif any(word in text for word in ["가격", "price", "btc"]):
            return "btc_price"
            
        return None
    
    def _extract_operator(self, text: str) -> Optional[str]:
        """조건 연산자 추출"""
        if any(word in text for word in ["초과", "넘으면", "오르면", "높아지면", "상승", ">"]):
            return ">"
        elif any(word in text for word in ["미만", "떨어지면", "내려가면", "낮아지면", "하락", "아래", "<"]):
            return "<"
        elif any(word in text for word in ["같으면", "도달하면", "되면", "="]):
            return "="
        return None
    
    def _extract_threshold(self, text: str) -> Optional[float]:
        """임계값 추출"""
        import re
        
        # 숫자 패턴 찾기 (소수점, 음수 포함)
        numbers = re.findall(r'-?\d+\.?\d*', text)
        
        if numbers:
            # 가장 큰 숫자를 임계값으로 사용
            return float(max(numbers, key=lambda x: abs(float(x))))
        
        # 특수 케이스
        if "반" in text or "절반" in text:
            return 50.0
        elif "제로" in text or "영" in text:
            return 0.0
            
        return None
    
    def _generate_auto_message(self, indicator: str, operator: str, threshold: float, original_text: str) -> str:
        """자동 메시지 생성"""
        indicator_kr = self.extended_indicators.get(indicator, indicator)
        
        # 조건부 표현
        if operator == ">":
            condition = f"{threshold} 초과"
        elif operator == "<":
            condition = f"{threshold} 미만"
        else:
            condition = f"{threshold} 도달"
        
        # 템플릿 선택
        if "fear_greed" in indicator:
            return f"시장 심리 {condition} 감지!"
        elif "whale" in indicator:
            return f"대형거래 {condition} 포착!"
        elif "funding" in indicator:
            return f"펀딩비 {condition} 이상징후!"
        else:
            return f"{indicator_kr} {condition} 감지!"

    def get_all_supported_indicators(self) -> Dict[str, List[str]]:
        """지원되는 모든 지표를 카테고리별로 반환"""
        categories = {
            "기본 가격": [k for k in self.extended_indicators.keys() if "price" in k or "change" in k],
            "기술적 지표": [k for k in self.extended_indicators.keys() if k in ["rsi", "macd", "bollinger_upper", "bollinger_lower", "sma_20", "sma_50", "ema_12", "atr", "stochastic"]],
            "거래량": [k for k in self.extended_indicators.keys() if "volume" in k or "obv" in k],
            "온체인": [k for k in self.extended_indicators.keys() if k in ["fear_greed", "whale_activity", "exchange_flows", "miner_flows", "active_addresses", "hash_rate"]],
            "파생상품": [k for k in self.extended_indicators.keys() if k in ["funding_rate", "open_interest", "futures_premium", "options_put_call", "implied_volatility"]],
            "감정지표": [k for k in self.extended_indicators.keys() if "sentiment" in k or "social" in k or "news" in k],
            "거시경제": [k for k in self.extended_indicators.keys() if k in ["dollar_index", "gold_price", "sp500", "vix", "inflation_rate"]]
        }
        
        return categories

    def format_indicator_guide(self) -> str:
        """지표 사용 가이드 메시지"""
        categories = self.get_all_supported_indicators()
        
        guide = "📊 **사용 가능한 지표들** (100+ 개)\n\n"
        
        for category, indicators in categories.items():
            guide += f"**{category}** ({len(indicators)}개):\n"
            # 상위 5개만 표시
            for indicator in indicators[:5]:
                kr_name = self.extended_indicators.get(indicator, indicator)
                guide += f"• `{indicator}` - {kr_name}\n"
            if len(indicators) > 5:
                guide += f"• ... 외 {len(indicators) - 5}개\n"
            guide += "\n"
        
        guide += "💡 **자연어 명령 예시**:\n"
        guide += "• '공포지수가 30 이하로 떨어지면 알려줘'\n"
        guide += "• 'RSI가 70 넘으면 과매수 경고'\n"
        guide += "• '펀딩비가 마이너스로 가면 알림'\n"
        guide += "• '고래활동이 80 초과하면 감지'\n"
        guide += "• 'BTC가격이 10만달러 넘으면'\n\n"
        
        guide += "⚙️ **정확한 명령어도 가능**:\n"
        guide += "`/set_alert [지표] [조건] [값] \"메시지\"`"
        
        return guide

# 테스트 함수
def test_natural_language():
    """자연어 처리 테스트"""
    
    system = EnhancedNaturalLanguageAlert()
    
    test_commands = [
        "공포지수가 30 이하로 떨어지면 알려줘",
        "RSI가 70 넘으면 과매수 경고해줘", 
        "펀딩비가 마이너스로 가면 알림",
        "고래활동이 80 초과하면 감지해줘",
        "비트코인 가격이 10만달러 넘으면",
        "거래량이 평소의 2배 오르면",
        "소셜감정이 90점 이상 되면"
    ]
    
    print("🧠 자연어 명령 처리 테스트\n")
    
    for cmd in test_commands:
        result = system.parse_natural_command(cmd)
        print(f"📝 '{cmd}'")
        if result and result.get("valid"):
            print(f"✅ 파싱: {result['indicator']} {result['operator']} {result['threshold']}")
            print(f"💬 메시지: {result['message']}")
        else:
            print(f"❌ 오류: {result.get('error', '알 수 없는 오류')}")
        print()

if __name__ == "__main__":
    test_natural_language()