#!/usr/bin/env python3
"""
사용자 커스텀 1회성 감시 시스템
텔레그램을 통한 개인 맞춤 알림 설정 및 관리
"""

import asyncio
import re
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging

class CustomWatchlistManager:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        # 1회성 알림 조건들 저장
        self.active_watchlists = []
        self.watchlist_counter = 1
        
    def parse_command(self, message: str) -> Optional[Dict]:
        """자연어 명령을 파싱하여 감시 조건으로 변환"""
        try:
            # 메시지 정리 (공백, 대소문자)
            msg = message.strip().lower()
            
            # 기본 패턴들
            patterns = {
                # RSI 조건
                r'rsi\s*([<>]=?)\s*(\d+)': self._parse_rsi,
                r'rsi\s*(\d+)\s*(이하|이상|미만|초과)': self._parse_rsi_korean,
                
                # 가격 변동 조건
                r'(\d+)%\s*(상승|하락|급등|급락)\s*(\d+)(분|시간)': self._parse_price_change,
                r'(\d+)(분|시간)\s*(\d+)%\s*(상승|하락)': self._parse_price_change_reverse,
                
                # 가격 레벨 조건
                r'(\d+)\s*(달러|원|만원)\s*(돌파|터치|도달)': self._parse_price_level,
                r'(\d+,?\d*)\s*(달러|원)\s*(돌파|터치)': self._parse_price_level_comma,
                
                # 거래량 조건
                r'거래량\s*(\d+)배\s*(증가|상승)': self._parse_volume,
                
                # 기술지표 조건
                r'macd\s*(골든크로스|데드크로스)': self._parse_macd,
                r'볼린저밴드?\s*(상한|하한)\s*(터치|돌파)': self._parse_bollinger,
                
                # 파생상품 조건
                r'펀딩비\s*([<>]=?)\s*([\d.]+)%': self._parse_funding,
                r'펀딩비\s*([\d.]+)%\s*(이상|이하|초과|미만)': self._parse_funding_korean,
            }
            
            for pattern, parser in patterns.items():
                match = re.search(pattern, msg)
                if match:
                    condition = parser(match.groups())
                    if condition:
                        condition['id'] = f"W{self.watchlist_counter:03d}"
                        condition['created_at'] = datetime.utcnow().isoformat()
                        condition['status'] = 'active'
                        self.watchlist_counter += 1
                        return condition
                        
            return None
            
        except Exception as e:
            self.logger.error(f"명령 파싱 오류: {e}")
            return None
    
    def _parse_rsi(self, groups: Tuple) -> Dict:
        """RSI 조건 파싱: rsi < 30"""
        operator, value = groups
        return {
            'type': 'indicator',
            'indicator': 'rsi',
            'operator': operator,
            'value': float(value),
            'description': f"RSI {operator} {value}"
        }
    
    def _parse_rsi_korean(self, groups: Tuple) -> Dict:
        """RSI 한국어 조건: rsi 30 이하"""
        value, direction = groups
        op_map = {'이하': '<=', '이상': '>=', '미만': '<', '초과': '>'}
        operator = op_map.get(direction, '<=')
        return {
            'type': 'indicator',
            'indicator': 'rsi',
            'operator': operator,
            'value': float(value),
            'description': f"RSI {value} {direction}"
        }
    
    def _parse_price_change(self, groups: Tuple) -> Dict:
        """가격 변동 조건: 5% 상승 10분"""
        percentage, direction, time_val, time_unit = groups
        
        # 시간 단위 변환
        if time_unit == '분':
            minutes = int(time_val)
        else:  # 시간
            minutes = int(time_val) * 60
            
        return {
            'type': 'price_change',
            'percentage': float(percentage),
            'direction': 'up' if direction in ['상승', '급등'] else 'down',
            'timeframe_minutes': minutes,
            'description': f"{time_val}{time_unit} 내 {percentage}% {direction}"
        }
    
    def _parse_price_change_reverse(self, groups: Tuple) -> Dict:
        """가격 변동 조건 역순: 10분 5% 상승"""
        time_val, time_unit, percentage, direction = groups
        
        if time_unit == '분':
            minutes = int(time_val)
        else:
            minutes = int(time_val) * 60
            
        return {
            'type': 'price_change',
            'percentage': float(percentage),
            'direction': 'up' if direction in ['상승', '급등'] else 'down',
            'timeframe_minutes': minutes,
            'description': f"{time_val}{time_unit} 내 {percentage}% {direction}"
        }
    
    def _parse_price_level(self, groups: Tuple) -> Dict:
        """가격 레벨 조건: 60000달러 돌파"""
        price, currency, action = groups
        
        # 통화 단위 처리
        if currency == '만원':
            price_usd = float(price) * 10000 / 1300  # 대략적 환율
        elif currency == '원':
            price_usd = float(price) / 1300
        else:  # 달러
            price_usd = float(price)
            
        return {
            'type': 'price_level',
            'target_price': price_usd,
            'direction': 'above' if action in ['돌파', '초과'] else 'touch',
            'description': f"{price}{currency} {action}"
        }
    
    def _parse_price_level_comma(self, groups: Tuple) -> Dict:
        """콤마 포함 가격: 60,000달러 돌파"""
        price_str, currency, action = groups
        price = float(price_str.replace(',', ''))
        
        if currency == '원':
            price_usd = price / 1300
        else:
            price_usd = price
            
        return {
            'type': 'price_level',
            'target_price': price_usd,
            'direction': 'above' if action == '돌파' else 'touch',
            'description': f"{price_str}{currency} {action}"
        }
    
    def _parse_volume(self, groups: Tuple) -> Dict:
        """거래량 조건: 거래량 3배 증가"""
        multiplier = groups[0]
        return {
            'type': 'volume_spike',
            'multiplier': float(multiplier),
            'description': f"거래량 {multiplier}배 증가"
        }
    
    def _parse_macd(self, groups: Tuple) -> Dict:
        """MACD 크로스 조건"""
        cross_type = groups[0]
        return {
            'type': 'macd_cross',
            'cross_type': 'golden' if cross_type == '골든크로스' else 'dead',
            'description': f"MACD {cross_type}"
        }
    
    def _parse_bollinger(self, groups: Tuple) -> Dict:
        """볼린저밴드 조건"""
        band, action = groups
        return {
            'type': 'bollinger_band',
            'band': 'upper' if band == '상한' else 'lower',
            'action': action,
            'description': f"볼린저밴드 {band} {action}"
        }
    
    def _parse_funding(self, groups: Tuple) -> Dict:
        """펀딩비 조건: 펀딩비 > 0.1%"""
        operator, value = groups
        return {
            'type': 'funding_rate',
            'operator': operator,
            'value': float(value),
            'description': f"펀딩비 {operator} {value}%"
        }
    
    def _parse_funding_korean(self, groups: Tuple) -> Dict:
        """펀딩비 한국어: 펀딩비 0.1% 이상"""
        value, direction = groups
        op_map = {'이하': '<=', '이상': '>=', '미만': '<', '초과': '>'}
        operator = op_map.get(direction, '>=')
        return {
            'type': 'funding_rate',
            'operator': operator,
            'value': float(value),
            'description': f"펀딩비 {value}% {direction}"
        }
    
    def add_watchlist(self, condition: Dict) -> str:
        """감시 조건 추가"""
        try:
            # 만료 시간 설정 (24시간 후 자동 삭제)
            condition['expires_at'] = (datetime.utcnow() + timedelta(hours=24)).isoformat()
            
            self.active_watchlists.append(condition)
            
            response = f"✅ {condition['description']} 알림 설정 완료\n"
            response += f"📋 ID: {condition['id']}\n"
            response += f"⏰ 24시간 후 자동 만료"
            
            self.logger.info(f"감시 조건 추가: {condition['id']} - {condition['description']}")
            return response
            
        except Exception as e:
            self.logger.error(f"감시 조건 추가 실패: {e}")
            return "❌ 감시 조건 추가 중 오류가 발생했습니다."
    
    def get_active_watchlists(self) -> str:
        """활성 감시 목록 반환"""
        if not self.active_watchlists:
            return "📋 현재 활성화된 감시 조건이 없습니다."
        
        response = "📋 현재 감시 중인 조건들:\n\n"
        for i, condition in enumerate(self.active_watchlists, 1):
            response += f"{i}. [{condition['id']}] {condition['description']}\n"
            
            # 만료 시간 표시
            try:
                expires = datetime.fromisoformat(condition['expires_at'])
                remaining = expires - datetime.utcnow()
                hours_left = int(remaining.total_seconds() / 3600)
                response += f"   ⏰ {hours_left}시간 후 만료\n\n"
            except:
                response += f"   ⏰ 만료 시간 확인 불가\n\n"
        
        return response
    
    def remove_watchlist(self, watchlist_id: str) -> str:
        """특정 감시 조건 제거"""
        for i, condition in enumerate(self.active_watchlists):
            if condition['id'] == watchlist_id:
                removed = self.active_watchlists.pop(i)
                self.logger.info(f"감시 조건 제거: {removed['id']}")
                return f"✅ {removed['description']} 감시 해제됨"
        
        return f"❌ ID '{watchlist_id}'를 찾을 수 없습니다."
    
    def clear_expired_watchlists(self):
        """만료된 감시 조건들 정리"""
        current_time = datetime.utcnow()
        before_count = len(self.active_watchlists)
        
        self.active_watchlists = [
            condition for condition in self.active_watchlists
            if datetime.fromisoformat(condition['expires_at']) > current_time
        ]
        
        removed_count = before_count - len(self.active_watchlists)
        if removed_count > 0:
            self.logger.info(f"만료된 감시 조건 {removed_count}개 정리됨")
    
    async def check_conditions(self, current_data: Dict) -> List[Dict]:
        """현재 데이터로 감시 조건들 체크"""
        triggered_alerts = []
        conditions_to_remove = []
        
        for condition in self.active_watchlists:
            try:
                if await self._evaluate_condition(condition, current_data):
                    # 조건 달성 - 알림 생성
                    alert = {
                        'id': condition['id'],
                        'description': condition['description'],
                        'message': self._generate_triggered_message(condition, current_data),
                        'priority': 'INFO'
                    }
                    triggered_alerts.append(alert)
                    conditions_to_remove.append(condition)
                    
            except Exception as e:
                self.logger.error(f"조건 체크 오류 {condition['id']}: {e}")
        
        # 달성된 조건들 제거 (1회성)
        for condition in conditions_to_remove:
            self.active_watchlists.remove(condition)
            self.logger.info(f"조건 달성으로 제거: {condition['id']}")
        
        return triggered_alerts
    
    async def _evaluate_condition(self, condition: Dict, current_data: Dict) -> bool:
        """개별 조건 평가"""
        try:
            condition_type = condition['type']
            
            if condition_type == 'indicator':
                return self._check_indicator_condition(condition, current_data)
            elif condition_type == 'price_change':
                return self._check_price_change_condition(condition, current_data)
            elif condition_type == 'price_level':
                return self._check_price_level_condition(condition, current_data)
            elif condition_type == 'volume_spike':
                return self._check_volume_condition(condition, current_data)
            elif condition_type == 'funding_rate':
                return self._check_funding_condition(condition, current_data)
            # 추가 조건 타입들...
            
            return False
            
        except Exception as e:
            self.logger.error(f"조건 평가 오류: {e}")
            return False
    
    def _check_indicator_condition(self, condition: Dict, current_data: Dict) -> bool:
        """기술적 지표 조건 체크"""
        indicator = condition['indicator']
        operator = condition['operator']
        target_value = condition['value']
        
        # 현재 데이터에서 지표 값 추출
        if 'derived_metrics' in current_data and indicator in current_data['derived_metrics']:
            current_value = current_data['derived_metrics'][indicator]
        elif 'immediate_risk' in current_data and indicator in current_data['immediate_risk']:
            current_value = current_data['immediate_risk'][indicator]
        else:
            return False
        
        # 조건 비교
        if operator in ['<', '<=']:
            return current_value <= target_value
        elif operator in ['>', '>=']:
            return current_value >= target_value
        elif operator == '==':
            return abs(current_value - target_value) < 0.1
        
        return False
    
    def _check_price_change_condition(self, condition: Dict, current_data: Dict) -> bool:
        """가격 변동 조건 체크 (단순 버전)"""
        # 실제로는 히스토리컬 데이터 필요
        # 여기서는 24시간 변동률로 근사치 체크
        if 'price_data' not in current_data:
            return False
            
        change_24h = abs(current_data['price_data'].get('change_24h', 0))
        target_percentage = condition['percentage']
        direction = condition['direction']
        
        # 단순 체크 (실제로는 더 정교한 시간별 체크 필요)
        if direction == 'down':
            return current_data['price_data'].get('change_24h', 0) <= -target_percentage
        else:
            return current_data['price_data'].get('change_24h', 0) >= target_percentage
    
    def _check_price_level_condition(self, condition: Dict, current_data: Dict) -> bool:
        """가격 레벨 조건 체크"""
        if 'price_data' not in current_data:
            return False
            
        current_price = current_data['price_data'].get('current_price', 0)
        target_price = condition['target_price']
        direction = condition['direction']
        
        if direction == 'above':
            return current_price >= target_price
        else:  # touch
            return abs(current_price - target_price) / target_price < 0.01  # 1% 이내
    
    def _check_volume_condition(self, condition: Dict, current_data: Dict) -> bool:
        """거래량 조건 체크"""
        # 단순 버전 - 실제로는 평균 거래량과 비교 필요
        if 'volume_data' not in current_data:
            return False
        
        # 임시로 24시간 거래량이 특정 값 이상인지 체크
        volume_24h = current_data['volume_data'].get('volume_24h', 0)
        # 평균 거래량 대비 배수는 별도 계산 필요
        return volume_24h > 20000000000  # 200억 달러 이상 (임시 기준)
    
    def _check_funding_condition(self, condition: Dict, current_data: Dict) -> bool:
        """펀딩비 조건 체크"""
        if 'derivatives_data' not in current_data:
            return False
        
        # 펀딩비 데이터 확인 (구현 예정)
        return False
    
    def _generate_triggered_message(self, condition: Dict, current_data: Dict) -> str:
        """조건 달성 알림 메시지 생성"""
        message = f"🎯 [설정 알림] {condition['description']} 달성\n\n"
        
        # 현재 상황 추가
        if 'price_data' in current_data:
            price = current_data['price_data'].get('current_price', 0)
            change = current_data['price_data'].get('change_24h', 0)
            message += f"💰 현재가: ${price:,.0f} ({change:+.2f}%)\n"
        
        message += f"✅ 알림 완료 (자동 삭제됨)\n"
        message += f"📅 {datetime.utcnow().strftime('%H:%M:%S')}"
        
        return message

# 테스트 함수
async def test_custom_watchlist():
    """커스텀 감시 시스템 테스트"""
    print("🧪 커스텀 감시 시스템 테스트...")
    
    manager = CustomWatchlistManager()
    
    # 테스트 명령들
    test_commands = [
        "RSI 30 이하",
        "5% 급락 10분",
        "60000달러 돌파",
        "거래량 3배 증가",
        "펀딩비 0.1% 이상"
    ]
    
    print("\n📋 명령어 파싱 테스트:")
    for cmd in test_commands:
        condition = manager.parse_command(cmd)
        if condition:
            print(f"✅ '{cmd}' → {condition['description']}")
            manager.add_watchlist(condition)
        else:
            print(f"❌ '{cmd}' → 파싱 실패")
    
    print(f"\n{manager.get_active_watchlists()}")
    return True

if __name__ == "__main__":
    asyncio.run(test_custom_watchlist())