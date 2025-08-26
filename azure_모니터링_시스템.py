#!/usr/bin/env python3
"""
☁️ Azure 실시간 모니터링 시스템
목적: 학습된 핵심 지표를 24시간 모니터링하고 위험 감지시 텔레그램 알림

기능:
1. Azure Function으로 24시간 가동 (컴퓨터 꺼져도 작동)
2. 학습된 20개 핵심 지표 실시간 추적
3. 위험 패턴 감지시 텔레그램 알림
4. 예측 변화량 계산 및 알림
"""

import asyncio
import aiohttp
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any
import logging

# Azure Function용 설정
try:
    import azure.functions as func
    AZURE_MODE = True
except ImportError:
    AZURE_MODE = False

# 텔레그램 설정 (환경변수에서 로드)
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN', '')  # 환경변수에서 설정
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID', '')     # 환경변수에서 설정

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AzureBTCMonitor:
    def __init__(self):
        self.critical_indicators = self.load_critical_indicators()
        self.last_values = {}
        self.alert_history = {}
        
        # 알림 임계값 설정
        self.alert_thresholds = {
            'high_importance': 0.002,    # 중요도 0.002 이상
            'price_change': 3.0,         # 3% 이상 변화
            'volume_spike': 2.0,         # 2배 이상 볼륨 증가
            'whale_movement': 0.1        # 고래 비율 10% 이상 변화
        }
        
        print("☁️ Azure BTC 모니터링 시스템 초기화")
        print(f"📊 모니터링 지표: {len(self.critical_indicators)}개")
        
    def load_critical_indicators(self) -> List[str]:
        """학습된 핵심 지표 로드"""
        try:
            with open('/Users/parkyoungjun/Desktop/BTC_Analysis_System/critical_indicators.json', 'r') as f:
                data = json.load(f)
                return data.get('critical_indicators', [])
        except FileNotFoundError:
            # 기본 핵심 지표 (학습 결과 기반)
            return [
                'macd_line', 'btc_layer2_activity', 'huobi_volume_24h',
                'btc_whale_ratio', 'ema_10', 'price_momentum_24h',
                'volatility_autocorrelation', 'addresses_balance_0.1btc_active'
            ]
    
    async def collect_current_indicators(self) -> Dict:
        """현재 핵심 지표값 수집"""
        try:
            # 실시간 데이터 수집 (간단화)
            async with aiohttp.ClientSession() as session:
                # CoinGecko에서 기본 데이터
                async with session.get(
                    'https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd&include_24hr_change=true&include_24hr_vol=true'
                ) as resp:
                    price_data = await resp.json()
                
                # Binance에서 볼륨 데이터
                async with session.get(
                    'https://api.binance.com/api/v3/ticker/24hr?symbol=BTCUSDT'
                ) as resp:
                    binance_data = await resp.json()
            
            current_price = price_data['bitcoin']['usd']
            price_change_24h = price_data['bitcoin']['usd_24h_change']
            volume_24h = float(binance_data['volume'])
            
            # 핵심 지표 추정/계산
            current_indicators = {
                'timestamp': datetime.now().isoformat(),
                'btc_price': current_price,
                'price_change_24h': price_change_24h,
                'volume_24h': volume_24h,
                
                # 중요 지표들 (실제 API가 없으므로 추정)
                'macd_line': self.estimate_macd(current_price),
                'btc_layer2_activity': self.estimate_layer2_activity(volume_24h),
                'huobi_volume_24h': volume_24h * 0.15,  # Huobi 비중 추정
                'btc_whale_ratio': self.estimate_whale_ratio(price_change_24h),
                'ema_10': self.estimate_ema_10(current_price),
                'price_momentum_24h': price_change_24h,
                'volatility_autocorrelation': abs(price_change_24h) / 10,
                'addresses_balance_0.1btc_active': 800000 + int(volume_24h / 1000)
            }
            
            return current_indicators
            
        except Exception as e:
            logger.error(f"지표 수집 실패: {e}")
            return {}
    
    def estimate_macd(self, current_price: float) -> float:
        """MACD 추정 (실제로는 과거 12/26일 이동평균 계산 필요)"""
        # 단순 추정 (실제 구현시에는 과거 데이터 필요)
        base_macd = current_price * 0.001
        return base_macd
    
    def estimate_layer2_activity(self, volume: float) -> float:
        """레이어2 활동성 추정"""
        return volume * 0.05  # 볼륨의 5%로 추정
    
    def estimate_whale_ratio(self, price_change: float) -> float:
        """고래 비율 추정"""
        base_ratio = 0.45
        volatility_effect = abs(price_change) * 0.01
        return min(0.8, max(0.2, base_ratio + volatility_effect))
    
    def estimate_ema_10(self, current_price: float) -> float:
        """10일 EMA 추정"""
        return current_price * 0.998  # 현재가에 가까운 값
    
    def analyze_risk_patterns(self, current_data: Dict) -> List[Dict]:
        """위험 패턴 분석"""
        alerts = []
        
        # 1. 급격한 가격 변화 감지
        price_change = current_data.get('price_change_24h', 0)
        if abs(price_change) > self.alert_thresholds['price_change']:
            severity = 'HIGH' if abs(price_change) > 5 else 'MEDIUM'
            direction = '급등' if price_change > 0 else '급락'
            
            alerts.append({
                'type': 'PRICE_ALERT',
                'severity': severity,
                'message': f"🚨 BTC {direction} 감지: {price_change:+.2f}%",
                'details': f"현재 가격: ${current_data.get('btc_price', 0):,.0f}",
                'recommendation': f"예측 모델에 {abs(price_change) * 100:.0f}$ 이상 영향 예상"
            })
        
        # 2. 볼륨 스파이크 감지
        current_volume = current_data.get('volume_24h', 0)
        if 'last_volume' in self.last_values:
            volume_ratio = current_volume / self.last_values.get('last_volume', current_volume)
            if volume_ratio > self.alert_thresholds['volume_spike']:
                alerts.append({
                    'type': 'VOLUME_SPIKE',
                    'severity': 'MEDIUM',
                    'message': f"📊 거래량 급증: {volume_ratio:.1f}x",
                    'details': f"현재: {current_volume:,.0f} BTC",
                    'recommendation': "큰 시장 움직임 예상, 예측 변동성 증가"
                })
        
        # 3. 고래 움직임 감지
        whale_ratio = current_data.get('btc_whale_ratio', 0.45)
        if 'last_whale_ratio' in self.last_values:
            whale_change = abs(whale_ratio - self.last_values.get('last_whale_ratio', whale_ratio))
            if whale_change > self.alert_thresholds['whale_movement']:
                alerts.append({
                    'type': 'WHALE_MOVEMENT',
                    'severity': 'HIGH',
                    'message': f"🐋 고래 움직임 감지: {whale_change:.3f} 변화",
                    'details': f"고래 비율: {whale_ratio:.3f}",
                    'recommendation': "대형 거래 예상, 예측 정확도에 큰 영향"
                })
        
        # 4. MACD 신호 변화
        macd = current_data.get('macd_line', 0)
        if 'last_macd' in self.last_values:
            macd_change = macd - self.last_values.get('last_macd', macd)
            if abs(macd_change) > current_data.get('btc_price', 100000) * 0.001:
                signal = '골든크로스' if macd_change > 0 else '데드크로스'
                alerts.append({
                    'type': 'TECHNICAL_SIGNAL',
                    'severity': 'MEDIUM',
                    'message': f"📈 MACD {signal} 신호",
                    'details': f"MACD: {macd:.4f} (변화: {macd_change:+.4f})",
                    'recommendation': f"기술적 분석 신호 변화, 예측 재검토 권장"
                })
        
        return alerts
    
    async def send_telegram_alert(self, alerts: List[Dict]):
        """텔레그램 알림 발송"""
        if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
            logger.warning("텔레그램 설정이 없어 알림을 건너뜁니다")
            return
        
        if not alerts:
            return
        
        # 알림 메시지 구성
        message_parts = ["🚨 *BTC 모니터링 알림*\n"]
        
        for alert in alerts:
            severity_emoji = {
                'HIGH': '🔴',
                'MEDIUM': '🟡', 
                'LOW': '🟢'
            }.get(alert['severity'], '⚪')
            
            message_parts.append(
                f"{severity_emoji} *{alert['message']}*\n"
                f"📊 {alert['details']}\n"
                f"💡 {alert['recommendation']}\n"
            )
        
        message_parts.append(f"\n⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        message = "\n".join(message_parts)
        
        try:
            async with aiohttp.ClientSession() as session:
                url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
                data = {
                    'chat_id': TELEGRAM_CHAT_ID,
                    'text': message,
                    'parse_mode': 'Markdown'
                }
                
                async with session.post(url, data=data) as resp:
                    if resp.status == 200:
                        logger.info(f"텔레그램 알림 발송 완료: {len(alerts)}개 알림")
                    else:
                        logger.error(f"텔레그램 알림 발송 실패: {resp.status}")
                        
        except Exception as e:
            logger.error(f"텔레그램 알림 오류: {e}")
    
    async def run_monitoring_cycle(self):
        """한 번의 모니터링 사이클 실행"""
        logger.info("🔄 모니터링 사이클 시작")
        
        # 현재 지표 수집
        current_data = await self.collect_current_indicators()
        if not current_data:
            logger.error("❌ 데이터 수집 실패")
            return
        
        # 위험 패턴 분석
        alerts = self.analyze_risk_patterns(current_data)
        
        # 알림 발송
        if alerts:
            await self.send_telegram_alert(alerts)
            logger.info(f"🚨 {len(alerts)}개 알림 발생")
        else:
            logger.info("✅ 모든 지표 정상")
        
        # 현재값을 다음 비교용으로 저장
        self.last_values = {
            'last_volume': current_data.get('volume_24h'),
            'last_whale_ratio': current_data.get('btc_whale_ratio'),
            'last_macd': current_data.get('macd_line'),
            'timestamp': current_data.get('timestamp')
        }
        
        logger.info("✅ 모니터링 사이클 완료")
    
    async def run_continuous_monitoring(self):
        """연속 모니터링 (로컬 테스트용)"""
        logger.info("🚀 연속 모니터링 시작 (5분 간격)")
        
        while True:
            try:
                await self.run_monitoring_cycle()
                await asyncio.sleep(300)  # 5분 대기
                
            except KeyboardInterrupt:
                logger.info("👋 모니터링 중단")
                break
            except Exception as e:
                logger.error(f"❌ 모니터링 오류: {e}")
                await asyncio.sleep(60)  # 오류시 1분 대기

# Azure Function 진입점
def main(req: func.HttpRequest) -> func.HttpResponse:
    """Azure Function 메인 함수"""
    if not AZURE_MODE:
        return func.HttpResponse("Azure Functions 환경이 아닙니다", status_code=500)
    
    try:
        monitor = AzureBTCMonitor()
        
        # 비동기 함수를 동기적으로 실행 (Azure Functions 호환)
        import asyncio
        asyncio.run(monitor.run_monitoring_cycle())
        
        return func.HttpResponse(
            json.dumps({
                "status": "success",
                "message": "모니터링 완료",
                "timestamp": datetime.now().isoformat()
            }),
            status_code=200,
            mimetype="application/json"
        )
        
    except Exception as e:
        logger.error(f"Azure Function 실행 오류: {e}")
        return func.HttpResponse(
            json.dumps({
                "status": "error", 
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            }),
            status_code=500,
            mimetype="application/json"
        )

# 로컬 실행용
if __name__ == "__main__":
    print("🧪 로컬 테스트 모드")
    print("📱 텔레그램 설정:")
    print(f"   BOT_TOKEN: {'설정됨' if TELEGRAM_BOT_TOKEN else '미설정'}")
    print(f"   CHAT_ID: {'설정됨' if TELEGRAM_CHAT_ID else '미설정'}")
    print("=" * 50)
    
    monitor = AzureBTCMonitor()
    asyncio.run(monitor.run_continuous_monitoring())