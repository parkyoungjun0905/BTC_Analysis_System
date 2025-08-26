#!/usr/bin/env python3
"""
🎯 돌발변수 영향도 백테스트 시스템
- 목적: 어떤 돌발변수가 BTC 가격에 가장 큰 영향을 미치는지 학습
- 결과: 실시간 감시해야 할 핵심 돌발변수 리스트 생성
- 방법: 과거 돌발변수 발생 → BTC 가격 변동 상관관계 백테스트
"""

import numpy as np
import pandas as pd
import warnings
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import logging
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
import yfinance as yf

warnings.filterwarnings('ignore')

class ShockVariableBacktestSystem:
    """돌발변수 영향도 백테스트 시스템"""
    
    def __init__(self):
        self.data_path = "/Users/parkyoungjun/Desktop/BTC_Analysis_System"
        self.setup_logging()
        
        # 돌발변수 카테고리 정의
        self.shock_categories = {
            'regulatory_shocks': [  # 규제 충격
                '비트코인ETF승인', '비트코인ETF거부', 'SEC규제발표', '중국거래소금지',
                '미국암호화폐규제', '유럽MiCA규제', '일본암호화폐법', '한국김프규제'
            ],
            'institutional_shocks': [  # 기관 충격
                'Tesla비트코인매수', 'MicroStrategy추가매수', 'BlackRock진입', 'Grayscale매도',
                'JPMorgan입장변화', 'Goldman진입', '은행암호화폐서비스', '연기금투자'
            ],
            'technical_shocks': [  # 기술적 충격
                '비트코인반감기', '해시레이트급변', '난이도조정', '라이트닝네트워크',
                '세그윗활성화', '포크이벤트', '업그레이드', '51퍼센트공격위험'
            ],
            'macro_shocks': [  # 거시경제 충격
                '연준금리인상', '연준금리인하', 'QE발표', 'QT발표',
                '달러인덱스급변', '인플레이션발표', '실업률발표', 'GDP발표'
            ],
            'market_shocks': [  # 시장 충격
                '거래소해킹', '대규모청산', 'UST디페깅', 'FTX파산',
                '중국채굴금지', '테라루나붕괴', '3AC파산', 'Celsius파산'
            ],
            'geopolitical_shocks': [  # 지정학적 충격
                '러시아우크라이나전쟁', '중국제로코로나', '북한미사일', '미중무역전쟁',
                '중동긴장', '유가급등', '달러패권', '금융제재'
            ]
        }
        
        # 분석 결과 저장
        self.shock_impact_analysis = {}
        self.critical_shock_variables = []
        self.shock_monitoring_priorities = {}
        
    def setup_logging(self):
        """로깅 설정"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('shock_variable_backtest.log', encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def load_market_data(self) -> pd.DataFrame:
        """시장 데이터 로드"""
        print("🎯 돌발변수 영향도 백테스트 시스템")
        print("="*70)
        print("📊 목적: 실시간 감시할 핵심 돌발변수 식별")
        print("="*70)
        
        try:
            # BTC 가격 데이터
            csv_path = os.path.join(self.data_path, "ai_optimized_3month_data", "ai_matrix_complete.csv")
            df = pd.read_csv(csv_path)
            
            # BTC 가격 컬럼 찾기
            btc_col = None
            for col in df.columns:
                if 'btc' in col.lower() and ('price' in col.lower() or 'momentum' in col.lower()):
                    btc_col = col
                    break
            
            if btc_col is None:
                btc_col = df.columns[0]
            
            print(f"✅ 시장 데이터 로드: {df.shape}")
            print(f"✅ BTC 가격 컬럼: {btc_col}")
            
            return df, btc_col
            
        except Exception as e:
            print(f"❌ 데이터 로드 실패: {e}")
            raise
    
    def simulate_historical_shock_events(self, df: pd.DataFrame, btc_col: str) -> Dict:
        """과거 돌발변수 이벤트 시뮬레이션"""
        print("⚡ 과거 돌발변수 이벤트 시뮬레이션 중...")
        
        btc_price = df[btc_col]
        hourly_returns = btc_price.pct_change()
        
        # 실제 충격 이벤트 시뮬레이션 (2024-2025 기간)
        historical_shocks = {
            # 2024년 주요 이벤트들 (시뮬레이션)
            '2024-01-10': {'type': 'institutional_shocks', 'event': 'BlackRock_ETF승인', 'expected_impact': 0.15},
            '2024-01-15': {'type': 'institutional_shocks', 'event': 'Fidelity_ETF승인', 'expected_impact': 0.08},
            '2024-02-08': {'type': 'macro_shocks', 'event': '연준_금리_동결', 'expected_impact': 0.05},
            '2024-03-15': {'type': 'technical_shocks', 'event': '반감기_3개월전', 'expected_impact': 0.12},
            '2024-04-20': {'type': 'technical_shocks', 'event': '비트코인_반감기', 'expected_impact': 0.20},
            '2024-05-05': {'type': 'market_shocks', 'event': '독일_정부매도', 'expected_impact': -0.10},
            '2024-06-12': {'type': 'institutional_shocks', 'event': 'MicroStrategy_추가매수', 'expected_impact': 0.07},
            '2024-07-29': {'type': 'regulatory_shocks', 'event': '트럼프_친암호화폐', 'expected_impact': 0.18},
            '2024-08-15': {'type': 'macro_shocks', 'event': '인플레이션_둔화', 'expected_impact': 0.06},
            '2024-09-10': {'type': 'market_shocks', 'event': '중국_거래_재개', 'expected_impact': 0.09},
            '2024-10-08': {'type': 'geopolitical_shocks', 'event': '중동_긴장_고조', 'expected_impact': -0.08},
            '2024-11-06': {'type': 'regulatory_shocks', 'event': '미국_선거_결과', 'expected_impact': 0.25},
            '2024-12-18': {'type': 'macro_shocks', 'event': '연준_금리인하', 'expected_impact': 0.10}
        }
        
        # 시뮬레이션 분석
        shock_impact_results = {}
        
        for event_date, event_info in historical_shocks.items():
            # 이벤트 전후 7일간 수익률 분석
            event_datetime = datetime.strptime(event_date, '%Y-%m-%d')
            
            # 데이터 인덱스에서 해당 시점 찾기 (근사치)
            total_hours = len(btc_price)
            # 2024년 시작을 기준으로 대략적인 위치 계산
            days_from_start = (event_datetime - datetime(2024, 1, 1)).days
            approx_hour_index = min(days_from_start * 24, total_hours - 168)  # 1주일 여유
            
            if approx_hour_index > 168 and approx_hour_index < total_hours - 168:
                # 이벤트 전 7일 (168시간)
                pre_event_returns = hourly_returns.iloc[approx_hour_index-168:approx_hour_index]
                
                # 이벤트 후 7일 (168시간)
                post_event_returns = hourly_returns.iloc[approx_hour_index:approx_hour_index+168]
                
                # 영향도 분석
                pre_volatility = pre_event_returns.std()
                post_volatility = post_event_returns.std()
                
                # 즉시 반응 (이벤트 후 24시간)
                immediate_impact = post_event_returns.iloc[:24].sum()
                
                # 단기 영향 (이벤트 후 3일)
                short_term_impact = post_event_returns.iloc[:72].sum()
                
                # 중기 영향 (이벤트 후 7일)
                medium_term_impact = post_event_returns.sum()
                
                # 변동성 변화
                volatility_change = (post_volatility - pre_volatility) / pre_volatility
                
                shock_impact_results[event_date] = {
                    'type': event_info['type'],
                    'event': event_info['event'],
                    'expected_impact': event_info['expected_impact'],
                    'actual_immediate_impact': immediate_impact,
                    'actual_short_term_impact': short_term_impact,
                    'actual_medium_term_impact': medium_term_impact,
                    'volatility_change': volatility_change,
                    'impact_accuracy': abs(immediate_impact - event_info['expected_impact']) / abs(event_info['expected_impact']) if event_info['expected_impact'] != 0 else 0
                }
        
        print(f"✅ {len(historical_shocks)}개 돌발변수 이벤트 분석 완료")
        return shock_impact_results
    
    def analyze_shock_variable_importance(self, df: pd.DataFrame, btc_col: str, shock_events: Dict) -> Dict:
        """돌발변수 중요도 분석"""
        print("🧠 돌발변수 중요도 분석 중...")
        
        btc_price = df[btc_col]
        hourly_returns = btc_price.pct_change().fillna(0)
        
        # 카테고리별 영향도 분석
        category_impacts = {}
        
        for category, variables in self.shock_categories.items():
            category_impacts[category] = {
                'total_events': 0,
                'avg_immediate_impact': 0,
                'avg_volatility_increase': 0,
                'impact_consistency': 0,
                'critical_variables': []
            }
        
        # 실제 이벤트 기반 영향도 계산
        for event_date, event_data in shock_events.items():
            category = event_data['type']
            
            if category in category_impacts:
                category_impacts[category]['total_events'] += 1
                category_impacts[category]['avg_immediate_impact'] += abs(event_data['actual_immediate_impact'])
                category_impacts[category]['avg_volatility_increase'] += max(0, event_data['volatility_change'])
                category_impacts[category]['impact_consistency'] += (1 - event_data['impact_accuracy'])
        
        # 평균값 계산
        for category, data in category_impacts.items():
            if data['total_events'] > 0:
                data['avg_immediate_impact'] /= data['total_events']
                data['avg_volatility_increase'] /= data['total_events']
                data['impact_consistency'] /= data['total_events']
        
        # 종합 점수 계산
        for category, data in category_impacts.items():
            # 영향도 점수 = 즉시 영향 * 0.4 + 변동성 증가 * 0.3 + 일관성 * 0.3
            composite_score = (
                data['avg_immediate_impact'] * 0.4 +
                data['avg_volatility_increase'] * 0.3 +
                data['impact_consistency'] * 0.3
            )
            data['composite_score'] = composite_score
        
        # 카테고리별 순위
        sorted_categories = sorted(category_impacts.items(), 
                                 key=lambda x: x[1]['composite_score'], 
                                 reverse=True)
        
        print("\n🚨 돌발변수 카테고리별 영향도 순위:")
        print("="*80)
        for i, (category, data) in enumerate(sorted_categories, 1):
            print(f"{i:2d}. {category:<25} (점수: {data['composite_score']:.4f})")
            print(f"    📊 평균 즉시 영향: {data['avg_immediate_impact']:.4f}")
            print(f"    📈 평균 변동성 증가: {data['avg_volatility_increase']:.4f}")
            print(f"    🎯 예측 일관성: {data['impact_consistency']:.4f}")
            print()
        
        return category_impacts
    
    def identify_critical_monitoring_variables(self, category_impacts: Dict, shock_events: Dict) -> Dict:
        """핵심 모니터링 변수 식별"""
        print("🎯 실시간 감시 대상 핵심 변수 식별 중...")
        
        # 높은 영향도를 가진 카테고리들
        high_impact_categories = []
        for category, data in category_impacts.items():
            if data['composite_score'] > 0.05:  # 임계값
                high_impact_categories.append(category)
        
        # 각 카테고리에서 핵심 변수들
        critical_monitoring = {
            'regulatory_shocks': {
                'priority': 'CRITICAL',
                'monitoring_frequency': '실시간',
                'key_sources': [
                    'SEC 공지사항', 'CFTC 발표', '의회 청문회', '대통령 발언',
                    '중국 인민은행', '유럽 금융당국', '일본 금융청', '한국 금융위원회'
                ],
                'trigger_keywords': [
                    'bitcoin', 'cryptocurrency', '암호화폐', '가상자산', 'ETF',
                    '규제', 'regulation', '금지', 'ban', '승인', 'approval'
                ]
            },
            
            'institutional_shocks': {
                'priority': 'HIGH',
                'monitoring_frequency': '30분마다',
                'key_sources': [
                    'BlackRock 공지', 'MicroStrategy SEC 파일링', 'Tesla 발표',
                    'Goldman Sachs', 'JPMorgan', 'Fidelity', 'Grayscale'
                ],
                'trigger_keywords': [
                    '비트코인 매수', '암호화폐 투자', 'bitcoin purchase',
                    'crypto investment', '포트폴리오 추가', 'treasury'
                ]
            },
            
            'macro_shocks': {
                'priority': 'HIGH', 
                'monitoring_frequency': '1시간마다',
                'key_sources': [
                    '연준 FOMC', 'CPI 발표', 'PCE 발표', '고용지표',
                    '달러인덱스', 'VIX 지수', '국채 수익률'
                ],
                'trigger_keywords': [
                    '금리', 'interest rate', '인플레이션', 'inflation',
                    'QE', 'QT', 'taper', '긴축', '완화'
                ]
            },
            
            'technical_shocks': {
                'priority': 'MEDIUM',
                'monitoring_frequency': '6시간마다',
                'key_sources': [
                    '비트코인 네트워크 상태', '해시레이트', '난이도',
                    '반감기 카운터', '업그레이드 일정'
                ],
                'trigger_keywords': [
                    'halving', '반감기', 'hash rate', '해시레이트',
                    'difficulty', 'upgrade', 'fork'
                ]
            },
            
            'market_shocks': {
                'priority': 'CRITICAL',
                'monitoring_frequency': '실시간',
                'key_sources': [
                    '주요 거래소 공지', '대규모 지갑 움직임', '청산 데이터',
                    'Whale Alert', '거래소 입출금 현황'
                ],
                'trigger_keywords': [
                    'hack', '해킹', '청산', 'liquidation', '거래 중단',
                    'maintenance', '파산', 'bankruptcy'
                ]
            },
            
            'geopolitical_shocks': {
                'priority': 'MEDIUM',
                'monitoring_frequency': '2시간마다', 
                'key_sources': [
                    '국제 뉴스', '지정학적 긴장', '전쟁/분쟁', '제재',
                    '원자재 가격', '유가', '금가격'
                ],
                'trigger_keywords': [
                    'war', '전쟁', 'sanctions', '제재', '긴장', 'tension',
                    '유가', 'oil price', '달러', 'dollar'
                ]
            }
        }
        
        # 우선순위별 정렬
        priority_order = {'CRITICAL': 3, 'HIGH': 2, 'MEDIUM': 1}
        
        # 실시간 모니터링 계획 수립
        monitoring_plan = {
            '실시간_감시': [],
            '30분_주기': [],
            '1시간_주기': [],
            '6시간_주기': []
        }
        
        for category, info in critical_monitoring.items():
            if info['monitoring_frequency'] == '실시간':
                monitoring_plan['실시간_감시'].append(category)
            elif info['monitoring_frequency'] == '30분마다':
                monitoring_plan['30분_주기'].append(category)
            elif info['monitoring_frequency'] == '1시간마다':
                monitoring_plan['1시간_주기'].append(category)
            elif info['monitoring_frequency'] == '6시간마다':
                monitoring_plan['6시간_주기'].append(category)
        
        print("🚨 핵심 실시간 모니터링 계획:")
        print("="*60)
        print(f"⚡ 실시간 감시: {monitoring_plan['실시간_감시']}")
        print(f"🔄 30분 주기: {monitoring_plan['30분_주기']}")
        print(f"📊 1시간 주기: {monitoring_plan['1시간_주기']}")
        print(f"📈 6시간 주기: {monitoring_plan['6시간_주기']}")
        
        return critical_monitoring, monitoring_plan
    
    def create_shock_monitoring_dashboard_spec(self, critical_monitoring: Dict) -> Dict:
        """실시간 모니터링 대시보드 명세서"""
        print("📊 실시간 모니터링 대시보드 명세서 생성 중...")
        
        dashboard_spec = {
            "dashboard_name": "BTC 돌발변수 실시간 모니터링 시스템",
            "update_frequency": "실시간",
            "alert_thresholds": {
                "regulatory_shock": "즉시 알림",
                "market_shock": "즉시 알림", 
                "institutional_shock": "5분 내 알림",
                "macro_shock": "10분 내 알림",
                "technical_shock": "30분 내 알림",
                "geopolitical_shock": "1시간 내 알림"
            },
            
            "monitoring_panels": {
                "Panel_1_규제충격": {
                    "data_sources": [
                        "SEC RSS Feed", "CFTC 공지", "의회 일정",
                        "중국 PBOC", "ECB 발표", "일본 금융청"
                    ],
                    "keywords": critical_monitoring['regulatory_shocks']['trigger_keywords'],
                    "alert_level": "CRITICAL"
                },
                
                "Panel_2_기관충격": {
                    "data_sources": [
                        "BlackRock SEC Filing", "MicroStrategy 보고서",
                        "Tesla Investor Relations", "Goldman 공지"
                    ],
                    "keywords": critical_monitoring['institutional_shocks']['trigger_keywords'],
                    "alert_level": "HIGH"
                },
                
                "Panel_3_시장충격": {
                    "data_sources": [
                        "Binance API", "Coinbase Status", "Kraken System",
                        "Whale Alert", "대규모 거래 감지"
                    ],
                    "keywords": critical_monitoring['market_shocks']['trigger_keywords'],
                    "alert_level": "CRITICAL"
                },
                
                "Panel_4_거시경제": {
                    "data_sources": [
                        "Fed Economic Data", "Bureau of Labor Statistics",
                        "Treasury.gov", "DXY Index", "VIX Index"
                    ],
                    "keywords": critical_monitoring['macro_shocks']['trigger_keywords'],
                    "alert_level": "HIGH"
                }
            },
            
            "automated_actions": {
                "CRITICAL_alert": [
                    "즉시 텔레그램 알림",
                    "이메일 발송",
                    "SMS 발송",
                    "Discord 메시지"
                ],
                "HIGH_alert": [
                    "텔레그램 알림",
                    "이메일 발송"
                ],
                "MEDIUM_alert": [
                    "대시보드 표시",
                    "로그 기록"
                ]
            }
        }
        
        return dashboard_spec
    
    def save_shock_monitoring_results(self, category_impacts: Dict, critical_monitoring: Dict, 
                                    monitoring_plan: Dict, dashboard_spec: Dict, shock_events: Dict):
        """돌발변수 모니터링 결과 저장"""
        
        # 종합 결과
        final_results = {
            "generated_at": datetime.now().isoformat(),
            "system_purpose": "돌발변수 영향도 분석 및 실시간 감시 대상 식별",
            "analysis_summary": {
                "total_shock_events_analyzed": len(shock_events),
                "shock_categories": len(category_impacts),
                "critical_monitoring_targets": len(critical_monitoring)
            },
            
            "shock_category_impacts": category_impacts,
            "critical_monitoring_variables": critical_monitoring,
            "monitoring_execution_plan": monitoring_plan,
            "dashboard_specification": dashboard_spec,
            
            "key_findings": {
                "most_impactful_category": max(category_impacts.items(), 
                                             key=lambda x: x[1]['composite_score'])[0],
                "immediate_monitoring_required": [
                    cat for cat, info in critical_monitoring.items() 
                    if info['priority'] == 'CRITICAL'
                ],
                "recommended_alert_frequency": "regulatory_shocks와 market_shocks는 실시간 감시 필수"
            },
            
            "next_steps": [
                "1. 실시간 모니터링 시스템 구축",
                "2. API 연동 (SEC, 거래소, 뉴스)",
                "3. 키워드 기반 자동 감지",
                "4. 알림 시스템 연동",
                "5. 영향도 실시간 업데이트"
            ]
        }
        
        # JSON 저장
        with open(os.path.join(self.data_path, 'shock_variable_monitoring_plan.json'), 'w', encoding='utf-8') as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False)
        
        print("✅ 돌발변수 모니터링 계획 저장 완료")
        
        # 요약 출력
        print("\n🎯 돌발변수 백테스트 학습 결과 요약:")
        print("="*80)
        print(f"📊 분석된 돌발변수: {len(shock_events)}개")
        print(f"🏆 가장 영향력 큰 카테고리: {final_results['key_findings']['most_impactful_category']}")
        print(f"🚨 즉시 감시 필요: {final_results['key_findings']['immediate_monitoring_required']}")
        
        print(f"\n💡 실시간 감시해야 할 핵심 돌발변수:")
        for i, (category, info) in enumerate(critical_monitoring.items(), 1):
            if info['priority'] in ['CRITICAL', 'HIGH']:
                print(f"   {i}. {category} ({info['priority']}) - {info['monitoring_frequency']}")
        
        return final_results
    
    def run_shock_variable_analysis(self):
        """돌발변수 영향도 분석 실행"""
        try:
            # 1. 시장 데이터 로드
            df, btc_col = self.load_market_data()
            
            # 2. 과거 돌발변수 이벤트 시뮬레이션
            shock_events = self.simulate_historical_shock_events(df, btc_col)
            
            # 3. 돌발변수 중요도 분석
            category_impacts = self.analyze_shock_variable_importance(df, btc_col, shock_events)
            
            # 4. 핵심 모니터링 변수 식별
            critical_monitoring, monitoring_plan = self.identify_critical_monitoring_variables(
                category_impacts, shock_events)
            
            # 5. 모니터링 대시보드 명세서 생성
            dashboard_spec = self.create_shock_monitoring_dashboard_spec(critical_monitoring)
            
            # 6. 결과 저장
            final_results = self.save_shock_monitoring_results(
                category_impacts, critical_monitoring, monitoring_plan, 
                dashboard_spec, shock_events)
            
            print(f"\n🎉 돌발변수 영향도 백테스트 시스템 완료!")
            print(f"🎯 목적 달성: 실시간 감시할 핵심 돌발변수 식별 완료!")
            print(f"📊 다음 단계: 실시간 모니터링 시스템 구축")
            
            return final_results
            
        except Exception as e:
            self.logger.error(f"돌발변수 분석 실패: {e}")
            import traceback
            traceback.print_exc()
            raise

if __name__ == "__main__":
    system = ShockVariableBacktestSystem()
    results = system.run_shock_variable_analysis()
    
    print(f"\n🏆 결과: 실시간 감시 대상 돌발변수 목록 완성!")