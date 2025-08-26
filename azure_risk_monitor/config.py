#!/usr/bin/env python3
"""
Azure 위험 감지 시스템 설정
"""

import os
from datetime import datetime

# 텔레그램 봇 설정
TELEGRAM_CONFIG = {
    "BOT_TOKEN": "8333838666:AAECXOavuX4aaUI9bFttGTqzNY2vb_tqGJI",
    "CHAT_ID": "5373223115",
    "MAX_MESSAGE_LENGTH": 4096
}

# 위험 감지 임계값 설정
RISK_THRESHOLDS = {
    # 🔴 긴급 위험 (즉시 알림)
    "CRITICAL": {
        "price_change_5min": 0.05,      # 5분간 5% 이상 변동
        "price_change_1hour": 0.10,     # 1시간 10% 이상 변동
        "volume_spike": 5.0,            # 거래량 5배 이상 급증
        "funding_rate": 0.001,          # 펀딩비 0.1% 이상
        "exchange_inflow": 5000,        # 거래소 5000 BTC 이상 유입
        "vix_spike": 5.0,               # VIX 5포인트 이상 급등
        "liquidation_volume": 100000000 # 청산량 1억 달러 이상
    },
    
    # 🟡 경고 위험 (주의 알림)  
    "WARNING": {
        "price_change_5min": 0.03,      # 5분간 3% 이상 변동
        "funding_rate_3d": 0.0005,      # 3일 평균 펀딩비 0.05% 이상
        "mvrv_threshold": 2.4,          # MVRV 2.4 이상 (과열)
        "fear_greed_extreme": 80,       # 공포탐욕지수 80 이상
        "correlation_break": 0.3        # 상관관계 30% 이상 변화
    },
    
    # 🔵 정보 위험 (참고 알림)
    "INFO": {
        "trend_change": 0.02,           # 트렌드 변화 감지
        "support_resistance": 0.01,     # 주요 지지/저항 근접
        "cycle_position": 0.1           # 사이클 위치 변화
    }
}

# 데이터 소스 설정 (무료 API 우선)
DATA_SOURCES = {
    "coingecko": {
        "base_url": "https://api.coingecko.com/api/v3",
        "rate_limit": 10,  # 10 calls per minute
        "endpoints": {
            "price": "/simple/price",
            "market_data": "/coins/bitcoin/market_chart",
            "fear_greed": "/indexes/fear-and-greed"
        }
    },
    
    "alpha_vantage": {
        "base_url": "https://www.alphavantage.co/query",
        "api_key": "demo",  # 무료 API 키로 시작
        "rate_limit": 5,    # 5 calls per minute
        "symbols": ["SPY", "QQQ", "GLD", "VIX"]
    },
    
    "fred": {
        "base_url": "https://api.stlouisfed.org/fred/series/observations",
        "api_key": "demo",  # 무료 연준 데이터
        "series": ["DGS10", "DGS2", "DEXUSEU", "DTWEXBGS"]
    },
    
    "alternative_me": {
        "base_url": "https://api.alternative.me",
        "endpoints": {
            "fear_greed": "/fng/",
            "crypto_data": "/v2/ticker/bitcoin/"
        }
    }
}

# Azure 설정
AZURE_CONFIG = {
    "function_timeout": 300,  # 5분 타임아웃
    "storage_account": "",    # Azure Storage 계정명 (나중에 설정)
    "container_name": "btc-risk-data",
    "log_level": "INFO"
}

# 분석 설정
ANALYSIS_CONFIG = {
    "lookback_periods": {
        "micro": 60,      # 1시간 (1분 간격)
        "short": 1440,    # 1일 (1분 간격)
        "medium": 10080,  # 1주 (1분 간격)  
        "long": 43200     # 1개월 (1분 간격)
    },
    
    "indicators_priority": {
        "priority_1": ["price", "volume", "funding_rate", "liquidations"],
        "priority_2": ["mvrv", "nvt", "exchange_flows", "derivatives"],
        "priority_3": ["onchain_metrics", "correlation", "sentiment"],
        "priority_4": ["macro_data", "cycle_indicators"],
        "priority_5": ["alternative_metrics", "social_metrics"]
    },
    
    "ml_models": {
        "anomaly_detection": True,
        "change_point_detection": True,
        "correlation_analysis": True,
        "pattern_matching": True
    }
}

# 알림 설정
NOTIFICATION_CONFIG = {
    "cooldown_minutes": {
        "CRITICAL": 5,   # 긴급: 5분 쿨다운
        "WARNING": 30,   # 경고: 30분 쿨다운  
        "INFO": 120      # 정보: 2시간 쿨다운
    },
    
    "message_templates": {
        "CRITICAL": "🚨 긴급 위험 신호",
        "WARNING": "⚠️ 주의 신호 감지",
        "INFO": "📊 참고 정보"
    },
    
    "max_alerts_per_hour": {
        "CRITICAL": 12,  # 시간당 최대 12건
        "WARNING": 6,    # 시간당 최대 6건
        "INFO": 2        # 시간당 최대 2건
    }
}

# 로깅 설정
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "max_file_size": 10 * 1024 * 1024,  # 10MB
    "backup_count": 5
}

# 디버그 모드 (개발 시 사용)
DEBUG_MODE = True

print("✅ Azure 위험 감지 시스템 설정 로드 완료")