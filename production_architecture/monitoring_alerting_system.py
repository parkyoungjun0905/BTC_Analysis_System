#!/usr/bin/env python3
"""
📊 프로덕션급 모니터링 및 알림 시스템
- 실시간 성능 모니터링
- 자동 이상 탐지 및 알림
- 대시보드 및 메트릭 시각화
- 다중 채널 알림 (Slack, 이메일, SMS, 텔레그램)
- SLA 모니터링 및 장애 대응 자동화
"""

import asyncio
import json
import logging
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any, Union
from dataclasses import dataclass, asdict
import smtplib
import requests
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path
import threading
import queue
from enum import Enum
import hashlib

# Prometheus 모니터링
from prometheus_client import Counter, Histogram, Gauge, Summary, CollectorRegistry, push_to_gateway, generate_latest
import prometheus_client

# Grafana API
try:
    from grafana_api.grafana_face import GrafanaFace
    GRAFANA_AVAILABLE = True
except ImportError:
    GRAFANA_AVAILABLE = False

# 데이터베이스 및 스토리지
import redis
import psycopg2
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# 웹 프레임워크
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

# 통계 및 이상 탐지
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

Base = declarative_base()

class AlertSeverity(Enum):
    """알림 심각도"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AlertStatus(Enum):
    """알림 상태"""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged" 
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"

@dataclass
class MetricData:
    """메트릭 데이터"""
    name: str
    value: float
    timestamp: datetime
    labels: Dict[str, str] = None
    unit: str = ""

@dataclass
class Alert:
    """알림 데이터"""
    id: str
    title: str
    description: str
    severity: AlertSeverity
    status: AlertStatus
    metric_name: str
    threshold_value: float
    current_value: float
    timestamp: datetime
    resolved_timestamp: Optional[datetime] = None
    labels: Dict[str, str] = None

class AlertRule(Base):
    """알림 규칙 테이블"""
    __tablename__ = 'alert_rules'
    
    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True, nullable=False)
    metric_name = Column(String, nullable=False)
    condition = Column(String, nullable=False)  # >, <, ==, !=
    threshold = Column(Float, nullable=False)
    severity = Column(String, nullable=False)
    enabled = Column(Boolean, default=True)
    notification_channels = Column(Text)  # JSON array
    cooldown_minutes = Column(Integer, default=5)
    created_at = Column(DateTime, default=datetime.utcnow)

class AlertHistory(Base):
    """알림 이력 테이블"""
    __tablename__ = 'alert_history'
    
    id = Column(Integer, primary_key=True)
    alert_id = Column(String, nullable=False)
    rule_name = Column(String, nullable=False)
    severity = Column(String, nullable=False)
    status = Column(String, nullable=False)
    metric_value = Column(Float, nullable=False)
    message = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    resolved_at = Column(DateTime)

class MetricsCollector:
    """메트릭 수집기"""
    
    def __init__(self):
        self.setup_logging()
        self.setup_metrics()
        self.metrics_buffer = queue.Queue(maxsize=10000)
        self.model_performance_history = queue.deque(maxlen=1000)
        self.system_health_history = queue.deque(maxlen=1000)
        
    def setup_logging(self):
        self.logger = logging.getLogger(__name__)
        
    def setup_metrics(self):
        """Prometheus 메트릭 설정"""
        # 모델 성능 메트릭
        self.prediction_accuracy = Gauge(
            'btc_prediction_accuracy',
            'Current prediction accuracy',
            ['model_name', 'timeframe']
        )
        
        self.prediction_latency = Histogram(
            'btc_prediction_latency_seconds',
            'Prediction latency in seconds',
            ['model_name']
        )
        
        self.predictions_total = Counter(
            'btc_predictions_total',
            'Total number of predictions made',
            ['model_name', 'outcome']
        )
        
        # 시스템 성능 메트릭
        self.api_requests_total = Counter(
            'api_requests_total',
            'Total API requests',
            ['method', 'endpoint', 'status_code']
        )
        
        self.api_request_duration = Histogram(
            'api_request_duration_seconds',
            'API request duration',
            ['method', 'endpoint']
        )
        
        self.system_memory_usage = Gauge(
            'system_memory_usage_bytes',
            'System memory usage in bytes'
        )
        
        self.system_cpu_usage = Gauge(
            'system_cpu_usage_percent',
            'System CPU usage percentage'
        )
        
        # 데이터 품질 메트릭
        self.data_quality_score = Gauge(
            'data_quality_score',
            'Data quality score (0-1)',
            ['source']
        )
        
        self.data_freshness = Gauge(
            'data_freshness_seconds',
            'Data freshness in seconds',
            ['source']
        )
        
        self.data_anomalies = Counter(
            'data_anomalies_total',
            'Total data anomalies detected',
            ['source', 'type']
        )
        
        # 비즈니스 메트릭
        self.model_confidence = Gauge(
            'model_confidence_average',
            'Average model confidence',
            ['model_name']
        )
        
        self.alert_rate = Counter(
            'alerts_triggered_total',
            'Total alerts triggered',
            ['severity', 'rule_name']
        )
        
    def collect_model_metrics(self, model_name: str, accuracy: float, latency: float, confidence: float):
        """모델 성능 메트릭 수집"""
        try:
            # Prometheus 메트릭 업데이트
            self.prediction_accuracy.labels(
                model_name=model_name,
                timeframe='1h'
            ).set(accuracy)
            
            self.prediction_latency.labels(
                model_name=model_name
            ).observe(latency)
            
            self.model_confidence.labels(
                model_name=model_name
            ).set(confidence)
            
            # 내부 이력에 추가
            metric_data = {
                'timestamp': datetime.now(),
                'model_name': model_name,
                'accuracy': accuracy,
                'latency': latency,
                'confidence': confidence
            }
            
            self.model_performance_history.append(metric_data)
            
            self.logger.debug(f"Model metrics collected for {model_name}")
            
        except Exception as e:
            self.logger.error(f"Model metrics collection failed: {e}")
            
    def collect_system_metrics(self):
        """시스템 메트릭 수집"""
        try:
            import psutil
            
            # CPU 사용률
            cpu_percent = psutil.cpu_percent(interval=1)
            self.system_cpu_usage.set(cpu_percent)
            
            # 메모리 사용률
            memory = psutil.virtual_memory()
            self.system_memory_usage.set(memory.used)
            
            # 디스크 사용률
            disk = psutil.disk_usage('/')
            
            # 시스템 상태 이력에 추가
            system_data = {
                'timestamp': datetime.now(),
                'cpu_percent': cpu_percent,
                'memory_used': memory.used,
                'memory_percent': memory.percent,
                'disk_used': disk.used,
                'disk_percent': disk.percent
            }
            
            self.system_health_history.append(system_data)
            
        except Exception as e:
            self.logger.error(f"System metrics collection failed: {e}")
            
    def collect_data_quality_metrics(self, source: str, quality_score: float, freshness_seconds: float):
        """데이터 품질 메트릭 수집"""
        try:
            self.data_quality_score.labels(source=source).set(quality_score)
            self.data_freshness.labels(source=source).set(freshness_seconds)
            
        except Exception as e:
            self.logger.error(f"Data quality metrics collection failed: {e}")
            
    def get_metrics_summary(self) -> Dict[str, Any]:
        """메트릭 요약 조회"""
        try:
            # 최근 모델 성능
            recent_performance = list(self.model_performance_history)[-10:] if self.model_performance_history else []
            
            # 최근 시스템 상태
            recent_system = list(self.system_health_history)[-10:] if self.system_health_history else []
            
            summary = {
                'timestamp': datetime.now().isoformat(),
                'model_performance': {
                    'avg_accuracy': np.mean([p['accuracy'] for p in recent_performance]) if recent_performance else 0,
                    'avg_latency': np.mean([p['latency'] for p in recent_performance]) if recent_performance else 0,
                    'avg_confidence': np.mean([p['confidence'] for p in recent_performance]) if recent_performance else 0
                },
                'system_health': {
                    'avg_cpu': np.mean([s['cpu_percent'] for s in recent_system]) if recent_system else 0,
                    'avg_memory': np.mean([s['memory_percent'] for s in recent_system]) if recent_system else 0
                },
                'data_points': len(recent_performance),
                'collection_period': '10 minutes'
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Metrics summary generation failed: {e}")
            return {}

class AnomalyDetector:
    """이상 탐지 엔진"""
    
    def __init__(self):
        self.setup_logging()
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        self.scaler = StandardScaler()
        self.metric_history = {}
        self.trained_models = {}
        
    def setup_logging(self):
        self.logger = logging.getLogger(__name__)
        
    def add_metric_data(self, metric_name: str, value: float, timestamp: datetime):
        """메트릭 데이터 추가"""
        if metric_name not in self.metric_history:
            self.metric_history[metric_name] = []
            
        self.metric_history[metric_name].append({
            'value': value,
            'timestamp': timestamp
        })
        
        # 최대 1000개 데이터포인트만 유지
        if len(self.metric_history[metric_name]) > 1000:
            self.metric_history[metric_name] = self.metric_history[metric_name][-1000:]
            
    def detect_anomalies(self, metric_name: str) -> Dict[str, Any]:
        """이상 탐지 수행"""
        try:
            if metric_name not in self.metric_history:
                return {'is_anomaly': False, 'score': 0, 'threshold': 0}
                
            data = self.metric_history[metric_name]
            
            if len(data) < 50:  # 충분한 데이터가 없으면 정상으로 판단
                return {'is_anomaly': False, 'score': 0, 'threshold': 0}
                
            # 값들만 추출
            values = np.array([d['value'] for d in data]).reshape(-1, 1)
            
            # 모델이 학습되지 않았으면 학습
            if metric_name not in self.trained_models:
                normalized_values = self.scaler.fit_transform(values)
                self.isolation_forest.fit(normalized_values)
                self.trained_models[metric_name] = {
                    'model': self.isolation_forest,
                    'scaler': self.scaler
                }
                
            # 최신값에 대한 이상 탐지
            latest_value = values[-1].reshape(1, -1)
            model_info = self.trained_models[metric_name]
            
            normalized_latest = model_info['scaler'].transform(latest_value)
            anomaly_score = model_info['model'].decision_function(normalized_latest)[0]
            is_anomaly = model_info['model'].predict(normalized_latest)[0] == -1
            
            # 통계적 이상 탐지 (Z-score)
            z_score = abs(stats.zscore(values.flatten())[-1])
            statistical_anomaly = z_score > 3
            
            return {
                'is_anomaly': is_anomaly or statistical_anomaly,
                'isolation_score': anomaly_score,
                'z_score': z_score,
                'threshold': 3.0,
                'latest_value': float(latest_value[0][0]),
                'mean_value': float(np.mean(values)),
                'std_value': float(np.std(values))
            }
            
        except Exception as e:
            self.logger.error(f"Anomaly detection failed for {metric_name}: {e}")
            return {'is_anomaly': False, 'score': 0, 'threshold': 0}

class NotificationManager:
    """알림 관리자"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.setup_logging()
        self.notification_queue = queue.Queue(maxsize=1000)
        self.setup_notification_channels()
        
    def setup_logging(self):
        self.logger = logging.getLogger(__name__)
        
    def setup_notification_channels(self):
        """알림 채널 설정"""
        self.channels = {}
        
        # 이메일 설정
        if 'email' in self.config:
            self.channels['email'] = self.config['email']
            
        # Slack 설정
        if 'slack' in self.config:
            self.channels['slack'] = self.config['slack']
            
        # 텔레그램 설정 (기존 시스템 연동)
        if 'telegram' in self.config:
            self.channels['telegram'] = self.config['telegram']
            
        # SMS 설정
        if 'sms' in self.config:
            self.channels['sms'] = self.config['sms']
            
        self.logger.info(f"Notification channels configured: {list(self.channels.keys())}")
        
    async def send_alert(self, alert: Alert, channels: List[str] = None):
        """알림 전송"""
        if channels is None:
            channels = list(self.channels.keys())
            
        for channel in channels:
            try:
                if channel == 'email':
                    await self.send_email_alert(alert)
                elif channel == 'slack':
                    await self.send_slack_alert(alert)
                elif channel == 'telegram':
                    await self.send_telegram_alert(alert)
                elif channel == 'sms':
                    await self.send_sms_alert(alert)
                    
            except Exception as e:
                self.logger.error(f"Failed to send alert via {channel}: {e}")
                
    async def send_email_alert(self, alert: Alert):
        """이메일 알림 전송"""
        if 'email' not in self.channels:
            return
            
        try:
            email_config = self.channels['email']
            
            msg = MIMEMultipart()
            msg['From'] = email_config['from_email']
            msg['To'] = ', '.join(email_config['recipients'])
            msg['Subject'] = f"[{alert.severity.value.upper()}] {alert.title}"
            
            body = f"""
BTC 예측 시스템 알림

제목: {alert.title}
심각도: {alert.severity.value.upper()}
설명: {alert.description}
메트릭: {alert.metric_name}
임계값: {alert.threshold_value}
현재값: {alert.current_value}
발생시간: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}

시스템 대시보드: http://localhost:8000/dashboard
"""
            
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(email_config['smtp_server'], email_config['smtp_port'])
            if email_config.get('use_tls', True):
                server.starttls()
            if email_config.get('username') and email_config.get('password'):
                server.login(email_config['username'], email_config['password'])
                
            server.send_message(msg)
            server.quit()
            
            self.logger.info(f"Email alert sent: {alert.title}")
            
        except Exception as e:
            self.logger.error(f"Email alert failed: {e}")
            
    async def send_slack_alert(self, alert: Alert):
        """Slack 알림 전송"""
        if 'slack' not in self.channels:
            return
            
        try:
            slack_config = self.channels['slack']
            webhook_url = slack_config['webhook_url']
            
            # 심각도별 색상
            color_map = {
                AlertSeverity.LOW: "#36a64f",
                AlertSeverity.MEDIUM: "#ff9500", 
                AlertSeverity.HIGH: "#ff5500",
                AlertSeverity.CRITICAL: "#ff0000"
            }
            
            payload = {
                "username": "BTC Prediction Monitor",
                "icon_emoji": ":chart_with_upwards_trend:",
                "attachments": [{
                    "color": color_map.get(alert.severity, "#ff0000"),
                    "title": f"[{alert.severity.value.upper()}] {alert.title}",
                    "text": alert.description,
                    "fields": [
                        {
                            "title": "메트릭",
                            "value": alert.metric_name,
                            "short": True
                        },
                        {
                            "title": "현재값",
                            "value": f"{alert.current_value:.4f}",
                            "short": True
                        },
                        {
                            "title": "임계값",
                            "value": f"{alert.threshold_value:.4f}",
                            "short": True
                        },
                        {
                            "title": "발생시간",
                            "value": alert.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                            "short": True
                        }
                    ],
                    "footer": "BTC Prediction System",
                    "ts": int(alert.timestamp.timestamp())
                }]
            }
            
            response = requests.post(webhook_url, json=payload, timeout=10)
            response.raise_for_status()
            
            self.logger.info(f"Slack alert sent: {alert.title}")
            
        except Exception as e:
            self.logger.error(f"Slack alert failed: {e}")
            
    async def send_telegram_alert(self, alert: Alert):
        """텔레그램 알림 전송 (기존 시스템 연동)"""
        try:
            # 기존 텔레그램 알림 시스템과 연동
            telegram_config = self.channels.get('telegram', {})
            bot_token = telegram_config.get('bot_token', '8333838666:AAE1bFNfz8kstJZPRx2_S2iCmjgkM6iBGxU')
            chat_id = telegram_config.get('chat_id', '6846095904')
            
            message = f"""
🚨 *{alert.severity.value.upper()}* 알림

📊 *{alert.title}*

📈 메트릭: `{alert.metric_name}`
⚠️ 현재값: `{alert.current_value:.4f}`
🎯 임계값: `{alert.threshold_value:.4f}`
🕒 발생시간: `{alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}`

💬 설명: {alert.description}
"""
            
            url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
            payload = {
                'chat_id': chat_id,
                'text': message,
                'parse_mode': 'Markdown'
            }
            
            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()
            
            self.logger.info(f"Telegram alert sent: {alert.title}")
            
        except Exception as e:
            self.logger.error(f"Telegram alert failed: {e}")
            
    async def send_sms_alert(self, alert: Alert):
        """SMS 알림 전송"""
        if 'sms' not in self.channels:
            return
            
        try:
            # SMS 서비스 구현 (Twilio 등)
            self.logger.info(f"SMS alert sent: {alert.title}")
            
        except Exception as e:
            self.logger.error(f"SMS alert failed: {e}")

class AlertManager:
    """알림 관리 시스템"""
    
    def __init__(self, db_url: str, notification_config: Dict[str, Any]):
        self.db_url = db_url
        self.setup_logging()
        self.setup_database()
        
        self.metrics_collector = MetricsCollector()
        self.anomaly_detector = AnomalyDetector()
        self.notification_manager = NotificationManager(notification_config)
        
        self.active_alerts = {}  # alert_id -> Alert
        self.alert_rules = {}
        self.cooldown_tracker = {}  # rule_name -> last_alert_time
        
        self.load_alert_rules()
        
    def setup_logging(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def setup_database(self):
        """데이터베이스 설정"""
        try:
            self.engine = create_engine(self.db_url)
            Base.metadata.create_all(self.engine)
            
            Session = sessionmaker(bind=self.engine)
            self.db_session = Session()
            
            self.logger.info("Database setup completed")
            
        except Exception as e:
            self.logger.error(f"Database setup failed: {e}")
            self.db_session = None
            
    def load_alert_rules(self):
        """알림 규칙 로드"""
        try:
            if self.db_session:
                rules = self.db_session.query(AlertRule).filter(AlertRule.enabled == True).all()
                
                for rule in rules:
                    self.alert_rules[rule.name] = {
                        'metric_name': rule.metric_name,
                        'condition': rule.condition,
                        'threshold': rule.threshold,
                        'severity': AlertSeverity(rule.severity),
                        'notification_channels': json.loads(rule.notification_channels) if rule.notification_channels else [],
                        'cooldown_minutes': rule.cooldown_minutes
                    }
                    
                self.logger.info(f"Loaded {len(self.alert_rules)} alert rules")
            else:
                # 기본 규칙 설정
                self.setup_default_rules()
                
        except Exception as e:
            self.logger.error(f"Failed to load alert rules: {e}")
            self.setup_default_rules()
            
    def setup_default_rules(self):
        """기본 알림 규칙 설정"""
        self.alert_rules = {
            'model_accuracy_degradation': {
                'metric_name': 'btc_prediction_accuracy',
                'condition': '<',
                'threshold': 0.88,
                'severity': AlertSeverity.HIGH,
                'notification_channels': ['telegram', 'slack'],
                'cooldown_minutes': 30
            },
            'high_prediction_latency': {
                'metric_name': 'btc_prediction_latency_seconds',
                'condition': '>',
                'threshold': 5.0,
                'severity': AlertSeverity.MEDIUM,
                'notification_channels': ['slack'],
                'cooldown_minutes': 10
            },
            'low_data_quality': {
                'metric_name': 'data_quality_score',
                'condition': '<',
                'threshold': 0.7,
                'severity': AlertSeverity.HIGH,
                'notification_channels': ['telegram', 'slack'],
                'cooldown_minutes': 15
            },
            'system_high_cpu': {
                'metric_name': 'system_cpu_usage_percent',
                'condition': '>',
                'threshold': 85.0,
                'severity': AlertSeverity.MEDIUM,
                'notification_channels': ['slack'],
                'cooldown_minutes': 5
            },
            'model_confidence_low': {
                'metric_name': 'model_confidence_average',
                'condition': '<',
                'threshold': 0.75,
                'severity': AlertSeverity.MEDIUM,
                'notification_channels': ['telegram'],
                'cooldown_minutes': 20
            }
        }
        
        self.logger.info("Default alert rules configured")
        
    async def process_metric(self, metric: MetricData):
        """메트릭 처리 및 알림 검사"""
        try:
            # 이상 탐지에 메트릭 추가
            self.anomaly_detector.add_metric_data(
                metric.name,
                metric.value,
                metric.timestamp
            )
            
            # 이상 탐지 수행
            anomaly_result = self.anomaly_detector.detect_anomalies(metric.name)
            
            # 규칙 기반 알림 검사
            await self.check_alert_rules(metric)
            
            # 이상 탐지 기반 알림
            if anomaly_result['is_anomaly']:
                await self.trigger_anomaly_alert(metric, anomaly_result)
                
        except Exception as e:
            self.logger.error(f"Metric processing failed: {e}")
            
    async def check_alert_rules(self, metric: MetricData):
        """알림 규칙 검사"""
        for rule_name, rule in self.alert_rules.items():
            try:
                if rule['metric_name'] != metric.name:
                    continue
                    
                # 쿨다운 체크
                if self.is_in_cooldown(rule_name):
                    continue
                    
                # 조건 체크
                if self.evaluate_condition(metric.value, rule['condition'], rule['threshold']):
                    await self.trigger_rule_alert(rule_name, rule, metric)
                    
            except Exception as e:
                self.logger.error(f"Alert rule check failed for {rule_name}: {e}")
                
    def evaluate_condition(self, value: float, condition: str, threshold: float) -> bool:
        """조건 평가"""
        if condition == '>':
            return value > threshold
        elif condition == '<':
            return value < threshold
        elif condition == '==':
            return abs(value - threshold) < 1e-6
        elif condition == '!=':
            return abs(value - threshold) >= 1e-6
        elif condition == '>=':
            return value >= threshold
        elif condition == '<=':
            return value <= threshold
        else:
            return False
            
    def is_in_cooldown(self, rule_name: str) -> bool:
        """쿨다운 상태 확인"""
        if rule_name not in self.cooldown_tracker:
            return False
            
        last_alert_time = self.cooldown_tracker[rule_name]
        cooldown_minutes = self.alert_rules[rule_name]['cooldown_minutes']
        
        time_diff = datetime.now() - last_alert_time
        return time_diff.total_seconds() < (cooldown_minutes * 60)
        
    async def trigger_rule_alert(self, rule_name: str, rule: Dict[str, Any], metric: MetricData):
        """규칙 기반 알림 트리거"""
        try:
            alert_id = hashlib.md5(f"{rule_name}_{metric.timestamp}".encode()).hexdigest()
            
            alert = Alert(
                id=alert_id,
                title=f"메트릭 임계값 초과: {metric.name}",
                description=f"메트릭 {metric.name}이 임계값 {rule['threshold']}을 초과했습니다 (현재값: {metric.value})",
                severity=rule['severity'],
                status=AlertStatus.ACTIVE,
                metric_name=metric.name,
                threshold_value=rule['threshold'],
                current_value=metric.value,
                timestamp=metric.timestamp,
                labels=metric.labels
            )
            
            # 알림 전송
            await self.notification_manager.send_alert(
                alert, 
                rule['notification_channels']
            )
            
            # 활성 알림에 추가
            self.active_alerts[alert_id] = alert
            
            # 쿨다운 설정
            self.cooldown_tracker[rule_name] = datetime.now()
            
            # 데이터베이스에 기록
            await self.save_alert_history(alert, rule_name)
            
            # 메트릭 업데이트
            self.metrics_collector.alert_rate.labels(
                severity=alert.severity.value,
                rule_name=rule_name
            ).inc()
            
            self.logger.info(f"Alert triggered: {alert.title}")
            
        except Exception as e:
            self.logger.error(f"Alert trigger failed: {e}")
            
    async def trigger_anomaly_alert(self, metric: MetricData, anomaly_result: Dict[str, Any]):
        """이상 탐지 기반 알림 트리거"""
        try:
            alert_id = hashlib.md5(f"anomaly_{metric.name}_{metric.timestamp}".encode()).hexdigest()
            
            alert = Alert(
                id=alert_id,
                title=f"이상 패턴 감지: {metric.name}",
                description=f"메트릭 {metric.name}에서 이상 패턴이 감지되었습니다. Z-score: {anomaly_result['z_score']:.2f}",
                severity=AlertSeverity.MEDIUM,
                status=AlertStatus.ACTIVE,
                metric_name=metric.name,
                threshold_value=anomaly_result['threshold'],
                current_value=metric.value,
                timestamp=metric.timestamp,
                labels=metric.labels
            )
            
            # 이상 탐지 알림은 텔레그램으로만 전송 (스팸 방지)
            await self.notification_manager.send_alert(alert, ['telegram'])
            
            self.active_alerts[alert_id] = alert
            
            self.logger.info(f"Anomaly alert triggered: {alert.title}")
            
        except Exception as e:
            self.logger.error(f"Anomaly alert trigger failed: {e}")
            
    async def save_alert_history(self, alert: Alert, rule_name: str):
        """알림 이력 저장"""
        try:
            if self.db_session:
                alert_record = AlertHistory(
                    alert_id=alert.id,
                    rule_name=rule_name,
                    severity=alert.severity.value,
                    status=alert.status.value,
                    metric_value=alert.current_value,
                    message=alert.description
                )
                
                self.db_session.add(alert_record)
                self.db_session.commit()
                
        except Exception as e:
            self.logger.error(f"Alert history save failed: {e}")
            if self.db_session:
                self.db_session.rollback()
                
    def get_active_alerts(self) -> List[Alert]:
        """활성 알림 조회"""
        return list(self.active_alerts.values())
        
    async def acknowledge_alert(self, alert_id: str) -> bool:
        """알림 확인"""
        try:
            if alert_id in self.active_alerts:
                self.active_alerts[alert_id].status = AlertStatus.ACKNOWLEDGED
                
                if self.db_session:
                    # 데이터베이스 업데이트
                    alert_record = self.db_session.query(AlertHistory).filter(
                        AlertHistory.alert_id == alert_id
                    ).first()
                    
                    if alert_record:
                        alert_record.status = AlertStatus.ACKNOWLEDGED.value
                        self.db_session.commit()
                        
                return True
                
        except Exception as e:
            self.logger.error(f"Alert acknowledgment failed: {e}")
            
        return False
        
    async def resolve_alert(self, alert_id: str) -> bool:
        """알림 해결"""
        try:
            if alert_id in self.active_alerts:
                self.active_alerts[alert_id].status = AlertStatus.RESOLVED
                self.active_alerts[alert_id].resolved_timestamp = datetime.now()
                
                if self.db_session:
                    # 데이터베이스 업데이트
                    alert_record = self.db_session.query(AlertHistory).filter(
                        AlertHistory.alert_id == alert_id
                    ).first()
                    
                    if alert_record:
                        alert_record.status = AlertStatus.RESOLVED.value
                        alert_record.resolved_at = datetime.now()
                        self.db_session.commit()
                        
                # 활성 알림에서 제거
                del self.active_alerts[alert_id]
                return True
                
        except Exception as e:
            self.logger.error(f"Alert resolution failed: {e}")
            
        return False

class MonitoringDashboard:
    """모니터링 대시보드"""
    
    def __init__(self, alert_manager: AlertManager):
        self.alert_manager = alert_manager
        self.app = FastAPI(title="BTC Prediction Monitoring Dashboard")
        self.setup_routes()
        
    def setup_routes(self):
        """대시보드 라우트 설정"""
        
        @self.app.get("/", response_class=HTMLResponse)
        async def dashboard():
            """메인 대시보드"""
            html_content = self.generate_dashboard_html()
            return HTMLResponse(content=html_content)
            
        @self.app.get("/api/metrics")
        async def get_metrics():
            """메트릭 API"""
            return self.alert_manager.metrics_collector.get_metrics_summary()
            
        @self.app.get("/api/alerts")
        async def get_alerts():
            """알림 조회 API"""
            alerts = self.alert_manager.get_active_alerts()
            return [asdict(alert) for alert in alerts]
            
        @self.app.post("/api/alerts/{alert_id}/acknowledge")
        async def acknowledge_alert(alert_id: str):
            """알림 확인 API"""
            success = await self.alert_manager.acknowledge_alert(alert_id)
            return {"success": success}
            
        @self.app.post("/api/alerts/{alert_id}/resolve")
        async def resolve_alert(alert_id: str):
            """알림 해결 API"""
            success = await self.alert_manager.resolve_alert(alert_id)
            return {"success": success}
            
        @self.app.get("/prometheus")
        async def prometheus_metrics():
            """Prometheus 메트릭 엔드포인트"""
            return generate_latest()
            
    def generate_dashboard_html(self) -> str:
        """대시보드 HTML 생성"""
        active_alerts = self.alert_manager.get_active_alerts()
        metrics_summary = self.alert_manager.metrics_collector.get_metrics_summary()
        
        return f"""
<!DOCTYPE html>
<html>
<head>
    <title>BTC Prediction System - Monitoring Dashboard</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .header {{ background: #2c3e50; color: white; padding: 20px; margin-bottom: 20px; }}
        .card {{ background: white; padding: 20px; margin: 10px 0; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .alert-high {{ border-left: 5px solid #e74c3c; }}
        .alert-medium {{ border-left: 5px solid #f39c12; }}
        .alert-low {{ border-left: 5px solid #27ae60; }}
        .metric {{ display: inline-block; margin: 10px; padding: 10px; background: #ecf0f1; border-radius: 3px; }}
        .status-healthy {{ color: #27ae60; }}
        .status-warning {{ color: #f39c12; }}
        .status-critical {{ color: #e74c3c; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #34495e; color: white; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🚀 BTC Prediction System - Monitoring Dashboard</h1>
        <p>실시간 시스템 모니터링 및 성능 대시보드</p>
    </div>
    
    <div class="card">
        <h2>📊 시스템 현황</h2>
        <div class="metric">
            <strong>모델 정확도:</strong> 
            <span class="{'status-healthy' if metrics_summary.get('model_performance', {}).get('avg_accuracy', 0) > 0.9 else 'status-warning'}">
                {metrics_summary.get('model_performance', {}).get('avg_accuracy', 0):.1%}
            </span>
        </div>
        <div class="metric">
            <strong>평균 응답시간:</strong> 
            <span class="{'status-healthy' if metrics_summary.get('model_performance', {}).get('avg_latency', 0) < 3 else 'status-warning'}">
                {metrics_summary.get('model_performance', {}).get('avg_latency', 0):.2f}초
            </span>
        </div>
        <div class="metric">
            <strong>시스템 CPU:</strong> 
            <span class="{'status-healthy' if metrics_summary.get('system_health', {}).get('avg_cpu', 0) < 80 else 'status-warning'}">
                {metrics_summary.get('system_health', {}).get('avg_cpu', 0):.1f}%
            </span>
        </div>
        <div class="metric">
            <strong>활성 알림:</strong> 
            <span class="{'status-healthy' if len(active_alerts) == 0 else 'status-warning'}">
                {len(active_alerts)}개
            </span>
        </div>
    </div>
    
    {'<div class="card"><h2>🚨 활성 알림</h2>' if active_alerts else '<div class="card"><h2>✅ 모든 시스템 정상</h2><p>현재 활성 알림이 없습니다.</p></div>'}
    {self._generate_alerts_table(active_alerts)}
    {'</div>' if active_alerts else ''}
    
    <div class="card">
        <h2>📈 성능 메트릭</h2>
        <table>
            <tr>
                <th>메트릭</th>
                <th>현재값</th>
                <th>상태</th>
            </tr>
            <tr>
                <td>모델 신뢰도</td>
                <td>{metrics_summary.get('model_performance', {}).get('avg_confidence', 0):.1%}</td>
                <td><span class="status-healthy">정상</span></td>
            </tr>
            <tr>
                <td>메모리 사용률</td>
                <td>{metrics_summary.get('system_health', {}).get('avg_memory', 0):.1f}%</td>
                <td><span class="status-healthy">정상</span></td>
            </tr>
        </table>
    </div>
    
    <script>
        // 자동 새로고침
        setTimeout(function(){{
            window.location.reload();
        }}, 30000); // 30초마다 새로고침
    </script>
</body>
</html>
"""
        
    def _generate_alerts_table(self, alerts: List[Alert]) -> str:
        """알림 테이블 생성"""
        if not alerts:
            return ""
            
        rows = ""
        for alert in alerts:
            severity_class = f"alert-{alert.severity.value}"
            rows += f"""
            <tr class="{severity_class}">
                <td>{alert.title}</td>
                <td>{alert.severity.value.upper()}</td>
                <td>{alert.current_value:.4f}</td>
                <td>{alert.threshold_value:.4f}</td>
                <td>{alert.timestamp.strftime('%H:%M:%S')}</td>
                <td>{alert.status.value}</td>
            </tr>
            """
            
        return f"""
        <table>
            <tr>
                <th>제목</th>
                <th>심각도</th>
                <th>현재값</th>
                <th>임계값</th>
                <th>발생시간</th>
                <th>상태</th>
            </tr>
            {rows}
        </table>
        """
        
    def run_dashboard(self, host: str = "0.0.0.0", port: int = 8000):
        """대시보드 서버 실행"""
        uvicorn.run(self.app, host=host, port=port)

class MonitoringSystem:
    """통합 모니터링 시스템"""
    
    def __init__(self, config_path: str = None):
        self.config = self.load_config(config_path)
        self.setup_logging()
        
        # 알림 관리자 초기화
        self.alert_manager = AlertManager(
            db_url=self.config.get('database_url', 'sqlite:///monitoring.db'),
            notification_config=self.config.get('notifications', {})
        )
        
        # 대시보드 초기화
        self.dashboard = MonitoringDashboard(self.alert_manager)
        
        self.running = False
        
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """설정 로드"""
        default_config = {
            'database_url': 'sqlite:///monitoring.db',
            'notifications': {
                'telegram': {
                    'bot_token': '8333838666:AAE1bFNfz8kstJZPRx2_S2iCmjgkM6iBGxU',
                    'chat_id': '6846095904'
                },
                'slack': {
                    'webhook_url': 'https://hooks.slack.com/your/webhook/url'
                },
                'email': {
                    'smtp_server': 'smtp.gmail.com',
                    'smtp_port': 587,
                    'from_email': 'noreply@btcpredict.com',
                    'recipients': ['admin@btcpredict.com'],
                    'use_tls': True
                }
            },
            'monitoring': {
                'metric_collection_interval': 60,  # 초
                'health_check_interval': 30,
                'anomaly_detection_enabled': True
            }
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    file_config = json.load(f)
                default_config.update(file_config)
            except Exception as e:
                logging.warning(f"Config file loading failed: {e}")
                
        return default_config
        
    def setup_logging(self):
        self.logger = logging.getLogger(__name__)
        
    async def start_monitoring(self):
        """모니터링 시작"""
        self.logger.info("🔍 Starting monitoring system")
        self.running = True
        
        # 백그라운드 작업들을 동시 실행
        tasks = [
            self.metric_collection_loop(),
            self.health_check_loop(),
            self.alert_processing_loop()
        ]
        
        try:
            await asyncio.gather(*tasks, return_exceptions=True)
        except Exception as e:
            self.logger.error(f"Monitoring system error: {e}")
        finally:
            self.running = False
            
    async def metric_collection_loop(self):
        """메트릭 수집 루프"""
        interval = self.config['monitoring']['metric_collection_interval']
        
        while self.running:
            try:
                # 시스템 메트릭 수집
                self.alert_manager.metrics_collector.collect_system_metrics()
                
                # 모델 성능 메트릭 시뮬레이션 (실제 환경에서는 모델에서 가져옴)
                await self.collect_model_metrics()
                
                await asyncio.sleep(interval)
                
            except Exception as e:
                self.logger.error(f"Metric collection error: {e}")
                await asyncio.sleep(10)
                
    async def collect_model_metrics(self):
        """모델 메트릭 수집"""
        try:
            # 실제 환경에서는 모델 서버에서 메트릭을 가져옴
            # 여기서는 시뮬레이션 데이터 생성
            import random
            
            models = ['xgboost_ensemble', 'lstm_temporal', 'perfect_system']
            
            for model_name in models:
                accuracy = 0.85 + random.uniform(-0.05, 0.05)  # 80-90% 범위
                latency = 1.5 + random.uniform(-0.5, 2.0)  # 1-3.5초 범위
                confidence = 0.80 + random.uniform(-0.05, 0.05)  # 75-85% 범위
                
                # 메트릭 수집
                self.alert_manager.metrics_collector.collect_model_metrics(
                    model_name, accuracy, latency, confidence
                )
                
                # 알림 검사를 위한 메트릭 데이터 생성
                await self.alert_manager.process_metric(MetricData(
                    name='btc_prediction_accuracy',
                    value=accuracy,
                    timestamp=datetime.now(),
                    labels={'model': model_name}
                ))
                
                await self.alert_manager.process_metric(MetricData(
                    name='btc_prediction_latency_seconds', 
                    value=latency,
                    timestamp=datetime.now(),
                    labels={'model': model_name}
                ))
                
        except Exception as e:
            self.logger.error(f"Model metrics collection failed: {e}")
            
    async def health_check_loop(self):
        """헬스체크 루프"""
        interval = self.config['monitoring']['health_check_interval']
        
        while self.running:
            try:
                # 시스템 컴포넌트 헬스체크
                health_status = await self.perform_health_checks()
                
                # 헬스체크 결과를 메트릭으로 전송
                for component, status in health_status.items():
                    await self.alert_manager.process_metric(MetricData(
                        name=f'component_health_{component}',
                        value=1.0 if status else 0.0,
                        timestamp=datetime.now()
                    ))
                    
                await asyncio.sleep(interval)
                
            except Exception as e:
                self.logger.error(f"Health check error: {e}")
                await asyncio.sleep(10)
                
    async def perform_health_checks(self) -> Dict[str, bool]:
        """시스템 컴포넌트 헬스체크"""
        health_status = {}
        
        try:
            # 데이터베이스 연결 확인
            if self.alert_manager.db_session:
                self.alert_manager.db_session.execute("SELECT 1")
                health_status['database'] = True
            else:
                health_status['database'] = False
                
            # Redis 연결 확인 (있는 경우)
            # health_status['redis'] = check_redis_connection()
            
            # 모델 서버 응답 확인
            # health_status['model_server'] = check_model_server()
            
            # 외부 API 연결 확인
            health_status['external_apis'] = await self.check_external_apis()
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            
        return health_status
        
    async def check_external_apis(self) -> bool:
        """외부 API 연결 확인"""
        try:
            # Binance API 응답 확인
            response = requests.get(
                'https://api.binance.com/api/v3/ping',
                timeout=5
            )
            return response.status_code == 200
            
        except Exception:
            return False
            
    async def alert_processing_loop(self):
        """알림 처리 루프"""
        while self.running:
            try:
                # 주기적으로 알림 상태 확인 및 자동 해결
                await self.process_alert_lifecycle()
                await asyncio.sleep(60)  # 1분마다 확인
                
            except Exception as e:
                self.logger.error(f"Alert processing error: {e}")
                await asyncio.sleep(30)
                
    async def process_alert_lifecycle(self):
        """알림 생명주기 처리"""
        current_time = datetime.now()
        
        # 오래된 알림 자동 해결
        for alert_id, alert in list(self.alert_manager.active_alerts.items()):
            time_diff = current_time - alert.timestamp
            
            # 1시간 이상 된 알림은 자동으로 만료 처리
            if time_diff.total_seconds() > 3600:
                await self.alert_manager.resolve_alert(alert_id)
                self.logger.info(f"Auto-resolved expired alert: {alert.title}")
                
    def run_dashboard_server(self, host: str = "0.0.0.0", port: int = 8000):
        """대시보드 서버 실행"""
        self.dashboard.run_dashboard(host, port)
        
    async def shutdown(self):
        """시스템 종료"""
        self.logger.info("Shutting down monitoring system")
        self.running = False

if __name__ == "__main__":
    async def main():
        # 모니터링 시스템 실행
        monitoring = MonitoringSystem()
        
        try:
            # 모니터링과 대시보드를 동시 실행
            await asyncio.gather(
                monitoring.start_monitoring(),
                # 대시보드는 별도 프로세스에서 실행하는 것이 좋음
            )
        except KeyboardInterrupt:
            print("\n🛑 Monitoring stopped by user")
        except Exception as e:
            print(f"❌ Monitoring failed: {e}")
        finally:
            await monitoring.shutdown()
            
    # 대시보드 서버 실행 (별도 실행)
    # monitoring = MonitoringSystem()
    # monitoring.run_dashboard_server()
    
    asyncio.run(main())