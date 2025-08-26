#!/usr/bin/env python3
"""
🔒 보안 및 컴플라이언스 시스템
- 다중 계층 보안 (인증, 인가, 암호화)
- 데이터 보호 및 개인정보 처리
- 감사 로그 및 컴플라이언스 추적
- 위험 평가 및 보안 모니터링
- 법적 요구사항 준수 (GDPR, SOX, ISO27001)
"""

import asyncio
import json
import logging
import time
import hashlib
import hmac
import secrets
import base64
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import os
import re
from enum import Enum
import ipaddress
from urllib.parse import urlparse

# 암호화 및 보안
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
import jwt
from passlib.context import CryptContext
from passlib.hash import bcrypt

# 웹 보안
from fastapi import FastAPI, HTTPException, Depends, Request, Response
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, OAuth2PasswordBearer
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
import aioredis
from starlette.middleware.sessions import SessionMiddleware
from starlette.middleware.httpsredirect import HTTPSRedirectMiddleware

# 데이터베이스 보안
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, Boolean, LargeBinary
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy_utils import EncryptedType
from sqlalchemy_utils.types.encrypted.encrypted_type import AesEngine

# 모니터링 및 감사
import structlog
from prometheus_client import Counter, Histogram, Gauge

Base = declarative_base()

class SecurityLevel(Enum):
    """보안 레벨"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AccessType(Enum):
    """접근 유형"""
    READ = "read"
    WRITE = "write"
    ADMIN = "admin"
    SYSTEM = "system"

class ComplianceFramework(Enum):
    """컴플라이언스 프레임워크"""
    GDPR = "gdpr"
    SOX = "sox"
    ISO27001 = "iso27001"
    PCI_DSS = "pci_dss"

@dataclass
class SecurityEvent:
    """보안 이벤트"""
    event_id: str
    event_type: str
    severity: SecurityLevel
    user_id: Optional[str]
    ip_address: str
    user_agent: Optional[str]
    endpoint: Optional[str]
    success: bool
    timestamp: datetime
    details: Dict[str, Any]

@dataclass
class AccessAttempt:
    """접근 시도"""
    ip_address: str
    user_id: Optional[str]
    endpoint: str
    method: str
    success: bool
    timestamp: datetime
    response_time: float

class User(Base):
    """사용자 테이블"""
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True)
    username = Column(String(50), unique=True, nullable=False)
    email = Column(String(100), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    salt = Column(String(32), nullable=False)
    role = Column(String(20), default='user')
    is_active = Column(Boolean, default=True)
    failed_login_attempts = Column(Integer, default=0)
    locked_until = Column(DateTime)
    last_login = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class AuditLog(Base):
    """감사 로그 테이블"""
    __tablename__ = 'audit_logs'
    
    id = Column(Integer, primary_key=True)
    event_id = Column(String(64), unique=True, nullable=False)
    event_type = Column(String(50), nullable=False)
    severity = Column(String(20), nullable=False)
    user_id = Column(String(50))
    ip_address = Column(String(45))
    user_agent = Column(Text)
    endpoint = Column(String(255))
    method = Column(String(10))
    success = Column(Boolean, nullable=False)
    details = Column(Text)  # JSON 형태
    timestamp = Column(DateTime, default=datetime.utcnow)

class EncryptedData(Base):
    """암호화된 데이터 테이블"""
    __tablename__ = 'encrypted_data'
    
    id = Column(Integer, primary_key=True)
    data_type = Column(String(50), nullable=False)
    key_id = Column(String(64), nullable=False)
    encrypted_content = Column(LargeBinary, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    accessed_at = Column(DateTime)

class SecurityManager:
    """보안 관리자"""
    
    def __init__(self, config_path: str = None):
        self.config = self.load_security_config(config_path)
        self.setup_logging()
        self.setup_database()
        self.setup_encryption()
        self.setup_metrics()
        
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        self.oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
        
        # 보안 이벤트 큐
        self.security_events = asyncio.Queue(maxsize=10000)
        self.failed_attempts = {}  # IP별 실패 시도 추적
        self.blacklisted_ips = set()
        self.rate_limits = {}  # IP별 요청 속도 제한
        
        # 세션 관리
        self.active_sessions = {}
        self.session_cleanup_interval = 3600  # 1시간
        
    def load_security_config(self, config_path: str) -> Dict[str, Any]:
        """보안 설정 로드"""
        default_config = {
            "jwt": {
                "secret_key": secrets.token_urlsafe(32),
                "algorithm": "HS256",
                "expire_minutes": 30,
                "refresh_expire_days": 7
            },
            "encryption": {
                "master_key": None,  # 환경변수에서 로드
                "key_rotation_days": 90,
                "algorithm": "AES-256-GCM"
            },
            "rate_limiting": {
                "max_requests_per_minute": 60,
                "max_requests_per_hour": 1000,
                "ban_duration_minutes": 60
            },
            "password_policy": {
                "min_length": 12,
                "require_uppercase": True,
                "require_lowercase": True,
                "require_numbers": True,
                "require_symbols": True,
                "max_failed_attempts": 5,
                "lockout_duration_minutes": 30
            },
            "session": {
                "timeout_minutes": 30,
                "max_concurrent_sessions": 3
            },
            "compliance": {
                "data_retention_days": 2555,  # 7년 (SOX 요구사항)
                "log_retention_days": 2555,
                "encryption_at_rest": True,
                "encryption_in_transit": True
            }
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    file_config = json.load(f)
                default_config.update(file_config)
            except Exception as e:
                logging.warning(f"Security config loading failed: {e}")
                
        return default_config
        
    def setup_logging(self):
        """구조화된 보안 로깅 설정"""
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="ISO"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.JSONRenderer()
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
        
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            handlers=[
                logging.FileHandler("security_audit.log"),
                logging.StreamHandler()
            ]
        )
        
        self.logger = structlog.get_logger(__name__)
        
    def setup_database(self):
        """보안 데이터베이스 설정"""
        try:
            # 보안을 위해 별도 데이터베이스 사용
            db_path = Path("security_data.db")
            self.engine = create_engine(
                f'sqlite:///{db_path}',
                echo=False,  # 보안을 위해 쿼리 로깅 비활성화
                pool_pre_ping=True
            )
            
            Base.metadata.create_all(self.engine)
            
            Session = sessionmaker(bind=self.engine)
            self.db_session = Session()
            
            self.logger.info("Security database setup completed")
            
        except Exception as e:
            self.logger.error("Security database setup failed", error=str(e))
            raise
            
    def setup_encryption(self):
        """암호화 시스템 설정"""
        try:
            # 마스터 키 로드 또는 생성
            master_key = self.config["encryption"].get("master_key")
            if not master_key:
                master_key = os.environ.get("SECURITY_MASTER_KEY")
                
            if not master_key:
                # 개발 환경에서만 자동 생성
                master_key = Fernet.generate_key().decode()
                self.logger.warning("Auto-generated master key for development")
                
            self.master_key = master_key.encode() if isinstance(master_key, str) else master_key
            self.fernet = Fernet(self.master_key)
            
            # RSA 키 쌍 생성 (비대칭 암호화용)
            self.private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048,
            )
            self.public_key = self.private_key.public_key()
            
            self.logger.info("Encryption system initialized")
            
        except Exception as e:
            self.logger.error("Encryption setup failed", error=str(e))
            raise
            
    def setup_metrics(self):
        """보안 메트릭 설정"""
        self.security_events_counter = Counter(
            'security_events_total',
            'Total security events',
            ['event_type', 'severity', 'success']
        )
        
        self.authentication_attempts = Counter(
            'authentication_attempts_total',
            'Total authentication attempts',
            ['method', 'success']
        )
        
        self.failed_logins_by_ip = Counter(
            'failed_logins_by_ip_total',
            'Failed login attempts by IP',
            ['ip_address']
        )
        
        self.active_sessions_gauge = Gauge(
            'active_sessions_count',
            'Number of active user sessions'
        )
        
    async def start_security_monitoring(self):
        """보안 모니터링 시작"""
        self.logger.info("🔒 Starting security monitoring")
        
        try:
            await asyncio.gather(
                self.security_event_processor(),
                self.session_cleanup_loop(),
                self.security_scanner_loop(),
                self.compliance_checker_loop(),
                return_exceptions=True
            )
            
        except Exception as e:
            self.logger.error("Security monitoring failed", error=str(e))
            raise
            
    async def security_event_processor(self):
        """보안 이벤트 처리기"""
        while True:
            try:
                # 보안 이벤트 큐에서 이벤트 처리
                event = await self.security_events.get()
                await self.process_security_event(event)
                
            except Exception as e:
                self.logger.error("Security event processing failed", error=str(e))
                await asyncio.sleep(1)
                
    async def process_security_event(self, event: SecurityEvent):
        """보안 이벤트 처리"""
        try:
            # 메트릭 업데이트
            self.security_events_counter.labels(
                event_type=event.event_type,
                severity=event.severity.value,
                success=str(event.success)
            ).inc()
            
            # 감사 로그에 기록
            await self.log_audit_event(event)
            
            # 위협 탐지 및 대응
            await self.threat_detection(event)
            
            # 실시간 알림 (심각한 보안 이벤트의 경우)
            if event.severity in [SecurityLevel.HIGH, SecurityLevel.CRITICAL]:
                await self.send_security_alert(event)
                
        except Exception as e:
            self.logger.error("Security event processing failed", 
                            event_id=event.event_id, error=str(e))
            
    async def log_audit_event(self, event: SecurityEvent):
        """감사 로그 기록"""
        try:
            audit_entry = AuditLog(
                event_id=event.event_id,
                event_type=event.event_type,
                severity=event.severity.value,
                user_id=event.user_id,
                ip_address=event.ip_address,
                user_agent=event.user_agent,
                endpoint=event.endpoint,
                success=event.success,
                details=json.dumps(event.details, default=str),
                timestamp=event.timestamp
            )
            
            self.db_session.add(audit_entry)
            self.db_session.commit()
            
        except Exception as e:
            self.logger.error("Audit logging failed", error=str(e))
            if self.db_session:
                self.db_session.rollback()
                
    async def threat_detection(self, event: SecurityEvent):
        """위협 탐지 및 대응"""
        try:
            # 1. 무차별 대입 공격 탐지
            if event.event_type == "login_failed":
                await self.detect_brute_force_attack(event)
                
            # 2. 비정상적인 API 요청 패턴 탐지
            if event.event_type == "api_request":
                await self.detect_abnormal_api_usage(event)
                
            # 3. 권한 상승 시도 탐지
            if event.event_type == "permission_escalation":
                await self.detect_privilege_escalation(event)
                
            # 4. 데이터 유출 시도 탐지
            if event.event_type == "data_access":
                await self.detect_data_exfiltration(event)
                
        except Exception as e:
            self.logger.error("Threat detection failed", error=str(e))
            
    async def detect_brute_force_attack(self, event: SecurityEvent):
        """무차별 대입 공격 탐지"""
        try:
            ip_address = event.ip_address
            
            # IP별 실패 시도 카운트
            if ip_address not in self.failed_attempts:
                self.failed_attempts[ip_address] = []
                
            self.failed_attempts[ip_address].append(event.timestamp)
            
            # 최근 10분간의 실패 시도만 유지
            cutoff_time = datetime.now() - timedelta(minutes=10)
            self.failed_attempts[ip_address] = [
                ts for ts in self.failed_attempts[ip_address] 
                if ts > cutoff_time
            ]
            
            # 임계값 초과 시 IP 차단
            if len(self.failed_attempts[ip_address]) > 10:
                await self.block_ip_address(ip_address, "brute_force_attack")
                
                # 심각한 보안 이벤트 생성
                security_event = SecurityEvent(
                    event_id=secrets.token_hex(16),
                    event_type="brute_force_detected",
                    severity=SecurityLevel.CRITICAL,
                    user_id=None,
                    ip_address=ip_address,
                    user_agent=event.user_agent,
                    endpoint=event.endpoint,
                    success=False,
                    timestamp=datetime.now(),
                    details={"failed_attempts": len(self.failed_attempts[ip_address])}
                )
                
                await self.security_events.put(security_event)
                
        except Exception as e:
            self.logger.error("Brute force detection failed", error=str(e))
            
    async def detect_abnormal_api_usage(self, event: SecurityEvent):
        """비정상적인 API 사용 패턴 탐지"""
        try:
            ip_address = event.ip_address
            current_time = datetime.now()
            
            # IP별 요청 속도 추적
            if ip_address not in self.rate_limits:
                self.rate_limits[ip_address] = []
                
            self.rate_limits[ip_address].append(current_time)
            
            # 최근 1분간의 요청만 유지
            cutoff_time = current_time - timedelta(minutes=1)
            self.rate_limits[ip_address] = [
                ts for ts in self.rate_limits[ip_address] 
                if ts > cutoff_time
            ]
            
            # 분당 요청 수 제한
            max_requests_per_minute = self.config["rate_limiting"]["max_requests_per_minute"]
            if len(self.rate_limits[ip_address]) > max_requests_per_minute:
                await self.rate_limit_ip(ip_address, "excessive_api_requests")
                
        except Exception as e:
            self.logger.error("API usage detection failed", error=str(e))
            
    async def block_ip_address(self, ip_address: str, reason: str):
        """IP 주소 차단"""
        try:
            self.blacklisted_ips.add(ip_address)
            
            self.logger.warning("IP address blocked", 
                              ip_address=ip_address, reason=reason)
                              
            # 차단 해제 스케줄링 (1시간 후)
            ban_duration = self.config["rate_limiting"]["ban_duration_minutes"]
            await asyncio.create_task(
                self.schedule_ip_unblock(ip_address, ban_duration)
            )
            
        except Exception as e:
            self.logger.error("IP blocking failed", error=str(e))
            
    async def schedule_ip_unblock(self, ip_address: str, minutes: int):
        """IP 차단 해제 스케줄링"""
        await asyncio.sleep(minutes * 60)
        
        try:
            if ip_address in self.blacklisted_ips:
                self.blacklisted_ips.remove(ip_address)
                self.logger.info("IP address unblocked", ip_address=ip_address)
                
        except Exception as e:
            self.logger.error("IP unblocking failed", error=str(e))
            
    def create_access_token(self, data: dict, expires_delta: Optional[timedelta] = None):
        """액세스 토큰 생성"""
        try:
            to_encode = data.copy()
            
            if expires_delta:
                expire = datetime.utcnow() + expires_delta
            else:
                expire = datetime.utcnow() + timedelta(
                    minutes=self.config["jwt"]["expire_minutes"]
                )
                
            to_encode.update({"exp": expire})
            to_encode.update({"iat": datetime.utcnow()})
            to_encode.update({"jti": secrets.token_hex(16)})  # JWT ID
            
            encoded_jwt = jwt.encode(
                to_encode, 
                self.config["jwt"]["secret_key"], 
                algorithm=self.config["jwt"]["algorithm"]
            )
            
            return encoded_jwt
            
        except Exception as e:
            self.logger.error("Token creation failed", error=str(e))
            raise HTTPException(status_code=500, detail="Token creation failed")
            
    def verify_access_token(self, token: str) -> Dict[str, Any]:
        """액세스 토큰 검증"""
        try:
            payload = jwt.decode(
                token, 
                self.config["jwt"]["secret_key"], 
                algorithms=[self.config["jwt"]["algorithm"]]
            )
            
            # 토큰 만료 확인
            exp_timestamp = payload.get("exp")
            if exp_timestamp and datetime.utcnow().timestamp() > exp_timestamp:
                raise HTTPException(status_code=401, detail="Token expired")
                
            # 세션 유효성 확인
            jti = payload.get("jti")
            if jti and jti not in self.active_sessions:
                raise HTTPException(status_code=401, detail="Session invalid")
                
            return payload
            
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token expired")
        except jwt.JWTError as e:
            raise HTTPException(status_code=401, detail="Token validation failed")
            
    def hash_password(self, password: str) -> Tuple[str, str]:
        """비밀번호 해시화"""
        try:
            # 솔트 생성
            salt = secrets.token_hex(16)
            
            # 비밀번호 정책 검증
            self.validate_password_policy(password)
            
            # 해시화
            password_hash = self.pwd_context.hash(password + salt)
            
            return password_hash, salt
            
        except Exception as e:
            self.logger.error("Password hashing failed", error=str(e))
            raise
            
    def verify_password(self, plain_password: str, hashed_password: str, salt: str) -> bool:
        """비밀번호 검증"""
        try:
            return self.pwd_context.verify(plain_password + salt, hashed_password)
        except Exception as e:
            self.logger.error("Password verification failed", error=str(e))
            return False
            
    def validate_password_policy(self, password: str) -> bool:
        """비밀번호 정책 검증"""
        policy = self.config["password_policy"]
        
        # 최소 길이
        if len(password) < policy["min_length"]:
            raise ValueError(f"Password must be at least {policy['min_length']} characters")
            
        # 대문자 포함
        if policy["require_uppercase"] and not re.search(r'[A-Z]', password):
            raise ValueError("Password must contain uppercase letters")
            
        # 소문자 포함
        if policy["require_lowercase"] and not re.search(r'[a-z]', password):
            raise ValueError("Password must contain lowercase letters")
            
        # 숫자 포함
        if policy["require_numbers"] and not re.search(r'\d', password):
            raise ValueError("Password must contain numbers")
            
        # 특수문자 포함
        if policy["require_symbols"] and not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
            raise ValueError("Password must contain symbols")
            
        # 일반적인 비밀번호 금지
        common_passwords = [
            "password", "123456", "qwerty", "admin", "letmein",
            "welcome", "monkey", "dragon", "bitcoin", "crypto"
        ]
        
        if password.lower() in common_passwords:
            raise ValueError("Password is too common")
            
        return True
        
    def encrypt_sensitive_data(self, data: str) -> Tuple[str, str]:
        """민감 데이터 암호화"""
        try:
            # 데이터 암호화
            encrypted_data = self.fernet.encrypt(data.encode())
            
            # 키 ID 생성 (키 회전 추적용)
            key_id = hashlib.sha256(self.master_key).hexdigest()[:16]
            
            return base64.b64encode(encrypted_data).decode(), key_id
            
        except Exception as e:
            self.logger.error("Data encryption failed", error=str(e))
            raise
            
    def decrypt_sensitive_data(self, encrypted_data: str, key_id: str) -> str:
        """민감 데이터 복호화"""
        try:
            # 키 ID 검증
            current_key_id = hashlib.sha256(self.master_key).hexdigest()[:16]
            if key_id != current_key_id:
                raise ValueError("Key ID mismatch - key rotation may be required")
                
            # 데이터 복호화
            encrypted_bytes = base64.b64decode(encrypted_data.encode())
            decrypted_data = self.fernet.decrypt(encrypted_bytes)
            
            return decrypted_data.decode()
            
        except Exception as e:
            self.logger.error("Data decryption failed", error=str(e))
            raise
            
    async def create_user_session(self, user_id: str, ip_address: str, user_agent: str) -> str:
        """사용자 세션 생성"""
        try:
            # 기존 세션 수 확인
            user_sessions = [
                sid for sid, session_info in self.active_sessions.items() 
                if session_info['user_id'] == user_id
            ]
            
            max_sessions = self.config["session"]["max_concurrent_sessions"]
            if len(user_sessions) >= max_sessions:
                # 가장 오래된 세션 제거
                oldest_session = min(user_sessions, 
                                   key=lambda sid: self.active_sessions[sid]['created_at'])
                await self.invalidate_session(oldest_session)
                
            # 새 세션 생성
            session_id = secrets.token_hex(32)
            
            self.active_sessions[session_id] = {
                'user_id': user_id,
                'ip_address': ip_address,
                'user_agent': user_agent,
                'created_at': datetime.now(),
                'last_activity': datetime.now()
            }
            
            # 메트릭 업데이트
            self.active_sessions_gauge.set(len(self.active_sessions))
            
            return session_id
            
        except Exception as e:
            self.logger.error("Session creation failed", error=str(e))
            raise
            
    async def invalidate_session(self, session_id: str):
        """세션 무효화"""
        try:
            if session_id in self.active_sessions:
                del self.active_sessions[session_id]
                self.active_sessions_gauge.set(len(self.active_sessions))
                
        except Exception as e:
            self.logger.error("Session invalidation failed", error=str(e))
            
    async def session_cleanup_loop(self):
        """세션 정리 루프"""
        while True:
            try:
                await asyncio.sleep(self.session_cleanup_interval)
                await self.cleanup_expired_sessions()
                
            except Exception as e:
                self.logger.error("Session cleanup failed", error=str(e))
                await asyncio.sleep(300)  # 5분 후 재시도
                
    async def cleanup_expired_sessions(self):
        """만료된 세션 정리"""
        try:
            current_time = datetime.now()
            timeout_minutes = self.config["session"]["timeout_minutes"]
            
            expired_sessions = []
            
            for session_id, session_info in self.active_sessions.items():
                last_activity = session_info['last_activity']
                if (current_time - last_activity).total_seconds() > (timeout_minutes * 60):
                    expired_sessions.append(session_id)
                    
            for session_id in expired_sessions:
                await self.invalidate_session(session_id)
                
            if expired_sessions:
                self.logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
                
        except Exception as e:
            self.logger.error("Session cleanup failed", error=str(e))
            
    async def security_scanner_loop(self):
        """보안 스캐너 루프"""
        while True:
            try:
                await asyncio.sleep(3600)  # 1시간마다 실행
                
                await self.scan_for_vulnerabilities()
                await self.check_system_integrity()
                await self.audit_user_permissions()
                
            except Exception as e:
                self.logger.error("Security scanner failed", error=str(e))
                await asyncio.sleep(1800)  # 30분 후 재시도
                
    async def scan_for_vulnerabilities(self):
        """취약점 스캔"""
        try:
            vulnerabilities = []
            
            # 1. 기본 보안 설정 확인
            if not self.config["compliance"]["encryption_at_rest"]:
                vulnerabilities.append("Encryption at rest not enabled")
                
            if not self.config["compliance"]["encryption_in_transit"]:
                vulnerabilities.append("Encryption in transit not enabled")
                
            # 2. 비밀번호 정책 확인
            if self.config["password_policy"]["min_length"] < 12:
                vulnerabilities.append("Weak password policy")
                
            # 3. 세션 구성 확인
            if self.config["session"]["timeout_minutes"] > 60:
                vulnerabilities.append("Session timeout too long")
                
            # 4. JWT 설정 확인
            if self.config["jwt"]["expire_minutes"] > 60:
                vulnerabilities.append("JWT expiration time too long")
                
            if vulnerabilities:
                security_event = SecurityEvent(
                    event_id=secrets.token_hex(16),
                    event_type="vulnerabilities_detected",
                    severity=SecurityLevel.MEDIUM,
                    user_id=None,
                    ip_address="system",
                    user_agent=None,
                    endpoint=None,
                    success=False,
                    timestamp=datetime.now(),
                    details={"vulnerabilities": vulnerabilities}
                )
                
                await self.security_events.put(security_event)
                
        except Exception as e:
            self.logger.error("Vulnerability scan failed", error=str(e))
            
    async def compliance_checker_loop(self):
        """컴플라이언스 확인 루프"""
        while True:
            try:
                await asyncio.sleep(86400)  # 24시간마다 실행
                
                await self.check_gdpr_compliance()
                await self.check_sox_compliance()
                await self.check_iso27001_compliance()
                
            except Exception as e:
                self.logger.error("Compliance check failed", error=str(e))
                await asyncio.sleep(3600)  # 1시간 후 재시도
                
    async def check_gdpr_compliance(self):
        """GDPR 컴플라이언스 확인"""
        try:
            issues = []
            
            # 데이터 보존 기간 확인
            retention_days = self.config["compliance"]["data_retention_days"]
            if retention_days <= 0:
                issues.append("Data retention policy not defined")
                
            # 암호화 확인
            if not self.config["compliance"]["encryption_at_rest"]:
                issues.append("Personal data not encrypted at rest")
                
            # 감사 로그 확인
            log_retention = self.config["compliance"]["log_retention_days"]
            if log_retention < 2555:  # 7년
                issues.append("Insufficient log retention for GDPR")
                
            if issues:
                self.logger.warning("GDPR compliance issues detected", issues=issues)
                
        except Exception as e:
            self.logger.error("GDPR compliance check failed", error=str(e))
            
    async def send_security_alert(self, event: SecurityEvent):
        """보안 알림 전송"""
        try:
            # 텔레그램으로 보안 알림 전송 (기존 알림 시스템 활용)
            import requests
            
            bot_token = "8333838666:AAE1bFNfz8kstJZPRx2_S2iCmjgkM6iBGxU"
            chat_id = "6846095904"
            
            message = f"""
🚨 *SECURITY ALERT* 🚨

🔒 **이벤트**: {event.event_type}
⚠️ **심각도**: {event.severity.value.upper()}
📍 **IP**: `{event.ip_address}`
🕒 **시간**: `{event.timestamp.strftime('%Y-%m-%d %H:%M:%S')}`
✅ **성공여부**: {'성공' if event.success else '실패'}

📋 **상세정보**: {json.dumps(event.details, indent=2)}
"""
            
            url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
            payload = {
                'chat_id': chat_id,
                'text': message,
                'parse_mode': 'Markdown'
            }
            
            response = requests.post(url, json=payload, timeout=10)
            
            if response.status_code == 200:
                self.logger.info("Security alert sent successfully")
            else:
                self.logger.error("Security alert sending failed")
                
        except Exception as e:
            self.logger.error("Security alert failed", error=str(e))

class SecurityMiddleware:
    """보안 미들웨어"""
    
    def __init__(self, security_manager: SecurityManager):
        self.security_manager = security_manager
        
    async def __call__(self, request: Request, call_next):
        """요청 보안 검사"""
        start_time = time.time()
        
        try:
            # IP 차단 확인
            client_ip = self.get_client_ip(request)
            if client_ip in self.security_manager.blacklisted_ips:
                return JSONResponse(
                    status_code=403,
                    content={"detail": "IP address blocked"}
                )
                
            # 속도 제한 확인
            if not await self.check_rate_limit(client_ip):
                return JSONResponse(
                    status_code=429,
                    content={"detail": "Rate limit exceeded"}
                )
                
            # 요청 헤더 보안 검사
            security_headers = self.validate_security_headers(request)
            
            # 요청 처리
            response = await call_next(request)
            
            # 보안 헤더 추가
            self.add_security_headers(response)
            
            # 보안 이벤트 기록
            await self.log_request_event(request, response, start_time)
            
            return response
            
        except Exception as e:
            # 보안 오류 이벤트 생성
            security_event = SecurityEvent(
                event_id=secrets.token_hex(16),
                event_type="middleware_error",
                severity=SecurityLevel.HIGH,
                user_id=None,
                ip_address=self.get_client_ip(request),
                user_agent=request.headers.get("User-Agent"),
                endpoint=str(request.url),
                success=False,
                timestamp=datetime.now(),
                details={"error": str(e)}
            )
            
            await self.security_manager.security_events.put(security_event)
            
            return JSONResponse(
                status_code=500,
                content={"detail": "Internal security error"}
            )
            
    def get_client_ip(self, request: Request) -> str:
        """클라이언트 IP 주소 획득"""
        # X-Forwarded-For 헤더 확인 (프록시 환경)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
            
        # X-Real-IP 헤더 확인
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
            
        # 직접 연결
        return request.client.host if request.client else "unknown"
        
    def add_security_headers(self, response: Response):
        """보안 헤더 추가"""
        security_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Content-Security-Policy": "default-src 'self'",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Permissions-Policy": "geolocation=(), microphone=(), camera=()"
        }
        
        for header, value in security_headers.items():
            response.headers[header] = value

def create_secure_app(security_manager: SecurityManager) -> FastAPI:
    """보안이 강화된 FastAPI 앱 생성"""
    
    app = FastAPI(
        title="Secure BTC Prediction API",
        description="보안이 강화된 비트코인 예측 API",
        version="2.0.0",
        docs_url=None,  # 프로덕션에서는 문서 비활성화
        redoc_url=None
    )
    
    # 보안 미들웨어 추가
    app.add_middleware(SecurityMiddleware, security_manager=security_manager)
    
    # HTTPS 리다이렉트
    app.add_middleware(HTTPSRedirectMiddleware)
    
    # 신뢰할 수 있는 호스트만 허용
    app.add_middleware(
        TrustedHostMiddleware, 
        allowed_hosts=["localhost", "127.0.0.1", "*.yourdomain.com"]
    )
    
    # CORS 설정 (매우 제한적)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["https://yourdomain.com"],
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["Authorization", "Content-Type"],
        max_age=600
    )
    
    # 세션 미들웨어
    app.add_middleware(
        SessionMiddleware, 
        secret_key=security_manager.config["jwt"]["secret_key"]
    )
    
    return app

if __name__ == "__main__":
    async def main():
        # 보안 시스템 실행
        security_manager = SecurityManager()
        
        try:
            await security_manager.start_security_monitoring()
        except KeyboardInterrupt:
            print("\n🛑 Security system stopped by user")
        except Exception as e:
            print(f"❌ Security system failed: {e}")
            
    asyncio.run(main())