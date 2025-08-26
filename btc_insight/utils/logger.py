#!/usr/bin/env python3
"""
📝 로깅 시스템
- BTC Insight 프로그램 전용 로거
- 학습 과정 추적 및 디버깅 지원
"""

import logging
import os
from datetime import datetime
from pathlib import Path

def setup_logger(name: str = 'btc_insight') -> logging.Logger:
    """
    메인 로거 설정
    
    Args:
        name: 로거 이름
        
    Returns:
        설정된 로거 객체
    """
    # 로그 디렉토리 생성
    log_dir = Path(__file__).parent.parent / 'logs'
    log_dir.mkdir(exist_ok=True)
    
    # 로그 파일 이름 (날짜별)
    today = datetime.now().strftime("%Y%m%d")
    log_file = log_dir / f"btc_insight_{today}.log"
    
    # 로거 생성
    logger = logging.getLogger(name)
    
    # 이미 핸들러가 설정되어 있다면 스킵
    if logger.handlers:
        return logger
        
    logger.setLevel(logging.INFO)
    
    # 포맷터 설정
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 파일 핸들러
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # 콘솔 핸들러 (개발용)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger

def get_logger(name: str = None) -> logging.Logger:
    """
    로거 가져오기
    
    Args:
        name: 로거 이름 (보통 __name__ 사용)
        
    Returns:
        로거 객체
    """
    if name:
        return logging.getLogger(f'btc_insight.{name}')
    else:
        return logging.getLogger('btc_insight')

class LearningLogger:
    """백테스트 학습 전용 로거"""
    
    def __init__(self):
        self.logger = get_logger('learning')
        
    def log_backtest_start(self, iteration: int, start_idx: int, prediction_hours: int):
        """백테스트 시작 로그"""
        self.logger.info(f"백테스트 시작 - 반복: {iteration}, 시점: {start_idx}, 예측시간: {prediction_hours}h")
        
    def log_backtest_result(self, iteration: int, accuracy: float, error_pct: float):
        """백테스트 결과 로그"""
        self.logger.info(f"백테스트 결과 - 반복: {iteration}, 정확도: {accuracy:.2f}%, 오차: {error_pct:.2f}%")
        
    def log_learning_progress(self, current_accuracy: float, target_accuracy: float, iterations: int):
        """학습 진행 상황 로그"""
        self.logger.info(f"학습 진행 - 현재 정확도: {current_accuracy:.2f}%, 목표: {target_accuracy:.2f}%, 반복: {iterations}")
        
    def log_failure_analysis(self, error_pct: float, shock_events: int, main_indicators: list):
        """실패 분석 로그"""
        indicators_str = ', '.join(main_indicators[:3]) if main_indicators else 'None'
        self.logger.info(f"실패 분석 - 오차: {error_pct:.2f}%, 돌발사건: {shock_events}, 주요지표: {indicators_str}")
        
    def log_pattern_learned(self, pattern: str, occurrences: int):
        """학습된 패턴 로그"""
        self.logger.info(f"패턴 학습 - {pattern}: {occurrences}회 발생")

class PerformanceLogger:
    """성능 모니터링 로거"""
    
    def __init__(self):
        self.logger = get_logger('performance')
        
    def log_execution_time(self, operation: str, duration: float):
        """실행 시간 로그"""
        self.logger.info(f"실행시간 - {operation}: {duration:.2f}초")
        
    def log_memory_usage(self, operation: str, memory_mb: float):
        """메모리 사용량 로그"""
        self.logger.info(f"메모리 사용 - {operation}: {memory_mb:.2f}MB")
        
    def log_data_processing(self, rows: int, columns: int, processing_time: float):
        """데이터 처리 성능 로그"""
        self.logger.info(f"데이터 처리 - {rows}행 x {columns}열, {processing_time:.2f}초")