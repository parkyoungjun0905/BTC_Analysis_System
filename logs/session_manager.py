#!/usr/bin/env python3
"""
멀티 클로드 세션 충돌 방지 시스템
세션별 로그 파일 자동 생성 및 관리
"""

import os
import time
import json
import fcntl
from datetime import datetime
from pathlib import Path

class SessionManager:
    def __init__(self):
        self.session_id = f"SESSION_{int(time.time())}_{os.getpid()}"
        self.log_dir = Path(__file__).parent
        self.session_log = self.log_dir / f"claude_session_{self.session_id}.md"
        self.lock_file = self.log_dir / "session_lock.txt"
        
        # 세션 정보 등록
        self.register_session()
        
    def register_session(self):
        """현재 활성 세션 등록"""
        sessions_file = self.log_dir / "active_sessions.json"
        
        # 파일 잠금으로 안전하게 업데이트
        with open(sessions_file, 'a+') as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            f.seek(0)
            try:
                sessions = json.load(f) if f.read().strip() else {}
            except:
                sessions = {}
            
            sessions[self.session_id] = {
                "start_time": datetime.now().isoformat(),
                "log_file": str(self.session_log),
                "status": "active"
            }
            
            f.seek(0)
            f.truncate()
            json.dump(sessions, f, indent=2, ensure_ascii=False)
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            
        print(f"🔒 세션 등록 완료: {self.session_id}")
        
    def safe_log_write(self, content):
        """안전한 로그 쓰기 (파일 잠금 사용)"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        with open(self.session_log, 'a', encoding='utf-8') as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            f.write(f"\n[{timestamp}] {content}\n")
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            
    def check_other_sessions(self):
        """다른 활성 세션 확인"""
        sessions_file = self.log_dir / "active_sessions.json"
        
        if not sessions_file.exists():
            return []
            
        with open(sessions_file, 'r') as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_SH)
            sessions = json.load(f)
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            
        other_sessions = [sid for sid in sessions if sid != self.session_id]
        return other_sessions
        
    def warn_about_conflicts(self):
        """다른 세션과 충돌 경고"""
        others = self.check_other_sessions()
        if others:
            print(f"⚠️ 경고: {len(others)}개의 다른 클로드 세션이 활성화됨!")
            print("📋 권장사항: 작업 완료 후 다른 세션들을 종료하세요.")
            return True
        return False

# 세션 관리자 초기화
if __name__ == "__main__":
    session_mgr = SessionManager()
    session_mgr.warn_about_conflicts()
    
    # 현재 세션 정보 출력
    print(f"🎯 현재 세션: {session_mgr.session_id}")
    print(f"📝 로그 파일: {session_mgr.session_log}")