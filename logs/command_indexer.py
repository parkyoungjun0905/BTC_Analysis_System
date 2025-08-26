#!/usr/bin/env python3
"""
사용자 명령어 색인 및 복구 시스템
- 모든 사용자 명령을 시간순으로 색인
- 특정 시점의 상태로 복구 가능한 백업 시스템
- 날짜별 로그 관리 및 검색 기능
"""

import os
import json
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import hashlib
import shutil

class CommandIndexer:
    def __init__(self, base_path: str = None):
        self.base_path = base_path or "/Users/parkyoungjun/Desktop/BTC_Analysis_System"
        self.logs_path = os.path.join(self.base_path, "logs")
        self.backup_path = os.path.join(self.base_path, "backups")
        self.db_path = os.path.join(self.logs_path, "command_index.db")
        
        os.makedirs(self.logs_path, exist_ok=True)
        os.makedirs(self.backup_path, exist_ok=True)
        
        self._init_database()
    
    def _init_database(self):
        """명령어 색인 데이터베이스 초기화"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 사용자 명령 테이블
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_commands (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                date TEXT NOT NULL,
                user_command TEXT NOT NULL,
                claude_response_summary TEXT,
                files_affected TEXT,  -- JSON array
                backup_id TEXT,
                status TEXT DEFAULT 'executed',  -- executed, failed, restored
                session_id TEXT,
                command_hash TEXT UNIQUE
            )
        ''')
        
        # 파일 상태 스냅샷 테이블
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS file_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                backup_id TEXT NOT NULL,
                file_path TEXT NOT NULL,
                file_size INTEGER,
                file_hash TEXT,
                backup_file_path TEXT
            )
        ''')
        
        # 복구 포인트 테이블
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS restore_points (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                date TEXT NOT NULL,
                description TEXT NOT NULL,
                backup_id TEXT UNIQUE,
                files_count INTEGER,
                total_size INTEGER
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def index_user_command(self, user_command: str, files_affected: List[str] = None) -> str:
        """사용자 명령어 색인 및 백업 생성"""
        timestamp = datetime.now().isoformat()
        date = datetime.now().strftime('%Y-%m-%d')
        
        # 명령어 해시 생성 (중복 방지)
        command_hash = hashlib.md5(f"{timestamp}_{user_command}".encode()).hexdigest()[:12]
        backup_id = f"backup_{timestamp.replace(':', '').replace('-', '').replace('.', '')}"
        
        # 현재 상태 백업 생성
        self._create_system_backup(backup_id, files_affected or [])
        
        # 데이터베이스에 기록
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT INTO user_commands 
                (timestamp, date, user_command, files_affected, backup_id, command_hash)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (timestamp, date, user_command, json.dumps(files_affected or []), backup_id, command_hash))
            
            # 복구 포인트 생성
            cursor.execute('''
                INSERT INTO restore_points
                (timestamp, date, description, backup_id, files_count)
                VALUES (?, ?, ?, ?, ?)
            ''', (timestamp, date, f"사용자 명령: {user_command[:50]}...", backup_id, len(files_affected or [])))
            
            conn.commit()
            print(f"📝 명령어 색인 완료: {command_hash}")
            return backup_id
            
        except sqlite3.IntegrityError:
            print(f"⚠️ 중복 명령어 감지: {command_hash}")
            return None
        finally:
            conn.close()
    
    def _create_system_backup(self, backup_id: str, files_affected: List[str]):
        """시스템 상태 백업 생성"""
        backup_dir = os.path.join(self.backup_path, backup_id)
        os.makedirs(backup_dir, exist_ok=True)
        
        # 주요 파일들 백업
        important_files = [
            "enhanced_data_collector.py",
            "timeseries_accumulator.py", 
            "collection_tracking.json",
            "CLAUDE.md"
        ]
        
        all_files = list(set(important_files + (files_affected or [])))
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for file_path in all_files:
            try:
                full_path = os.path.join(self.base_path, file_path) if not file_path.startswith('/') else file_path
                
                if os.path.exists(full_path):
                    # 파일 정보 수집
                    file_size = os.path.getsize(full_path)
                    file_hash = self._get_file_hash(full_path)
                    
                    # 백업 파일 경로
                    backup_file_path = os.path.join(backup_dir, os.path.basename(file_path))
                    shutil.copy2(full_path, backup_file_path)
                    
                    # 데이터베이스에 기록
                    cursor.execute('''
                        INSERT INTO file_snapshots
                        (timestamp, backup_id, file_path, file_size, file_hash, backup_file_path)
                        VALUES (?, ?, ?, ?, ?, ?)
                    ''', (datetime.now().isoformat(), backup_id, file_path, file_size, file_hash, backup_file_path))
                    
            except Exception as e:
                print(f"⚠️ 백업 실패 - {file_path}: {e}")
        
        conn.commit()
        conn.close()
        
        # 백업 요약 파일 생성
        summary = {
            "backup_id": backup_id,
            "timestamp": datetime.now().isoformat(),
            "files_count": len(all_files),
            "files": all_files
        }
        
        with open(os.path.join(backup_dir, "backup_summary.json"), 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
    
    def _get_file_hash(self, file_path: str) -> str:
        """파일 해시 계산"""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except:
            return "unknown"
    
    def search_commands(self, query: str = None, date: str = None, limit: int = 20) -> List[Dict]:
        """명령어 검색"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        sql = "SELECT * FROM user_commands WHERE 1=1"
        params = []
        
        if query:
            sql += " AND user_command LIKE ?"
            params.append(f"%{query}%")
        
        if date:
            sql += " AND date = ?"
            params.append(date)
        
        sql += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        cursor.execute(sql, params)
        results = cursor.fetchall()
        conn.close()
        
        # 결과를 딕셔너리로 변환
        columns = ['id', 'timestamp', 'date', 'user_command', 'claude_response_summary', 
                  'files_affected', 'backup_id', 'status', 'session_id', 'command_hash']
        
        return [dict(zip(columns, row)) for row in results]
    
    def restore_to_point(self, backup_id: str, confirm: bool = False) -> bool:
        """특정 시점으로 복구"""
        if not confirm:
            print("⚠️ 복구는 현재 파일들을 덮어씁니다. confirm=True로 다시 호출하세요.")
            return False
        
        backup_dir = os.path.join(self.backup_path, backup_id)
        
        if not os.path.exists(backup_dir):
            print(f"❌ 백업을 찾을 수 없습니다: {backup_id}")
            return False
        
        # 복구 전 현재 상태 백업
        emergency_backup = f"emergency_before_restore_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self._create_system_backup(emergency_backup, [])
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 해당 백업의 파일 목록 조회
        cursor.execute('SELECT * FROM file_snapshots WHERE backup_id = ?', (backup_id,))
        snapshots = cursor.fetchall()
        
        restored_files = 0
        for snapshot in snapshots:
            try:
                _, _, backup_file_path, file_path = snapshot[0], snapshot[1], snapshot[-1], snapshot[3]
                
                if os.path.exists(backup_file_path):
                    # 원본 위치로 복구
                    target_path = os.path.join(self.base_path, file_path) if not file_path.startswith('/') else file_path
                    os.makedirs(os.path.dirname(target_path), exist_ok=True)
                    shutil.copy2(backup_file_path, target_path)
                    restored_files += 1
                    print(f"✅ 복구: {file_path}")
                
            except Exception as e:
                print(f"❌ 복구 실패 - {file_path}: {e}")
        
        conn.close()
        
        print(f"🎯 복구 완료: {restored_files}개 파일")
        print(f"💾 비상 백업: {emergency_backup}")
        
        return True
    
    def get_restore_points(self, days: int = 7) -> List[Dict]:
        """복구 가능한 시점 목록"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        since_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        cursor.execute('''
            SELECT r.*, COUNT(f.id) as files_count, SUM(f.file_size) as total_size
            FROM restore_points r
            LEFT JOIN file_snapshots f ON r.backup_id = f.backup_id
            WHERE r.date >= ?
            GROUP BY r.backup_id
            ORDER BY r.timestamp DESC
        ''', (since_date,))
        
        results = cursor.fetchall()
        conn.close()
        
        columns = ['id', 'timestamp', 'date', 'description', 'backup_id', 
                  'files_count_db', 'total_size_db', 'files_count', 'total_size']
        
        return [dict(zip(columns, row)) for row in results]
    
    def generate_daily_summary(self, date: str = None) -> Dict:
        """일일 명령어 요약"""
        if not date:
            date = datetime.now().strftime('%Y-%m-%d')
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 해당 날짜 명령어 통계
        cursor.execute('''
            SELECT COUNT(*) as total_commands,
                   COUNT(CASE WHEN status = 'executed' THEN 1 END) as successful,
                   COUNT(CASE WHEN status = 'failed' THEN 1 END) as failed,
                   COUNT(DISTINCT backup_id) as backups_created
            FROM user_commands WHERE date = ?
        ''', (date,))
        
        stats = cursor.fetchone()
        
        # 자주 사용된 명령어
        cursor.execute('''
            SELECT user_command, COUNT(*) as frequency
            FROM user_commands 
            WHERE date = ? 
            GROUP BY user_command 
            ORDER BY frequency DESC LIMIT 5
        ''', (date,))
        
        frequent_commands = cursor.fetchall()
        conn.close()
        
        return {
            "date": date,
            "statistics": {
                "total_commands": stats[0],
                "successful": stats[1], 
                "failed": stats[2],
                "backups_created": stats[3]
            },
            "frequent_commands": [{"command": cmd, "frequency": freq} for cmd, freq in frequent_commands]
        }

# 글로벌 인덱서 인스턴스
command_indexer = CommandIndexer()

# 편의 함수들
def index_command(user_command: str, files_affected: List[str] = None) -> str:
    """사용자 명령어 색인"""
    return command_indexer.index_user_command(user_command, files_affected)

def search_commands(query: str = None, date: str = None) -> List[Dict]:
    """명령어 검색"""
    return command_indexer.search_commands(query, date)

def restore_to_point(backup_id: str, confirm: bool = False) -> bool:
    """복구 실행"""
    return command_indexer.restore_to_point(backup_id, confirm)

def get_restore_points(days: int = 7) -> List[Dict]:
    """복구 포인트 조회"""
    return command_indexer.get_restore_points(days)

def daily_summary(date: str = None) -> Dict:
    """일일 요약"""
    return command_indexer.generate_daily_summary(date)

if __name__ == "__main__":
    # 테스트
    backup_id = index_command("테스트 명령어", ["test.py"])
    print(f"백업 ID: {backup_id}")
    
    # 검색 테스트
    results = search_commands("테스트")
    print(f"검색 결과: {len(results)}개")
    
    # 복구 포인트 조회
    points = get_restore_points()
    print(f"복구 가능한 시점: {len(points)}개")
    
    print("명령어 색인 시스템 테스트 완료")