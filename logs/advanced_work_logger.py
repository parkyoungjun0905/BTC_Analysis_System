#!/usr/bin/env python3
"""
고도화된 작업 로깅 시스템
- 사용자 명령어 자동 색인
- 시간대별 작업 복구 지원
- 날짜별 로그 파일 관리
- 실시간 상태 추적
"""

import os
import json
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from command_indexer import CommandIndexer, command_indexer

class AdvancedWorkLogger:
    def __init__(self, base_path: str = None):
        self.base_path = base_path or "/Users/parkyoungjun/Desktop/BTC_Analysis_System"
        self.logs_path = os.path.join(self.base_path, "logs")
        os.makedirs(self.logs_path, exist_ok=True)
        
        # 명령어 색인 시스템 연동
        self.command_indexer = command_indexer
        
        self.today = datetime.now().strftime("%Y-%m-%d")
        self.current_session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # 날짜별 로그 파일
        self.md_log_file = os.path.join(self.logs_path, f"claude_work_log_{self.today}.md")
        self.json_log_file = os.path.join(self.logs_path, f"claude_work_log_{self.today}.json") 
        self.command_log_file = os.path.join(self.logs_path, f"user_commands_{self.today}.json")
        
        # 로그 파일 초기화
        self._init_daily_logs()
    
    def _init_daily_logs(self):
        """일일 로그 파일 초기화"""
        # JSON 로그 초기화
        if not os.path.exists(self.json_log_file):
            initial_data = {
                "date": self.today,
                "created_at": datetime.now().isoformat(),
                "session_id": self.current_session_id,
                "work_sessions": [],
                "file_modifications": [],
                "user_commands": [],
                "errors": [],
                "completions": [],
                "system_snapshots": []
            }
            
            with open(self.json_log_file, 'w', encoding='utf-8') as f:
                json.dump(initial_data, f, ensure_ascii=False, indent=2)
        
        # 마크다운 로그 헤더 추가 (파일이 없을 때만)
        if not os.path.exists(self.md_log_file):
            header = f"""# 🤖 Claude 작업 로그 - {self.today}

## 📋 세션 정보
- **날짜**: {self.today}
- **세션 ID**: `{self.current_session_id}`
- **시작 시간**: {datetime.now().strftime('%H:%M:%S')}

---

"""
            with open(self.md_log_file, 'w', encoding='utf-8') as f:
                f.write(header)
        
        # 사용자 명령어 로그 초기화
        if not os.path.exists(self.command_log_file):
            command_data = {
                "date": self.today,
                "commands": [],
                "summary": {
                    "total_commands": 0,
                    "unique_commands": 0,
                    "most_frequent": None
                }
            }
            
            with open(self.command_log_file, 'w', encoding='utf-8') as f:
                json.dump(command_data, f, ensure_ascii=False, indent=2)
    
    def log_user_command(self, command: str, context: str = "", files_mentioned: List[str] = None) -> str:
        """사용자 명령어 로그 및 색인"""
        timestamp = datetime.now().isoformat()
        
        # 명령어 색인 시스템에 등록 (백업 생성 포함)
        backup_id = self.command_indexer.index_user_command(command, files_mentioned or [])
        
        # 로그 엔트리 생성
        command_entry = {
            "timestamp": timestamp,
            "command": command,
            "context": context,
            "files_mentioned": files_mentioned or [],
            "backup_id": backup_id,
            "session_id": self.current_session_id
        }
        
        # JSON 로그 업데이트
        self._update_json_log("user_commands", command_entry)
        
        # 사용자 명령어 전용 로그 업데이트
        self._update_command_log(command_entry)
        
        # 마크다운 로그 업데이트
        md_entry = f"""
## 👤 {timestamp[:19]} - 사용자 명령
### 📝 명령어
```
{command}
```
### 📋 상황
{context or '상황 정보 없음'}

### 📁 관련 파일
{', '.join(files_mentioned) if files_mentioned else '없음'}

### 💾 백업 ID
`{backup_id}`

---
"""
        self._append_to_md(md_entry)
        
        print(f"📝 사용자 명령 기록: {command[:50]}...")
        return backup_id
    
    def log_claude_response(self, user_command: str, response_summary: str, actions_taken: List[str] = None, files_modified: List[str] = None):
        """클로드 응답 로그"""
        timestamp = datetime.now().isoformat()
        
        response_entry = {
            "timestamp": timestamp,
            "user_command": user_command,
            "response_summary": response_summary,
            "actions_taken": actions_taken or [],
            "files_modified": files_modified or [],
            "session_id": self.current_session_id
        }
        
        # JSON 로그 업데이트
        self._update_json_log("claude_responses", response_entry)
        
        # 마크다운 로그 업데이트
        actions_str = '\n'.join([f"- {action}" for action in actions_taken]) if actions_taken else "없음"
        files_str = '\n'.join([f"- `{file}`" for file in files_modified]) if files_modified else "없음"
        
        md_entry = f"""
## 🤖 {timestamp[:19]} - Claude 응답
### 📥 사용자 명령
```
{user_command}
```

### 📤 응답 요약
{response_summary}

### ⚡ 수행 작업
{actions_str}

### 📁 수정된 파일
{files_str}

---
"""
        self._append_to_md(md_entry)
    
    def create_system_snapshot(self, description: str = "정기 스냅샷") -> str:
        """시스템 상태 스냅샷 생성"""
        timestamp = datetime.now().isoformat()
        snapshot_id = f"snapshot_{timestamp.replace(':', '').replace('-', '').replace('.', '')}"
        
        # 중요 파일들의 현재 상태 백업
        important_files = [
            "enhanced_data_collector.py",
            "timeseries_accumulator.py",
            "collection_tracking.json",
            "CLAUDE.md"
        ]
        
        backup_id = self.command_indexer._create_system_backup(snapshot_id, important_files)
        
        snapshot_entry = {
            "timestamp": timestamp,
            "snapshot_id": snapshot_id,
            "description": description,
            "files_count": len(important_files),
            "backup_id": backup_id
        }
        
        self._update_json_log("system_snapshots", snapshot_entry)
        
        md_entry = f"""
## 📸 {timestamp[:19]} - 시스템 스냅샷
### 📋 설명
{description}

### 🆔 스냅샷 ID
`{snapshot_id}`

### 📁 백업된 파일
{len(important_files)}개 파일

---
"""
        self._append_to_md(md_entry)
        
        print(f"📸 시스템 스냅샷 생성: {snapshot_id}")
        return snapshot_id
    
    def find_command_by_keyword(self, keyword: str, days: int = 7) -> List[Dict]:
        """키워드로 명령어 검색"""
        results = []
        
        # 최근 N일간의 로그 파일 검색
        for i in range(days):
            search_date = (datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d')
            command_file = os.path.join(self.logs_path, f"user_commands_{search_date}.json")
            
            if os.path.exists(command_file):
                try:
                    with open(command_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    for cmd in data.get("commands", []):
                        if keyword.lower() in cmd.get("command", "").lower():
                            results.append({
                                "date": search_date,
                                "timestamp": cmd.get("timestamp"),
                                "command": cmd.get("command"),
                                "backup_id": cmd.get("backup_id"),
                                "files_mentioned": cmd.get("files_mentioned", [])
                            })
                except Exception as e:
                    print(f"⚠️ 로그 파일 읽기 오류 ({search_date}): {e}")
        
        return sorted(results, key=lambda x: x['timestamp'], reverse=True)
    
    def restore_to_command(self, backup_id: str, confirm: bool = False) -> bool:
        """특정 명령어 시점으로 복구"""
        print(f"🔄 복구 시작: {backup_id}")
        
        if not confirm:
            print("⚠️ 이 작업은 현재 파일들을 덮어씁니다.")
            print("복구를 진행하려면 confirm=True로 다시 호출하세요.")
            return False
        
        # 복구 전 현재 상태 백업
        emergency_backup = self.create_system_snapshot("복구 전 비상 백업")
        
        # 명령어 색인 시스템을 통한 복구
        success = self.command_indexer.restore_to_point(backup_id, confirm=True)
        
        if success:
            restore_entry = {
                "timestamp": datetime.now().isoformat(),
                "backup_id": backup_id,
                "emergency_backup_id": emergency_backup,
                "status": "success"
            }
            
            self._update_json_log("restorations", restore_entry)
            
            md_entry = f"""
## 🔄 {datetime.now().strftime('%H:%M:%S')} - 시스템 복구
### 🆔 복구 포인트
`{backup_id}`

### 💾 비상 백업
`{emergency_backup}`

### ✅ 상태
복구 성공

---
"""
            self._append_to_md(md_entry)
            
            print(f"✅ 복구 완료: {backup_id}")
            print(f"💾 비상 백업: {emergency_backup}")
        
        return success
    
    def get_daily_command_summary(self, date: str = None) -> Dict:
        """일일 명령어 요약"""
        if not date:
            date = self.today
        
        command_file = os.path.join(self.logs_path, f"user_commands_{date}.json")
        
        if not os.path.exists(command_file):
            return {"date": date, "error": "로그 파일 없음"}
        
        try:
            with open(command_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            commands = data.get("commands", [])
            
            # 통계 계산
            total_commands = len(commands)
            unique_commands = len(set(cmd.get("command") for cmd in commands))
            
            # 가장 빈번한 명령어
            command_freq = {}
            for cmd in commands:
                command_text = cmd.get("command", "")
                command_freq[command_text] = command_freq.get(command_text, 0) + 1
            
            most_frequent = max(command_freq.items(), key=lambda x: x[1]) if command_freq else None
            
            return {
                "date": date,
                "total_commands": total_commands,
                "unique_commands": unique_commands,
                "most_frequent_command": most_frequent[0] if most_frequent else None,
                "most_frequent_count": most_frequent[1] if most_frequent else 0,
                "command_frequency": dict(sorted(command_freq.items(), key=lambda x: x[1], reverse=True)[:10])
            }
            
        except Exception as e:
            return {"date": date, "error": str(e)}
    
    def _update_command_log(self, command_entry: Dict):
        """사용자 명령어 전용 로그 업데이트"""
        try:
            with open(self.command_log_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            data["commands"].append(command_entry)
            
            # 요약 정보 업데이트
            commands = data["commands"]
            data["summary"]["total_commands"] = len(commands)
            data["summary"]["unique_commands"] = len(set(cmd["command"] for cmd in commands))
            
            # 가장 빈번한 명령어 계산
            command_freq = {}
            for cmd in commands:
                command_freq[cmd["command"]] = command_freq.get(cmd["command"], 0) + 1
            
            if command_freq:
                most_frequent = max(command_freq.items(), key=lambda x: x[1])
                data["summary"]["most_frequent"] = {
                    "command": most_frequent[0],
                    "count": most_frequent[1]
                }
            
            with open(self.command_log_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            print(f"⚠️ 명령어 로그 업데이트 오류: {e}")
    
    def _update_json_log(self, section: str, entry: Dict):
        """JSON 로그 업데이트"""
        try:
            with open(self.json_log_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if section not in data:
                data[section] = []
            
            data[section].append(entry)
            data["last_updated"] = datetime.now().isoformat()
            
            with open(self.json_log_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            print(f"⚠️ JSON 로그 업데이트 오류: {e}")
    
    def _append_to_md(self, content: str):
        """마크다운 로그에 내용 추가"""
        try:
            with open(self.md_log_file, 'a', encoding='utf-8') as f:
                f.write(content)
        except Exception as e:
            print(f"⚠️ 마크다운 로그 업데이트 오류: {e}")

# 글로벌 로거 인스턴스
advanced_logger = AdvancedWorkLogger()

# 편의 함수들
def log_user_command(command: str, context: str = "", files: List[str] = None) -> str:
    """사용자 명령어 기록"""
    return advanced_logger.log_user_command(command, context, files)

def log_claude_response(user_cmd: str, summary: str, actions: List[str] = None, files: List[str] = None):
    """클로드 응답 기록"""
    advanced_logger.log_claude_response(user_cmd, summary, actions, files)

def create_snapshot(description: str = "정기 스냅샷") -> str:
    """시스템 스냅샷 생성"""
    return advanced_logger.create_system_snapshot(description)

def find_command(keyword: str, days: int = 7) -> List[Dict]:
    """명령어 검색"""
    return advanced_logger.find_command_by_keyword(keyword, days)

def restore_to_command(backup_id: str, confirm: bool = False) -> bool:
    """명령어 시점으로 복구"""
    return advanced_logger.restore_to_command(backup_id, confirm)

def daily_command_summary(date: str = None) -> Dict:
    """일일 명령어 요약"""
    return advanced_logger.get_daily_command_summary(date)

if __name__ == "__main__":
    # 테스트
    backup_id = log_user_command("테스트 명령어", "시스템 테스트", ["test.py"])
    log_claude_response("테스트 명령어", "테스트 응답", ["파일 생성"], ["test.py"])
    
    snapshot_id = create_snapshot("테스트 스냅샷")
    
    commands = find_command("테스트")
    print(f"검색된 명령어: {len(commands)}개")
    
    summary = daily_command_summary()
    print(f"오늘의 명령어 요약: {summary}")
    
    print("고도화된 로깅 시스템 테스트 완료")