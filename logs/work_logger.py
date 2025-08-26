#!/usr/bin/env python3
"""
클로드 작업 로그 자동 기록 시스템
모든 작업 내역을 실시간으로 기록하여 투명성과 추적성 확보
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Any, Optional

class WorkLogger:
    def __init__(self, base_path: str = None):
        self.base_path = base_path or os.path.dirname(os.path.abspath(__file__))
        self.log_dir = os.path.join(os.path.dirname(self.base_path), "logs")
        os.makedirs(self.log_dir, exist_ok=True)
        
        self.today = datetime.now().strftime("%Y-%m-%d")
        self.log_file = os.path.join(self.log_dir, f"claude_work_log_{self.today}.md")
        self.json_log = os.path.join(self.log_dir, f"claude_work_log_{self.today}.json")
        
        # JSON 로그 구조 초기화
        if not os.path.exists(self.json_log):
            self._init_json_log()
    
    def _init_json_log(self):
        """JSON 로그 파일 초기화"""
        initial_data = {
            "date": self.today,
            "created_at": datetime.now().isoformat(),
            "work_sessions": [],
            "file_modifications": [],
            "errors": [],
            "completions": []
        }
        
        with open(self.json_log, 'w', encoding='utf-8') as f:
            json.dump(initial_data, f, ensure_ascii=False, indent=2)
    
    def log_work_start(self, task_description: str, files_involved: List[str] = None) -> str:
        """작업 시작 로그"""
        timestamp = datetime.now().isoformat()
        session_id = f"work_{timestamp.replace(':', '').replace('-', '').replace('.', '')}"
        
        work_entry = {
            "session_id": session_id,
            "start_time": timestamp,
            "task": task_description,
            "files_involved": files_involved or [],
            "status": "started",
            "checkpoints": [],
            "end_time": None,
            "result": "in_progress"
        }
        
        # JSON 로그 업데이트
        self._update_json_log("work_sessions", work_entry)
        
        # 마크다운 로그 업데이트
        md_entry = f"""
## 🚀 {timestamp[:16]} - 작업 시작
### 작업 ID: `{session_id}`
### 작업 내용: {task_description}
### 관련 파일: {', '.join(files_involved) if files_involved else '없음'}
### 상태: 🔄 진행중

---
"""
        self._append_to_md(md_entry)
        
        return session_id
    
    def log_checkpoint(self, session_id: str, checkpoint_desc: str, status: str = "progress"):
        """작업 중간 체크포인트 로그"""
        timestamp = datetime.now().isoformat()
        
        checkpoint = {
            "time": timestamp,
            "description": checkpoint_desc,
            "status": status
        }
        
        # JSON 업데이트 - 해당 세션에 체크포인트 추가
        self._add_checkpoint_to_session(session_id, checkpoint)
        
        # 마크다운 업데이트
        status_icon = "✅" if status == "success" else "❌" if status == "error" else "🔄"
        md_entry = f"""
### {status_icon} {timestamp[:19]} - 체크포인트
- **세션**: `{session_id}`
- **내용**: {checkpoint_desc}
- **상태**: {status}

"""
        self._append_to_md(md_entry)
    
    def log_file_modification(self, file_path: str, operation: str, backup_created: bool = False):
        """파일 수정 로그"""
        timestamp = datetime.now().isoformat()
        
        mod_entry = {
            "timestamp": timestamp,
            "file_path": file_path,
            "operation": operation,  # create, modify, delete, backup
            "backup_created": backup_created,
            "file_size": os.path.getsize(file_path) if os.path.exists(file_path) else 0
        }
        
        self._update_json_log("file_modifications", mod_entry)
        
        # 마크다운 로그
        op_icon = {"create": "📝", "modify": "✏️", "delete": "🗑️", "backup": "💾"}.get(operation, "📄")
        md_entry = f"""
### {op_icon} {timestamp[:19]} - 파일 {operation}
- **파일**: `{file_path}`
- **백업**: {'✅ 생성됨' if backup_created else '❌ 없음'}
- **크기**: {mod_entry['file_size']:,} bytes

"""
        self._append_to_md(md_entry)
    
    def log_error(self, session_id: str, error_desc: str, error_details: str = None):
        """오류 로그"""
        timestamp = datetime.now().isoformat()
        
        error_entry = {
            "timestamp": timestamp,
            "session_id": session_id,
            "error": error_desc,
            "details": error_details,
            "severity": "high"
        }
        
        self._update_json_log("errors", error_entry)
        
        # 마크다운 로그
        md_entry = f"""
### ❌ {timestamp[:19]} - 오류 발생
- **세션**: `{session_id}`
- **오류**: {error_desc}
- **상세**: {error_details or '상세 정보 없음'}
- **심각도**: 🔴 HIGH

"""
        self._append_to_md(md_entry)
    
    def log_work_complete(self, session_id: str, result: str, verification_passed: bool = True):
        """작업 완료 로그"""
        timestamp = datetime.now().isoformat()
        
        # 해당 세션 업데이트
        self._complete_session(session_id, timestamp, result, verification_passed)
        
        completion_entry = {
            "timestamp": timestamp,
            "session_id": session_id,
            "result": result,
            "verification_passed": verification_passed,
            "quality": "high" if verification_passed else "low"
        }
        
        self._update_json_log("completions", completion_entry)
        
        # 마크다운 로그
        result_icon = "✅" if verification_passed else "⚠️"
        md_entry = f"""
### {result_icon} {timestamp[:19]} - 작업 완료
- **세션**: `{session_id}`
- **결과**: {result}
- **검증**: {'✅ 통과' if verification_passed else '❌ 실패'}
- **품질**: {'🟢 HIGH' if verification_passed else '🟡 LOW'}

---
"""
        self._append_to_md(md_entry)
    
    def _update_json_log(self, section: str, data: Dict):
        """JSON 로그 업데이트"""
        try:
            with open(self.json_log, 'r', encoding='utf-8') as f:
                log_data = json.load(f)
            
            log_data[section].append(data)
            log_data["last_updated"] = datetime.now().isoformat()
            
            with open(self.json_log, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"JSON 로그 업데이트 오류: {e}")
    
    def _add_checkpoint_to_session(self, session_id: str, checkpoint: Dict):
        """세션에 체크포인트 추가"""
        try:
            with open(self.json_log, 'r', encoding='utf-8') as f:
                log_data = json.load(f)
            
            for session in log_data["work_sessions"]:
                if session["session_id"] == session_id:
                    session["checkpoints"].append(checkpoint)
                    break
            
            log_data["last_updated"] = datetime.now().isoformat()
            
            with open(self.json_log, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"체크포인트 추가 오류: {e}")
    
    def _complete_session(self, session_id: str, end_time: str, result: str, verified: bool):
        """세션 완료 처리"""
        try:
            with open(self.json_log, 'r', encoding='utf-8') as f:
                log_data = json.load(f)
            
            for session in log_data["work_sessions"]:
                if session["session_id"] == session_id:
                    session["end_time"] = end_time
                    session["status"] = "completed"
                    session["result"] = result
                    session["verification_passed"] = verified
                    break
            
            log_data["last_updated"] = datetime.now().isoformat()
            
            with open(self.json_log, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"세션 완료 처리 오류: {e}")
    
    def _append_to_md(self, content: str):
        """마크다운 파일에 내용 추가"""
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(content)
        except Exception as e:
            print(f"마크다운 로그 업데이트 오류: {e}")
    
    def get_daily_summary(self) -> Dict:
        """일일 작업 요약"""
        try:
            with open(self.json_log, 'r', encoding='utf-8') as f:
                log_data = json.load(f)
            
            total_sessions = len(log_data["work_sessions"])
            completed_sessions = len([s for s in log_data["work_sessions"] if s["status"] == "completed"])
            total_errors = len(log_data["errors"])
            files_modified = len(log_data["file_modifications"])
            
            return {
                "date": self.today,
                "total_work_sessions": total_sessions,
                "completed_sessions": completed_sessions,
                "completion_rate": completed_sessions / total_sessions if total_sessions > 0 else 0,
                "total_errors": total_errors,
                "files_modified": files_modified,
                "last_activity": log_data.get("last_updated", "N/A")
            }
        except Exception as e:
            return {"error": str(e)}

# 글로벌 로거 인스턴스
work_logger = WorkLogger()

# 편의 함수들
def start_work(task: str, files: List[str] = None) -> str:
    """작업 시작"""
    return work_logger.log_work_start(task, files)

def checkpoint(session_id: str, desc: str, status: str = "progress"):
    """체크포인트"""
    work_logger.log_checkpoint(session_id, desc, status)

def file_modified(path: str, operation: str, backup: bool = False):
    """파일 수정"""
    work_logger.log_file_modification(path, operation, backup)

def log_error(session_id: str, error: str, details: str = None):
    """오류 기록"""
    work_logger.log_error(session_id, error, details)

def complete_work(session_id: str, result: str, verified: bool = True):
    """작업 완료"""
    work_logger.log_work_complete(session_id, result, verified)

def daily_summary() -> Dict:
    """일일 요약"""
    return work_logger.get_daily_summary()

if __name__ == "__main__":
    # 테스트
    session = start_work("테스트 작업", ["test.py"])
    checkpoint(session, "파일 읽기 완료", "success")
    file_modified("test.py", "modify", True)
    complete_work(session, "테스트 성공적 완료", True)
    print("로그 시스템 테스트 완료")