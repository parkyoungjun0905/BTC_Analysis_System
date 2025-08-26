#!/usr/bin/env python3
"""
🔧 Git 자동화 로그 시스템 (초보자용)
- 모든 작업을 자동으로 Git에 저장해주는 도구
- 언제든 이전 상태로 되돌릴 수 있어요
"""

import os
import sys
import json
import subprocess
from datetime import datetime
import argparse

class AutoGitLogger:
    def __init__(self):
        # 기본 설정
        self.work_counter_file = "work_counter.json"  # 작업 번호를 저장하는 파일
        self.log_file = "coding_log.md"  # 작업 로그를 저장하는 파일
        self.backup_branch = "backup"  # 백업용 브랜치 이름
        
    def init_project(self):
        """프로젝트 초기 설정 - Git 저장소와 로그 파일을 만들어요"""
        print("📁 프로젝트 초기 설정을 시작합니다...")
        
        # Git 저장소가 없으면 만들기
        if not os.path.exists(".git"):
            print("🔧 Git 저장소를 만들고 있어요...")
            subprocess.run(["git", "init"], check=True)
            print("✅ Git 저장소 생성 완료!")
        
        # 작업 카운터 파일 초기화
        if not os.path.exists(self.work_counter_file):
            with open(self.work_counter_file, 'w', encoding='utf-8') as f:
                json.dump({"current_work": 0, "works": {}}, f, ensure_ascii=False, indent=2)
            print("📊 작업 카운터 파일 생성 완료!")
        
        # 로그 파일 초기화
        if not os.path.exists(self.log_file):
            self.create_log_template()
            print("📝 작업 로그 파일 생성 완료!")
        
        # gitignore 설정
        gitignore_content = """
# Python 캐시 파일
__pycache__/
*.pyc
*.pyo

# 개발 도구 파일
.vscode/
.idea/

# 로그 파일 (너무 크면)
*.log

# 민감한 설정 파일
*.env
config.secret
"""
        with open(".gitignore", "w", encoding="utf-8") as f:
            f.write(gitignore_content.strip())
        
        print("🎉 프로젝트 초기 설정이 완료되었습니다!")
        
    def create_log_template(self):
        """작업 로그 템플릿을 만들어요"""
        template = f"""# 📝 코딩 작업 로그

**프로젝트**: {os.path.basename(os.getcwd())}
**시작 날짜**: {datetime.now().strftime('%Y-%m-%d')}

---

## 📋 오늘의 작업 목록

### 완료된 작업 ✅
_(아직 없음)_

### 진행 중인 작업 🔄
_(아직 없음)_

### 대기 중인 작업 ⏳
_(아직 없음)_

---

## 📊 작업 히스토리

<!-- 여기에 작업 내역이 자동으로 추가됩니다 -->

---

## 🔧 되돌리기 명령어

문제가 생겼을 때 사용하세요:

```bash
# 최근 작업으로 되돌리기
python auto_git_logger.py rollback

# 특정 작업 번호로 되돌리기 (예: 작업 5번으로)
python auto_git_logger.py rollback 5

# 오늘 시작 시점으로 되돌리기
python auto_git_logger.py rollback today
```

---

**💡 팁**: 작업을 시작하기 전에 항상 `python auto_git_logger.py start "작업내용"`을 실행하세요!
"""
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write(template)
    
    def get_next_work_number(self):
        """다음 작업 번호를 가져와요"""
        if os.path.exists(self.work_counter_file):
            with open(self.work_counter_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            data = {"current_work": 0, "works": {}}
        
        data["current_work"] += 1
        return data["current_work"], data
    
    def save_work_data(self, data):
        """작업 데이터를 저장해요"""
        with open(self.work_counter_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def start_work(self, description):
        """새로운 작업을 시작해요"""
        work_number, data = self.get_next_work_number()
        timestamp = datetime.now()
        
        print(f"🚀 작업 #{work_number} 시작: {description}")
        
        # 작업 정보 저장
        work_info = {
            "description": description,
            "start_time": timestamp.isoformat(),
            "status": "진행중",
            "commit_hash": None
        }
        data["works"][str(work_number)] = work_info
        self.save_work_data(data)
        
        # Git 커밋 생성 (체크포인트)
        try:
            # 모든 변경사항 추가
            subprocess.run(["git", "add", "."], check=True)
            
            # 커밋 메시지 작성
            commit_message = f"🚀 작업 #{work_number} 시작: {description}\n\n시작 시간: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}"
            subprocess.run(["git", "commit", "-m", commit_message], check=True)
            
            # 커밋 해시 가져오기
            result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
            commit_hash = result.stdout.strip()
            
            # 커밋 해시 저장
            data["works"][str(work_number)]["commit_hash"] = commit_hash
            self.save_work_data(data)
            
            print(f"✅ Git 체크포인트 생성 완료! (커밋: {commit_hash[:8]})")
            
        except subprocess.CalledProcessError as e:
            print(f"⚠️  Git 커밋 생성 중 오류가 발생했어요: {e}")
            print("💡 해결 방법: 먼저 git config로 이름과 이메일을 설정해주세요")
            print("   git config user.name \"당신의 이름\"")
            print("   git config user.email \"당신의 이메일\"")
        
        # 로그 파일 업데이트
        self.update_log_file(work_number, description, "시작", timestamp)
        
        print(f"📝 작업 로그가 {self.log_file}에 기록되었습니다")
        print(f"🔧 문제가 생기면 다음 명령어로 되돌릴 수 있어요:")
        print(f"   python auto_git_logger.py rollback {work_number}")
    
    def complete_work(self, description):
        """현재 작업을 완료해요"""
        with open(self.work_counter_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        current_work = data["current_work"]
        timestamp = datetime.now()
        
        print(f"✅ 작업 #{current_work} 완료: {description}")
        
        # 작업 정보 업데이트
        if str(current_work) in data["works"]:
            data["works"][str(current_work)]["status"] = "완료"
            data["works"][str(current_work)]["end_time"] = timestamp.isoformat()
            data["works"][str(current_work)]["completion_description"] = description
            self.save_work_data(data)
        
        # Git 커밋 생성
        try:
            subprocess.run(["git", "add", "."], check=True)
            commit_message = f"✅ 작업 #{current_work} 완료: {description}\n\n완료 시간: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}"
            subprocess.run(["git", "commit", "-m", commit_message], check=True)
            print("✅ Git 커밋 생성 완료!")
            
        except subprocess.CalledProcessError as e:
            print(f"⚠️  Git 커밋 생성 중 오류가 발생했어요: {e}")
        
        # 로그 파일 업데이트
        self.update_log_file(current_work, description, "완료", timestamp)
        print("🎉 작업 완료! 로그에 기록되었습니다")
    
    def rollback(self, target=None):
        """이전 상태로 되돌려요"""
        with open(self.work_counter_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if target is None:
            # 최근 작업으로 되돌리기
            target = data["current_work"]
        elif target == "today":
            # 오늘 첫 작업으로 되돌리기
            target = 1
        else:
            target = int(target)
        
        if str(target) not in data["works"]:
            print(f"❌ 작업 #{target}을 찾을 수 없어요")
            return
        
        work_info = data["works"][str(target)]
        commit_hash = work_info.get("commit_hash")
        
        if not commit_hash:
            print(f"❌ 작업 #{target}의 커밋 정보가 없어요")
            return
        
        print(f"🔄 작업 #{target}로 되돌리기: {work_info['description']}")
        
        try:
            # 현재 변경사항 임시 저장
            subprocess.run(["git", "stash"], check=False)  # 실패해도 계속 진행
            
            # 지정된 커밋으로 되돌리기
            subprocess.run(["git", "reset", "--hard", commit_hash], check=True)
            
            print(f"✅ 작업 #{target} 상태로 되돌리기 완료!")
            print(f"📝 작업 내용: {work_info['description']}")
            print(f"🕐 시작 시간: {work_info['start_time']}")
            
        except subprocess.CalledProcessError as e:
            print(f"❌ 되돌리기 실패: {e}")
            print("💡 해결 방법: 다음 명령어를 직접 실행해보세요")
            print(f"   git reset --hard {commit_hash}")
    
    def update_log_file(self, work_number, description, action, timestamp):
        """로그 파일을 업데이트해요"""
        log_entry = f"\n### 📝 작업 #{work_number} - {action}\n"
        log_entry += f"- **시간**: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n"
        log_entry += f"- **내용**: {description}\n"
        log_entry += f"- **되돌리기**: `python auto_git_logger.py rollback {work_number}`\n"
        
        # 로그 파일에 추가
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_entry)
    
    def show_status(self):
        """현재 작업 상황을 보여줘요"""
        if not os.path.exists(self.work_counter_file):
            print("📝 아직 작업이 없어요. 첫 작업을 시작해보세요!")
            print("💡 사용법: python auto_git_logger.py start \"작업 내용\"")
            return
        
        with open(self.work_counter_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print("📊 작업 현황")
        print("=" * 50)
        
        total_works = len(data["works"])
        completed_works = sum(1 for w in data["works"].values() if w["status"] == "완료")
        in_progress_works = total_works - completed_works
        
        print(f"📋 총 작업 수: {total_works}")
        print(f"✅ 완료된 작업: {completed_works}")
        print(f"🔄 진행 중인 작업: {in_progress_works}")
        
        if data["works"]:
            print("\n📝 최근 작업 5개:")
            recent_works = list(data["works"].items())[-5:]
            for work_id, work_info in recent_works:
                status_icon = "✅" if work_info["status"] == "완료" else "🔄"
                print(f"  {status_icon} 작업 #{work_id}: {work_info['description']}")
    
    def daily_backup(self):
        """하루 종료시 전체 백업을 만들어요"""
        print("💾 일일 백업을 시작합니다...")
        
        today = datetime.now().strftime('%Y-%m-%d')
        backup_branch_name = f"backup-{today}"
        
        try:
            # 현재 변경사항 커밋
            subprocess.run(["git", "add", "."], check=True)
            subprocess.run(["git", "commit", "-m", f"📦 일일 백업 - {today}"], check=False)
            
            # 백업 브랜치 생성
            subprocess.run(["git", "branch", backup_branch_name], check=False)
            
            print(f"✅ 백업 브랜치 '{backup_branch_name}' 생성 완료!")
            print("💡 이 브랜치로 되돌리려면:")
            print(f"   git checkout {backup_branch_name}")
            
        except subprocess.CalledProcessError as e:
            print(f"⚠️  백업 생성 중 오류: {e}")


def main():
    """메인 함수 - 초보자도 쉽게 사용할 수 있게 만들어요"""
    logger = AutoGitLogger()
    
    parser = argparse.ArgumentParser(
        description="🔧 Git 자동화 로그 시스템 (초보자용)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  python auto_git_logger.py init                    # 프로젝트 초기 설정
  python auto_git_logger.py start "로그인 기능 추가"   # 작업 시작
  python auto_git_logger.py complete "로그인 완료"    # 작업 완료
  python auto_git_logger.py status                  # 현재 상황 보기
  python auto_git_logger.py rollback                # 최근 작업으로 되돌리기
  python auto_git_logger.py rollback 3              # 작업 3번으로 되돌리기
  python auto_git_logger.py backup                  # 일일 백업
        """
    )
    
    parser.add_argument('action', choices=['init', 'start', 'complete', 'rollback', 'status', 'backup'],
                       help='실행할 작업을 선택하세요')
    parser.add_argument('description', nargs='?', 
                       help='작업 설명 (start, complete에서 필요)')
    parser.add_argument('target', nargs='?',
                       help='되돌릴 작업 번호 (rollback에서 선택적)')
    
    # 인자가 없으면 도움말 출력
    if len(sys.argv) == 1:
        print("🔧 Git 자동화 로그 시스템에 오신 것을 환영합니다!")
        print("=" * 50)
        print("💡 사용법:")
        print("  python auto_git_logger.py init                    # 처음 설정")
        print("  python auto_git_logger.py start \"작업 내용\"        # 작업 시작")
        print("  python auto_git_logger.py complete \"완료 내용\"     # 작업 완료")
        print("  python auto_git_logger.py status                  # 상황 확인")
        print("  python auto_git_logger.py rollback                # 되돌리기")
        print("")
        print("❓ 더 자세한 도움말을 보려면: python auto_git_logger.py --help")
        return
    
    args = parser.parse_args()
    
    try:
        if args.action == 'init':
            logger.init_project()
        
        elif args.action == 'start':
            if not args.description:
                print("❌ 작업 설명을 입력해주세요!")
                print("💡 사용법: python auto_git_logger.py start \"작업 설명\"")
                return
            logger.start_work(args.description)
        
        elif args.action == 'complete':
            if not args.description:
                print("❌ 완료 설명을 입력해주세요!")
                print("💡 사용법: python auto_git_logger.py complete \"완료 설명\"")
                return
            logger.complete_work(args.description)
        
        elif args.action == 'rollback':
            logger.rollback(args.target)
        
        elif args.action == 'status':
            logger.show_status()
        
        elif args.action == 'backup':
            logger.daily_backup()
            
    except KeyboardInterrupt:
        print("\n\n👋 작업이 중단되었습니다. 안전하게 종료합니다.")
    except Exception as e:
        print(f"\n❌ 오류가 발생했어요: {e}")
        print("💡 문제가 계속되면 다음 명령어로 Git 상태를 확인해보세요:")
        print("   git status")
        print("   git log --oneline -5")


if __name__ == "__main__":
    main()