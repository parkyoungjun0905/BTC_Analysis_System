#!/bin/bash
# 🌱 초급: Git 시스템 초기 설정 (복사해서 사용)
# 처음 한 번만 실행하면 되는 설정 스크립트예요

echo "🔧 Git 자동화 시스템 초기 설정을 시작합니다..."
echo ""

# Git 사용자 정보 설정 (아직 설정되지 않은 경우)
git_name=$(git config user.name)
git_email=$(git config user.email)

if [ -z "$git_name" ]; then
    echo "Git 사용자 이름을 입력해주세요:"
    read user_name
    git config user.name "$user_name"
    echo "✅ 사용자 이름 설정 완료: $user_name"
fi

if [ -z "$git_email" ]; then
    echo "Git 이메일 주소를 입력해주세요:"
    read user_email
    git config user.email "$user_email"
    echo "✅ 이메일 주소 설정 완료: $user_email"
fi

# 실행 권한 부여
echo "🔐 스크립트 실행 권한을 설정합니다..."
chmod +x start_work.sh
chmod +x complete_work.sh
chmod +x rollback_work.sh
chmod +x check_status.sh
chmod +x daily_backup.sh
chmod +x auto_git_logger.py

echo "✅ 실행 권한 설정 완료!"

# Git 자동화 시스템 초기화
echo "📁 프로젝트 초기화를 진행합니다..."
python3 auto_git_logger.py init

echo ""
echo "🎉 Git 자동화 시스템 설정이 완료되었습니다!"
echo ""
echo "💡 이제 다음 명령어들을 사용할 수 있어요:"
echo "   ./start_work.sh     - 작업 시작"
echo "   ./complete_work.sh  - 작업 완료"
echo "   ./check_status.sh   - 현황 확인"
echo "   ./rollback_work.sh  - 되돌리기"
echo "   ./daily_backup.sh   - 하루 종료 백업"
echo ""
echo "🚀 첫 작업을 시작해보세요: ./start_work.sh"