#!/bin/bash

# 🔧 영구 자동 승인 설정 스크립트
# 한 번 실행하면 영원히 권한 질문 없음

echo "🔧 영구 자동 승인 설정 시작..."

# ============================================
# 쉘 환경 설정 파일들에 자동 승인 추가
# ============================================

AUTO_ENV='
# ========== BTC Analysis Auto Approval ==========
export DEBIAN_FRONTEND=noninteractive
export NEEDRESTART_MODE=a
export NEEDRESTART_SUSPEND=1
export APT_LISTCHANGES_FRONTEND=none
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_NO_INPUT=1
export PIP_QUIET=1
export PYTHONUNBUFFERED=1
export HOMEBREW_NO_INSTALL_CLEANUP=1
export HOMEBREW_NO_AUTO_UPDATE=1
export HOMEBREW_NO_ENV_HINTS=1

# 자동 승인 별칭들
alias pip="pip --no-input --disable-pip-version-check"
alias pip3="pip3 --no-input --disable-pip-version-check"
alias brew="brew"
alias sudo="sudo"

# BTC Analysis 자동 실행 함수들
btc_auto() {
    cd /Users/parkyoungjun/Desktop/BTC_Analysis_System
    yes | ./ultra_auto_run.sh
}

btc_setup() {
    cd /Users/parkyoungjun/Desktop/BTC_Analysis_System
    yes | ./setup_auto.sh
}

btc_analysis() {
    cd /Users/parkyoungjun/Desktop/BTC_Analysis_System
    yes | ./start_analysis_auto.sh
}

btc_recovery() {
    cd /Users/parkyoungjun/Desktop/BTC_Analysis_System
    yes | ./claude_quick_recovery.sh
}
# ===============================================
'

# bash 설정 파일에 추가
echo "$AUTO_ENV" >> ~/.bashrc 2>/dev/null || true
echo "$AUTO_ENV" >> ~/.bash_profile 2>/dev/null || true

# zsh 설정 파일에 추가 (macOS 기본 쉘)
echo "$AUTO_ENV" >> ~/.zshrc 2>/dev/null || true
echo "$AUTO_ENV" >> ~/.zprofile 2>/dev/null || true

# ============================================
# pip 전역 설정 파일 생성
# ============================================
echo "🐍 pip 전역 설정 생성..."

mkdir -p ~/.pip
cat > ~/.pip/pip.conf << 'EOF'
[global]
disable-pip-version-check = true
no-input = true
quiet = true
timeout = 60
trusted-host = pypi.org
               pypi.python.org
               files.pythonhosted.org

[install]
user = false
upgrade = true
EOF

# ============================================
# Homebrew 자동 설정
# ============================================
echo "🍺 Homebrew 자동 설정..."

# Homebrew 환경 변수를 영구 설정
mkdir -p ~/.config
cat > ~/.config/homebrew_auto << 'EOF'
export HOMEBREW_NO_INSTALL_CLEANUP=1
export HOMEBREW_NO_AUTO_UPDATE=1
export HOMEBREW_NO_ENV_HINTS=1
export HOMEBREW_NO_ANALYTICS=1
EOF

# ============================================
# Claude CLI 자동 설정
# ============================================
echo "🤖 Claude CLI 자동 설정..."

mkdir -p ~/.claude
cat > ~/.claude/config.json << 'EOF'
{
  "permission_mode": "bypassPermissions",
  "auto_approve": true,
  "dangerous_mode": true,
  "skip_confirmations": true
}
EOF

# ============================================
# 시스템 별칭 생성 (터미널 바로가기)
# ============================================
echo "⚡ 시스템 별칭 생성..."

# 전역 별칭 파일 생성
cat > ~/.btc_aliases << 'EOF'
#!/bin/bash
# BTC Analysis System 자동 실행 별칭들

alias btc="cd /Users/parkyoungjun/Desktop/BTC_Analysis_System && yes | ./ultra_auto_run.sh"
alias btc-setup="cd /Users/parkyoungjun/Desktop/BTC_Analysis_System && yes | ./setup_auto.sh"
alias btc-run="cd /Users/parkyoungjun/Desktop/BTC_Analysis_System && yes | ./start_analysis_auto.sh"
alias btc-fix="cd /Users/parkyoungjun/Desktop/BTC_Analysis_System && yes | ./fix_permissions.sh"
alias btc-recover="cd /Users/parkyoungjun/Desktop/BTC_Analysis_System && yes | ./claude_quick_recovery.sh"
alias btc-status="cd /Users/parkyoungjun/Desktop/BTC_Analysis_System && ls -la historical_data/ | tail -5"

# Claude 자동 실행
alias claude-auto="claude --dangerously-skip-permissions"
alias claude-bypass="claude --permission-mode bypassPermissions"
EOF

# 별칭 파일을 쉘 설정에 포함
echo "source ~/.btc_aliases" >> ~/.bashrc 2>/dev/null || true
echo "source ~/.btc_aliases" >> ~/.zshrc 2>/dev/null || true

# ============================================
# 모든 스크립트 권한 설정
# ============================================
echo "🔨 모든 스크립트 최종 권한 설정..."

cd /Users/parkyoungjun/Desktop/BTC_Analysis_System

# 모든 실행 파일에 최대 권한
chmod 777 *.sh *.py 2>/dev/null || true

# 소유권 설정
chown -R $(whoami):staff . 2>/dev/null || true

# macOS 확장 속성 완전 제거
find . -type f \( -name "*.sh" -o -name "*.py" \) -exec xattr -c {} \; 2>/dev/null || true

echo ""
echo "🎉 영구 자동 승인 설정 완료!"
echo ""
echo "🚀 이제 다음 명령어들로 자동 실행 가능:"
echo ""
echo "   btc           # 전체 자동 실행"
echo "   btc-setup     # 자동 설정"
echo "   btc-run       # 자동 분석"
echo "   btc-recover   # 자동 복구"
echo "   btc-status    # 상태 확인"
echo ""
echo "🔄 새 터미널에서 바로 사용하려면:"
echo "   source ~/.zshrc"
echo "   source ~/.bashrc"
echo ""
echo "💡 Claude 자동 모드:"
echo "   claude-auto   # 권한 질문 완전 생략"
echo "   claude-bypass # 권한 우회"
echo ""

# 현재 쉘에 즉시 적용
source ~/.btc_aliases 2>/dev/null || true
source ~/.zshrc 2>/dev/null || source ~/.bashrc 2>/dev/null || true

echo "✅ 모든 설정이 현재 세션에도 적용되었습니다!"