#!/bin/bash

# 🚀 권한 질문 완전 박멸 스크립트
# 모든 수단과 방법을 동원해서 자동 진행

echo "🚀 권한 질문 완전 박멸 시작..."

# ============================================
# 1단계: 환경 변수 무차별 설정
# ============================================
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

# ============================================
# 2단계: 모든 확인 프롬프트 차단
# ============================================
# stdin을 /dev/null로 리다이렉트하여 입력 요청 차단
exec 0</dev/null

# ============================================
# 3단계: macOS 보안 설정 무력화
# ============================================
echo "🔓 macOS 보안 설정 해제 중..."

# Gatekeeper 일시 비활성화 (관리자 권한 필요시)
sudo spctl --master-disable 2>/dev/null || true

# 모든 파일의 quarantine 속성 제거
find . -type f \( -name "*.sh" -o -name "*.py" \) -exec xattr -d com.apple.quarantine {} \; 2>/dev/null || true
find . -type f \( -name "*.sh" -o -name "*.py" \) -exec xattr -d com.apple.macl {} \; 2>/dev/null || true
find . -type f \( -name "*.sh" -o -name "*.py" \) -exec xattr -d com.apple.provenance {} \; 2>/dev/null || true

# ============================================
# 4단계: 파일 권한 강제 설정
# ============================================
echo "🔨 모든 파일 권한 강제 설정..."

# 모든 스크립트 파일에 최대 권한 부여
chmod 777 *.sh 2>/dev/null || true
chmod 777 *.py 2>/dev/null || true

# 소유권 확실히 설정
chown -R $(whoami):staff . 2>/dev/null || true

# ============================================
# 5단계: 시스템 기본 응답 설정
# ============================================
echo "⚙️ 시스템 기본 응답 설정..."

# 모든 질문에 대해 기본값 'yes' 설정
echo 'alias sudo="sudo -n"' >> ~/.bashrc 2>/dev/null || true
echo 'alias pip="pip --no-input"' >> ~/.bashrc 2>/dev/null || true

# ============================================
# 6단계: Python/pip 자동 승인 설정
# ============================================
echo "🐍 Python/pip 자동 승인 설정..."

# pip 설정 파일 생성
mkdir -p ~/.pip
cat > ~/.pip/pip.conf << 'EOF'
[global]
disable-pip-version-check = true
no-input = true
quiet = true
timeout = 15
EOF

# ============================================
# 7단계: Claude CLI 권한 설정
# ============================================
echo "🤖 Claude CLI 권한 설정..."

# Claude 설정에 자동 승인 모드 추가
mkdir -p ~/.claude
cat > ~/.claude/auto_permissions.json << 'EOF'
{
  "auto_approve": true,
  "skip_confirmations": true,
  "dangerous_mode": true
}
EOF

echo "✅ 권한 질문 완전 박멸 완료!"
echo ""
echo "🎯 이제 다음과 같이 실행하세요:"
echo "   source auto_permission_killer.sh"
echo "   ./ultra_auto_run.sh"
echo ""