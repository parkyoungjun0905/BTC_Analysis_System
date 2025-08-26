#!/bin/bash

echo "🔄 로그인시 자동 실행 설정..."

# 설정 변수
SCRIPT_DIR="/Users/parkyoungjun/Desktop/BTC_Analysis_System"
PYTHON_PATH=$(which python3)
PLIST_DIR="$HOME/Library/LaunchAgents"
PLIST_FILE="$PLIST_DIR/com.btc.cryptoquant.downloader.plist"

# LaunchAgents 디렉토리 생성
mkdir -p "$PLIST_DIR"

# LaunchAgent plist 파일 생성
cat > "$PLIST_FILE" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.btc.cryptoquant.downloader</string>
    
    <key>ProgramArguments</key>
    <array>
        <string>/bin/bash</string>
        <string>-c</string>
        <string>cd $SCRIPT_DIR && $PYTHON_PATH cryptoquant_downloader.py >> logs/auto_download.log 2>&1</string>
    </array>
    
    <key>RunAtLoad</key>
    <true/>
    
    <key>WorkingDirectory</key>
    <string>$SCRIPT_DIR</string>
    
    <key>StandardOutPath</key>
    <string>$SCRIPT_DIR/logs/launchagent.log</string>
    
    <key>StandardErrorPath</key>
    <string>$SCRIPT_DIR/logs/launchagent_error.log</string>
    
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/usr/local/bin:/usr/bin:/bin</string>
    </dict>
</dict>
</plist>
EOF

echo "📁 LaunchAgent 파일 생성: $PLIST_FILE"

# LaunchAgent 로드
launchctl unload "$PLIST_FILE" 2>/dev/null || true
launchctl load "$PLIST_FILE"

echo ""
echo "✅ 로그인시 자동 실행 설정 완료!"
echo ""
echo "📋 동작 방식:"
echo "   • 로그인할 때마다 즉시 1회 실행"
echo "   • CryptoQuant 1일 제한에 맞춘 설정"
echo "   • 중복 다운로드 자동 방지"
echo ""
echo "🔧 관리 명령어:"
echo "launchctl list | grep btc  # 실행 상태 확인"
echo "launchctl unload $PLIST_FILE  # 자동 실행 중지"
echo "launchctl load $PLIST_FILE    # 자동 실행 시작"
echo ""
echo "📁 로그 파일:"
echo "   • $SCRIPT_DIR/logs/auto_download.log"
echo "   • $SCRIPT_DIR/logs/launchagent.log"
echo ""
echo "🧪 수동 테스트:"
echo "cd $SCRIPT_DIR && python3 cryptoquant_downloader.py"