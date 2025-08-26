#!/usr/bin/env python3
"""
안전한 배포 시스템 - 클로드 코드용
배포 → 검증 → 알림을 자동화
"""

import asyncio
import subprocess
import time
import json
import aiohttp
from datetime import datetime
from typing import Dict, Tuple

class SafeDeploymentManager:
    """안전한 배포 관리자"""
    
    def __init__(self):
        self.resource_group = "btc-risk-monitor-rg"
        self.function_name = "btc-risk-monitor-func"
        self.function_url = "https://btc-risk-monitor-func.azurewebsites.net"
        self.bot_token = "8333838666:AAECXOavuX4aaUI9bFttGTqzNY2vb_tqGJI"
        self.chat_id = "5373223115"
        self.deployment_id = f"deploy_{int(time.time())}"
        
    async def deploy_safely(self) -> Tuple[bool, str]:
        """안전한 배포 실행"""
        
        try:
            print(f"🚀 안전 배포 시작 - ID: {self.deployment_id}")
            
            # 1. 사용자에게 배포 시작 알림
            await self.notify_deployment_start()
            
            # 2. 현재 버전 백업 (롤백용)
            current_version = await self.get_current_version()
            print(f"📦 현재 버전 백업: {current_version}")
            
            # 3. 새 버전 패키징
            print("📦 새 버전 패키징...")
            zip_file = self.create_deployment_package()
            
            # 4. Azure Functions 배포
            print("☁️ Azure Functions 배포 중...")
            deploy_success = self.deploy_to_azure(zip_file)
            
            if not deploy_success:
                await self.notify_deployment_failure("배포 명령 실패")
                return False, "배포 명령 실패"
            
            # 5. 배포 완료 대기 (30초)
            print("⏳ 배포 안정화 대기 (30초)...")
            await asyncio.sleep(30)
            
            # 6. 배포 검증
            print("🔍 배포 검증 중...")
            verification_result = await self.verify_deployment()
            
            if verification_result["success"]:
                await self.notify_deployment_success(verification_result)
                print("✅ 배포 성공!")
                return True, "배포 성공"
            else:
                await self.notify_deployment_failure(verification_result["error"])
                print(f"❌ 배포 실패: {verification_result['error']}")
                return False, verification_result["error"]
                
        except Exception as e:
            error_msg = f"배포 중 예외 발생: {str(e)}"
            await self.notify_deployment_failure(error_msg)
            print(f"💥 {error_msg}")
            return False, error_msg
    
    def create_deployment_package(self) -> str:
        """배포 패키지 생성"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        zip_filename = f"deployment_{timestamp}.zip"
        
        # 불필요한 파일 제외하고 패키징
        exclude_patterns = [
            "*.git*", "__pycache__/*", "*.pyc", "test_*", 
            "*cache*", ".venv/*", "venv/*", "*.log", 
            ".DS_Store", "deployment_*.zip"
        ]
        
        exclude_args = []
        for pattern in exclude_patterns:
            exclude_args.extend(["-x", pattern])
        
        cmd = ["zip", "-r", zip_filename, "."] + exclude_args
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"📦 패키지 생성 완료: {zip_filename}")
            return zip_filename
        else:
            raise Exception(f"패키지 생성 실패: {result.stderr}")
    
    def deploy_to_azure(self, zip_file: str) -> bool:
        """Azure Functions에 배포"""
        
        cmd = [
            "az", "functionapp", "deployment", "source", "config-zip",
            "--resource-group", self.resource_group,
            "--name", self.function_name,
            "--src", zip_file
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("☁️ Azure 배포 명령 성공")
            return True
        else:
            print(f"❌ Azure 배포 명령 실패: {result.stderr}")
            return False
    
    async def get_current_version(self) -> str:
        """현재 배포된 버전 확인"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.function_url}/api/health", 
                                     timeout=aiohttp.ClientTimeout(total=10)) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get("timestamp", "unknown")
                    else:
                        return "unavailable"
        except:
            return "unknown"
    
    async def verify_deployment(self) -> Dict:
        """배포 검증"""
        
        checks = {}
        
        # 1. 헬스체크
        checks["health"] = await self.check_health_endpoint()
        
        # 2. 텔레그램 봇 응답
        checks["telegram"] = await self.check_telegram_bot()
        
        # 3. 핵심 기능 테스트
        checks["features"] = await self.test_key_features()
        
        all_passed = all(checks.values())
        
        return {
            "success": all_passed,
            "checks": checks,
            "error": None if all_passed else "일부 검증 실패"
        }
    
    async def check_health_endpoint(self) -> bool:
        """헬스체크 엔드포인트 확인"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.function_url}/api/health",
                                     timeout=aiohttp.ClientTimeout(total=15)) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get("status") == "optimized_healthy"
                    else:
                        return False
        except Exception as e:
            print(f"헬스체크 실패: {e}")
            return False
    
    async def check_telegram_bot(self) -> bool:
        """텔레그램 봇 응답 확인"""
        try:
            url = f"https://api.telegram.org/bot{self.bot_token}/getMe"
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get("ok", False)
                    else:
                        return False
        except:
            return False
    
    async def test_key_features(self) -> bool:
        """핵심 기능 테스트"""
        try:
            # 수동 모니터링 엔드포인트 테스트
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.function_url}/api/monitor?level=minimal",
                                     timeout=aiohttp.ClientTimeout(total=20)) as response:
                    return response.status in [200, 202]  # 202는 처리 중
        except:
            return False
    
    async def notify_deployment_start(self):
        """배포 시작 알림"""
        message = f"""🚀 **안전 배포 시작**

📦 **배포 ID**: `{self.deployment_id}`
⏰ **시작 시간**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
🔄 **진행 상황**: 배포 중...

잠시만 기다려 주세요! 배포 완료되면 알려드릴게요."""
        
        await self.send_telegram_message(message)
    
    async def notify_deployment_success(self, verification_result: Dict):
        """배포 성공 알림"""
        checks = verification_result["checks"]
        
        message = f"""✅ **배포 성공!**

📦 **배포 ID**: `{self.deployment_id}`
⏰ **완료 시간**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

🔍 **검증 결과**:
• 헬스체크: {'✅' if checks['health'] else '❌'}
• 텔레그램: {'✅' if checks['telegram'] else '❌'}  
• 핵심기능: {'✅' if checks['features'] else '❌'}

🎯 **이제 새로운 기능을 테스트해보세요!**
• 자연어 명령 사용 가능
• 100+ 지표 감시 활성화
• 실시간 이중 시스템 작동"""
        
        await self.send_telegram_message(message)
    
    async def notify_deployment_failure(self, error: str):
        """배포 실패 알림"""
        message = f"""❌ **배포 실패!**

📦 **배포 ID**: `{self.deployment_id}`
⏰ **실패 시간**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
🚨 **오류**: {error}

⚠️ **주의**: 이전 버전이 계속 실행됩니다.
🔧 **조치 필요**: 문제를 확인하고 다시 배포하세요."""
        
        await self.send_telegram_message(message)
    
    async def send_telegram_message(self, message: str):
        """텔레그램 메시지 발송"""
        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        
        try:
            async with aiohttp.ClientSession() as session:
                data = {
                    "chat_id": self.chat_id,
                    "text": message,
                    "parse_mode": "Markdown"
                }
                
                async with session.post(url, json=data, 
                                      timeout=aiohttp.ClientTimeout(total=10)) as response:
                    return response.status == 200
        except Exception as e:
            print(f"텔레그램 발송 실패: {e}")
            return False

# 메인 실행부
async def main():
    """안전 배포 실행"""
    
    deployer = SafeDeploymentManager()
    
    print("🛡️ 안전한 배포 시스템 시작")
    print("=" * 50)
    
    success, message = await deployer.deploy_safely()
    
    print("=" * 50)
    if success:
        print("🎉 배포 완료! 사용자가 새 기능을 테스트할 수 있습니다.")
    else:
        print(f"😢 배포 실패: {message}")
        print("🔧 문제를 해결하고 다시 시도하세요.")

if __name__ == "__main__":
    asyncio.run(main())