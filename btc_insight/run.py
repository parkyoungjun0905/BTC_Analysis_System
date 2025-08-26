#!/usr/bin/env python3
"""
🚀 BTC Insight 실행 스크립트
- 빠르고 쉬운 실행을 위한 유틸리티
"""

import os
import sys
from pathlib import Path

def main():
    print("🎯 BTC Insight 코인분석프로그램")
    print("=" * 50)
    print()
    print("실행 옵션을 선택하세요:")
    print("1. 💡 처음 실행 (백테스트 학습 + 90% 정확도 달성)")
    print("2. ⚡ 빠른 실행 (저장된 모델 사용)")
    print("3. 📚 저장된 모델 목록 보기")
    print("4. ❌ 종료")
    print()
    
    while True:
        try:
            choice = input("선택하세요 (1-4): ").strip()
            
            if choice == "1":
                print("\n🔥 처음 실행: 백테스트 학습 시작")
                print("⚠️  90% 정확도 달성까지 시간이 걸릴 수 있습니다")
                confirm = input("계속 진행하시겠습니까? (y/n): ").strip().lower()
                if confirm in ['y', 'yes', '예']:
                    os.system("python3 main.py")
                break
                
            elif choice == "2":
                print("\n⚡ 빠른 실행: 저장된 모델 사용")
                os.system("python3 main.py --fast")
                break
                
            elif choice == "3":
                print("\n📚 저장된 모델 목록:")
                os.system("python3 main.py --list-models")
                print("\n다시 선택하세요.")
                continue
                
            elif choice == "4":
                print("👋 프로그램을 종료합니다.")
                sys.exit(0)
                
            else:
                print("❌ 잘못된 선택입니다. 1-4 중에서 선택하세요.")
                continue
                
        except KeyboardInterrupt:
            print("\n👋 프로그램을 종료합니다.")
            sys.exit(0)
        except Exception as e:
            print(f"❌ 오류 발생: {e}")
            break

if __name__ == "__main__":
    # 현재 디렉터리를 btc_insight로 변경
    current_file = Path(__file__).resolve()
    btc_insight_dir = current_file.parent
    os.chdir(btc_insight_dir)
    
    main()