#!/usr/bin/env python3
"""
🧪 BTC 멀티 에이전트 시스템 간단 테스트
데이터 로딩과 에이전트 초기화만 테스트 (학습은 제외)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from btc_multiagent_deeplearning_system import MultiAgentBTCLearningSystem

def test_system():
    """시스템 기본 기능 테스트"""
    print("🧪 BTC 멀티 에이전트 시스템 기본 테스트 시작")
    print("="*60)
    
    try:
        # 시스템 초기화
        system = MultiAgentBTCLearningSystem()
        print("✅ 시스템 초기화 성공")
        
        # 데이터 로딩 테스트 (간단 버전)
        print("\n📊 데이터 로딩 테스트...")
        data_path = system.data_path
        csv_file = os.path.join(data_path, "ai_matrix_complete.csv")
        
        if os.path.exists(csv_file):
            file_size = os.path.getsize(csv_file) / (1024*1024)
            print(f"✅ 데이터 파일 존재: {file_size:.1f}MB")
        else:
            print("❌ 데이터 파일 없음")
            return False
        
        # 에이전트 초기화 테스트
        print("\n🤖 에이전트 초기화 테스트...")
        if system.initialize_agents():
            print(f"✅ {len(system.agents)}개 에이전트 초기화 성공")
            
            # 에이전트 정보 출력
            for agent_id, agent in system.agents.items():
                print(f"  Agent {agent_id}: {agent.specialization} "
                      f"({agent.target_hours[0]}-{agent.target_hours[1]}시간)")
        else:
            print("❌ 에이전트 초기화 실패")
            return False
        
        # 예측 공식 가이드 생성 테스트
        print("\n📋 예측 공식 가이드 생성 테스트...")
        formula_guide = system.generate_prediction_formula_guide()
        if formula_guide:
            print("✅ 예측 공식 가이드 생성 성공")
            print(f"  - 에이전트 공식: {len(formula_guide['agent_formulas'])}개")
            print(f"  - 돌발변수 감지: {len(formula_guide['anomaly_detection'])}개 카테고리")
        else:
            print("❌ 예측 공식 가이드 생성 실패")
        
        print("\n🎉 기본 테스트 완료!")
        print("="*60)
        print("📝 참고:")
        print("  - 실제 학습은 python btc_multiagent_deeplearning_system.py 실행")
        print("  - 학습에는 수시간이 걸릴 수 있습니다")
        print("  - Ctrl+C로 중단 가능합니다")
        
        return True
        
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        return False

if __name__ == "__main__":
    success = test_system()
    if success:
        print("\n✅ 모든 기본 테스트 통과!")
    else:
        print("\n❌ 테스트 실패")
        sys.exit(1)