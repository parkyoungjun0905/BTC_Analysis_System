#!/usr/bin/env python3

import asyncio
from integrated_adaptive_system import IntegratedAdaptiveSystem
import numpy as np

async def quick_integrated_test():
    print('🚀 통합 적응형 시스템 빠른 테스트')
    print('='*50)
    
    # 시스템 초기화
    system = IntegratedAdaptiveSystem()
    
    print('⚙️ 시스템 초기화 완료')
    
    # 가상 시장 데이터 생성
    market_data = {
        'price': 52000 + np.random.normal(0, 500),
        'volume': np.random.exponential(1500000),
        'volatility': np.random.uniform(0.02, 0.04),
        'rsi': 50 + np.random.normal(0, 15),
        'macd': np.random.normal(0, 10),
        'fear_greed_index': 50 + np.random.normal(0, 20),
        'timestamp': '2025-08-26T15:25:00'
    }
    
    print('📊 통합 예측 생성 테스트...')
    
    # 통합 예측 생성
    prediction = await system.generate_integrated_prediction(market_data)
    
    if prediction:
        print(f'✅ 통합 예측 성공!')
        print(f'   현재 가격: ${prediction.current_price:,.0f}')
        print(f'   예측 가격: ${prediction.predicted_price:,.0f}')
        print(f'   방향: {prediction.direction}')
        print(f'   신뢰도: {prediction.confidence:.2f}')
        print(f'   시장 상황: {prediction.market_condition}')
        print(f'   전략: {prediction.strategy_used}')
        print(f'   정확도 추정: {prediction.accuracy_estimate:.1%}')
    else:
        print('❌ 통합 예측 생성 실패')
    
    # 시스템 상태 확인
    status = await system.get_current_status()
    print(f'\n🏥 시스템 상태: {"running" if status["running"] else "stopped"}')
    
    print('🎉 통합 시스템 테스트 완료!')

if __name__ == "__main__":
    asyncio.run(quick_integrated_test())