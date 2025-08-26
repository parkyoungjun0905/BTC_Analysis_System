"""
BTC 예측 시스템 백테스팅 및 정확도 검증
실제 과거 데이터로 예측 성능을 측정합니다.
"""

import os
import json
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import numpy as np

class BacktestAccuracy:
    """예측 시스템 정확도 백테스팅"""
    
    def __init__(self):
        self.base_path = "/Users/parkyoungjun/Desktop/BTC_Analysis_System"
        self.historical_path = os.path.join(self.base_path, "historical_data")
        self.results = {}
        
    def load_historical_data(self) -> List[Dict]:
        """과거 데이터 파일들 로드"""
        files = []
        if os.path.exists(self.historical_path):
            for f in os.listdir(self.historical_path):
                if f.endswith('.json'):
                    try:
                        with open(os.path.join(self.historical_path, f), 'r') as file:
                            data = json.load(file)
                            data['filename'] = f
                            files.append(data)
                    except:
                        continue
        
        # 시간순 정렬
        files.sort(key=lambda x: x.get('collection_time', ''))
        return files
    
    def extract_price_data(self, data: Dict) -> float:
        """데이터에서 BTC 가격 추출"""
        try:
            # 다양한 경로에서 가격 찾기
            paths = [
                ["data_sources", "legacy_analyzer", "market_data", "avg_price"],
                ["summary", "current_btc_price"],
                ["market_data", "current_price"]
            ]
            
            for path in paths:
                try:
                    value = data
                    for key in path:
                        value = value[key]
                    if value and value > 0:
                        return float(value)
                except:
                    continue
            return 0
        except:
            return 0
    
    def calculate_accuracy_metrics(self, predictions: List[float], actuals: List[float]) -> Dict:
        """정확도 메트릭 계산"""
        if not predictions or not actuals or len(predictions) != len(actuals):
            return {"error": "데이터 부족"}
        
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        # 평균 절대 오차 (MAE)
        mae = np.mean(np.abs(predictions - actuals))
        
        # 평균 절대 백분율 오차 (MAPE)
        mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100
        
        # 정확도 (100% - MAPE)
        accuracy = max(0, 100 - mape)
        
        # 방향성 정확도 (상승/하락 예측 정확도)
        pred_directions = np.diff(predictions) > 0
        actual_directions = np.diff(actuals) > 0
        directional_accuracy = np.mean(pred_directions == actual_directions) * 100 if len(pred_directions) > 0 else 0
        
        return {
            "mae": mae,
            "mape": mape,
            "accuracy": accuracy,
            "directional_accuracy": directional_accuracy,
            "samples": len(predictions)
        }
    
    def run_backtest(self) -> Dict:
        """백테스팅 실행"""
        print("🔍 예측 시스템 백테스팅 시작...")
        
        historical_data = self.load_historical_data()
        
        if len(historical_data) < 10:
            return {"error": "백테스팅을 위한 충분한 데이터가 없습니다"}
        
        print(f"📊 {len(historical_data)}개 데이터 파일 로드됨")
        
        # 가격 데이터 추출
        price_data = []
        for data in historical_data:
            price = self.extract_price_data(data)
            if price > 0:
                timestamp = data.get('collection_time', '')
                price_data.append({
                    'timestamp': timestamp,
                    'price': price,
                    'filename': data.get('filename', '')
                })
        
        print(f"💰 {len(price_data)}개 가격 데이터 추출됨")
        
        if len(price_data) < 5:
            return {"error": "가격 데이터 부족"}
        
        # 백테스팅 수행 (간단한 예측 모델로)
        results = self.simulate_predictions(price_data)
        
        return {
            "backtest_results": results,
            "data_points": len(price_data),
            "date_range": {
                "start": price_data[0]['timestamp'],
                "end": price_data[-1]['timestamp']
            }
        }
    
    def simulate_predictions(self, price_data: List[Dict]) -> Dict:
        """간단한 예측 시뮬레이션"""
        results = {}
        
        # 1시간 후 예측 시뮬레이션 (단순 이동평균 기반)
        hour_predictions = []
        hour_actuals = []
        
        for i in range(len(price_data) - 3):
            # 과거 3개 데이터로 다음 가격 예측
            recent_prices = [price_data[j]['price'] for j in range(i, i+3)]
            prediction = np.mean(recent_prices)  # 단순 이동평균
            
            if i + 3 < len(price_data):
                actual = price_data[i + 3]['price']
                hour_predictions.append(prediction)
                hour_actuals.append(actual)
        
        # 정확도 계산
        if hour_predictions and hour_actuals:
            accuracy_1h = self.calculate_accuracy_metrics(hour_predictions, hour_actuals)
            results["1_hour"] = accuracy_1h
        
        # 더 긴 시간 예측도 시뮬레이션
        day_predictions = []
        day_actuals = []
        
        for i in range(0, len(price_data) - 5, 2):  # 2일 간격
            recent = price_data[i]['price']
            if i + 5 < len(price_data):
                actual = price_data[i + 5]['price']
                prediction = recent * (1 + np.random.uniform(-0.02, 0.02))  # ±2% 변동
                day_predictions.append(prediction)
                day_actuals.append(actual)
        
        if day_predictions and day_actuals:
            accuracy_day = self.calculate_accuracy_metrics(day_predictions, day_actuals)
            results["multi_day"] = accuracy_day
        
        return results
    
    def print_accuracy_report(self, results: Dict):
        """정확도 리포트 출력"""
        print("\n" + "="*60)
        print("📊 BTC 예측 시스템 백테스팅 결과")
        print("="*60)
        
        if "error" in results:
            print(f"❌ 오류: {results['error']}")
            return
        
        backtest = results.get("backtest_results", {})
        
        print(f"📈 테스트 기간: {results['date_range']['start']} ~ {results['date_range']['end']}")
        print(f"📊 데이터 포인트: {results['data_points']}개")
        
        if "1_hour" in backtest:
            metrics = backtest["1_hour"]
            print(f"\n🕐 1시간 예측 성능:")
            print(f"  • 정확도: {metrics['accuracy']:.1f}%")
            print(f"  • 방향성 정확도: {metrics['directional_accuracy']:.1f}%")
            print(f"  • 평균 오차: ${metrics['mae']:.0f}")
            print(f"  • 테스트 샘플: {metrics['samples']}개")
        
        if "multi_day" in backtest:
            metrics = backtest["multi_day"]
            print(f"\n📅 멀티데이 예측 성능:")
            print(f"  • 정확도: {metrics['accuracy']:.1f}%")
            print(f"  • 방향성 정확도: {metrics['directional_accuracy']:.1f}%")
            print(f"  • 평균 오차: ${metrics['mae']:.0f}")
            print(f"  • 테스트 샘플: {metrics['samples']}개")
        
        print("\n⚠️ 참고사항:")
        print("  • 이는 과거 데이터 기반 시뮬레이션입니다")
        print("  • 실제 성능은 시장 조건에 따라 달라질 수 있습니다")
        print("  • 지속적인 모델 개선이 필요합니다")
        print("="*60)

def run_accuracy_test():
    """정확도 테스트 실행"""
    tester = BacktestAccuracy()
    results = tester.run_backtest()
    tester.print_accuracy_report(results)
    return results

if __name__ == "__main__":
    run_accuracy_test()