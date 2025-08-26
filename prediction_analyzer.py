"""
📊 BTC 30일 예측 시스템 분석 도구
- 예측 정확도 분석
- 불확실성 모델링 검증
- 리스크 프로파일 생성
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

class PredictionAnalyzer:
    """예측 결과 분석기"""
    
    def __init__(self, json_path: str):
        """초기화"""
        self.json_path = json_path
        self.data = self.load_predictions()
        
    def load_predictions(self) -> dict:
        """예측 데이터 로드"""
        try:
            with open(self.json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ 데이터 로드 실패: {e}")
            return {}
    
    def analyze_prediction_quality(self):
        """예측 품질 분석"""
        if not self.data:
            return
        
        print("🔍 BTC 30일 예측 시스템 분석")
        print("=" * 60)
        
        predictions = self.data.get('predictions', [])
        if not predictions:
            print("❌ 예측 데이터 없음")
            return
        
        # 기본 정보
        current_price = self.data.get('current_price', 0)
        generation_time = self.data.get('generation_time', '')
        
        print(f"📊 기본 정보:")
        print(f"  • 현재 가격: ${current_price:,.0f}")
        print(f"  • 분석 시간: {generation_time}")
        print(f"  • 예측 기간: {len(predictions)}일")
        
        # 신뢰도 분석
        confidence_data = [p['confidence'] for p in predictions]
        price_data = [p['price'] for p in predictions]
        change_data = [p['change_pct'] for p in predictions]
        
        print(f"\n📈 예측 신뢰도 분석:")
        print(f"  • 평균 신뢰도: {np.mean(confidence_data):.1f}%")
        print(f"  • 신뢰도 범위: {min(confidence_data):.1f}% ~ {max(confidence_data):.1f}%")
        print(f"  • 신뢰도 표준편차: {np.std(confidence_data):.1f}%")
        
        # 가격 예측 분석
        print(f"\n💰 가격 예측 분석:")
        print(f"  • 예측 가격 범위: ${min(price_data):,.0f} ~ ${max(price_data):,.0f}")
        print(f"  • 평균 예측 가격: ${np.mean(price_data):,.0f}")
        print(f"  • 가격 변동성: ${np.std(price_data):,.0f}")
        print(f"  • 최대 상승 예상: +{max(change_data):.2f}%")
        print(f"  • 최대 하락 예상: {min(change_data):+.2f}%")
        
        # 기간별 신뢰도 변화
        print(f"\n⏰ 기간별 신뢰도 변화:")
        for period, days in [("단기", range(1, 8)), ("중기", range(8, 15)), ("장기", range(15, 31))]:
            period_confidence = [predictions[i-1]['confidence'] for i in days if i <= len(predictions)]
            if period_confidence:
                print(f"  • {period} ({days.start}-{days.stop-1}일): 평균 {np.mean(period_confidence):.1f}%")
        
        # 리스크 분석
        self.analyze_risk_profile(predictions, current_price)
        
        # 시나리오 분석
        self.scenario_analysis(predictions, current_price)
    
    def analyze_risk_profile(self, predictions: list, current_price: float):
        """리스크 프로파일 분석"""
        print(f"\n⚠️ 리스크 프로파일 분석:")
        
        # 신뢰구간 너비 분석
        confidence_widths = []
        for p in predictions:
            width = p['upper_bound'] - p['lower_bound']
            confidence_widths.append(width)
        
        print(f"  • 평균 신뢰구간 너비: ${np.mean(confidence_widths):,.0f}")
        print(f"  • 신뢰구간 확장률: {(max(confidence_widths) / min(confidence_widths)):.1f}배")
        
        # 변동성 리스크
        price_changes = [abs(p['change_pct']) for p in predictions]
        high_volatility_days = sum(1 for change in price_changes if change > 5)
        
        print(f"  • 고변동성 구간 (±5% 이상): {high_volatility_days}일")
        print(f"  • 평균 일간 변동성: ±{np.mean(price_changes):.2f}%")
        
        # 방향성 리스크 (연속 하락/상승)
        consecutive_down = 0
        consecutive_up = 0
        max_consecutive_down = 0
        max_consecutive_up = 0
        
        for p in predictions:
            if p['change_pct'] < 0:
                consecutive_down += 1
                consecutive_up = 0
                max_consecutive_down = max(max_consecutive_down, consecutive_down)
            else:
                consecutive_up += 1
                consecutive_down = 0
                max_consecutive_up = max(max_consecutive_up, consecutive_up)
        
        print(f"  • 최대 연속 상승 예상: {max_consecutive_up}일")
        print(f"  • 최대 연속 하락 예상: {max_consecutive_down}일")
    
    def scenario_analysis(self, predictions: list, current_price: float):
        """시나리오 분석"""
        print(f"\n🎯 시나리오 분석:")
        
        # 30일 후 가격 기준
        final_prediction = predictions[-1]
        final_price = final_prediction['price']
        final_change = final_prediction['change_pct']
        
        # 시나리오 정의
        if final_change >= 10:
            scenario = "🚀 강세 시나리오"
        elif final_change >= 5:
            scenario = "📈 상승 시나리오"
        elif final_change >= -5:
            scenario = "📊 횡보 시나리오"
        elif final_change >= -10:
            scenario = "📉 하락 시나리오"
        else:
            scenario = "💥 약세 시나리오"
        
        print(f"  • 주요 시나리오: {scenario}")
        print(f"  • 30일 후 예상: ${final_price:,.0f} ({final_change:+.2f}%)")
        print(f"  • 신뢰도: {final_prediction['confidence']:.1f}%")
        
        # 확률별 구간
        positive_days = sum(1 for p in predictions if p['change_pct'] > 0)
        negative_days = len(predictions) - positive_days
        
        print(f"\n📊 확률 분포:")
        print(f"  • 상승 예상 일수: {positive_days}일 ({positive_days/len(predictions)*100:.1f}%)")
        print(f"  • 하락 예상 일수: {negative_days}일 ({negative_days/len(predictions)*100:.1f}%)")
        
        # 주요 이정표
        print(f"\n🎯 주요 이정표:")
        milestones = [7, 14, 21, 30]
        for day in milestones:
            if day <= len(predictions):
                p = predictions[day-1]
                print(f"  • {day:2d}일 후: ${p['price']:,.0f} ({p['change_pct']:+.2f}%) [신뢰도: {p['confidence']:.1f}%]")
    
    def export_analysis_report(self):
        """분석 보고서 내보내기"""
        if not self.data:
            return
        
        predictions = self.data.get('predictions', [])
        current_price = self.data.get('current_price', 0)
        
        # 분석 보고서 생성
        report = {
            "analysis_time": datetime.now().isoformat(),
            "data_source": self.json_path,
            "basic_info": {
                "current_price": current_price,
                "prediction_days": len(predictions),
                "generation_time": self.data.get('generation_time', '')
            },
            "confidence_analysis": {
                "avg_confidence": np.mean([p['confidence'] for p in predictions]),
                "min_confidence": min([p['confidence'] for p in predictions]),
                "max_confidence": max([p['confidence'] for p in predictions]),
                "confidence_std": np.std([p['confidence'] for p in predictions])
            },
            "price_analysis": {
                "avg_price": np.mean([p['price'] for p in predictions]),
                "price_range": {
                    "min": min([p['price'] for p in predictions]),
                    "max": max([p['price'] for p in predictions])
                },
                "price_volatility": np.std([p['price'] for p in predictions]),
                "final_prediction": predictions[-1]['price'] if predictions else current_price,
                "total_change_pct": predictions[-1]['change_pct'] if predictions else 0
            },
            "risk_metrics": {
                "high_volatility_days": sum(1 for p in predictions if abs(p['change_pct']) > 5),
                "avg_daily_volatility": np.mean([abs(p['change_pct']) for p in predictions]),
                "confidence_interval_avg_width": np.mean([p['upper_bound'] - p['lower_bound'] for p in predictions])
            }
        }
        
        # 보고서 저장
        report_path = self.json_path.replace('.json', '_analysis.json')
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"\n📄 분석 보고서 저장: {report_path}")
        return report_path

def main():
    """메인 실행"""
    json_path = "/Users/parkyoungjun/Desktop/BTC_Analysis_System/monthly_predictions.json"
    
    analyzer = PredictionAnalyzer(json_path)
    analyzer.analyze_prediction_quality()
    analyzer.export_analysis_report()
    
    print(f"\n" + "=" * 60)
    print("🎉 BTC 예측 시스템 분석 완료!")
    print("=" * 60)

if __name__ == "__main__":
    main()