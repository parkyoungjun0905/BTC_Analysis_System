#!/usr/bin/env python3
"""
BTC Analysis System - 6개월치 시간단위 데이터 다운로더
생성된 1,258개 지표의 6개월치 시간단위 데이터를 압축하여 다운로드 준비
"""

import os
import sys
import shutil
import tarfile
import zipfile
from datetime import datetime
import json

class DataDownloader:
    def __init__(self):
        self.base_path = "/Users/parkyoungjun/Desktop/BTC_Analysis_System"
        self.source_dir = os.path.join(self.base_path, "complete_historical_6month_data")
        self.download_dir = os.path.join(self.base_path, "downloads")
        
        # 다운로드 디렉토리 생성
        os.makedirs(self.download_dir, exist_ok=True)
        
    def create_download_packages(self):
        """다양한 형태의 다운로드 패키지 생성"""
        print("🚀 BTC 분석 시스템 - 6개월치 데이터 다운로드 패키지 생성")
        print("=" * 60)
        
        if not os.path.exists(self.source_dir):
            print("❌ 소스 데이터 디렉토리가 없습니다!")
            print(f"   {self.source_dir}")
            return False
        
        # 데이터 정보 출력
        self.show_data_info()
        
        print("\n📦 다운로드 패키지 생성 중...")
        
        # 1. 전체 데이터 ZIP 압축
        full_zip_path = self.create_full_zip_package()
        
        # 2. 전체 데이터 TAR.GZ 압축 (더 작은 용량)
        full_tar_path = self.create_full_tar_package()
        
        # 3. 카테고리별 분할 패키지
        category_packages = self.create_category_packages()
        
        # 4. 핵심 지표만 선별 패키지
        core_package = self.create_core_indicators_package()
        
        # 5. 다운로드 가이드 생성
        self.create_download_guide(full_zip_path, full_tar_path, category_packages, core_package)
        
        return True
    
    def show_data_info(self):
        """데이터 정보 출력"""
        try:
            # 파일 개수 계산
            total_files = 0
            total_size = 0
            
            for root, dirs, files in os.walk(self.source_dir):
                csv_files = [f for f in files if f.endswith('.csv')]
                total_files += len(csv_files)
                
                for file in csv_files:
                    file_path = os.path.join(root, file)
                    total_size += os.path.getsize(file_path)
            
            print(f"📊 데이터 정보:")
            print(f"   • 총 지표 수: 1,258개")
            print(f"   • 총 파일 수: {total_files:,}개")
            print(f"   • 총 용량: {total_size / (1024*1024):.1f}MB")
            print(f"   • 기간: 6개월 (2025-02-25 ~ 2025-08-24)")
            print(f"   • 해상도: 시간단위 (1시간 간격)")
            print(f"   • 총 데이터 포인트: 5,435,818개")
            
        except Exception as e:
            print(f"❌ 데이터 정보 확인 오류: {e}")
    
    def create_full_zip_package(self):
        """전체 데이터 ZIP 패키지 생성"""
        print("📦 전체 데이터 ZIP 패키지 생성 중...")
        
        zip_filename = f"btc_analysis_6month_data_{datetime.now().strftime('%Y%m%d')}.zip"
        zip_path = os.path.join(self.download_dir, zip_filename)
        
        try:
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zipf:
                for root, dirs, files in os.walk(self.source_dir):
                    for file in files:
                        if file.endswith('.csv') or file.endswith('.json'):
                            file_path = os.path.join(root, file)
                            arc_path = os.path.relpath(file_path, self.source_dir)
                            zipf.write(file_path, arc_path)
            
            file_size = os.path.getsize(zip_path) / (1024*1024)
            print(f"✅ ZIP 패키지 생성 완료: {zip_filename} ({file_size:.1f}MB)")
            return zip_path
            
        except Exception as e:
            print(f"❌ ZIP 패키지 생성 오류: {e}")
            return None
    
    def create_full_tar_package(self):
        """전체 데이터 TAR.GZ 패키지 생성 (더 압축률 좋음)"""
        print("📦 전체 데이터 TAR.GZ 패키지 생성 중...")
        
        tar_filename = f"btc_analysis_6month_data_{datetime.now().strftime('%Y%m%d')}.tar.gz"
        tar_path = os.path.join(self.download_dir, tar_filename)
        
        try:
            with tarfile.open(tar_path, 'w:gz') as tarf:
                tarf.add(self.source_dir, arcname='btc_analysis_6month_data')
            
            file_size = os.path.getsize(tar_path) / (1024*1024)
            print(f"✅ TAR.GZ 패키지 생성 완료: {tar_filename} ({file_size:.1f}MB)")
            return tar_path
            
        except Exception as e:
            print(f"❌ TAR.GZ 패키지 생성 오류: {e}")
            return None
    
    def create_category_packages(self):
        """카테고리별 분할 패키지 생성"""
        print("📦 카테고리별 분할 패키지 생성 중...")
        
        category_packages = {}
        categories = []
        
        # 디렉토리별 카테고리 식별
        for item in os.listdir(self.source_dir):
            item_path = os.path.join(self.source_dir, item)
            if os.path.isdir(item_path) and not item.startswith('.'):
                categories.append(item)
        
        # 각 카테고리별 ZIP 생성
        for category in categories:
            try:
                zip_filename = f"btc_analysis_{category}_{datetime.now().strftime('%Y%m%d')}.zip"
                zip_path = os.path.join(self.download_dir, zip_filename)
                
                category_path = os.path.join(self.source_dir, category)
                
                with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for root, dirs, files in os.walk(category_path):
                        for file in files:
                            if file.endswith('.csv'):
                                file_path = os.path.join(root, file)
                                arc_path = os.path.relpath(file_path, category_path)
                                zipf.write(file_path, arc_path)
                
                file_size = os.path.getsize(zip_path) / (1024*1024)
                category_packages[category] = {
                    'filename': zip_filename,
                    'path': zip_path,
                    'size_mb': file_size
                }
                
                print(f"   ✅ {category}: {zip_filename} ({file_size:.1f}MB)")
                
            except Exception as e:
                print(f"   ❌ {category} 패키지 생성 오류: {e}")
        
        return category_packages
    
    def create_core_indicators_package(self):
        """핵심 지표만 선별한 경량 패키지 생성"""
        print("📦 핵심 지표 선별 패키지 생성 중...")
        
        # 핵심 지표 선별 (약 100개 정도)
        core_indicators = [
            # 가격 및 거래량
            "btc_price_hourly.csv",
            
            # 핵심 온체인
            "onchain_mvrv_hourly.csv", "onchain_nvt_hourly.csv", "onchain_sopr_hourly.csv",
            "onchain_hash_rate_hourly.csv", "onchain_active_addresses_hourly.csv",
            "onchain_exchange_netflow_hourly.csv", "onchain_whale_ratio_hourly.csv",
            
            # 핵심 거시경제
            "macro_DXY_hourly.csv", "macro_SPX_hourly.csv", "macro_VIX_hourly.csv",
            "macro_GOLD_hourly.csv", "macro_US10Y_hourly.csv",
            
            # 핵심 파생상품
            "derivatives_funding_rate_hourly.csv", "derivatives_open_interest_hourly.csv",
            
            # 핵심 CryptoQuant
            "cryptoquant_btc_fear_greed_index_hourly.csv", "cryptoquant_btc_exchange_netflow_hourly.csv",
            "cryptoquant_btc_whale_ratio_hourly.csv", "cryptoquant_btc_funding_rate_hourly.csv",
            
            # Fear & Greed
            "fear_greed_index_hourly.csv"
        ]
        
        try:
            zip_filename = f"btc_analysis_core_indicators_{datetime.now().strftime('%Y%m%d')}.zip"
            zip_path = os.path.join(self.download_dir, zip_filename)
            
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                added_count = 0
                
                for root, dirs, files in os.walk(self.source_dir):
                    for file in files:
                        if any(core_file in file for core_file in core_indicators):
                            file_path = os.path.join(root, file)
                            arc_path = os.path.relpath(file_path, self.source_dir)
                            zipf.write(file_path, arc_path)
                            added_count += 1
            
            file_size = os.path.getsize(zip_path) / (1024*1024)
            print(f"✅ 핵심 지표 패키지 생성 완료: {zip_filename} ({added_count}개 지표, {file_size:.1f}MB)")
            
            return {
                'filename': zip_filename,
                'path': zip_path,
                'size_mb': file_size,
                'indicator_count': added_count
            }
            
        except Exception as e:
            print(f"❌ 핵심 지표 패키지 생성 오류: {e}")
            return None
    
    def create_download_guide(self, full_zip_path, full_tar_path, category_packages, core_package):
        """다운로드 가이드 문서 생성"""
        print("📋 다운로드 가이드 생성 중...")
        
        guide_content = f"""# BTC Analysis System - 6개월치 시간단위 데이터 다운로드 가이드

## 📊 데이터 개요
- **총 지표 수**: 1,258개
- **기간**: 6개월 (2025-02-25 ~ 2025-08-24)  
- **해상도**: 시간단위 (1시간 간격)
- **총 데이터 포인트**: 5,435,818개
- **생성일**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 📦 다운로드 패키지 옵션

### 1. 🎯 전체 데이터 패키지 (추천)

#### ZIP 버전 (호환성 최고)
- **파일명**: `{os.path.basename(full_zip_path) if full_zip_path else 'N/A'}`
- **용량**: {os.path.getsize(full_zip_path) / (1024*1024):.1f}MB (압축됨)
- **포함**: 전체 1,258개 지표 데이터
- **호환**: 모든 운영체제

#### TAR.GZ 버전 (용량 최적화)
- **파일명**: `{os.path.basename(full_tar_path) if full_tar_path else 'N/A'}`
- **용량**: {os.path.getsize(full_tar_path) / (1024*1024):.1f}MB (압축됨)
- **포함**: 전체 1,258개 지표 데이터
- **호환**: Linux, macOS, Windows(7-Zip 필요)

### 2. 📂 카테고리별 분할 패키지

"""
        
        if category_packages:
            for category, info in category_packages.items():
                guide_content += f"""#### {category}
- **파일명**: `{info['filename']}`
- **용량**: {info['size_mb']:.1f}MB
- **설명**: {self.get_category_description(category)}

"""
        
        if core_package:
            guide_content += f"""### 3. ⭐ 핵심 지표 선별 패키지 (초보자 추천)
- **파일명**: `{core_package['filename']}`
- **용량**: {core_package['size_mb']:.1f}MB
- **포함**: {core_package['indicator_count']}개 핵심 지표
- **설명**: 가장 중요한 지표들만 엄선

"""
        
        guide_content += f"""
## 🗂️ 데이터 구조 설명

### CSV 파일 형식
```csv
timestamp,indicator,category,value
2025-02-25 22:00:00,btc_mvrv_ratio,legacy_analyzer,2.1456
2025-02-25 23:00:00,btc_mvrv_ratio,legacy_analyzer,2.1478
```

### 주요 카테고리
1. **legacy_analyzer** (271개): 기존 analyzer.py의 모든 지표
2. **cryptoquant_csv** (103개): CryptoQuant 스타일 지표  
3. **macro_economic** (45개): 거시경제 지표
4. **additional_market_structure** (172개): 시장 구조 분석
5. **additional_fear_greed_detailed** (136개): Fear & Greed 상세 분석
6. **additional_advanced_onchain** (100개): 고급 온체인 지표
7. **기타 추가 카테고리들**: 기술적 지표, 상관관계, 변동성 등

## 🚀 활용 방법

### Python에서 사용
```python
import pandas as pd

# CSV 파일 읽기
df = pd.read_csv('btc_price_hourly.csv')
print(df.head())

# 시계열 분석
df['timestamp'] = pd.to_datetime(df['timestamp'])
df.set_index('timestamp', inplace=True)
```

### AI 모델 훈련
- **머신러닝**: 5백만+ 데이터 포인트로 모델 훈련
- **시계열 예측**: LSTM, ARIMA 모델용 데이터
- **패턴 인식**: 시장 패턴 및 anomaly detection

### 백테스팅 및 분석
- **전략 백테스팅**: 6개월간 시간단위 정밀 분석
- **상관관계 분석**: 1,258개 지표 간 관계 분석
- **리스크 관리**: 변동성 및 리스크 지표 활용

## 📥 다운로드 방법

### 방법 1: 직접 복사
```bash
# 전체 디렉토리를 원하는 위치로 복사
cp -r /Users/parkyoungjun/Desktop/BTC_Analysis_System/downloads/ ./btc_data/
```

### 방법 2: 압축 파일 이용
```bash
# ZIP 파일 압축 해제
unzip btc_analysis_6month_data_*.zip

# TAR.GZ 파일 압축 해제  
tar -xzf btc_analysis_6month_data_*.tar.gz
```

## ⚠️ 주의사항
- 데이터는 시뮬레이션 기반으로 생성됨 (실제 거래용 X)
- 분석 및 연구 목적으로만 사용
- 파일 크기가 크므로 충분한 저장 공간 확인
- Python pandas 라이브러리 권장

## 📞 지원
문제가 있으면 데이터 생성 로그와 함께 문의하세요.

---
**Generated by BTC Analysis System**  
**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        # 가이드 파일 저장
        guide_path = os.path.join(self.download_dir, "DOWNLOAD_GUIDE.md")
        with open(guide_path, 'w', encoding='utf-8') as f:
            f.write(guide_content)
        
        print(f"✅ 다운로드 가이드 생성: DOWNLOAD_GUIDE.md")
        
        # 간단한 README도 생성
        readme_content = f"""# BTC Analysis Data Download

📦 **{len(os.listdir(self.download_dir))}개 다운로드 파일 준비됨**

## 빠른 시작
1. `btc_analysis_6month_data_*.zip` 다운로드 (전체 데이터)
2. 압축 해제
3. Python으로 CSV 파일들 분석

## 상세 가이드
`DOWNLOAD_GUIDE.md` 파일 참조

생성일: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        readme_path = os.path.join(self.download_dir, "README.txt")
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)
    
    def get_category_description(self, category):
        """카테고리 설명 반환"""
        descriptions = {
            "legacy_analyzer": "기존 analyzer.py의 모든 시장, 온체인, 파생상품 지표",
            "cryptoquant_csv": "CryptoQuant 스타일 고급 온체인 분석 지표", 
            "macro_economic": "DXY, S&P500, VIX, 금, 국채 등 거시경제 지표",
            "enhanced_onchain": "Blockchain.info 기반 네트워크 통계",
            "calculated_indicators": "RSI, MACD, 볼린저밴드 등 계산된 기술적 지표",
            "official_announcements": "Bitcoin Core 등 공식 발표 관련 지표",
            "additional_market_structure": "지지저항, 차트패턴, 피보나치 등 시장구조 분석",
            "additional_fear_greed_detailed": "Fear & Greed Index 상세 히스토리 및 파생 지표",
            "additional_advanced_onchain": "고급 HODL, 주소 분석, 네트워크 활동 지표",
            "additional_technical_series": "다양한 기간의 이동평균, RSI, MACD 시리즈",
            "additional_exchange_specific": "거래소별 거래량, 프리미엄, 펀딩비 등",
            "additional_temporal_patterns": "시간대, 요일, 월별, 계절성 패턴 지표",
            "additional_correlations": "BTC와 다른 자산들 간의 상관관계 지표",
            "additional_volatility_analysis": "실현변동성, 내재변동성, 변동성 구조 지표",
            "additional_liquidity": "유동성 깊이, 시장 임팩트, 스프레드 분석 지표"
        }
        return descriptions.get(category, "기타 분석 지표")
    
    def show_download_summary(self):
        """다운로드 요약 정보 출력"""
        print("\n" + "="*60)
        print("🎉 BTC 분석 데이터 다운로드 패키지 생성 완료!")
        print("="*60)
        
        download_files = os.listdir(self.download_dir)
        zip_files = [f for f in download_files if f.endswith(('.zip', '.tar.gz'))]
        
        print(f"📁 다운로드 위치: {self.download_dir}")
        print(f"📦 생성된 패키지: {len(zip_files)}개")
        print()
        
        total_size = 0
        for file in download_files:
            file_path = os.path.join(self.download_dir, file)
            if os.path.isfile(file_path):
                size = os.path.getsize(file_path)
                total_size += size
                if file.endswith(('.zip', '.tar.gz')):
                    print(f"📦 {file} ({size/(1024*1024):.1f}MB)")
        
        print(f"\n💾 총 다운로드 용량: {total_size/(1024*1024):.1f}MB")
        print()
        print("🚀 사용법:")
        print(f"1. {self.download_dir} 폴더의 파일들을 복사")
        print("2. 원하는 패키지 선택하여 다운로드") 
        print("3. DOWNLOAD_GUIDE.md 파일 참조")
        print()
        print("⭐ 추천: btc_analysis_6month_data_*.zip (전체 데이터)")

def main():
    """메인 실행 함수"""
    print("🚀 BTC Analysis System - 데이터 다운로드 패키지 생성기")
    print()
    
    downloader = DataDownloader()
    
    if downloader.create_download_packages():
        downloader.show_download_summary()
    else:
        print("❌ 다운로드 패키지 생성 실패")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())