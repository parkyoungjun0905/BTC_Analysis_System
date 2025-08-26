#!/usr/bin/env python3
"""
기존 6개월 데이터에서 최근 3개월만 추출하는 스크립트
"""

import os
import pandas as pd
from datetime import datetime, timedelta
import shutil

def extract_3month_data():
    """6개월 데이터에서 최근 3개월만 추출"""
    
    source_dir = "/Users/parkyoungjun/Desktop/BTC_Analysis_System/complete_historical_6month_data"
    target_dir = "/Users/parkyoungjun/Desktop/BTC_Analysis_System/three_month_timeseries_data"
    
    # 3개월 기준 시점 계산
    three_months_ago = datetime.now() - timedelta(days=90)
    three_months_hours = 90 * 24  # 2160시간
    
    print(f"🗂️ 3개월 데이터 추출 시작...")
    print(f"📅 기준 시점: {three_months_ago.strftime('%Y-%m-%d %H:%M')}")
    print(f"⏱️ 목표 시간: {three_months_hours}시간")
    
    # 타겟 디렉토리 생성
    if os.path.exists(target_dir):
        print(f"🗑️ 기존 {target_dir} 삭제")
        shutil.rmtree(target_dir)
    
    os.makedirs(target_dir, exist_ok=True)
    
    extracted_count = 0
    total_files = 0
    
    # 소스 디렉토리의 모든 CSV 파일 처리
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.endswith('_hourly.csv'):
                total_files += 1
                source_path = os.path.join(root, file)
                
                # 상대 경로 계산하여 타겟에 동일 구조 생성
                relative_path = os.path.relpath(root, source_dir)
                target_subdir = os.path.join(target_dir, relative_path)
                os.makedirs(target_subdir, exist_ok=True)
                
                target_path = os.path.join(target_subdir, file)
                
                try:
                    # CSV 파일 읽기
                    df = pd.read_csv(source_path)
                    
                    if len(df) == 0:
                        continue
                    
                    # timestamp 컬럼이 있는지 확인
                    if 'timestamp' in df.columns:
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                        df = df.sort_values('timestamp')
                        
                        # 최근 3개월(2160시간)만 추출
                        df_3month = df.tail(three_months_hours)
                        
                        if len(df_3month) > 0:
                            df_3month.to_csv(target_path, index=False)
                            extracted_count += 1
                    else:
                        # timestamp 컬럼이 없는 경우 그대로 복사
                        df_3month = df.tail(three_months_hours)
                        df_3month.to_csv(target_path, index=False)
                        extracted_count += 1
                        
                except Exception as e:
                    print(f"⚠️ 파일 처리 오류 {file}: {e}")
                    continue
                
                # 진행상황 표시
                if extracted_count % 100 == 0:
                    print(f"📊 진행률: {extracted_count}/{total_files} 파일 처리됨")
    
    print(f"✅ 3개월 데이터 추출 완료!")
    print(f"📁 소스: {source_dir}")
    print(f"📁 타겟: {target_dir}")
    print(f"📊 처리 결과: {extracted_count}/{total_files} 파일 추출")
    
    return target_dir

if __name__ == "__main__":
    extract_3month_data()