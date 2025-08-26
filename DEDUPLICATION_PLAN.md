# 🔥 중복 지표 제거 계획

## 📊 현재 상황
- **전체 지표**: 2,208개
- **중복 그룹**: 19개 
- **가장 심각한 중복**: current_value (150+ 중복)

## 🎯 제거 전략

### 1️⃣ **해시레이트 & 난이도** (2개 지표)
```
유지: legacy_analyzer (더 안정적)
제거: enhanced_onchain.blockchain_info (중복)
```

### 2️⃣ **거시경제 지표** (심각한 중복)
```
문제: current_value, change_1d, high_7d, low_7d, volume_avg 모두 중복
해결: macro_economic만 유지, 다른 소스에서 제거
```

### 3️⃣ **Accumulated Timeseries** (가장 심각)
```
문제: 모든 지표를 다시 저장 (current_value, change_1d 등)
해결: timeseries 분석용으로만 사용, 중복 필드 제거
```

### 4️⃣ **CryptoQuant CSV**
```
문제: current_value 중복
해결: CryptoQuant는 고유 지표만 유지
```

## 🔧 구체적 제거 대상

### A. enhanced_onchain에서 제거
- blockchain_info.network_stats.hash_rate (→ legacy_analyzer 유지)
- blockchain_info.network_stats.difficulty (→ legacy_analyzer 유지)

### B. accumulated_timeseries에서 제거
- 모든 current_value 필드 (원본 소스에서 이미 제공)
- 모든 change_1d 필드 (macro_economic에서 이미 제공)
- 중복되는 기본 지표들 (price, volume, hash_rate 등)

### C. CryptoQuant CSV 정리
- 중복되는 기본 지표들만 제거 (hash_rate, difficulty 등)
- CryptoQuant 고유 지표는 유지

## 📈 예상 효과
- **제거 전**: 2,208개
- **제거 후**: 약 800-1,000개 (50% 감소)
- **중복 그룹**: 19개 → 0개