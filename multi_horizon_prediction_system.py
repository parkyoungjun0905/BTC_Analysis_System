#!/usr/bin/env python3
"""
🎯 Multi-Horizon Bitcoin Price Prediction System
정교한 다중 시간대 예측 시스템으로 90%+ 정확도 달성

주요 기능:
1. Multi-Task Learning Architecture - 공유 특성 인코더
2. Temporal Hierarchy Modeling - 장/중/단기 트렌드 분석
3. Uncertainty Quantification - 몬테카를로 드롭아웃과 앙상블 신뢰도
4. Dynamic Horizon Weighting - 시장 변동성 기반 시간대 최적화
5. Integration Strategies - 계층적 예측 통합

목표 시간대: 1h, 4h, 24h, 72h, 168h
목표 정확도: 90%+
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import warnings
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
import joblib
from dataclasses import dataclass
from collections import defaultdict
import logging

warnings.filterwarnings('ignore')

@dataclass
class HorizonConfig:
    """시간대별 설정"""
    horizon_hours: int
    weight: float
    lookback_periods: int
    feature_dims: int
    uncertainty_samples: int

class MultiHorizonDataset(Dataset):
    """다중 시간대 데이터셋"""
    
    def __init__(self, data: np.ndarray, horizons: List[int], lookback: int = 168):
        self.data = data
        self.horizons = horizons
        self.lookback = lookback
        self.samples = []
        
        # 다중 시간대 샘플 생성
        for i in range(lookback, len(data) - max(horizons)):
            sample = {
                'features': data[i-lookback:i],
                'targets': {}
            }
            
            for horizon in horizons:
                if i + horizon < len(data):
                    sample['targets'][horizon] = data[i + horizon, 0]  # 가격만 예측
            
            self.samples.append(sample)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        features = torch.FloatTensor(sample['features'])
        targets = {h: torch.FloatTensor([sample['targets'][h]]) for h in sample['targets']}
        return features, targets

class SharedFeatureEncoder(nn.Module):
    """공유 특성 인코더"""
    
    def __init__(self, input_dim: int, hidden_dims: List[int], dropout_rate: float = 0.2):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*layers)
        self.output_dim = prev_dim
    
    def forward(self, x):
        # x shape: (batch, sequence, features)
        batch_size, seq_len, features = x.shape
        
        # Flatten for processing
        x = x.view(batch_size * seq_len, features)
        encoded = self.encoder(x)
        
        # Reshape back
        encoded = encoded.view(batch_size, seq_len, -1)
        
        # Global pooling for sequence
        encoded = torch.mean(encoded, dim=1)
        
        return encoded

class TemporalAttention(nn.Module):
    """시간적 어텐션 메커니즘"""
    
    def __init__(self, input_dim: int, num_heads: int = 8):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.1
        )
        self.norm = nn.LayerNorm(input_dim)
    
    def forward(self, x):
        attended, weights = self.attention(x, x, x)
        return self.norm(attended + x), weights

class HorizonSpecificHead(nn.Module):
    """시간대별 예측 헤드"""
    
    def __init__(self, input_dim: int, hidden_dim: int, dropout_rate: float = 0.3):
        super().__init__()
        
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, x):
        return self.head(x)

class MultiHorizonPredictor(nn.Module):
    """다중 시간대 예측 모델"""
    
    def __init__(self, 
                 input_dim: int, 
                 horizons: List[int],
                 encoder_dims: List[int] = [512, 256, 128],
                 head_dim: int = 64,
                 dropout_rate: float = 0.2):
        super().__init__()
        
        self.horizons = horizons
        self.encoder = SharedFeatureEncoder(input_dim, encoder_dims, dropout_rate)
        self.attention = TemporalAttention(self.encoder.output_dim)
        
        # 시간대별 예측 헤드
        self.heads = nn.ModuleDict({
            str(h): HorizonSpecificHead(self.encoder.output_dim, head_dim, dropout_rate)
            for h in horizons
        })
        
        # 크로스 호라이즌 일관성 제약
        self.consistency_weight = nn.Parameter(torch.tensor(0.1))
    
    def forward(self, x, return_attention=False):
        # 공유 특성 인코딩
        encoded = self.encoder(x)
        
        # 시간적 어텐션 적용
        if len(encoded.shape) == 2:
            encoded = encoded.unsqueeze(1)  # Add sequence dimension
        
        attended, attention_weights = self.attention(encoded)
        attended = attended.squeeze(1)  # Remove sequence dimension
        
        # 시간대별 예측
        predictions = {}
        for horizon in self.horizons:
            predictions[horizon] = self.heads[str(horizon)](attended)
        
        if return_attention:
            return predictions, attention_weights
        return predictions
    
    def compute_consistency_loss(self, predictions):
        """크로스 호라이즌 일관성 손실 계산"""
        consistency_loss = 0.0
        sorted_horizons = sorted(self.horizons)
        
        for i in range(len(sorted_horizons) - 1):
            h1, h2 = sorted_horizons[i], sorted_horizons[i + 1]
            pred1, pred2 = predictions[h1], predictions[h2]
            
            # 단기 예측이 장기 예측과 일관성을 가져야 함
            consistency_loss += torch.mean(torch.abs(pred1 - pred2)) * (h2 - h1) / max(self.horizons)
        
        return self.consistency_weight * consistency_loss

class UncertaintyQuantifier:
    """불확실성 정량화 시스템"""
    
    def __init__(self, model: MultiHorizonPredictor, num_samples: int = 100):
        self.model = model
        self.num_samples = num_samples
    
    def monte_carlo_predictions(self, x: torch.Tensor) -> Dict[int, Dict[str, float]]:
        """몬테카를로 드롭아웃을 이용한 불확실성 추정"""
        self.model.train()  # 드롭아웃 활성화
        
        predictions = {h: [] for h in self.model.horizons}
        
        with torch.no_grad():
            for _ in range(self.num_samples):
                preds = self.model(x)
                for horizon in self.model.horizons:
                    predictions[horizon].append(preds[horizon].cpu().numpy())
        
        # 통계 계산
        results = {}
        for horizon in self.model.horizons:
            pred_array = np.array(predictions[horizon])
            results[horizon] = {
                'mean': float(np.mean(pred_array)),
                'std': float(np.std(pred_array)),
                'lower_95': float(np.percentile(pred_array, 2.5)),
                'upper_95': float(np.percentile(pred_array, 97.5)),
                'confidence': float(1.0 - (np.std(pred_array) / np.abs(np.mean(pred_array))))
            }
        
        return results

class TemporalHierarchyAnalyzer:
    """시간 계층 분석기"""
    
    def __init__(self):
        self.trend_analyzers = {
            'long_term': {'window': 168, 'weight': 0.5},    # 1주일 장기 트렌드
            'medium_term': {'window': 72, 'weight': 0.3},   # 3일 중기 트렌드  
            'short_term': {'window': 24, 'weight': 0.2}     # 1일 단기 트렌드
        }
    
    def analyze_trends(self, data: np.ndarray) -> Dict[str, float]:
        """다층 트렌드 분석"""
        trends = {}
        
        for trend_type, config in self.trend_analyzers.items():
            window = config['window']
            if len(data) >= window:
                recent_data = data[-window:]
                
                # 트렌드 강도 계산
                x = np.arange(len(recent_data))
                slope = np.polyfit(x, recent_data[:, 0], 1)[0]
                
                # R² 계산으로 트렌드 일관성 측정
                y_pred = slope * x + np.polyfit(x, recent_data[:, 0], 1)[1]
                r2 = 1 - np.sum((recent_data[:, 0] - y_pred) ** 2) / np.sum((recent_data[:, 0] - np.mean(recent_data[:, 0])) ** 2)
                
                trends[trend_type] = {
                    'slope': float(slope),
                    'consistency': float(max(0, r2)),
                    'volatility': float(np.std(recent_data[:, 0])),
                    'weight': config['weight']
                }
        
        return trends

class DynamicHorizonWeighter:
    """동적 시간대 가중치 조정기"""
    
    def __init__(self):
        self.volatility_threshold = 0.05
        self.performance_history = defaultdict(list)
    
    def update_performance(self, horizon: int, accuracy: float, volatility: float):
        """성능 기록 업데이트"""
        self.performance_history[horizon].append({
            'accuracy': accuracy,
            'volatility': volatility,
            'timestamp': datetime.now()
        })
        
        # 최근 100개 기록만 유지
        if len(self.performance_history[horizon]) > 100:
            self.performance_history[horizon].pop(0)
    
    def compute_dynamic_weights(self, current_volatility: float) -> Dict[int, float]:
        """현재 시장 상황에 따른 동적 가중치 계산"""
        base_weights = {1: 0.1, 4: 0.2, 24: 0.3, 72: 0.25, 168: 0.15}
        
        if current_volatility > self.volatility_threshold:
            # 고변동성 시장: 단기 예측에 더 많은 가중치
            adjusted_weights = {1: 0.25, 4: 0.3, 24: 0.25, 72: 0.15, 168: 0.05}
        else:
            # 저변동성 시장: 장기 예측에 더 많은 가중치
            adjusted_weights = {1: 0.05, 4: 0.1, 24: 0.25, 72: 0.3, 168: 0.3}
        
        # 과거 성능 기반 조정
        for horizon in base_weights:
            if horizon in self.performance_history and self.performance_history[horizon]:
                recent_performance = np.mean([p['accuracy'] for p in self.performance_history[horizon][-10:]])
                performance_multiplier = max(0.5, min(2.0, recent_performance / 0.8))  # 80% 기준
                adjusted_weights[horizon] *= performance_multiplier
        
        # 정규화
        total_weight = sum(adjusted_weights.values())
        return {h: w / total_weight for h, w in adjusted_weights.items()}

class MultiHorizonPredictionSystem:
    """완전한 다중 시간대 예측 시스템"""
    
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.horizons = [1, 4, 24, 72, 168]  # 시간 단위
        self.lookback = 168  # 1주일 룩백
        
        # 구성 요소 초기화
        self.model = None
        self.scaler = StandardScaler()
        self.uncertainty_quantifier = None
        self.temporal_analyzer = TemporalHierarchyAnalyzer()
        self.horizon_weighter = DynamicHorizonWeighter()
        
        # 성능 추적
        self.performance_history = defaultdict(list)
        self.setup_logging()
    
    def setup_logging(self):
        """로깅 설정"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('multi_horizon_prediction.log', encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def load_and_prepare_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """데이터 로드 및 전처리"""
        self.logger.info("🔍 데이터 로드 및 전처리 시작")
        
        # AI 최적화된 데이터 로드
        csv_path = os.path.join(self.data_path, "ai_optimized_3month_data", "ai_matrix_complete.csv")
        
        if not os.path.exists(csv_path):
            # 대체 데이터 경로 시도
            csv_path = os.path.join(self.data_path, "historical_6month_data", "btc_price_hourly.csv")
        
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {csv_path}")
        
        df = pd.read_csv(csv_path)
        self.logger.info(f"데이터 로드 완료: {df.shape}")
        
        # 시간 컬럼 처리
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
            df = df.sort_values('datetime')
        elif 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
        
        # 숫자형 컬럼만 선택
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df_numeric = df[numeric_columns].copy()
        
        # 결측치 처리
        df_numeric = df_numeric.ffill().bfill()
        
        # 이상치 제거 (IQR 방법)
        Q1 = df_numeric.quantile(0.25)
        Q3 = df_numeric.quantile(0.75)
        IQR = Q3 - Q1
        df_clean = df_numeric[~((df_numeric < (Q1 - 1.5 * IQR)) | (df_numeric > (Q3 + 1.5 * IQR))).any(axis=1)]
        
        # 특성과 타겟 분리
        if 'btc_price' in df_clean.columns:
            price_col = 'btc_price'
        elif 'close' in df_clean.columns:
            price_col = 'close'
        else:
            price_col = df_clean.columns[0]  # 첫 번째 컬럼을 가격으로 가정
        
        # 타겟을 첫 번째 컬럼으로 이동
        cols = [price_col] + [col for col in df_clean.columns if col != price_col]
        df_clean = df_clean[cols]
        
        # 정규화
        data_scaled = self.scaler.fit_transform(df_clean.values)
        
        # 훈련/검증 분할 (시간순 분할)
        train_size = int(0.8 * len(data_scaled))
        train_data = data_scaled[:train_size]
        test_data = data_scaled[train_size:]
        
        self.logger.info(f"훈련 데이터: {train_data.shape}, 테스트 데이터: {test_data.shape}")
        
        return train_data, test_data
    
    def create_model(self, input_dim: int) -> MultiHorizonPredictor:
        """모델 생성"""
        model = MultiHorizonPredictor(
            input_dim=input_dim,
            horizons=self.horizons,
            encoder_dims=[512, 256, 128],
            head_dim=64,
            dropout_rate=0.2
        )
        
        self.uncertainty_quantifier = UncertaintyQuantifier(model, num_samples=100)
        return model
    
    def train_model(self, train_data: np.ndarray, val_data: np.ndarray = None) -> Dict:
        """모델 훈련"""
        self.logger.info("🚀 Multi-Horizon 모델 훈련 시작")
        
        # 데이터셋 생성
        train_dataset = MultiHorizonDataset(train_data, self.horizons, self.lookback)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        
        if val_data is not None:
            val_dataset = MultiHorizonDataset(val_data, self.horizons, self.lookback)
            val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        # 모델 생성
        input_dim = train_data.shape[1]
        self.model = self.create_model(input_dim)
        
        # 옵티마이저 및 손실 함수
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)
        
        # 훈련 루프
        best_loss = float('inf')
        patience_counter = 0
        training_history = {'train_loss': [], 'val_loss': []}
        
        epochs = 100
        for epoch in range(epochs):
            # 훈련
            self.model.train()
            train_losses = []
            
            for features, targets in train_loader:
                optimizer.zero_grad()
                
                predictions = self.model(features)
                
                # 다중 시간대 손실 계산
                total_loss = 0.0
                for horizon in self.horizons:
                    if horizon in targets:
                        horizon_loss = F.mse_loss(predictions[horizon], targets[horizon])
                        total_loss += horizon_loss
                
                # 일관성 손실 추가
                consistency_loss = self.model.compute_consistency_loss(predictions)
                total_loss += consistency_loss
                
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_losses.append(total_loss.item())
            
            avg_train_loss = np.mean(train_losses)
            training_history['train_loss'].append(avg_train_loss)
            
            # 검증
            if val_data is not None:
                self.model.eval()
                val_losses = []
                
                with torch.no_grad():
                    for features, targets in val_loader:
                        predictions = self.model(features)
                        
                        total_loss = 0.0
                        for horizon in self.horizons:
                            if horizon in targets:
                                horizon_loss = F.mse_loss(predictions[horizon], targets[horizon])
                                total_loss += horizon_loss
                        
                        consistency_loss = self.model.compute_consistency_loss(predictions)
                        total_loss += consistency_loss
                        
                        val_losses.append(total_loss.item())
                
                avg_val_loss = np.mean(val_losses)
                training_history['val_loss'].append(avg_val_loss)
                
                # 스케줄러 업데이트
                scheduler.step(avg_val_loss)
                
                # 조기 종료 검사
                if avg_val_loss < best_loss:
                    best_loss = avg_val_loss
                    patience_counter = 0
                    # 모델 저장
                    torch.save(self.model.state_dict(), 'best_multi_horizon_model.pth')
                else:
                    patience_counter += 1
                
                if patience_counter >= 20:
                    self.logger.info(f"조기 종료: {epoch+1} 에포크")
                    break
                
                if (epoch + 1) % 10 == 0:
                    self.logger.info(f"에포크 {epoch+1}/{epochs}: 훈련 손실={avg_train_loss:.6f}, 검증 손실={avg_val_loss:.6f}")
        
        # 최적 모델 로드
        if val_data is not None:
            self.model.load_state_dict(torch.load('best_multi_horizon_model.pth'))
        
        self.logger.info("모델 훈련 완료")
        return training_history
    
    def predict_with_uncertainty(self, data: np.ndarray) -> Dict:
        """불확실성을 포함한 예측"""
        if self.model is None:
            raise ValueError("모델이 훈련되지 않았습니다.")
        
        self.model.eval()
        
        # 최근 데이터로 예측
        recent_data = torch.FloatTensor(data[-self.lookback:]).unsqueeze(0)
        
        # 확정적 예측
        with torch.no_grad():
            predictions = self.model(recent_data)
            deterministic_preds = {h: float(pred.item()) for h, pred in predictions.items()}
        
        # 불확실성 정량화
        uncertainty_results = self.uncertainty_quantifier.monte_carlo_predictions(recent_data)
        
        # 시간 계층 분석
        trend_analysis = self.temporal_analyzer.analyze_trends(data)
        
        # 현재 변동성 계산
        current_volatility = float(np.std(data[-24:, 0]))  # 최근 24시간 변동성
        
        # 동적 가중치 계산
        dynamic_weights = self.horizon_weighter.compute_dynamic_weights(current_volatility)
        
        return {
            'deterministic_predictions': deterministic_preds,
            'uncertainty_analysis': uncertainty_results,
            'trend_analysis': trend_analysis,
            'market_volatility': current_volatility,
            'dynamic_weights': dynamic_weights,
            'prediction_timestamp': datetime.now().isoformat()
        }
    
    def evaluate_performance(self, test_data: np.ndarray) -> Dict:
        """성능 평가"""
        self.logger.info("📊 모델 성능 평가 시작")
        
        if self.model is None:
            raise ValueError("모델이 훈련되지 않았습니다.")
        
        test_dataset = MultiHorizonDataset(test_data, self.horizons, self.lookback)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        
        horizon_results = {h: {'predictions': [], 'actuals': []} for h in self.horizons}
        
        self.model.eval()
        with torch.no_grad():
            for features, targets in test_loader:
                predictions = self.model(features)
                
                for horizon in self.horizons:
                    if horizon in targets:
                        pred_value = float(predictions[horizon].item())
                        actual_value = float(targets[horizon].item())
                        
                        horizon_results[horizon]['predictions'].append(pred_value)
                        horizon_results[horizon]['actuals'].append(actual_value)
        
        # 성능 메트릭 계산
        performance_metrics = {}
        for horizon in self.horizons:
            if horizon_results[horizon]['predictions']:
                preds = np.array(horizon_results[horizon]['predictions'])
                actuals = np.array(horizon_results[horizon]['actuals'])
                
                # 역정규화
                preds_denorm = self.scaler.inverse_transform(
                    np.column_stack([preds, np.zeros((len(preds), self.scaler.scale_.shape[0] - 1))])
                )[:, 0]
                actuals_denorm = self.scaler.inverse_transform(
                    np.column_stack([actuals, np.zeros((len(actuals), self.scaler.scale_.shape[0] - 1))])
                )[:, 0]
                
                # 메트릭 계산
                mae = mean_absolute_error(actuals_denorm, preds_denorm)
                rmse = np.sqrt(mean_squared_error(actuals_denorm, preds_denorm))
                
                # 방향 정확도 (상승/하락 예측 정확도)
                actual_changes = np.sign(np.diff(actuals_denorm))
                pred_changes = np.sign(np.diff(preds_denorm))
                direction_accuracy = np.mean(actual_changes == pred_changes) * 100
                
                # MAPE 계산
                mape = np.mean(np.abs((actuals_denorm - preds_denorm) / actuals_denorm)) * 100
                
                performance_metrics[f'{horizon}h'] = {
                    'mae': float(mae),
                    'rmse': float(rmse),
                    'direction_accuracy': float(direction_accuracy),
                    'mape': float(mape),
                    'samples': len(preds)
                }
                
                # 성능 기록 업데이트
                current_volatility = float(np.std(actuals_denorm))
                self.horizon_weighter.update_performance(horizon, direction_accuracy, current_volatility)
        
        # 전체 성능 요약
        overall_metrics = {
            'overall_direction_accuracy': np.mean([m['direction_accuracy'] for m in performance_metrics.values()]),
            'overall_mape': np.mean([m['mape'] for m in performance_metrics.values()]),
            'target_achieved': np.mean([m['direction_accuracy'] for m in performance_metrics.values()]) >= 90.0,
            'evaluation_timestamp': datetime.now().isoformat()
        }
        
        self.logger.info(f"전체 방향 정확도: {overall_metrics['overall_direction_accuracy']:.2f}%")
        self.logger.info(f"90% 목표 달성: {'✅ YES' if overall_metrics['target_achieved'] else '❌ NO'}")
        
        return {
            'horizon_metrics': performance_metrics,
            'overall_metrics': overall_metrics,
            'detailed_results': horizon_results
        }
    
    def save_system(self, filepath: str):
        """시스템 저장"""
        save_data = {
            'model_state': self.model.state_dict() if self.model else None,
            'scaler': self.scaler,
            'horizons': self.horizons,
            'lookback': self.lookback,
            'performance_history': dict(self.performance_history)
        }
        
        joblib.dump(save_data, filepath)
        self.logger.info(f"시스템 저장 완료: {filepath}")
    
    def load_system(self, filepath: str):
        """시스템 로드"""
        save_data = joblib.load(filepath)
        
        self.scaler = save_data['scaler']
        self.horizons = save_data['horizons']
        self.lookback = save_data['lookback']
        self.performance_history = defaultdict(list, save_data.get('performance_history', {}))
        
        if save_data['model_state']:
            # 모델 구조 재생성 필요
            input_dim = save_data['scaler'].scale_.shape[0]
            self.model = self.create_model(input_dim)
            self.model.load_state_dict(save_data['model_state'])
        
        self.logger.info(f"시스템 로드 완료: {filepath}")

def main():
    """메인 실행 함수"""
    system = MultiHorizonPredictionSystem("/Users/parkyoungjun/Desktop/BTC_Analysis_System")
    
    try:
        # 데이터 로드
        train_data, test_data = system.load_and_prepare_data()
        
        # 모델 훈련
        training_history = system.train_model(train_data, test_data)
        
        # 성능 평가
        performance_results = system.evaluate_performance(test_data)
        
        # 실시간 예측 예시
        prediction_results = system.predict_with_uncertainty(test_data)
        
        # 결과 저장
        results = {
            'training_history': training_history,
            'performance_results': performance_results,
            'prediction_example': prediction_results,
            'system_info': {
                'horizons': system.horizons,
                'target_accuracy': 90.0,
                'achieved_accuracy': performance_results['overall_metrics']['overall_direction_accuracy'],
                'target_met': performance_results['overall_metrics']['target_achieved']
            }
        }
        
        # JSON으로 결과 저장
        with open('multi_horizon_results.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # 시스템 저장
        system.save_system('multi_horizon_system.pkl')
        
        print("🎯 Multi-Horizon Bitcoin Prediction System")
        print("="*60)
        print(f"📊 전체 방향 정확도: {performance_results['overall_metrics']['overall_direction_accuracy']:.2f}%")
        print(f"🎯 90% 목표 달성: {'✅ YES' if performance_results['overall_metrics']['target_achieved'] else '❌ NO'}")
        print(f"📈 시간대별 성능:")
        
        for horizon, metrics in performance_results['horizon_metrics'].items():
            print(f"  {horizon}: {metrics['direction_accuracy']:.2f}% (MAPE: {metrics['mape']:.2f}%)")
        
        return results
        
    except Exception as e:
        system.logger.error(f"시스템 실행 중 오류 발생: {str(e)}")
        raise

if __name__ == "__main__":
    main()