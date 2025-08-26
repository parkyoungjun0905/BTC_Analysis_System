"""
🎯 90%+ 정확도 비트코인 예측 시스템
최첨단 딥러닝 아키텍처 기반 고정밀 예측기

참고 논문:
- Helformer: Attention-based Deep Learning Model for Cryptocurrency Price Forecasting (2025)
- Temporal Fusion Transformer-Based Trading Strategy (2024)
- Deep Learning-Enhanced Temporal Fusion Transformer (ADE-TFT)

Architecture Features:
1. Temporal Fusion Transformer with attention mechanisms
2. Helformer integration (Holt-Winters + Transformer)
3. Multi-scale CNN-LSTM hybrid architecture
4. Advanced feature engineering with 100+ indicators
5. Ensemble methods with dynamic weight allocation
6. Uncertainty quantification and confidence intervals
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_absolute_percentage_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# 추가 라이브러리
from typing import Dict, List, Tuple, Optional
import json
import sqlite3
from datetime import datetime, timedelta
import logging
import optuna
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HoltWintersIntegrator:
    """
    Helformer의 핵심: Holt-Winters 지수평활법을 Transformer와 통합
    시계열 데이터를 level, trend, seasonality로 분해
    """
    def __init__(self, seasonal_periods: int = 24, alpha: float = 0.3, beta: float = 0.3, gamma: float = 0.3):
        self.seasonal_periods = seasonal_periods
        self.alpha = alpha  # level smoothing parameter
        self.beta = beta    # trend smoothing parameter
        self.gamma = gamma  # seasonal smoothing parameter
        
    def decompose(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        시계열 데이터를 level, trend, seasonality 성분으로 분해
        
        Args:
            data: 시계열 데이터 (1D array)
            
        Returns:
            Dict containing level, trend, and seasonal components
        """
        n = len(data)
        level = np.zeros(n)
        trend = np.zeros(n)
        seasonal = np.zeros(n + self.seasonal_periods)
        
        # 초기값 설정
        level[0] = data[0]
        trend[0] = (data[1] - data[0]) if n > 1 else 0
        
        # 초기 계절성 추정
        if n >= self.seasonal_periods:
            for i in range(self.seasonal_periods):
                seasonal[i] = data[i] / level[0] if level[0] != 0 else 1
        else:
            seasonal[:self.seasonal_periods] = 1
        
        # Holt-Winters 알고리즘 적용
        for t in range(1, n):
            prev_level = level[t-1]
            prev_trend = trend[t-1]
            prev_seasonal = seasonal[t-1]
            
            # Level update
            if prev_seasonal != 0:
                level[t] = self.alpha * (data[t] / prev_seasonal) + (1 - self.alpha) * (prev_level + prev_trend)
            else:
                level[t] = self.alpha * data[t] + (1 - self.alpha) * (prev_level + prev_trend)
            
            # Trend update
            trend[t] = self.beta * (level[t] - prev_level) + (1 - self.beta) * prev_trend
            
            # Seasonal update
            if level[t] != 0:
                seasonal[t] = self.gamma * (data[t] / level[t]) + (1 - self.gamma) * prev_seasonal
            else:
                seasonal[t] = prev_seasonal
        
        return {
            'level': level,
            'trend': trend,
            'seasonal': seasonal[:n]
        }

class MultiHeadAttention(nn.Module):
    """
    Temporal Fusion Transformer의 멀티헤드 어텐션
    시간 의존성과 변수 간 상관관계를 동시에 학습
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        residual = query
        
        # Linear transformations
        Q = self.w_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        output = self.w_o(context)
        output = self.layer_norm(output + residual)
        
        return output, attention_weights

class GatedResidualNetwork(nn.Module):
    """
    Temporal Fusion Transformer의 핵심 구성요소
    비선형 변환과 gating mechanism을 통한 특성 선택
    """
    def __init__(self, input_size: int, hidden_size: int, output_size: int, dropout: float = 0.1):
        super().__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear_out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
        
        # Gating mechanism
        self.gate_linear = nn.Linear(hidden_size, output_size)
        
        # Skip connection
        if input_size != output_size:
            self.skip_linear = nn.Linear(input_size, output_size)
        else:
            self.skip_linear = nn.Identity()
            
        self.layer_norm = nn.LayerNorm(output_size)
        
    def forward(self, x):
        # Skip connection
        skip = self.skip_linear(x)
        
        # Non-linear transformation
        hidden = self.dropout(F.elu(self.linear1(x)))
        hidden = self.dropout(F.elu(self.linear2(hidden)))
        
        # Output transformation
        output = self.linear_out(hidden)
        
        # Gating
        gate = torch.sigmoid(self.gate_linear(hidden))
        output = gate * output
        
        # Residual connection and normalization
        output = self.layer_norm(output + skip)
        
        return output

class VariableSelectionNetwork(nn.Module):
    """
    입력 변수의 중요도를 동적으로 결정하는 네트워크
    100+ 지표 중 중요한 특성만 선택하여 과적합 방지
    """
    def __init__(self, input_size: int, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Variable selection weights
        self.flattened_grn = GatedResidualNetwork(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=input_size,
            dropout=dropout
        )
        
        self.single_variable_grns = nn.ModuleList([
            GatedResidualNetwork(
                input_size=1,
                hidden_size=hidden_size,
                output_size=hidden_size,
                dropout=dropout
            ) for _ in range(input_size)
        ])
        
    def forward(self, flattened_embedding):
        # Variable selection weights
        sparse_weights = self.flattened_grn(flattened_embedding)
        sparse_weights = F.softmax(sparse_weights, dim=-1)
        
        # Transform individual variables
        var_outputs = []
        for i, grn in enumerate(self.single_variable_grns):
            var_input = flattened_embedding[..., i:i+1]
            var_output = grn(var_input)
            var_outputs.append(var_output)
        
        var_outputs = torch.stack(var_outputs, dim=-1)
        
        # Weight and combine
        outputs = var_outputs * sparse_weights.unsqueeze(-2)
        outputs = torch.sum(outputs, dim=-1)
        
        return outputs, sparse_weights

class TemporalFusionTransformer(nn.Module):
    """
    최첨단 Temporal Fusion Transformer 구현
    - Multi-horizon forecasting
    - Attention-based feature selection
    - Interpretable predictions with confidence intervals
    """
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        output_horizon: int = 24,  # 24시간 예측
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.output_horizon = output_horizon
        
        # Variable selection network
        self.variable_selection = VariableSelectionNetwork(
            input_size=input_size,
            hidden_size=hidden_size,
            dropout=dropout
        )
        
        # LSTM encoder for temporal patterns
        self.lstm_encoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=dropout
        )
        
        # Multi-head attention layers
        self.attention_layers = nn.ModuleList([
            MultiHeadAttention(hidden_size, n_heads, dropout) 
            for _ in range(n_layers)
        ])
        
        # Position encoding
        self.position_encoding = nn.Parameter(torch.randn(1000, hidden_size))
        
        # Output networks
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_horizon)
        )
        
        # Quantile prediction for uncertainty
        self.quantile_layers = nn.ModuleDict({
            '0.1': nn.Linear(hidden_size, output_horizon),
            '0.5': nn.Linear(hidden_size, output_horizon), 
            '0.9': nn.Linear(hidden_size, output_horizon)
        })
        
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # Variable selection
        selected_features, selection_weights = self.variable_selection(x.reshape(batch_size * seq_len, -1))
        selected_features = selected_features.reshape(batch_size, seq_len, -1)
        
        # Position encoding
        pos_enc = self.position_encoding[:seq_len, :].unsqueeze(0).expand(batch_size, -1, -1)
        selected_features = selected_features + pos_enc
        
        # LSTM encoding
        lstm_out, (hidden, cell) = self.lstm_encoder(selected_features)
        
        # Multi-head attention
        attention_out = lstm_out
        for attention_layer in self.attention_layers:
            attention_out, _ = attention_layer(attention_out, attention_out, attention_out)
        
        # Global pooling
        global_context = torch.mean(attention_out, dim=1)
        
        # Predictions
        point_forecast = self.output_layer(global_context)
        
        # Quantile predictions for uncertainty
        quantile_forecasts = {}
        for quantile, layer in self.quantile_layers.items():
            quantile_forecasts[quantile] = layer(global_context)
        
        return {
            'point_forecast': point_forecast,
            'quantile_forecasts': quantile_forecasts,
            'selection_weights': selection_weights.reshape(batch_size, seq_len, -1),
            'attention_weights': attention_out
        }

class CNNLSTMHybrid(nn.Module):
    """
    CNN-LSTM 하이브리드 아키텍처
    다중 스케일 패턴 인식과 시간적 의존성 학습을 결합
    """
    def __init__(
        self,
        input_size: int,
        cnn_channels: List[int] = [64, 128, 256],
        kernel_sizes: List[int] = [3, 5, 7],
        lstm_hidden: int = 256,
        lstm_layers: int = 2,
        output_size: int = 24,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        
        # Multi-scale CNN layers
        self.conv_layers = nn.ModuleList()
        for i, (channels, kernel_size) in enumerate(zip(cnn_channels, kernel_sizes)):
            if i == 0:
                in_channels = 1
            else:
                in_channels = cnn_channels[i-1]
                
            conv_block = nn.Sequential(
                nn.Conv1d(in_channels, channels, kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(channels),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Conv1d(channels, channels, kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(channels), 
                nn.ReLU(),
                nn.MaxPool1d(2)
            )
            self.conv_layers.append(conv_block)
        
        # Calculate CNN output size
        cnn_output_size = cnn_channels[-1] * (input_size // (2 ** len(cnn_channels)))
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=cnn_output_size,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        
        # Output layers
        self.output_layer = nn.Sequential(
            nn.Linear(lstm_hidden * 2, lstm_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, output_size)
        )
        
    def forward(self, x):
        batch_size, seq_len, features = x.shape
        
        # Reshape for CNN (batch_size, channels, sequence)
        x = x.transpose(1, 2).unsqueeze(2)  # Add channel dimension
        x = x.view(-1, 1, features)
        
        # CNN feature extraction
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        
        # Reshape for LSTM
        x = x.view(batch_size, seq_len, -1)
        
        # LSTM temporal modeling
        lstm_out, _ = self.lstm(x)
        
        # Global pooling and prediction
        global_features = torch.mean(lstm_out, dim=1)
        output = self.output_layer(global_features)
        
        return output

class AdvancedFeatureEngineer:
    """
    100+ 지표를 활용한 고급 특성 엔지니어링
    """
    def __init__(self):
        self.scalers = {}
        self.feature_names = []
        
    def create_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        고급 기술적 지표 생성
        """
        features_df = df.copy()
        
        # 다중 시간축 이동평균
        for window in [5, 10, 20, 50, 100, 200]:
            features_df[f'sma_{window}'] = df['close'].rolling(window).mean()
            features_df[f'ema_{window}'] = df['close'].ewm(span=window).mean()
            
        # 볼린저 밴드 (다중 편차)
        for window in [20, 50]:
            for std_dev in [1.5, 2.0, 2.5]:
                rolling_mean = df['close'].rolling(window).mean()
                rolling_std = df['close'].rolling(window).std()
                features_df[f'bb_upper_{window}_{std_dev}'] = rolling_mean + (rolling_std * std_dev)
                features_df[f'bb_lower_{window}_{std_dev}'] = rolling_mean - (rolling_std * std_dev)
                features_df[f'bb_width_{window}_{std_dev}'] = features_df[f'bb_upper_{window}_{std_dev}'] - features_df[f'bb_lower_{window}_{std_dev}']
                features_df[f'bb_position_{window}_{std_dev}'] = (df['close'] - features_df[f'bb_lower_{window}_{std_dev}']) / features_df[f'bb_width_{window}_{std_dev}']
        
        # RSI (다중 기간)
        for period in [7, 14, 21, 30]:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            features_df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        
        # MACD 변형
        for fast in [8, 12, 16]:
            for slow in [21, 26, 35]:
                if fast < slow:
                    exp1 = df['close'].ewm(span=fast).mean()
                    exp2 = df['close'].ewm(span=slow).mean()
                    features_df[f'macd_{fast}_{slow}'] = exp1 - exp2
                    features_df[f'macd_signal_{fast}_{slow}'] = features_df[f'macd_{fast}_{slow}'].ewm(span=9).mean()
                    features_df[f'macd_histogram_{fast}_{slow}'] = features_df[f'macd_{fast}_{slow}'] - features_df[f'macd_signal_{fast}_{slow}']
        
        # 스토캐스틱 오실레이터
        for k_period in [14, 21]:
            for d_period in [3, 5]:
                low_min = df['low'].rolling(window=k_period).min()
                high_max = df['high'].rolling(window=k_period).max()
                features_df[f'stoch_k_{k_period}_{d_period}'] = 100 * ((df['close'] - low_min) / (high_max - low_min))
                features_df[f'stoch_d_{k_period}_{d_period}'] = features_df[f'stoch_k_{k_period}_{d_period}'].rolling(window=d_period).mean()
        
        # ATR (Average True Range)
        for period in [7, 14, 21]:
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            true_range = np.maximum(high_low, np.maximum(high_close, low_close))
            features_df[f'atr_{period}'] = true_range.rolling(window=period).mean()
        
        return features_df
    
    def create_market_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        마켓 마이크로스트럭처 특성 생성
        """
        features_df = df.copy()
        
        # 가격 변화율과 변동성
        for period in [1, 3, 6, 12, 24]:
            features_df[f'return_{period}h'] = df['close'].pct_change(period)
            features_df[f'volatility_{period}h'] = features_df[f'return_{period}h'].rolling(window=24).std()
            
        # 거래량 기반 지표
        features_df['volume_sma_24'] = df['volume'].rolling(window=24).mean()
        features_df['volume_ratio'] = df['volume'] / features_df['volume_sma_24']
        features_df['price_volume_trend'] = ((df['close'] - df['close'].shift()) / df['close'].shift()) * df['volume']
        
        # 가격 갭과 바디/그림자 비율
        features_df['price_gap'] = (df['open'] - df['close'].shift()) / df['close'].shift()
        features_df['upper_shadow'] = (df['high'] - np.maximum(df['open'], df['close'])) / df['close']
        features_df['lower_shadow'] = (np.minimum(df['open'], df['close']) - df['low']) / df['close']
        features_df['body_size'] = np.abs(df['close'] - df['open']) / df['close']
        
        return features_df
        
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        시간적 특성 생성
        """
        features_df = df.copy()
        
        # 시간 기반 특성
        features_df['hour'] = pd.to_datetime(df.index).hour
        features_df['day_of_week'] = pd.to_datetime(df.index).dayofweek
        features_df['month'] = pd.to_datetime(df.index).month
        features_df['quarter'] = pd.to_datetime(df.index).quarter
        
        # 주기적 인코딩
        features_df['hour_sin'] = np.sin(2 * np.pi * features_df['hour'] / 24)
        features_df['hour_cos'] = np.cos(2 * np.pi * features_df['hour'] / 24)
        features_df['day_sin'] = np.sin(2 * np.pi * features_df['day_of_week'] / 7)
        features_df['day_cos'] = np.cos(2 * np.pi * features_df['day_of_week'] / 7)
        features_df['month_sin'] = np.sin(2 * np.pi * features_df['month'] / 12)
        features_df['month_cos'] = np.cos(2 * np.pi * features_df['month'] / 12)
        
        return features_df

class EnsemblePredictor:
    """
    다중 모델 앙상블 예측기
    동적 가중치 할당과 불확실성 정량화
    """
    def __init__(self, models: List[nn.Module], weights: Optional[List[float]] = None):
        self.models = models
        self.num_models = len(models)
        
        if weights is None:
            self.weights = np.ones(self.num_models) / self.num_models
        else:
            self.weights = np.array(weights) / np.sum(weights)
            
        self.performance_history = []
        
    def predict(self, x: torch.Tensor) -> Dict:
        """
        앙상블 예측 수행
        """
        predictions = []
        uncertainties = []
        
        for model in self.models:
            model.eval()
            with torch.no_grad():
                if isinstance(model, TemporalFusionTransformer):
                    output = model(x)
                    pred = output['point_forecast']
                    # Uncertainty from quantile predictions
                    q_10 = output['quantile_forecasts']['0.1']
                    q_90 = output['quantile_forecasts']['0.9']
                    uncertainty = (q_90 - q_10) / 2  # IQR/2 as uncertainty measure
                    uncertainties.append(uncertainty)
                else:
                    pred = model(x)
                    # Simple uncertainty estimate based on model complexity
                    uncertainty = torch.std(pred, dim=-1, keepdim=True) * 0.1
                    uncertainties.append(uncertainty)
                    
                predictions.append(pred)
        
        # Weighted ensemble
        predictions = torch.stack(predictions)
        uncertainties = torch.stack(uncertainties)
        
        weights_tensor = torch.tensor(self.weights).view(-1, 1, 1).to(predictions.device)
        ensemble_pred = torch.sum(predictions * weights_tensor, dim=0)
        ensemble_uncertainty = torch.sqrt(torch.sum(uncertainties**2 * weights_tensor, dim=0))
        
        return {
            'prediction': ensemble_pred,
            'uncertainty': ensemble_uncertainty,
            'individual_predictions': predictions,
            'confidence_interval': {
                'lower': ensemble_pred - 1.96 * ensemble_uncertainty,
                'upper': ensemble_pred + 1.96 * ensemble_uncertainty
            }
        }
    
    def update_weights(self, predictions: np.ndarray, actual: np.ndarray):
        """
        성능 기반 가중치 업데이트
        """
        individual_errors = []
        
        for i in range(self.num_models):
            pred = predictions[i].detach().cpu().numpy() if torch.is_tensor(predictions[i]) else predictions[i]
            error = mean_absolute_percentage_error(actual, pred)
            individual_errors.append(error)
        
        # 역 오차 기반 가중치 (낮은 오차에 높은 가중치)
        inv_errors = 1 / (np.array(individual_errors) + 1e-8)
        self.weights = inv_errors / np.sum(inv_errors)
        
        self.performance_history.append({
            'timestamp': datetime.now(),
            'errors': individual_errors,
            'weights': self.weights.copy()
        })

class AdvancedBTCDataset(Dataset):
    """
    고급 비트코인 데이터셋
    Holt-Winters 분해와 다중 스케일 특성 포함
    """
    def __init__(
        self,
        data: pd.DataFrame,
        sequence_length: int = 168,  # 1주 = 168시간
        prediction_horizon: int = 24,  # 24시간 예측
        features: Optional[List[str]] = None,
        use_holt_winters: bool = True
    ):
        self.data = data.copy()
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.use_holt_winters = use_holt_winters
        
        # Feature engineering
        self.feature_engineer = AdvancedFeatureEngineer()
        
        # 기본 특성 생성
        self.data = self.feature_engineer.create_technical_features(self.data)
        self.data = self.feature_engineer.create_market_microstructure_features(self.data)
        self.data = self.feature_engineer.create_temporal_features(self.data)
        
        # Holt-Winters 분해
        if self.use_holt_winters:
            hw_integrator = HoltWintersIntegrator()
            hw_components = hw_integrator.decompose(self.data['close'].values)
            
            self.data['hw_level'] = hw_components['level']
            self.data['hw_trend'] = hw_components['trend']  
            self.data['hw_seasonal'] = hw_components['seasonal']
        
        # 특성 선택
        if features is None:
            # 수치형 컬럼만 선택 (NaN 제거)
            numeric_columns = self.data.select_dtypes(include=[np.number]).columns
            self.features = [col for col in numeric_columns if col != 'close']
        else:
            self.features = features
            
        # 데이터 정규화
        self.scaler = RobustScaler()
        feature_data = self.data[self.features].fillna(method='ffill').fillna(0)
        self.scaled_features = self.scaler.fit_transform(feature_data)
        
        # 타겟 정규화
        self.target_scaler = StandardScaler()
        self.scaled_targets = self.target_scaler.fit_transform(
            self.data['close'].values.reshape(-1, 1)
        ).flatten()
        
        # 유효한 인덱스 계산
        self.valid_indices = list(range(
            self.sequence_length, 
            len(self.data) - self.prediction_horizon
        ))
        
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        actual_idx = self.valid_indices[idx]
        
        # 입력 시퀀스
        start_idx = actual_idx - self.sequence_length
        end_idx = actual_idx
        
        x = torch.FloatTensor(self.scaled_features[start_idx:end_idx])
        
        # 타겟 (미래 prediction_horizon 시간)
        target_start = actual_idx
        target_end = actual_idx + self.prediction_horizon
        
        y = torch.FloatTensor(self.scaled_targets[target_start:target_end])
        
        return x, y
    
    def inverse_transform_target(self, scaled_target):
        """
        정규화된 타겟을 원래 스케일로 변환
        """
        if torch.is_tensor(scaled_target):
            scaled_target = scaled_target.detach().cpu().numpy()
            
        return self.target_scaler.inverse_transform(
            scaled_target.reshape(-1, 1)
        ).flatten()

class Advanced90PercentPredictor:
    """
    90%+ 정확도를 목표로 하는 최첨단 비트코인 예측기
    """
    def __init__(
        self,
        input_size: int,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.device = device
        self.input_size = input_size
        
        # 모델 초기화
        self.tft_model = TemporalFusionTransformer(
            input_size=input_size,
            hidden_size=512,
            n_heads=16,
            n_layers=6,
            output_horizon=24,
            dropout=0.1
        ).to(device)
        
        self.cnn_lstm_model = CNNLSTMHybrid(
            input_size=input_size,
            cnn_channels=[128, 256, 512],
            kernel_sizes=[3, 5, 7],
            lstm_hidden=512,
            lstm_layers=3,
            output_size=24,
            dropout=0.1
        ).to(device)
        
        # 앙상블 예측기
        self.ensemble = EnsemblePredictor([self.tft_model, self.cnn_lstm_model])
        
        # 최적화 관련
        self.optimizers = {
            'tft': torch.optim.AdamW(self.tft_model.parameters(), lr=1e-4, weight_decay=1e-5),
            'cnn_lstm': torch.optim.AdamW(self.cnn_lstm_model.parameters(), lr=1e-4, weight_decay=1e-5)
        }
        
        self.schedulers = {
            'tft': torch.optim.lr_scheduler.OneCycleLR(
                self.optimizers['tft'], max_lr=1e-3, epochs=100, steps_per_epoch=100
            ),
            'cnn_lstm': torch.optim.lr_scheduler.OneCycleLR(
                self.optimizers['cnn_lstm'], max_lr=1e-3, epochs=100, steps_per_epoch=100
            )
        }
        
        # 훈련 기록
        self.training_history = {
            'tft': {'train_loss': [], 'val_loss': [], 'accuracy': []},
            'cnn_lstm': {'train_loss': [], 'val_loss': [], 'accuracy': []}
        }
        
    def quantile_loss(self, predictions, targets, quantiles):
        """
        분위수 손실 함수 (불확실성 정량화용)
        """
        losses = []
        for i, q in enumerate(quantiles):
            errors = targets - predictions[:, i]
            losses.append(torch.max((q-1) * errors, q * errors))
        return torch.mean(torch.stack(losses))
    
    def directional_accuracy_loss(self, predictions, targets, alpha=0.5):
        """
        방향 정확도를 고려한 손실 함수
        """
        # 가격 변화 방향
        pred_direction = torch.sign(predictions[:, 1:] - predictions[:, :-1])
        target_direction = torch.sign(targets[:, 1:] - targets[:, :-1])
        
        # 방향 일치 보너스
        direction_match = (pred_direction * target_direction > 0).float()
        direction_bonus = torch.mean(direction_match)
        
        # 일반 MSE 손실
        mse_loss = F.mse_loss(predictions, targets)
        
        # 결합된 손실
        return (1 - alpha) * mse_loss - alpha * direction_bonus
    
    def train_epoch(self, model, train_loader, optimizer, model_name):
        """
        단일 에포크 훈련
        """
        model.train()
        total_loss = 0
        num_batches = 0
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            optimizer.zero_grad()
            
            if model_name == 'tft':
                output = model(batch_x)
                prediction = output['point_forecast']
                
                # 메인 예측 손실
                main_loss = self.directional_accuracy_loss(prediction, batch_y)
                
                # 분위수 손실
                quantile_preds = torch.stack([
                    output['quantile_forecasts']['0.1'],
                    output['quantile_forecasts']['0.5'], 
                    output['quantile_forecasts']['0.9']
                ], dim=-1)
                quantile_loss = self.quantile_loss(
                    quantile_preds, batch_y, [0.1, 0.5, 0.9]
                )
                
                total_loss_batch = main_loss + 0.1 * quantile_loss
                
            else:  # cnn_lstm
                prediction = model(batch_x)
                total_loss_batch = self.directional_accuracy_loss(prediction, batch_y)
            
            total_loss_batch.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += total_loss_batch.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def validate(self, model, val_loader, model_name):
        """
        검증 수행
        """
        model.eval()
        total_loss = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                if model_name == 'tft':
                    output = model(batch_x)
                    prediction = output['point_forecast']
                else:
                    prediction = model(batch_x)
                
                loss = F.mse_loss(prediction, batch_y)
                total_loss += loss.item()
                
                all_predictions.append(prediction.cpu())
                all_targets.append(batch_y.cpu())
        
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # 방향 정확도 계산
        pred_direction = torch.sign(all_predictions[:, 1:] - all_predictions[:, :-1])
        target_direction = torch.sign(all_targets[:, 1:] - all_targets[:, :-1])
        direction_accuracy = torch.mean((pred_direction * target_direction > 0).float()).item()
        
        return total_loss / len(val_loader), direction_accuracy
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 100,
        patience: int = 15
    ):
        """
        모델 훈련
        """
        logger.info("🚀 고급 90% 정확도 예측 모델 훈련 시작")
        
        best_val_loss = {'tft': float('inf'), 'cnn_lstm': float('inf')}
        patience_counter = {'tft': 0, 'cnn_lstm': 0}
        
        for epoch in range(epochs):
            # TFT 모델 훈련
            tft_train_loss = self.train_epoch(
                self.tft_model, train_loader, self.optimizers['tft'], 'tft'
            )
            tft_val_loss, tft_accuracy = self.validate(self.tft_model, val_loader, 'tft')
            
            # CNN-LSTM 모델 훈련  
            cnn_train_loss = self.train_epoch(
                self.cnn_lstm_model, train_loader, self.optimizers['cnn_lstm'], 'cnn_lstm'
            )
            cnn_val_loss, cnn_accuracy = self.validate(self.cnn_lstm_model, val_loader, 'cnn_lstm')
            
            # 학습률 스케줄러 업데이트
            self.schedulers['tft'].step()
            self.schedulers['cnn_lstm'].step()
            
            # 기록 업데이트
            self.training_history['tft']['train_loss'].append(tft_train_loss)
            self.training_history['tft']['val_loss'].append(tft_val_loss)
            self.training_history['tft']['accuracy'].append(tft_accuracy)
            
            self.training_history['cnn_lstm']['train_loss'].append(cnn_train_loss)
            self.training_history['cnn_lstm']['val_loss'].append(cnn_val_loss)
            self.training_history['cnn_lstm']['accuracy'].append(cnn_accuracy)
            
            # Early stopping check
            for model_name in ['tft', 'cnn_lstm']:
                if model_name == 'tft':
                    current_val_loss = tft_val_loss
                else:
                    current_val_loss = cnn_val_loss
                    
                if current_val_loss < best_val_loss[model_name]:
                    best_val_loss[model_name] = current_val_loss
                    patience_counter[model_name] = 0
                    # 최고 모델 저장
                    if model_name == 'tft':
                        torch.save(self.tft_model.state_dict(), f'best_{model_name}_model.pth')
                    else:
                        torch.save(self.cnn_lstm_model.state_dict(), f'best_{model_name}_model.pth')
                else:
                    patience_counter[model_name] += 1
            
            # 로깅
            if epoch % 10 == 0:
                logger.info(
                    f"Epoch {epoch}: "
                    f"TFT - Train: {tft_train_loss:.4f}, Val: {tft_val_loss:.4f}, Acc: {tft_accuracy:.4f} | "
                    f"CNN-LSTM - Train: {cnn_train_loss:.4f}, Val: {cnn_val_loss:.4f}, Acc: {cnn_accuracy:.4f}"
                )
            
            # Early stopping
            if all(patience_counter[name] >= patience for name in ['tft', 'cnn_lstm']):
                logger.info(f"Early stopping at epoch {epoch}")
                break
        
        # 최고 모델 로드
        self.tft_model.load_state_dict(torch.load('best_tft_model.pth'))
        self.cnn_lstm_model.load_state_dict(torch.load('best_cnn_lstm_model.pth'))
        
        logger.info("✅ 모델 훈련 완료")
    
    def predict_with_confidence(self, x: torch.Tensor) -> Dict:
        """
        신뢰구간을 포함한 예측
        """
        x = x.to(self.device)
        
        # 앙상블 예측
        ensemble_output = self.ensemble.predict(x)
        
        return {
            'prediction': ensemble_output['prediction'].cpu().numpy(),
            'uncertainty': ensemble_output['uncertainty'].cpu().numpy(),
            'confidence_interval': {
                'lower': ensemble_output['confidence_interval']['lower'].cpu().numpy(),
                'upper': ensemble_output['confidence_interval']['upper'].cpu().numpy()
            }
        }
    
    def evaluate_accuracy(self, test_loader: DataLoader, dataset: AdvancedBTCDataset) -> Dict:
        """
        종합적인 정확도 평가
        """
        self.tft_model.eval()
        self.cnn_lstm_model.eval()
        
        all_predictions = []
        all_targets = []
        all_uncertainties = []
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(self.device)
                
                # 앙상블 예측
                ensemble_output = self.ensemble.predict(batch_x)
                
                all_predictions.append(ensemble_output['prediction'].cpu())
                all_targets.append(batch_y)
                all_uncertainties.append(ensemble_output['uncertainty'].cpu())
        
        predictions = torch.cat(all_predictions, dim=0)
        targets = torch.cat(all_targets, dim=0)
        uncertainties = torch.cat(all_uncertainties, dim=0)
        
        # 스케일 복원
        pred_original = []
        target_original = []
        
        for i in range(predictions.shape[0]):
            pred_orig = dataset.inverse_transform_target(predictions[i])
            target_orig = dataset.inverse_transform_target(targets[i])
            
            pred_original.append(pred_orig)
            target_original.append(target_orig)
        
        pred_original = np.array(pred_original)
        target_original = np.array(target_original)
        
        # 다양한 정확도 지표 계산
        mape = mean_absolute_percentage_error(target_original.flatten(), pred_original.flatten())
        r2 = r2_score(target_original.flatten(), pred_original.flatten())
        
        # 방향 정확도
        pred_direction = np.sign(pred_original[:, 1:] - pred_original[:, :-1])
        target_direction = np.sign(target_original[:, 1:] - target_original[:, :-1])
        direction_accuracy = np.mean(pred_direction * target_direction > 0)
        
        # 가격 범위별 정확도
        price_changes = target_original[:, -1] - target_original[:, 0]
        small_changes = np.abs(price_changes) < np.percentile(np.abs(price_changes), 33)
        medium_changes = (np.abs(price_changes) >= np.percentile(np.abs(price_changes), 33)) & (np.abs(price_changes) < np.percentile(np.abs(price_changes), 67))
        large_changes = np.abs(price_changes) >= np.percentile(np.abs(price_changes), 67)
        
        small_accuracy = direction_accuracy if np.sum(small_changes) == 0 else np.mean((pred_direction * target_direction > 0)[small_changes[:-1]])
        medium_accuracy = direction_accuracy if np.sum(medium_changes) == 0 else np.mean((pred_direction * target_direction > 0)[medium_changes[:-1]])
        large_accuracy = direction_accuracy if np.sum(large_changes) == 0 else np.mean((pred_direction * target_direction > 0)[large_changes[:-1]])
        
        results = {
            'mape': mape * 100,  # 퍼센트로 변환
            'r2_score': r2,
            'direction_accuracy': direction_accuracy * 100,  # 퍼센트로 변환
            'accuracy_by_magnitude': {
                'small_changes': small_accuracy * 100,
                'medium_changes': medium_accuracy * 100,
                'large_changes': large_accuracy * 100
            },
            'overall_accuracy': (1 - mape) * 100,  # MAPE 기반 전체 정확도
            'predictions': pred_original,
            'targets': target_original,
            'uncertainties': uncertainties.numpy()
        }
        
        return results

def main():
    """
    메인 실행 함수
    """
    logger.info("🎯 90%+ 정확도 비트코인 예측 시스템 초기화")
    
    # 데이터 로드 (기존 시스템과 연동)
    try:
        # 6개월 데이터 로드
        data_path = "/Users/parkyoungjun/Desktop/BTC_Analysis_System/ai_optimized_6month_data/ai_matrix_6month_20250824_2213.csv"
        df = pd.read_csv(data_path, index_col=0, parse_dates=True)
        logger.info(f"✅ 데이터 로드 완료: {df.shape}")
        
        # 기본 OHLCV 데이터가 있는지 확인
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_columns):
            logger.error("❌ 필수 OHLCV 데이터가 없습니다")
            return
        
        # 데이터셋 생성
        dataset = AdvancedBTCDataset(
            data=df,
            sequence_length=168,  # 1주
            prediction_horizon=24,  # 24시간 예측
            use_holt_winters=True
        )
        
        logger.info(f"✅ 데이터셋 생성 완료: {len(dataset)} 샘플")
        logger.info(f"특성 수: {len(dataset.features)}")
        
        # 데이터 분할 (80% 훈련, 10% 검증, 10% 테스트)
        total_size = len(dataset)
        train_size = int(0.8 * total_size)
        val_size = int(0.1 * total_size)
        test_size = total_size - train_size - val_size
        
        train_dataset = torch.utils.data.Subset(dataset, range(train_size))
        val_dataset = torch.utils.data.Subset(dataset, range(train_size, train_size + val_size))
        test_dataset = torch.utils.data.Subset(dataset, range(train_size + val_size, total_size))
        
        # 데이터 로더
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)
        
        # 모델 초기화
        predictor = Advanced90PercentPredictor(
            input_size=len(dataset.features),
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        # 훈련
        predictor.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=200,
            patience=20
        )
        
        # 평가
        logger.info("🔍 모델 성능 평가 중...")
        results = predictor.evaluate_accuracy(test_loader, dataset)
        
        # 결과 출력
        logger.info("📊 최종 성능 결과:")
        logger.info(f"  • 전체 정확도: {results['overall_accuracy']:.2f}%")
        logger.info(f"  • 방향 예측 정확도: {results['direction_accuracy']:.2f}%") 
        logger.info(f"  • MAPE: {results['mape']:.2f}%")
        logger.info(f"  • R² Score: {results['r2_score']:.4f}")
        logger.info("  • 변화폭별 방향 정확도:")
        logger.info(f"    - 소폭 변화: {results['accuracy_by_magnitude']['small_changes']:.2f}%")
        logger.info(f"    - 중간 변화: {results['accuracy_by_magnitude']['medium_changes']:.2f}%")
        logger.info(f"    - 대폭 변화: {results['accuracy_by_magnitude']['large_changes']:.2f}%")
        
        # 결과 저장
        results_path = "/Users/parkyoungjun/Desktop/BTC_Analysis_System/advanced_90_percent_results.json"
        results_to_save = {k: v for k, v in results.items() if k not in ['predictions', 'targets', 'uncertainties']}
        
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results_to_save, f, ensure_ascii=False, indent=2)
        
        logger.info(f"✅ 결과 저장 완료: {results_path}")
        
        # 모델 저장
        model_path = "/Users/parkyoungjun/Desktop/BTC_Analysis_System/advanced_90_percent_model.pth"
        torch.save({
            'tft_state_dict': predictor.tft_model.state_dict(),
            'cnn_lstm_state_dict': predictor.cnn_lstm_model.state_dict(),
            'ensemble_weights': predictor.ensemble.weights,
            'feature_names': dataset.features,
            'scaler': dataset.scaler,
            'target_scaler': dataset.target_scaler,
            'training_history': predictor.training_history
        }, model_path)
        
        logger.info(f"✅ 모델 저장 완료: {model_path}")
        
        # 성공 여부 판단
        if results['overall_accuracy'] >= 90:
            logger.info("🎉 90%+ 정확도 달성 성공!")
        else:
            logger.info(f"⚠️  목표 정확도 미달성: {results['overall_accuracy']:.2f}% (목표: 90%+)")
            logger.info("💡 하이퍼파라미터 튜닝이나 더 많은 데이터가 필요할 수 있습니다.")
        
    except Exception as e:
        logger.error(f"❌ 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()