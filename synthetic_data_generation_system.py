#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
합성 비트코인 데이터 생성 시스템
- GAN 기반 합성 데이터 생성
- VAE 잠재공간 데이터 생성
- 몬테카를로 시뮬레이션
- 부트스트랩 샘플링
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

import os
import sys
import json
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
import logging
import pickle

# 과학 계산
from scipy import stats
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import minimize
from scipy.stats import multivariate_normal

# 머신러닝
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA, FastICA
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import IsolationForest

# 딥러닝 모델 구현
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.models import Model, Sequential
    from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization
    from tensorflow.keras.layers import Input, Reshape, Conv1D, Conv1DTranspose
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("⚠️ TensorFlow 없음 - GAN/VAE 기능 제한")

# 통계 및 시각화
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.dates import DateFormatter
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SyntheticBitcoinDataGenerator:
    """
    🔮 합성 비트코인 데이터 생성 시스템
    
    주요 기능:
    1. GAN 기반 시계열 합성 데이터
    2. VAE 잠재공간 데이터 생성
    3. 몬테카를로 가격 경로 시뮬레이션
    4. 부트스트랩 리샘플링
    5. 시장 충격 시나리오 생성
    """
    
    def __init__(self, sequence_length: int = 168):
        """
        시스템 초기화
        
        Args:
            sequence_length: 시계열 시퀀스 길이 (기본: 7일 = 168시간)
        """
        self.sequence_length = sequence_length
        self.logger = logging.getLogger(__name__)
        
        # 모델 저장소
        self.gan_generator = None
        self.gan_discriminator = None
        self.vae_encoder = None
        self.vae_decoder = None
        self.vae_model = None
        
        # 데이터 저장소
        self.training_data = None
        self.scalers = {}
        
        # 하이퍼파라미터
        self.hyperparams = {
            'gan': {
                'latent_dim': 100,
                'generator_lr': 0.0002,
                'discriminator_lr': 0.0002,
                'batch_size': 32,
                'epochs': 1000,
                'beta_1': 0.5
            },
            'vae': {
                'latent_dim': 50,
                'learning_rate': 0.001,
                'batch_size': 32,
                'epochs': 500,
                'beta': 1.0  # KL divergence weight
            },
            'monte_carlo': {
                'n_simulations': 10000,
                'time_horizon': 168,  # 7 days
                'confidence_levels': [0.05, 0.95]
            }
        }
        
        self.logger.info("🔮 합성 데이터 생성 시스템 초기화 완료")
    
    def prepare_training_data(self, data: pd.DataFrame) -> np.ndarray:
        """
        GAN/VAE 훈련용 데이터 준비
        
        Args:
            data: 원본 시계열 데이터
            
        Returns:
            훈련용 시퀀스 배열
        """
        self.logger.info("📊 훈련 데이터 준비 중...")
        
        # 숫자형 컬럼만 선택
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        numeric_data = data[numeric_cols].fillna(method='ffill').fillna(method='bfill')
        
        # 스케일링
        scaler = RobustScaler()
        scaled_data = scaler.fit_transform(numeric_data)
        self.scalers['main'] = scaler
        
        # 시퀀스 생성
        sequences = []
        for i in range(len(scaled_data) - self.sequence_length + 1):
            seq = scaled_data[i:i + self.sequence_length]
            sequences.append(seq)
        
        sequences = np.array(sequences)
        self.training_data = sequences
        
        self.logger.info(f"✅ {len(sequences)}개 시퀀스 준비 완료, 형태: {sequences.shape}")
        return sequences
    
    def build_gan_generator(self, input_dim: int, output_dim: int) -> Model:
        """
        GAN 생성자 모델 구축
        
        Args:
            input_dim: 입력 차원 (잠재 공간)
            output_dim: 출력 차원 (시퀀스 * 특성 수)
            
        Returns:
            생성자 모델
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow가 필요합니다")
        
        model = Sequential([
            Dense(256, activation='leaky_relu', input_dim=input_dim),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(512, activation='leaky_relu'),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(1024, activation='leaky_relu'),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(output_dim, activation='tanh'),
            Reshape((self.sequence_length, -1))
        ])
        
        return model
    
    def build_gan_discriminator(self, input_shape: Tuple[int, int]) -> Model:
        """
        GAN 판별자 모델 구축
        
        Args:
            input_shape: 입력 형태 (시퀀스 길이, 특성 수)
            
        Returns:
            판별자 모델
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow가 필요합니다")
        
        model = Sequential([
            Conv1D(64, 3, activation='leaky_relu', input_shape=input_shape),
            Dropout(0.3),
            
            Conv1D(128, 3, activation='leaky_relu'),
            Dropout(0.3),
            
            Conv1D(256, 3, activation='leaky_relu'),
            Dropout(0.3),
            
            layers.GlobalMaxPooling1D(),
            Dense(512, activation='leaky_relu'),
            Dropout(0.3),
            Dense(1, activation='sigmoid')
        ])
        
        return model
    
    def build_vae_encoder(self, input_shape: Tuple[int, int], latent_dim: int) -> Model:
        """
        VAE 인코더 모델 구축
        
        Args:
            input_shape: 입력 형태
            latent_dim: 잠재 공간 차원
            
        Returns:
            인코더 모델
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow가 필요합니다")
        
        inputs = Input(shape=input_shape)
        
        # 인코더 네트워크
        x = Conv1D(64, 3, activation='relu', padding='same')(inputs)
        x = Conv1D(128, 3, activation='relu', padding='same')(x)
        x = layers.GlobalAveragePooling1D()(x)
        x = Dense(256, activation='relu')(x)
        
        # 잠재 변수 파라미터
        z_mean = Dense(latent_dim, name='z_mean')(x)
        z_log_var = Dense(latent_dim, name='z_log_var')(x)
        
        # 샘플링 레이어
        def sampling(args):
            z_mean, z_log_var = args
            batch = tf.shape(z_mean)[0]
            dim = tf.shape(z_mean)[1]
            epsilon = tf.random.normal(shape=(batch, dim))
            return z_mean + tf.exp(0.5 * z_log_var) * epsilon
        
        z = layers.Lambda(sampling, name='z')([z_mean, z_log_var])
        
        encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
        return encoder
    
    def build_vae_decoder(self, latent_dim: int, output_shape: Tuple[int, int]) -> Model:
        """
        VAE 디코더 모델 구축
        
        Args:
            latent_dim: 잠재 공간 차원
            output_shape: 출력 형태
            
        Returns:
            디코더 모델
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow가 필요합니다")
        
        latent_inputs = Input(shape=(latent_dim,))
        
        # 디코더 네트워크
        x = Dense(256, activation='relu')(latent_inputs)
        x = Dense(output_shape[0] * 64, activation='relu')(x)
        x = Reshape((output_shape[0], 64))(x)
        
        x = Conv1DTranspose(128, 3, activation='relu', padding='same')(x)
        x = Conv1DTranspose(64, 3, activation='relu', padding='same')(x)
        outputs = Conv1DTranspose(output_shape[1], 3, activation='linear', padding='same')(x)
        
        decoder = Model(latent_inputs, outputs, name='decoder')
        return decoder
    
    def train_gan(self, data: np.ndarray, epochs: int = None) -> Dict[str, List[float]]:
        """
        GAN 모델 훈련
        
        Args:
            data: 훈련 데이터
            epochs: 에폭 수
            
        Returns:
            훈련 기록
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow가 필요합니다")
        
        self.logger.info("🔥 GAN 모델 훈련 시작...")
        
        epochs = epochs or self.hyperparams['gan']['epochs']
        batch_size = self.hyperparams['gan']['batch_size']
        latent_dim = self.hyperparams['gan']['latent_dim']
        
        # 모델 구축
        input_shape = (data.shape[1], data.shape[2])
        output_dim = data.shape[1] * data.shape[2]
        
        self.gan_generator = self.build_gan_generator(latent_dim, output_dim)
        self.gan_discriminator = self.build_gan_discriminator(input_shape)
        
        # 옵티마이저
        g_optimizer = Adam(learning_rate=self.hyperparams['gan']['generator_lr'],
                          beta_1=self.hyperparams['gan']['beta_1'])
        d_optimizer = Adam(learning_rate=self.hyperparams['gan']['discriminator_lr'],
                          beta_1=self.hyperparams['gan']['beta_1'])
        
        # 손실 함수
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        
        def discriminator_loss(real_output, fake_output):
            real_loss = cross_entropy(tf.ones_like(real_output), real_output)
            fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
            return real_loss + fake_loss
        
        def generator_loss(fake_output):
            return cross_entropy(tf.ones_like(fake_output), fake_output)
        
        # 훈련 루프
        history = {'d_loss': [], 'g_loss': []}
        
        for epoch in range(epochs):
            # 배치 생성
            idx = np.random.randint(0, data.shape[0], batch_size)
            real_batch = data[idx]
            
            # 판별자 훈련
            with tf.GradientTape() as disc_tape:
                noise = tf.random.normal([batch_size, latent_dim])
                generated_data = self.gan_generator(noise, training=True)
                
                real_output = self.gan_discriminator(real_batch, training=True)
                fake_output = self.gan_discriminator(generated_data, training=True)
                
                disc_loss = discriminator_loss(real_output, fake_output)
            
            disc_gradients = disc_tape.gradient(disc_loss, self.gan_discriminator.trainable_variables)
            d_optimizer.apply_gradients(zip(disc_gradients, self.gan_discriminator.trainable_variables))
            
            # 생성자 훈련
            with tf.GradientTape() as gen_tape:
                noise = tf.random.normal([batch_size, latent_dim])
                generated_data = self.gan_generator(noise, training=True)
                fake_output = self.gan_discriminator(generated_data, training=True)
                gen_loss = generator_loss(fake_output)
            
            gen_gradients = gen_tape.gradient(gen_loss, self.gan_generator.trainable_variables)
            g_optimizer.apply_gradients(zip(gen_gradients, self.gan_generator.trainable_variables))
            
            # 기록
            history['d_loss'].append(float(disc_loss))
            history['g_loss'].append(float(gen_loss))
            
            # 진행상황 출력
            if epoch % 100 == 0:
                self.logger.info(f"에폭 {epoch}/{epochs}, D 손실: {disc_loss:.4f}, G 손실: {gen_loss:.4f}")
        
        self.logger.info("✅ GAN 훈련 완료")
        return history
    
    def train_vae(self, data: np.ndarray, epochs: int = None) -> Dict[str, List[float]]:
        """
        VAE 모델 훈련
        
        Args:
            data: 훈련 데이터
            epochs: 에폭 수
            
        Returns:
            훈련 기록
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow가 필요합니다")
        
        self.logger.info("🔥 VAE 모델 훈련 시작...")
        
        epochs = epochs or self.hyperparams['vae']['epochs']
        batch_size = self.hyperparams['vae']['batch_size']
        latent_dim = self.hyperparams['vae']['latent_dim']
        beta = self.hyperparams['vae']['beta']
        
        # 모델 구축
        input_shape = (data.shape[1], data.shape[2])
        
        self.vae_encoder = self.build_vae_encoder(input_shape, latent_dim)
        self.vae_decoder = self.build_vae_decoder(latent_dim, input_shape)
        
        # VAE 전체 모델
        inputs = Input(shape=input_shape)
        z_mean, z_log_var, z = self.vae_encoder(inputs)
        outputs = self.vae_decoder(z)
        
        self.vae_model = Model(inputs, outputs, name='vae')
        
        # 손실 함수
        def vae_loss(x, x_decoded_mean):
            # 재구성 손실
            reconstruction_loss = tf.reduce_mean(tf.keras.losses.mse(x, x_decoded_mean))
            
            # KL 발산 손실
            kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            
            return reconstruction_loss + beta * kl_loss
        
        # 컴파일 및 훈련
        self.vae_model.compile(optimizer=Adam(learning_rate=self.hyperparams['vae']['learning_rate']),
                              loss=vae_loss)
        
        # 콜백
        callbacks = [
            EarlyStopping(patience=50, restore_best_weights=True),
            ReduceLROnPlateau(patience=25, factor=0.5)
        ]
        
        # 훈련
        history = self.vae_model.fit(
            data, data,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=1
        )
        
        self.logger.info("✅ VAE 훈련 완료")
        return history.history
    
    def generate_gan_samples(self, n_samples: int = 1000) -> np.ndarray:
        """
        GAN으로 합성 샘플 생성
        
        Args:
            n_samples: 생성할 샘플 수
            
        Returns:
            합성 데이터 배열
        """
        if self.gan_generator is None:
            raise ValueError("GAN 생성자가 훈련되지 않았습니다")
        
        self.logger.info(f"🎲 GAN으로 {n_samples}개 합성 샘플 생성...")
        
        latent_dim = self.hyperparams['gan']['latent_dim']
        noise = tf.random.normal([n_samples, latent_dim])
        synthetic_data = self.gan_generator(noise, training=False)
        
        return synthetic_data.numpy()
    
    def generate_vae_samples(self, n_samples: int = 1000) -> np.ndarray:
        """
        VAE로 합성 샘플 생성
        
        Args:
            n_samples: 생성할 샘플 수
            
        Returns:
            합성 데이터 배열
        """
        if self.vae_decoder is None:
            raise ValueError("VAE 디코더가 훈련되지 않았습니다")
        
        self.logger.info(f"🎲 VAE로 {n_samples}개 합성 샘플 생성...")
        
        latent_dim = self.hyperparams['vae']['latent_dim']
        
        # 잠재 공간에서 샘플링
        z_samples = tf.random.normal([n_samples, latent_dim])
        synthetic_data = self.vae_decoder(z_samples)
        
        return synthetic_data.numpy()
    
    def monte_carlo_price_simulation(self, 
                                   initial_price: float,
                                   returns_data: pd.Series,
                                   n_simulations: int = None,
                                   time_horizon: int = None) -> np.ndarray:
        """
        몬테카를로 가격 경로 시뮬레이션
        
        Args:
            initial_price: 초기 가격
            returns_data: 수익률 데이터
            n_simulations: 시뮬레이션 횟수
            time_horizon: 시간 지평선
            
        Returns:
            시뮬레이션된 가격 경로들
        """
        self.logger.info("📈 몬테카를로 가격 시뮬레이션 시작...")
        
        n_simulations = n_simulations or self.hyperparams['monte_carlo']['n_simulations']
        time_horizon = time_horizon or self.hyperparams['monte_carlo']['time_horizon']
        
        # 수익률 분포 파라미터 추정
        mu = returns_data.mean()
        sigma = returns_data.std()
        
        # 시뮬레이션
        price_paths = np.zeros((n_simulations, time_horizon))
        price_paths[:, 0] = initial_price
        
        for t in range(1, time_horizon):
            # 정규분포에서 수익률 샘플링
            random_returns = np.random.normal(mu, sigma, n_simulations)
            
            # 가격 업데이트 (로그 정규 가정)
            price_paths[:, t] = price_paths[:, t-1] * (1 + random_returns)
        
        self.logger.info(f"✅ {n_simulations}개 시뮬레이션 경로 생성 완료")
        return price_paths
    
    def bootstrap_resample(self, data: pd.DataFrame, n_samples: int = 1000) -> List[pd.DataFrame]:
        """
        부트스트랩 리샘플링
        
        Args:
            data: 원본 데이터
            n_samples: 샘플 수
            
        Returns:
            부트스트랩 샘플들
        """
        self.logger.info(f"🔄 부트스트랩 {n_samples}개 샘플 생성...")
        
        bootstrap_samples = []
        n_observations = len(data)
        
        for i in range(n_samples):
            # 복원 추출
            indices = np.random.choice(n_observations, n_observations, replace=True)
            bootstrap_sample = data.iloc[indices].reset_index(drop=True)
            bootstrap_samples.append(bootstrap_sample)
        
        self.logger.info("✅ 부트스트랩 샘플링 완료")
        return bootstrap_samples
    
    def generate_market_shock_scenarios(self, 
                                      base_data: pd.DataFrame,
                                      shock_types: List[str] = None) -> Dict[str, pd.DataFrame]:
        """
        시장 충격 시나리오 생성
        
        Args:
            base_data: 기준 데이터
            shock_types: 충격 유형 리스트
            
        Returns:
            충격 시나리오들
        """
        self.logger.info("⚡ 시장 충격 시나리오 생성...")
        
        shock_types = shock_types or ['flash_crash', 'bubble_burst', 'regulatory_shock', 'whale_dump']
        scenarios = {}
        
        # 가격 컬럼 찾기
        price_col = None
        for col in base_data.columns:
            if 'price' in col.lower() or 'close' in col.lower():
                price_col = col
                break
        
        if price_col is None:
            price_col = base_data.select_dtypes(include=[np.number]).columns[0]
        
        for shock_type in shock_types:
            shocked_data = base_data.copy()
            
            if shock_type == 'flash_crash':
                # 갑작스러운 급락 (30-50% 하락 후 부분 회복)
                shock_point = len(shocked_data) // 2
                crash_magnitude = np.random.uniform(0.3, 0.5)  # 30-50% 하락
                recovery_factor = np.random.uniform(0.6, 0.8)  # 60-80% 회복
                
                # 급락
                shocked_data.loc[shock_point:shock_point+5, price_col] *= (1 - crash_magnitude)
                # 부분 회복
                recovery_points = min(20, len(shocked_data) - shock_point - 5)
                recovery_slope = crash_magnitude * recovery_factor / recovery_points
                
                for i in range(recovery_points):
                    idx = shock_point + 5 + i
                    if idx < len(shocked_data):
                        shocked_data.loc[idx, price_col] *= (1 + recovery_slope * (i + 1) / recovery_points)
            
            elif shock_type == 'bubble_burst':
                # 버블 붕괴 (점진적 대폭락)
                burst_start = int(len(shocked_data) * 0.6)
                burst_duration = min(50, len(shocked_data) - burst_start)
                total_decline = np.random.uniform(0.6, 0.8)  # 60-80% 하락
                
                for i in range(burst_duration):
                    idx = burst_start + i
                    if idx < len(shocked_data):
                        daily_decline = total_decline * (1 - np.exp(-i / 10)) / burst_duration
                        shocked_data.loc[idx, price_col] *= (1 - daily_decline)
            
            elif shock_type == 'regulatory_shock':
                # 규제 충격 (계단식 하락)
                shock_points = np.random.choice(len(shocked_data), 3, replace=False)
                shock_points = np.sort(shock_points)
                
                for point in shock_points:
                    decline = np.random.uniform(0.15, 0.25)  # 15-25% 하락
                    shocked_data.loc[point:, price_col] *= (1 - decline)
            
            elif shock_type == 'whale_dump':
                # 고래 매도 (대량 매도로 인한 급락과 빠른 회복)
                dump_point = np.random.randint(10, len(shocked_data) - 10)
                dump_magnitude = np.random.uniform(0.2, 0.35)  # 20-35% 하락
                
                # 급락
                shocked_data.loc[dump_point:dump_point+2, price_col] *= (1 - dump_magnitude)
                # 빠른 회복 (24시간 내)
                recovery_points = min(24, len(shocked_data) - dump_point - 2)
                for i in range(recovery_points):
                    idx = dump_point + 2 + i
                    if idx < len(shocked_data):
                        recovery = dump_magnitude * np.exp(-i / 8) / recovery_points  # 지수적 회복
                        shocked_data.loc[idx, price_col] *= (1 + recovery)
            
            scenarios[shock_type] = shocked_data
        
        self.logger.info(f"✅ {len(scenarios)}개 충격 시나리오 생성 완료")
        return scenarios
    
    def inverse_transform_synthetic_data(self, synthetic_data: np.ndarray) -> List[pd.DataFrame]:
        """
        합성 데이터를 원본 스케일로 역변환
        
        Args:
            synthetic_data: 합성 데이터
            
        Returns:
            역변환된 데이터프레임 리스트
        """
        if 'main' not in self.scalers:
            self.logger.warning("스케일러가 없습니다. 역변환을 건너뜁니다.")
            return []
        
        scaler = self.scalers['main']
        transformed_dfs = []
        
        for i, sequence in enumerate(synthetic_data):
            # 역변환
            original_shape = sequence.shape
            flattened = sequence.reshape(-1, sequence.shape[-1])
            inverse_scaled = scaler.inverse_transform(flattened)
            restored_sequence = inverse_scaled.reshape(original_shape)
            
            # 데이터프레임으로 변환
            df = pd.DataFrame(restored_sequence)
            df.index = pd.date_range(start='2024-01-01', periods=len(df), freq='H')
            
            transformed_dfs.append(df)
        
        return transformed_dfs
    
    def evaluate_synthetic_data_quality(self, 
                                       original_data: np.ndarray,
                                       synthetic_data: np.ndarray) -> Dict[str, float]:
        """
        합성 데이터 품질 평가
        
        Args:
            original_data: 원본 데이터
            synthetic_data: 합성 데이터
            
        Returns:
            품질 메트릭
        """
        metrics = {}
        
        try:
            # 1. 분포 유사성 (Wasserstein 거리)
            from scipy.stats import wasserstein_distance
            
            wasserstein_distances = []
            for feature_idx in range(original_data.shape[-1]):
                orig_feature = original_data[:, :, feature_idx].flatten()
                synth_feature = synthetic_data[:, :, feature_idx].flatten()
                
                distance = wasserstein_distance(orig_feature, synth_feature)
                wasserstein_distances.append(distance)
            
            metrics['wasserstein_distance'] = np.mean(wasserstein_distances)
            
            # 2. 통계적 특성 보존
            orig_mean = np.mean(original_data, axis=(0, 1))
            synth_mean = np.mean(synthetic_data, axis=(0, 1))
            orig_std = np.std(original_data, axis=(0, 1))
            synth_std = np.std(synthetic_data, axis=(0, 1))
            
            mean_similarity = 1 - np.mean(np.abs(orig_mean - synth_mean) / (np.abs(orig_mean) + 1e-8))
            std_similarity = 1 - np.mean(np.abs(orig_std - synth_std) / (orig_std + 1e-8))
            
            metrics['statistical_similarity'] = (mean_similarity + std_similarity) / 2
            
            # 3. 시계열 특성 (자기상관)
            orig_autocorr = []
            synth_autocorr = []
            
            for i in range(min(10, original_data.shape[0])):
                for j in range(original_data.shape[-1]):
                    orig_series = original_data[i, :, j]
                    synth_series = synthetic_data[i, :, j]
                    
                    if len(orig_series) > 1:
                        orig_ac = np.corrcoef(orig_series[:-1], orig_series[1:])[0, 1]
                        synth_ac = np.corrcoef(synth_series[:-1], synth_series[1:])[0, 1]
                        
                        if not (np.isnan(orig_ac) or np.isnan(synth_ac)):
                            orig_autocorr.append(orig_ac)
                            synth_autocorr.append(synth_ac)
            
            if orig_autocorr and synth_autocorr:
                autocorr_similarity = 1 - np.mean(np.abs(np.array(orig_autocorr) - np.array(synth_autocorr)))
                metrics['temporal_similarity'] = max(0, autocorr_similarity)
            else:
                metrics['temporal_similarity'] = 0.5
            
            # 4. 종합 품질 점수
            metrics['overall_quality'] = np.mean([
                1 - metrics['wasserstein_distance'] / (1 + metrics['wasserstein_distance']),
                metrics['statistical_similarity'],
                metrics['temporal_similarity']
            ])
            
        except Exception as e:
            self.logger.warning(f"품질 평가 오류: {e}")
            metrics = {'error': str(e), 'overall_quality': 0}
        
        return metrics
    
    def save_synthetic_data(self, 
                          synthetic_data: Dict[str, np.ndarray],
                          output_dir: str = "synthetic_btc_data") -> None:
        """
        합성 데이터 저장
        
        Args:
            synthetic_data: 합성 데이터 딕셔너리
            output_dir: 출력 디렉토리
        """
        self.logger.info(f"💾 합성 데이터 저장: {output_dir}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        for method, data in synthetic_data.items():
            method_dir = os.path.join(output_dir, method)
            os.makedirs(method_dir, exist_ok=True)
            
            # NumPy 배열로 저장
            np.save(os.path.join(method_dir, "synthetic_sequences.npy"), data)
            
            # CSV 형태로도 저장 (일부만)
            if len(data) > 0:
                sample_size = min(100, len(data))
                sample_data = data[:sample_size]
                
                for i, sequence in enumerate(sample_data):
                    df = pd.DataFrame(sequence)
                    df.to_csv(os.path.join(method_dir, f"sample_{i:03d}.csv"), index=False)
        
        self.logger.info("✅ 합성 데이터 저장 완료")


def main():
    """
    메인 실행 함수
    """
    print("🔮 합성 비트코인 데이터 생성 시스템 시작")
    
    # 시스템 초기화
    generator = SyntheticBitcoinDataGenerator(sequence_length=168)
    
    # 예제 데이터 생성 (실제로는 실제 데이터를 로드)
    print("\n📊 예제 데이터 생성...")
    np.random.seed(42)
    
    # 가상의 비트코인 시계열 데이터 생성
    n_samples = 1000
    seq_length = 168
    n_features = 5
    
    example_data = np.random.randn(n_samples, seq_length, n_features)
    example_data = np.cumsum(example_data, axis=1)  # 시계열 특성 부여
    
    print(f"예제 데이터 형태: {example_data.shape}")
    
    if TENSORFLOW_AVAILABLE:
        try:
            # GAN 훈련
            print("\n🔥 GAN 모델 훈련...")
            gan_history = generator.train_gan(example_data, epochs=100)  # 테스트용으로 적은 에폭
            
            # VAE 훈련
            print("\n🔥 VAE 모델 훈련...")
            vae_history = generator.train_vae(example_data, epochs=50)  # 테스트용으로 적은 에폭
            
            # 합성 데이터 생성
            print("\n🎲 합성 데이터 생성...")
            gan_samples = generator.generate_gan_samples(100)
            vae_samples = generator.generate_vae_samples(100)
            
            # 품질 평가
            print("\n🔍 품질 평가...")
            gan_quality = generator.evaluate_synthetic_data_quality(example_data[:100], gan_samples)
            vae_quality = generator.evaluate_synthetic_data_quality(example_data[:100], vae_samples)
            
            print(f"GAN 품질 점수: {gan_quality.get('overall_quality', 0):.3f}")
            print(f"VAE 품질 점수: {vae_quality.get('overall_quality', 0):.3f}")
            
            # 저장
            synthetic_data = {
                'gan_samples': gan_samples,
                'vae_samples': vae_samples
            }
            generator.save_synthetic_data(synthetic_data)
            
        except Exception as e:
            print(f"⚠️ 딥러닝 모델 훈련 오류: {e}")
    
    # 몬테카를로 시뮬레이션 (TensorFlow 없이도 실행 가능)
    print("\n📈 몬테카를로 시뮬레이션...")
    returns_data = pd.Series(np.random.normal(0.001, 0.02, 1000))  # 예제 수익률
    price_paths = generator.monte_carlo_price_simulation(
        initial_price=50000,
        returns_data=returns_data,
        n_simulations=1000,
        time_horizon=168
    )
    
    print(f"몬테카를로 경로 형태: {price_paths.shape}")
    
    # 시장 충격 시나리오
    print("\n⚡ 시장 충격 시나리오 생성...")
    example_df = pd.DataFrame({'price': np.random.randn(100).cumsum() + 50000})
    shock_scenarios = generator.generate_market_shock_scenarios(example_df)
    
    print(f"생성된 충격 시나리오: {list(shock_scenarios.keys())}")
    
    print("\n✅ 합성 데이터 생성 시스템 완료!")


if __name__ == "__main__":
    main()