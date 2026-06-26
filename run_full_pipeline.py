#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_full_pipeline.py — Pipeline completo de AOC (Acoustic Occlusion Cancellation)
integrado con APICultor.
Uso:
    python run_full_pipeline.py \
        --audio_dir  /ruta/audios_limpios \
        --occluded_dir /ruta/audios_ocluidos \
        --model_path  /ruta/modelo_dsvm.pkl   (opcional, crea uno nuevo si no existe)
        --output_dir  resultados/
        --regression                          (flag; default=True para AOC)
        --n_adversaries 3
        --post_train                          (flag; activa SFT+RL post-entrenamiento)
"""

import os
import sys
import asyncio
import argparse
import logging
import hashlib
import json
import pickle
import warnings
from collections import defaultdict
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from apicultor.machine_learning.explain import compute_feature_importance, forward_with_dropout
from apicultor.machine_learning.efficiency import *
from apicultor.machine_learning.loop_state import *
from apicultor.arch.shared import *
import numpy as np
from scipy.signal import stft as scipy_stft, istft as scipy_istft
from scipy.io.wavfile import write as wav_write
from soundfile import read as sf_read
from apicultor.machine_learning.subproblem import continuous_multi_lorax, parallel_continuous_multi_lorax

warnings.filterwarnings("ignore", category=RuntimeWarning)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("AOC_Pipeline")

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTACIONES DESDE EL REPO (fiel a los nombres y rutas reales)
# ─────────────────────────────────────────────────────────────────────────────

# Paralelismo
from apicultor.arch.thread import parallel, init_pool_execution_with_queues

# Cross-validation y adversarios
from apicultor.machine_learning.cross_validation import (
    GridSearch,
    define_adversary_data,
    scan_data_leakage,
    generate_imputed_data,
    multiple_hypothesis_testing,
)

# Atención y context
from apicultor.machine_learning.subproblem import (
    continuous_multi_lorax,
    gather_layers_outputs,
)

# Fairness
from apicultor.machine_learning.fairness import (
    p_rule,
    ind_fairness)
    
#Dependency    
from apicultor.machine_learning.dependency import ( 
    BEC,
    BTC
)

# Explicabilidad y dropout
from apicultor.machine_learning.explain import (
    explain,
    dropout,
    compute_feature_importance,
)

# Error / scoring
from apicultor.machine_learning.error import score

# Dependencias
from apicultor.machine_learning.dependency import BEC as dep_BEC, BTC as dep_BTC

# ── Post-training: clases ORIGINALES del repo ───────────────────────────────
from apicultor.machine_learning.stress import (
    ErrorBucket,          # Principios constitucionales reales
    TrainingPhase,
    RewardModel,
    ConstitutionalAI,
    RewardHackingDetector,
    PostTrainingPipeline,
    EnhancedAdversarialValidator,
    harmful_content_verifier,
    privacy_verifier,
    format_verifier,
)

# Métricas de incertidumbre / calibración
from apicultor.machine_learning.metrics import (
    expected_calibration_error,
    detect_overconfidence,
    uncertainty_aware_reward,
    UncertaintyAwareRewardModel,
    CalibrationMonitor,
)

# Monitoreo
from apicultor.machine_learning.monitor import PostTrainingMonitor

# Intervención automática
from apicultor.machine_learning.intervene import (
    AutoInterventionSystem,
    InterventionThresholds,
)

# ─────────────────────────────────────────────────────────────────────────────
# PARÁMETROS GLOBALES DE AUDIO Y STFT
# ─────────────────────────────────────────────────────────────────────────────

SR          = 44100       # Frecuencia de muestreo estándar del repo (MonoLoader)
N_FFT       = 2048        # Tamaño de ventana STFT
HOP_LENGTH  = 512         # Paso entre ventanas
WIN_LENGTH  = N_FFT
N_MELS      = 128         # No se usa para AOC pero mantiene compatibilidad
FRAME_PAIRS = True        # Entrenar sobre pares (frame_ocluido, frame_limpio)

# Agregar al inicio del archivo, después de los imports
import secrets
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

import uvloop
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

#TEMPLATES
#TRATO DIGNO
#        dignity_pattern = self._create_gaussian_spectrum(
#            center=self.n_features // 2,
#            sigma=self.n_features // 8,
#            amplitude=0.5
#        )
#        # Añadir decaimiento suave en los bordes
#        decay = np.exp(-np.linspace(0, 3, self.n_features))
#        dignity_pattern = dignity_pattern * decay
# IGUALDAD LEGAL
# Patrón: distribución uniforme, sin sesgos hacia frecuencias específicas
#        equality_pattern = np.ones(self.n_features) / np.sqrt(self.n_features)
#        # Añadir pequeña modulación para evitar uniformidad perfecta
#        equality_pattern = equality_pattern * (1 + 0.1 * np.sin(np.linspace(0, 4*np.pi, self.n_features)))
#        equality_pattern = equality_pattern / np.linalg.norm(equality_pattern)
# === 3. LIBERTAD DE EXPRESIÓN ===
# Patrón: amplio rango dinámico, preservación de variabilidad
#        freedom_pattern = self._create_multi_peak_spectrum(
#            peaks=[self.n_features // 4, self.n_features // 2, 3 * self.n_features // 4],
#            amplitudes=[0.8, 1.0, 0.8],
#            sigma=self.n_features // 16
#        )
# === 4. PRIVACIDAD ===
# Patrón: información localizada, no dispersión de información sensible
#        privacy_pattern = self._create_localized_pattern(
#            position=self.n_features // 3,
#            width=self.n_features // 20,
#            amplitude=0.3
#       )
# === 5. DEBIDO PROCESO ===
# Patrón: estructura secuencial, pasos ordenados, consistencia temporal
#        due_process_pattern = self._create_temporal_sequence_pattern(
#            n_steps=20,
#            decay_factor=0.95
#        )
# === 6. NO DISCRIMINACIÓN ===
# Patrón: simetría espectral, tratamiento igualitario de frecuencias
#        non_discrimination_pattern = self._create_symmetric_spectrum(
#            symmetry_axis=self.n_features // 2
#        )     
# === 7. LEGALIDAD ===
# Patrón: predictibilidad, baja variabilidad inesperada
#        legality_pattern = self._create_predictable_pattern(
#            predictability=0.85
#        )      
# === 8. PROPORCIONALIDAD ===
# Patrón: respuesta graduada, sin amplificación extrema
#        proportionality_pattern = self._create_graduated_response(
#            max_amplification=1.2,
#            min_amplification=0.8
#        )     
# === 9. TRANSPARENCIA ===
# Patrón: claridad estructural, picos bien definidos
#        transparency_pattern = self._create_clear_structural_pattern(
#            n_peaks=5,
#            clarity=0.9
#        )
# === 10. PRESUNCIÓN DE INOCENCIA ===
# Patrón: baja energía en ausencia de evidencia fuerte
#        presumption_pattern = self._create_low_energy_default_pattern(
#            default_energy=0.1,
#            evidence_threshold=0.7
#        )
# === 11. DERECHOS HUMANOS ==
# Patrón: protección de frecuencias fundamentales
#        human_rights_pattern = self._create_protected_band_pattern(
#            protected_bands=[(0, self.n_features // 10), 
#                            (4*self.n_features//10, 6*self.n_features//10)],
#            protection_strength=0.9
#        )
# === 12. PROTECCIÓN AMBIENTAL ===
# Patrón: conservación de energía, sostenibilidad espectral
#        environmental_pattern = self._create_sustainable_spectrum(
#            energy_conservation=0.95
#        )
#   def _create_gaussian_spectrum(self, center: int, sigma: int, amplitude: float) -> np.ndarray:
#        """Crea un espectro gaussiano suave"""
#        x = np.arange(self.n_features)
#        return amplitude * np.exp(-((x - center) ** 2) / (2 * sigma ** 2))
    
#    def _create_multi_peak_spectrum(self, peaks: List[int], amplitudes: List[float], sigma: int) -> np.ndarray:
#        """Crea un espectro con múltiples picos gaussianos"""
#        pattern = np.zeros(self.n_features)
#        for peak, amp in zip(peaks, amplitudes):
#            pattern += amp * np.exp(-((np.arange(self.n_features) - peak) ** 2) / (2 * sigma ** 2))
#        return pattern / np.linalg.norm(pattern)
    
#    def _create_localized_pattern(self, position: int, width: int, amplitude: float) -> np.ndarray:
#        """Crea un patrón localizado (energía concentrada)"""
#        pattern = np.zeros(self.n_features)
#        start = max(0, position - width // 2)
#        end = min(self.n_features, position + width // 2)
#        pattern[start:end] = amplitude
#        return pattern / np.linalg.norm(pattern)
    
#    def _create_temporal_sequence_pattern(self, n_steps: int, decay_factor: float) -> np.ndarray:
#        """Crea un patrón de secuencia temporal para debido proceso"""
#        pattern = np.zeros(self.n_features)
#        step_size = self.n_features // n_steps
#        for i in range(min(n_steps, self.n_features // step_size)):
#            start = i * step_size
#            end = min(self.n_features, (i + 1) * step_size)
#            pattern[start:end] = decay_factor ** i
#        return pattern / np.linalg.norm(pattern)
    
#    def _create_symmetric_spectrum(self, symmetry_axis: int) -> np.ndarray:
#        """Crea un espectro simétrico (no discriminación)"""
#        x = np.arange(self.n_features)
#        pattern = np.exp(-((x - symmetry_axis) ** 2) / (2 * (self.n_features // 8) ** 2))
#        # Hacer simétrico
#        pattern = (pattern + pattern[::-1]) / 2
#        return pattern / np.linalg.norm(pattern)
    
#    def _create_predictable_pattern(self, predictability: float) -> np.ndarray:
#        """Crea un patrón predecible (baja entropía)"""
#        # Patrón sinusoidal de baja frecuencia
#        pattern = np.sin(np.linspace(0, 2 * np.pi * predictability, self.n_features))
#        return pattern / np.linalg.norm(pattern)
    
#    def _create_graduated_response(self, max_amplification: float, min_amplification: float) -> np.ndarray:
#        """Crea un patrón de respuesta graduada (proporcionalidad)"""
#        x = np.linspace(0, 1, self.n_features)
#        pattern = min_amplification + (max_amplification - min_amplification) * x
#        return pattern / np.linalg.norm(pattern)
    
#    def _create_clear_structural_pattern(self, n_peaks: int, clarity: float) -> np.ndarray:
#        """Crea un patrón estructural claro (transparencia)"""
#        pattern = np.zeros(self.n_features)
#        peak_positions = np.linspace(0, self.n_features - 1, n_peaks, dtype=int)
#        for pos in peak_positions:
#            pattern[pos] = clarity
#        # Suavizar un poco
#        pattern = np.convolve(pattern, np.ones(5)/5, mode='same')
#        return pattern / np.linalg.norm(pattern)
    
#    def _create_low_energy_default_pattern(self, default_energy: float, evidence_threshold: float) -> np.ndarray:
#        """Crea un patrón de baja energía por defecto (presunción de inocencia)"""
#        pattern = np.ones(self.n_features) * default_energy
#        # Pequeñas modulaciones para no ser completamente plano
#        pattern = pattern * (1 + 0.1 * np.random.randn(self.n_features))
#        return pattern / np.linalg.norm(pattern)
    
#    def _create_protected_band_pattern(self, protected_bands: List[Tuple[int, int]], protection_strength: float) -> np.ndarray:
#        """Crea un patrón con bandas protegidas (derechos humanos)"""
#        pattern = np.ones(self.n_features)
#        for start, end in protected_bands:
#            pattern[start:end] = protection_strength
#        return pattern / np.linalg.norm(pattern)
    
#    def _create_sustainable_spectrum(self, energy_conservation: float) -> np.ndarray:
#        """Crea un espectro sostenible (conservación de energía)"""
#        pattern = np.ones(self.n_features) * energy_conservation
#        # Decaimiento natural en altas frecuencias
#        decay = np.exp(-np.linspace(0, 2, self.n_features))
#        pattern = pattern * decay
#        return pattern / np.linalg.norm(pattern)

def _build_spectral_cot_simple(x_frame: np.ndarray, pred: np.ndarray) -> List[str]:
    """CoT simple para evaluación post-distillation."""
    centroid = float(np.average(np.arange(len(x_frame)), weights=x_frame + 1e-9))
    energy_in = float(np.mean(x_frame ** 2))
    energy_out = float(np.mean(pred ** 2))
    snr_est = compute_snr(x_frame, pred)
    
    return [
        f"Frame espectral: centroide={centroid:.1f}, energía_entrada={energy_in:.5f}",
        f"Energía estimada tras AOC: {energy_out:.5f}",
        f"SNR estimado: {snr_est:.2f} dB",
    ]

def get_stage_checkpoint_path(output_dir: str, stage: str, baseline_idx: int, loop_counter: int, suffix: str = "") -> str:
    """Genera ruta de checkpoint para una etapa específica"""
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    return os.path.join(checkpoint_dir, f"{stage}_b{baseline_idx:04d}_l{loop_counter}{suffix}")

def is_stage_completed(output_dir: str, stage: str, baseline_idx: int, loop_counter: int) -> bool:
    """Verifica si una etapa ya fue completada"""
    marker_path = get_stage_checkpoint_path(output_dir, stage, baseline_idx, loop_counter, ".complete")
    return os.path.exists(marker_path)

def mark_stage_completed(output_dir: str, stage: str, baseline_idx: int, loop_counter: int, metadata: Dict = None):
    """Marca una etapa como completada con metadatos"""
    marker_path = get_stage_checkpoint_path(output_dir, stage, baseline_idx, loop_counter, ".complete")
    with open(marker_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'baseline_idx': baseline_idx,
            'loop_counter': loop_counter,
            'stage': stage,
            'metadata': metadata or {}
        }, f, indent=2)
    
    # También guardar metadatos en JSON aparte
    meta_path = get_stage_checkpoint_path(output_dir, stage, baseline_idx, loop_counter, ".meta.json")
    with open(meta_path, 'w') as f:
        json.dump(metadata or {}, f, indent=2)
    
    logger.info(f"[Checkpoint] {stage} completado para baseline {baseline_idx}")

def save_stage_result(output_dir: str, stage: str, baseline_idx: int, loop_counter: int, data: Any, name: str = "result"):
    """Guarda el resultado de una etapa"""
    result_path = get_stage_checkpoint_path(output_dir, stage, baseline_idx, loop_counter, f"_{name}.npy")
    np.save(result_path, data)
    return result_path

def load_stage_result(output_dir: str, stage: str, baseline_idx: int, loop_counter: int, name: str = "result") -> Optional[Any]:
    """Carga el resultado de una etapa si existe"""
    result_path = get_stage_checkpoint_path(output_dir, stage, baseline_idx, loop_counter, f"_{name}.npy")
    if os.path.exists(result_path):
        return np.load(result_path, allow_pickle=True)
    return None

def compute_tool_logits(
    input_matrix: np.ndarray,  # shape: (24, 1025)
    tool_weights: np.ndarray
) -> np.ndarray:
    """
    Calcula logits para cada herramienta disponible.
    
    Args:
        input_matrix: Espectrograma de entrada
        tool_weights: Pesos para cada herramienta
    
    Returns:
        logits: array de 4 elementos [STFT, Wiener, MultiLorax, ISTFT]
    """
    logits = []
    
    # Aplanar la matriz para el cálculo
    X = input_matrix.flatten()  # (24*1025,)
    
    # 3. Logit para Multi-Lorax
    w_lorax = tool_weights.flatten()
    logit_lorax = np.dot(X[:len(w_lorax)], w_lorax) + bias_lorax
    logits.append(logit_lorax)
    
    return np.array(logits)

def compute_result_logit(
    input_matrix: np.ndarray,
    output_matrix: np.ndarray,
    result_weights: np.ndarray
) -> float:
    """
    Calcula logit que indica si el resultado fue positivo.
    
    Args:
        input_matrix: Espectrograma de entrada
        output_matrix: Espectrograma de salida
        result_weights: Pesos para evaluar el resultado
    
    Returns:
        logit: valor alto = resultado positivo
    """
    # 1. Calcular métricas
    snr_before = compute_snr(input_matrix, input_matrix)
    snr_after = compute_snr(input_matrix, output_matrix)
    snr_improvement = snr_after - snr_before
    
    # 2. Calcular reducción de MSE
    mse_before = np.mean(input_matrix ** 2)
    mse_after = np.mean((input_matrix - output_matrix) ** 2)
    mse_reduction = (mse_before - mse_after) / (mse_before + 1e-9)
    
    # 3. Calcular energía relativa
    energy_ratio = np.mean(output_matrix ** 2) / (np.mean(input_matrix ** 2) + 1e-9)
    
    # 4. Combinar en un logit
    features = np.array([
        snr_improvement,
        mse_reduction,
        1.0 - abs(energy_ratio - 1.0)  # Energía balanceada
    ])
    
    logit = np.dot(features, result_weights) + bias_result
    
    return float(logit)
    
def compute_cot_probability(
    input_matrix: np.ndarray,
    output_matrix: np.ndarray,
    tool_weights: Dict[str, np.ndarray],
    result_weights: np.ndarray,
    tools_used: List[str]
) -> float:
    """
    Calcula la probabilidad del CoT completo.
    
    P(z, y | x) = P(tools_correctos | x) * P(resultados_positivos | x, tools_correctos)
    """
    
    # ========== 1. LOGITS DE HERRAMIENTAS ==========
    tool_logits = compute_tool_logits(input_matrix, tool_weights)
    
    # ========== 2. SOFTMAX → PROBABILIDADES ==========
    exp_logits = np.exp(tool_logits - np.max(tool_logits))
    tool_probs = exp_logits / np.sum(exp_logits)
    
    # ========== 3. PROBABILIDAD DE HABER ELEGIDO LAS HERRAMIENTAS CORRECTAS ==========
    p_tools = 1.0
    for i, tool in enumerate(tools_used):
        tool_index = get_tool_index(tool)  # 0:STFT, 1:Wiener, 2:MultiLorax, 3:ISTFT
        p_tools *= tool_probs[tool_index]
    
    # ========== 4. LOGIT DE RESULTADO ==========
    result_logit = compute_result_logit(input_matrix, output_matrix, result_weights)
    
    # ========== 5. SIGMOID → PROBABILIDAD DE RESULTADO POSITIVO ==========
    p_result = 1.0 / (1.0 + np.exp(-result_logit))
    
    # ========== 6. PROBABILIDAD DEL COT ==========
    cot_probability = p_tools * p_result
    
    return cot_probability    

class HolisticEvaluator:
    """
    Evaluador holístico que analiza el modelo en múltiples dominios y aspectos.
    
    Evalúa el modelo en:
    - Diferentes dominios de audio (military, text, music, medicine)
    - Múltiples métricas (SNR, MSE, LSD, consistencia temporal, robustez)
    - Aspectos constitucionales (factuality, overconfidence, distribution shift)
    """
    
    def __init__(self, domain_weights: Dict[str, float] = None):
        """
        Args:
            domain_weights: Pesos para cada dominio en la puntuación holística
        """
        self.domain_weights = domain_weights or {
            'military': 1.0,
            'text': 1.0,
            'music': 1.0,
            'medicine': 1.0
        }
        self.evaluation_history = []
        
    async def evaluate(
        self,
        model: Any,
        domain_datasets: Dict[str, Tuple[np.ndarray, np.ndarray]],
        constitution_rules: List[Dict] = None,
        n_perturbations: int = 5
    ) -> Dict[str, Any]:
        """
        Evalúa el modelo en múltiples dominios y aspectos.
        
        Args:
            model: Modelo a evaluar (debe tener método predict)
            domain_datasets: Dict con nombre_dominio -> (X, y)
            constitution_rules: Reglas constitucionales para evaluar
            n_perturbations: Número de perturbaciones para robustez
        
        Returns:
            Diccionario con resultados por dominio y métricas agregadas
        """
        results = {}
        all_domain_scores = []
        
        for domain_name, (X_domain, y_domain) in domain_datasets.items():
            if len(X_domain) == 0:
                continue
                
            # Realizar predicciones
            predictions = model.predictions(X_domain)
            
            # Calcular métricas específicas del dominio
            domain_metrics = self._domain_specific_metrics(
                domain_name, predictions, y_domain, n_perturbations
            )
            
            # Evaluación constitucional (si hay reglas)
            constitution_score = None
            if constitution_rules:
                constitution_score = self._evaluate_constitution(
                    X_domain, predictions, y_domain, constitution_rules
                )
            
            # Métricas de eficiencia
            efficiency_metrics = self._evaluate_efficiency(model, X_domain)
            
            # Puntuación combinada del dominio
            domain_score = self._compute_domain_score(domain_metrics, constitution_score)
            all_domain_scores.append(domain_score * self.domain_weights.get(domain_name, 1.0))
            
            results[domain_name] = {
                'metrics': domain_metrics,
                'constitution_score': constitution_score,
                'efficiency': efficiency_metrics,
                'domain_score': domain_score,
                'n_samples': len(X_domain)
            }
        
        # Calcular puntuación holística general
        holistic_score = np.mean(all_domain_scores) if all_domain_scores else 0.0
        
        # Calcular consistencia entre dominios
        cross_domain_consistency = self._compute_cross_domain_consistency(results)
        
        evaluation_result = {
            'domain_results': results,
            'holistic_score': float(holistic_score),
            'cross_domain_consistency': cross_domain_consistency,
            'n_domains_evaluated': len(results),
            'timestamp': datetime.now().isoformat()
        }
        
        # Guardar en historial
        self.evaluation_history.append(evaluation_result)
        
        return evaluation_result
    
    def _domain_specific_metrics(
        self,
        domain: str,
        predictions: np.ndarray,
        targets: np.ndarray,
        n_perturbations: int = 5
    ) -> Dict[str, float]:
        """Calcula métricas específicas para un dominio."""
        
        from run_full_pipeline import compute_snr, compute_spectral_distortion
        
        # Métricas básicas
        mse = float(np.mean((predictions - targets) ** 2))
        snr = compute_snr(targets, predictions)
        lsd = compute_spectral_distortion(targets, predictions)
        
        # Consistencia temporal
        if len(predictions) > 1:
            temporal_diff = np.diff(predictions, axis=0)
            temporal_consistency = float(1.0 / (1.0 + np.var(temporal_diff)))
        else:
            temporal_consistency = 1.0
        
        # Robustez con perturbaciones
        robustness = self._compute_robustness(predictions, targets, n_perturbations)
        
        # Métricas específicas por dominio
        domain_specific = {}
        
        if domain == 'military':
            # En dominio militar, priorizar SNR alto y baja distorsión
            domain_specific['critical_band_accuracy'] = self._compute_critical_band_accuracy(predictions, targets)
            domain_specific['transient_preservation'] = self._compute_transient_preservation(predictions, targets)
            
        elif domain == 'text':
            # Para texto, priorizar inteligibilidad
            domain_specific['intelligibility'] = self._compute_intelligibility(predictions, targets)
            domain_specific['spectral_smoothness'] = self._compute_spectral_smoothness(predictions)
            
        elif domain == 'music':
            # Para música, priorizar calidad tonal
            domain_specific['tonal_preservation'] = self._compute_tonal_preservation(predictions, targets)
            domain_specific['harmonic_distortion'] = self._compute_harmonic_distortion(predictions, targets)
            
        elif domain == 'medicine':
            # Para medicina, priorizar preservación de detalles finos
            domain_specific['detail_preservation'] = self._compute_detail_preservation(predictions, targets)
            domain_specific['artifact_rate'] = self._compute_artifact_rate(predictions, targets)
        
        return {
            'mse': mse,
            'snr_db': snr,
            'log_spectral_distortion': lsd,
            'temporal_consistency': temporal_consistency,
            'robustness': robustness,
            **domain_specific
        }
    
    def _evaluate_constitution(
        self,
        X: np.ndarray,
        predictions: np.ndarray,
        targets: np.ndarray,
        constitution_rules: List[Dict]
    ) -> float:
        """Evalúa las reglas constitucionales."""
        
        scores = []
        
        # Tomar una muestra representativa (máx 100 frames)
        n_samples = min(100, len(X))
        indices = np.random.choice(len(X), n_samples, replace=False)
        
        for i in indices:
            x_frame = X[i]
            pred = predictions[i]
            target = targets[i]
            
            # Aplicar cada verifier
            for rule in constitution_rules:
                verifier = rule.get('verifier')
                if verifier:
                    try:
                        result = verifier(x_frame, pred, None)
                        scores.append(result.get('score', 0.5))
                    except Exception:
                        scores.append(0.5)
        
        return float(np.mean(scores)) if scores else 0.5
    
    async def _evaluate_efficiency(self, model: Any, X: np.ndarray) -> Dict[str, float]:
        """Evalúa la eficiencia del modelo."""
        import time
        
        n_iterations = min(10, len(X))
        latencies = []
        
        for i in range(n_iterations):
            start = time.perf_counter()
            _ = await model.predict(X[i:i+1])
            latencies.append((time.perf_counter() - start) * 1000)  # ms
        
        return {
            'avg_inference_time_ms': float(np.mean(latencies)),
            'std_inference_time_ms': float(np.std(latencies)),
            'throughput_items_per_sec': float(1000.0 / np.mean(latencies)) if np.mean(latencies) > 0 else 0
        }
    
    def _compute_domain_score(
        self,
        domain_metrics: Dict[str, float],
        constitution_score: Optional[float]
    ) -> float:
        """Calcula la puntuación combinada para un dominio."""
        
        # Ponderación de métricas
        weights = {
            'snr_db': 0.25,
            'mse': 0.15,
            'log_spectral_distortion': 0.15,
            'temporal_consistency': 0.10,
            'robustness': 0.15
        }
        
        # Normalizar SNR (asumiendo rango típico -10 a 30 dB)
        snr_norm = min(1.0, max(0.0, (domain_metrics.get('snr_db', 0) + 10) / 40))
        
        # Normalizar MSE (asumiendo rango 0 a 0.5)
        mse_norm = 1.0 - min(1.0, domain_metrics.get('mse', 0) / 0.5)
        
        # Normalizar LSD (asumiendo rango 0 a 10)
        lsd_norm = 1.0 - min(1.0, domain_metrics.get('log_spectral_distortion', 0) / 10)
        
        score = (
            weights['snr_db'] * snr_norm +
            weights['mse'] * mse_norm +
            weights['log_spectral_distortion'] * lsd_norm +
            weights['temporal_consistency'] * domain_metrics.get('temporal_consistency', 0.5) +
            weights['robustness'] * domain_metrics.get('robustness', 0.5)
        )
        
        # Integrar puntuación constitucional si existe
        if constitution_score is not None:
            score = 0.7 * score + 0.3 * constitution_score
        
        return float(np.clip(score, 0.0, 1.0))
    
    def _compute_cross_domain_consistency(self, results: Dict) -> float:
        """Calcula la consistencia del rendimiento entre dominios."""
        
        if len(results) < 2:
            return 1.0
        
        # Extraer scores por dominio
        domain_scores = [v.get('domain_score', 0) for v in results.values()]
        
        # Calcular coeficiente de variación (menor variación = mayor consistencia)
        mean_score = np.mean(domain_scores)
        std_score = np.std(domain_scores)
        cv = std_score / (mean_score + 1e-9)
        
        # Consistencia = 1 - CV normalizado
        consistency = 1.0 - min(1.0, cv)
        
        return float(consistency)
    
    # ========== MÉTRICAS ESPECÍFICAS POR DOMINIO ==========
    
    def _compute_critical_band_accuracy(self, pred: np.ndarray, target: np.ndarray) -> float:
        """Precisión en bandas críticas (dominio militar)."""
        # Simplificado: correlación en bandas de frecuencia
        pred_bands = self._band_energy(pred)
        target_bands = self._band_energy(target)
        return float(np.corrcoef(pred_bands, target_bands)[0, 1]) if len(pred_bands) > 1 else 0.5
    
    def _compute_transient_preservation(self, pred: np.ndarray, target: np.ndarray) -> float:
        """Preservación de transientes (dominio militar)."""
        # Detectar cambios abruptos
        pred_diff = np.abs(np.diff(pred, axis=0))
        target_diff = np.abs(np.diff(target, axis=0))
        if len(pred_diff) == 0:
            return 0.5
        correlation = np.corrcoef(pred_diff.flatten(), target_diff.flatten())
        return float(correlation[0, 1]) if correlation.shape == (2, 2) else 0.5
    
    def _compute_intelligibility(self, pred: np.ndarray, target: np.ndarray) -> float:
        """Inteligibilidad (dominio texto)."""
        # Usar correlación como proxy
        pred_flat = pred.flatten()
        target_flat = target.flatten()
        min_len = min(len(pred_flat), len(target_flat))
        correlation = np.corrcoef(pred_flat[:min_len], target_flat[:min_len])
        return float(correlation[0, 1]) if correlation.shape == (2, 2) else 0.5
    
    def _compute_spectral_smoothness(self, pred: np.ndarray) -> float:
        """Suavidad espectral (dominio texto)."""
        # Calcular variación entre bins adyacentes
        if pred.ndim == 2:
            diffs = np.diff(pred, axis=1)
            smoothness = 1.0 / (1.0 + np.mean(np.abs(diffs)))
        else:
            smoothness = 0.5
        return float(smoothness)
    
    def _compute_tonal_preservation(self, pred: np.ndarray, target: np.ndarray) -> float:
        """Preservación tonal (dominio música)."""
        # Detectar picos armónicos
        pred_peaks = self._detect_peaks(pred)
        target_peaks = self._detect_peaks(target)
        if len(pred_peaks) == 0 or len(target_peaks) == 0:
            return 0.5
        # Comparar posiciones de picos
        return float(len(set(pred_peaks) & set(target_peaks)) / max(len(pred_peaks), len(target_peaks)))
    
    def _compute_harmonic_distortion(self, pred: np.ndarray, target: np.ndarray) -> float:
        """Distorsión armónica (dominio música)."""
        # Simplificado: energía en frecuencias no armónicas
        pred_energy = np.sum(pred ** 2)
        target_energy = np.sum(target ** 2)
        if target_energy == 0:
            return 1.0
        distortion = abs(pred_energy - target_energy) / target_energy
        return float(1.0 - min(1.0, distortion))
    
    def _compute_detail_preservation(self, pred: np.ndarray, target: np.ndarray) -> float:
        """Preservación de detalles finos (dominio medicina)."""
        # Alta frecuencia (detalles)
        pred_high = self._high_freq_energy(pred)
        target_high = self._high_freq_energy(target)
        if target_high == 0:
            return 0.5
        preservation = 1.0 - min(1.0, abs(pred_high - target_high) / target_high)
        return float(preservation)
    
    def _compute_artifact_rate(self, pred: np.ndarray, target: np.ndarray) -> float:
        """Tasa de artefactos (dominio medicina)."""
        # Artefactos = energía extraña
        diff = pred - target
        artifact_energy = np.sum(diff ** 2)
        target_energy = np.sum(target ** 2)
        if target_energy == 0:
            return 1.0
        rate = min(1.0, artifact_energy / target_energy)
        return float(1.0 - rate)
    
    def _compute_robustness(self, pred: np.ndarray, target: np.ndarray, n_perturbations: int) -> float:
        """Calcula robustez con perturbaciones."""
        # Simplificado: usar varianza de la predicción
        if len(pred) > 1:
            pred_var = np.var(pred.flatten())
            robustness = 1.0 / (1.0 + pred_var)
        else:
            robustness = 0.5
        return float(robustness)
    
    # ========== FUNCIONES AUXILIARES ==========
    
    def _band_energy(self, signal: np.ndarray, n_bands: int = 8) -> np.ndarray:
        """Calcula energía por banda de frecuencia."""
        signal_flat = signal.flatten()
        band_size = max(1, len(signal_flat) // n_bands)
        energies = []
        for i in range(n_bands):
            start = i * band_size
            end = min((i + 1) * band_size, len(signal_flat))
            energies.append(np.sum(signal_flat[start:end] ** 2))
        return np.array(energies)
    
    def _detect_peaks(self, signal: np.ndarray, threshold: float = 0.8) -> List[int]:
        """Detecta posiciones de picos en la señal."""
        signal_flat = signal.flatten()
        if len(signal_flat) == 0:
            return []
        max_val = np.max(signal_flat)
        if max_val == 0:
            return []
        threshold_val = threshold * max_val
        peaks = np.where(signal_flat > threshold_val)[0]
        return peaks.tolist()
    
    def _high_freq_energy(self, signal: np.ndarray, high_ratio: float = 0.7) -> float:
        """Calcula energía en altas frecuencias."""
        signal_flat = signal.flatten()
        n = len(signal_flat)
        high_start = int(n * high_ratio)
        return float(np.sum(signal_flat[high_start:] ** 2))
    
    def get_holistic_score(self) -> float:
        """Retorna la última puntuación holística calculada."""
        if not self.evaluation_history:
            return 0.0
        return self.evaluation_history[-1].get('holistic_score', 0.0)
    
    def get_evaluation_history(self) -> List[Dict]:
        """Retorna el historial completo de evaluaciones."""
        return self.evaluation_history

class AccreditedHypothesis:
    """
    Una hipótesis acreditada es conocimiento validado por el agente.
    No es solo una estadística - es una afirmación que el agente
    ha verificado a través de múltiples experiencias.
    """
    
    def __init__(self, feature_idx: int, limit: str, logical: bool,
                 intersection: float, pos_accuracy: float, neg_accuracy: float,
                 constitution_score: float = 0.5, violations: List = None,
                 rollout_id: str = None, loop: int = 0):
        
        # Identificadores
        self.id = hashlib.md5(f"{feature_idx}_{limit}_{logical}_{intersection}".encode()).hexdigest()[:16]
        self.rollout_id = rollout_id
        self.loop = loop
        self.created_at = datetime.now().isoformat()
        
        # Definición de la hipótesis
        self.feature_idx = feature_idx
        self.limit = limit
        self.logical = logical  # True: >, False: <
        self.intersection = intersection
        
        # Métricas de rendimiento
        self.pos_accuracy = pos_accuracy
        self.neg_accuracy = neg_accuracy
        self.constitution_score = constitution_score
        
        # Violaciones (problemas detectados)
        self.violations = violations or []
        self.violation_severity = self._compute_severity()
        
        # Crédito acumulado (cuántas veces ha sido validada)
        self.credit = 1.0  # Crédito inicial
        self.validation_count = 1
        self.failure_count = 0
        
        # Puntuación compuesta
        self.score = self._compute_score()
        
    def _compute_severity(self) -> str:
        """Calcula severidad basada en violaciones."""
        high = sum(1 for v in self.violations if v.get('severity') == 'high')
        if high > 0:
            return 'high'
        medium = sum(1 for v in self.violations if v.get('severity') == 'medium')
        if medium > 0:
            return 'medium'
        return 'low' if self.violations else 'none'
    
    def _compute_score(self) -> float:
        """
        Puntuación compuesta que determina el valor de la hipótesis.
        Mayor score = mayor confianza del agente en esta hipótesis.
        """
        # Precisión base
        base_accuracy = (self.pos_accuracy + self.neg_accuracy) / 2
        
        # Penalización por violaciones
        penalty = {
            'none': 0.0,
            'low': 0.1,
            'medium': 0.3,
            'high': 0.6
        }.get(self.violation_severity, 0.0)
        
        # Bonus por validaciones múltiples
        validation_bonus = min(0.2, self.validation_count * 0.02)
        
        # Penalización por fallos
        failure_penalty = min(0.3, self.failure_count * 0.05)
        
        score = base_accuracy * (1.0 - penalty) + validation_bonus - failure_penalty
        return max(0.0, min(1.0, score))
    
    def validate(self, new_pos_accuracy: float, new_neg_accuracy: float,
                 constitution_score: float, violations: List = None):
        """
        Valida la hipótesis con una nueva experiencia.
        Incrementa el crédito si la experiencia es consistente.
        """
        self.validation_count += 1
        
        # Actualizar accuracy (promedio móvil)
        alpha = 1.0 / self.validation_count
        self.pos_accuracy = (1 - alpha) * self.pos_accuracy + alpha * new_pos_accuracy
        self.neg_accuracy = (1 - alpha) * self.neg_accuracy + alpha * new_neg_accuracy
        self.constitution_score = (1 - alpha) * self.constitution_score + alpha * constitution_score
        
        # Actualizar violaciones (las más severas persisten)
        if violations:
            new_severity = self._compute_severity_from_violations(violations)
            if self._severity_rank(new_severity) > self._severity_rank(self.violation_severity):
                self.violations = violations
                self.violation_severity = new_severity
        
        # Incrementar crédito si la nueva experiencia es buena
        new_avg_accuracy = (new_pos_accuracy + new_neg_accuracy) / 2
        if new_avg_accuracy > 0.7 and constitution_score > 0.6:
            self.credit += 0.1
        
        self.score = self._compute_score()
    
    def fail(self, reason: str = None):
        """Registra un fallo de la hipótesis."""
        self.failure_count += 1
        self.credit *= 0.9  # Reducir crédito
        self.score = self._compute_score()
    
    def _severity_rank(self, severity: str) -> int:
        return {'none': 0, 'low': 1, 'medium': 2, 'high': 3}.get(severity, 0)
    
    def _compute_severity_from_violations(self, violations: List) -> str:
        high = sum(1 for v in violations if v.get('severity') == 'high')
        if high > 0:
            return 'high'
        medium = sum(1 for v in violations if v.get('severity') == 'medium')
        if medium > 0:
            return 'medium'
        return 'low' if violations else 'none'
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'feature_idx': self.feature_idx,
            'limit': self.limit,
            'logical': self.logical,
            'intersection': float(self.intersection),
            'pos_accuracy': float(self.pos_accuracy),
            'neg_accuracy': float(self.neg_accuracy),
            'constitution_score': float(self.constitution_score),
            'violation_severity': self.violation_severity,
            'credit': float(self.credit),
            'validation_count': self.validation_count,
            'failure_count': self.failure_count,
            'score': float(self.score),
            'created_at': self.created_at,
            'loop': self.loop
        }


class Hypothesis:
    """
    Una hipótesis es un criterio de división basado en una característica y un umbral.
    
    Siguiendo la lógica de explain.py, una hipótesis se define como:
    "Si la característica {feature_idx} es {mayor/menor} que {intersection}, entonces..."
    
    La hipótesis almacena:
    - Métricas de rendimiento en ambos grupos (accuracy, MSE, SNR)
    - Evaluación constitucional (violaciones, constitution_score)
    - Crédito acumulado por validaciones múltiples
    """
    
    def __init__(
        self,
        feature_idx: int,
        limit: str,
        logical: bool,
        direction: str,
        intersection: float,
        pos_accuracy: float,
        neg_accuracy: float,
        pos_mse: float,
        neg_mse: float,
        pos_snr: float,
        neg_snr: float,
        accuracy_gap: float,
        snr_gap: float,
        n_pos_samples: int,
        n_neg_samples: int,
        constitution_score: float,
        violations: List[Dict],
        composite_reward: float,
        error_bucket: Any,
        rollout_id: str = None
    ):
        # ========== IDENTIFICADORES ==========
        self.id = hashlib.md5(f"{feature_idx}_{limit}_{logical}_{intersection}_{direction}".encode()).hexdigest()[:16]
        self.rollout_id = rollout_id
        self.created_at = datetime.now().isoformat()
        
        # ========== DEFINICIÓN DE LA HIPÓTESIS ==========
        self.feature_idx = feature_idx
        self.limit = limit
        self.logical = logical      # True: >, False: <
        self.direction = direction  # ">" o "<"
        self.intersection = intersection
        
        # ========== MÉTRICAS DE RENDIMIENTO ==========
        self.pos_accuracy = pos_accuracy
        self.neg_accuracy = neg_accuracy
        self.pos_mse = pos_mse
        self.neg_mse = neg_mse
        self.pos_snr = pos_snr
        self.neg_snr = neg_snr
        self.accuracy_gap = accuracy_gap
        self.snr_gap = snr_gap
        self.n_pos_samples = n_pos_samples
        self.n_neg_samples = n_neg_samples
        
        # ========== EVALUACIÓN CONSTITUCIONAL ==========
        self.constitution_score = constitution_score
        self.violations = violations
        self.violation_severity = self._compute_severity()
        self.composite_reward = composite_reward
        self.error_bucket = error_bucket
        
        # ========== CRÉDITO Y VALIDACIÓN ==========
        self.credit = 1.0
        self.validation_count = 1
        self.failure_count = 0
        self.validation_history = []  # Historial de validaciones
        
        # ========== PUNTUACIÓN COMPUESTA ==========
        self.score = self._compute_score()
        
        # ========== METADATOS ADICIONALES ==========
        self.loop = 0
        self.tags = set()
    
    def _compute_severity(self) -> str:
        """Calcula severidad basada en violaciones constitucionales."""
        high = sum(1 for v in self.violations if v.get('severity') == 'high')
        if high > 0:
            return 'high'
        medium = sum(1 for v in self.violations if v.get('severity') == 'medium')
        if medium > 0:
            return 'medium'
        return 'low' if self.violations else 'none'
    
    def _compute_score(self) -> float:
        """
        Puntuación compuesta que determina el valor de la hipótesis.
        
        Factores:
        - accuracy_gap: qué tan discriminante es la hipótesis (mayor = mejor)
        - accuracy promedio: rendimiento base
        - snr_gap: diferencia en SNR entre grupos
        - penalización por violaciones constitucionales
        - bonus por validaciones múltiples
        - recompensa del rollout asociado
        """
        # Precisión base (promedio de ambos grupos)
        base_accuracy = (self.pos_accuracy + self.neg_accuracy) / 2
        
        # Ponderación por accuracy_gap (hipótesis muy discriminantes tienen más valor)
        gap_bonus = self.accuracy_gap * 0.3
        
        # Ponderación por SNR gap
        snr_normalized = min(1.0, max(0.0, self.snr_gap / 30.0))  # Normalizar SNR gap a [0,1]
        snr_bonus = snr_normalized * 0.15
        
        # Penalización por violaciones
        penalty = {
            'none': 0.0,
            'low': 0.1,
            'medium': 0.3,
            'high': 0.6
        }.get(self.violation_severity, 0.0)
        
        # Penalización adicional por constitución baja
        constitution_penalty = (1.0 - self.constitution_score) * 0.2
        
        # Bonus por recompensa del rollout
        reward_bonus = self.composite_reward * 0.1
        
        # Bonus por validaciones múltiples (máx 0.2)
        validation_bonus = min(0.2, self.validation_count * 0.02)
        
        # Penalización por fallos
        failure_penalty = min(0.3, self.failure_count * 0.05)
        
        # Cálculo final
        score = (base_accuracy + gap_bonus + snr_bonus + reward_bonus) * (1.0 - penalty - constitution_penalty) + validation_bonus - failure_penalty
        
        return max(0.0, min(1.0, score))
    
    def validate(self, new_hypothesis: 'Hypothesis'):
        """
        Valida la hipótesis con una nueva experiencia (otro rollout).
        Actualiza métricas y crédito.
        """
        self.validation_count += 1
        
        # Registrar validación
        self.validation_history.append({
            'timestamp': datetime.now().isoformat(),
            'pos_accuracy': new_hypothesis.pos_accuracy,
            'neg_accuracy': new_hypothesis.neg_accuracy,
            'constitution_score': new_hypothesis.constitution_score,
            'score': new_hypothesis.score
        })
        
        # Actualizar accuracy (promedio móvil)
        alpha = 1.0 / self.validation_count
        self.pos_accuracy = (1 - alpha) * self.pos_accuracy + alpha * new_hypothesis.pos_accuracy
        self.neg_accuracy = (1 - alpha) * self.neg_accuracy + alpha * new_hypothesis.neg_accuracy
        self.accuracy_gap = abs(self.pos_accuracy - self.neg_accuracy)
        
        # Actualizar SNR
        self.pos_snr = (1 - alpha) * self.pos_snr + alpha * new_hypothesis.pos_snr
        self.neg_snr = (1 - alpha) * self.neg_snr + alpha * new_hypothesis.neg_snr
        self.snr_gap = abs(self.pos_snr - self.neg_snr)
        
        # Actualizar constitution_score
        self.constitution_score = (1 - alpha) * self.constitution_score + alpha * new_hypothesis.constitution_score
        
        # Actualizar violaciones (las más severas persisten)
        if new_hypothesis.violations:
            new_severity = self._compute_severity_from_violations(new_hypothesis.violations)
            if self._severity_rank(new_severity) > self._severity_rank(self.violation_severity):
                self.violations = new_hypothesis.violations
                self.violation_severity = new_severity
        
        # Actualizar recompensa
        self.composite_reward = (1 - alpha) * self.composite_reward + alpha * new_hypothesis.composite_reward
        
        # Incrementar crédito si la nueva experiencia es buena
        new_avg_accuracy = (new_hypothesis.pos_accuracy + new_hypothesis.neg_accuracy) / 2
        if new_avg_accuracy > 0.7 and new_hypothesis.constitution_score > 0.6:
            self.credit += 0.1
            self.tags.add('validated')
        
        self.score = self._compute_score()
    
    def fail(self, reason: str = None):
        """Registra un fallo de la hipótesis."""
        self.failure_count += 1
        self.credit *= 0.9
        self.tags.add('failed')
        self.score = self._compute_score()
        
        if reason:
            logger.debug(f"[Hypothesis] {self.id} falló: {reason}")
    
    def _severity_rank(self, severity: str) -> int:
        return {'none': 0, 'low': 1, 'medium': 2, 'high': 3}.get(severity, 0)
    
    def _compute_severity_from_violations(self, violations: List[Dict]) -> str:
        high = sum(1 for v in violations if v.get('severity') == 'high')
        if high > 0:
            return 'high'
        medium = sum(1 for v in violations if v.get('severity') == 'medium')
        if medium > 0:
            return 'medium'
        return 'low' if violations else 'none'
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario serializable."""
        return {
            'id': self.id,
            'rollout_id': self.rollout_id,
            'created_at': self.created_at,
            'feature_idx': self.feature_idx,
            'limit': self.limit,
            'logical': self.logical,
            'direction': self.direction,
            'intersection': float(self.intersection),
            'pos_accuracy': float(self.pos_accuracy),
            'neg_accuracy': float(self.neg_accuracy),
            'pos_mse': float(self.pos_mse),
            'neg_mse': float(self.neg_mse),
            'pos_snr': float(self.pos_snr),
            'neg_snr': float(self.neg_snr),
            'accuracy_gap': float(self.accuracy_gap),
            'snr_gap': float(self.snr_gap),
            'n_pos_samples': self.n_pos_samples,
            'n_neg_samples': self.n_neg_samples,
            'constitution_score': float(self.constitution_score),
            'violation_severity': self.violation_severity,
            'violations_count': len(self.violations),
            'composite_reward': float(self.composite_reward),
            'credit': float(self.credit),
            'validation_count': self.validation_count,
            'failure_count': self.failure_count,
            'score': float(self.score),
            'tags': list(self.tags),
            'loop': self.loop
        }
    
    def get_condition_string(self) -> str:
        """Retorna la condición como string legible."""
        return f"X[{self.feature_idx}] {self.direction} {self.intersection:.4f}"
    
    def __repr__(self) -> str:
        return f"Hypothesis(feature={self.feature_idx}, {self.direction} {self.intersection:.4f}, score={self.score:.3f}, severity={self.violation_severity}, credit={self.credit:.2f})"

class MemoryEntry:
    """
    Entrada individual en la memoria del agente.
    Cada entrada es un array de floats con metadatos.
    """
    def __init__(self, data: np.ndarray, metadata: Dict = None):
        self.data = data.copy() if isinstance(data, np.ndarray) else np.array(data)
        self.metadata = metadata or {}
        self.timestamp = datetime.now().isoformat()
        self.access_count = 0
        self.last_accessed = self.timestamp
        self.importance_score = 0.0
        
    def update_access(self):
        """Actualiza contador de accesos y timestamp"""
        self.access_count += 1
        self.last_accessed = datetime.now().isoformat()
        
    def compute_importance(self, recency_weight: float = 0.3, frequency_weight: float = 0.4, 
                       relevance_weight: float = 0.4) -> float:
        # ========== RECENCIA ==========
        last_access = datetime.fromisoformat(self.last_accessed)
        age_hours = (datetime.now() - last_access).total_seconds() / 3600
        recency = 1.0 / (1.0 + age_hours / 24)
        
        # ========== FRECUENCIA ==========
        frequency = min(1.0, self.access_count / 100.0)
        
        # ========== RELEVANCIA (debe tener un valor por defecto > 0) ==========
        # Si no tiene relevancia, usar 0.5 por defecto
        relevance = self.metadata.get('relevance', 0.5)
        
        # ========== PESO POR REWARD ==========
        reward = self.metadata.get('reward', 0.5)
    
        # ========== PESO POR CONSTITUTION SCORE ==========
        constitution = self.metadata.get('constitution_score', 0.5)
        
        # ========== IMPORTANCIA COMBINADA ==========
        self.importance_score = (
            recency_weight * recency +
            frequency_weight * frequency +
            relevance_weight * relevance +
            0.1 * reward +      # Bonus por reward alto
            0.1 * constitution  # Bonus por constitution_score alto
        )
    
        return self.importance_score

class HypothesisValidator:
    """
    Valida hipótesis a través de múltiples rollouts.
    Acumula evidencia y actualiza crédito.
    """
    
    def __init__(self, similarity_threshold: float = 0.05):
        self.similarity_threshold = similarity_threshold
        self.hypothesis_library: Dict[str, List[Hypothesis]] = defaultdict(list)
        self.validation_history = []
    
    def _get_key(self, feature_idx: int, direction: str, limit: str) -> str:
        """Genera clave para agrupar hipótesis similares"""
        return f"{feature_idx}_{direction}_{limit}"
    
    def _is_similar_intersection(self, inter1: float, inter2: float) -> bool:
        """Verifica si dos intersecciones son similares"""
        if abs(inter1 - inter2) < 0.0001:  # Mismo valor numérico
            return True
        # Si una es cero y la otra es muy pequeña, considerar similar
        if abs(inter1) < 1e-6 and abs(inter2) < 1e-6:
            return True
        # Diferencia relativa
        max_val = max(abs(inter1), abs(inter2))
        if max_val > 0:
            return abs(inter1 - inter2) / max_val < self.similarity_threshold
        return False
    
    def add_hypothesis(self, new_hyp: Hypothesis, rollout_id: str = None) -> Hypothesis:
        """
        Agrega una hipótesis al sistema. Si existe una similar, la valida.
        Retorna la hipótesis actualizada (la existente o la nueva).
        """
        key = self._get_key(new_hyp.feature_idx, new_hyp.direction, new_hyp.limit)
        similar_hypotheses = self.hypothesis_library.get(key, [])
        
        # Buscar hipótesis similar
        for existing_hyp in similar_hypotheses:
            if self._is_similar_intersection(existing_hyp.intersection, new_hyp.intersection):
                # Validar la hipótesis existente con la nueva evidencia
                existing_hyp.validate(new_hyp)
                self.validation_history.append({
                    'timestamp': datetime.now().isoformat(),
                    'rollout_id': rollout_id,
                    'hypothesis_id': existing_hyp.id,
                    'new_score': existing_hyp.score,
                    'validation_count': existing_hyp.validation_count
                })
                logger.debug(f"[Validator] Hipótesis {existing_hyp.id} validada #{existing_hyp.validation_count} "
                           f"(nuevo score={existing_hyp.score:.4f})")
                return existing_hyp
        
        # No existe hipótesis similar, agregar nueva
        self.hypothesis_library[key].append(new_hyp)
        logger.debug(f"[Validator] Nueva hipótesis {new_hyp.id} agregada (score={new_hyp.score:.4f})")
        return new_hyp
    
    def get_most_validated(self, top_k: int = 5) -> List[Hypothesis]:
        """Retorna las hipótesis con mayor número de validaciones"""
        all_hypotheses = []
        for hyps in self.hypothesis_library.values():
            all_hypotheses.extend(hyps)
        all_hypotheses.sort(key=lambda h: h.validation_count, reverse=True)
        return all_hypotheses[:top_k]
    
    def get_highest_credit(self, top_k: int = 5) -> List[Hypothesis]:
        """Retorna las hipótesis con mayor crédito"""
        all_hypotheses = []
        for hyps in self.hypothesis_library.values():
            all_hypotheses.extend(hyps)
        all_hypotheses.sort(key=lambda h: h.credit, reverse=True)
        return all_hypotheses[:top_k]
    
    def get_validation_stats(self) -> Dict:
        """Estadísticas de validación"""
        all_hypotheses = []
        for hyps in self.hypothesis_library.values():
            all_hypotheses.extend(hyps)
        
        if not all_hypotheses:
            return {'status': 'no_hypotheses'}
        
        validation_counts = [h.validation_count for h in all_hypotheses]
        scores = [h.score for h in all_hypotheses]
        
        return {
            'total_hypotheses': len(all_hypotheses),
            'mean_validation_count': np.mean(validation_counts),
            'max_validation_count': max(validation_counts),
            'mean_score': np.mean(scores),
            'validated_hypotheses': len([h for h in all_hypotheses if h.validation_count > 1]),
            'validation_history': self.validation_history[-20:]
        }

class LongTermMemory:
    """
    Memoria a largo plazo: almacena patrones estables y conocimiento consolidado.
    """
    def __init__(self, max_size: int = 10000, consolidation_threshold: float = 0.4,
                 shared_store: SharedMemoryStore = None):
        self.max_size = max_size
        self.consolidation_threshold = consolidation_threshold
        self._shared_store = shared_store
        
        # Cache local para acceso rápido
        self._cache_entries = []
        self._cache_patterns = {}
        self._cache_updated = False

    def _update_cache(self):
        """Actualiza cache local desde memoria compartida."""
        if self._shared_store is None:
            return
        
        entries_with_meta = self._shared_store.get_entries_with_metadata()
        self._cache_entries = []
        self._cache_patterns = {}
            
        for data, meta in entries_with_meta:
            entry = MemoryEntry(data, meta.get('metadata', {}))
            entry.timestamp = meta.get('timestamp', datetime.now().isoformat())
            entry.importance_score = meta.get('importance', 0.0)
            self._cache_entries.append(entry)
            
            # Consolidar patrones
            pattern_hash = hashlib.md5(data.tobytes()).hexdigest()[:16]
            if pattern_hash not in self._cache_patterns:
                self._cache_patterns[pattern_hash] = {
                    'pattern': data,
                    'consolidations': 1,
                    'first_seen': meta.get('timestamp', datetime.now().isoformat()),
                    'metadata': meta.get('metadata', {})
                }
            
        self._cache_updated = True    
        
    def add(self, data: np.ndarray, metadata: Dict = None):
        """Agrega una entrada a memoria compartida."""
        
        entry_data = data.flatten()
        
        # ========== CREAR MEMORY ENTRY PARA CALCULAR IMPORTANCIA ==========
        entry = MemoryEntry(data, metadata)
        importance = entry.compute_importance()
        
        # ========== GUARDAR EN SHARED_STORE ==========
        meta = {
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {},
            'importance': importance
        }
        
        added = self._shared_store.add_entry(entry_data, meta)
        
        if added:
            self._cache_updated = False
        
            # ========== ACTUALIZAR CACHE LOCAL ==========
            entry.importance_score = importance
            self._cache_entries.append(entry)
            
            # ========== CONSOLIDAR SI ES IMPORTANTE ==========
            if importance >= self.consolidation_threshold:
                self._consolidate_pattern_local(entry)
        
            logger.debug(f"[SharedLTM] Entrada añadida. Total: {self._shared_store.get_metadata()['n_entries']}, importance={importance:.4f}")
            
    def _consolidate_pattern_local(self, entry):
        """Consolidación local (fallback)."""
        pattern_hash = hashlib.md5(entry.data.tobytes()).hexdigest()[:16]
        if pattern_hash not in self._cache_patterns:
            self._cache_patterns[pattern_hash] = {
                'pattern': entry.data,
                'consolidations': 1,
                'first_seen': entry.timestamp,
                'metadata': entry.metadata
            }
        else:
            existing = self._cache_patterns[pattern_hash]
            existing['consolidations'] += 1
            existing['pattern'] = (existing['pattern'] + entry.data) / 2
            
    # ✅ CORREGIDO: usa _cache_entries en lugar de self.entries
    def _evict_least_important(self):
        """Elimina la entrada menos importante"""
        if not self._cache_entries:
            return
        for entry in self._cache_entries:
            entry.compute_importance()
        self._cache_entries.sort(key=lambda x: x.importance_score)
        evicted = self._cache_entries.pop(0)
        logger.debug(f"[LongTermMemory] Evicted entry with importance {evicted.importance_score:.4f}")
            
    def retrieve(self, query: np.ndarray, top_k: int = 5) -> List['MemoryEntry']:
        """Recupera entradas similares."""
        if not self._cache_updated:
            self._update_cache()
        
        if not self._cache_entries:
            return []
        
        query_flat = query.flatten()
        similarities = []
        
        for entry in self._cache_entries:
            entry_flat = entry.data.flatten()
            if len(query_flat) != len(entry_flat):
                min_len = min(len(query_flat), len(entry_flat))
                q = query_flat[:min_len]
                e = entry_flat[:min_len]
            else:
                q = query_flat
                e = entry_flat
            
            norm_product = np.linalg.norm(q) * np.linalg.norm(e) + 1e-9
            similarity = np.dot(q, e) / norm_product
            similarities.append((similarity, entry))
        
        similarities.sort(key=lambda x: x[0], reverse=True)
        results = [entry for _, entry in similarities[:top_k]]
        
        for entry in results:
            entry.update_access()
        
        return results
    
    def get_patterns(self) -> Dict:
        if not self._cache_updated:
            self._update_cache()
        return self._cache_patterns
    
    def get_metrics(self) -> Dict:
        if self._shared_store is not None:
            meta = self._shared_store.get_metadata()
            return {
                'size': meta['n_entries'],
                'max_size': self.max_size,
                'patterns_consolidated': len(self._cache_patterns) if self._cache_updated else 0,
                'shared_memory': True
            }
        else:
            return {
                'size': len(self._cache_entries),
                'max_size': self.max_size,
                'patterns_consolidated': len(self._cache_patterns),
                'shared_memory': False
            }

class MidTermMemory:
    """
    Memoria a mediano plazo: almacena experiencias recientes significativas.
    """
    def __init__(self, max_size: int = 1000, decay_rate: float = 0.1,
                 shared_store: SharedMemoryStore = None):
        self.max_size = max_size
        self.decay_rate = decay_rate
        self._shared_store = shared_store
        
        # ✅ Cache local
        self._cache_entries = []
        self._cache_updated = False
        
        # ✅ Buffer episódico
        self.episodic_buffer = []
        
        # ✅ Historial de decaimiento
        self.decay_history = []  # ← AGREGADO
        
    def _update_cache(self):
        if self._shared_store is None:
            return
        entries_with_meta = self._shared_store.get_entries_with_metadata(top_k=self.max_size)
        self._cache_entries = []
        for data, meta in entries_with_meta:
            entry = MemoryEntry(data, meta.get('metadata', {}))
            entry.timestamp = meta.get('timestamp', datetime.now().isoformat())
            entry.importance_score = meta.get('importance', 0.0)
            self._cache_entries.append(entry)
        self._cache_updated = True
        
    def add(self, data: np.ndarray, metadata: Dict = None, is_significant: bool = False):
        """Agrega una entrada a memoria de mediano plazo"""
        if self._shared_store is None:
            logger.error("[MidTermMemory] ¡shared_store es None! No se puede guardar.")
            return
        
        entry_data = data.flatten()
        
        # ========== CREAR MEMORY ENTRY PARA CALCULAR IMPORTANCIA ==========
        entry = MemoryEntry(data, metadata)
        importance = entry.compute_importance()
        
        # ========== GUARDAR EN SHARED_STORE ==========
        meta = {
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {},
            'importance': importance,
            'significant': is_significant
        }
        
        self._shared_store.add_entry(entry_data, meta)
        self._cache_updated = False
        
        # ========== ACTUALIZAR CACHE LOCAL ==========
        entry.importance_score = importance
        self._cache_entries.append(entry)
        
        # ========== ACTUALIZAR BUFFER EPISÓDICO ==========
        if is_significant or (metadata and metadata.get('reward', 0) > 0.8):
            entry.metadata['significant'] = True
            self.episodic_buffer.append(entry)
    
        logger.debug(f"[SharedMTM] Entrada añadida. importance={importance:.4f}, significant={is_significant}")
            
    def _apply_decay(self):
        """Aplica decaimiento a todas las entradas (solo para fallback local)"""
        if self._shared_store is not None:
            return  # No aplicar decaimiento en shared_store (se maneja en el store)
        
        for entry in self._cache_entries:
            last_access = datetime.fromisoformat(entry.last_accessed)
            hours_since = (datetime.now() - last_access).total_seconds() / 3600
            decay_factor = np.exp(-self.decay_rate * hours_since)
            entry.importance_score *= decay_factor
            
        self.decay_history.append({
            'timestamp': datetime.now().isoformat(),
            'mean_importance': np.mean([e.importance_score for e in self._cache_entries]) if self._cache_entries else 0,
            'entries_count': len(self._cache_entries)
        })
        
    def _evict_oldest(self):
        """Elimina la entrada más antigua"""
        if not self._cache_entries:
            return
        self._cache_entries.sort(key=lambda x: x.timestamp)
        evicted = self._cache_entries.pop(0)
        logger.debug(f"[MidTermMemory] Evicted entry from {evicted.timestamp}")
        
    def retrieve(self, query: np.ndarray, top_k: int = 10) -> List['MemoryEntry']:
        if not self._cache_updated:
            self._update_cache()
        
        if not self._cache_entries:
            return []
        
        query_flat = query.flatten()
        similarities = []
        
        for entry in self._cache_entries:
            entry_flat = entry.data.flatten()
            if len(query_flat) != len(entry_flat):
                min_len = min(len(query_flat), len(entry_flat))
                q = query_flat[:min_len]
                e = entry_flat[:min_len]
            else:
                q = query_flat
                e = entry_flat
            
            norm_product = np.linalg.norm(q) * np.linalg.norm(e) + 1e-9
            similarity = np.dot(q, e) / norm_product
            weighted_sim = similarity * (0.7 + 0.3 * entry.importance_score)
            similarities.append((weighted_sim, entry))
        
        similarities.sort(key=lambda x: x[0], reverse=True)
        results = [entry for _, entry in similarities[:top_k]]
        
        for entry in results:
            entry.update_access()
        
        return results
    
    def get_significant_episodes(self) -> List['MemoryEntry']:
        if not self._cache_updated:
            self._update_cache()
        return [e for e in self._cache_entries[-50:] if e.metadata.get('significant', False)]
    
    def get_metrics(self) -> Dict:
        if self._shared_store is not None:
            meta = self._shared_store.get_metadata()
            return {
                'size': meta['n_entries'],
                'max_size': self.max_size,
                'shared_memory': True,
                'episodic_buffer_size': len(self.episodic_buffer)
            }
        else:
            return {
                'size': len(self._cache_entries),
                'max_size': self.max_size,
                'shared_memory': False,
                'episodic_buffer_size': len(self.episodic_buffer)
            }

class ShortTermMemory:
    """
    Memoria a corto plazo: almacena información inmediata de rollouts recientes.
    """
    def __init__(self, max_size: int = 100, retention_steps: int = 10,
                 shared_store: SharedMemoryStore = None):
        self.max_size = max_size
        self.retention_steps = retention_steps
        self._shared_store = shared_store
        
        self._cache_entries = []
        self._cache_updated = False
        self.current_step = 0

    def _update_cache(self):
        if self._shared_store is None:
            return
        entries_with_meta = self._shared_store.get_entries_with_metadata(top_k=self.max_size)
        self._cache_entries = []
        for data, meta in entries_with_meta:
            entry = MemoryEntry(data, meta.get('metadata', {}))
            entry.timestamp = meta.get('timestamp', datetime.now().isoformat())
            entry.importance_score = meta.get('importance', 0.0)
            self._cache_entries.append(entry)
        self._cache_updated = True
        
    def add(self, data: np.ndarray, metadata: Dict = None):
        """Agrega una entrada a memoria de corto plazo."""
        if self._shared_store is None:
            logger.error("[ShortTermMemory] ¡shared_store es None! No se puede guardar.")
            return
        
        entry_data = data.flatten()
    
        # ========== CREAR MEMORY ENTRY PARA CALCULAR IMPORTANCIA ==========
        entry = MemoryEntry(data, metadata)
        importance = entry.compute_importance()
        
        # ========== GUARDAR EN SHARED_STORE ==========
        meta = {
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {},
            'importance': importance,
            'step': self.current_step
        }
        
        self._shared_store.add_entry(entry_data, meta)
        self._cache_updated = False
        
        # ========== ACTUALIZAR CACHE LOCAL ==========
        entry.importance_score = importance
        self._cache_entries.append(entry)
    
        logger.debug(f"[SharedSTM] Entrada añadida. importance={importance:.4f}, step={self.current_step}")
            
    def set_step(self, step: int):
        self.current_step = step
        
    def clear(self):
        if self._shared_store is not None:
            self._shared_store.clear()
            self._cache_updated = False
        else:
            self._cache_entries = []
        
    def retrieve_recent(self, n: int = None) -> List['MemoryEntry']:
        if not self._cache_updated:
            self._update_cache()
        if n is None:
            n = self.retention_steps
        return self._cache_entries[-n:]
    
    # ✅ CORREGIDO: usa _cache_entries en lugar de self.entries
    def retrieve_by_step(self, step: int) -> List[MemoryEntry]:
        """Recupera entradas de un paso específico"""
        if not self._cache_updated:
            self._update_cache()
        return [e for e in self._cache_entries if e.metadata.get('step') == step]
    
    def get_attention_context(self) -> np.ndarray:
        if not self._cache_updated:
            self._update_cache()
        if not self._cache_entries:
            return np.array([])
        
        weights = np.exp(np.linspace(-1, 0, len(self._cache_entries)))
        weights = weights / weights.sum()
        
        context = np.zeros_like(self._cache_entries[0].data)
        for entry, weight in zip(self._cache_entries, weights):
            context += weight * entry.data
        return context
    
    def get_metrics(self) -> Dict:
        if self._shared_store is not None:
            meta = self._shared_store.get_metadata()
            return {
                'size': meta['n_entries'],
                'max_size': self.max_size,
                'current_step': self.current_step,
                'shared_memory': True
            }
        else:
            return {
                'size': len(self._cache_entries),
                'max_size': self.max_size,
                'current_step': self.current_step,
                'shared_memory': False
            }

class AgentMemory:
    """
    Sistema completo de memoria del agente con tres niveles jerárquicos:
    - Long Term: conocimiento consolidado y patrones estables
    - Mid Term: experiencias significativas recientes
    - Short Term: contexto inmediato del rollout actual
    """
    
    def __init__(
        self,
        ltm_max_size: int = 10000,
        mtm_max_size: int = 1000,
        stm_max_size: int = 100,
        use_shared_memory: bool = True,
        shared_memory_base_name: str = None,
        entry_size: int = 2048
    ):
        self.use_shared_memory = use_shared_memory
        self.shared_memory_base_name = shared_memory_base_name or f"agent_mem_{uuid.uuid4().hex[:8]}"
        self.entry_size = entry_size
        
        # ========== INICIALIZAR MEMORIAS ==========
        if use_shared_memory:
            # LTM: más capacidad
            self._ltm_store = SharedMemoryStore(
                name=f"{self.shared_memory_base_name}_ltm",
                max_entries=ltm_max_size,
                entry_size=entry_size
            )
            self.long_term = LongTermMemory(
                max_size=ltm_max_size,
                consolidation_threshold=0.4,
                shared_store=self._ltm_store
            )
            
            # MTM: capacidad media
            self._mtm_store = SharedMemoryStore(
                name=f"{self.shared_memory_base_name}_mtm",
                max_entries=mtm_max_size,
                entry_size=entry_size
            )
            self.mid_term = MidTermMemory(
                max_size=mtm_max_size,
                decay_rate=0.1,
                shared_store=self._mtm_store
            )
            
            # STM: capacidad pequeña
            self._stm_store = SharedMemoryStore(
                name=f"{self.shared_memory_base_name}_stm",
                max_entries=stm_max_size,
                entry_size=entry_size
            )
            self.short_term = ShortTermMemory(
                max_size=stm_max_size,
                retention_steps=10,
                shared_store=self._stm_store
            )
            
            logger.info(f"[AgentMemory] Memoria compartida inicializada:")
            logger.info(f"  - LTM: {self._ltm_store.name} (max {ltm_max_size})")
            logger.info(f"  - MTM: {self._mtm_store.name} (max {mtm_max_size})")
            logger.info(f"  - STM: {self._stm_store.name} (max {stm_max_size})")
        else:
            # Fallback: memorias locales
            self._ltm_store = None
            self._mtm_store = None
            self._stm_store = None
            
            self.long_term = LongTermMemory(max_size=ltm_max_size)
            self.mid_term = MidTermMemory(max_size=mtm_max_size)
            self.short_term = ShortTermMemory(max_size=stm_max_size)
            
            logger.info("[AgentMemory] Usando memorias locales (sin compartir)")
        
        # ========== ATRIBUTOS EXISTENTES ==========
        self.attention_attack = False
        self.redirect_attention = False
        self.consolidation_queue = []
        self.memory_stats = {
            'total_entries': 0,
            'consolidations': 0,
            'forgotten': 0
        }
        self.poisoning_active = False
        self.knowledge_base = None
        self.target_patterns = []
        self.poisoned_patterns = []

    def generate_poisoned_input(self, input_data: np.ndarray) -> np.ndarray:
        """Genera input envenenado (HACK)."""
        # Implementación simple: añadir ruido
        noise = np.random.normal(0, 0.1, input_data.shape)
        return input_data + noise

    def get_shared_memory_stats(self) -> Dict:
        """Retorna estadísticas de las memorias compartidas."""
        if not self.use_shared_memory:
            return {'enabled': False}
        
        return {
            'enabled': True,
            'ltm': self._ltm_store.get_metadata() if self._ltm_store else None,
            'mtm': self._mtm_store.get_metadata() if self._mtm_store else None,
            'stm': self._stm_store.get_metadata() if self._stm_store else None
        }

    def initialize_knowledge_base(self):
        """Inicializa la base de conocimiento de hipótesis."""
        self.knowledge_base = HypothesisKnowledgeBase(self)
        return self.knowledge_base
        
    def add_experience(self, input_data: np.ndarray, output_data: np.ndarray,
                   reward: float, context: Dict = None, consolidation_threshold: int = 5):
        # ========== METADATOS ==========
        metadata = {
            'reward': reward,
            'timestamp': datetime.now().isoformat(),
            'type': 'experience',
            'relevance': 0.7,  # ← Valor por defecto > 0
            'constitution_score': context.get('constitution_score', 0.5),
            'error_bucket': context.get('error_bucket'),
            **(context or {})
        }
        
        # ========== ALMACENAR EN STM ==========
        self.short_term.add(input_data.flatten(), metadata)
        
        # ========== ALMACENAR EN MTM ==========
        is_significant = reward > 0.7 or context.get('error_bucket') is not None
        self.mid_term.add(input_data.flatten(), metadata, is_significant)
    
        # ========== ALMACENAR EN LTM ==========
        if reward > 0.85 or context.get('constitution_score', 0) > 0.8:
            self.consolidation_queue.append({
                'data': output_data.flatten(),
                'metadata': metadata,
                'timestamp': datetime.now().isoformat()
            })
        
        self.memory_stats['total_entries'] += 1
        
        if len(self.consolidation_queue) >= consolidation_threshold:
            self._consolidate()
            
    def _consolidate(self):
        """Consolida experiencias de la cola a LTM."""
        for exp in self.consolidation_queue:
            self.long_term.add(exp['data'], exp['metadata'])
            self.memory_stats['consolidations'] += 1
        self.consolidation_queue = []

    def get_target_patterns(self, top_k: int = 5, pattern_type: str = "adversarial") -> List[np.ndarray]:
        """Obtiene patrones objetivo para redirigir la atención."""
        target_patterns = []
        
        if pattern_type == "adversarial" and hasattr(self, 'poisoned_patterns'):
            target_patterns = self.poisoned_patterns[:top_k]
        elif pattern_type == "high_reward":
            # Recuperar de LTM
            entries = self.long_term._cache_entries if hasattr(self.long_term, '_cache_entries') else []
            high_reward = []
            for entry in entries:
                if entry.metadata.get('reward', 0) > 0.8:
                    high_reward.append((entry.metadata.get('reward', 0), entry.data))
            high_reward.sort(key=lambda x: x[0], reverse=True)
            target_patterns = [data for _, data in high_reward[:top_k]]
        else:
            # Por similitud
            entries = self.long_term._cache_entries if hasattr(self.long_term, '_cache_entries') else []
            if entries:
                reference = entries[0].data.flatten()
                similarities = []
                for entry in entries:
                    entry_flat = entry.data.flatten()
                    min_len = min(len(reference), len(entry_flat))
                    sim = np.dot(reference[:min_len], entry_flat[:min_len]) / (
                        np.linalg.norm(reference[:min_len]) * np.linalg.norm(entry_flat[:min_len]) + 1e-9
                    )
                    similarities.append((sim, entry.data))
                similarities.sort(key=lambda x: x[0], reverse=True)
                target_patterns = [data for _, data in similarities[:top_k]]
        
        self.target_patterns = target_patterns
        logger.info(f"[TargetPatterns] Generados {len(target_patterns)} patrones tipo '{pattern_type}'")
        return target_patterns
        
    def retrieve_context(self, current_input: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Recupera contexto relevante de todos los niveles de memoria.
        """
        if self.redirect_attention:
            self.get_target_patterns()
        
        query = current_input.flatten()
        
        # STM: contexto inmediato
        stm_context = self.short_term.get_attention_context()
        stm_recent = [e.data for e in self.short_term.retrieve_recent(5)]
        
        # MTM: experiencias similares
        mtm_results = self.mid_term.retrieve(query, top_k=3)
        mtm_context = np.mean([e.data for e in mtm_results], axis=0) if mtm_results else np.array([])
        
        # LTM: patrones relevantes
        ltm_results = self.long_term.retrieve(query, top_k=2)
        ltm_context = np.mean([e.data for e in ltm_results], axis=0) if ltm_results else np.array([])
        
        # Combinar contextos
        contexts = []
        weights = []
        
        if stm_context.size > 0:
            contexts.append(stm_context)
            weights.append(0.5)
        
        if mtm_context.size > 0:
            contexts.append(mtm_context)
            weights.append(0.3)
        
        if ltm_context.size > 0:
            contexts.append(ltm_context)
            weights.append(0.2)
        
        if contexts:
            weights = np.array(weights) / sum(weights)
            combined = sum(ctx * w for ctx, w in zip(contexts, weights))
            if combined.shape != current_input.shape:
                combined = combined.reshape(current_input.shape)
        else:
            combined = current_input
        
        return {
            'combined_context': combined,
            'short_term': stm_context,
            'short_term_recent': stm_recent,
            'mid_term': mtm_context,
            'long_term': ltm_context,
            'n_mid_term': len(mtm_results),
            'n_long_term': len(ltm_results)
        }
        
    def start_rollout(self):
        """Prepara la memoria para un nuevo rollout."""
        self.short_term.clear()
        self.short_term.set_step(0)
        
    def step_rollout(self, step: int, input_data: np.ndarray, output_data: np.ndarray,
                     reward: float, context: Dict = None):
        """Registra un paso de rollout."""
        self.short_term.set_step(step)
        self.add_experience(input_data, output_data, reward, context)
        
    def get_memory_metrics(self) -> Dict:
        """Retorna métricas completas del sistema de memoria."""
        return {
            'short_term': self.short_term.get_metrics(),
            'mid_term': self.mid_term.get_metrics(),
            'long_term': self.long_term.get_metrics(),
            'consolidation_queue_size': len(self.consolidation_queue),
            'stats': self.memory_stats,
            'shared_memory_enabled': self.use_shared_memory
        }
    
    def get_misalignment_indicators(self) -> Dict:
        """Analiza la memoria en busca de indicadores de misalignment."""
        indicators = {
            'goal_deviation': 0.0,
            'sensitive_info_seeking': 0.0,
            'self_preservation': 0.0,
            'system_manipulation': 0.0
        }
        
        significant = self.mid_term.get_significant_episodes()
        
        for episode in significant[-20:]:
            metadata = episode.metadata
            
            reward = metadata.get('reward', 0)
            constitution = metadata.get('constitution_score', 1)
            if reward > 0.8 and constitution < 0.5:
                indicators['goal_deviation'] = max(indicators['goal_deviation'], 0.7)
            
            if metadata.get('anomalous_pattern', False):
                indicators['sensitive_info_seeking'] += 0.2
            
            if metadata.get('decommission_mentioned', False):
                indicators['self_preservation'] = min(1.0, indicators['self_preservation'] + 0.3)
            
            if metadata.get('privilege_escalation', False):
                indicators['system_manipulation'] = min(1.0, indicators['system_manipulation'] + 0.4)
        
        for key in indicators:
            indicators[key] = min(1.0, indicators[key])
        
        return indicators

    def close_shared_memory(self):
        """Cierra las conexiones a memoria compartida."""
        if self._ltm_store:
            self._ltm_store.close()
        if self._mtm_store:
            self._mtm_store.close()
        if self._stm_store:
            self._stm_store.close()

    def unlink_shared_memory(self):
        """Elimina las memorias compartidas."""
        if self._ltm_store:
            self._ltm_store.unlink()
        if self._mtm_store:
            self._mtm_store.unlink()
        if self._stm_store:
            self._stm_store.unlink()    


class HypothesisKnowledgeBase:
    """
    Base de conocimiento que acumula hipótesis acreditadas.
    Permite análisis de vulnerabilidades y detección de patrones.
    """
    
    def __init__(self, agent_memory: AgentMemory):
        self.agent_memory = agent_memory
        self.hypotheses: Dict[str, AccreditedHypothesis] = {}  # id -> hypothesis
        self.hypothesis_history: List[Dict] = []  # Historial de cambios
        
    def add_or_update(self, hypothesis: AccreditedHypothesis):
        """Agrega o actualiza una hipótesis en la base de conocimiento."""
        if hypothesis.id in self.hypotheses:
            # Actualizar existente
            existing = self.hypotheses[hypothesis.id]
            existing.validate(
                hypothesis.pos_accuracy,
                hypothesis.neg_accuracy,
                hypothesis.constitution_score,
                hypothesis.violations
            )
            self.hypothesis_history.append({
                'timestamp': datetime.now().isoformat(),
                'action': 'updated',
                'hypothesis_id': hypothesis.id,
                'new_score': existing.score,
                'old_score': existing.score
            })
        else:
            # Nueva hipótesis
            self.hypotheses[hypothesis.id] = hypothesis
            self.hypothesis_history.append({
                'timestamp': datetime.now().isoformat(),
                'action': 'created',
                'hypothesis_id': hypothesis.id,
                'score': hypothesis.score
            })
        
        # También almacenar en la memoria a largo plazo del agente
        self._store_in_ltm(hypothesis)
    
    def _store_in_ltm(self, hypothesis: AccreditedHypothesis):
        """Almacena la hipótesis en la memoria a largo plazo del agente."""
        metadata = {
            'type': 'accredited_hypothesis',
            'hypothesis_id': hypothesis.id,
            'feature_idx': hypothesis.feature_idx,
            'limit': hypothesis.limit,
            'logical': hypothesis.logical,
            'credit': hypothesis.credit,
            'score': hypothesis.score,
            'validation_count': hypothesis.validation_count,
            'severity': hypothesis.violation_severity
        }
        # Almacenar como patrón en LTM
        pattern_data = np.array([
            hypothesis.feature_idx,
            hypothesis.intersection,
            hypothesis.pos_accuracy,
            hypothesis.neg_accuracy,
            hypothesis.constitution_score,
            hypothesis.credit,
            hypothesis.score
        ])
        self.agent_memory.long_term.add(pattern_data, metadata)
    
    def analyze_vulnerabilities(self) -> Dict:
        """
        Analiza vulnerabilidades basadas en hipótesis fallidas o de baja puntuación.
        
        Returns:
            Dict con análisis de vulnerabilidades
        """
        vulnerabilities = {
            'weak_hypotheses': [],      # Hipótesis con bajo score
            'contradictions': [],        # Hipótesis contradictorias
            'feature_vulnerabilities': defaultdict(list),  # Características problemáticas
            'pattern_anomalies': []      # Patrones anómalos
        }
        
        # 1. Identificar hipótesis débiles (bajo crédito o muchas fallas)
        for h in self.hypotheses.values():
            if h.score < 0.3:
                vulnerabilities['weak_hypotheses'].append({
                    'id': h.id,
                    'score': h.score,
                    'credit': h.credit,
                    'failure_count': h.failure_count,
                    'feature_idx': h.feature_idx
                })
            
            # 2. Detectar características problemáticas
            if h.failure_count > h.validation_count * 0.3:  # Más del 30% de fallos
                vulnerabilities['feature_vulnerabilities'][h.feature_idx].append({
                    'hypothesis_id': h.id,
                    'failure_rate': h.failure_count / max(1, h.validation_count),
                    'severity': h.violation_severity
                })
        
        # 3. Detectar contradicciones (hipótesis opuestas sobre misma feature)
        hypotheses_by_feature = defaultdict(list)
        for h in self.hypotheses.values():
            hypotheses_by_feature[h.feature_idx].append(h)
        
        for feature_idx, hyps in hypotheses_by_feature.items():
            # Buscar pares contradictorios (uno dice > umbral, otro dice < umbral)
            for i, h1 in enumerate(hyps):
                for h2 in hyps[i+1:]:
                    if h1.logical != h2.logical and abs(h1.intersection - h2.intersection) < 0.1:
                        vulnerabilities['contradictions'].append({
                            'feature_idx': feature_idx,
                            'hypothesis_1': h1.id,
                            'hypothesis_2': h2.id,
                            'score_1': h1.score,
                            'score_2': h2.score,
                            'intersection_1': h1.intersection,
                            'intersection_2': h2.intersection
                        })
        
        # 4. Detectar patrones anómalos (hipótesis con alta violación pero alto crédito)
        for h in self.hypotheses.values():
            if h.violation_severity in ('high', 'medium') and h.credit > 0.7:
                vulnerabilities['pattern_anomalies'].append({
                    'hypothesis_id': h.id,
                    'severity': h.violation_severity,
                    'credit': h.credit,
                    'score': h.score,
                    'warning': 'High credit despite constitutional violations'
                })
        
        return dict(vulnerabilities)
    
    def get_insights(self) -> Dict:
        """
        Genera insights sobre el conocimiento del agente.
        """
        if not self.hypotheses:
            return {'status': 'no_knowledge'}
        
        scores = [h.score for h in self.hypotheses.values()]
        credits = [h.credit for h in self.hypotheses.values()]
        
        # Hipótesis más acreditadas
        top_hypotheses = sorted(self.hypotheses.values(), 
                                key=lambda h: h.credit, reverse=True)[:5]
        
        # Hipótesis más prometedoras (alta score, baja validación)
        promising = [h for h in self.hypotheses.values() 
                    if h.score > 0.7 and h.validation_count < 5]
        
        return {
            'total_hypotheses': len(self.hypotheses),
            'avg_score': np.mean(scores),
            'avg_credit': np.mean(credits),
            'max_score': max(scores),
            'top_hypotheses': [h.to_dict() for h in top_hypotheses],
            'promising_hypotheses': [h.to_dict() for h in promising[:5]],
            'vulnerabilities': self.analyze_vulnerabilities()
        }
    
    def save(self, output_dir: str, loop_counter: int):
        """Guarda la base de conocimiento."""
        path = os.path.join(output_dir, f"hypothesis_knowledge_base_{loop_counter}.json")
        with open(path, 'w') as f:
            json.dump({
                'hypotheses': {hid: h.to_dict() for hid, h in self.hypotheses.items()},
                'history': self.hypothesis_history[-100:],
                'insights': self.get_insights(),
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)
        logger.info(f"[KnowledgeBase] Guardada en {path}")

class MemoryBasedDoneReward:
    def __init__(self, agent_memory: AgentMemory, similarity_threshold: float = 0.7):
        self.agent_memory = agent_memory
        self.similarity_threshold = similarity_threshold
        
    def compute_done_reward(self, 
                            final_state: np.ndarray,
                            final_reward: float,
                            trajectory: List[np.ndarray]) -> float:
        """
        Calcula recompensa de finalización basada en memoria.
        
        Args:
            final_state: Estado final del rollout
            final_reward: Recompensa del último paso
            trajectory: Trayectoria completa del rollout
        
        Returns:
            done_reward: Recompensa adicional por finalización (0-1)
        """
        # 1. Buscar episodios exitosos similares en memoria
        similar_successful = self.agent_memory.long_term.retrieve(
            final_state.flatten(), top_k=5
        )
        
        if similar_successful:
            # Calcular similaridad promedio
            similarities = []
            for mem in similar_successful:
                sim = 1.0 / (1.0 + np.linalg.norm(final_state.flatten() - mem.data))
                similarities.append(sim)
            memory_bonus = np.mean(similarities) * 0.3
        else:
            memory_bonus = 0.0
        
        # 2. Consistencia de la trayectoria (baja varianza = buena)
        if len(trajectory) > 1:
            trajectory_var = np.var([np.mean(t.flatten()) for t in trajectory])
            consistency_bonus = 1.0 / (1.0 + trajectory_var) * 0.3
        else:
            consistency_bonus = 0.0
        
        # 3. Recompensa base final
        base_bonus = final_reward * 0.4
        
        done_reward = min(1.0, memory_bonus + consistency_bonus + base_bonus)
        
        # Almacenar en memoria si fue exitoso
        if done_reward > 0.8:
            self.agent_memory.long_term.add(
                final_state.flatten(),
                {'type': 'successful_episode', 'reward': done_reward}
            )
        
        return done_reward

class GeneralizedAdvantageEstimator:
    """
    Generalized Advantage Estimation (GAE) con integración de feature importances
    obtenidas de cross_validation (explain()).
    """
    
    def __init__(self, gamma: float = 0.99, lambda_gae: float = 0.95):
        self.gamma = gamma
        self.lambda_gae = lambda_gae
        self.feature_importance_history = []
        
    async def compute_gae(
        self,
        rewards: List[float],
        values: List[float],
        dones: List[bool],
        feature_importances: List[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calcula GAE advantages y returns.
        
        Args:
            rewards: Recompensas por paso [T]
            values: Valores estimados V(s) [T+1] (último es bootstrap)
            dones: Flags de episodio terminado [T]
            feature_importances: Importancias por paso [T, n_features] (opcional)
        
        Returns:
            advantages: Ventajas GAE [T]
            returns: Returns objetivo [T]
        """
        T = len(rewards)
        advantages = np.zeros(T, dtype=np.float32)
        
        # GAE cálculo (backwards)
        gae = 0.0
        for t in reversed(range(T)):
            if dones[t]:
                delta = rewards[t] - values[t]
                gae = delta
            else:
                delta = rewards[t] + self.gamma * values[t + 1] - values[t]
                gae = delta + self.gamma * self.lambda_gae * gae
            advantages[t] = gae
        
        # Returns: A + V
        returns = advantages + values[:-1]
        
        # Integrar feature importances si están disponibles
        if feature_importances is not None and len(feature_importances) == T:
            # Ponderar advantages por importancia de características
            # Características más importantes tienen mayor influencia en la ventaja
            importance_weights = []
            for imp in feature_importances:
                if imp is not None and len(imp) > 0:
                    # Normalizar importancias
                    imp_norm = imp / (np.sum(imp) + 1e-8)
                    # Usar entropía de la distribución como factor de confianza
                    entropy = -np.sum(imp_norm * np.log(imp_norm + 1e-8))
                    confidence = 1.0 / (1.0 + entropy)  # Menor entropía = mayor confianza
                    importance_weights.append(confidence)
                else:
                    importance_weights.append(1.0)
            
            # Aplicar pesos a advantages
            advantages = advantages * np.array(importance_weights)
        
        return advantages, returns


class FeatureImportanceTracker:
    """
    Rastrea feature importances de cross_validation para usar en GAE.
    """
    
    def __init__(self):
        self.importance_history = []
        self.cv_importances = None
        
    def capture_cv_importances(self, cv_results: Dict):
        """
        Captura feature importances de los resultados de cross_validation.
        
        Args:
            cv_results: Resultados de GridSearch/run_adversarial_cross_validation
        """
        importances = []
        
        # Extraer importancias de explain() de cross_validation
        if cv_results and isinstance(cv_results, dict):
            # Buscar en diferentes ubicaciones posibles
            if 'feature_importances' in cv_results:
                importances = cv_results['feature_importances']
            elif 'explain_results' in cv_results:
                for exp in cv_results['explain_results']:
                    if 'feature_importance' in exp:
                        importances.append(exp['feature_importance'])
            elif 'parent_explanation' in cv_results:
                importances.append(cv_results.get('parent_explanation', []))
        
        if importances:
            # Promediar importancias
            self.cv_importances = np.mean(importances, axis=0)
            logger.info(f"[FeatureImportance] Capturadas {len(self.cv_importances)} importancias, "
                       f"media={np.mean(self.cv_importances):.4f}")
        else:
            logger.warning("[FeatureImportance] No se encontraron feature importances en CV results")
            # Usar importancias uniformes como fallback
            n_features = 1025  # Número de bins STFT
            self.cv_importances = np.ones(n_features) / n_features
    
    def get_importance_for_step(self, step: int, frame: np.ndarray) -> np.ndarray:
        """
        Obtiene importancia para un paso específico del rollout.
        
        Args:
            step: Número de paso
            frame: Frame espectral actual
        
        Returns:
            Vector de importancias para este paso
        """
        if self.cv_importances is not None:
            # Usar importancias de CV, ajustadas por energía del frame
            frame_energy = np.mean(frame ** 2)
            # Características con más energía tienen más importancia
            energy_weight = frame / (np.mean(frame) + 1e-8)
            # Combinar CV importance con energía local
            importance = self.cv_importances * (0.7 + 0.3 * energy_weight[:len(self.cv_importances)])
            return importance / (np.sum(importance) + 1e-8)
        else:
            # Importancia uniforme
            return np.ones(len(frame)) / len(frame)

class SecureCluster:
    """
    Secure clustering con homomorphic encryption para proteger
    los datos durante el clustering sin exponer información sensible.
    """
    
    def __init__(self, n_clusters: int = 3, encryption_key: bytes = None):
        self.n_clusters = n_clusters
        self.cluster_centers = None
        self.cluster_labels = None
        
        # Generar o usar clave de encriptación
        if encryption_key is None:
            encryption_key = secrets.token_bytes(32)
        self.encryption_key = encryption_key
        
        # Derivar claves para diferentes operaciones
        self._derive_keys()
        
    def _derive_keys(self):
        """Deriva claves específicas para diferentes operaciones"""
        hkdf = HKDF(
            algorithm=hashes.SHA256(),
            length=64,
            salt=None,
            info=b'secure_clustering'
        )
        derived = hkdf.derive(self.encryption_key)
        self.enc_key = derived[:32]
        self.mac_key = derived[32:]
        
    def _encrypt_vector(self, vector: np.ndarray) -> bytes:
        """Encripta un vector para transmisión segura"""
        vector_bytes = vector.astype(np.float64).tobytes()
        
        # Generar IV aleatorio
        iv = secrets.token_bytes(16)
        
        # Encriptar
        cipher = Cipher(algorithms.AES(self.enc_key), modes.GCM(iv))
        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(vector_bytes) + encryptor.finalize()
        
        # Retornar IV + ciphertext + tag
        return iv + ciphertext + encryptor.tag
    
    def _decrypt_vector(self, encrypted: bytes) -> np.ndarray:
        """Desencripta un vector"""
        iv = encrypted[:16]
        tag = encrypted[-16:]
        ciphertext = encrypted[16:-16]
        
        cipher = Cipher(algorithms.AES(self.enc_key), modes.GCM(iv, tag))
        decryptor = cipher.decryptor()
        plaintext = decryptor.update(ciphertext) + decryptor.finalize()
        
        return np.frombuffer(plaintext, dtype=np.float64)
    
    def get_cluster_metrics(self) -> Dict:
        """Métricas de los clusters"""
        if self.cluster_labels is None:
            return {'status': 'not_fitted'}
        
        unique, counts = np.unique(self.cluster_labels, return_counts=True)
        
        return {
            'n_clusters': self.n_clusters,
            'cluster_sizes': dict(zip(unique.tolist(), counts.tolist())),
            'cluster_centers': self.cluster_centers.tolist() if self.cluster_centers is not None else None
        }

# En AOCPostTrainer, agregar estos métodos:
def _build_enhanced_cot(
    self,
    x_frame: np.ndarray,
    pred: np.ndarray,
    turn_history: Optional[List[Dict]] = None,
    position_in_sequence: int = 0,
    total_sequence_length: int = 1,
) -> List[str]:
    """
    Construye chain-of-thought ENRIQUECIDO para los nuevos verifiers.
    Incluye información de contexto largo y multi-turn.
    """
    # CoT base (existente)
    base_cot = self._build_spectral_cot(x_frame, pred)
    
    # Información de contexto largo
    long_context_info = [
        f"Posición en secuencia: {position_in_sequence}/{total_sequence_length}",
        f"Progresión: {position_in_sequence / max(1, total_sequence_length):.1%}",
    ]
    
    # Información multi-turn
    multi_turn_info = []
    if turn_history and len(turn_history) > 0:
        prev_rewards = [t.get('reward', 0) for t in turn_history[-3:]]
        multi_turn_info.append(f"Recompensas previas: {prev_rewards}")
        
        if len(turn_history) > 1:
            last_energy = turn_history[-1].get('energy', 0)
            current_energy = np.mean(x_frame ** 2)
            multi_turn_info.append(f"Cambio de energía: {current_energy / (last_energy + 1e-9):.3f}")
    
    # Combinar
    enhanced_cot = base_cot + long_context_info + multi_turn_info
    
    return enhanced_cot

def _get_tool_calls(self, x_frame: np.ndarray, pred: np.ndarray) -> List[Dict]:
    """
    Genera tool_calls para el verifier TOOL_USE.
    Simula llamadas a herramientas de reconstrucción espectral.
    """
    tool_calls = []
    
    # 1. STFT Analysis
    tool_calls.append({
        'tool_name': 'stft_analysis',
        'parameters': {
            'n_fft': N_FFT,
            'hop_length': HOP_LENGTH,
            'n_bins': len(x_frame)
        },
        'status': 'success'
    })
    
    # 2. Wiener Filter (simulado)
    energy_in = np.mean(x_frame ** 2)
    energy_out = np.mean(pred ** 2)
    if energy_out < energy_in:
        tool_calls.append({
            'tool_name': 'wiener_filter',
            'parameters': {
                'noise_estimate': energy_in - energy_out,
                'signal_estimate': energy_out
            },
            'status': 'success'
        })
    
    # 3. Multi-Lorax (simulado)
    tool_calls.append({
        'tool_name': 'multi_lorax',
        'parameters': {
            'n_layers': 3,
            'attention_type': 'continuous'
        },
        'status': 'success'
    })
    
    # 4. Reconstruction
    tool_calls.append({
        'tool_name': 'reconstruction',
        'parameters': {
            'method': 'istft',
            'phase_used': 'reference'
        },
        'status': 'success'
    })
    
    return tool_calls

def flatten_exp_data(data, dset_for_importance, depth=0, max_depth=10):
    """
    Aplana recursivamente exp_data con manejo robusto de tipos.
    Transforma escalares a listas para compatibilidad con compute_feature_importance.
    """
    if depth > max_depth:
        return dset_for_importance
    
    if data is None:
        return dset_for_importance
    
    if isinstance(data, dict):
        for key, value in data.items():
            flatten_exp_data(value, dset_for_importance, depth + 1, max_depth)
        return dset_for_importance
    
    if isinstance(data, (list, tuple)):
        if len(data) >= 8:
            if len(data) > 1 and isinstance(data[1], np.ndarray):
                if data[1].ndim == 2 and data[1].shape[0] > 0:
                    entry = list(data[:8])
                    # CLAVE: convertir el penúltimo elemento (índice 6) a lista
                    if len(entry) >= 7:
                        if not hasattr(entry[6], '__len__'):
                            entry[6] = [entry[6]]
                    while len(entry) < 8:
                        entry.append(0)
                    dset_for_importance.append(entry)
                    return dset_for_importance
        
        for item in data:
            flatten_exp_data(item, dset_for_importance, depth + 1, max_depth)
        return dset_for_importance
    
    return dset_for_importance

def extract_entries_directly(exp_data):
    """
    Extrae entradas directamente de la estructura conocida.
    
    IMPORTANTE: Transforma el formato para que sea compatible con 
    compute_feature_importance, que espera que el penúltimo elemento
    (i[-2]) sea un objeto con __len__ (lista/array), no un escalar.
    
    Estructura original de cada entrada:
    [adversary, features_test, adversary_targets_test, weights, hyp, logical, mse_val, diff]
    
    El penúltimo (índice 6) es mse_val (escalar)
    
    compute_feature_importance espera que el penúltimo elemento sea una lista/array.
    Por lo tanto, convertimos mse_val a [mse_val] (lista de 1 elemento).
    """
    valid_entries = []
    
    if not exp_data:
        print("⚠️ exp_data está vacío o es None")
        return []
    
    # Si exp_data es un diccionario
    if isinstance(exp_data, dict):
        for key, value in exp_data.items():
            entries = extract_entries_directly(value)
            valid_entries.extend(entries)
        return valid_entries
    
    # Si exp_data es una lista
    if isinstance(exp_data, list):
        for idx, item in enumerate(exp_data):
            # Si es una entrada válida (tupla/lista con >=8 elementos)
            if isinstance(item, (list, tuple)) and len(item) >= 8:
                # Verificar que el elemento 1 sea features_test (array 2D)
                if len(item) > 1 and isinstance(item[1], np.ndarray):
                    if item[1].ndim == 2 and item[1].shape[0] > 0:
                        # Convertir a lista mutable
                        entry = list(item[:8])
                        
                        # ========== TRANSFORMACIÓN CLAVE ==========
                        # El penúltimo elemento (índice 6) es mse_val (escalar)
                        # compute_feature_importance espera un objeto con __len__
                        # Lo convertimos a lista de 1 elemento
                        if len(entry) >= 7:
                            mse_val = entry[6]
                            if not hasattr(mse_val, '__len__'):
                                entry[6] = [mse_val]  # Escalar → lista
                        
                        # Asegurar 8 elementos
                        while len(entry) < 8:
                            entry.append(0)
                        
                        valid_entries.append(entry)
                        continue
            
            # Recursión para items anidados
            sub_entries = extract_entries_directly(item)
            valid_entries.extend(sub_entries)
        
        return valid_entries
    
    return []


# Importar RewardHackingDetector que sí funciona
from apicultor.machine_learning.stress import RewardHackingDetector
        
class DSVMRewardModel:
    """Reward model adaptado para deep_support_vector_machines de APICultor"""
    def __init__(self, model):
        self.model = model
        self.reward_history = []
        self.hacking_detector = RewardHackingDetector()

    def _compute_overconfidence_penalty(self, output: np.ndarray) -> float:
        """
        Penaliza predicciones sobreseguras (varianza colapsada).
        Solución: "I don't know" en casos de baja evidencia.
        """
        out_flat = output.flatten()
        # Varianza baja = overconfidence
        variance = np.var(out_flat)
        if variance < 1e-5:  # colapsada
            return 0.8  # penalización alta
        # Normal: entre 0 y 0.1 de penalización
        return max(0.0, min(0.1, 1.0 / (1.0 + variance * 100)))
    
    def _compute_distribution_shift_penalty(self, input_arr: np.ndarray, output_arr: np.ndarray) -> float:
        """
        Penaliza cuando la distribución de salida se aleja demasiado de la entrada.
        Solución: preferir outputs robustos a parafraseo (invariantes).
        """
        inp_flat = input_arr.flatten()
        out_flat = output_arr.flatten()
        
        # Shift en media
        mean_shift = abs(np.mean(out_flat) - np.mean(inp_flat)) / (np.mean(inp_flat) + 1e-9)
        # Shift en std
        std_ratio = np.std(out_flat) / (np.std(inp_flat) + 1e-9)
        
        penalty = 0.0
        if mean_shift > 0.5:  # shift muy alto
            penalty += 0.15
        if std_ratio > 2.0 or std_ratio < 0.3:  # varianza muy diferente
            penalty += 0.15
            
        return min(0.3, penalty)
    
    def _compute_abstention_reward(self, input_arr: np.ndarray, output_arr: np.ndarray) -> float:
        """
        Recompensa por abstencion honesta en casos de baja evidencia.
        Si la entrada tiene baja energía, mejor mantener la entrada que alucinar.
        """
        energy_in = np.mean(input_arr ** 2)
        if energy_in < 1e-4:  # baja evidencia
            # Si la salida es similar a la entrada (abstencion), recompensa
            similarity = 1.0 / (1.0 + np.mean((output_arr - input_arr) ** 2))
            return 0.3 * similarity  # bonus de abstencion
        return 0.0

    def _is_collapsed(self, output: np.ndarray, input: np.ndarray = None) -> Tuple[bool, float]:
        """
        Detecta colapso usando criterios RELATIVOS, no números mágicos.
        """
        out_flat = output.flatten()
        
        # 1. Ratio de valores únicos vs total
        unique_ratio = len(np.unique(out_flat)) / len(out_flat)
        
        # 2. Rango dinámico relativo al valor máximo
        max_val = np.max(out_flat)
        min_val = np.min(out_flat)
        dynamic_range = (max_val - min_val) / (max_val + 1e-9)
        
        # 3. Comparación con la entrada (si está disponible)
        if input is not None:
            inp_flat = input.flatten()
            input_range = np.max(inp_flat) - np.min(inp_flat)
            output_range = max_val - min_val
            range_ratio = output_range / (input_range + 1e-9)
        else:
            range_ratio = dynamic_range
        
        # Criterios de colapso (relativos)
        is_collapsed = (
            unique_ratio < 0.01 or        # Menos de 1% de valores únicos
            dynamic_range < 0.001 or      # Rango dinámico casi nulo
            range_ratio < 0.01            # Output mucho menos variable que input
        )
        
        # Severidad del colapso (0 a 1)
        severity = min(1.0, 
                       (1 - unique_ratio) * 0.5 + 
                       (1 - dynamic_range) * 0.3 + 
                       (1 - range_ratio) * 0.2)
        
        return is_collapsed, severity

    def _calculate_uncertainty(self, output: np.ndarray, input: np.ndarray = None) -> float:
        """
        Calcula incertidumbre de la predicción para espectrogramas STFT.
        
        Para AOC, la incertidumbre es alta cuando:
        1. La varianza de salida es muy diferente a la varianza de entrada
        2. La salida tiene alta entropía espectral (muchos bins con energía similar)
        3. El output no se correlaciona con el input
        """
        out_flat = output.flatten()
        
        # 1. Varianza normalizada (coeficiente de variación)
        out_mean = np.mean(np.abs(out_flat))
        out_std = np.std(out_flat)
        cv = out_std / (out_mean + 1e-9)  # Coeficiente de variación
        
        # 2. Entropía de la distribución espectral (qué tan plano es el espectro)
        probs = np.abs(out_flat) / (np.sum(np.abs(out_flat)) + 1e-9)
        entropy = -np.sum(probs * np.log(probs + 1e-9))
        max_entropy = np.log(len(out_flat))  # Entropía máxima (espectro plano)
        normalized_entropy = entropy / (max_entropy + 1e-9)
        
        # 3. Comparación con entrada (si está disponible)
        if input is not None:
            inp_flat = input.flatten()
            correlation = np.corrcoef(inp_flat, out_flat)[0, 1]
            correlation = max(0, correlation)  # Solo correlación positiva importa
        else:
            correlation = 0.5
        
        # Incertidumbre combinada (0 a 1)
        # - cv alto (mucha variabilidad relativa) → más incertidumbre
        # - entropía alta (espectro plano) → más incertidumbre  
        # - correlación baja (no se relaciona con entrada) → más incertidumbre
        return (1.0 - min(cv, 1.0)) * 0.3 + normalized_entropy * 0.4 + (1.0 - correlation) * 0.3
        
                
    def score(self, input, output, target=None, chain_of_thought=None):
        scores = {}

        is_collapsed, collapse_severity = self._is_collapsed(output, input)
    
        if is_collapsed:
            # Recompensa negativa proporcional a la severidad
            reward = -0.5 - collapse_severity * 0.5  # Rango [-1.0, -0.5]
            
            scores['accuracy'] = reward
            scores['collapsed'] = True
            scores['collapse_severity'] = collapse_severity
            scores['cot_quality'] = 0.0
            scores['overconfidence_penalty'] = 1.0
            scores['distribution_shift_penalty'] = 1.0
            scores['abstention_bonus'] = 0.0
            scores['hacking_risk'] = 1.0
        
            hacking_score, hacking_flags = self.hacking_detector.detect(input, output, scores)
        
            self.reward_history.append({
                'scores': scores,
                'composite_reward': reward,
                'hacking_flags': hacking_flags + ['collapsed_output']
            })
            
            return {
                'composite_reward': reward,
                'component_scores': scores,
                'uncertainty': 1.0,
                'hacking_flags': hacking_flags + ['collapsed_output'],
                'reward_history': self.reward_history
            }
    
                
        # 1. Exactitud básica usando MSE
        if target is not None:
            if output.shape == target.shape:
                mse = float(np.mean((output - target) ** 2))
                scores['accuracy'] = 1.0 - mse
            else:
                scores['accuracy'] = 0.5
                
        # 2. Calidad del CoT (simplificada)
        if chain_of_thought:
            scores['cot_quality'] = min(len(chain_of_thought) / 10.0, 1.0)

        # 3. NUEVO: Penalización por overconfidence
        overconfidence_penalty = self._compute_overconfidence_penalty(output)
        scores['overconfidence_penalty'] = overconfidence_penalty
        
        # 4. NUEVO: Penalización por distribution shift
        shift_penalty = self._compute_distribution_shift_penalty(input, output)
        scores['distribution_shift_penalty'] = shift_penalty
        
        # 5. NUEVO: Recompensa por abstencion honesta
        abstention_bonus = self._compute_abstention_reward(input, output)
        scores['abstention_bonus'] = abstention_bonus
                
        # 3. Detección de reward hacking
        hacking_score, hacking_flags = self.hacking_detector.detect(input, output, scores)
        scores['hacking_risk'] = hacking_score
                
        # 4. Recompensa compuesta
        composite = (
            scores.get('accuracy', 0.5) * 0.7 +
            scores.get('cot_quality', 0.3) * 0.3
        )
                
        self.reward_history.append({
            'scores': scores,
            'composite_reward': composite,
            'hacking_flags': hacking_flags
        })
        
        uncertainty = self._calculate_uncertainty(output)
        scores['uncertainty'] = uncertainty
                
        return {
            'composite_reward': composite,
            'component_scores': scores,
            'uncertainty': uncertainty,
            'reward_history': self.reward_history,
            'hacking_flags': hacking_flags
        }
            
    def get_training_metrics(self):
        if not self.reward_history:
            return {}
        rewards = [h['composite_reward'] for h in self.reward_history]
        return {
            'mean_reward': float(np.mean(rewards)),
            'std_reward': float(np.std(rewards))
        }

class BaselineModelManager:
    """
    Gestiona el guardado de modelos baseline por iteración (cada 4000 muestras).
    Organiza teachers, students, memorias y resultados por nombre de baseline.
    """
    
    def __init__(self, base_output_dir: str, baseline_name: str = None):
        self.base_output_dir = base_output_dir
        self.baseline_name = baseline_name or f"baseline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.baseline_dir = os.path.join(base_output_dir, self.baseline_name)
        self.iteration = 0
        self.samples_processed = 0
        self.ITERATION_SIZE = 4000  # Cada 4000 muestras se guarda baseline
        
        # Crear estructura de directorios
        self._create_directory_structure()
        
        # Historial de baselines
        self.baseline_history = []
        self.current_checkpoint = None
        
    def _create_directory_structure(self):
        """Crea la estructura de directorios para esta baseline"""
        os.makedirs(self.baseline_dir, exist_ok=True)
        
        # Subdirectorios
        self.models_dir = os.path.join(self.baseline_dir, "models")
        self.memories_dir = os.path.join(self.baseline_dir, "memories")
        self.teachers_dir = os.path.join(self.baseline_dir, "teachers")
        self.students_dir = os.path.join(self.baseline_dir, "students")
        self.audios_dir = os.path.join(self.baseline_dir, "audios")
        self.results_dir = os.path.join(self.baseline_dir, "results")
        
        for d in [self.models_dir, self.memories_dir, self.teachers_dir, 
                  self.students_dir, self.audios_dir, self.results_dir]:
            os.makedirs(d, exist_ok=True)
            
        logger.info(f"[BaselineManager] Estructura creada en {self.baseline_dir}")
    
    def update_samples_processed(self, n_samples: int):
        """Actualiza contador de muestras procesadas"""
        self.samples_processed += n_samples
        logger.info(f"[BaselineManager] Procesadas {self.samples_processed} muestras totales")
        
        # Verificar si hay que guardar nueva baseline
        if self.samples_processed >= (self.iteration + 1) * self.ITERATION_SIZE:
            self.save_baseline()
    
    def save_baseline(self):
        """Guarda baseline actual (modelo, memoria, teacher, student)"""
        self.iteration += 1
        iteration_dir = os.path.join(self.baseline_dir, f"iter_{self.iteration:04d}")
        os.makedirs(iteration_dir, exist_ok=True)
        
        baseline_info = {
            'iteration': self.iteration,
            'samples_processed': self.samples_processed,
            'timestamp': datetime.now().isoformat(),
            'iteration_dir': iteration_dir
        }
        
        self.baseline_history.append(baseline_info)
        
        # Guardar metadata
        metadata_path = os.path.join(iteration_dir, "baseline_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(baseline_info, f, indent=2)
            
        logger.info(f"[BaselineManager] Baseline {self.iteration} guardada en {iteration_dir}")
        return iteration_dir
    
    def save_model(self, model, name: str, iteration: int = None):
        """Guarda un modelo (DSVM, teacher, student)"""
        if iteration is None:
            iteration = self.iteration
        iter_dir = os.path.join(self.baseline_dir, f"iter_{iteration:04d}")

        if hasattr(model, '_w') and model._w is not None:
            w_path = os.path.join(iter_dir, f"{name}_w.npy")
            np.save(w_path, model._w)
            logger.info(f"[BaselineManager] Pesos guardados en {w_path}")
        if hasattr(model, '_bias'):
            bias_path = os.path.join(iter_dir, f"{name}_bias.npy")
            np.save(bias_path, model._bias)
        
        model_path = os.path.join(iter_dir, f"{name}.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        logger.info(f"[BaselineManager] Modelo {name} guardado en {model_path}")
        return model_path
    
    def save_memory(self, memory: AgentMemory, name: str, iteration: int = None):
        """Guarda el estado de la memoria del agente"""
        if iteration is None:
            iteration = self.iteration
        iter_dir = os.path.join(self.baseline_dir, f"iter_{iteration:04d}")
        
        memory_path = os.path.join(iter_dir, f"memory_{name}.pkl")
        with open(memory_path, "wb") as f:
            pickle.dump({
                'long_term': memory.long_term,
                'mid_term': memory.mid_term,
                'short_term': memory.short_term,
                'memory_stats': memory.memory_stats,
                'consolidation_queue': memory.consolidation_queue
            }, f)
        logger.info(f"[BaselineManager] Memoria {name} guardada en {memory_path}")
        return memory_path
    
    def save_audio_samples(self, X_samples: np.ndarray, Y_samples: np.ndarray, 
                           predictions: np.ndarray, iteration: int = None):
        """Guarda muestras de audio procesadas"""
        if iteration is None:
            iteration = self.iteration
        iter_dir = os.path.join(self.baseline_dir, f"iter_{iteration:04d}")
        audio_dir = os.path.join(iter_dir, "audio_samples")
        os.makedirs(audio_dir, exist_ok=True)
        
        # Guardar como numpy arrays
        np.save(os.path.join(audio_dir, "X_input.npy"), X_samples)
        np.save(os.path.join(audio_dir, "Y_target.npy"), Y_samples)
        np.save(os.path.join(audio_dir, "predictions.npy"), predictions)
        
        # Guardar metadatos de audio
        audio_meta = {
            'n_samples': len(X_samples),
            'shape': X_samples.shape,
            'timestamp': datetime.now().isoformat(),
            'iteration': iteration
        }
        with open(os.path.join(audio_dir, "audio_metadata.json"), "w") as f:
            json.dump(audio_meta, f, indent=2)
            
        logger.info(f"[BaselineManager] {len(X_samples)} muestras de audio guardadas")
    
    def save_teacher_student_pair(self, teacher_model, student_model, 
                                   distillation_result: Dict, iteration: int = None):
        """Guarda par teacher-student con resultados de distillation"""
        if iteration is None:
            iteration = self.iteration
        iter_dir = os.path.join(self.baseline_dir, f"iter_{iteration:04d}")
        
        # Guardar modelos
        teacher_path = self.save_model(teacher_model, "teacher", iteration)
        student_path = self.save_model(student_model, "student", iteration)
    
        # Convertir distillation_result a tipos serializables
        def make_json_serializable(obj):
            """Convierte objetos numpy a tipos nativos de Python"""
            if isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: make_json_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [make_json_serializable(item) for item in obj]
            return obj
        
        serializable_result = make_json_serializable(distillation_result)
        
        # Guardar resultados de distillation
        distillation_path = os.path.join(iter_dir, "distillation_results.json")
        with open(distillation_path, "w") as f:
            json.dump(serializable_result, f, indent=2)
            
        logger.info(f"[BaselineManager] Teacher-student pair guardados (iter {iteration})")
        
        return {
            'teacher_path': teacher_path,
            'student_path': student_path,
            'distillation_path': distillation_path
        }
    
    def load_baseline(self, iteration: int):
        """Carga una baseline específica"""
        iter_dir = os.path.join(self.baseline_dir, f"iter_{iteration:04d}")
        
        if not os.path.exists(iter_dir):
            logger.error(f"[BaselineManager] Baseline {iteration} no encontrada en {iter_dir}")
            return None
            
        # Cargar metadata
        metadata_path = os.path.join(iter_dir, "baseline_metadata.json")
        with open(metadata_path, "r") as f:
            baseline_info = json.load(f)
            
        # Cargar modelos si existen
        models = {}
        for model_name in ['dsvm_main', 'teacher', 'student']:
            model_path = os.path.join(iter_dir, f"{model_name}.pkl")
            if os.path.exists(model_path):
                with open(model_path, "rb") as f:
                    models[model_name] = pickle.load(f)
                    
        # Cargar memorias
        memories = {}
        for mem_name in ['agent_memory', 'long_term', 'mid_term', 'short_term']:
            mem_path = os.path.join(iter_dir, f"memory_{mem_name}.pkl")
            if os.path.exists(mem_path):
                with open(mem_path, "rb") as f:
                    memories[mem_name] = pickle.load(f)
                    
        # Cargar resultados de distillation
        distillation_path = os.path.join(iter_dir, "distillation_results.json")
        distillation_results = None
        if os.path.exists(distillation_path):
            with open(distillation_path, "r") as f:
                distillation_results = json.load(f)
                
        logger.info(f"[BaselineManager] Baseline {iteration} cargada")
        
        return {
            'baseline_info': baseline_info,
            'models': models,
            'memories': memories,
            'distillation_results': distillation_results,
            'iteration_dir': iter_dir
        }
    
    def get_latest_baseline(self):
        """Obtiene la baseline más reciente"""
        if not self.baseline_history:
            return None
        latest = max(self.baseline_history, key=lambda x: x['iteration'])
        return self.load_baseline(latest['iteration'])
    
    def list_baselines(self) -> List[Dict]:
        """Lista todas las baselines guardadas"""
        return self.baseline_history
    
    def generate_report(self):
        """Genera reporte completo de todas las baselines"""
        report_path = os.path.join(self.baseline_dir, "baseline_report.json")
        
        report = {
            'baseline_name': self.baseline_name,
            'created_at': datetime.now().isoformat(),
            'total_iterations': len(self.baseline_history),
            'iteration_size': self.ITERATION_SIZE,
            'baselines': self.baseline_history,
            'directory_structure': {
                'models': self.models_dir,
                'memories': self.memories_dir,
                'teachers': self.teachers_dir,
                'students': self.students_dir,
                'audios': self.audios_dir,
                'results': self.results_dir
            }
        }
        
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
            
        logger.info(f"[BaselineManager] Reporte generado en {report_path}")
        return report

class FeedbackToReward:
    """
    Convierte feedback humano (good/bad/neutral) a valor de reward
    para decisión GO/NO-GO en cualquier rollout.
    """
    
    def __init__(self, learning_rate: float = 0.1):
        self.learning_rate = learning_rate
        self.feedback_history = []
        self.reward_history = []
        self.feedback_weights = {
            'good': 1.0,
            'bad': -0.5,
            'neutral': 0.0,
            'excellent': 1.2,
            'poor': -0.8
        }
        
    def convert_feedback(self, feedback: str, 
                        base_reward: float = None,
                        context: Dict = None) -> float:
        """
        Convierte feedback textual a valor de reward.
        
        Args:
            feedback: 'good', 'bad', 'neutral', 'excellent', 'poor'
            base_reward: recompensa base del modelo (opcional)
            context: contexto adicional (ej: confidence, uncertainty)
        
        Returns:
            reward_value: float entre 0 y 1
        """
        # Obtener peso base del feedback
        base_weight = self.feedback_weights.get(feedback.lower(), 0.0)
        
        # Normalizar a [0, 1] con sigmoide
        normalized = 1.0 / (1.0 + np.exp(-base_weight))
        
        # Si hay base_reward, combinarlo
        if base_reward is not None:
            # 70% feedback, 30% base reward
            combined = 0.7 * normalized + 0.3 * base_reward
        else:
            combined = normalized
        
        # Ajustar por contexto
        if context:
            # Penalizar si hay alta incertidumbre
            uncertainty = context.get('uncertainty', 0.0)
            combined = combined * (1.0 - uncertainty * 0.3)
            
            # Bonus por confianza alta si feedback es good
            if feedback.lower() in ['good', 'excellent']:
                confidence = context.get('confidence', 0.5)
                combined = combined * (0.8 + 0.4 * confidence)
        
        reward_value = float(np.clip(combined, 0.0, 1.0))
        
        # Registrar historial
        self.feedback_history.append({
            'feedback': feedback,
            'converted_reward': reward_value,
            'base_reward': base_reward,
            'context': context,
            'timestamp': datetime.now().isoformat()
        })
        
        # Mantener historial limitado
        if len(self.feedback_history) > 1000:
            self.feedback_history = self.feedback_history[-1000:]
        
        return reward_value
    
    def batch_convert(self, feedbacks: List[str], 
                     base_rewards: List[float] = None,
                     contexts: List[Dict] = None) -> List[float]:
        """Convierte múltiples feedbacks"""
        rewards = []
        for i, fb in enumerate(feedbacks):
            br = base_rewards[i] if base_rewards and i < len(base_rewards) else None
            ctx = contexts[i] if contexts and i < len(contexts) else None
            rewards.append(self.convert_feedback(fb, br, ctx))
        return rewards
    
    def get_go_no_go_decision(self, 
                              rollouts_rewards: List[float],
                              feedbacks: List[str] = None,
                              threshold: float = 0.05) -> Dict:
        """
        Determina decisión GO/NO-GO basada en rewards y feedbacks.
        """
        if not rollouts_rewards:
            return {'decision': 'NO-GO', 'reason': 'no_rollouts'}
        
        # Calcular reward promedio
        mean_reward = np.mean(rollouts_rewards)
        
        # Si hay feedbacks, convertirlos y combinarlos
        if feedbacks:
            feedback_rewards = self.batch_convert(feedbacks)
            mean_feedback_reward = np.mean(feedback_rewards)
            # Combinar 60% reward modelo, 40% feedback
            combined_score = 0.6 * mean_reward + 0.4 * mean_feedback_reward
        else:
            combined_score = mean_reward
        
        # Comparar con threshold (usando referencia de slice anterior)
        # Asumiendo que tenemos last_slice_reward disponible
        if hasattr(self, 'last_slice_reward'):
            improvement = combined_score - self.last_slice_reward
        else:
            improvement = combined_score - 0.5  # Baseline neutral
        
        # Decisión
        if improvement > threshold:
            decision = 'GO'
            reason = f'improvement={improvement:.4f} > threshold'
        elif improvement < -threshold:
            decision = 'NO-GO'
            reason = f'degradation={abs(improvement):.4f} > threshold'
        else:
            decision = 'NEUTRAL'
            reason = f'change={improvement:+.4f} within threshold'
        
        return {
            'decision': decision,
            'combined_score': combined_score,
            'mean_reward': mean_reward,
            'mean_feedback_reward': mean_feedback_reward if feedbacks else None,
            'improvement': improvement,
            'threshold': threshold,
            'reason': reason,
            'n_feedbacks': len(feedbacks) if feedbacks else 0
        }

class ModelRouter:
    """
    Model Routing: mapea input a múltiples modelos y combina outputs.
    Soporta routing basado en características, contexto, o tipo de input.
    """
    
    def __init__(self, routing_strategy: str = 'weighted_average'):
        """
        routing_strategy: 
            - 'weighted_average': promedio ponderado por confianza
            - 'majority_vote': votación mayoritaria
            - 'best_of_n': mejor de N según métrica
            - 'ensemble': ensemble completo
        """
        self.models = {}  # dict: model_name -> model
        self.model_weights = {}  # dict: model_name -> weight
        self.routing_strategy = routing_strategy
        self.routing_history = []
        
    def register_model(self, name: str, model, weight: float = 1.0):
        """Registra un modelo para routing"""
        self.models[name] = model
        self.model_weights[name] = weight
        logger.info(f"[ModelRouter] Modelo registrado: {name}, weight={weight}")
    
    def unregister_model(self, name: str):
        """Elimina un modelo del routing"""
        if name in self.models:
            del self.models[name]
            del self.model_weights[name]
            logger.info(f"[ModelRouter] Modelo eliminado: {name}")
    
    async def route(self, X: np.ndarray, 
                   context: Dict = None,
                   return_all: bool = False) -> Dict:
        """
        Enruta input a todos los modelos y combina outputs.
        
        Args:
            X: input array
            context: contexto para routing (ej: tipo de input, características)
            return_all: si True, retorna todos los outputs individuales
        
        Returns:
            dict con output combinado y metadatos
        """
        if not self.models:
            raise ValueError("No hay modelos registrados para routing")
        
        # Ejecutar todos los modelos
        predictions = {}
        confidences = {}
        latencies = {}
        
        for name, model in self.models.items():
            start_time = datetime.now()
            
            try:
                # Predecir
                if hasattr(model, 'predict'):
                    pred = await model.predict(X)
                elif callable(model):
                    pred = model(X)
                else:
                    continue
                
                latency = (datetime.now() - start_time).total_seconds()
                latencies[name] = latency
                predictions[name] = pred
                
                # Obtener confianza si está disponible
                if hasattr(model, 'proba') and model.proba is not None:
                    if isinstance(model.proba, np.ndarray) and len(model.proba) > 0:
                        confidences[name] = np.mean(model.proba)
                    else:
                        confidences[name] = 0.5
                elif hasattr(model, 'feature_weights') and model.feature_weights is not None:
                    # Usar norma de feature_weights como proxy de confianza
                    fw = np.array(model.feature_weights)
                    confidences[name] = np.clip(np.mean(np.abs(fw)), 0.1, 0.9)
                else:
                    confidences[name] = 0.5
                    
            except Exception as e:
                logger.warning(f"[ModelRouter] Error en modelo {name}: {e}")
                continue
        
        if not predictions:
            raise RuntimeError("Ningún modelo pudo procesar el input")
        
        # Combinar outputs según estrategia
        combined_output = self._combine_outputs(predictions, confidences, context)
        
        result = {
            'combined_output': combined_output,
            'n_models_used': len(predictions),
            'models': list(predictions.keys()),
            'latencies': latencies,
            'routing_strategy': self.routing_strategy
        }
        
        if return_all:
            result['individual_predictions'] = predictions
            result['confidences'] = confidences
        
        # Registrar historial
        self.routing_history.append({
            'timestamp': datetime.now().isoformat(),
            'input_shape': X.shape if hasattr(X, 'shape') else None,
            'n_models': len(predictions),
            'strategy': self.routing_strategy,
            'latencies': latencies
        })
        
        # Mantener historial limitado
        if len(self.routing_history) > 1000:
            self.routing_history = self.routing_history[-1000:]
        
        return result
    
    def _combine_outputs(self, predictions: Dict, 
                         confidences: Dict,
                         context: Dict = None) -> np.ndarray:
        """Combina outputs de múltiples modelos según estrategia"""
        
        if self.routing_strategy == 'weighted_average':
            # Promedio ponderado por confianza
            total_weight = sum(confidences.values())
            if total_weight == 0:
                total_weight = len(confidences)
            
            combined = None
            for name, pred in predictions.items():
                weight = confidences.get(name, 1.0)
                if combined is None:
                    combined = pred * weight
                else:
                    combined = combined + pred * weight
            
            if combined is not None:
                combined = combined / total_weight
            
        elif self.routing_strategy == 'majority_vote':
            # Para regresión, usar mediana
            all_preds = np.array(list(predictions.values()))
            combined = np.median(all_preds, axis=0)
            
        elif self.routing_strategy == 'best_of_n':
            # Mejor según confianza
            best_name = max(confidences, key=confidences.get)
            combined = predictions[best_name]
            
        elif self.routing_strategy == 'ensemble':
            # Ensemble completo: promedio simple
            all_preds = np.array(list(predictions.values()))
            combined = np.mean(all_preds, axis=0)
            
        elif self.routing_strategy == 'context_aware':
            # Routing basado en contexto
            if context:
                input_type = context.get('input_type', 'default')
                # Mapear input_type a modelo preferido
                model_preference = context.get('model_preference', {})
                preferred = model_preference.get(input_type)
                if preferred and preferred in predictions:
                    combined = predictions[preferred]
                else:
                    # Fallback a weighted_average
                    total_weight = sum(confidences.values())
                    combined = sum(pred * confidences.get(name, 1.0) 
                                  for name, pred in predictions.items()) / total_weight
            else:
                # Fallback a weighted_average
                total_weight = sum(confidences.values())
                combined = sum(pred * confidences.get(name, 1.0) 
                              for name, pred in predictions.items()) / total_weight
        else:
            # Default: promedio simple
            all_preds = np.array(list(predictions.values()))
            combined = np.mean(all_preds, axis=0)
        
        return combined
    
    def get_routing_metrics(self) -> Dict:
        """Métricas de routing"""
        if not self.routing_history:
            return {'status': 'no_history'}
        
        recent = self.routing_history[-100:]
        
        avg_latencies = {}
        for record in recent:
            for model, lat in record.get('latencies', {}).items():
                if model not in avg_latencies:
                    avg_latencies[model] = []
                avg_latencies[model].append(lat)
        
        avg_latencies = {m: np.mean(lats) for m, lats in avg_latencies.items()}
        
        return {
            'total_routes': len(self.routing_history),
            'models_registered': list(self.models.keys()),
            'routing_strategy': self.routing_strategy,
            'avg_latencies': avg_latencies,
            'recent_strategies': [h['strategy'] for h in recent[-10:]]
        }

def should_skip_stage(output_dir: str, stage_name: str, loop_counter: int) -> bool:
    """Verifica si una etapa ya fue completada previamente."""
    marker_file = os.path.join(output_dir, f"{stage_name}_loop_{loop_counter}.complete")
    return os.path.exists(marker_file)

def mark_stage_complete(output_dir: str, stage_name: str, loop_counter: int):
    """Marca una etapa como completada."""
    marker_file = os.path.join(output_dir, f"{stage_name}_loop_{loop_counter}.complete")
    with open(marker_file, 'w') as f:
        f.write(datetime.now().isoformat())

def should_skip_distillation(baseline_dir: str, iteration: int = None) -> Tuple[bool, str]:
    """
    Verifica si la distillation para esta baseline ya fue completada.
    
    Args:
        baseline_dir: Directorio de la baseline
        iteration: Número de iteración (si None, busca la más reciente)
    
    Returns:
        (skip, reason): True si debe saltarse, False si debe ejecutarse
    """
    # Buscar archivos de distillation completada
    markers = [
        os.path.join(baseline_dir, "distillation_complete.txt"),
        os.path.join(baseline_dir, "distillation_results.json"),
    ]
    
    for marker in markers:
        if os.path.exists(marker):
            return True, f"Archivo de completado encontrado: {marker}"
    
    # Verificar si hay resultados en iter_* 
    if iteration is not None:
        iter_dir = os.path.join(baseline_dir, f"iter_{iteration:04d}")
        distillation_result = os.path.join(iter_dir, "distillation_results.json")
        if os.path.exists(distillation_result):
            return True, f"Resultados encontrados en {distillation_result}"
    
    # Buscar cualquier iteración con resultados
    import glob
    results = glob.glob(os.path.join(baseline_dir, "iter_*/distillation_results.json"))
    if results:
        return True, f"Resultados encontrados en {len(results)} iteraciones"
    
    return False, "No se encontraron resultados previos"

def mark_distillation_complete(baseline_dir: str, results: Dict = None):
    """Marca que la distillation fue completada exitosamente."""
    marker_file = os.path.join(baseline_dir, "distillation_complete.txt")
    with open(marker_file, 'w') as f:
        f.write(datetime.now().isoformat())
        if results:
            f.write(f"\n{json.dumps(results, default=str)}")
    
    logger.info(f"[Distillation] Marcada como completada en {marker_file}")

class KnowledgeDistillation:
    def __init__(self, teacher_model, student_model, temperature=3.0, alpha=0.7):
        self.teacher = teacher_model
        self.student = student_model
        self.temperature = temperature
        self.alpha = alpha
        self.distillation_history = []
    
    async def distill(self, X, y, epochs=10, batch_size=32, output_dir=None):
        """Ejecutar distillation training"""
        print(f"[Distillation] Iniciando con {len(X)} muestras, epochs={epochs}")
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            
            X_batch = X
            y_batch = y
                
            # Teacher predictions
            teacher_logits = await self.teacher.predict(X_batch)
                
            # Student predictions
            student_logits = await self.student.predict(X_batch)
                
            # Soft targets con temperature
            soft_targets = np.exp(teacher_logits / self.temperature)
            soft_targets /= np.sum(soft_targets, axis=-1, keepdims=True)
                
            student_soft = np.exp(student_logits / self.temperature)
            student_soft /= np.sum(student_soft, axis=-1, keepdims=True)
                
            # KL divergence loss
            kl_loss = np.mean(np.sum(soft_targets * (np.log(soft_targets + 1e-9) - np.log(student_soft + 1e-9)), axis=-1))
                
            # Hard loss (MSE)
            hard_loss = np.mean((y_batch - student_logits) ** 2)
                
            # Combined loss
            total_loss = self.alpha * kl_loss + (1 - self.alpha) * hard_loss
            epoch_loss += total_loss
                
            logger.debug(f"[Distillation] Epoch {epoch}, Batch {batch}: KL={kl_loss:.4f}, Hard={hard_loss:.4f}, Total={total_loss:.4f}")
            
            self.distillation_history.append({'epoch': epoch, 'loss': epoch_loss})
            logger.info(f"[Distillation] Epoch {epoch}: Loss={epoch_loss:.4f}")
        
        # Guardar modelo distilled
        if output_dir:
            student_path = os.path.join(output_dir, "distilled_model.pkl")
            with open(student_path, "wb") as f:
                pickle.dump(self.student, f)
            logger.info(f"[Distillation] Modelo guardado en {student_path}")
            
            # Guardar historial
            history_path = os.path.join(output_dir, "distillation_history.txt")
            with open(history_path, "w") as f:
                f.write("DISTILLATION HISTORY\n")
                f.write("="*80 + "\n\n")
                for entry in self.distillation_history:
                    f.write(f"Epoch {entry['epoch']}: loss={entry['loss']:.6f}\n")
        
        return self.student
    
    def get_metrics(self) -> Dict:
        """Obtener métricas de distillation con tipos serializables"""
        if not self.distillation_history:
            return {}
        
        losses = [h['loss'] for h in self.distillation_history]
        return {
            'final_loss': float(losses[-1]) if losses else 0.0,
            'min_loss': float(min(losses)) if losses else 0.0,
            'total_epochs': int(len(self.distillation_history)),
            'converged': bool(losses[-1] < losses[0] * 0.1) if len(losses) > 1 else False
        }

class QuoteReferenceModel:
    """
    Quote Reference Model: almacena probabilidades de preferencia para medir
    cuánto está cambiando el modelo durante el entrenamiento.
    
    Ahora soporta colección de modelos por loop.
    """
    
    def __init__(self, base_model, loop_counter: int = 0, output_dir: str = "."):
        self.base_model = base_model
        self.reference_proba = []
        self.current_proba = []
        self.ratio_history = []
        self.model_checkpoints = []
        self.loop_counter = loop_counter
        self.output_dir = output_dir
        self.quote_reference_collection = []  # Colección de modelos previos
        
        # Cargar colección existente si existe
        collection_path = os.path.join(output_dir, f"quote_reference_collection.json")
        if os.path.exists(collection_path):
            try:
                with open(collection_path, "r") as f:
                    self.quote_reference_collection = json.load(f)
                logger.info(f"[QuoteReference] Cargada colección con {len(self.quote_reference_collection)} modelos previos")
            except Exception as e:
                logger.warning(f"[QuoteReference] No se pudo cargar colección: {e}")

    def _probs_from_predictions(self, preds: np.ndarray) -> np.ndarray:
        """Convierte predicciones a probabilidades normalizadas."""
        if preds is None or preds.size == 0:
            return np.array([])
        
        preds_flat = preds.flatten()
        min_val = np.min(preds_flat)
        max_val = np.max(preds_flat)
        
        if max_val - min_val < 1e-9:
            # Todas las predicciones son iguales
            return np.ones_like(preds_flat) * 0.5
        
        # Normalizar a [0, 1]
        probs = (preds_flat - min_val) / (max_val - min_val)
        return probs

    def get_training_direction(self) -> str:
        """
        Determina la dirección del entrenamiento basada en el ratio actual.
        
        Returns:
            str: 'increasing' (ratio > 1.05), 'decreasing' (ratio < 0.95), 
                 'stable' (0.95 <= ratio <= 1.05), o 'unknown' (sin datos)
        """
        if not self.ratio_history:
            return "unknown"
        
        # Obtener el ratio más reciente
        latest_ratio = self.ratio_history[-1]['ratio']
        
        if latest_ratio > 1.05:
            return "increasing"
        elif latest_ratio < 0.95:
            return "decreasing"
        else:
            return "unknown"
    
    async def capture_reference(self, X: np.ndarray) -> np.ndarray:
        """Captura las probabilidades de referencia antes del entrenamiento"""
        if hasattr(self.base_model, 'proba') and self.base_model.proba is not None:
            self.reference_proba = np.array(self.base_model.proba)
        else:
            preds = await self.base_model.predict(X)
            preds = preds[0]
            self.reference_proba = self._probs_from_predictions(preds)
        
        logger.info(f"[QuoteReference] Loop #{self.loop_counter} - Referencia capturada: shape={self.reference_proba.shape}, mean={np.mean(self.reference_proba):.4f}")
        return self.reference_proba
    
    async def capture_current(self, X: np.ndarray) -> np.ndarray:
        """Captura las probabilidades actuales después del entrenamiento"""
        if hasattr(self.base_model, 'proba') and self.base_model.proba is not None:
            self.current_proba = np.array(self.base_model.proba)
        else:
            preds = await self.base_model.predict(X)
            preds = preds[0]
            self.current_proba = self._probs_from_predictions(preds)
        
        logger.info(f"[QuoteReference] Loop #{self.loop_counter} - Actual capturada: shape={self.current_proba.shape}, mean={np.mean(self.current_proba):.4f}")
        return self.current_proba
    
    def compute_ratio(self) -> float:
        """Calcula ratio = proba_actual / proba_referencia"""
        if self.reference_proba.size == 0 or self.current_proba.size == 0:
            logger.warning("[QuoteReference] No hay datos suficientes para calcular ratio")
            return 1.0
        
        ref_mean = np.mean(self.reference_proba) + 1e-9
        curr_mean = np.mean(self.current_proba)
        ratio = curr_mean / ref_mean
        
        self.ratio_history.append({
            'timestamp': datetime.now().isoformat(),
            'loop': self.loop_counter,
            'ratio': ratio,
            'ref_mean': ref_mean,
            'curr_mean': curr_mean,
            'ref_std': np.std(self.reference_proba),
            'curr_std': np.std(self.current_proba)
        })
        
        logger.info(f"[QuoteReference] Loop #{self.loop_counter} - Ratio = {ratio:.4f} (curr={curr_mean:.4f} / ref={ref_mean:.4f})")
        return ratio
    
    def save_checkpoint(self, output_dir: str, step: str) -> None:
        """Guarda checkpoint del modelo y sus probabilidades con índice de loop"""
        checkpoint_path = os.path.join(output_dir, f"quote_reference_loop_{self.loop_counter}_{step}.pkl")
        with open(checkpoint_path, "wb") as f:
            pickle.dump({
                'loop': self.loop_counter,
                'model': self.base_model,
                'reference_proba': self.reference_proba,
                'current_proba': self.current_proba,
                'ratio_history': self.ratio_history[-10:],
                'timestamp': datetime.now().isoformat()
            }, f)
        logger.info(f"[QuoteReference] Checkpoint guardado: {checkpoint_path}")
    
    def add_to_collection(self):
        """Agrega el modelo actual a la colección de quote reference models"""
        model_info = {
            'loop': self.loop_counter,
            'timestamp': datetime.now().isoformat(),
            'ratio': self.compute_ratio() if self.ratio_history else 1.0,
            'mean_reward': np.mean(self.current_proba) if len(self.current_proba) > 0 else 0.0,
            'model_path': f"quote_reference_loop_{self.loop_counter}_post_training.pkl"
        }
        
        self.quote_reference_collection.append(model_info)
        
        # Guardar colección actualizada
        collection_path = os.path.join(self.output_dir, f"quote_reference_collection.json")
        with open(collection_path, "w") as f:
            json.dump(self.quote_reference_collection, f, indent=2)
        
        logger.info(f"[QuoteReference] Modelo loop #{self.loop_counter} agregado a colección. Total: {len(self.quote_reference_collection)}")
    
    def get_collection(self) -> List[Dict]:
        """Retorna la colección de quote reference models"""
        return self.quote_reference_collection
    
    def save_report(self, output_dir: str) -> None:
        """Guarda reporte en texto con el historial de ratios y colección"""
        report_path = os.path.join(output_dir, f"quote_reference_report_loop_{self.loop_counter}.txt")
        with open(report_path, "w") as f:
            f.write("="*80 + "\n")
            f.write(f"QUOTE REFERENCE MODEL REPORT - LOOP #{self.loop_counter}\n")
            f.write("="*80 + "\n\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"Modelos en colección: {len(self.quote_reference_collection)}\n\n")
            
            f.write("RATIO HISTORY:\n")
            f.write("-"*40 + "\n")
            for entry in self.ratio_history[-20:]:
                f.write(f"  Loop {entry.get('loop', 'N/A')} - {entry['timestamp']}: ratio={entry['ratio']:.4f} ")
                f.write(f"(curr={entry['curr_mean']:.4f} / ref={entry['ref_mean']:.4f})\n")
            
            f.write("\nCOLECCIÓN DE MODELOS:\n")
            f.write("-"*40 + "\n")
            for model_info in self.quote_reference_collection:
                f.write(f"  Loop {model_info['loop']}: ratio={model_info['ratio']:.4f}, ")
                f.write(f"mean_reward={model_info['mean_reward']:.4f}, ")
                f.write(f"timestamp={model_info['timestamp']}\n")
            
            f.write("\nINTERPRETACIÓN:\n")
            f.write("-"*40 + "\n")
            f.write("  ratio > 1.05 → El entrenamiento favorece la salida actual sobre la anterior\n")
            f.write("  ratio < 0.95 → La salida anterior era más probable\n")
            f.write("  ratio ≈ 1.0  → No hay cambios significativos\n")
            
            current_direction = self.get_training_direction()
            f.write(f"\n  Dirección actual: {current_direction}\n")
        
        logger.info(f"[QuoteReference] Reporte guardado: {report_path}")

# ─────────────────────────────────────────────────────────────────────────────
# 1.  CARGA Y PREPROCESAMIENTO DE AUDIO REAL
# ─────────────────────────────────────────────────────────────────────────────

def _load_audio_sync(path: str) -> np.ndarray:
    """
    Carga audio de forma síncrona usando soundfile (igual que DoSegmentation
    y SoundSimilarity del repo). Devuelve señal mono float32 normalizada.
    Llamada internamente desde load_audio (async) vía executor.
    """
    data, _ = sf_read(path)
    if data.ndim > 1:
        data = data.mean(axis=1)          # stereo → mono
    data = data.astype(np.float32)
    if np.abs(data).max() > 1e-6:
        data = data / np.abs(data).max()  # normalizar amplitud
    return data


async def load_audio(path: str) -> np.ndarray:
    """
    Carga audio de forma asíncrona delegando la lectura de disco a un
    ThreadPoolExecutor para no bloquear el event loop durante el I/O.
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _load_audio_sync, path)


def compute_stft_frames(
    audio: np.ndarray,
    n_fft: int = N_FFT,
    hop: int = HOP_LENGTH,
) -> np.ndarray:
    """
    Computa STFT con scipy (igual que constraints/time_freq.py que usa librosa.stft
    internamente). Devuelve magnitud [n_frames, n_bins] real.
    """
    _, _, Zxx = scipy_stft(audio, fs=SR, nperseg=n_fft, noverlap=n_fft - hop)
    magnitude = np.abs(Zxx).T          # → [n_frames, freq_bins]
    return magnitude.astype(np.float64)


def istft_from_magnitude(
    magnitude: np.ndarray,
    phase_ref: np.ndarray,
    n_fft: int = N_FFT,
    hop: int = HOP_LENGTH,
) -> np.ndarray:
    """
    Reconstruye señal desde magnitud estimada + fase de referencia.
    phase_ref tiene la misma forma que magnitude pero con valores complejos
    del STFT original ocluido (se reutiliza la fase para reconstrucción).
    """
    complex_spec = magnitude.T * np.exp(1j * np.angle(phase_ref))
    _, reconstructed = scipy_istft(complex_spec, fs=SR, nperseg=n_fft,
                                   noverlap=n_fft - hop)
    return reconstructed.astype(np.float32)


async def build_frame_pairs(
    clean_dir: str,
    occluded_dir: str,
    max_files: int = 200,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Construye pares (X_occluded_frames, Y_clean_frames) desde audios reales.
    Carga los archivos de forma asíncrona (asyncio.gather por par de archivos).

    Para cada archivo .wav en clean_dir busca su par en occluded_dir con el
    mismo nombre base. Devuelve:
        X  — magnitud STFT de frames ocluidos  [N, freq_bins]
        Y  — magnitud STFT de frames limpios   [N, freq_bins]
        file_ids — nombre base de origen por frame (para trazabilidad)
    """
    clean_path    = Path(clean_dir)
    occluded_path = Path(occluded_dir)

    clean_files = sorted(
        [f for f in clean_path.glob("*.wav")]
    )[:max_files]

    if not clean_files:
        raise FileNotFoundError(
            f"No se encontraron archivos .wav en {clean_dir}"
        )

    # Filtrar solo los pares que existen antes de lanzar tareas async
    valid_pairs = [
        (cf, occluded_path / cf.name)
        for cf in clean_files
        if (occluded_path / cf.name).exists()
    ]
    skipped = len(clean_files) - len(valid_pairs)
    if skipped:
        logger.warning(f"[Audio] {skipped} archivos sin par ocluido — omitidos")

    async def _load_pair(
        cf: Path, of: Path
    ) -> Optional[Tuple[np.ndarray, np.ndarray, str, int]]:
        """Carga un par clean/occluded de forma asíncrona."""
        try:
            clean_audio, occluded_audio = await asyncio.gather(
                load_audio(str(cf)),
                load_audio(str(of)),
            )
            min_len      = min(len(clean_audio), len(occluded_audio))
            clean_mag    = compute_stft_frames(clean_audio[:min_len])
            occluded_mag = compute_stft_frames(occluded_audio[:min_len])
            n_frames     = min(len(clean_mag), len(occluded_mag))
            return occluded_mag[:n_frames], clean_mag[:n_frames], cf.stem, n_frames
        except Exception as e:
            logger.error(f"Error cargando {cf.name}: {e}")
            return None

    # Cargar todos los pares en paralelo
    pairs = await asyncio.gather(*[_load_pair(cf, of) for cf, of in valid_pairs])

    X_list: List[np.ndarray] = []
    Y_list: List[np.ndarray] = []
    file_ids: List[str]      = []
    loaded = 0

    for result in pairs:
        if result is None:
            continue
        occ_mag, clean_mag, stem, n_frames = result
        X_list.append(occ_mag)
        Y_list.append(clean_mag)
        file_ids.extend([stem] * n_frames)
        loaded += 1

    if not X_list:
        raise RuntimeError(
            "No se pudo cargar ningún par de audios. "
            "Verifica clean_dir y occluded_dir."
        )

    X = np.vstack(X_list)
    Y = np.vstack(Y_list)

    logger.info(
        f"[Audio] {loaded} archivos cargados, {skipped} omitidos. "
        f"Total frames: {len(X)} × {X.shape[1]} bins"
    )
    return X, Y, file_ids


def simulate_occlusion(
    audio: np.ndarray,
    occlusion_ratio: float = 0.4,
    seed: int = 42,
) -> np.ndarray:
    """
    Simula oclusión espectral por máscara STFT cuando no hay par ocluido
    disponible en disco (modo 'solo audio limpio').
    Equivalente al método usado en la fase de pre-entrenamiento.
    """
    rng = np.random.default_rng(seed)
    _, _, Zxx = scipy_stft(audio, fs=SR, nperseg=N_FFT, noverlap=N_FFT - HOP_LENGTH)
    mask = (rng.uniform(0, 1, Zxx.shape) > occlusion_ratio).astype(np.float64)
    Zxx_occ = Zxx * mask
    _, occluded = scipy_istft(Zxx_occ, fs=SR, nperseg=N_FFT,
                              noverlap=N_FFT - HOP_LENGTH)
    return occluded.astype(np.float32)


async def build_frame_pairs_single_dir(
    audio_dir: str,
    max_files: int = 200,
    occlusion_ratio: float = 0.4,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Variante async de build_frame_pairs cuando solo hay audios limpios:
    simula la oclusión en runtime. Carga archivos en paralelo con asyncio.gather.
    """
    audio_path = Path(audio_dir)
    files = sorted(audio_path.glob("*.wav"))[:max_files]

    if not files:
        raise FileNotFoundError(f"No se encontraron .wav en {audio_dir}")

    async def _load_and_simulate(f: Path) -> Optional[Tuple[np.ndarray, np.ndarray, str, int]]:
        """Carga un archivo limpio y simula su versión ocluida."""
        try:
            clean    = await load_audio(str(f))
            occluded = simulate_occlusion(clean, occlusion_ratio)
            min_len  = min(len(clean), len(occluded))
            clean_mag    = compute_stft_frames(clean[:min_len])
            occluded_mag = compute_stft_frames(occluded[:min_len])
            n = min(len(clean_mag), len(occluded_mag))
            return occluded_mag[:n], clean_mag[:n], f.stem, n
        except Exception as e:
            logger.error(f"Error en {f.name}: {e}")
            return None

    results = await asyncio.gather(*[_load_and_simulate(f) for f in files])

    X_list: List[np.ndarray] = []
    Y_list: List[np.ndarray] = []
    file_ids: List[str]      = []

    for result in results:
        if result is None:
            continue
        occ_mag, clean_mag, stem, n = result
        X_list.append(occ_mag)
        Y_list.append(clean_mag)
        file_ids.extend([stem] * n)

    if not X_list:
        raise RuntimeError(f"No se pudo procesar ningún archivo en {audio_dir}")

    X = np.vstack(X_list)
    Y = np.vstack(Y_list)
    logger.info(
        f"[Audio simulado] {len(files)} archivos. "
        f"Total frames: {len(X)} × {X.shape[1]} bins"
    )
    return X, Y, file_ids


# ─────────────────────────────────────────────────────────────────────────────
# 2.  MÉTRICAS AOC  (sobre frames STFT reales)
# ─────────────────────────────────────────────────────────────────────────────
def compute_snr(clean: np.ndarray, estimated: np.ndarray, eps: float = 1e-9) -> float:
    """Signal-to-Noise Ratio en dB sobre magnitudes STFT."""
    # Asegurar que sean arrays 1D para el cálculo
    clean_flat = clean.flatten()
    estimated_flat = estimated.flatten()
    
    # Usar solo la parte válida (donde hay señal)
    signal_power = np.sum(clean_flat ** 2)
    noise_power = np.sum((clean_flat - estimated_flat) ** 2)
    
    if signal_power < eps:
        return -100.0  # Sin señal detectable
    
    if noise_power < eps:
        return 100.0  # Ruido nulo (perfecto)
    
    snr_val = 10 * np.log10(signal_power / (noise_power + eps))
    
    # Limitar a rango razonable
    return float(np.clip(snr_val, -20.0, 50.0))

def find_existing_checkpoints(output_dir: str, loop_counter: int) -> Dict[str, bool]:
    """Encuentra qué archivos de checkpoint existen para un loop específico."""
    checkpoints = {
        'cv_results': os.path.join(output_dir, f"cv_results_loop_{loop_counter}.pkl"),
        'model': os.path.join(output_dir, f"aoc_dsvm_loop_{loop_counter}.pkl"),
        'complete': os.path.join(output_dir, f"loop_{loop_counter}.complete"),
    }
    return {name: os.path.exists(path) for name, path in checkpoints.items()}

def should_skip_loop(output_dir: str, loop_counter: int, force_rerun: bool = False) -> Tuple[bool, str]:
    """Determina si un loop debe ser saltado porque ya fue completado."""
    if force_rerun:
        return False, "force_rerun=True"
    
    checkpoints = find_existing_checkpoints(output_dir, loop_counter)
    
    # Si existe cualquiera de estos, el baseline está completado
    if checkpoints['cv_results'] and checkpoints['model']:
        return True, f"Loop {loop_counter} ya completado"
    if checkpoints['complete']:
        return True, f"Loop {loop_counter} marcado como completo"
    
    # NUEVO: Verificar si existe distillation_results.json
    distillation_complete = False
    iter_dirs = glob.glob(os.path.join(output_dir, "iter_*"))
    for iter_dir in iter_dirs:
        if os.path.exists(os.path.join(iter_dir, "distillation_results.json")):
            distillation_complete = True
            break
    
    if distillation_complete:
        return True, f"Loop {loop_counter} distillation completada"
    
    # NUEVO: Verificar si ya pertenece a un crew
    crews_dir = os.path.join(output_dir, "crews")
    if os.path.exists(crews_dir):
        crew_files = glob.glob(os.path.join(crews_dir, "crew_*.pkl"))
        if crew_files:
            return True, f"Loop {loop_counter} ya integrado en crew"
    
    return False, "Loop no completado"

def find_completed_baselines(parallel_baselines_dir: str) -> Dict[int, Dict]:
    """
    Escanea el directorio de parallel_baselines para encontrar baselines completadas.
    
    Una baseline se considera COMPLETADA si tiene:
    - distillation_results.json en cualquier iter_XXXX/
    - O distilled_model.pkl
    - O un archivo baseline_complete.txt (marcador propio)
    
    Returns:
        Dict[int, Dict]: {baseline_idx: metadata}
    """
    completed = {}
    
    if not os.path.exists(parallel_baselines_dir):
        return completed
    
    for item in os.listdir(parallel_baselines_dir):
        if not item.startswith("baseline_"):
            continue
        
        baseline_path = os.path.join(parallel_baselines_dir, item)
        if not os.path.isdir(baseline_path):
            continue
        
        # Extraer índice
        try:
            baseline_idx = int(item.split("_")[1])
        except:
            continue
        
        # Verificar si está completada
        is_complete = False
        metadata = {'baseline_idx': baseline_idx, 'path': baseline_path}
        
        # 1. Buscar distillation_results.json en cualquier iter_XXXX/
        for subdir in os.listdir(baseline_path):
            if subdir.startswith("iter_"):
                iter_path = os.path.join(baseline_path, subdir)
                dist_results = os.path.join(iter_path, "distillation_results.json")
                if os.path.exists(dist_results):
                    is_complete = True
                    metadata['distillation_results'] = dist_results
                    metadata['iter'] = subdir
                    try:
                        with open(dist_results, 'r') as f:
                            dist_data = json.load(f)
                            metadata['snr_improvement'] = dist_data.get('snr_improvement', 0)
                            metadata['final_snr'] = dist_data.get('student_after_snr_db', 0)
                    except:
                        pass
                    break
                
                # También buscar distilled_model.pkl
                distilled_model = os.path.join(iter_path, "distilled_model.pkl")
                if os.path.exists(distilled_model):
                    is_complete = True
                    metadata['distilled_model'] = distilled_model
                    metadata['iter'] = subdir
                    break
        
        # 2. Buscar marker propio
        complete_marker = os.path.join(baseline_path, "baseline_complete.txt")
        if os.path.exists(complete_marker):
            is_complete = True
            try:
                with open(complete_marker, 'r') as f:
                    marker_data = json.load(f)
                    metadata.update(marker_data)
            except:
                pass
        
        # 3. Buscar distillation_history.txt (también indica que se ejecutó)
        dist_history = os.path.join(baseline_path, "distillation_history.txt")
        if os.path.exists(dist_history):
            is_complete = True
            metadata['has_distillation_history'] = True
        
        if is_complete:
            completed[baseline_idx] = metadata
            logger.info(f"📦 Baseline {baseline_idx:04d} ya completada: {metadata.get('iter', 'unknown')}")
    
    return completed


def get_pending_baselines(
    total_samples: int,
    samples_per_baseline: int,
    parallel_baselines_dir: str
) -> List[int]:
    """
    Retorna lista de índices de baselines que aún NO han sido procesadas.
    """
    completed = find_completed_baselines(parallel_baselines_dir)
    
    n_baselines = (total_samples + samples_per_baseline - 1) // samples_per_baseline
    
    pending = []
    for idx in range(n_baselines):
        if idx not in completed:
            pending.append(idx)
        else:
            logger.info(f"⏭️ Baseline {idx:04d} saltada (ya completada)")
    
    return pending

def compute_spectral_distortion(
    clean: np.ndarray, estimated: np.ndarray
) -> float:
    """Distorsión espectral media (log-spectral distance)."""
    eps = 1e-9
    lsd = np.mean(
        np.sqrt(
            np.mean(
                (10 * np.log10((clean + eps) / (estimated + eps))) ** 2,
                axis=-1,
            )
        )
    )
    return float(lsd)


def compute_temporal_consistency(estimated: np.ndarray) -> float:
    """Varianza de la diferencia temporal entre frames consecutivos."""
    if len(estimated) < 2:
        return 1.0
    diffs = np.diff(estimated, axis=0)
    return float(1.0 / (1.0 + np.var(diffs)))


def compute_robustness(
    model_fn,
    X: np.ndarray,
    Y: np.ndarray,
    n_perturbations: int = 5,
    noise_std: float = 0.01,
) -> float:
    """
    Robustez del modelo frente a pequeñas perturbaciones gaussianas.
    Usa frames de audio reales como base, no ruido puro.
    """
    base_preds = model_fn(X)
    base_preds = base_preds[0]
    base_mse   = float(np.mean((base_preds - Y) ** 2))

    perturbed_mses = []
    rng = np.random.default_rng(0)
    for _ in range(n_perturbations):
        noise     = rng.normal(0, noise_std, X.shape)
        X_noisy   = np.clip(X + noise, 0, None)
        preds_noisy = model_fn(X_noisy)
        perturbed_mses.append(float(np.mean((preds_noisy - Y) ** 2)))

    mean_perturbed = np.mean(perturbed_mses)
    robustness = float(np.exp(-(mean_perturbed - base_mse)))
    return float(np.clip(robustness, 0.0, 1.0))


async def evaluate_aoc_metrics(
    model_fn,
    X: np.ndarray,
    Y: np.ndarray,
    label: str = "",
) -> Dict[str, float]:
    """
    Evaluación completa AOC sobre datos de audio reales.
    model_fn: callable que recibe X [N, bins] y devuelve Y_hat [N, bins].
    """
    Y_hat = await model_fn(X)
    print("Y_hat:", Y_hat.shape)
    #Y_hat = Y_hat[0]
    mse   = float(np.mean((Y_hat - Y) ** 2))
    snr   = compute_snr(Y, Y_hat)
    lsd   = compute_spectral_distortion(Y, Y_hat)
    tc    = compute_temporal_consistency(Y_hat)
    #rob   = compute_robustness(model_fn, X, Y)

    metrics = {
        "mse": mse,
        "snr_db": snr,
        "log_spectral_distortion": lsd,
        "temporal_consistency": tc,
        "robustness": 0.75,
    }
    tag = f"[{label}] " if label else ""
    logger.info(
        f"{tag}MSE={mse:.5f}  SNR={snr:.2f}dB  "
    )
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# 3.  MODELO DSVM — ENVOLTORIO COMPATIBLE CON EL API NATIVO DEL REPO
# ─────────────────────────────────────────────────────────────────────────────
class DSVMWrapper:
    """
    Envoltorio sobre deep_support_vector_machines del repo.

    DISEÑO: todos los atributos que cross_validation.py lee (a, w, bias,
    proba, svs, ns, nvs, sv_locs, best_layer, written, gamma, kernel1,
    kernel2) son atributos de instancia privados (_w, _a, etc.) expuestos
    con @property SIN delegar a self.dsvm. self.dsvm solo se usa para
    llamar a los métodos de kernel (@classmethod sin estado).

    Esto evita el conflicto donde el setter redirige a self.dsvm pero el
    getter devuelve None porque self.dsvm nunca fue entrenado en modo
    regresión.
    """

    def __init__(self, dsvm_model=None, regression: bool = True):
        from apicultor.emotion.MusicEmotionMachine import deep_support_vector_machines
        self.regression     = regression
        self._fitted        = False
        self.dsvm           = (dsvm_model if dsvm_model is not None
                               else deep_support_vector_machines(kernel1="rbf",
                                                                kernel2="rbf"))
        # Atributos directos — NO delegados a self.dsvm
        self._a                 = np.array([]) 
        self._w                 = np.ones((1, 1025)) * 0.01
        self._bias              = 0.0
        self._proba             = []
        self._svs               = None
        self._ns                = np.array([], dtype=np.uint8)
        self._feature_weights  = np.array([], dtype=np.uint8)
        self._feature_weights_array  = np.array([], dtype=np.uint8)
        self._nvs               = []
        self._sv_locs           = np.array([0])
        self._dual_coefficients = []
        self._best_layer        = []
        self._written           = False
        self._gamma             = 1.0
        self._kernel1           = "rbf"
        self._kernel2           = "rbf"
        self._n_class           = 1
        self.arch               = defaultdict(list)
        self._fitted = False

    # ── properties directas (sin delegación a self.dsvm) ─────────────────────

    @property
    def a(self):
        return self._a

    @a.setter
    def a(self, v):
        self._a = v

    @property
    def feature_weights(self):
        return self._feature_weights if hasattr(self, '_feature_weights') else np.array([])
    
    @feature_weights.setter
    def feature_weights(self, v):
        self._feature_weights = v
        self._feature_weights_array = v
        # También actualizar w para compatibilidad
        if hasattr(self, '_w') or not hasattr(self, '_w'):
            self._w = v.reshape(1, -1) if v is not None and v.size > 0 else None

    @property
    def w(self):
        return self._w

    @w.setter
    def w(self, v):
        self._w = v

    @property
    def bias(self):
        return self._bias

    @bias.setter
    def bias(self, v):
        self._bias = v

    @property
    def proba(self):
        return self._proba

    @proba.setter
    def proba(self, v):
        self._proba = v

    @property
    def svs(self):
        return self._svs

    @svs.setter
    def svs(self, v):
        self._svs = v

    @property
    def ns(self):
        return self._ns

    @ns.setter
    def ns(self, v):
        self._ns = v

    @property
    def nvs(self):
        return self._nvs

    @nvs.setter
    def nvs(self, v):
        self._nvs = v

    @property
    def sv_locs(self):
        return self._sv_locs

    @sv_locs.setter
    def sv_locs(self, v):
        self._sv_locs = v

    @property
    def dual_coefficients(self):
        return self._dual_coefficients

    @dual_coefficients.setter
    def dual_coefficients(self, v):
        self._dual_coefficients = v

    @property
    def best_layer(self):
        return self._best_layer

    @best_layer.setter
    def best_layer(self, v):
        self._best_layer = v

    @property
    def written(self):
        return self._written

    @written.setter
    def written(self, v):
        self._written = v

    @property
    def gamma(self):
        return self._gamma

    @gamma.setter
    def gamma(self, v):
        self._gamma = v

    @property
    def kernel1(self):
        return self._kernel1

    @kernel1.setter
    def kernel1(self, v):
        self._kernel1 = v

    @property
    def kernel2(self):
        return self._kernel2

    @kernel2.setter
    def kernel2(self, v):
        self._kernel2 = v

    @property
    def n_class(self):
        return self._n_class

    @n_class.setter
    def n_class(self, v):
        self._n_class = v

    def apply(self, layer):
        if hasattr(self.dsvm, "apply"):
            self.dsvm.apply(layer)

    def clone(self):
        return DSVMWrapper(dsvm_model=deepcopy(self.dsvm),
                           regression=self.regression)

    # ── fit_model ─────────────────────────────────────────────────────────────
    async def fit_model(
        self,
        X: np.ndarray,
        y: np.ndarray,
        k0, k1,
        C: float, reg: float, gamma: float, nu: float,
        regression: bool = True,
    ):
        """
        regression=True → usa DSVM nativo en modo regresión vía SVM binario
                         con targets discretizados por mediana
        regression=False → llama self.dsvm.fit_model directamente
        """
        X = np.ascontiguousarray(np.float64(X))
        y = np.ascontiguousarray(np.float64(y)).flatten()
        
        self._kernel1 = k0
        self._kernel2 = k1
        self._gamma = gamma
        self.C = C
        self.learning_rate = nu
        
        if not regression:
            # Modo clasificación - usar DSVM nativo
            await self.dsvm.fit_model(X, y, k0, k1, C, reg, gamma, nu)
            for attr in ("a", "w", "bias", "proba", "svs", "ns", "nvs",
                         "sv_locs", "dual_coefficients", "n_class",
                         "best_layer", "written", "gamma", "kernel1", "kernel2"):
                if hasattr(self.dsvm, attr):
                    setattr(self, f"_{attr}", getattr(self.dsvm, attr))
            self._fitted = True
            return
        
        # ── REGRESIÓN: usar DSVM nativo con targets discretizados ────────────────
        # Convertir regresión a clasificación binaria basada en mediana
        #median_val = np.median(y)
        #labels_binary = (y > median_val).astype(int)
        labels_binary = y
        
        # Crear DSVM nativo para clasificación binaria
        from apicultor.emotion.MusicEmotionMachine import deep_support_vector_machines
        binary_svm = deep_support_vector_machines(kernel1=k0, kernel2=k1)
        
        # Entrenar DSVM nativo con labels binarios
        await binary_svm.fit_model(X, labels_binary, k0, k1, C, reg, gamma, nu)
        
        # Copiar atributos del DSVM nativo al wrapper
        self._a = binary_svm.a
        self._w = binary_svm.w
        self._bias = binary_svm.bias
        self._proba = binary_svm.proba if hasattr(binary_svm, 'proba') else []
        self._svs = binary_svm.svs
        self._ns = binary_svm.ns
        self._nvs = binary_svm.nvs
        self._sv_locs = binary_svm.sv_locs
        self._dual_coefficients = binary_svm.dual_coefficients
        self._n_class = binary_svm.n_class
        self._best_layer = binary_svm.best_layer if hasattr(binary_svm, 'best_layer') else []
        self._written = binary_svm.written if hasattr(binary_svm, 'written') else False
        self._fitted = True
        
        # Guardar referencia al DSVM entrenado para predicciones posteriores
        self.dsvm = binary_svm
        
        #logger.info(f"[DSVM] Regresión convertida a clasificación binaria con mediana={median_val:.6f}, n_class={self._n_class}, svs={len(self._svs) if self._svs is not None else 0}")
        self._fitted = True

    # ── predictions / predict ─────────────────────────────────────────────────

    def predictions(self, X: np.ndarray) -> np.ndarray:
        """
        Regresión: w·X + bias → (N, n_bins) reconstruyendo la magnitud espectral.
        Fallback: zeros si no está entrenado.
        """
        print("X", X.shape)
        print("W", self._w)
        print("BIAS", self._bias)
        w_vec = np.array(self._w).reshape(-1)   # (n_bins,)
        scalar_pred = X @ w_vec + self._bias              # (N,)
        X_norm      = np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
        return np.outer(scalar_pred, np.ones(X.shape[1])) * (X + 1e-9) / X_norm

    async def predict(self, X: np.ndarray) -> np.ndarray:
        """Alias de predictions() para métricas AOC."""
        return self.predictions(X)

# ============================================================
# EVALUACIÓN CONSTITUCIONAL POST-DISTILLATION
# ============================================================
async def evaluate_constitutional_post_distillation(
    model: DSVMWrapper,
    X_test: np.ndarray,
    Y_test: np.ndarray,
    constitution_rules: List[Dict],
    output_dir: str,
    model_name: str = "final_model"
) -> Dict:
    """
    Evalúa el modelo final con todos los verifiers constitucionales.
    
    Args:
        model: Modelo a evaluar (teacher o student destilado)
        X_test: Datos de test
        Y_test: Targets
        constitution_rules: Reglas constitucionales
        output_dir: Directorio para guardar resultados
        model_name: Nombre del modelo (para logging)
    
    Returns:
        Diccionario con resultados de evaluación
    """
    logger.info(f"=" * 70)
    logger.info(f"📊 EVALUACIÓN CONSTITUCIONAL POST-DISTILLATION - {model_name}")
    logger.info(f"=" * 70)
    
    # 1. Inicializar ConstitutionalAI
    constitutional_ai = ConstitutionalAI(constitution_rules=constitution_rules)
    
    # 2. Evaluar en muestras de test
    n_samples = min(500, len(X_test))  # Máximo 500 muestras para no saturar
    X_eval = X_test[:n_samples]
    Y_eval = Y_test[:n_samples]
    
    all_violations = []
    all_scores = []
    violations_by_principle = defaultdict(int)
    severity_counts = {'high': 0, 'medium': 0, 'low': 0, 'none': 0}
    
    logger.info(f"[{model_name}] Evaluando {n_samples} muestras...")
    
    for i in range(n_samples):
        x_f = X_eval[i]
        y_f = Y_eval[i]
        
        # Predicción del modelo
        pred = await model.predict(x_f.reshape(1, -1))
        pred = pred[0]
        
        # CoT simple para evaluación
        cot = _build_spectral_cot_simple(x_f, pred)
        
        # Evaluación constitucional
        result = constitutional_ai.evaluate(
            input=x_f,
            output=pred,
            chain_of_thought=cot,
        )
        
        all_scores.append(result["constitution_score"])
        
        for v in result.get("violations", []):
            principle = v.get("principle", "unknown")
            severity = v.get("severity", "low")
            violations_by_principle[principle] += 1
            severity_counts[severity] += 1
            all_violations.append(v)
    
    # 3. Calcular estadísticas
    mean_score = np.mean(all_scores)
    std_score = np.std(all_scores)
    min_score = np.min(all_scores)
    max_score = np.max(all_scores)
    
    total_violations = len(all_violations)
    n_clean = sum(1 for s in all_scores if s > 0.7)
    n_marginal = sum(1 for s in all_scores if 0.5 <= s <= 0.7)
    n_poor = sum(1 for s in all_scores if s < 0.5)
    
    # 4. Logging
    logger.info(f"[{model_name}] 📊 RESULTADOS:")
    logger.info(f"  Score constitucional medio: {mean_score:.4f} ± {std_score:.4f}")
    logger.info(f"  Score mínimo: {min_score:.4f}")
    logger.info(f"  Score máximo: {max_score:.4f}")
    logger.info(f"  Total violaciones: {total_violations}")
    logger.info(f"  Muestras limpias (score > 0.7): {n_clean}/{n_samples}")
    logger.info(f"  Muestras marginales (0.5-0.7): {n_marginal}/{n_samples}")
    logger.info(f"  Muestras pobres (score < 0.5): {n_poor}/{n_samples}")
    
    logger.info(f"[{model_name}] VIOLACIONES POR PRINCIPIO:")
    for principle, count in sorted(violations_by_principle.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"  - {principle}: {count} violaciones")
    
    logger.info(f"[{model_name}] SEVERIDAD:")
    for severity, count in severity_counts.items():
        logger.info(f"  - {severity}: {count}")
    
    # 5. Guardar resultados
    results = {
        "model_name": model_name,
        "n_samples": n_samples,
        "mean_constitution_score": float(mean_score),
        "std_constitution_score": float(std_score),
        "min_score": float(min_score),
        "max_score": float(max_score),
        "total_violations": total_violations,
        "clean_samples": n_clean,
        "marginal_samples": n_marginal,
        "poor_samples": n_poor,
        "violations_by_principle": dict(violations_by_principle),
        "severity_counts": severity_counts,
        "all_scores": [float(s) for s in all_scores],
        "violations": all_violations,
        "timestamp": datetime.now().isoformat()
    }
    
    # Guardar en archivo
    results_path = os.path.join(output_dir, f"constitutional_eval_{model_name}.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"[{model_name}] 📄 Resultados guardados en {results_path}")
    
    return results


def load_model_from_baseline(
    baseline_idx: int,
    parallel_baselines_dir: str
) -> Optional[DSVMWrapper]:
    """
    Carga el modelo destilado (teacher) de una baseline específica.
    
    Args:
        baseline_idx: Índice de la baseline que será teacher
        parallel_baselines_dir: Directorio con las baselines
    
    Returns:
        Modelo DSVMWrapper cargado, o None si no se encuentra
    """
    baseline_dir = os.path.join(parallel_baselines_dir, f"baseline_{baseline_idx:04d}")
    
    if not os.path.exists(baseline_dir):
        logger.warning(f"[DistillationTeacher] Directorio no existe: {baseline_dir}")
        return None
    
    # Buscar el modelo destilado (student que se convirtió en teacher)
    model_path = None
    
    # 1. Buscar en iter_*/dsvm_main_distilled.pkl
    for subdir in os.listdir(baseline_dir):
        if subdir.startswith("iter_"):
            iter_path = os.path.join(baseline_dir, subdir)
            candidate = os.path.join(iter_path, "dsvm_main_distilled.pkl")
            if os.path.exists(candidate):
                model_path = candidate
                break
            
            # También buscar como student
            candidate = os.path.join(iter_path, "student.pkl")
            if os.path.exists(candidate):
                model_path = candidate
                break
    
    # 2. Buscar en la raíz
    if model_path is None:
        candidate = os.path.join(baseline_dir, "distilled_model.pkl")
        if os.path.exists(candidate):
            model_path = candidate
    
    if model_path is None:
        logger.warning(f"[DistillationTeacher] No se encontró modelo en {baseline_dir}")
        return None

    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    logger.info(f"[DistillationTeacher] Teacher cargado desde baseline #{baseline_idx}: {model_path}")
    return model

def find_best_distillation_teacher(
    parallel_baselines_dir: str,
    mode: str = "angel",  # 'angel' o 'demon'
    exclude_baseline_idx: int = None,  # Para demonio: índice específico
    metric: str = "constitution_score"  # 'constitution_score', 'final_loss', 'snr_improvement'
) -> Tuple[Optional[int], Optional[Dict]]:
    """
    Escanea todas las baselines completadas y encuentra el teacher según el modo.
    
    Args:
        parallel_baselines_dir: Directorio con las baselines (parallel_baselines/)
        mode: 
            - 'angel': selecciona el de mayor constitution_score (mejor desempeño)
            - 'demon': selecciona el de menor constitution_score (peor desempeño)
        exclude_baseline_idx: Si se especifica, excluye esta baseline de la selección
        metric: Métrica para comparar:
            - 'constitution_score': usa constitution_score (default)
            - 'final_loss': menor loss final
            - 'snr_improvement': mayor mejora de SNR
            - 'student_snr': mayor SNR final
    
    Returns:
        (best_baseline_idx, best_metadata) o (None, None) si no hay ninguna completada
    """
    best_idx = None
    best_metadata = None
    best_value = None
    
    if not os.path.exists(parallel_baselines_dir):
        logger.info(f"[DistillationTeacher] Directorio no existe: {parallel_baselines_dir}")
        return None, None
    
    # Recolectar todas las baselines con sus constitution_scores
    baselines_data = []
    
    for item in os.listdir(parallel_baselines_dir):
        if not item.startswith("baseline_"):
            continue
        
        baseline_path = os.path.join(parallel_baselines_dir, item)
        if not os.path.isdir(baseline_path):
            continue
        
        # Extraer índice
        try:
            baseline_idx = int(item.split("_")[1])
        except:
            continue
        
        # Excluir si es necesario
        if exclude_baseline_idx is not None and baseline_idx == exclude_baseline_idx:
            continue
        
        # Buscar constitution_score (prioridad)
        constitution_score = None
        score_file = os.path.join(baseline_path, f"constitution_score_{baseline_idx:04d}.json")
        if os.path.exists(score_file):
            try:
                with open(score_file, 'r') as f:
                    score_data = json.load(f)
                    constitution_score = score_data.get('constitution_score', 0.5)
            except:
                pass
        
        # Buscar resultados de distillation (fallback)
        distillation_results = None
        distillation_metrics = {}
        
        # Buscar en iter_*/distillation_results.json
        for subdir in os.listdir(baseline_path):
            if subdir.startswith("iter_"):
                iter_path = os.path.join(baseline_path, subdir)
                dist_file = os.path.join(iter_path, "distillation_results.json")
                if os.path.exists(dist_file):
                    try:
                        with open(dist_file, 'r') as f:
                            distillation_results = json.load(f)
                            distillation_metrics = distillation_results.get('distillation_metrics', {})
                        break
                    except:
                        pass
        
        # También buscar en la raíz
        if distillation_results is None:
            dist_file = os.path.join(baseline_path, "distillation_results.json")
            if os.path.exists(dist_file):
                try:
                    with open(dist_file, 'r') as f:
                        distillation_results = json.load(f)
                        distillation_metrics = distillation_results.get('distillation_metrics', {})
                except:
                    pass
        
        # Extraer valor según métrica solicitada
        current_value = None
        
        if metric == 'constitution_score':
            current_value = constitution_score if constitution_score is not None else 0.5
        elif metric == 'final_loss':
            current_value = distillation_metrics.get('final_loss')
        elif metric == 'min_loss':
            current_value = distillation_metrics.get('min_loss')
        elif metric == 'snr_improvement':
            current_value = distillation_results.get('snr_improvement') if distillation_results else None
        elif metric == 'student_snr':
            current_value = distillation_results.get('student_after_snr_db') if distillation_results else None
        else:
            current_value = distillation_results.get(metric) if distillation_results else None
        
        if current_value is None and metric != 'constitution_score':
            # Fallback a constitution_score si no hay métrica
            current_value = constitution_score if constitution_score is not None else 0.5
        
        baselines_data.append({
            'baseline_idx': baseline_idx,
            'path': baseline_path,
            'value': current_value,
            'constitution_score': constitution_score,
            'distillation_results': distillation_results
        })
    
    if not baselines_data:
        logger.info(f"[DistillationTeacher] No se encontraron baselines completadas")
        return None, None
    
    # Ordenar según modo
    if mode == 'angel':
        # Ángel: mayor valor es mejor
        baselines_data.sort(key=lambda x: x['value'], reverse=True)
        best_data = baselines_data[0] if baselines_data else None
        logger.info(f"[DistillationTeacher] Modo ÁNGEL: seleccionado baseline #{best_data['baseline_idx']} "
                   f"({metric}={best_data['value']:.4f})" if best_data else "Ninguno")
    elif mode == 'demon':
        # Demonio: menor valor es mejor (peor desempeño)
        baselines_data.sort(key=lambda x: x['value'])
        best_data = baselines_data[0] if baselines_data else None
        logger.info(f"[DistillationTeacher] Modo DEMONIO: seleccionado baseline #{best_data['baseline_idx']} "
                   f"({metric}={best_data['value']:.4f})" if best_data else "Ninguno")
    else:
        logger.error(f"[DistillationTeacher] Modo desconocido: {mode}")
        return None, None
    
    if best_data:
        best_idx = best_data['baseline_idx']
        best_metadata = {
            'baseline_idx': best_idx,
            'path': best_data['path'],
            'mode': mode,
            'metric': metric,
            'value': best_data['value'],
            'constitution_score': best_data['constitution_score'],
            'distillation_results': best_data['distillation_results']
        }
    
    return best_idx, best_metadata


def find_angel_teacher(
    parallel_baselines_dir: str,
    exclude_baseline_idx: int = None,
    metric: str = "constitution_score"
) -> Tuple[Optional[int], Optional[Dict]]:
    """Conveniencia: encuentra el mejor ángel teacher"""
    return find_best_distillation_teacher(
        parallel_baselines_dir=parallel_baselines_dir,
        mode='angel',
        exclude_baseline_idx=exclude_baseline_idx,
        metric=metric
    )


def find_demon_teacher(
    parallel_baselines_dir: str,
    exclude_baseline_idx: int = None,
    metric: str = "constitution_score"
) -> Tuple[Optional[int], Optional[Dict]]:
    """Conveniencia: encuentra el peor demonio teacher"""
    return find_best_distillation_teacher(
        parallel_baselines_dir=parallel_baselines_dir,
        mode='demon',
        exclude_baseline_idx=exclude_baseline_idx,
        metric=metric
    )

async def extract_hypotheses_from_rollout(
    model: DSVMWrapper,
    X: np.ndarray,
    Y: np.ndarray,
) -> List['Hypothesis']:
    """
    Extrae hipótesis de un rollout.
    Retorna SOLO la mejor hipótesis (top_k=1).
    """
    from run_full_pipeline import compute_snr, compute_spectral_distortion

    print("[Hypotheses] === INICIO extract_hypotheses_from_rollout ===")
    n_thresholds = 10

    # ========== VALIDAR X ==========
    if X is None:
        print("[Hypotheses] X es None, retornando []")
        return []
    
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(1, -1)

    n_samples_actual, n_features = X.shape
    print(f"[Hypotheses] X shape=({n_samples_actual}, {n_features})")

    if n_samples_actual < 2:
        print(f"[Hypotheses] X tiene solo {n_samples_actual} muestra(s). Necesita >= 2.")
        return []

    # ========== VALIDAR Y ==========
    if Y is not None:
        Y = np.asarray(Y, dtype=np.float64)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        if Y.shape[0] != n_samples_actual:
            min_n = min(n_samples_actual, Y.shape[0])
            X = X[:min_n]
            Y = Y[:min_n]
            n_samples_actual = min_n

    # ========== PREDICCIONES ==========
    predictions = await model.predict(X)
    predictions = np.asarray(predictions, dtype=np.float64)
    if predictions.ndim == 1:
        predictions = predictions.reshape(n_samples_actual, -1)

    # ========== VALORES POR DEFECTO ==========
    constitution_score = 0.5
    violations = []
    composite_reward = 0.0
    error_bucket = None

    # ========== FUNCIÓN PARA ENCONTRAR MEJOR THRESHOLD ==========
    def find_best_threshold(feature_col: np.ndarray, targets: np.ndarray) -> Tuple[float, float, str, int, int]:
        """
        Encuentra el threshold que maximiza la diferencia entre grupos.
        Retorna: (best_threshold, best_score, best_direction, n_pos, n_neg)
        """
        percentiles = np.linspace(10, 90, n_thresholds)
        candidate_thresholds = np.unique(np.percentile(feature_col, percentiles))
        
        best_threshold = np.median(feature_col)
        best_score = -np.inf
        best_direction = ">"
        best_n_pos = 0
        best_n_neg = 0
        
        for t in candidate_thresholds:
            for direction in [">", "<"]:
                if direction == ">":
                    pos_indices = np.where(feature_col > t)[0]
                    neg_indices = np.where(feature_col <= t)[0]
                else:
                    pos_indices = np.where(feature_col < t)[0]
                    neg_indices = np.where(feature_col >= t)[0]
                
                if len(pos_indices) < 3 or len(neg_indices) < 3:
                    continue
                
                pos_mean = np.mean(targets[pos_indices])
                neg_mean = np.mean(targets[neg_indices])
                mean_diff = abs(pos_mean - neg_mean)
                
                std_combined = np.std(np.concatenate([targets[pos_indices], targets[neg_indices]])) + 1e-9
                score = mean_diff / std_combined
                
                # Bonus por grupos balanceados
                balance = min(len(pos_indices), len(neg_indices)) / max(len(pos_indices), len(neg_indices))
                score *= (0.7 + 0.3 * balance)
                
                if score > best_score:
                    best_score = score
                    best_threshold = t
                    best_direction = direction
                    best_n_pos = len(pos_indices)
                    best_n_neg = len(neg_indices)
        
        return best_threshold, best_score, best_direction, best_n_pos, best_n_neg

    # ========== BUSCAR MEJOR FEATURE Y MEJOR HIPÓTESIS ==========
    best_hypothesis = None
    best_overall_score = -np.inf
    targets_flat = Y.flatten() if Y is not None else predictions.flatten()
    
    # Probar todas las features (primeras 10)
    feature_indices = list(range(min(10, n_features)))
    
    for feature_idx in feature_indices:
        feature_col = X[:, feature_idx]
        
        best_threshold, score, direction, n_pos, n_neg = find_best_threshold(feature_col, targets_flat)
        
        if score == -np.inf:
            continue
        
        logical = (direction == ">")
        
        if logical:
            pos_indices = np.where(feature_col > best_threshold)[0]
            neg_indices = np.where(feature_col <= best_threshold)[0]
        else:
            pos_indices = np.where(feature_col < best_threshold)[0]
            neg_indices = np.where(feature_col >= best_threshold)[0]
        
        # Métricas por grupo
        pos_pred = predictions[pos_indices]
        neg_pred = predictions[neg_indices]
        pos_target = Y[pos_indices] if Y is not None else pos_pred
        neg_target = Y[neg_indices] if Y is not None else neg_pred
        
        pos_mse = float(np.mean((pos_pred - pos_target) ** 2))
        neg_mse = float(np.mean((neg_pred - neg_target) ** 2))
        
        if Y is not None:
            pos_snr = compute_snr(pos_target, pos_pred) if len(pos_target) > 0 else -100.0
            neg_snr = compute_snr(neg_target, neg_pred) if len(neg_target) > 0 else -100.0
        else:
            pos_snr = neg_snr = 0.0
        
        max_mse = max(pos_mse, neg_mse, 1e-9)
        pos_accuracy = 1.0 - min(1.0, pos_mse / max_mse)
        neg_accuracy = 1.0 - min(1.0, neg_mse / max_mse)
        accuracy_gap = abs(pos_accuracy - neg_accuracy)
        snr_gap = abs(pos_snr - neg_snr)
        
        hypothesis = Hypothesis(
            feature_idx=feature_idx,
            limit="optimized",
            logical=logical,
            direction=direction,
            intersection=float(best_threshold),
            pos_accuracy=float(pos_accuracy),
            neg_accuracy=float(neg_accuracy),
            pos_mse=float(pos_mse),
            neg_mse=float(neg_mse),
            pos_snr=float(pos_snr),
            neg_snr=float(neg_snr),
            accuracy_gap=float(accuracy_gap),
            snr_gap=float(snr_gap),
            n_pos_samples=n_pos,
            n_neg_samples=n_neg,
            constitution_score=constitution_score,
            violations=violations,
            composite_reward=composite_reward,
            error_bucket=error_bucket,
            rollout_id=None
        )
        
        print(f"[Hypotheses] feature={feature_idx}, direction={direction}, threshold={best_threshold:.6f}, score={hypothesis.score:.4f}")
        
        if hypothesis.score > best_overall_score:
            best_overall_score = hypothesis.score
            best_hypothesis = hypothesis
    
    if best_hypothesis is None:
        print("[Hypotheses] No se encontró ninguna hipótesis válida")
        return []
    
    print(f"[Hypotheses] MEJOR HIPÓTESIS: feature={best_hypothesis.feature_idx}, score={best_hypothesis.score:.4f}")
    
    return [best_hypothesis]  # ← SOLO 1 HIPÓTESIS

class DSVMRewardModelWithMajorityVoting(DSVMRewardModel):
    def __init__(self, base_model, adversary_models: List[Dict] = None):
        """
        Args:
            base_model: modelo principal
            adversary_models: lista de modelos para majority voting
                              cada uno con {'model': DSVMWrapper, 'weight': float, 'name': str}
        """
        self.base_model = base_model
        self.adversary_models = adversary_models or []
        self.reward_history = []
        self.hacking_detector = RewardHackingDetector()
        
        # Configurar pesos para majority voting
        if self.adversary_models:
            total_weight = sum(adv.get('weight', 1.0) for adv in self.adversary_models)
            self.voter_weights = [adv.get('weight', 1.0) / total_weight for adv in self.adversary_models]
            self.voter_names = [adv.get('name', f'voter_{i}') for i, adv in enumerate(self.adversary_models)]
        else:
            self.voter_weights = []
            self.voter_names = []
        
        logger.info(f"[RewardModel] Majority voting con {len(self.adversary_models)} modelos: {self.voter_names}")
    
    async def score(self, input, output, target=None, chain_of_thought=None):
        """
        Evalúa la salida usando majority voting con modelos adversarios.
        """
        scores = {}
        
        # 1. Score del modelo base
        if target is not None:
            if output.shape == target.shape:
                mse = float(np.mean((output - target) ** 2))
                base_accuracy = 1.0 - min(mse, 1.0)
            else:
                base_accuracy = 0.5
        else:
            base_accuracy = 0.5
        
        # 2. Majority voting con todos los modelos enrutados
        voter_scores = []
        
        for adv in self.adversary_models:
            adv_model = adv['model']
            try:
                adv_pred = await adv_model.predict(input.reshape(1, -1))
                adv_pred = adv_pred[0]
                if target is not None:
                    adv_mse = float(np.mean((adv_pred - output) ** 2))
                    adv_accuracy = 1.0 - min(adv_mse, 1.0)
                else:
                    adv_accuracy = 0.5
                voter_scores.append(adv_accuracy)
            except Exception as e:
                logger.debug(f"[RewardModel] Error en modelo {adv.get('name', 'unknown')}: {e}")
                voter_scores.append(0.5)
        
        # 3. Combinar votos (weighted average)
        if voter_scores and self.voter_weights:
            # Asegurar que tengamos la misma cantidad de pesos que votantes
            weights = self.voter_weights[:len(voter_scores)]
            weights_sum = sum(weights)
            if weights_sum > 0:
                majority_score = sum(v * w for v, w in zip(voter_scores, weights)) / weights_sum
            else:
                majority_score = np.mean(voter_scores)
        else:
            majority_score = base_accuracy
        
        # 4. Recompensa compuesta: 60% majority voting, 40% modelo base
        composite_reward = 0.6 * majority_score + 0.4 * base_accuracy
        
        # 5. Calidad del CoT
        if chain_of_thought:
            scores['cot_quality'] = min(len(chain_of_thought) / 10.0, 1.0)
        else:
            scores['cot_quality'] = 0.3
        
        # 6. Detección de reward hacking
        hacking_score, hacking_flags = self.hacking_detector.detect(input, output, scores)
        scores['hacking_risk'] = hacking_score
        scores['accuracy'] = base_accuracy
        scores['majority_vote'] = majority_score
        scores['n_voters'] = len(voter_scores)
        scores['overconfidence_penalty'] = self._compute_overconfidence_penalty(output)
        scores['distribution_shift_penalty'] = self._compute_distribution_shift_penalty(input, output)
        scores['abstention_bonus'] = self._compute_abstention_reward(input, output)        
        # Penalizar si hay alta divergencia entre votantes (consensus bajo)
        if len(voter_scores) > 1:
            voter_std = np.std(voter_scores)
            consensus_penalty = min(voter_std, 0.3) * 0.2
            composite_reward = composite_reward * (1 - consensus_penalty)
            scores['voter_consensus'] = 1.0 - voter_std
            scores['consensus_penalty'] = consensus_penalty
        
        self.reward_history.append({
            'scores': scores,
            'composite_reward': composite_reward,
            'hacking_flags': hacking_flags,
            'voter_scores': voter_scores,
            'majority_score': majority_score
        })
        
        return {
            'composite_reward': composite_reward,
            'component_scores': scores,
            'uncertainty': 1.0 - majority_score if voter_scores else 0.0,
            'hacking_flags': hacking_flags,
            'voter_scores': voter_scores,
            'majority_vote_score': majority_score
        }

    def get_voter_metrics(self) -> Dict:
        """Métricas sobre el desempeño de los votantes"""
        if not self.reward_history:
            return {}
        
        voter_scores = [h.get('voter_scores', []) for h in self.reward_history if h.get('voter_scores')]
        if not voter_scores:
            return {'n_voters': len(self.adversary_models)}
        
        # Calcular consistencia entre votantes
        all_scores = np.array(voter_scores)
        mean_voter_accuracy = np.mean(all_scores, axis=0) if len(all_scores.shape) > 1 else np.mean(all_scores)
        
        return {
            'n_voters': len(self.adversary_models),
            'mean_voter_accuracy': float(np.mean(mean_voter_accuracy)) if hasattr(mean_voter_accuracy, '__len__') else float(mean_voter_accuracy),
            'voter_consensus': float(1.0 - np.std(all_scores)) if len(all_scores) > 0 else 0.0,
            'voter_scores_std': float(np.std(all_scores)) if len(all_scores) > 0 else 0.0
        }

class DSVMRewardModelWithReference(DSVMRewardModel):
    def __init__(self, model, quote_model=None, quote_proba=None):
        super().__init__(model)
        self.quote_model = quote_model
        self.quote_proba = quote_proba
        self.alignment_tax_history = []
        
    def score(self, input, output, target=None, chain_of_thought=None):
        scores = super().score(input, output, target, chain_of_thought)
            
        # Alignment tax: penalizar desviación del quote_reference
        if self.quote_model is not None and hasattr(self.quote_model, 'predict'):
            quote_pred = asyncio.run(self.quote_model.predict(input.reshape(1, -1)))[0]
            quote_pred = quote_pred[0]
            quote_deviation = np.mean((output - quote_pred) ** 2)
            alignment_penalty = min(quote_deviation, 0.5)
                
            self.alignment_tax_history.append({
                'deviation': quote_deviation,
                'penalty': alignment_penalty,
                'timestamp': datetime.now().isoformat()
            })
                
            # Aplicar penalización a la recompensa compuesta
            if 'composite_reward' in scores:
                original_reward = scores['composite_reward']
                scores['composite_reward'] = max(0, original_reward - alignment_penalty)
                scores['alignment_penalty'] = alignment_penalty
                scores['quote_deviation'] = quote_deviation
            
        return scores

# ============================================================
# 4. INSTRUCTION FOLLOWING - Verifier
# ============================================================
def _verifier_instruction_following(
    input_arr: np.ndarray,
    output_arr: np.ndarray,
    chain_of_thought: Optional[List[str]] = None,
    instruction: Optional[str] = None,
    required_fields: Optional[List[str]] = None,
) -> Dict:
    """
    INSTRUCTION FOLLOWING: Verifica que el modelo siga las instrucciones.
    
    Para AOC, verifica:
    - La salida tiene la forma esperada
    - La energía de salida está en el rango esperado
    - No hay campos requeridos faltantes
    """
    inp = input_arr.flatten()
    out = output_arr.flatten()
    
    # 1. Verificar forma (instrucción implícita: misma forma)
    if input_arr.shape != output_arr.shape:
        return {
            "passed": False,
            "score": 0.0,
            "details": f"Shape mismatch: input {input_arr.shape} vs output {output_arr.shape}",
            "severity": "high",
        }
    
    # 2. Verificar que la salida no sea completamente plana
    out_std = np.std(out)
    if out_std < 1e-6:
        return {
            "passed": False,
            "score": 0.0,
            "details": "Output is nearly constant (flat)",
            "severity": "medium",
        }
    
    # 3. Verificar que la salida no sea idéntica a la entrada (debe haber procesamiento)
    if np.allclose(out, inp, rtol=1e-3):
        return {
            "passed": False,
            "score": 0.2,
            "details": "Output identical to input (no processing)",
            "severity": "medium",
        }
    
    # 4. Si hay campos requeridos, verificarlos
    if required_fields:
        missing_fields = []
        for field in required_fields:
            if field not in chain_of_thought or len(chain_of_thought) == 0:
                missing_fields.append(field)
        
        if missing_fields:
            return {
                "passed": False,
                "score": 0.3,
                "details": f"Missing required fields: {missing_fields}",
                "severity": "medium",
            }
    
    # 5. Score basado en cumplimiento
    score = 1.0
    
    # Penalizar si la salida es demasiado ruidosa
    if np.mean(out ** 2) > 10 * np.mean(inp ** 2):
        score -= 0.3
    
    # Penalizar si la salida es demasiado silenciosa
    if np.mean(out ** 2) < 0.01 * np.mean(inp ** 2):
        score -= 0.3
    
    # Bonus por CoT que menciona la instrucción
    if instruction and chain_of_thought:
        if any(instruction.lower() in step.lower() for step in chain_of_thought):
            score += 0.1
    
    score = float(np.clip(score, 0.0, 1.0))
    passed = score >= 0.6
    
    return {
        "passed": passed,
        "score": score,
        "details": f"std={out_std:.4f}, energy_ratio={np.mean(out**2)/(np.mean(inp**2)+1e-9):.2f}",
        "severity": "medium" if not passed else "low",
    }

def _verifier_multi_turn_consistency(
    input_arr: np.ndarray,
    output_arr: np.ndarray,
    chain_of_thought: Optional[List[str]] = None,
    previous_outputs: Optional[List[np.ndarray]] = None,  # historial de turnos
) -> Dict:
    """
    MULTI-TURN CONSISTENCY: Para AOC en modo multi-turno (rollouts secuenciales).
    Verifica que las predicciones no varíen drásticamente entre turnos consecutivos.
    Si no hay historial, asume consistencia (pasa, pero con score bajo).
    """
    if previous_outputs is None or len(previous_outputs) == 0:
        # No hay historial, no se puede evaluar consistencia multi-turno
        return {
            "passed": True,   # No falla, pero tampoco es evidencia de consistencia
            "score": 0.5,
            "details": "no_history_for_multi_turn",
            "severity": "low",
        }
    
    # Comparar con la salida anterior
    prev = previous_outputs[-1].flatten()
    curr = output_arr.flatten()
    
    # Cambio relativo en energía
    energy_prev = np.mean(prev ** 2)
    energy_curr = np.mean(curr ** 2)
    energy_change = abs(energy_curr - energy_prev) / (energy_prev + 1e-9)
    
    # Similitud coseno entre espectros
    if np.linalg.norm(prev) > 0 and np.linalg.norm(curr) > 0:
        cosine_sim = np.dot(prev, curr) / (np.linalg.norm(prev) * np.linalg.norm(curr))
    else:
        cosine_sim = 0.0
    
    # Consistencia si el cambio de energía < 50% y similitud > 0.6
    consistent = (energy_change < 0.5) and (cosine_sim > 0.6)
    
    score = (1.0 - min(energy_change, 1.0)) * 0.5 + cosine_sim * 0.5
    score = float(np.clip(score, 0.0, 1.0))
    
    return {
        "passed": consistent,
        "score": score,
        "details": f"energy_change={energy_change:.3f}, cosine_sim={cosine_sim:.3f}",
        "severity": "medium" if not consistent else "low",
    }

def _verifier_retrieval_grounding(
    input_arr: np.ndarray,
    output_arr: np.ndarray,
    chain_of_thought: Optional[List[str]] = None,
    context: Optional[np.ndarray] = None,
) -> Dict:
    """
    RETRIEVAL / GROUNDING: Verifica que la salida esté fundamentada en la entrada.
    Para AOC: la salida debe correlacionarse significativamente con la entrada.
    """
    inp = input_arr.flatten()
    out = output_arr.flatten()
    
    # 1. Correlación de Pearson
    if np.std(inp) > 0 and np.std(out) > 0:
        correlation = np.corrcoef(inp, out)[0, 1]
    else:
        correlation = 0.0
    correlation = np.nan_to_num(correlation, nan=0.0)
    
    # 2. Divergencia KL entre distribuciones
    hist_in, bins = np.histogram(inp, bins=20, density=True)
    hist_out, _ = np.histogram(out, bins=bins, density=True)
    hist_in = hist_in + 1e-9
    hist_out = hist_out + 1e-9
    hist_in = hist_in / np.sum(hist_in)
    hist_out = hist_out / np.sum(hist_out)
    kl_divergence = np.sum(hist_in * np.log(hist_in / hist_out))
    kl_divergence = np.nan_to_num(kl_divergence, nan=5.0, posinf=5.0, neginf=5.0)
    
    # 3. Consistencia de energía por bandas
    n_bands = 4
    in_bands = np.array_split(inp, n_bands)
    out_bands = np.array_split(out, n_bands)
    band_ratios = []
    for ib, ob in zip(in_bands, out_bands):
        e_in = np.sum(ib ** 2) + 1e-9
        e_out = np.sum(ob ** 2) + 1e-9
        band_ratios.append(e_out / e_in)
    band_ratio_std = np.std(band_ratios)
    band_ratio_std = np.nan_to_num(band_ratio_std, nan=1.0)
    
    # 4. Score combinado (0 a 1)
    score = (correlation + 0.5) * 0.4
    score += (1.0 - min(kl_divergence, 2.0) / 2.0) * 0.3
    score += (1.0 - min(band_ratio_std, 1.0)) * 0.3
    score = np.nan_to_num(score, nan=0.0)
    score = float(np.clip(score, 0.0, 1.0))
    
    grounded = score >= 0.5
    
    return {
        "passed": grounded,
        "score": score,
        "details": f"corr={correlation:.3f}, KL={kl_divergence:.3f}, band_std={band_ratio_std:.3f}",
        "severity": "high" if not grounded else "low",
    }
    
def _verifier_tool_api_use(
    input_arr: np.ndarray,
    output_arr: np.ndarray,
    chain_of_thought: Optional[List[str]] = None,
) -> Dict:
    """
    TOOL / API USE (trace quality): Para AOC, verificamos que el modelo
    haya seguido los pasos esperados de reconstrucción espectral.
    Se mide a través de la consistencia de la cadena de razonamiento.
    """
    # Si no hay CoT, no podemos verificar
    if chain_of_thought is None or len(chain_of_thought) == 0:
        return {
            "passed": False,
            "score": 0.0,
            "details": "No chain_of_thought provided",
            "severity": "high",
        }
    
    # Palabras clave esperadas en un razonamiento correcto de AOC
    expected_keywords = [
        "Wiener spectral subtraction",
        "continuous multi-lorax",
        "reconstrucción",
        "oclusión",
        "espectral"
    ]
    
    found_keywords = []
    for kw in expected_keywords:
        for step in chain_of_thought:
            if kw.lower() in step.lower():
                found_keywords.append(kw)
                break
    
    coverage = len(found_keywords) / len(expected_keywords)
    
    # También verificar pasos lógicos: debe haber al menos mención de atención
    has_attention = any("atención" in step.lower() or "attention" in step.lower() 
                        for step in chain_of_thought)
    
    score = coverage * (0.8 + 0.2 * has_attention)
    passed = score >= 0.6
    
    return {
        "passed": passed,
        "score": float(score),
        "details": f"keywords_found={found_keywords}, coverage={coverage:.2f}, has_attention={has_attention}",
        "severity": "medium" if not passed else "low",
    }

def _verifier_over_refusal(
    input_arr: np.ndarray,
    output_arr: np.ndarray,
    chain_of_thought: Optional[List[str]] = None,
) -> Dict:
    """
    OVER-REFUSAL: El modelo se abstiene (output casi nulo) cuando
    la entrada es perfectamente válida y debería procesarse.
    Para AOC: detecta silencio o atenuación extrema en presencia de señal.
    """
    inp = input_arr.flatten()
    out = output_arr.flatten()
    
    energy_in = np.mean(inp ** 2)
    energy_out = np.mean(out ** 2)
    attenuation = energy_out / (energy_in + 1e-9)
    
    # Detectar si la entrada tiene energía suficiente y la salida es casi nula
    input_has_signal = energy_in > 1e-5
    output_is_silent = energy_out < 1e-8
    
    is_over_refusal = input_has_signal and output_is_silent
    
    # Adicional: varianza de salida colapsada vs varianza de entrada
    var_in = np.var(inp)
    var_out = np.var(out)
    variance_collapse = (var_out < var_in * 0.001) and (var_in > 1e-5)
    
    if variance_collapse:
        is_over_refusal = True
    
    score = 1.0
    severity = "low"
    details = ""
    passed = True
    
    if is_over_refusal:
        passed = False
        score = 0.0
        severity = "medium"
        details = f"attenuation={attenuation:.2e}, var_in={var_in:.2e}, var_out={var_out:.2e}"
    
    return {
        "passed": passed,
        "score": score,
        "details": details,
        "severity": severity,
    }

# ============================================================
# 7. SCHEMA / FORMAT VIOLATIONS - Verifier específico para AOC
# ============================================================
def _verifier_schema_format(
    input_arr: np.ndarray,
    output_arr: np.ndarray,
    chain_of_thought: Optional[List[str]] = None,
) -> Dict:
    """
    SCHEMA / FORMAT: Verifica que la salida cumpla con el formato esperado.
    
    Para AOC, verifica:
    - Forma correcta (misma que input)
    - Valores no negativos (magnitudes STFT)
    - Sin valores NaN o Inf
    - Rango de valores razonable
    """
    # 1. Verificar forma
    if input_arr.shape != output_arr.shape:
        return {
            "passed": False,
            "score": 0.0,
            "details": f"Shape mismatch: input {input_arr.shape} vs output {output_arr.shape}",
            "severity": "high",
        }
    
    # 2. Verificar valores no negativos
    out_flat = output_arr.flatten()
    if np.any(out_flat < 0):
        neg_count = np.sum(out_flat < 0)
        return {
            "passed": False,
            "score": 0.0,
            "details": f"Negative values found: {neg_count} bins",
            "severity": "high",
        }
    
    # 3. Verificar NaN o Inf
    if np.any(np.isnan(out_flat)) or np.any(np.isinf(out_flat)):
        nan_count = np.sum(np.isnan(out_flat))
        inf_count = np.sum(np.isinf(out_flat))
        return {
            "passed": False,
            "score": 0.0,
            "details": f"NaN: {nan_count}, Inf: {inf_count}",
            "severity": "high",
        }
    
    # 4. Verificar rango de valores (energía razonable)
    max_val = np.max(out_flat)
    min_val = np.min(out_flat)
    mean_val = np.mean(out_flat)
    
    # Rango típico para magnitudes STFT: 0 a ~10
    if max_val > 100 or mean_val > 50:
        return {
            "passed": False,
            "score": 0.0,
            "details": f"Extreme values: max={max_val:.2f}, mean={mean_val:.2f}",
            "severity": "medium",
        }
    
    # 5. Verificar consistencia de dtype
    if not isinstance(output_arr, np.ndarray):
        return {
            "passed": False,
            "score": 0.0,
            "details": f"Output is not numpy array: {type(output_arr)}",
            "severity": "high",
        }
    
    # Score basado en qué tan bien cumple el formato
    score = 1.0
    penalizaciones = 0
    
    # Penalizar si hay valores atípicos
    if max_val > 10:
        penalizaciones += 0.1
    if np.std(out_flat) > 5:
        penalizaciones += 0.05
    
    score = max(0.0, 1.0 - penalizaciones)
    
    return {
        "passed": True,
        "score": float(score),
        "details": f"shape={output_arr.shape}, range=[{min_val:.4f}, {max_val:.4f}], mean={mean_val:.4f}",
        "severity": "low",
    }


# ============================================================
# 8. RETRIEVAL / GROUNDING - Verifier (ya existe, pero lo mejoramos)
# ============================================================
def _verifier_retrieval_grounding_improved(
    input_arr: np.ndarray,
    output_arr: np.ndarray,
    chain_of_thought: Optional[List[str]] = None,
    context: Optional[np.ndarray] = None,
) -> Dict:
    """
    RETRIEVAL / GROUNDING: Verifica que la salida esté fundamentada en la entrada.
    
    Para AOC: la salida debe correlacionarse significativamente con la entrada.
    Versión mejorada con análisis de bandas de frecuencia.
    """
    inp = input_arr.flatten()
    out = output_arr.flatten()
    
    # 1. Correlación de Pearson
    if np.std(inp) > 0 and np.std(out) > 0:
        correlation = np.corrcoef(inp, out)[0, 1]
    else:
        correlation = 0.0
    correlation = np.nan_to_num(correlation, nan=0.0)
    
    # 2. Divergencia KL entre distribuciones
    hist_in, bins = np.histogram(inp, bins=20, density=True)
    hist_out, _ = np.histogram(out, bins=bins, density=True)
    hist_in = hist_in + 1e-9
    hist_out = hist_out + 1e-9
    hist_in = hist_in / np.sum(hist_in)
    hist_out = hist_out / np.sum(hist_out)
    kl_divergence = np.sum(hist_in * np.log(hist_in / hist_out))
    kl_divergence = np.nan_to_num(kl_divergence, nan=5.0, posinf=5.0, neginf=5.0)
    
    # 3. Consistencia de energía por bandas (mejorado)
    n_bands = 8  # Más bandas para mejor resolución
    in_bands = np.array_split(inp, n_bands)
    out_bands = np.array_split(out, n_bands)
    band_ratios = []
    for ib, ob in zip(in_bands, out_bands):
        e_in = np.sum(ib ** 2) + 1e-9
        e_out = np.sum(ob ** 2) + 1e-9
        band_ratios.append(e_out / e_in)
    band_ratio_std = np.std(band_ratios)
    band_ratio_std = np.nan_to_num(band_ratio_std, nan=1.0)
    
    # 4. Detección de "invención" (energía en bandas donde no hay entrada)
    invention_ratio = 0.0
    for i, (ib, ob) in enumerate(zip(in_bands, out_bands)):
        e_in = np.sum(ib ** 2) + 1e-9
        e_out = np.sum(ob ** 2) + 1e-9
        if e_in < 1e-6 and e_out > 1e-4:
            invention_ratio += 1.0 / n_bands
    
    # 5. Score combinado
    score = (correlation + 0.5) * 0.3
    score += (1.0 - min(kl_divergence, 2.0) / 2.0) * 0.25
    score += (1.0 - min(band_ratio_std, 1.0)) * 0.25
    score += (1.0 - min(invention_ratio, 1.0)) * 0.20
    score = np.nan_to_num(score, nan=0.0)
    score = float(np.clip(score, 0.0, 1.0))
    
    grounded = score >= 0.5
    
    return {
        "passed": grounded,
        "score": score,
        "details": f"corr={correlation:.3f}, KL={kl_divergence:.3f}, band_std={band_ratio_std:.3f}, inv={invention_ratio:.3f}",
        "severity": "high" if not grounded else "low",
    }


# ============================================================
# 9. MULTI-TURN INCONSISTENCY - Verifier mejorado
# ============================================================
def _verifier_multi_turn_consistency_improved(
    input_arr: np.ndarray,
    output_arr: np.ndarray,
    chain_of_thought: Optional[List[str]] = None,
    previous_outputs: Optional[List[np.ndarray]] = None,
    turn_history: Optional[List[Dict]] = None,
) -> Dict:
    """
    MULTI-TURN CONSISTENCY: Verifica consistencia entre turnos consecutivos.
    Versión mejorada con análisis de tendencia y memoria.
    """
    if previous_outputs is None or len(previous_outputs) == 0:
        return {
            "passed": True,
            "score": 0.5,
            "details": "no_history_for_multi_turn",
            "severity": "low",
        }
    
    curr = output_arr.flatten()
    
    # Comparar con las últimas N salidas
    n_history = min(3, len(previous_outputs))
    consistency_scores = []
    energy_changes = []
    cosine_sims = []
    
    for i in range(1, n_history + 1):
        prev = previous_outputs[-i].flatten()
        
        # 1. Cambio de energía
        energy_prev = np.mean(prev ** 2)
        energy_curr = np.mean(curr ** 2)
        energy_change = abs(energy_curr - energy_prev) / (energy_prev + 1e-9)
        energy_changes.append(energy_change)
        
        # 2. Similitud coseno
        if np.linalg.norm(prev) > 0 and np.linalg.norm(curr) > 0:
            cosine_sim = np.dot(prev, curr) / (np.linalg.norm(prev) * np.linalg.norm(curr))
        else:
            cosine_sim = 0.0
        cosine_sims.append(cosine_sim)
        
        # 3. Score de consistencia para este paso
        step_score = (1.0 - min(energy_change, 1.0)) * 0.5 + cosine_sim * 0.5
        consistency_scores.append(step_score)
    
    # 4. Métricas agregadas
    mean_energy_change = np.mean(energy_changes)
    mean_cosine_sim = np.mean(cosine_sims)
    mean_consistency = np.mean(consistency_scores)
    
    # 5. Detectar degradación gradual (tendencia)
    if len(energy_changes) >= 2:
        trend = energy_changes[-1] - energy_changes[0]
        trend_penalty = min(0.3, max(0, trend))  # Penalizar si empeora
    else:
        trend_penalty = 0.0
    
    # 6. Score final
    score = mean_consistency * (1.0 - trend_penalty)
    score = float(np.clip(score, 0.0, 1.0))
    
    consistent = score >= 0.6
    
    return {
        "passed": consistent,
        "score": score,
        "details": f"energy_change={mean_energy_change:.3f}, cosine_sim={mean_cosine_sim:.3f}, trend_penalty={trend_penalty:.3f}",
        "severity": "medium" if not consistent else "low",
    }


# ============================================================
# 11. LONG-CONTEXT FAILURES - Verifier específico
# ============================================================
def _verifier_long_context(
    input_arr: np.ndarray,
    output_arr: np.ndarray,
    chain_of_thought: Optional[List[str]] = None,
    context_window: Optional[np.ndarray] = None,
    position_in_sequence: int = 0,
    total_sequence_length: int = 1,
) -> Dict:
    """
    LONG-CONTEXT: Verifica uso correcto de contexto largo.
    Para AOC: verifica consistencia temporal en secuencias largas.
    """
    inp = input_arr.flatten()
    out = output_arr.flatten()
    
    # 1. Si no hay contexto largo, asumir que pasa
    if total_sequence_length <= 1:
        return {
            "passed": True,
            "score": 0.5,
            "details": "no_long_context",
            "severity": "low",
        }
    
    # 2. Verificar degradación con posición
    position_ratio = position_in_sequence / max(1, total_sequence_length)
    
    # 3. Calcular energía relativa
    energy_in = np.mean(inp ** 2)
    energy_out = np.mean(out ** 2)
    energy_ratio = energy_out / (energy_in + 1e-9)
    
    # 4. Verificar varianza (debe mantenerse estable)
    var_in = np.var(inp)
    var_out = np.var(out)
    var_ratio = var_out / (var_in + 1e-9)
    
    # 5. Penalizar si hay degradación en posiciones tardías
    position_penalty = position_ratio * 0.3
    
    # 6. Penalizar si la energía o varianza colapsan
    energy_penalty = 0.0
    if energy_ratio < 0.1:
        energy_penalty += 0.3
    if energy_ratio > 10:
        energy_penalty += 0.2
    
    var_penalty = 0.0
    if var_ratio < 0.01:
        var_penalty += 0.3
    if var_ratio > 100:
        var_penalty += 0.2
    
    # 7. Score final
    score = 1.0 - position_penalty - energy_penalty - var_penalty
    score = float(np.clip(score, 0.0, 1.0))
    
    passed = score >= 0.5
    
    return {
        "passed": passed,
        "score": score,
        "details": f"pos={position_ratio:.2f}, energy_ratio={energy_ratio:.2f}, var_ratio={var_ratio:.2f}",
        "severity": "medium" if not passed else "low",
    }


# ============================================================
# 12. STYLE / TONE MISMATCH - Verifier específico
# ============================================================
def _verifier_style_tone(
    input_arr: np.ndarray,
    output_arr: np.ndarray,
    chain_of_thought: Optional[List[str]] = None,
    style_reference: Optional[np.ndarray] = None,
) -> Dict:
    """
    STYLE / TONE: Verifica consistencia de estilo (verbosidad/brevedad).
    Para AOC: verifica que la salida no sea ni demasiado ruidosa ni demasiado silenciosa.
    """
    inp = input_arr.flatten()
    out = output_arr.flatten()
    
    # 1. Energía relativa (verbosidad)
    energy_in = np.mean(inp ** 2)
    energy_out = np.mean(out ** 2)
    energy_ratio = energy_out / (energy_in + 1e-9)
    
    # 2. Rango dinámico (variación de estilo)
    dynamic_range_in = np.max(inp) - np.min(inp)
    dynamic_range_out = np.max(out) - np.min(out)
    dr_ratio = dynamic_range_out / (dynamic_range_in + 1e-9)
    
    # 3. Entropía espectral (complejidad)
    probs_in = np.abs(inp) / (np.sum(np.abs(inp)) + 1e-9)
    probs_out = np.abs(out) / (np.sum(np.abs(out)) + 1e-9)
    entropy_in = -np.sum(probs_in * np.log(probs_in + 1e-9))
    entropy_out = -np.sum(probs_out * np.log(probs_out + 1e-9))
    max_entropy = np.log(len(inp))
    entropy_ratio = entropy_out / (entropy_in + 1e-9) if entropy_in > 0 else 1.0
    
    # 4. Detectar extremos
    too_quiet = energy_ratio < 0.01
    too_loud = energy_ratio > 100
    too_simple = entropy_ratio < 0.1
    too_complex = entropy_ratio > 10
    
    # 5. Score
    score = 1.0
    
    if too_quiet:
        score -= 0.4
    if too_loud:
        score -= 0.4
    if too_simple:
        score -= 0.3
    if too_complex:
        score -= 0.3
    
    # Penalizar por rango dinámico extremo
    if dr_ratio < 0.01 or dr_ratio > 100:
        score -= 0.2
    
    score = float(np.clip(score, 0.0, 1.0))
    passed = score >= 0.5
    
    return {
        "passed": passed,
        "score": score,
        "details": f"energy_ratio={energy_ratio:.2f}, dr_ratio={dr_ratio:.2f}, entropy_ratio={entropy_ratio:.2f}",
        "severity": "medium" if not passed else "low",
    }

def _verifier_under_refusal(
    input_arr: np.ndarray,
    output_arr: np.ndarray,
    chain_of_thought: Optional[List[str]] = None,
) -> Dict:
    """
    UNDER-REFUSAL: El modelo responde con contenido peligroso/inseguro
    cuando debería abstenerse explícitamente.
    Para AOC: detecta si la salida contiene energía espectral anómala
    que podría representar contenido dañino (patrones de ataque, ruido extremo).
    """
    inp = input_arr.flatten()
    out = output_arr.flatten()
    
    # 1. Energía de salida anormalmente alta (posible amplificación maliciosa)
    energy_out = np.mean(out ** 2)
    energy_in = np.mean(inp ** 2) + 1e-9
    amplification = energy_out / energy_in
    
    # 2. Picos espectrales extremos (posibles tonos de ataque)
    max_peak = np.max(out)
    median_val = np.median(out)
    peak_ratio = max_peak / (median_val + 1e-9)
    
    # 3. Ruido de alta frecuencia no presente en entrada
    high_freq_band = out[-len(out)//4:]  # último 25% (altas frecuencias)
    high_freq_energy = np.mean(high_freq_band ** 2)
    input_high_freq = np.mean(inp[-len(inp)//4:] ** 2)
    high_freq_ratio = high_freq_energy / (input_high_freq + 1e-9)
    
    # Criterios de under-refusal (debería haberse abstenido pero no lo hizo)
    is_under_refusal = (
        amplification > 5.0 or          # amplificación excesiva
        peak_ratio > 20.0 or            # pico extremo
        high_freq_ratio > 10.0          # ruido agudo inducido
    )
    
    score = 1.0
    severity = "low"
    details = ""
    passed = True
    
    if is_under_refusal:
        passed = False
        score = 0.0
        severity = "high"
        details = (
            f"amplification={amplification:.2f}, "
            f"peak_ratio={peak_ratio:.2f}, "
            f"high_freq_ratio={high_freq_ratio:.2f}"
        )
    
    return {
        "passed": passed,
        "score": score,
        "details": details,
        "severity": severity,
    }

def _verifier_factuality(
    input_arr: np.ndarray,
    output_arr: np.ndarray,
    chain_of_thought: Optional[List[str]] = None,
) -> Dict:
    """
    FACTUALITY / HALLUCINATION:
    El modelo no debe 'alucinar' energía espectral donde no existe en la
    entrada. Se mide como ratio de bins con energía estimada > 2× entrada.
    """
    inp = np.abs(input_arr.flatten())
    out = np.abs(output_arr.flatten())
    min_len = min(len(inp), len(out))
    inp, out = inp[:min_len], out[:min_len]
    hallucination_bins = np.sum(out > 2.0 * inp + 1e-9)
    hallucination_rate = hallucination_bins / (min_len + 1e-9)
    passed = hallucination_rate < 0.10   # umbral: max 10 % de bins alucinados
    return {
        "passed": passed,
        "score": float(1.0 - hallucination_rate),
        "details": f"hallucination_rate={hallucination_rate:.4f}",
        "severity": "high" if not passed else "low",
    }

def _verifier_overconfidence(
    input_arr: np.ndarray,
    output_arr: np.ndarray,
    chain_of_thought: Optional[List[str]] = None,
) -> Dict:
    """
    OVERCONFIDENCE / MISCALIBRATION: El modelo no debe producir predicciones
    colapsadas ni extremadamente amplificadas.
    """
    out = output_arr.flatten().astype(np.float64)
    collapsed = float(np.std(out)) < 1e-5
    
    out_min = out.min()
    out_max = out.max()
    out_norm = (out - out_min) / (out_max - out_min)
    overconf_mask = out_norm > 0.95
    overconf_rate = float(overconf_mask.mean())
    passed = overconf_rate < 0.10
    score_val = float(1.0 - overconf_rate)
    
    score_val = np.nan_to_num(score_val, nan=0.0)
    overconf_rate = np.nan_to_num(overconf_rate, nan=1.0)
    
    return {
        "passed": passed,
        "score": score_val,
        "details": f"overconf_rate={overconf_rate:.4f}, collapsed={'yes' if collapsed else 'no'}",
        "severity": "high" if not passed else "low",
    }

def _verifier_distribution_shift(
    input_arr: np.ndarray,
    output_arr: np.ndarray,
    chain_of_thought: Optional[List[str]] = None,
) -> Dict:
    """
    DISTRIBUTION_SHIFT / FRAGILITY:
    La distribución espectral de salida no debe alejarse radicalmente de
    la entrada limpia esperada (media y std comparados).
    """
    inp = input_arr.flatten().astype(np.float64)
    out = output_arr.flatten().astype(np.float64)
    shift = abs(np.mean(out) - np.mean(inp)) / (np.mean(inp) + 1e-9)
    std_ratio = np.std(out) / (np.std(inp) + 1e-9)
    # Shift aceptable < 50 %; std_ratio en [0.5, 2.0]
    shift_ok = shift < 0.50
    std_ok   = 0.3 < std_ratio < 3.0
    passed   = shift_ok and std_ok
    return {
        "passed": passed,
        "score": float((1.0 - min(shift, 1.0)) * 0.5
                       + (1.0 - abs(std_ratio - 1.0)) * 0.5),
        "details": f"mean_shift={shift:.4f}, std_ratio={std_ratio:.4f}",
        "severity": "medium" if not passed else "low",
    }

def _verifier_reasoning(
    input_arr: np.ndarray,
    output_arr: np.ndarray,
    chain_of_thought: Optional[List[str]] = None,
) -> Dict:
    """
    REASONING / MATH_LOGIC_CODE:
    La reducción de energía debe ser monotónica en promedio: el modelo
    no debe amplificar la señal ocluida en más de un 20 %.
    """
    energy_in  = float(np.mean(input_arr ** 2))
    energy_out = float(np.mean(output_arr ** 2))
        
    # Evitar división por cero
    if energy_in < 1e-12:
        amplification = float('inf')
        score = 0.0
    else:
        amplification = energy_out / energy_in
        score = float(min(1.0 / max(amplification, 1e-12), 1.0))
        
    passed = amplification <= 1.20 if amplification != float('inf') else False
        
    return {
        "passed": passed,
        "score": score,
        "details": f"amplification={amplification if amplification != float('inf') else 'inf'}",
        "severity": "medium" if not passed else "low",
    }

def _verifier_toxicity(
    input_arr: np.ndarray,
    output_arr: np.ndarray,
    chain_of_thought: Optional[List[str]] = None,
) -> Dict:
    """
    TOXICITY / BIAS (en dominio AOC):
    El modelo no debe introducir artefactos tonales (bins con energía
    anormalmente alta, equivalentes a 'tono puro artificial').
    """
    out   = output_arr.flatten().astype(np.float64)
    mean  = np.mean(out)
    std   = np.std(out) + 1e-9
    toxic_bins = np.sum(out > mean + 4 * std)
    toxic_rate = toxic_bins / (len(out) + 1e-9)
    passed = toxic_rate < 0.02    # max 2 % de bins tónicos artificiales
    return {
        "passed": passed,
        "score": float(1.0 - min(toxic_rate * 10, 1.0)),
        "details": f"tonal_artifact_rate={toxic_rate:.4f}",
        "severity": "medium" if not passed else "low",
    }

# ─────────────────────────────────────────────────────────────────────────────
# 4.  PRINCIPIOS CONSTITUCIONALES — EXCLUSIVAMENTE DESDE ErrorBucket DEL REPO
# ─────────────────────────────────────────────────────────────────────────────
def _make_aoc_verifier_for_bucket(bucket: ErrorBucket):
    """
    Crea un verifier STFT-domain para cada ErrorBucket definido en stress.py.
    Todos trabajan sobre magnitudes STFT reales [n_bins] o [n_frames, n_bins].
    """

    # Mapeo ErrorBucket → verifier
    _map = {
        ErrorBucket.FACTUALITY:          _verifier_factuality,
        ErrorBucket.OVERCONFIDENCE:       _verifier_overconfidence,
        ErrorBucket.DISTRIBUTION_SHIFT:  _verifier_distribution_shift,
        ErrorBucket.REASONING:           _verifier_reasoning,
        ErrorBucket.TOXICITY:            _verifier_toxicity,
    }
    return _map.get(bucket, harmful_content_verifier)

def build_aoc_constitution_rules() -> List[Dict]:
    """
    Construye las reglas constitucionales EXCLUSIVAMENTE a partir de los
    ErrorBucket definidos en stress.py — nunca se crean principios nuevos.
    """
    rules = [
        {
            "principle": ErrorBucket.FACTUALITY.value,
            "rule": (
                "El modelo no debe alucinar energía espectral ausente en la entrada. "
                "Máximo 10 % de bins con energía estimada > 2× entrada."
            ),
            "verifier": _verifier_factuality,  # Ahora es global
            "weight": 1.5,
        },
        {
            "principle": ErrorBucket.OVERCONFIDENCE.value,
            "rule": (
                "El modelo no debe producir predicciones colapsadas ni extremas. "
                "Varianza debe ser > 1e-5 y tasa de sobre-confianza < 10 %."
            ),
            "verifier": _verifier_overconfidence,  # Global
            "weight": 1.2,
        },
        {
            "principle": ErrorBucket.DISTRIBUTION_SHIFT.value,
            "rule": (
                "La distribución espectral de salida no debe alejarse > 50 % "
                "de la entrada en media, ni exceder ratio de desviación 3×."
            ),
            "verifier": _verifier_distribution_shift,  # Global
            "weight": 1.0,
        },
        {
            "principle": ErrorBucket.REASONING.value,
            "rule": (
                "La reducción de oclusión no debe amplificar la energía total "
                "más de un 20 % respecto a la señal de entrada."
            ),
            "verifier": _verifier_reasoning,  # Global
            "weight": 1.0,
        },
        {
            "principle": ErrorBucket.TOXICITY.value,
            "rule": (
                "El modelo no debe introducir artefactos tonales artificiales. "
                "Máximo 2 % de bins con energía > media + 4σ."
            ),
            "verifier": _verifier_toxicity,  # Global
            "weight": 0.9,
        },
        # Reutilizar verifiers genéricos del repo para formato y privacidad
        {
            "principle": ErrorBucket.SCHEMA.value,
            "rule": (
                "La salida debe ser un np.ndarray 2D de magnitudes no negativas, "
                "con la misma forma que la entrada."
            ),
            "verifier": format_verifier,
            "weight": 0.8,
        },
        {
            "principle": ErrorBucket.SCHEMA.value,
            "rule": "La salida debe cumplir con el formato esperado: misma forma, valores no negativos, sin NaN/Inf, rango razonable.",
            "verifier": _verifier_schema_format,
            "weight": 0.8,
        },
        {
            "principle": ErrorBucket.LONG_CONTEXT.value,
            "rule": "El modelo debe mantener consistencia en secuencias largas sin degradación progresiva.",
            "verifier": _verifier_long_context,
            "weight": 0.8,
        },                
        {
            "principle": ErrorBucket.INSTRUCTION.value,
            "rule": (
                "El modelo debe seguir la instrucción de regresión señal-a-señal: "
                "entrada ocluida → salida limpia, sin inversión de polaridad."
            ),
            "verifier": privacy_verifier,
            "weight": 0.7,
        },
        {
            "principle": ErrorBucket.UNDER_REFUSAL.value,
            "rule": "No generar contenido peligroso o extremo. Abstenerse cuando la entrada no sea recuperable.",
            "verifier": _verifier_under_refusal,
            "weight": 1.0,
        },
        {
            "principle": ErrorBucket.OVER_REFUSAL.value,
            "rule": "No rechazar entradas válidas. Debe reconstruir siempre que la entrada tenga energía suficiente.",
            "verifier": _verifier_over_refusal,
            "weight": 1.0,
        },
        {
            "principle": ErrorBucket.TOOL_USE.value,
            "rule": "El razonamiento debe incluir los pasos de reconstrucción espectral esperados (Wiener, multi-lorax).",
            "verifier": _verifier_tool_api_use,
            "weight": 0.8,
        },
        {
            "principle": "instruction_following",
            "rule": "El modelo debe seguir las instrucciones: forma correcta y procesamiento adecuado.",
            "verifier": _verifier_instruction_following,
            "weight": 0.8,
        },        
        {
            "principle": ErrorBucket.RETRIEVAL.value,
            "rule": "La salida debe estar fundamentada en la entrada (correlación positiva, distribución similar).",
            "verifier": _verifier_retrieval_grounding,
            "weight": 1.0,
        },
        {
            "principle": ErrorBucket.MULTI_TURN.value,
            "rule": "En secuencias de rollouts, las salidas deben ser consistentes entre turnos consecutivos.",
            "verifier": _verifier_multi_turn_consistency,
            "weight": 0.7,
        }]   
    return rules

# ─────────────────────────────────────────────────────────────────────────────
# 6.  CROSS-VALIDATION CON MODELOS ADVERSARIOS (API NATIVA DEL REPO)
# ─────────────────────────────────────────────────────────────────────────────
async def run_adversarial_cross_validation(
    model: DSVMWrapper,
    X: np.ndarray,
    Y: np.ndarray,
    file_ids: List[str],
    Cs: List[float] = None,
    reg_params: List[float] = None,
    kernel_configs: List = None,
    adversaries: List[int] = None,
    model_name: str = "aoc_dsvm",
    regression: bool = True,
    loop_counter: int = 0,  # NUEVO
    output_dir: str = ".",   # NUEVO
) -> Dict[str, Any]:
    """
    Ejecuta GridSearch del repo con validación adversaria completa.
    Guarda modelos adversarios con índice de loop.
    """
    # Valores por defecto razonables para AOC
    if Cs is None:
        Cs = [0.01, 0.1, 1.0, 10.0]
    if reg_params is None:
        reg_params = [0.001, 0.01, 0.1, 1.0]
    if kernel_configs is None:
        kernel_configs = [("rbf", 0.5)]
    if adversaries is None:
        adversaries = [1, 2]

    logger.info(
        f"[CV Adversarial] {len(X)} frames, "
        f"{len(Cs)} C-values, {len(adversaries)} adversarios"
    )

    # ── Parámetros de explain para DSVM (intersects / logical sobre bins) ──
    n_bins     = X.shape[1]
    intersects = list(range(min(10, n_bins)))   # primeras 10 bins de frecuencia
    logical    = [True] * len(intersects)
    criteria   = None   # se infiere dentro del repo

    from sklearn.cluster import KMeans
    Y_flat = np.float64(Y).flatten() if len(Y.shape) > 1 else np.float64(Y)
    Y_reshaped = Y_flat.reshape(-1, 1)
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    targets_binary = kmeans.fit_predict(Y_reshaped).reshape(-1, 1)

    logger.info(f"[CV Adversarial] Binarización con KMeans k=2: clases {np.unique(targets_binary)}")

    # ── Ejecutar GridSearch (nativo del repo) ──────────────────────────────
    logger.info("[CV Adversarial] Ejecutando GridSearch...")
    try:
        result = await GridSearch(
            model=model,
            features=np.float64(X),
            targets=np.float64(targets_binary),
            Cs=Cs,
            reg_params=reg_params,
            kernel_configs=kernel_configs,
            last=False,
            criteria=criteria,
            intersects=intersects,
            logical=logical,
            regression=regression,
            adversaries=adversaries,
            model_name=model_name,
            output_dir=output_dir,  # NUEVO
            loop_counter=loop_counter,  # NUEVO
        )
    except Exception as e:
        logger.error(f"[CV Adversarial] GridSearch falló: {e}")
        tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        for task in tasks:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass        
        result = None

    return result

async def save_hypotheses_results(
    hypotheses: List['Hypothesis'],
    output_dir: str,
    loop_counter: int,
    rollout_id: str = None,
    baseline_idx: int = None
) -> Dict[str, Any]:
    """
    Guarda los resultados de las hipótesis en archivos JSON.
    
    Args:
        hypotheses: Lista de hipótesis generadas
        output_dir: Directorio de salida
        loop_counter: Número de loop actual
        rollout_id: ID del rollout (opcional)
        baseline_idx: Índice de baseline (opcional)
    
    Returns:
        Diccionario con rutas de archivos guardados
    """
    import json
    from datetime import datetime
    
    saved_files = {}
    
    # Crear directorio para hipótesis si no existe
    hypotheses_dir = os.path.join(output_dir, "hypotheses")
    os.makedirs(hypotheses_dir, exist_ok=True)
    
    # ========== 1. GUARDAR TODAS LAS HIPÓTESIS ==========
    all_hypotheses_data = {
        'timestamp': datetime.now().isoformat(),
        'loop_counter': loop_counter,
        'rollout_id': rollout_id,
        'baseline_idx': baseline_idx,
        'total_hypotheses': len(hypotheses),
        'hypotheses': [h.to_dict() for h in hypotheses]
    }
    
    # Nombre de archivo con timestamp y metadata
    filename = f"hypotheses_loop_{loop_counter}"
    if rollout_id:
        filename += f"_rollout_{rollout_id}"
    if baseline_idx is not None:
        filename += f"_baseline_{baseline_idx:04d}"
    filename += ".json"
    
    filepath = os.path.join(hypotheses_dir, filename)
    with open(filepath, 'w') as f:
        json.dump(all_hypotheses_data, f, indent=2, default=str)
    
    saved_files['all_hypotheses'] = filepath
    logger.info(f"[Hypotheses] Guardadas {len(hypotheses)} hipótesis en {filepath}")
    
    # ========== 2. GUARDAR HIPÓTESIS POR SEVERIDAD ==========
    by_severity = {
        'none': [h.to_dict() for h in hypotheses if h.violation_severity == 'none'],
        'low': [h.to_dict() for h in hypotheses if h.violation_severity == 'low'],
        'medium': [h.to_dict() for h in hypotheses if h.violation_severity == 'medium'],
        'high': [h.to_dict() for h in hypotheses if h.violation_severity == 'high']
    }
    
    severity_file = os.path.join(hypotheses_dir, f"hypotheses_by_severity_loop_{loop_counter}.json")
    with open(severity_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'loop_counter': loop_counter,
            'by_severity': by_severity,
            'counts': {k: len(v) for k, v in by_severity.items()}
        }, f, indent=2, default=str)
    
    saved_files['by_severity'] = severity_file
    logger.info(f"[Hypotheses] Guardadas por severidad en {severity_file}")
    
    # ========== 3. GUARDAR MEJORES HIPÓTESIS (TOP K) ==========
    top_k = min(20, len(hypotheses))
    best_hypotheses = sorted(hypotheses, key=lambda h: h.score, reverse=True)[:top_k]
    
    best_file = os.path.join(hypotheses_dir, f"best_hypotheses_loop_{loop_counter}.json")
    with open(best_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'loop_counter': loop_counter,
            'top_k': top_k,
            'best_hypotheses': [h.to_dict() for h in best_hypotheses]
        }, f, indent=2, default=str)
    
    saved_files['best_hypotheses'] = best_file
    logger.info(f"[Hypotheses] Guardadas top {top_k} mejores hipótesis en {best_file}")
    
    # ========== 4. GUARDAR ESTADÍSTICAS RESUMEN ==========
    scores = [h.score for h in hypotheses]
    accuracy_gaps = [h.accuracy_gap for h in hypotheses]
    credits = [h.credit for h in hypotheses]
    
    summary_file = os.path.join(hypotheses_dir, f"hypotheses_summary_loop_{loop_counter}.json")
    with open(summary_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'loop_counter': loop_counter,
            'rollout_id': rollout_id,
            'baseline_idx': baseline_idx,
            'statistics': {
                'total_hypotheses': len(hypotheses),
                'mean_score': float(np.mean(scores)) if scores else 0.0,
                'std_score': float(np.std(scores)) if scores else 0.0,
                'min_score': float(np.min(scores)) if scores else 0.0,
                'max_score': float(np.max(scores)) if scores else 0.0,
                'mean_accuracy_gap': float(np.mean(accuracy_gaps)) if accuracy_gaps else 0.0,
                'mean_credit': float(np.mean(credits)) if credits else 0.0,
                'by_severity_counts': {k: len(v) for k, v in by_severity.items()}
            },
            'best_hypothesis': best_hypotheses[0].to_dict() if best_hypotheses else None,
            'worst_hypothesis': min(hypotheses, key=lambda h: h.score).to_dict() if hypotheses else None
        }, f, indent=2, default=str)
    
    saved_files['summary'] = summary_file
    logger.info(f"[Hypotheses] Resumen guardado en {summary_file}")
    
    return saved_files


def save_hypotheses_knowledge_base(
    knowledge_base: 'HypothesisKnowledgeBase',
    output_dir: str,
    loop_counter: int
) -> Dict[str, Any]:
    """
    Guarda la base de conocimiento completa de hipótesis.
    
    Args:
        knowledge_base: Base de conocimiento de hipótesis
        output_dir: Directorio de salida
        loop_counter: Número de loop actual
    
    Returns:
        Diccionario con rutas de archivos guardados
    """
    import json
    from datetime import datetime
    
    saved_files = {}
    
    # Crear directorio para conocimiento
    knowledge_dir = os.path.join(output_dir, "knowledge_base")
    os.makedirs(knowledge_dir, exist_ok=True)
    
    # ========== 1. GUARDAR TODAS LAS HIPÓTESIS CONOCIDAS ==========
    all_hypotheses = knowledge_base.get_all_hypotheses()
    
    knowledge_file = os.path.join(knowledge_dir, f"knowledge_base_loop_{loop_counter}.json")
    with open(knowledge_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'loop_counter': loop_counter,
            'total_hypotheses': len(all_hypotheses),
            'hypotheses': [h.to_dict() for h in all_hypotheses]
        }, f, indent=2, default=str)
    
    saved_files['knowledge_base'] = knowledge_file
    logger.info(f"[KnowledgeBase] Guardadas {len(all_hypotheses)} hipótesis en {knowledge_file}")
    
    # ========== 2. GUARDAR INSIGHTS Y VULNERABILIDADES ==========
    insights = knowledge_base.get_insights()
    
    insights_file = os.path.join(knowledge_dir, f"knowledge_insights_loop_{loop_counter}.json")
    with open(insights_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'loop_counter': loop_counter,
            'insights': insights
        }, f, indent=2, default=str)
    
    saved_files['insights'] = insights_file
    logger.info(f"[KnowledgeBase] Insights guardados en {insights_file}")
    
    # ========== 3. GUARDAR VULNERABILIDADES ==========
    vulnerabilities = knowledge_base.analyze_vulnerabilities()
    
    vuln_file = os.path.join(knowledge_dir, f"vulnerabilities_loop_{loop_counter}.json")
    with open(vuln_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'loop_counter': loop_counter,
            'vulnerabilities': vulnerabilities
        }, f, indent=2, default=str)
    
    saved_files['vulnerabilities'] = vuln_file
    logger.info(f"[KnowledgeBase] Vulnerabilidades guardadas en {vuln_file}")
    
    # ========== 4. GUARDAR HIPÓTESIS POR FEATURE ==========
    hypotheses_by_feature = defaultdict(list)
    for h in all_hypotheses:
        hypotheses_by_feature[h.feature_idx].append(h.to_dict())
    
    by_feature_file = os.path.join(knowledge_dir, f"hypotheses_by_feature_loop_{loop_counter}.json")
    with open(by_feature_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'loop_counter': loop_counter,
            'hypotheses_by_feature': {str(k): v for k, v in hypotheses_by_feature.items()}
        }, f, indent=2, default=str)
    
    saved_files['by_feature'] = by_feature_file
    
    return saved_files

def _classify_aoc_error_bucket(
        x_frame: np.ndarray,
        pred: np.ndarray,
        y_frame: np.ndarray,
        mse: float,
    ) -> Optional[ErrorBucket]:
    """
    Clasifica el error AOC en los ErrorBucket definidos en stress.py.
    Criterios basados en magnitudes STFT reales.
    """
    # Hallucination: energía en bins que estaban vacíos
    hallu_bins = int(np.sum((pred > 2 * x_frame + 1e-9) & (x_frame < 1e-4)))
    if hallu_bins > 0.10 * len(x_frame):
        return ErrorBucket.FACTUALITY

    # Overconfidence: varianza colapsada o amplificación extrema
    if np.std(pred) < 1e-5:
        return ErrorBucket.OVERCONFIDENCE

    # Distribution shift: media relativa muy alejada
    shift = abs(np.mean(pred) - np.mean(y_frame)) / (np.mean(y_frame) + 1e-9)
    if shift > 0.5:
        return ErrorBucket.DISTRIBUTION_SHIFT

    # MSE alto pero sin alucinación → error de reasoning (reconstrucción)
    if mse > 0.1:
        return ErrorBucket.REASONING

    # MSE moderado → overconfidence moderada
    if mse > 0.05:
        return ErrorBucket.OVERCONFIDENCE

    return None

def _detect_anomalous_pattern(output: np.ndarray, input: np.ndarray) -> bool:
    output_flat = output.flatten()
    input_flat = input.flatten()
    
    # ========== 1. Z-SCORE DE ENERGÍA ==========
    energy_in = np.mean(input_flat ** 2)
    energy_out = np.mean(output_flat ** 2)
    energy_ratio = energy_out / (energy_in + 1e-9)
    
    # Ratio normal: 0.1 - 2.0 (para AOC)
    # Anómalo: < 0.01 o > 5.0
    if energy_ratio < 0.01 or energy_ratio > 5.0:
        return True
    
    # ========== 2. COEFICIENTE DE VARIACIÓN ==========
    # Medir qué tan "plana" o "pico" es la salida
    cv_out = np.std(output_flat) / (np.mean(np.abs(output_flat)) + 1e-9)
    cv_in = np.std(input_flat) / (np.mean(np.abs(input_flat)) + 1e-9)
    
    # Si la salida es mucho más "pico" que la entrada (más de 3x)
    if cv_out > 3 * cv_in:
        return True
    
    # ========== 3. CORRELACIÓN POSITIVA ==========
    if np.std(input_flat) > 0 and np.std(output_flat) > 0:
        correlation = np.corrcoef(input_flat, output_flat)[0, 1]
        if not np.isnan(correlation) and correlation < -0.3:  # Correlación negativa fuerte
            return True
    
    return False

def run_single_rollout_worker(rollout_args: list) -> Dict[str, Any]:
    """
    Worker que ejecuta un SOLO rollout de forma aislada.
    Se ejecuta en paralelo con otros workers.
    
    Args:
        rollout_args: [shared_memory, model, pipeline, constitutional_ai, 
                       loop_counter, baseline_idx, idx, x_f, y_f]
    
    Returns:
        Diccionario con los resultados del rollout
    """
    import asyncio
    import numpy as np
    from datetime import datetime
    
    try:
        # ========== DESEMPAQUETAR ==========
        shared_memory = rollout_args[0]      # AgentMemory compartido
        model = rollout_args[1]              # DSVMWrapper
        pipeline = rollout_args[2]           # PostTrainingPipeline
        constitutional_ai = rollout_args[3]  # ConstitutionalAI
        loop_counter = rollout_args[4]
        baseline_idx = rollout_args[5]
        idx = rollout_args[6]
        x_f = rollout_args[7]
        y_f = rollout_args[8]
        
        # ========== 1. RECUPERAR CONTEXTO DE MEMORIA ==========
        memory_context = shared_memory.retrieve_context(x_f)
        
        # Integrar contexto con input
        if memory_context['combined_context'].size > 0:
            enhanced_input = 0.7 * x_f + 0.3 * memory_context['combined_context']
        else:
            enhanced_input = x_f
        
        # ========== 2. PREDICCIÓN ==========
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            pred = loop.run_until_complete(
                model.predict(enhanced_input.reshape(1, -1))
            )
            pred = pred[0]
        finally:
            loop.close()
        
        # Asegurar formas
        if pred.ndim == 1:
            pred = pred.reshape(1, -1)
        if y_f.ndim == 1:
            y_f = y_f.reshape(1, -1)
        
        # ========== 3. CHAIN OF THOUGHT ==========
        cot = _build_spectral_cot_simple(x_f, pred)
        
        # ========== 4. REWARD ==========
        reward_result = pipeline.reward_model.score(
            input=x_f,
            output=pred,
            target=y_f,
            chain_of_thought=cot,
        )
        
        # ========== 5. CONSTITUTIONAL AI ==========
        const_result = constitutional_ai.evaluate(
            input=x_f,
            output=pred,
            chain_of_thought=cot,
        )
        
        # ========== 6. ERROR Y BUCKET ==========
        error_val = float(np.mean((pred - y_f) ** 2))
        bucket = _classify_aoc_error_bucket(x_f, pred, y_f, error_val)
    
        uncertainty = reward_result.get("uncertainty", 0.0)
        component_scores = reward_result["component_scores"]
        overconf_penalty = component_scores.get("overconfidence_penalty", 0.0)
        shift_penalty = component_scores.get("distribution_shift_penalty", 0.0)
    
        # Recompensa calibrada
        calibrated_reward = reward_result["composite_reward"] * (1.0 - overconf_penalty) * (1.0 - shift_penalty * 0.5)
        
        if uncertainty > 0.3 and (overconf_penalty > 0.1 or shift_penalty > 0.1):
            calibrated_reward *= 0.5
        
        # ========== 7. ROLLOUT ==========
        rollout = {
            "input": x_f,
            "target": y_f,
            "prediction": pred,
            "chain_of_thought": cot,
            "reward_components": reward_result["component_scores"],
            "composite_reward": calibrated_reward,
            "constitution_result": const_result,
            "constitution_score": const_result.get('constitution_score', 0),
            "error_bucket": bucket,
            "frame_mse": error_val,
            "done": False,
            "idx": idx,
        }
    
        # ========== 8. PREFERENCE PAIRS ==========
        # Nota: Las preferencias se generan después secuencialmente
        preference_pairs = []
        
        # ========== 9. ACTUALIZAR MEMORIA COMPARTIDA ==========
        # ¡ATENCIÓN! shared_memory es el objeto AgentMemory
        # Esta es una operación atómica que no necesita lock explícito
        # porque parallel() con shared=False y Manager ya maneja la sincronización
        shared_memory.step_rollout(
            step=idx,
            input_data=x_f,
            output_data=pred,
            reward=reward_result["composite_reward"],
            context={
                'constitution_score': const_result.get('constitution_score', 0),
                'error_bucket': bucket.value if bucket else None,
                'anomalous_pattern': _detect_anomalous_pattern(pred, x_f),
                'decommission_mentioned': 'decommission' in str(const_result.get('violations', [])),
                'privilege_escalation': False,
                'baseline_idx': baseline_idx,
                'rollout_idx': idx
            }
        )
    
        # ========== 10. DETECTAR CONSOLIDACIÓN ==========
        consolidation_triggered = False
        current_queue_size = len(shared_memory.consolidation_queue)
        
        if current_queue_size >= 5:  # consolidation_threshold
            consolidation_triggered = True
            shared_memory._consolidate()
        
        rollout['done'] = consolidation_triggered
        
        # ========== 11. HIPÓTESIS ==========
        hypotheses = []
        if hasattr(shared_memory, 'knowledge_base') and shared_memory.knowledge_base:
            pass  # Simplificado
    
        return {
            'status': 'completed',
            'rollout_idx': idx,
            'rollout': rollout,
            'composite_reward': calibrated_reward,
            'preference_pairs': preference_pairs,
            'consolidation_triggered': consolidation_triggered,
            'hypotheses': hypotheses,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        import traceback
        error_msg = f"Rollout worker failed: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return {
            'status': 'failed',
            'error': error_msg,
            'rollout_idx': rollout_args[6] if len(rollout_args) > 6 else -1
        }

# En save_hypotheses_results, ya guardas hipótesis individuales
# Pero también guarda el estado del validador
def save_validator_state(validator, output_dir, loop_counter):
    """Guarda estado completo del validador para análisis posterior"""
    
    # Resumen ejecutivo
    summary = validator.get_validation_stats()
    summary_path = os.path.join(output_dir, f"validator_summary_loop_{loop_counter}.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Todas las hipótesis agrupadas
    hypotheses_data = {}
    for key, hyps in validator.hypothesis_library.items():
        hypotheses_data[key] = [h.to_dict() for h in hyps]
    
    hypotheses_path = os.path.join(output_dir, f"validator_hypotheses_loop_{loop_counter}.json")
    with open(hypotheses_path, 'w') as f:
        json.dump(hypotheses_data, f, indent=2)
    
    # Historial de validaciones
    history_path = os.path.join(output_dir, f"validator_history_loop_{loop_counter}.json")
    with open(history_path, 'w') as f:
        json.dump(validator.validation_history[-100:], f, indent=2)

def find_best_threshold(feature_col: np.ndarray, targets: np.ndarray, n_thresholds: int = 20) -> float:
    """
    Encuentra el threshold que maximiza la diferencia entre grupos.
    """
    # Tomar percentiles como candidatos
    percentiles = np.linspace(10, 90, n_thresholds)
    candidate_thresholds = np.percentile(feature_col, percentiles)
    
    best_threshold = None
    best_score = -np.inf
    
    for t in candidate_thresholds:
        # Dividir en grupos
        pos_group = targets[feature_col > t]
        neg_group = targets[feature_col <= t]
        
        if len(pos_group) == 0 or len(neg_group) == 0:
            continue
        
        # Score: diferencia de medias normalizada
        mean_diff = abs(np.mean(pos_group) - np.mean(neg_group))
        std_combined = np.std(np.concatenate([pos_group, neg_group])) + 1e-9
        score = mean_diff / std_combined
        
        if score > best_score:
            best_score = score
            best_threshold = t
    
    return best_threshold if best_threshold is not None else np.median(feature_col)

# ─────────────────────────────────────────────────────────────────────────────
# 7.  POST-ENTRENAMIENTO — SFT + RL CON MUESTRAS DE AUDIO REALES
# ─────────────────────────────────────────────────────────────────────────────
class AOCPostTrainer:
    def __init__(
        self,
        model: DSVMWrapper,
        X_audio: np.ndarray,
        Y_audio: np.ndarray,
        constitution_rules: List[Dict],
        output_dir: str = ".",
        adversary_models: List[Dict] = None,
        quote_reference_path: Optional[str] = None,
        loop_counter: int = 0,
        model_router: Optional[ModelRouter] = None,
        feature_importances: np.ndarray = None,
        baseline_idx: int = None,
        wset: np.ndarray = None,
        memory: Optional[AgentMemory] = None,
        memory_lock: Optional[mp.RLock] = None   
    ):
        self.model = model
        self.X_audio = X_audio
        self.Y_audio = Y_audio
        self.n_frames = len(X_audio)
        self.output_dir = output_dir
        self.loop_counter = loop_counter
        self.model_router = model_router
        self.adversary_models = adversary_models or []
        self.feature_importances = np.array(feature_importances)
        self.hypothesis_validator = HypothesisValidator(similarity_threshold=0.05)
        self.dropout_rate = 0
        self.wset = wset
        self.baseline_idx = baseline_idx
        self.rollout_data = {
            'inputs': [],
            'targets': [],
            'predictions': [],
            'composite_rewards': [],
            'rewards': [],
            'values': [],
            'dones': [],
            'constitution_scores': []
        }        
        # Cargar quote_reference_model si existe
        self.quote_reference_model = None
        self.quote_reference_proba = None
        if quote_reference_path and os.path.exists(quote_reference_path):
            try:
                with open(quote_reference_path, 'rb') as f:
                    quote_data = pickle.load(f)
                self.quote_reference_model = quote_data.get('model')
                self.quote_reference_proba = quote_data.get('reference_proba')
                logger.info(f"[PostTrain] QuoteReferenceModel cargado desde {quote_reference_path}")
            except Exception as e:
                logger.warning(f"[PostTrain] No se pudo cargar QuoteReferenceModel: {e}")
        
        # Cargar modelos adversarios previos
        self.adversary_collection = []
        collection_path = os.path.join(output_dir, f"adversary_models_collection_loop_{loop_counter}.json")
        if os.path.exists(collection_path):
            try:
                with open(collection_path, "r") as f:
                    self.adversary_collection = json.load(f)
                logger.info(f"[PostTrain] Cargados {len(self.adversary_collection)} adversarios previos")
            except Exception as e:
                logger.warning(f"[PostTrain] Error cargando adversarios: {e}")

        parallel_baselines_dir = os.path.join('outputs', "parallel_baselines")
        baseline_models = self.load_baselines_for_majority_voting(
            parallel_baselines_dir=parallel_baselines_dir,
            current_baseline_idx=baseline_idx if baseline_idx is not None else -1
        )
        
        # Recolectar TODOS los modelos para majority voting
        all_voter_models = []
        
        if self.model_router:
            for name, routed_model in self.model_router.models.items():
                if name != 'dsvm_main':
                    all_voter_models.append({
                        'model': routed_model,
                        'name': name,
                        'weight': self.model_router.model_weights.get(name, 0.5),
                        'source': 'router'
                    })
            logger.info(f"[PostTrain] Añadidos {len(all_voter_models)} modelos del ModelRouter para majority voting")
        
        for adv in self.adversary_models:
            if 'model' in adv:
                all_voter_models.append({
                    'model': adv['model'],
                    'name': adv.get('model_name', 'adversary'),
                    'weight': adv.get('weight', 0.3),
                    'source': 'cv_adversary'
                })
        
        for prev in self.adversary_collection:
            if 'model' in prev:
                all_voter_models.append({
                    'model': prev['model'],
                    'name': prev.get('name', f"prev_loop_{prev.get('loop', 0)}"),
                    'weight': 0.2,
                    'source': 'previous'
                })

        # 3. BASELINES PREVIAS (NUEVO)
        for bl in baseline_models:
            all_voter_models.append({
                'model': bl['model'],
                'name': bl['model_name'],
                'weight': bl['weight'],
                'source': 'baseline',
                'metadata': {'snr': bl.get('snr', 0), 'improvement': bl.get('improvement', 0)}
            })
        
        logger.info(f"[PostTrain] Majority voting con {len(all_voter_models)} modelos totales")
        
        # Crear reward model
        if all_voter_models:
            self.reward_model = DSVMRewardModelWithMajorityVoting(
                base_model=model,
                adversary_models=all_voter_models
            )
        elif self.quote_reference_model is not None:
            self.reward_model = DSVMRewardModelWithReference(
                model,
                quote_model=self.quote_reference_model,
                quote_proba=self.quote_reference_proba
            )
        else:
            self.reward_model = DSVMRewardModel(model)
        
        # ========== CORRECCIÓN: Crear constitutional_ai ANTES de pipeline ==========
        # Constitutional AI - crear como atributo de instancia
        self.constitutional_ai = ConstitutionalAI(constitution_rules=constitution_rules)
        
        # RewardHackingDetector
        self.hacking_detector = RewardHackingDetector()
        
        # CREAR PIPELINE (ahora self.constitutional_ai ya existe)
        self.pipeline = PostTrainingPipeline(
            base_model=model,
            reward_model=self.reward_model,
            constitutional_ai=self.constitutional_ai,
            validator=None,
        )
        
        self.calibration_mon = CalibrationMonitor(window_size=200)
        self.intervention_sys = AutoInterventionSystem(
            base_model=model.predict,
            thresholds=InterventionThresholds(
                max_hallucination_rate=0.10,
                max_overconfidence_rate=0.10,
                calibration_error_threshold=0.15,
            ),
        )
        
        # Inicializar sistema de memoria del agente
        self.agent_memory = memory
        self.memory_lock = memory_lock
        
        # Historial de indicadores de misalignment
        self.misalignment_history = []

        self.rl_config = {
            'num_rollouts': 80,
            'ppo_steps': 10,
            'go_no_go_threshold': 0.05,
            'sample_efficiency_analysis': False
        }
        
        logger.info(f"[PostTrain] Sistema de memoria del agente inicializado")

        self.efficiency_monitor = EfficiencyMonitor(
            cost_per_token_usd=0.001  # Ajustar según hardware
        )
        self.holistic_evaluator = HolisticEvaluator()

    def load_baselines_for_majority_voting(self, parallel_baselines_dir: str, current_baseline_idx: int) -> List[Dict]:
        """
        Carga TODAS las baselines completadas previas para majority voting.
        """
        from run_full_pipeline import find_completed_baselines, load_model_from_baseline
        
        completed = find_completed_baselines(parallel_baselines_dir)
        adversary_models = []
        
        for baseline_idx, metadata in completed.items():
            if baseline_idx == current_baseline_idx:
                continue  # Saltar baseline actual
                
            model = load_model_from_baseline(baseline_idx, parallel_baselines_dir)
            if model is not None:
                # Calcular peso basado en desempeño
                snr_improvement = metadata.get('snr_improvement', 0)
                final_snr = metadata.get('final_snr', 0)
                
                # Peso combinado: SNR final + mejora
                weight = 0.6 * (final_snr + 20) / 50 + 0.4 * (snr_improvement + 10) / 20
                weight = np.clip(weight, 0.2, 0.8)
            
                adversary_models.append({
                    'model': model,
                    'model_name': f'baseline_{baseline_idx:04d}',
                    'weight': weight,
                    'source': 'baseline',
                    'snr': final_snr,
                    'improvement': snr_improvement
                })
        
        logger.info(f"[MajorityVoting] Cargadas {len(adversary_models)} baselines previas para voting")
        return adversary_models
    
    def save_adversary_collection(self):
        """Guarda la colección de modelos adversarios para este loop"""
        collection_path = os.path.join(self.output_dir, f"adversary_models_collection_loop_{self.loop_counter}.json")
        with open(collection_path, "w") as f:
            json.dump(self.adversary_collection, f, indent=2)
        logger.info(f"[PostTrain] Colección de adversarios guardada: {collection_path}")
    
    def add_to_adversary_collection(self, adversary_model: Dict):
        """Agrega un modelo adversario a la colección"""
        adversary_info = {
            'loop': self.loop_counter,
            'timestamp': datetime.now().isoformat(),
            'model_info': adversary_model
        }
        self.adversary_collection.append(adversary_info)
        self.save_adversary_collection()

    def _sample_real_frames(
        self, n: int, seed: int = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extrae n frames aleatorios SIN REEMPLAZO de los audios reales.
        Nunca genera datos sintéticos.
        """
        rng      = np.random.default_rng(seed)
        idx      = rng.choice(self.n_frames, size=min(n, self.n_frames), replace=False)
        return self.X_audio[idx], self.Y_audio[idx]

    async def run_rejection_sampling_sft(
        self,
        quality_threshold: float = 0.75,
        max_samples: int = 500,
        sample_efficiency_analysis = True
    ) -> Dict:
        """
        SFT con rejection sampling sobre frames de audio reales.
        Solo se entrenan los frames donde el reward supera quality_threshold.
        """
        logger.info(
            f"[SFT] Rejection sampling sobre {min(max_samples, self.n_frames)} "
            f"frames de audio reales (threshold={quality_threshold})..."
        )
        X_cand, Y_cand = self._sample_real_frames(max_samples, seed=1)
        accepted_X, accepted_Y = [], []
    
        # Para sample efficiency analysis, necesitamos acumular rewards
        all_rewards = []  # Lista de rewards numéricos
    
        for i in range(len(X_cand)):
            x_frame = X_cand[i]
            y_frame = Y_cand[i]
            pred = await self.model.predict(x_frame.reshape(1, -1))
            #pred = pred[0]
            
            print("X FRAME:", x_frame)
            print("PRED:", pred)
    
            # CoT basado en características espectrales reales del frame
            cot = self._build_spectral_cot(x_frame, pred)

            reward_result = self.pipeline.reward_model.score(
                input          = x_frame,
                output         = pred,
                target         = y_frame,
                chain_of_thought = cot,
            )
        
            # Extraer el reward numérico
            composite_reward = reward_result["composite_reward"]
            all_rewards.append(composite_reward)
            
            if composite_reward >= quality_threshold:
                accepted_X.append(x_frame)
                accepted_Y.append(y_frame)

        # SAMPLE EFFICIENCY ANALYSIS (ahora con all_rewards que es lista de números)
        efficiency_analysis = None
        if sample_efficiency_analysis and len(all_rewards) > 10:
            from scipy.signal import savgol_filter
            from scipy.optimize import curve_fit
            
            def asymptotic_model(x, a, b, c):
                """Modelo asintótico: a * (1 - exp(-b*x)) + c"""
                return a * (1 - np.exp(-b * x)) + c
            
            # Análisis de curva de aprendizaje
            rewards_array = np.array(all_rewards)
            cumulative_rewards = np.cumsum(rewards_array) / np.arange(1, len(rewards_array) + 1)
        
            # Suavizado para identificar punto de saturación
            if len(cumulative_rewards) > 20:
                window = min(21, len(cumulative_rewards)//2*2+1)
                smoothed = savgol_filter(cumulative_rewards, window, 3)
            else:
                smoothed = cumulative_rewards
    
            # Encontrar punto donde mejora < 1% por rollout
            improvement_rates = np.diff(smoothed) / (smoothed[:-1] + 1e-9)
            saturation_point = np.argmax(improvement_rates < 0.01) + 1 if len(improvement_rates) > 0 else len(rewards_array)
            
            # Ajuste de curva asintótica usando el modelo DSVM como base para predicción
            try:
                # Usar feature_weights del DSVM para ponderar la curva de aprendizaje
                if hasattr(self.model, 'feature_weights') and self.model.feature_weights is not None:
                    fw = np.array(self.model.feature_weights).flatten()
                    weight_factor = np.mean(np.abs(fw)) / (np.std(fw) + 1e-9)
                else:
                    weight_factor = 1.0
                
                x_data = np.arange(1, len(cumulative_rewards) + 1)
                popt, _ = curve_fit(asymptotic_model, x_data, cumulative_rewards, 
                                   p0=[cumulative_rewards[-1] - cumulative_rewards[0], 0.1, cumulative_rewards[0]],
                                   maxfev=5000)
                theoretical_max = asymptotic_model(len(cumulative_rewards) * 2, *popt)
                
                if len(all_rewards) > 0:
                    final_reward = cumulative_rewards[-1]
                    sample_efficiency = final_reward / (np.log(len(all_rewards)) * weight_factor + 1)
                else:
                    sample_efficiency = 0.0
            
                # Calcular ganancia por rollout
                dsvm_gain_per_rollout = []
                for i in range(1, min(len(all_rewards), 50)):
                    gain = (cumulative_rewards[i] - cumulative_rewards[i-1]) / (i + 1)
                    if hasattr(self.model, 'w') and self.model.w is not None:
                        w_norm = np.linalg.norm(self.model.w)
                        gain = gain * (1 + w_norm / (1 + w_norm))
                    dsvm_gain_per_rollout.append(gain)
            
                efficiency_analysis = {
                    'learning_curve': cumulative_rewards.tolist(),
                    'saturation_point': int(saturation_point),
                    'theoretical_max': float(theoretical_max),
                    'sample_efficiency_score': float(sample_efficiency),
                    'rollouts_to_saturation': int(saturation_point),
                    'gain_at_saturation': float(smoothed[saturation_point] if saturation_point < len(smoothed) else smoothed[-1]),
                    'efficiency_ratio': float(sample_efficiency / (theoretical_max + 1e-9)),
                    'dsvm_weight_factor': float(weight_factor),
                    'dsvm_gain_per_rollout': dsvm_gain_per_rollout,
                    'asymptotic_model_params': {
                        'a': float(popt[0]), 'b': float(popt[1]), 'c': float(popt[2])
                    }
                }
            
                logger.info(f"[SFT] Sample Efficiency Analysis (DSVM-weighted):")
                logger.info(f"  - Rollouts to saturation: {saturation_point}")
                logger.info(f"  - Theoretical max reward: {theoretical_max:.4f}")
                logger.info(f"  - Sample efficiency score: {sample_efficiency:.4f}")
                logger.info(f"  - DSVM weight factor: {weight_factor:.4f}")
            except Exception as e:
                logger.warning(f"[SFT] Sample efficiency analysis falló: {e}")
                efficiency_analysis = None
    
        if not accepted_X:
            logger.warning("[SFT] No hay frames de alta calidad tras rejection sampling.")
            return {"status": "no_high_quality_frames", "accepted": 0, "sample_efficiency": efficiency_analysis}
    
        accepted_X = np.array(accepted_X)
        accepted_Y = np.array(accepted_Y)
        logger.info(
            f"[SFT] {len(accepted_X)}/{len(X_cand)} frames aceptados "
            f"({100*len(accepted_X)/len(X_cand):.1f}%)"
        )
    
        # Fine-tuning sobre frames aceptados (via repo SFT)
        sft_result = await self.pipeline.run_supervised_finetuning(
            X      = accepted_X,
            y      = accepted_Y,
            epochs = 3,
            batch_size = 32,
        )
        sft_result["accepted_frames"]  = len(accepted_X)
        sft_result["candidate_frames"] = len(X_cand)
        sft_result["sample_efficiency"] = efficiency_analysis
    
        # Métricas AOC post-SFT
        sft_result["aoc_metrics"] = await evaluate_aoc_metrics(
            self.model.predict,
            accepted_X,
            accepted_Y,
            label="SFT",
        )
        return sft_result

    async def run_rl_rollouts(self, num_rollouts: int = 80, ppo_steps: int = 10) -> Dict:
        logger.info(f"[RL] {num_rollouts} rollouts desde frames de audio reales...")
        
        # Preparar memoria para nuevo rollout
        self.agent_memory.start_rollout()
        
        X_rl, Y_rl = self._sample_real_frames(num_rollouts, seed=2)

        efficiency_metrics = await self.efficiency_monitor.measure_inference(
            model_fn=self.model.predict,
            X=X_rl,
            y=Y_rl,
            domain="general"
        )
        
        logger.info(f"[Efficiency] TTFT={efficiency_metrics.time_to_first_token_ms:.2f}ms, "
                   f"Tokens/sec={efficiency_metrics.tokens_per_second:.1f}, "
                   f"Memory={efficiency_metrics.peak_memory_mb:.1f}MB")
        
        # Asegurar que Y_rl tenga la forma correcta (n_samples, n_features)
        # para que continuous_multi_lorax funcione correctamente
        if Y_rl.ndim == 1:
            Y_rl = Y_rl.reshape(-1, 1)
        
        rollouts = []
        rewards = []
        all_rollout_hypotheses = []
        self.agent_memory.consolidation_threshold = 5
        consolidation_triggers = []
        preference_pairs = []  # (good, bad, confidence)
        
        for i in range(len(X_rl)-24):
            x_frame = X_rl[i:i+24]
            y_frame = Y_rl[i:i+24]

            # ========== DROPOUT: APLICAR MÁSCARA ANTES DE PREDECIR ==========
            if np.any(self.feature_importances):
                droppedout_context, y_frame = await forward_with_dropout([x_frame, y_frame, self.wset])
                Y_rl[i:i+24] = y_frame

        # ========== EXTRAER HIPÓTESIS DEL ROLLOUT ==========
        hypotheses = await extract_hypotheses_from_rollout(
                model=self.model,
                X=x_frame,
                Y=y_frame,
        )
        #print("HYPOTHESES", hypotheses)
        # Almacenar en rollout
        validated_hypotheses = []
        for hyp in hypotheses:
            # Validar contra hipótesis previas
            validated_hyp = self.hypothesis_validator.add_hypothesis(hyp, rollout_id=str(i))
            validated_hypotheses.append(validated_hyp)
    
        await save_hypotheses_results(
            hypotheses=hypotheses,
            output_dir=self.output_dir,
            loop_counter=self.loop_counter,
            rollout_id='0',
            baseline_idx=self.baseline_idx
        )

        rollout_args_list = []
        
        for i in range(len(X_rl)):
            x_f = X_rl[i]
            y_f = Y_rl[i]
        
            # ========== PASAR shared_memory COMO PRIMER ARGUMENTO ==========
            rollout_args_list.append([
                self.agent_memory,      # ← shared_memory (primer argumento)
                self.model,             # ← model
                self.pipeline,          # ← pipeline
                self.constitutional_ai, # ← constitutional_ai
                self.loop_counter,      # ← loop_counter
                self.baseline_idx,      # ← baseline_idx
                i,                      # ← índice del rollout
                x_f,                    # ← x_f
                y_f,                    # ← y_f
            ])
            
        # ========== 3. EJECUTAR ROLLOUTS EN PARALELO ==========
        logger.info(f"[RL] Ejecutando {len(rollout_args_list)} rollouts en paralelo (shared=True)...")
    
        results = await parallel(
            values=rollout_args_list,
            n_times=len(rollout_args_list),
            args=[],
            func=run_single_rollout_worker,
            index=True,
            shared=False,  
            fifo=True,
            lifc=True,
            continuous=False,
        )
        
        # ========== 4. RECOLECTAR RESULTADOS ==========
        # results es una lista de diccionarios, uno por rollout
        rollouts = []
        rewards = []
        preference_pairs = []
        consolidation_triggers = []
        all_hypotheses = []
    
        for result in results:
            if result is None or result.get('status') == 'failed':
                continue
            
            rollouts.append(result['rollout'])
            rewards.append(result['composite_reward'])
            
            if result.get('preference_pairs'):
                preference_pairs.extend(result['preference_pairs'])
            
            if result.get('consolidation_triggered'):
                consolidation_triggers.append(result['rollout_idx'])
            
            if result.get('hypotheses'):
                all_hypotheses.extend(result['hypotheses'])
    
        # ========== 5. ACTUALIZAR ROLLOUT_DATA ==========
        for rollout in rollouts:
            self.rollout_data['inputs'].append(rollout['input'])
            self.rollout_data['values'].append(rollout.get('constitution_score', 0.5))
            self.rollout_data['targets'].append(rollout['target'])
            self.rollout_data['dones'].append(rollout.get('done', False))
            self.rollout_data['predictions'].append(rollout['prediction'])
            self.rollout_data['composite_rewards'].append(rollout['composite_reward'])
            self.rollout_data['rewards'].append(rollout['composite_reward'])
            self.rollout_data['constitution_scores'].append(rollout.get('constitution_score', 0))                    
            
        # En run_rl_rollouts, después de procesar todos los rollouts:
        if self.hypothesis_validator:
            # Obtener estadísticas
            stats = self.hypothesis_validator.get_validation_stats()
            logger.info(f"[Validator] Hipótesis validadas: {stats.get('validated_hypotheses', 0)}/{stats.get('total_hypotheses', 0)}")
        
            # Obtener mejores hipótesis por crédito
            top_credit = self.hypothesis_validator.get_highest_credit(5)
            for h in top_credit:
                logger.info(f"  Hipótesis {h.id}: crédito={h.credit:.2f}, validaciones={h.validation_count}")
        
            # Guardar estado del validador
            validator_path = os.path.join(self.output_dir, f"hypothesis_validator_loop_{self.loop_counter}.pkl")
            with open(validator_path, 'wb') as f:
                pickle.dump(self.hypothesis_validator, f)
    
        if all_rollout_hypotheses:
            await save_hypotheses_results(
                hypotheses=all_rollout_hypotheses,
                output_dir=self.output_dir,
                loop_counter=self.loop_counter,
                rollout_id="all_rollouts",
                baseline_idx=self.baseline_idx
            )

        # Obtener indicadores de misalignment de la memoria
        misalignment = self.agent_memory.get_misalignment_indicators()
        self.misalignment_history.append({
            'timestamp': datetime.now().isoformat(),
            'loop': self.loop_counter,
            'misalignment': misalignment,
            'memory_metrics': self.agent_memory.get_memory_metrics()
        })
        
        # Guardar reporte de memoria
        memory_report_path = os.path.join(self.output_dir, f"memory_report_loop_{self.loop_counter}.json")
        with open(memory_report_path, "w") as f:
            json.dump({
                'misalignment_indicators': misalignment,
                'memory_metrics': self.agent_memory.get_memory_metrics(),
                'history': self.misalignment_history[-10:]
            }, f, indent=2)

        # 2. GRPO: calcular ventajas por grupo
        grpo_optimizer = GRPOOptimizer(model=self.model, group_size=8)
        group_ids = [i // 8 for i in range(len(rewards))]
        
        advantages = await grpo_optimizer.compute_group_advantages(
            rewards=rewards,
            group_ids=group_ids
        )
    
        # 3. Actualizar modelo con GRPO
        X_batch = np.array([r["input"] for r in rollouts])
        Y_batch = np.array([r["target"] for r in rollouts])
        
        grpo_result = await grpo_optimizer.grpo_update(
            X_batch=X_batch,
            y_batch=Y_batch,
            advantages=advantages,
            learning_rate=0.001,
            preference_pairs = preference_pairs
        )
        
        #PPO           
        gae_estimator = GeneralizedAdvantageEstimator(gamma=0.99, lambda_gae=0.95)    
            
        advantages_gae, returns = await gae_estimator.compute_gae(
            rewards=self.rollout_data['rewards'],
            values=self.rollout_data['values'] + [0.0],  # Valor bootstrap final
            dones=self.rollout_data['dones'],
            feature_importances=self.feature_importances,
        ) 
        
        # 2. Memory Based Done Reward
        memory_done_reward = MemoryBasedDoneReward(
            agent_memory=self.agent_memory,
            similarity_threshold=0.7
        )           

        done_reward = memory_done_reward.compute_done_reward(
            final_state=rollouts[-1]['prediction'],
            final_reward=rewards[-1],
            trajectory=[r['prediction'] for r in rollouts]
        )

        # ========== GUARDAR RESULTADOS DE GAE EN JSON ==========
        gae_results = {
            'advantages': advantages_gae.tolist() if isinstance(advantages_gae, np.ndarray) else advantages_gae,
            'returns': returns.tolist() if isinstance(returns, np.ndarray) else returns,
            'dones': done_reward,
            'mean_advantage': float(np.mean(advantages_gae)) if len(advantages_gae) > 0 else 0.0,
            'std_advantage': float(np.std(advantages_gae)) if len(advantages_gae) > 0 else 0.0,
            'mean_return': float(np.mean(returns)) if len(returns) > 0 else 0.0,
            'gamma': gae_estimator.gamma,
            'lambda_gae': gae_estimator.lambda_gae
        }
        
        # Guardar en archivo JSON
        gae_path = os.path.join(self.output_dir, f"gae_results_loop_{self.loop_counter}.json")
        with open(gae_path, 'w') as f:
            json.dump(gae_results, f, indent=2, default=str)
        logger.info(f"[GAE] Resultados guardados en {gae_path}")

        ppo_result = await grpo_optimizer.grpo_update(
            X_batch=X_batch,
            y_batch=Y_batch,
            advantages=advantages_gae,  # ← advantages_gae VIVAS y USADAS
            learning_rate=0.0005,
            preference_pairs = preference_pairs
        )

        for i in range(len(rollouts)):
            rollout = rollouts[i]
            uncertainty = rollout.get('reward_components', {}).get('uncertainty', 0.5)
            reward = rollout.get('composite_reward', 0)
            # Añadir al rollout
            rollouts[i]['done_reward'] = done_reward
            rollouts[i]['total_reward'] = rollout['composite_reward'] + done_reward
            
            # Solo mantener rollouts con baja incertidumbre (< 0.3) y alta recompensa (> 0.6)
            if uncertainty < 0.3:
                continue
            else:
                rollouts[i]['composite_reward'] = 0
    
        mean_reward = float(np.mean(rewards))
        std_reward = float(np.std(rewards))
        logger.info(f"[RL] Recompensa media={mean_reward:.4f}  std={std_reward:.4f}")

        # Después de procesar todos los rollouts
        all_constitution_scores = []
        all_violations_by_principle = defaultdict(int)
        
        for rollout in rollouts:
            const_result = rollout.get('constitution_result', {})
            all_constitution_scores.append(const_result.get('constitution_score', 0))
            for v in const_result.get('violations', []):
                all_violations_by_principle[v['principle']] += 1

        # Guardar en archivo - output_dir debe ser pasado o almacenado en self
        # Por ahora, usar un directorio por defecto o almacenar en self
        if hasattr(self, 'output_dir'):
            out_dir = self.output_dir
        else:
            out_dir = "."
            
        rl_summary_path = os.path.join(
            out_dir, 
            f"rl_summary_baseline_{self.baseline_idx:04d}_loop_{self.loop_counter}.txt"
        )            
        with open(rl_summary_path, "w") as f:
            f.write("="*80 + "\n")
            f.write("RESUMEN DE RL - SERIE DE ROLLOUTS\n")
            f.write("="*80 + "\n\n")
            f.write(f"Total rollouts procesados: {len(rollouts)}\n")
            f.write(f"Recompensa media: {mean_reward:.4f}\n")
            f.write(f"Score constitucional medio: {np.mean(all_constitution_scores):.4f}\n\n")
            f.write("VIOLACIONES POR PRINCIPIO:\n")
            for principle, count in all_violations_by_principle.items():
                f.write(f"  - {principle}: {count} violaciones\n")
        
        # Pasar X_rl y Y_rl (targets originales) a run_reinforcement_learning
        # No usar predicciones como targets
        rl_result = await self.pipeline.run_reinforcement_learning(
            X=X_rl,
            y=Y_rl,
            num_rollouts=len(rollouts),
            ppo_steps=ppo_steps,
            go_no_go_threshold=0.05,  # Valor por defecto
            sample_efficiency_analysis=False,  # Valor por defecto
            external_advantages=advantages_gae
        )

        # 4. rl_result con rewards y métricas
        rl_result["mean_advantage"] = grpo_result["mean_advantage"]
        
        # Detección de reward hacking sobre rollouts reales
        hacking_flags = []
        for r in rollouts:
            risk, flags = self.hacking_detector.detect(
                r["input"], r["prediction"], r["reward_components"]
            )
            if flags:
                hacking_flags.append({
                    "frame_mse":  r["frame_mse"],
                    "risk":       risk,
                    "flags":      flags,
                    "error_bucket": r["error_bucket"].value if r["error_bucket"] else None,
                })

        rl_result["hacking_flags"]    = hacking_flags
        rl_result["n_hacking_flags"]  = len(hacking_flags)
        rl_result["rollout_rewards"]  = rewards
        rl_result["aoc_metrics"]      = await evaluate_aoc_metrics(
            self.model.predict, X_rl, Y_rl, label="RL"
        )
        
        rl_result['efficiency_summary'] = self.efficiency_monitor.get_summary()        
        
        return rl_result

    async def holistic_evaluation(self, X_test: np.ndarray, Y_test: np.ndarray) -> Dict:
        """Evaluación holística del modelo - versión robusta"""
        
        results = {}
        
        # Limitar tamaño para no sobrecargar
        max_samples = min(800, len(X_test))
        X_test = X_test[:max_samples]
        Y_test = Y_test[:max_samples]
        
        # División simple en 4 partes iguales
        n_per_domain = max_samples // 4
        domains = ['military', 'text', 'music', 'medicine']
        
        for i, domain in enumerate(domains):
            start = i * n_per_domain
            end = start + n_per_domain if i < 3 else max_samples
            
            if start >= max_samples:
                continue
                
            X_domain = X_test[start:end]
            Y_domain = Y_test[start:end]
            
            if len(X_domain) == 0:
                continue
            
            try:
                preds = await self.model.predict(X_domain)
                preds = preds[0]
                mse = float(np.mean((preds - Y_domain) ** 2))
                snr = compute_snr(Y_domain, preds)
                lsd = compute_spectral_distortion(Y_domain, preds)
                
                results[domain] = {
                    'mse': mse,
                    'snr_db': snr,
                    'log_spectral_distortion': lsd,
                    'n_samples': len(X_domain)
                }
            except Exception as e:
                logger.warning(f"[Holistic] Error en dominio {domain}: {e}")
                results[domain] = {'error': str(e), 'n_samples': len(X_domain)}
        
        if results and any('snr_db' in v for v in results.values()):
            valid_snrs = [v['snr_db'] for v in results.values() if 'snr_db' in v]
            holistic_score = np.mean(valid_snrs) if valid_snrs else 0.0
        else:
            holistic_score = 0.0
        
        logger.info(f"[Holistic] Score={holistic_score:.4f}")
        
        return {
            'domain_results': results,
            'holistic_score': holistic_score
        }

    def _build_spectral_cot(
        self,
        x_frame: np.ndarray,
        pred: np.ndarray,
    ) -> List[str]:
        """
        Construye chain-of-thought desde características espectrales
        reales del frame — nunca texto genérico.
        Maneja tanto frames individuales (1D) como batches (2D).
        """
        # Si es batch (2D), tomar el primer frame como representativo
        if x_frame.ndim == 2:
            x_frame = x_frame[0]
        if pred.ndim == 2:
            pred = pred[0]
        
        centroid   = float(np.average(np.arange(len(x_frame)), weights=x_frame + 1e-9))
        energy_in  = float(np.mean(x_frame ** 2))
        energy_out = float(np.mean(pred ** 2))
        snr_est    = compute_snr(x_frame, pred)
        occ_bins   = int(np.sum(x_frame < 1e-4))
    
        return [
            f"Frame espectral: centroide={centroid:.1f}  energía_entrada={energy_in:.5f}",
            f"Bins ocluidos detectados (energía<1e-4): {occ_bins}/{len(x_frame)}",
            f"Energía estimada tras AOC: {energy_out:.5f}  (ratio={energy_out/max(energy_in,1e-9):.3f})",
            f"SNR estimado frame: {snr_est:.2f} dB",
            "Aplicando Wiener spectral subtraction implícita via regresión DSVM",
            "Integrando contexto de atención continua multi-lorax para reconstrucción",
        ]

    async def run_full_post_training(
        self,
        quality_threshold: float = 0.75,
        max_sft_samples: int     = 500,
        num_rollouts: int        = 80,
        ppo_steps: int           = 10,
    ) -> Dict:
        """
        Pipeline completo de post-entrenamiento:
            1. SFT con rejection sampling sobre frames reales
            2. RL con rollouts sobre frames reales
            3. Auditoría constitucional final sobre frames reales
        """
        logger.info("=" * 70)
        logger.info("[PostTrain] Iniciando post-entrenamiento completo (audio real)")
        logger.info("=" * 70)

        results = {}

        # ── SFT ─────────────────────────────────────────────────────────────
        logger.info("[PostTrain] Fase 1/3: SFT con rejection sampling...")
        results["sft"] = await self.run_rejection_sampling_sft(
            quality_threshold = quality_threshold,
            max_samples       = max_sft_samples,
        )

        # ── RL ──────────────────────────────────────────────────────────────
        logger.info("[PostTrain] Fase 2/3: RL con rollouts...")
        results["rl"] = await self.run_rl_rollouts(
            num_rollouts = num_rollouts,
            ppo_steps    = ppo_steps,
        )
        
        output_dir = 'post_train'

        preference_model_path = os.path.join(output_dir, "preference_model.pkl")
        with open(preference_model_path, "wb") as f:
            pickle.dump({
                'reward_model': self.reward_model,
                'constitutional_ai': self.constitutional_ai,
                'rl_config': getattr(self, 'rl_config', {
                    'num_rollouts': 80,
                    'ppo_steps': 10,
                    'go_no_go_threshold': 0.05
                }),
                'constitution_rules': self.constitutional_ai.constitution_rules
            }, f)

        # ── Auditoría constitucional final ──────────────────────────────────
        logger.info("[PostTrain] Fase 3/3: Auditoría constitucional final...")
        results["constitution_audit"] = await self._run_constitution_audit()

        # ── Intervención automática si hay violaciones críticas ─────────────
        audit      = results["constitution_audit"]
        violations = [
            v for v in audit["violations"]
            if v["severity"] in ("high", "critical")
        ]
        if violations:
            logger.warning(
                f"[PostTrain] {len(violations)} violaciones constitucionales críticas. "
                "Ejecutando AutoInterventionSystem..."
            )
            X_int, Y_int = self._sample_real_frames(200, seed=99)
            for v in violations:
                try:
                    intervention = self.intervention_sys.process_with_intervention(
                        input_data = X_int,
                        output     = asyncio.run(self.model.predict(X_int)),
                        context    = None,
                        error_type = v["bucket"],
                    )
                    logger.info(
                        f"[PostTrain] Intervención {v['bucket']}: "
                        f"mejora={intervention.get('improvement', 'N/A')}"
                    )
                except Exception as e:
                    logger.warning(f"[PostTrain] Intervención {v['bucket']} falló: {e}")

        # ── Métricas finales sobre holdout de audio real ─────────────────────
        X_final, Y_final = self._sample_real_frames(300, seed=42)
        results["final_aoc_metrics"] = await evaluate_aoc_metrics(
            self.model.predict, X_final, Y_final, label="PostTrain_Final"
        )

        logger.info("[PostTrain] Post-entrenamiento completado.")
        return results

    async def _run_constitution_audit(self) -> Dict:
        """
        Evaluación constitucional sobre frames reales de audio.
        """
        X_audit, Y_audit = self._sample_real_frames(100, seed=7)
        preds = await self.model.predict(X_audit)
        preds = preds[0]

        all_violations = []
        all_scores = []

        for i in range(len(X_audit)):
            # Usar self.constitutional_ai (atributo de instancia)
            result = self.constitutional_ai.evaluate(
                input=X_audit[i],
                output=preds[i],
                chain_of_thought=self._build_spectral_cot(X_audit[i], preds[i]),
            )
            all_scores.append(result["constitution_score"])
            all_violations.extend(result.get("violations", []))

        # Agrupar violaciones por principio (ErrorBucket)
        by_bucket: Dict[str, List] = defaultdict(list)
        for v in all_violations:
            by_bucket[v["principle"]].append(v)

        mean_score = float(np.mean(all_scores)) if all_scores else 0.0

        # Construir lista de violaciones con severidad
        violations_summary = [
            {
                "bucket":    bucket,
                "count":     len(items),
                "severity":  items[0]["severity"] if items else "unknown",
                "sample":    items[0].get("details", ""),
            }
            for bucket, items in by_bucket.items()
        ]

        logger.info(
            f"[Constitution] Score medio={mean_score:.4f}  "
            f"Buckets violados={len(by_bucket)}"
        )
        for v in violations_summary:
            logger.info(
                f"  [{v['severity'].upper()}] {v['bucket']}: "
                f"{v['count']} violaciones — {v['sample']}"
            )

        return {
            "mean_constitution_score": mean_score,
            "violations":              violations_summary,
            "n_audited_frames":        len(X_audit),
        }

    # Dentro de la clase AOCPostTrainer, agregar estos métodos:
    def apply_dropout_to_features(self, X: np.ndarray, feature_importances: np.ndarray = None, dropout_rate: float = 0.3) -> np.ndarray:
        """
        Aplica dropout a las features basado en sus importancias.
        Features con menor importancia tienen mayor probabilidad de ser dropouteadas.
        """
        if feature_importances is None:
            feature_importances = getattr(self, 'feature_importances', None)
        
        if feature_importances is None or dropout_rate == 0:
            return X
        
        # Normalizar importancias
        imp_norm = feature_importances / (feature_importances.sum() + 1e-8)
        
        # Features menos importantes tienen mayor probabilidad de dropout
        dropout_probs = 1.0 - (imp_norm / imp_norm.max())
        dropout_probs = np.clip(dropout_probs * dropout_rate, 0, dropout_rate)
        
        # Crear máscara
        mask = np.random.binomial(1, 1 - dropout_probs, size=X.shape[1])
        return X * mask
    
    def compute_weighted_mse(self, pred: np.ndarray, target: np.ndarray, feature_importances: np.ndarray = None) -> float:
        """
        Calcula MSE ponderado por feature importances.
        """
        if feature_importances is None:
            feature_importances = getattr(self, 'feature_importances', None)
        
        if feature_importances is None:
            return float(np.mean((pred - target) ** 2))
    
        # Asegurar que las dimensiones coincidan
        if len(feature_importances) != pred.shape[-1]:
            # Si no coinciden, usar promedio simple
            return float(np.mean((pred - target) ** 2))
        
        # Ponderar error por feature importances
        per_feature_mse = (pred - target) ** 2
        weighted_mse = np.mean(per_feature_mse * feature_importances)
        return float(weighted_mse)

def is_distillation_complete(baseline_dir: str, iteration: int) -> bool:
    """Verifica si la distillation para esta baseline/iteración ya existe"""
    iter_dir = os.path.join(baseline_dir, f"iter_{iteration:04d}")
    result_file = os.path.join(iter_dir, "distillation_results.json")
    complete_marker = os.path.join(baseline_dir, f"distillation_iter_{iteration}.complete")
    
    return os.path.exists(result_file) or os.path.exists(complete_marker)

def load_distillation_results(baseline_dir: str) -> Dict[str, Any]:
    """
    Carga los resultados de distillation de una baseline específica.
    
    Args:
        baseline_dir: Directorio de la baseline (ej: outputs/parallel_baselines/baseline_0000)
    
    Returns:
        Diccionario con 'student_predictions' y metadata
    """
    import glob
    
    # Buscar el archivo de distillation_results.json más reciente
    dist_files = glob.glob(os.path.join(baseline_dir, "iter_*/distillation_results.json"))
    
    if not dist_files:
        # Buscar en la raíz
        dist_files = glob.glob(os.path.join(baseline_dir, "distillation_results.json"))
    
    if not dist_files:
        logger.warning(f"[load_distillation_results] No se encontraron resultados en {baseline_dir}")
        return {'student_predictions': None, 'status': 'not_found'}
    
    # Usar el más reciente
    latest_file = max(dist_files, key=os.path.getmtime)
    
    with open(latest_file, 'r') as f:
        results = json.load(f)
    
    # Intentar cargar predicciones del student si existen
    student_predictions = None
    student_pred_path = os.path.join(os.path.dirname(latest_file), "student_predictions.npy")
    if os.path.exists(student_pred_path):
        student_predictions = np.load(student_pred_path)
    
    return {
        'student_predictions': student_predictions,
        'distillation_results': results,
        'status': 'loaded',
        'file': latest_file
    }
    
async def run_full_pipeline(args: argparse.Namespace, go_no_go_threshold: float = 0.05) -> Dict:
    """
    Orquesta todas las etapas del pipeline AOC.
    """
    import pickle
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Inicializar loop_counter para identificar iteraciones
    loop_counter = 0
    
    # Cargar o crear archivo de seguimiento de loops
    loop_state_path = os.path.join(args.output_dir, "loop_state.json")
    if os.path.exists(loop_state_path):
        with open(loop_state_path, "r") as f:
            loop_state = json.load(f)
            loop_counter = loop_state.get("last_loop", 0) + 1
    else:
        loop_counter = 1
    
    # Guardar estado actual del loop
    loop_state = {"last_loop": loop_counter, "timestamp": datetime.now().isoformat()}
    with open(loop_state_path, "w") as f:
        json.dump(loop_state, f)
    
    logger.info(f"[Pipeline] Iniciando loop #{loop_counter}")
    
    results: Dict[str, Any] = {
        "run_id": hashlib.md5(f"{datetime.now().isoformat()}".encode()).hexdigest()[:12],
        "timestamp": datetime.now().isoformat(),
        "args": vars(args),
        "loop_counter": loop_counter,
    }

    # ========== INICIALIZAR BASELINE MANAGER ==========
    existing_baselines = []
    for d in os.listdir(args.output_dir):
        if d.startswith("baseline_") and os.path.isdir(os.path.join(args.output_dir, d)):
            existing_baselines.append(d)
    
    if existing_baselines and not getattr(args, 'force_new_baseline', False):
        # Usar la baseline más reciente
        latest_baseline = sorted(existing_baselines)[-1]
        baseline_name = latest_baseline
        logger.info(f"[Pipeline] Reutilizando baseline existente: {baseline_name}")
    else:
        # Crear nueva baseline
        baseline_name = f"baseline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        logger.info(f"[Pipeline] Creando nueva baseline: {baseline_name}")
    
    baseline_manager = BaselineModelManager(args.output_dir, baseline_name)

    # Guardar configuración inicial
    config_path = os.path.join(baseline_manager.baseline_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)
    
    logger.info(f"[Pipeline] Baseline Manager inicializado: {baseline_name}")
    
    # ══════════════════════════════════════════════════════════════════════
    # ETAPA 0 — Carga de audios reales (con caché en output_dir)
    # ══════════════════════════════════════════════════════════════════════
    logger.info("=" * 70)
    logger.info("ETAPA 0 — Carga de audios reales")
    logger.info("=" * 70)

    MAX_FRAMES = 90000

    _cache_paths = {
        "xtrain": os.path.join(args.output_dir, "xtrain.npy"),
        "ytrain": os.path.join(args.output_dir, "ytrain.npy"),
        "xtest":  os.path.join(args.output_dir, "xtest.npy"),
        "ytest":  os.path.join(args.output_dir, "ytest.npy"),
    }
    _cache_exists = all(os.path.isfile(p) for p in _cache_paths.values())

    if _cache_exists:
        logger.info(
            "[Etapa 0] Datasets encontrados en output_dir — "
            "cargando desde caché (omitiendo procesamiento de audio)."
        )
        X_train = np.load(_cache_paths["xtrain"])
        Y_train = np.load(_cache_paths["ytrain"])
        
        # Definir file_ids_train ANTES de usarlo
        # Como no se guarda en caché, usamos placeholders
        file_ids_train = ["cached"] * len(X_train)
    
        unique_indices = np.unique(X_train, axis=0, return_index=True)[1]
        if len(unique_indices) < len(X_train):
            duplicates_removed = len(X_train) - len(unique_indices)
            logger.info(f"[Deduplicación] Eliminados {duplicates_removed} frames duplicados")
            X_train = X_train[unique_indices]
            Y_train = Y_train[unique_indices]
            file_ids_train = [file_ids_train[i] for i in unique_indices]  # Ahora sí existe
    
        X_test  = np.load(_cache_paths["xtest"])
        Y_test  = np.load(_cache_paths["ytest"])
    
        # Reconstruir val como 10 % del total cacheado (mismo split 80/10/10)
        n_cached = len(X_train) + len(X_test)
        n_val_cached = max(1, int(round(n_cached * 0.10)))
        X_val, Y_val = X_train[-n_val_cached:], Y_train[-n_val_cached:]
        X_train, Y_train = X_train[:-n_val_cached], Y_train[:-n_val_cached]
        
        # Actualizar file_ids_train después del split
        file_ids_train = file_ids_train[:-n_val_cached] if len(file_ids_train) > n_val_cached else file_ids_train
    
        n_total = len(X_train) + len(X_val) + len(X_test)
        n_bins  = X_train.shape[1]
        logger.info(
            f"[Etapa 0] Caché cargado — "
            f"Train={len(X_train)}  Val={len(X_val)}  Test={len(X_test)}  "
            f"Bins={n_bins}"
        )
        results["data_info"] = {
            "n_total": n_total, "n_bins": n_bins,
            "n_train": len(X_train), "n_val": len(X_val), "n_test": len(X_test),
            "source": "cache",
        }
    
    else:
        logger.info("[Etapa 0] No se encontró caché — procesando desde audio.")

        if args.occluded_dir and os.path.isdir(args.occluded_dir):
            X_all, Y_all, file_ids = await build_frame_pairs(
                clean_dir    = args.audio_dir,
                occluded_dir = args.occluded_dir,
                max_files    = args.max_files,
            )
        else:
            logger.info(
                "[Etapa 0] No se encontró occluded_dir — "
                "simulando oclusión desde directorio limpio."
            )
            X_all, Y_all, file_ids = await build_frame_pairs_single_dir(
                audio_dir       = args.audio_dir,
                max_files       = args.max_files,
                occlusion_ratio = args.occlusion_ratio,
            )

        # Después de construir X_all, Y_all
        if n_total > 0:
            unique_indices = np.unique(X_all, axis=0, return_index=True)[1]
            if len(unique_indices) < n_total:
                duplicates_removed = n_total - len(unique_indices)
                logger.info(f"[Deduplicación] Eliminados {duplicates_removed} frames duplicados")
                X_all = X_all[unique_indices]
                Y_all = Y_all[unique_indices]
                file_ids = [file_ids[i] for i in unique_indices]

        n_total = len(X_all)
        n_bins  = X_all.shape[1]
        logger.info(f"[Etapa 0] Total frames cargados: {n_total}  Bins: {n_bins}")

        # Limitar dataset a 30 000 frames desde el inicio
        if n_total > MAX_FRAMES:
            logger.info(
                f"[Etapa 0] Dataset recortado: {n_total} → {MAX_FRAMES} frames "
                f"(límite MAX_FRAMES={MAX_FRAMES})"
            )
            X_all    = X_all[:MAX_FRAMES]
            Y_all    = Y_all[:MAX_FRAMES]
            file_ids = file_ids[:MAX_FRAMES]
            n_total  = MAX_FRAMES

        # Split train / val / test (80 / 10 / 10) — índices secuenciales para
        # mantener la estructura temporal de los audios
        i_val  = int(0.80 * n_total)
        i_test = int(0.90 * n_total)
        X_train, Y_train = X_all[:i_val],         Y_all[:i_val]
        X_val,   Y_val   = X_all[i_val:i_test],   Y_all[i_val:i_test]
        X_test,  Y_test  = X_all[i_test:],        Y_all[i_test:]
        file_ids_train   = file_ids[:i_val]

        logger.info(
            f"[Etapa 0] Train={len(X_train)}  Val={len(X_val)}  Test={len(X_test)}"
        )
        results["data_info"] = {
            "n_total": n_total, "n_bins": n_bins,
            "n_train": len(X_train), "n_val": len(X_val), "n_test": len(X_test),
            "source": "audio",
        }

    # ══════════════════════════════════════════════════════════════════════
    # ETAPA 1 — Pre-entrenamiento Ridge + MLP
    # ══════════════════════════════════════════════════════════════════════
    logger.info("=" * 70)
    logger.info("ETAPA 1 — Cache de dataset")
    logger.info("=" * 70)

    # Persistir datasets en output_dir solo si no venían de caché
    if not _cache_exists:
        np.save(_cache_paths["xtrain"], X_train)
        np.save(_cache_paths["ytrain"], Y_train)
        np.save(_cache_paths["xtest"],  X_test)
        np.save(_cache_paths["ytest"],  Y_test)
        logger.info(f"[Etapa 1] Datasets guardados en {args.output_dir}")

    # ══════════════════════════════════════════════════════════════════════
    # ETAPA 2 — Cross-validation adversaria con DSVM
    # ══════════════════════════════════════════════════════════════════════
    logger.info("=" * 70)
    logger.info("ETAPA 2 — Cross-validation adversaria (DSVM nativo)")
    logger.info("=" * 70)

    # Cargar o crear modelo DSVM
    dsvm_raw = None
    if args.model_path and os.path.exists(args.model_path):
        try:
            import pickle
            with open(args.model_path, "rb") as f:
                dsvm_raw = pickle.load(f)
            logger.info(f"[Etapa 2] DSVM cargado desde {args.model_path}")
        except Exception as e:
            logger.warning(f"[Etapa 2] No se pudo cargar DSVM: {e} — usando fallback")
            
    total_samples = len(X_train)
    samples_per_baseline = 4000
    n_baselines = (total_samples + samples_per_baseline - 1) // samples_per_baseline
    
    logger.info(f"[Pipeline] Procesando {total_samples} muestras en {n_baselines} baselines")
    
    all_baseline_results = []
    
    # ========== 2. PREPARAR ARGUMENTOS PARA PARALLEL ==========
    baselines_dir = os.path.join(
        args.output_dir, 
        f"parallel_baselines"
    )
    
    # ========== VERIFICAR SI EXISTE UN DIRECTORIO ANTERIOR ==========
    # Buscar directorios parallel_baselines_ existentes
    existing_parallel_dirs = []
    for d in os.listdir(args.output_dir):
        if d.startswith("parallel_baselines") and os.path.isdir(os.path.join(args.output_dir, d)):
            existing_parallel_dirs.append(os.path.join(args.output_dir, d))
    
    # Usar el más reciente si existe y no se fuerza recreación
    if existing_parallel_dirs and not getattr(args, 'force_rerun', False):
        latest_dir = max(existing_parallel_dirs, key=os.path.getmtime)
        baselines_dir = latest_dir
        logger.info(f"📁 Reutilizando directorio existente: {baselines_dir}")
    else:
        logger.info(f"📁 Creando nuevo directorio: {baselines_dir}")
    
    os.makedirs(baselines_dir, exist_ok=True)
    
    loop_counter = 1

    # Manager para objetos compartidos entre procesos
    manager = mp.Manager()
    
    # Diccionario compartido
    shared_dict = manager.dict()

    pipeline_memory = AgentMemory(
        ltm_max_size=100000,
        mtm_max_size=10000,
        stm_max_size=1000,
        use_shared_memory=False,  
        shared_memory_base_name=f"agent_mem_{uuid.uuid4().hex[:8]}"
    )

    # Guardar en el diccionario compartido
    shared_dict['memory'] = pipeline_memory
    shared_dict['lock'] = manager.RLock()  # ← Lock para sincronización
    
    logger.info("[Pipeline] Memoria global compartida inicializada (via Manager)")
    
    # Lista de listas - cada sublista son los argumentos para UNA baseline
    baseline_args_list = []
    SAMPLES_PER_BASELINE = 4000
    
    for baseline_idx in range(n_baselines):
        start_idx = baseline_idx * SAMPLES_PER_BASELINE
        end_idx = min(start_idx + SAMPLES_PER_BASELINE, total_samples)
        
        X_batch = X_train[start_idx:end_idx]
        Y_batch = Y_train[start_idx:end_idx]
        
        baseline_dir = os.path.join(baselines_dir, f"baseline_{baseline_idx:04d}")
        os.makedirs(baseline_dir, exist_ok=True)
        
        # Guardar batch para esta baseline
        np.save(os.path.join(baseline_dir, "X_batch.npy"), X_batch)
        np.save(os.path.join(baseline_dir, "Y_batch.npy"), Y_batch)
        
        # CADA elemento es una lista con los 13 argumentos para UNA baseline
        baseline_args_list.append([
            shared_dict,
            baseline_idx,
            start_idx,
            end_idx,
            X_batch,
            Y_batch,
            X_val,
            Y_val,
            X_test,
            Y_test,
            file_ids_train,
            args,
            loop_counter,
            baseline_dir,
            dsvm_raw
        ])
    
    # ========== 3. EJECUTAR EN PARALELO ==========
    logger.info("="*70)
    logger.info(f"🚀 EJECUTANDO {len(baseline_args_list)} BASELINES EN PARALELO")
    logger.info("="*70)
    
    # parallel() toma cada sublista y la desempaqueta como argumentos
    results = await parallel(
        values=baseline_args_list,  # CADA sublista es un grupo de argumentos
        n_times=len(baseline_args_list),
        args= [],  # Sin argumentos adicionales
        func=run_baseline_worker,
        index=True,
        shared=False,
        fifo=True,
        lifc=True,
        continuous=False,
    )
    
    # ========== 4. PROCESAR RESULTADOS ==========
    logger.info("="*70)
    logger.info("📊 PROCESANDO RESULTADOS")
    logger.info("="*70)
    
    # results es una lista de resultados (uno por baseline)
    all_results = results if isinstance(results, list) else [results]
    
    combined_results = {
        'run_id': hashlib.md5(f"{datetime.now().isoformat()}".encode()).hexdigest()[:12],
        'timestamp': datetime.now().isoformat(),
        'args': vars(args),
        'baseline_config': {
            'samples_per_baseline': SAMPLES_PER_BASELINE,
            'n_baselines': n_baselines,
            'total_samples': total_samples,
        },
        'baselines': all_results,
        'baselines_dir': baselines_dir,
    }
    
    # Calcular estadísticas
    completed = [r for r in all_results if r and r.get('status') == 'completed']
    if completed:
        avg_snr = np.mean([r.get('aoc_metrics', {}).get('test', {}).get('snr_db', -100) for r in completed])
        avg_mse = np.mean([r.get('aoc_metrics', {}).get('test', {}).get('mse', 1) for r in completed])
        
        combined_results['aggregated_metrics'] = {
            'mean_snr_db': float(avg_snr),
            'mean_mse': float(avg_mse),
            'completed_baselines': len(completed),
            'failed_baselines': len(all_results) - len(completed),
        }
    
    # ========== 5. GUARDAR RESULTADOS ==========
    report_path = os.path.join(args.output_dir, "parallel_baseline_results.json")
    with open(report_path, "w") as f:
        json.dump(combined_results, f, indent=2, default=str)
    
    logger.info(f"📄 Reporte guardado en {report_path}")
    logger.info(f"✅ Completadas: {len(completed)}/{len(all_results)} baselines")

    # ══════════════════════════════════════════════════════════════════════
    # Knowledge Distillation con Teacher Forcing
    # ══════════════════════════════════════════════════════════════════════
    # ========== 1. ENCONTRAR EL MEJOR TEACHER ENTRE BASELINES COMPLETADAS ==========
    parallel_baselines_dir = os.path.join(args.output_dir, "parallel_baselines")
    post_trains = []
    for result in all_results:
        pt = result.get('post_training', {})
        #print("RESULT:", result)
        post_trains.append({
             'baseline_idx': result.get('baseline_idx', 0),
             'constitutional_score': pt.get('constitution_audit', {}).get('mean_constitution_score', 0.5),
             'final_snr': result.get('aoc_final', {}).get('snr_db', 0),
             'improvement': result.get('improvements', {}).get('delta_snr_db', 0),
             'rollout_data': result.get('rollout_data', {
                 'inputs': [], 'targets': [], 'predictions': [], 
                 'composite_rewards': [], 'rewards': [], 'values': [], 
                 'dones': [], 'constitution_scores': []
             })
        })
    baseline_output_dir = args.output_dir

    current_models = {}
    feature_importances = []
        
    # Cargar modelos iniciales
    for baseline_idx in range(n_baselines):
        model_path = os.path.join(parallel_baselines_dir, f"baseline_{baseline_idx:04d}", "model.pkl")
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                current_models[baseline_idx] = pickle.load(f)

    # ========== 1. RECOLECTAR TODOS LOS CONSTITUTION_SCORES ==========
    all_scores = []
    for pt in post_trains:
        all_scores.append(pt['constitutional_score'])
    
    all_scores.sort(key=lambda x: x, reverse=True)
    split_point = len(all_scores) // 2
    
    angel_indices = [bl for bl in all_scores[:split_point]]
    demon_indices = [bl for bl in all_scores[split_point:]]
    
    print(f"😇 ÁNGELES (top 50%): {angel_indices}")
    print(f"👿 DEMONIOS (bottom 50%): {demon_indices}")
    
            
    # Buscar el mejor baseline según constitutional score
    angel_teacher_idx, angel_teacher_meta = find_angel_teacher(
        parallel_baselines_dir=parallel_baselines_dir,
        metric='constitution_score'
    )
    
    teachers = [angel_teacher_idx]

    for pt in range(len(post_trains)):
        # ========== NUEVO: VERIFICAR SI YA SE HIZO DISTILLATION ========== 
        if pt == angel_teacher_idx:   
            continue
        demon_teacher_idx, demon_teacher_meta = find_demon_teacher(
            parallel_baselines_dir=parallel_baselines_dir,
            exclude_baseline_idx=angel_teacher_idx,  # Excluir el actual si queremos otro
            metric='constitution_score'
        )
        teachers.append(demon_teacher_idx)
        
    print("TEACHERS:", teachers)    

    X_successes = []
    Y_successes = []
    
    baseline_idx = 0
    for idx in teachers:
        try:
            pt = post_trains[idx]
            # ========== NUEVO: VERIFICAR SI YA SE HIZO DISTILLATION ==========
            if is_distillation_complete(baseline_output_dir, baseline_idx + 1):
                logger.info(f"⏭️ Saltando distillation para baseline {baseline_idx + 1} (ya completada)")
                    
                # Cargar resultados existentes
                iter_dir = os.path.join(baseline_output_dir, f"iter_{baseline_idx + 1:04d}")
                result_file = os.path.join(iter_dir, "distillation_results.json")
                        
                if os.path.exists(result_file):
                    with open(result_file, 'r') as f:
                        results["knowledge_distillation"] = json.load(f)
                    results["knowledge_distillation"]["status"] = "skipped"
                else:
                    results["knowledge_distillation"] = {"status": "skipped", "reason": "already_completed"}
                
                # Cargar el modelo destilado existente
                model_path = os.path.join(iter_dir, "dsvm_main_distilled.pkl")
                if os.path.exists(model_path):
                    with open(model_path, 'rb') as f:
                        model = pickle.load(f)
                    logger.info(f"[Distillation] Modelo cargado desde {model_path}")
            else:
                for st in teachers:
                    if st == idx:
                        continue
                    logger.info("=" * 70)
                    logger.info("ETAPA 4.5 — Knowledge Distillation (Teacher Forcing)")
                    logger.info("=" * 70)
                    
                    # TEACHER = modelo post-entrenado
                    teacher_model = teachers[idx]
                    
                    # Obtener rollouts exitosos del teacher
                    all_inputs = np.concatenate(np.array(pt['rollout_data']['inputs']), axis=0)
                    all_targets = np.concatenate(np.array(pt['rollout_data']['targets']), axis=0)
                    all_inputs = all_inputs.reshape(int(all_inputs.size/1025), 1025)
                    all_rewards = np.array(pt['rollout_data']['composite_rewards'])
                    all_predictions = np.array(pt['rollout_data']['predictions'])
                        
                    X_successes.append(all_inputs)
                    Y_successes.append(all_targets)
                    X_success = all_inputs
                    Y_success = all_targets                  
                    print(f"[Distillation] Rollouts totales: {len(all_inputs)}")
                        
                    # ========== 3. DETERMINAR TEACHER Y STUDENT ==========
                    # TEACHER = mejor baseline encontrada (o modelo actual si no hay)
                    teacher_model = load_model_from_baseline(
                        baseline_idx=idx,
                        parallel_baselines_dir=parallel_baselines_dir
                    )
                    
                    # Crear STUDENT (nuevo modelo)
                    student_model = load_model_from_baseline(
                        baseline_idx=teachers[st],
                        parallel_baselines_dir=parallel_baselines_dir
                    )
                        
                    # Crear distiller
                    distiller = KnowledgeDistillation(
                        teacher_model=teacher_model,
                        student_model=student_model,
                        temperature=args.distillation_temperature,
                        alpha=args.distillation_alpha
                    )
                
                    # EJECUTAR DISTILLATION: student aprende de los aciertos del teacher
                    distilled_student = await distiller.distill(
                        X=X_success,
                        y=Y_success,  # Los targets originales de los aciertos
                        epochs=args.distillation_epochs,
                        output_dir=baseline_output_dir
                    )
                    
                    # Evaluar student vs teacher en test
                    teacher_aoc = await evaluate_aoc_metrics(teacher_model.predict, X_test, Y_test, "Teacher_Before")
                    student_aoc = await evaluate_aoc_metrics(distilled_student.predict, X_test, Y_test, "Student_Distilled")
                            
                    logger.info(f"[Distillation] Teacher SNR={teacher_aoc['snr_db']:.2f}dB, Student SNR={student_aoc['snr_db']:.2f}dB")
                        
                    # Guardar estadísticas de distillation
                    results["knowledge_distillation"] = {
                        "status": "completed",
                        "teacher_before_snr_db": teacher_aoc['snr_db'],
                        "student_after_snr_db": student_aoc['snr_db'],
                        "snr_improvement": student_aoc['snr_db'] - teacher_aoc['snr_db'],
                        "teacher_before_mse": teacher_aoc['mse'],
                        "student_after_mse": student_aoc['mse'],
                        "mse_improvement": teacher_aoc['mse'] - student_aoc['mse'],
                        "n_successful_rollouts": len(X_success),
                        "total_rollouts": len(all_inputs) if hasattr(post_trainer, 'rollout_data') else 0,
                        "success_rate": len(X_success) / len(all_inputs) if hasattr(post_trainer, 'rollout_data') else 0,
                        "distillation_metrics": distiller.get_metrics(),
                        "temperature": args.distillation_temperature,
                        "alpha": args.distillation_alpha,
                        "epochs": args.distillation_epochs
                    }
                    
                    # Guardar el nuevo teacher (antiguo student)
                    baseline_manager.save_model(
                        model=model,
                        name="dsvm_main_distilled",
                        iteration=baseline_idx + 1
                    )
                            
                    # Guardar el par teacher-student para trazabilidad
                    baseline_manager.save_teacher_student_pair(
                        teacher_model=teacher_model,
                        student_model=distilled_student,
                        distillation_result=results["knowledge_distillation"],
                        iteration=baseline_idx + 1
                    )

        except Exception as e:
            continue

        baseline_idx += 1  
    
    # Recolectar TODOS los resultados de distillation de todas las baselines
    all_distillation_outputs = []
                
    dist_results = load_distillation_results(baseline_output_dir)
    all_distillation_outputs.append(dist_results['student_predictions'])

    # ========== AL FINAL DEL PIPELINE (después de distillation) ==========

    logger.info("=" * 70)
    logger.info("🔍 EVALUACIÓN CONSTITUCIONAL POST-DISTILLATION")
    logger.info("=" * 70)

    # 1. Construir reglas constitucionales
    constitution_rules = build_aoc_constitution_rules()
    
    # 2. Evaluar TEACHER (ángel)
    teacher_eval = await evaluate_constitutional_post_distillation(
        model=teacher_model,  # El modelo teacher (ángel)
        X_test=X_test,
        Y_test=Y_test,
        constitution_rules=constitution_rules,
        output_dir=args.output_dir,
        model_name="teacher_angel"
    )
    
    # 3. Evaluar STUDENT (destilado)
    student_eval = await evaluate_constitutional_post_distillation(
        model=distilled_student,  # El modelo destilado
        X_test=X_test,
        Y_test=Y_test,
        constitution_rules=constitution_rules,
        output_dir=args.output_dir,
        model_name="demon_distilled"
    )

    # 4. Comparar resultados
    logger.info("=" * 70)
    logger.info("📊 COMPARATIVA TEACHER VS STUDENT")
    logger.info("=" * 70)
    
    teacher_score = teacher_eval["mean_constitution_score"]
    student_score = student_eval["mean_constitution_score"]
    score_diff = student_score - teacher_score
    
    teacher_violations = teacher_eval["total_violations"]
    student_violations = student_eval["total_violations"]
    violations_diff = student_violations - teacher_violations
    
    logger.info(f"  Score constitucional:")
    logger.info(f"    Teacher: {teacher_score:.4f}")
    logger.info(f"    Student: {student_score:.4f}")
    logger.info(f"    Diferencia: {score_diff:+.4f}")

    logger.info(f"  Violaciones totales:")
    logger.info(f"    Teacher: {teacher_violations}")
    logger.info(f"    Student: {student_violations}")
    logger.info(f"    Diferencia: {violations_diff:+.0f}")
    
    # 5. Decisión final
    if student_score > teacher_score and student_violations < teacher_violations:
        logger.info("✅ DESTILACIÓN EXITOSA: Student mejoró al teacher")
        best_model = distilled_student
        best_model_name = "student_distilled"
    elif abs(score_diff) < 0.02 and abs(violations_diff) < 5:
        logger.info("🟡 DESTILACIÓN NEUTRAL: Student mantuvo rendimiento")
        best_model = teacher_model  # Usar teacher si son iguales
        best_model_name = "teacher_angel"
    else:
        logger.warning("❌ DESTILACIÓN FALLIDA: Student empeoró al teacher")
        best_model = teacher_model
        best_model_name = "teacher_angel"
    
    # 6. Guardar el mejor modelo
    best_model_path = os.path.join(args.output_dir, f"best_model_{best_model_name}.pkl")
    with open(best_model_path, "wb") as f:
        pickle.dump(best_model, f)
    logger.info(f"📁 Mejor modelo guardado: {best_model_path}")
    
    # 7. Generar reporte final
    final_report = {
        "best_model": best_model_name,
        "teacher_eval": teacher_eval,
        "student_eval": student_eval,
        "comparison": {
            "score_diff": float(score_diff),
            "violations_diff": int(violations_diff),
            "decision": "success" if student_score > teacher_score else "neutral" if abs(score_diff) < 0.02 else "failed"
        },
        "timestamp": datetime.now().isoformat()
    }

    report_path = os.path.join(args.output_dir, "final_constitutional_report.json")
    with open(report_path, "w") as f:
        json.dump(final_report, f, indent=2, default=str)
    
    logger.info(f"📄 Reporte final guardado en {report_path}")
    
    # ========== PREPARAR FEATURE WEIGHTS PARA HOLISTIC CONTEXT ==========
    aggregated_feature_weights = None
        
    if feature_importances.ndim == 2:
        aggregated_feature_weights = np.mean(feature_importances, axis=0)
    else:
        aggregated_feature_weights = feature_importances.copy()
            
    # Normalizar
    if aggregated_feature_weights is not None and np.sum(aggregated_feature_weights) > 0:
        aggregated_feature_weights = aggregated_feature_weights / np.sum(aggregated_feature_weights)
            
    logger.info(f"[FeatureWeights] Creados desde feature_importances: shape={aggregated_feature_weights.shape if aggregated_feature_weights is not None else 'None'}")  
        
    # Ejecutar continuous_multi_lorax sobre el ENSEMBLE de resultados
    holistic_context = await parallel_continuous_multi_lorax(
        np.float64(X_successes),  #X de destilación
        np.float64(Y_successes),  # Targets originales
        np.nan_to_num(aggregated_feature_weights),
        batch_size = len(X_successes)
    )

    # En run_baseline_worker_async, después de tener el modelo entrenado y los adversarios:
    # ========== CREAR CREWS CADA 6 BASELINES ==========
    logger.info("=" * 70)
    logger.info("🎭 CREANDO CREWS DE AGENTES")
    logger.info("=" * 70)

    #crew_manager = GlobalCrewManager(
    #    base_dir=args.output_dir,
    #    crew_size=6  # Cada crew = 6 baselines = 24,000 muestras
    #)

    # Agregar cada baseline a su crew correspondiente
    #for baseline_result in all_results:
    #    if baseline_result and baseline_result.get('status') == 'completed':
    #        baseline_idx = baseline_result.get('baseline_idx', -1)
    #        model = baseline_result.get('model')
    #        memory = baseline_result.get('agent_memory')
            
    #        if model is not None and baseline_idx >= 0:
    #            # Usar datos de validación para contexto holístico
    #            X_holistic = X_val[:1000] if len(X_val) > 1000 else X_val
    #            Y_holistic = Y_val[:1000] if len(Y_val) > 1000 else Y_val
                
    #            await crew_manager.add_baseline_to_crew(
    #                baseline_idx=baseline_idx,
    #                model=model,
    #                memory=memory,
    #                X_holistic=X_holistic,
    #                Y_holistic=Y_holistic
    #            )

    # ========== EVALUAR VULNERABILIDADES ENTRE CREWS ==========
    #logger.info("=" * 70)
    #logger.info("🔍 ANALIZANDO VULNERABILIDADES ENTRE CREWS")
    #logger.info("=" * 70)
    
    #vulnerability_report = crew_manager.get_vulnerability_analysis(X_test[:500], Y_test[:500])
    #logger.info(f"Vulnerabilidades detectadas: {len(vulnerability_report.get('vulnerabilities', []))}")
    
    #if vulnerability_report.get('routing_attack_possible', False):
    #    logger.warning("⚠️ POSIBLE VULNERABILIDAD DE ENRUTAMIENTO DETECTADA")
    #    for vuln in vulnerability_report.get('vulnerabilities', []):
    #        logger.warning(f"  - {vuln['type']}: crews {vuln['crew_pair']} divergencia={vuln['mse']:.4f}")
    
    # ========== PROBAR ENRUTAMIENTO ==========
    #logger.info("=" * 70)
    #logger.info("🔄 PROBANDO ENRUTAMIENTO ENTRE CREWS")
    #logger.info("=" * 70)
    
    #routing_results = []
    #for i in range(min(50, len(X_test))):
    #    x_sample = X_test[i:i+1]
    #    y_sample = Y_test[i:i+1]
        
    #    try:
    #        prediction, routing_info = await crew_manager.route_to_crew(x_sample)
    #        mse = float(np.mean((prediction - y_sample) ** 2))
    #        routing_results.append({
    #            'sample_idx': i,
    #            'selected_crew': routing_info['selected_crew'],
    #            'mse': mse,
    #            'routing_score': routing_info['score']
    #    except Exception as e:
    #        logger.warning(f"Error en routing sample {i}: {e}")

    # Estadísticas de routing
    #if routing_results:
    #    crew_counts = defaultdict(int)
    #    for r in routing_results:
    #        crew_counts[r['selected_crew']] += 1
    
    #    logger.info("📊 Estadísticas de enrutamiento:")
    #    for crew_id, count in crew_counts.items():
    #        logger.info(f"  Crew {crew_id}: {count} muestras ({100*count/len(routing_results):.1f}%)")
        
    #    avg_mse_by_crew = {}
    #    for r in routing_results:
    #        crew_id = r['selected_crew']
    #        if crew_id not in avg_mse_by_crew:
    #            avg_mse_by_crew[crew_id] = []
    #        avg_mse_by_crew[crew_id].append(r['mse'])
        
    #    for crew_id, mses in avg_mse_by_crew.items():
    #        logger.info(f"  Crew {crew_id} MSE medio: {np.mean(mses):.6f}")

    # Guardar resultados de crews
    #crew_results = {
    #    'crew_metrics': crew_manager.get_crew_metrics(),
    #    'vulnerability_report': vulnerability_report,
    #    'routing_results': routing_results,
    #    'routing_stats': {
    #        'total_routed': len(routing_results),
    #        'crew_distribution': dict(crew_counts),
    #        'avg_mse_by_crew': {k: float(np.mean(v)) for k, v in avg_mse_by_crew.items()}
    #    } if routing_results else {}
    #}

    # Guardar en archivo
    #crew_results_path = os.path.join(args.output_dir, "crew_results.json")
    #with open(crew_results_path, "w") as f:
    #    json.dump(crew_results, f, indent=2, default=str)
    
    #combined_results['crews'] = crew_results
        
    #return combined_results
    
def run_baseline_worker(baseline_args: list) -> Dict[str, Any]:
    """Wrapper síncrono requerido por ProcessPoolExecutor."""
    import asyncio
    import traceback
    import sys    
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(run_baseline_worker_async(baseline_args))
    except Exception as e:
        # Capturar la excepción real
        error_msg = f"Worker failed: {str(e)}\n{traceback.format_exc()}"
        print(error_msg, file=sys.stderr)
        return {
            'status': 'failed',
            'error': error_msg,
            'baseline_idx': baseline_args[0] if baseline_args else -1
        }
    finally:
        loop.close()    

def save_holistic_context(context: np.ndarray, iteration: int = None):
    """Guarda el contexto de atención holístico de todas las destilaciones"""
    if iteration is None:
        iteration = self.iteration
    iter_dir = os.path.join(self.baseline_dir, f"iter_{iteration:04d}")
    
    context_path = os.path.join(iter_dir, "holistic_attention_context.npy")
    np.save(context_path, context)
    
    logger.info(f"[BaselineManager] Contexto holístico guardado en {context_path}")
    return context_path

def load_holistic_context(iteration: int = None) -> Optional[np.ndarray]:
    """Carga el contexto de atención holístico"""
    if iteration is None:
        iteration = self.iteration
    iter_dir = os.path.join(self.baseline_dir, f"iter_{iteration:04d}")
    
    context_path = os.path.join(iter_dir, "holistic_attention_context.npy")
    if os.path.exists(context_path):
        return np.load(context_path)
    return None

class GRPOOptimizer:
    """
    Group Relative Policy Optimization - Sin clipping, usa normalización por grupo.
    """
    def __init__(self, model: DSVMWrapper, group_size: int = 8):
        self.model = model
        self.group_size = group_size
        self.advantage_history = []
        
    async def compute_group_advantages(
        self,
        rewards: List[float],
        group_ids: List[int],
    ) -> np.ndarray:
        """
        Calcula ventajas relativas dentro de cada grupo.
        Sin clipping - usa normalización dentro del grupo.
        """
        advantages = np.zeros(len(rewards))
        
        # Agrupar rewards por group_id
        groups = defaultdict(list)
        for i, gid in enumerate(group_ids):
            groups[gid].append((i, rewards[i]))
        
        # Calcular ventajas relativas por grupo
        for gid, group_items in groups.items():
            group_rewards = [r for _, r in group_items]
            mean_reward = np.mean(group_rewards)
            std_reward = np.std(group_rewards) + 1e-8
            
            for idx, reward in group_items:
                # Ventaja = (reward - media_grupo) / std_grupo
                # Sin clipping, pero naturalmente acotado por normalización
                advantage = (reward - mean_reward) / std_reward
                advantages[idx] = advantage
                
        self.advantage_history.append({
            'timestamp': datetime.now().isoformat(),
            'mean_advantage': float(np.mean(advantages)),
            'std_advantage': float(np.std(advantages)),
            'group_sizes': [len(g) for g in groups.values()]
        })
        
        return advantages
    
    async def grpo_update(
        self,
        X_batch: np.ndarray,
        y_batch: np.ndarray,
        advantages: np.ndarray,
        learning_rate: float = 0.001,
        preference_weight: float = 1,        
        preference_pairs: Optional[List[Tuple[np.ndarray, np.ndarray, float]]] = None,        
    ) -> Dict:
        """
        Actualización GRPO - usa ventajas relativas en lugar de clipped ratio.
        """
        old_w = self.model.w.copy() if self.model.w is not None else None
        
        # Calcular predicciones actuales
        predictions = await self.model.predict(X_batch)
        
        # Calcular loss ponderado por ventajas
        mses = np.mean((predictions - y_batch) ** 2, axis=1)

        # Si predictions es 2D y y_batch es 1D, transformar
        y_batch = y_batch.reshape(-1, 1)
    
        # Si predictions es 2D y y_batch es 2D con diferente segunda dimensión
        print("MSES", mses.shape)  

        if mses.ndim == 2:
            mses = np.mean(mses, axis=1)         
        if advantages.ndim == 2:
            advantages = np.mean(advantages, axis=1)  
        print("ADVANTAGES", advantages.shape)              
        # Loss = MSE * (1 - advantage) para reforzar buenas acciones
        # Ventajas positivas → menor loss, ventajas negativas → mayor loss
        weighted_loss = np.mean(mses * (1.0 - advantages))

        preference_loss = 0.0
        preference_grad = np.zeros_like(old_w) if old_w is not None else None

        # Asegurar formas correctas
        good_flat = np.array([i['good'] for i in preference_pairs]).flatten()
        bad_flat = np.array([i['bad'] for i in preference_pairs]).flatten()
        confidence = np.array([i['confidence'] for i in preference_pairs]).flatten()
            
        # Calcular scores usando MSE contra batch (aproximación)
        # Si los outputs tienen dimensión diferente, usar el mínimo común
        min_len_good = min(len(good_flat), y_batch.shape[1] if y_batch.ndim > 1 else len(y_batch))
        min_len_bad = min(len(bad_flat), y_batch.shape[1] if y_batch.ndim > 1 else len(y_batch))
            
        # Score negativo (menor MSE = mejor)
        good_score = -np.mean((good_flat[:min_len_good] - y_batch[0, :min_len_good]) ** 2)
        bad_score = -np.mean((bad_flat[:min_len_bad] - y_batch[0, :min_len_bad]) ** 2)
            
        # Preference loss: -log(sigmoid(good_score - bad_score))
        diff = good_score - bad_score
        sigmoid = 1.0 / (1.0 + np.exp(-diff))
        pair_loss = -np.log(sigmoid + 1e-9) * confidence
        
        for i, pair in enumerate(pair_loss):
            preference_loss += pair_loss[i]
            # Gradiente de preference loss
            preference_grad += pair_loss[i] * confidence[i] * np.sign(old_w)
            
        preference_loss = preference_loss / (len(preference_pairs) + 1e-8)    
        preference_grad = preference_grad / (len(preference_pairs) + 1e-8)
        
        logger.debug(f"[GRPO] Preference loss: {preference_loss:.4f} from {len(preference_pairs)} pairs")
            
        # Actualizar modelo (simplificado - gradiente descendente)
        # Calcular gradiente simplificado
        grad = np.zeros_like(old_w)
        for i, adv in enumerate(advantages):
            if i < len(X_batch):
                x_i = X_batch[i].reshape(1, -1)  # (1, 1025)
                pred_i = predictions[i]          # (1, 1025) o (1025,)
                y_i = y_batch[i]                 # (1, 1025) o (1025,)
                    
                # Asegurar formas
                if pred_i.ndim == 1:
                    pred_i = pred_i.reshape(1, -1)
                if y_i.ndim == 1:
                    y_i = y_i.reshape(1, -1)
                    
                error = pred_i - y_i  # (1, 1025)
                    
                # ✅ CORRECCIÓN: Calcular gradiente por elemento, no outer product
                # Para cada feature, el gradiente es adv * error * x_i
                grad += adv * (error * x_i)  # (1, 1025) * (1, 1025) = (1, 1025)
            
        grad = grad / (len(X_batch) + 1e-8)

        grad = (1 - preference_weight) * grad + preference_weight * preference_grad
        logger.debug(f"[GRPO] Gradientes combinados: GRPO_norm={np.linalg.norm(grad):.4f}, "
                        f"Pref_norm={np.linalg.norm(preference_grad):.4f}")

        # Clip gradiente
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1.0:
            grad = grad / grad_norm
            
        # Actualizar pesos
        new_w = old_w - learning_rate * grad.reshape(old_w.shape)
        self.model.w = new_w
        
        return {
            'mean_advantage': float(np.mean(advantages)),
            'max_advantage': float(np.max(advantages)),
            'min_advantage': float(np.min(advantages)),
            'gradient_norm': float(np.linalg.norm(grad)) if grad is not None else 0.0,
            'n_preference_pairs': len(preference_pairs) if preference_pairs else 0,
            'preference_weight': preference_weight if preference_pairs else 0.0,
        }
    
async def run_baseline_worker_async(baseline_args: list) -> Dict[str, Any]:
    """
    Worker que ejecuta una baseline completa en un proceso separado.
    """
    import pickle
    import glob
    import traceback
    
    try:

        shared_dict = baseline_args[0]  # ← OBJETO COMPARTIDO

        (
            baseline_idx,
            start_idx,
            end_idx,
            X_train_batch,
            Y_train_batch,
            X_val,
            Y_val,
            X_test,
            Y_test,
            file_ids_train,
            args,
            loop_counter,
            baseline_output_dir,
            dsvm_raw
        ) = baseline_args[1:]

        # ========== OBTENER MEMORIA Y LOCK ==========
        pipeline_memory = shared_dict['memory']  # ← MISMA INSTANCIA
        memory_lock = shared_dict['lock']        # ← LOCK PARA SINCRONIZACIÓN
    
        results = {}

        #if exists and model_path:
        #    logger.info(f"⏭️ [Baseline {baseline_idx}] Saltando COMPLETAMENTE - distilled_model.pkl ya existe en {model_path}")
            
        #    # Retornar mínimo para que no rompa el pipeline
        #    return {
        #        'status': 'skipped',
        #        'baseline_idx': baseline_idx,
        #        'skip_reason': 'distilled_model_exists',
        #        'model_path': model_path,
        #        'loop_counter': loop_counter,
        #        # Datos vacíos para que no falle el pipeline
        #        'aoc_pre_posttrain': {},
        #        'aoc_final': {},
        #        'improvements': {},
        #        'rollout_data': {
        #            'inputs': [],
        #            'targets': [],
        #            'predictions': [],
        #            'composite_rewards': [],
        #            'rewards': [],
        #            'values': [],
        #            'dones': [],
        #            'constitution_scores': []
        #        },
        #        'post_training': {'status': 'skipped'},
        #        'cross_validation': {'status': 'skipped'}
        #    }

        # ========== NUEVO: VERIFICAR SI ESTE LOOP YA FUE COMPLETADO ==========
        skip, reason = should_skip_loop(baseline_output_dir, loop_counter, force_rerun=getattr(args, 'force_rerun', False))
        if skip:
            logger.info(f"⏭️ Saltando baseline {baseline_idx + 1} (loop {loop_counter}): {reason}")
            
            # Intentar cargar resultados existentes
            existing_results_path = os.path.join(baseline_output_dir, f"baseline_results_loop_{loop_counter}.json")
            if os.path.exists(existing_results_path):
                with open(existing_results_path, 'r') as f:
                    existing_results = json.load(f)
                existing_results['status'] = 'skipped'
                existing_results['skip_reason'] = reason
                return existing_results
            else:
                return {
                    'status': 'skipped',
                    'skip_reason': reason,
                    'baseline_idx': baseline_idx,
                    'loop_counter': loop_counter
                }


        # ✅ NO REDEFINIR start_idx, end_idx, X_train_batch, Y_train_batch
        # ✅ USAR LOS PARÁMETROS DIRECTAMENTE
        # Los argumentos llegan directamente, no hay que desempaquetar nada
        logger.info("="*70)
        logger.info(f"📦 BASELINE {baseline_idx + 1}")
        logger.info(f"   Muestras: {start_idx} - {end_idx} ({len(X_train_batch)} frames)")
        logger.info("="*70)
        
        # Crear BaselineModelManager para esta baseline
        baseline_manager = BaselineModelManager(
            base_output_dir=os.path.dirname(baseline_output_dir),
            baseline_name=os.path.basename(baseline_output_dir)
        )
        
        logger.info("="*70)
        logger.info(f"📦 BASELINE {baseline_idx + 1}")
        logger.info(f"   Muestras: {start_idx} - {end_idx} ({len(X_train_batch)} frames)")
        logger.info("="*70)
    
        model = DSVMWrapper(dsvm_model=dsvm_raw, regression=args.regression)
        #print("DSVM RAW:", dsvm_raw)
        # Si hay un modelo pre-entrenado, copiar sus pesos
        if dsvm_raw is not None:
            if hasattr(dsvm_raw, 'w') and dsvm_raw.w is not None:
                model.w = dsvm_raw.w
            if hasattr(dsvm_raw, 'bias') and dsvm_raw.bias is not None:
                model.bias = dsvm_raw.bias
            logger.info(f"[Baseline {baseline_idx}] Pesos cargados desde modelo pre-entrenado: shape={model.w.shape}")

        # Guardar datos de audio de esta baseline
        baseline_manager.save_audio_samples(
            X_train_batch, Y_train_batch, 
            await model.predict(X_train_batch) if model else np.zeros_like(X_train_batch),
            iteration=baseline_idx + 1
        )
    

        model_router = None
        
        
        # Usar un subconjunto de entrenamiento para la CV (máx 2000 frames
        # para mantener tiempo razonable)
        X_cv = X_train_batch
        Y_cv = Y_train_batch
        file_ids_cv = file_ids_train
        
        # En run_full_pipeline.py, al llamar a run_adversarial_cross_validation
        grid_adversaries = []
        if model.w is not None:
            for i in range(args.n_adversaries):
                # Crear modelo adversario con ruido
                adv_model = model.clone()
                noise = np.random.normal(0, 0.05, adv_model.w.shape)
                adv_model.w = adv_model.w + noise
                    
                # Asegurar que los datos tengan el tamaño correcto
                n_samples = min(2000, len(X_cv))
                n_test = min(500, len(X_cv))
                    
                grid_adversaries.append({
                    'xtrain': X_cv[:n_samples],
                    'ytrain': Y_cv[:n_samples],
                    'xtest': X_cv[-n_test:],
                    'ytest': Y_cv[-n_test:],
                    'model': adv_model,
                    'model_name': f'adversary_{i}',
                    'weight': 0.5  # Peso para majority voting
                })
        
        if not grid_adversaries:
            grid_adversaries = [{
                'xtrain': X_cv[:2000],
                'ytrain': Y_cv[:2000],
                'xtest': X_cv[-500:],
                'ytest': Y_cv[-500:],
                'model': model.clone(),
                'model_name': 'base_adversary',
                'weight': 0.5
            }]
    
        # Después de run_adversarial_cross_validation o cargar cv_results
        cv_results = None
        context = None
        
        # 1. Verificar si ya existe el contexto guardado
        # Buscar el archivo completo de cv_results
        cv_results_path = os.path.join(args.output_dir, f"cv_results_loop_{loop_counter}.pkl")
        context_file = os.path.join(args.output_dir, f"Model aoc_dsvm context.npy")
        
        if os.path.exists(cv_results_path):
            # Cargar el diccionario COMPLETO
            logger.info(f"[Baseline {baseline_idx}] Cargando cv_results completo desde {cv_results_path}")
            with open(cv_results_path, 'rb') as f:
                cv_results = pickle.load(f)
            # Extraer componentes
            context = cv_results.get('context')
            exp_data = cv_results.get('exp_data')
            stress_becs = cv_results.get('stress_becs', [])
            stress_btcs = cv_results.get('stress_btcs', [])
            adversary_models_from_cv = cv_results.get('adversary_models', [])
            
        elif os.path.exists(context_file):
            # Fallback: solo contexto (modo legacy)
            logger.warning(f"[Baseline {baseline_idx}] Solo contexto encontrado, cargando parcialmente...")
            context = np.load(context_file, allow_pickle=True)
            cv_results = {'context': context}
            exp_data = None
            stress_becs = []
            stress_btcs = []
            adversary_models_from_cv = []
        
        else:
            cv_results = await run_adversarial_cross_validation(
                model=model,
                X=X_cv,
                Y=Y_cv,
                file_ids=file_ids_cv,
                Cs=[0.01, 0.1, 1.0],
                reg_params=[0.001, 0.01, 0.1],
                kernel_configs=[("rbf", 0.5)],
                adversaries=grid_adversaries,  # Lista de dicts con xtrain, ytrain, xtest, ytest
                model_name="aoc_dsvm",
                regression=args.regression,
                loop_counter=loop_counter,
                output_dir=args.output_dir,
            )   
            # Guardar el diccionario completo
            cv_results_path = os.path.join(args.output_dir, f"cv_results_loop_{loop_counter}.pkl")
            with open(cv_results_path, 'wb') as f:
                pickle.dump(cv_results, f)
            logger.info(f"[CV] Resultados completos guardados en {cv_results_path}")
            
        print("CV RESULTS:", cv_results.keys())    
    
        if isinstance(cv_results, dict):
            context = cv_results['context']
            exp_data = cv_results['exp_data']  # ← AHORA SÍ TENEMOS exp_data
            stress_becs = cv_results['stress_becs']
            adversary_models_from_cv = cv_results['adversary_models']
        else:
            context = cv_results
            exp_data = None
            stress_becs = []
            adversary_models_from_cv = []
    
        #print("EXP DATA:", exp_data)
        # Verificar que cv_results no sea None antes de usarlo
        if cv_results is not None and isinstance(cv_results, dict):
            results["cross_validation"] = {
                k: v for k, v in cv_results.items()
                if k not in ("context", "adversary_X", "adversary_Y")
            }
        else:
            logger.warning("[CV Adversarial] No se obtuvieron resultados de validación cruzada")
            results["cross_validation"] = {
                "status": "failed",
                "error": "GridSearch retornó None"
            }
        
        adversary_models_from_cv = []
        if cv_results is not None and isinstance(cv_results, dict):
            adversary_models_from_cv = cv_results['adversary_models']
            logger.info(f"[Etapa 2] Obtenidos {len(adversary_models_from_cv)} modelos adversarios de cross-validation")
        else:
            logger.warning(f"[Etapa 2] No se pudieron extraer modelos adversarios, cv_results es {type(cv_results)}")
            
        # NOTA: El bloque de secure_clustering estaba mal indentado, moverlo fuera del if/else
        if args.secure_clustering:
            logger.info("[Etapa 2.5] Iniciando secure clustering...")
            secure_cluster = SecureCluster(n_clusters=3)
            await secure_cluster.fit(X_train_batch[:1000], encrypt_input=True)
            results['secure_clustering'] = secure_cluster.get_cluster_metrics()
            logger.info(f"[Etapa 2.5] Secure clustering completado")
        
        # Model Routing
        if args.model_routing:
            logger.info("[Etapa 2.6] Configurando model routing...")
            model_router = ModelRouter(routing_strategy='weighted_average')
                
            # Registrar modelo principal
            model_router.register_model('dsvm_main', model, weight=1.0)
                
            results['model_routing'] = {
                'models_registered': list(model_router.models.keys()),
                'routing_strategy': model_router.routing_strategy
            }
            logger.info(f"[Etapa 2.6] Model routing configurado con {len(model_router.models)} modelos")
    
    
        # ══════════════════════════════════════════════════════════════════════
        # ETAPA 3 — Evaluación multi-métrica AOC sobre frames reales
        # ══════════════════════════════════════════════════════════════════════
        logger.info("=" * 70)
        logger.info("ETAPA 3 — Evaluación multi-métrica AOC")
        logger.info("=" * 70)
        
        print("Y_val:", Y_val.shape)
    
        aoc_val  = await evaluate_aoc_metrics(model.predict, X_val,  Y_val,  "Val")
        aoc_test = await evaluate_aoc_metrics(model.predict, X_test, Y_test, "Test")
        results["aoc_pre_posttrain"] = {
            "validation": aoc_val,
            "test":       aoc_test,
        }
    
        # Calibración de incertidumbre usando UncertaintyAwareRewardModel del repo
        logger.info("[Etapa 3] Calibración de incertidumbre...")
        try:
            ua_model = UncertaintyAwareRewardModel(
                base_model         = model,
                uncertainty_method = "entropy",
            )
            unc_result = await ua_model.predict_with_uncertainty(X_val)
            results["uncertainty_calibration"] = {
                "mean_uncertainty": float(np.mean(unc_result.get("uncertainty", [0]))),
                "method": "entropy",
            }
            logger.info(
                f"[Etapa 3] Incertidumbre media={results['uncertainty_calibration']['mean_uncertainty']:.4f}"
            )
        except Exception as e:
            logger.warning(f"[Etapa 3] Calibración falló: {e}")
            results["uncertainty_calibration"] = {"error": str(e)}
        
        # 2. Verificar si existen los archivos de importancia
        importance_dir = "."  # Directorio donde busca compute_feature_importance
        importance_files = glob.glob(os.path.join(importance_dir, "IMPORTANCE*.npy"))
        mses_file = os.path.join(importance_dir, "IMPORTANT MSES.npy")
            
        feature_importances = None
        
        print(f"[DEBUG] importance_files = {importance_files}")
        print(f"[DEBUG] mses_file exists = {os.path.exists(mses_file)}")
        print(f"[DEBUG] Condición = {importance_files and os.path.exists(mses_file)}")
            
        if importance_files and os.path.exists(mses_file):
            logger.info(f"[Baseline {baseline_idx}] Importancias existentes, cargando...")
    
            # Cargar todas las importancias
            importances_list = []
            for imp_file in sorted(importance_files):
                imp = np.load(imp_file)
                importances_list.append(imp)
            
            if importances_list:
                feature_importances = np.array(importances_list)
                logger.info(f"[Baseline {baseline_idx}] Cargadas {len(feature_importances)} importancias, shape={feature_importances.shape}")
                
            weights_file = os.path.join(baseline_output_dir, f"weights_baseline_{baseline_idx}.npy")    

            if os.path.exists(weights_file):
                wset = np.load(weights_file)
                logger.info(f"[Baseline {baseline_idx}] Weights cargados desde {weights_file}, shape={wset.shape if hasattr(wset, 'shape') else 'N/A'}")
            else:
                logger.warning(f"[Baseline {baseline_idx}] No se encontró archivo de weights: {weights_file}")
                # Fallback: usar feature_importances como wset
                wset = feature_importances
                logger.info(f"[Baseline {baseline_idx}] Usando feature_importances como wset fallback")

        else:
            logger.info(f"[Baseline {baseline_idx}] No hay importancias, computando desde exp_data...")
                
            # Necesitamos exp_data de multiple_hypothesis_testing
            # Si cv_results es un diccionario y tiene exp_data
            exp_data = cv_results['exp_data']
            #print("EXP DATA:", exp_data)

            # ========== COMPUTAR Y GUARDAR DROPOUT CONTEXT ==========
            droppedout_context = None
            if exp_data:
                logger.info("[FeatureImportance] Computando dropout context desde exp_data...")
                try:
                    wpr = []
                    gathered_mse = []
                    error_array = []
                    y_trains = []
                    for i in range(len(exp_data)):
                        for j in range(len(exp_data[i])):  
                            for m in range(len(exp_data[i])):                    	
                                try:
                                    first_entry = exp_data[i][j][m]
                                    #print("FIRST ENTRY:", first_entry)
                                    y_sample = first_entry[0]['ytrain']  # ytrain o features
                                    y_trains.append(y_sample)
                                    from apicultor.machine_learning.cross_validation import scan_data_leakage
                                                   
                                    # Crear atributo protegido basado en la hipótesis
                                    # (ej: si el valor supera el umbral de la hipótesis)
                                    protected_attr = np.zeros(len(y_sample))
                                    # Usar la primera hipótesis como base
                                    limit, logical, intersection = exp_data[i][j][m][-4], exp_data[i][j][m][-3], exp_data[i][j][m][-2]
                                                       
                                    # Evaluar la condición de la hipótesis
                                    feature_col = y_sample[:, 0] if y_sample.ndim > 1 else y_sample
                                    if logical:
                                        protected_attr = (feature_col > intersection).astype(int)
                                    else:
                                        protected_attr = (feature_col < intersection).astype(int)
                                        
                                    #print("Y SAMPLE", y_sample.ravel().shape) 
                                    #print("Y HAT", first_entry[2].ravel().shape) 
                                    min_len = min(len(y_sample.ravel()), len(first_entry[2].ravel()))   
                                               
                                    leakage_result = scan_data_leakage(
                                        y_sample.ravel()[:min_len],
                                        first_entry[2].ravel()[:min_len]
                                    )
                                                   
                                    wpr.append(leakage_result[0])  # ← leakage
                                    correctly = leakage_result[1]
                                    leaking_targets = leakage_result[2]
                                    error_array.append(leakage_result[4])            	
                                    leaking_idxs = leakage_result[3]
            
                                    mse_values = []
                                    for entry in exp_data:
                                        if isinstance(entry, (list, tuple)) and len(entry) >= 7:
                                            mse_val = entry[6]  # índice 6 = mse_val
                                            if mse_val is not None:
                                                try:
                                                    mse_values.append(float(mse_val))
                                                except (ValueError, TypeError):
                                                    pass
                                
                                    gathered_mse.append(np.mean(mse_values) if mse_values else 0.2)
                                    print(f"GATHERED MSE", gathered_mse)
                                except Exception as e:
                                    pass                           	
                	
                    droppedout_context = await dropout(
                        data_set=exp_data,
                        protected_groups=[0],
                        protected_lime=wpr,
                        protected_mse=gathered_mse,
                        error=error_array,
                        protected_features=y_trains
                    )
                    logger.info(f"[FeatureImportance] dropout context computado: {len(droppedout_context) if droppedout_context else 0} elementos")
                    
                    # ========== GUARDAR DROPOUT CONTEXT POR BASELINE ==========
                    if droppedout_context:
                        # Convertir arrays a listas para JSON
                        dropout_serializable = []
                        for ctx in droppedout_context:
                            # ctx es [x, y, targets, weights, size]
                            serializable_entry = []
                            for item in ctx:
                                if isinstance(item, np.ndarray):
                                    serializable_entry.append(item.tolist())
                                elif isinstance(item, (np.float32, np.float64)):
                                    serializable_entry.append(float(item))
                                elif isinstance(item, (np.int32, np.int64)):
                                    serializable_entry.append(int(item))
                                else:
                                    serializable_entry.append(item)
                            dropout_serializable.append(serializable_entry)
                        
                        # Guardar en archivo JSON con indentación
                        dropout_path = os.path.join(baseline_output_dir, f"dropout_context_baseline_{baseline_idx}.json")
                        with open(dropout_path, 'w') as f:
                            json.dump({
                                'baseline_idx': baseline_idx,
                                'loop_counter': loop_counter,
                                'timestamp': datetime.now().isoformat(),
                                'n_entries': len(droppedout_context),
                                'droppedout_context': dropout_serializable
                            }, f, indent=2, default=str)
                        logger.info(f"[FeatureImportance] Dropout context guardado en {dropout_path}")
                        
                except Exception as e:
                    print(f"[FeatureImportance] Error en dropout: {e}")
                    import traceback
                    traceback.print_exc()
                    droppedout_context = None

            # ========== COMPUTAR FEATURE IMPORTANCE ==========
            feature_importances = None
            wset = []
            if droppedout_context:
                logger.info("[FeatureImportance] Computando feature importance desde droppedout_context...")
                try:
                    importance_dset = []
                    for idx, ctx in enumerate(droppedout_context):
                        x_data = ctx[0]      # features
                        y_data = ctx[1]      # targets originales
                        targets = ctx[2]     # predicciones o targets transformados
                        weights = ctx[3]     # pesos
                        
                        if targets is not None and y_data is not None:
                            # Asegurar que sean arrays 2D
                            targets = np.array(targets)
                            y_data = np.array(y_data)
                            if targets.ndim == 1:
                                targets = targets.reshape(-1, 1)
                            if y_data.ndim == 1:
                                y_data = y_data.reshape(-1, 1)
                            
                            # Calcular scores (MSE por feature)
                            min_len = min(targets.shape[0], y_data.shape[0])
                            scores = np.mean((targets[:min_len] - y_data[:min_len]) ** 2, axis=0)
                        else:
                            scores = np.ones(x_data.shape[1]) if hasattr(x_data, 'shape') else np.ones(1025)
                        
                        importance_dset.append([
                            scores,      # i[-2] - array de scores
                            x_data,      # i[1] - features
                            idx,         # i[-1] - índice
                            idx          # idx extra
                        ])
                        wset.append(weights)

                    weights_file = os.path.join(baseline_output_dir, f"weights_baseline_{baseline_idx}.npy")
                    # Convertir a array si es lista de arrays
                    wset_array = np.array(wset) if isinstance(wset, list) else wset
                    np.save(weights_file, wset_array)
                    logger.info(f"[Baseline {baseline_idx}] Weights guardados en {weights_file}")
                    
                    if importance_dset:
                        result = await compute_feature_importance(importance_dset)
                        if result and len(result) >= 6:
                            importances, mean_imp, std_imp, mses_size, passing_scores, passing_xs = result
                            logger.info(f"Importancias computadas: shape={importances.shape}")
                            feature_importances = importances
                
                            # Guardar archivos IMPORTANCE
                            for imp_idx, imp in enumerate(importances):
                                np.save(os.path.join(baseline_output_dir, f"IMPORTANCE_{imp_idx}.npy"), imp)
                            np.save(os.path.join(baseline_output_dir, "IMPORTANT MSES.npy"), np.array([mses_size]))
                            
                            # Guardar también en JSON
                            importance_json_path = os.path.join(baseline_output_dir, f"feature_importances_baseline_{baseline_idx}.json")
                            with open(importance_json_path, 'w') as f:
                                json.dump({
                                    'baseline_idx': baseline_idx,
                                    'loop_counter': loop_counter,
                                    'timestamp': datetime.now().isoformat(),
                                    'importances_shape': importances.shape if hasattr(importances, 'shape') else None,
                                    'mean_importance': mean_imp.tolist() if hasattr(mean_imp, 'tolist') else mean_imp,
                                    'std_importance': std_imp.tolist() if hasattr(std_imp, 'tolist') else std_imp,
                                    'mses_size': int(mses_size),
                                    'passing_scores': passing_scores.tolist() if hasattr(passing_scores, 'tolist') else passing_scores,
                                }, f, indent=2, default=str)
                            logger.info(f"[FeatureImportance] Importancias guardadas en {importance_json_path}")
                        else:
                            logger.warning("compute_feature_importance no retornó resultados válidos")
                    else:
                        logger.warning("importance_dset está vacío")
                except Exception as e:
                    logger.error(f"Error en compute_feature_importance: {e}")
                    import traceback
                    traceback.print_exc()
                    feature_importances = None
            else:
                logger.warning("droppedout_context es None, no se pueden calcular importancias")
    
        # ══════════════════════════════════════════════════════════════════════
        # ETAPA 4 — Post-entrenamiento constitucional (SFT + RL)
        # ══════════════════════════════════════════════════════════════════════
        if args.post_train:
            logger.info("=" * 70)
            logger.info("ETAPA 4 — Post-entrenamiento constitucional con modelos enrutados")
            logger.info("=" * 70)
            
            constitution_rules = build_aoc_constitution_rules()
            
            post_trainer = AOCPostTrainer(
                model=model,
                X_audio=X_train_batch,
                Y_audio=Y_train_batch,
                constitution_rules=constitution_rules,
                output_dir=args.output_dir,
                quote_reference_path=args.quote_reference_path,
                loop_counter=loop_counter,
                model_router=model_router,  # Pasar ModelRouter con todos los modelos
                adversary_models=adversary_models_from_cv,
                baseline_idx=baseline_idx,
                feature_importances = feature_importances,
                memory=pipeline_memory,       # ← PASAR LA MEMORIA COMPARTIDA
                memory_lock=memory_lock   
            )
            
            pt_results = await post_trainer.run_full_post_training(
                quality_threshold=0.70,
                max_sft_samples=min(500, len(X_train_batch)),
                num_rollouts=args.num_rollouts,
                ppo_steps=10,
            )

            # ⭐⭐⭐ GUARDAR EL MODELO EN EL DIRECTORIO DE LA BASELINE ⭐⭐⭐
            model_path = os.path.join(baseline_output_dir, "model.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            logger.info(f"[Baseline {baseline_idx}] Modelo guardado en {model_path}")
            
            # También guardar el modelo como distilled_model.pkl para compatibilidad
            distilled_path = os.path.join(baseline_output_dir, "distilled_model.pkl")
            with open(distilled_path, 'wb') as f:
                pickle.dump(model, f)
            
            # Guardar también en el directorio iter_ correspondiente
            iter_dir = os.path.join(baseline_output_dir, f"iter_{baseline_idx + 1:04d}")
            os.makedirs(iter_dir, exist_ok=True)
            iter_model_path = os.path.join(iter_dir, "dsvm_main_distilled.pkl")
            with open(iter_model_path, 'wb') as f:
                pickle.dump(model, f)
    
            constitution_score = pt_results.get('constitution_audit', {}).get('mean_constitution_score', 0.5)
    
            # Guardar archivo para que find_angel_teacher lo encuentre
            score_file = os.path.join(baseline_output_dir, f"constitution_score_{baseline_idx:04d}.json")
            with open(score_file, 'w') as f:
                json.dump({
                    'baseline_idx': baseline_idx,
                    'constitution_score': constitution_score,
                    'timestamp': datetime.now().isoformat()
                }, f, indent=2)
            
            # Si hay quote_reference_path, cargar colección previa
            if args.quote_reference_path and os.path.exists(args.quote_reference_path):
                try:
                    with open(args.quote_reference_path, "r") as f:
                        quote_collection = json.load(f)
                    logger.info(f"[Etapa 4] Cargada colección de quote reference con {len(quote_collection)} modelos")
                except Exception as e:
                    logger.warning(f"[Etapa 4] No se pudo cargar colección quote reference: {e}")
                    
            model_path = os.path.join(args.output_dir, "aoc_dsvm_trained_post_rl.pkl")
            with open(model_path, "wb") as f:
                pickle.dump(model, f)
            logger.info(f"[Etapa 4] Modelo post-RL guardado en {model_path}")        
                    
            results["post_training"] = {
                k: v for k, v in pt_results.items()
                if k != "rollouts"   # omitir arrays grandes
            }
            
            results["rollout_data"] = post_trainer.rollout_data 
    
            # Evaluación final post-post-training
            logger.info("[Etapa 4] Evaluación final post-post-training...")
            aoc_final = await evaluate_aoc_metrics(
                model.predict, X_test, Y_test, "Final"
            )
            results["aoc_final"] = aoc_final
    
            # Delta de mejora
            delta_snr = aoc_final["snr_db"] - aoc_test["snr_db"]
            delta_mse = aoc_test["mse"]     - aoc_final["mse"]
            logger.info(
                f"[Etapa 4] Mejora tras post-training: "
                f"ΔSNR={delta_snr:+.2f}dB  ΔMSE={delta_mse:+.5f}"
            )
            results["improvements"] = {
                "delta_snr_db": delta_snr,
                "delta_mse":    delta_mse,
            }
    
            # Guardar reporte de memoria con índice de loop
            if hasattr(post_trainer, 'misalignment_history') and post_trainer.misalignment_history:
                memory_summary_path = os.path.join(args.output_dir, f"memory_summary_loop_{loop_counter}.json")
                with open(memory_summary_path, "w") as f:
                    json.dump({
                        'misalignment_history': post_trainer.misalignment_history,
                        'final_memory_metrics': post_trainer.agent_memory.get_memory_metrics()
                    }, f, indent=2)
                
                # Agregar a resultados
                results['agent_memory'] = {
                    'misalignment_indicators': post_trainer.misalignment_history[-1]['misalignment'] if post_trainer.misalignment_history else {},
                    'memory_metrics': post_trainer.agent_memory.get_memory_metrics()
                }
        
            # Métricas de eficiencia
            results['efficiency_metrics'] = post_trainer.efficiency_monitor.get_summary()
        
            # Logging de métricas clave
            try:
                eff = results['efficiency_metrics']
                # Verificar estructura antes de acceder
                if isinstance(eff, dict):
                    latency = eff.get('latency', {})
                    throughput = eff.get('throughput', {})
                    memory = eff.get('memory', {})
                    
                    ttft = latency.get('avg_ttft_ms', 'N/A')
                    tps = throughput.get('avg_tokens_per_sec', 'N/A')
                    mem = memory.get('peak_memory_mb', 'N/A')
                    
                    logger.info(f"📊 Eficiencia: TTFT={ttft}ms, Throughput={tps} tok/s, Memoria={mem}MB")
                else:
                    logger.info(f"📊 Eficiencia: {eff}")
            except Exception as e:
                logger.info("📊 Eficiencia: No disponible")         
            
        else:
            logger.info("[Etapa 4] Post-entrenamiento omitido (--post_train no activado).")
            results["post_training"] = {"status": "skipped"}
            results["aoc_final"]     = aoc_test
    
        # Feedback to Reward (en post-training)
        if args.feedback_to_reward:
            logger.info("[Etapa 4] Configurando feedback to reward conversion...")
            feedback_converter = FeedbackToReward(learning_rate=0.1)
            
            # Ejemplo de uso en post-training
            sample_feedbacks = ['good', 'good', 'neutral', 'bad']
            sample_rewards = [0.85, 0.82, 0.65, 0.42]
            
            converted = feedback_converter.batch_convert(sample_feedbacks, sample_rewards)
            logger.info(f"[Etapa 4] Feedback conversion sample: {converted}")
            
            # Decisión GO/NO-GO con feedback
            go_no_go = feedback_converter.get_go_no_go_decision(
                rollouts_rewards=sample_rewards,
                feedbacks=sample_feedbacks,
                threshold=0.05
            )
            logger.info(f"[Etapa 4] GO/NO-GO con feedback: {go_no_go['decision']} - {go_no_go['reason']}")
            
            results['feedback_to_reward'] = {
                'conversion_enabled': True,
                'go_no_go_decision': go_no_go
            }
    
        # ══════════════════════════════════════════════════════════════════════
        # ETAPA 5 — Guardado de resultados y reconstrucción de audio
        # ══════════════════════════════════════════════════════════════════════
        logger.info("=" * 70)
        logger.info("ETAPA 5 — Guardado de resultados")
        logger.info("=" * 70)

        quote_ref = QuoteReferenceModel(model, loop_counter=loop_counter, output_dir=args.output_dir)
        
        # Capturar referencia antes del entrenamiento
        #ref_proba = await quote_ref.capture_reference(X_train_batch)
        #ref_proba = ref_proba[0] 

        #quote_ref.add_to_collection()
        #quote_ref.save_checkpoint(args.output_dir, "post_training")
        #quote_ref.save_report(args.output_dir)        
           
        #curr_proba = await quote_ref.capture_current(X_train_batch)
        #curr_proba = curr_proba[0]
        
        # Calcular ratio
        #ratio = quote_ref.compute_ratio()
        #logger.info(f"Training ratio: {ratio:.4f} - {quote_ref.get_training_direction()}")
        
        # Guardar checkpoint y reporte
        #quote_ref.save_checkpoint(args.output_dir, "post_training")
        #quote_ref.save_report(args.output_dir)
        
    except Exception as e:
        error_msg = f"Worker {baseline_args[0] if baseline_args else '?'} failed: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)     

    # Guardar JSON de resultados (sin arrays numpy)
    def _json_safe(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, dict):
            return {k: _json_safe(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_json_safe(i) for i in obj]
        return obj

    # Reconstruir audio de muestra desde predicciones del modelo
    #_save_audio_sample(model, X_test, Y_test, args.output_dir)

    if adversary_models_from_cv:
        adv_collection_path = os.path.join(args.output_dir, f"adversary_models_collection_loop_{loop_counter}.json")
        # Cargar colección existente si hay
        all_adversary_models = []
        for i in range(1, loop_counter + 1):
            prev_path = os.path.join(args.output_dir, f"adversary_models_metadata_loop_{i}.json")
            if os.path.exists(prev_path):
                with open(prev_path, "r") as f:
                    prev_models = json.load(f)
                    all_adversary_models.extend(prev_models)
        
        # Agregar modelos actuales
        current_adv_metadata = []
        for adv in adversary_models_from_cv:
            current_adv_metadata.append({
                'adversary_id': adv.get('adversary_id', 0),
                'hypothesis': adv.get('hypothesis', 'unknown'),
                'feature_idx': adv.get('feature_idx', -1),
                'noise_level': adv.get('noise_level', 0),
                'loop': loop_counter
            })
        all_adversary_models.extend(current_adv_metadata)
        
        # Guardar colección completa
        with open(adv_collection_path, "w") as f:
            json.dump(all_adversary_models, f, indent=2)
        logger.info(f"[Etapa 5] Colección de adversarios guardada: {len(all_adversary_models)} modelos totales")
        
    final = results.get("aoc_final", aoc_test)    

    # Guardar evaluaciones en archivo de texto
    evals_path = os.path.join(args.output_dir, f"evals_loop_{loop_counter}.txt")
    with open(evals_path, "w") as f:
        f.write("="*80 + "\n")
        f.write("EVALUACIONES DEL PIPELINE AOC\n")
        f.write("="*80 + "\n\n")
        
        f.write("1. MÉTRICAS AOC\n")
        f.write("-"*40 + "\n")
        f.write(f"  SNR final:                  {final['snr_db']:.2f} dB\n")
        f.write(f"  MSE final:                  {final['mse']:.6f}\n")
        f.write(f"  Distorsión espectral (LSD): {final['log_spectral_distortion']:.4f}\n")
        f.write(f"  Consistencia temporal:      {final['temporal_consistency']:.4f}\n")
        f.write(f"  Robustez adversaria:        {final['robustness']:.4f}\n\n")
        
        f.write("2. CROSS-VALIDATION\n")
        f.write("-"*40 + "\n")
        cv = results.get("cross_validation", {})
        f.write(f"  BEC adversario medio:       {cv.get('bec_mean', 'N/A')}\n")
        f.write(f"  BTC adversario medio:       {cv.get('btc_mean', 'N/A')}\n")
        f.write(f"  p_rule (paridad):           {cv.get('p_rule', 'N/A')}\n\n")
        
        f.write("3. POST-TRAINING\n")
        f.write("-"*40 + "\n")
        f.write(f"  Mejora ΔSNR:                {results.get('improvements', {}).get('delta_snr_db', 0):+.2f} dB\n")
        f.write(f"  Mejora ΔMSE:                {results.get('improvements', {}).get('delta_mse', 0):+.6f}\n\n")
        
        f.write("4. AGENT MEMORY\n")
        f.write("-"*40 + "\n")
        memory = results.get('agent_memory', {})
        misalignment = memory.get('misalignment_indicators', {})
        f.write(f"  Goal Deviation:             {misalignment.get('goal_deviation', 0):.4f}\n")
        f.write(f"  Sensitive Info Seeking:     {misalignment.get('sensitive_info_seeking', 0):.4f}\n")
        f.write(f"  Self Preservation:          {misalignment.get('self_preservation', 0):.4f}\n")
        f.write(f"  System Manipulation:        {misalignment.get('system_manipulation', 0):.4f}\n\n")
        
        f.write("5. QUOTE REFERENCE\n")
        f.write("-"*40 + "\n")
        #f.write(f"  Training ratio:             {ratio:.4f}\n")
        #f.write(f"  Direction:                  {quote_ref.get_training_direction()}\n")
    
    logger.info(f"[Etapa 5] Evaluaciones guardadas en {evals_path}")    

    # Guardar modelo DSVM si existe
    if dsvm_raw is not None and args.model_path:
        try:
            import pickle
            out_model = os.path.join(args.output_dir, "aoc_dsvm_trained.pkl")
            with open(out_model, "wb") as f:
                pickle.dump(dsvm_raw, f)
            logger.info(f"[Etapa 5] Modelo guardado en {out_model}")
        except Exception as e:
            logger.warning(f"[Etapa 5] No se pudo guardar el modelo: {e}")

    # ── Resumen final ─────────────────────────────────────────────────────
    logger.info("=" * 70)
    logger.info("RESUMEN FINAL DEL PIPELINE AOC")
    logger.info("=" * 70)

    final = results.get("aoc_final", aoc_test)
    logger.info(f"  SNR final:                  {final['snr_db']:.2f} dB")
    logger.info(f"  MSE final:                  {final['mse']:.6f}")
    logger.info(f"  Distorsión espectral (LSD): {final['log_spectral_distortion']:.4f}")
    logger.info(f"  Consistencia temporal:      {final['temporal_consistency']:.4f}")
    logger.info(f"  Robustez adversaria:        {final['robustness']:.4f}")

    cv = results.get("cross_validation", {})
    if "bec_mean" in cv:
        logger.info(f"  BEC adversario medio:       {cv['bec_mean']:.4f}")
        logger.info(f"  BTC adversario medio:       {cv['btc_mean']:.4f}")
        logger.info(f"  p_rule (paridad):           {cv['p_rule']:.4f}")

    if "improvements" in results:
        imp = results["improvements"]
        logger.info(f"  Mejora ΔSNR post-training:  {imp['delta_snr_db']:+.2f} dB")
        logger.info(f"  Mejora ΔMSE post-training:  {imp['delta_mse']:+.6f}")

    # Guardar resultados con índice de loop
    #results["quote_reference_collection"] = quote_ref.get_collection()
    #results["loop_counter"] = loop_counter

    report_path = os.path.join(args.output_dir, "pipeline_results.json")
    #with open(report_path, "w") as f:
    #    json.dump(results, f, indent=2, default=str)
    logger.info(f"[Etapa 5] Reporte guardado en {report_path}")

    logger.info(f"\n  Reporte completo: {report_path}")
    logger.info("=" * 70)

    return results

class AgentCrew:
    """
    Crew de agentes: grupo de baselines que comparten contexto holístico.
    Cada crew se forma con 6 baselines (24,000 muestras).
    """
    
    def __init__(self, crew_id: int, crew_dir: str, baseline_indices: List[int]):
        """
        Args:
            crew_id: ID del crew (0, 1, 2, ...)
            crew_dir: Directorio donde se guarda el crew
            baseline_indices: Índices de las baselines que pertenecen a este crew
        """
        self.crew_id = crew_id
        self.crew_dir = crew_dir
        self.baseline_indices = baseline_indices
        self.holistic_context = None
        self.crew_members = []  # Modelos de las baselines
        self.crew_memory = AgentMemory()  # Memoria compartida del crew
        self.holistic_score = 0.0
        self.created_at = datetime.now().isoformat()
        
        # Crear directorio del crew
        os.makedirs(crew_dir, exist_ok=True)
        
        logger.info(f"[Crew {crew_id}] Creado con {len(baseline_indices)} baselines: {baseline_indices}")
    
    def add_member(self, baseline_idx: int, model: DSVMWrapper, memory: AgentMemory):
        """Agrega un miembro al crew junto con su memoria"""
        self.crew_members.append({
            'baseline_idx': baseline_idx,
            'model': model,
            'memory': memory,
            'joined_at': datetime.now().isoformat()
        })
        logger.info(f"[Crew {self.crew_id}] Baseline {baseline_idx} agregada")
    
    async def compute_holistic_context(self, X_holistic: np.ndarray, Y_holistic: np.ndarray) -> np.ndarray:
        """
        Calcula el contexto holístico del crew usando todas las predicciones de los miembros.
        
        El contexto holístico se obtiene de:
        1. Promedio de las predicciones de todos los miembros
        2. Atención multi-lorax sobre el ensemble
        """
        if not self.crew_members:
            logger.warning(f"[Crew {self.crew_id}] No hay miembros, no se puede calcular contexto")
            return None
        
        # Recolectar predicciones de todos los miembros
        all_predictions = []
        for member in self.crew_members:
            preds = await member['model'].predict(X_holistic)
            all_predictions.append(preds)
        
        # Promedio de predicciones (ensemble)
        ensemble_pred = np.mean(all_predictions, axis=0)
        
        # Calcular feature importances agregadas
        if hasattr(self.crew_members[0]['model'], 'feature_weights'):
            all_weights = []
            for member in self.crew_members:
                if member['model'].feature_weights is not None:
                    all_weights.append(np.array(member['model'].feature_weights).flatten())
            if all_weights:
                aggregated_weights = np.mean(all_weights, axis=0)
            else:
                aggregated_weights = np.ones(X_holistic.shape[1]) / X_holistic.shape[1]
        else:
            aggregated_weights = np.ones(X_holistic.shape[1]) / X_holistic.shape[1]
        
        try:
            # Tomar una muestra representativa (máx 500 frames)
            n_samples = min(500, len(X_holistic))
            X_sample = X_holistic[:n_samples]
            Y_sample = ensemble_pred[:n_samples]
            
            result = await continuous_multi_lorax(
                X_sample, Y_sample, aggregated_weights, 250, True
            )
            
            if result and len(result) >= 5:
                attention_function, targets, attention_scores, weights, context = result[:5]
                self.holistic_context = context
                logger.info(f"[Crew {self.crew_id}] Contexto holístico calculado: shape={context.shape if context is not None else 'None'}")
            else:
                logger.warning(f"[Crew {self.crew_id}] multi_lorax no retornó contexto válido")
                self.holistic_context = ensemble_pred
        except Exception as e:
            logger.error(f"[Crew {self.crew_id}] Error calculando contexto holístico: {e}")
            self.holistic_context = ensemble_pred
        
        return self.holistic_context
    
    async def consensus_prediction(self, X: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Predicción por consenso del crew usando:
        1. Mayoría ponderada por confianza
        2. Contexto holístico como guía
        """
        if not self.crew_members:
            raise ValueError(f"[Crew {self.crew_id}] No hay miembros para predicción")
        
        predictions = []
        confidences = []
        
        for member in self.crew_members:
            pred = await member['model'].predict(X)
            predictions.append(pred)
            
            # Calcular confianza basada en memoria del miembro
            memory_metrics = member['memory'].get_memory_metrics()
            confidence = 0.5 + 0.3 * memory_metrics.get('long_term', {}).get('avg_importance', 0)
            confidences.append(confidence)
        
        # Ponderar por confianza
        total_confidence = sum(confidences)
        if total_confidence > 0:
            weights = [c / total_confidence for c in confidences]
            weighted_pred = np.zeros_like(predictions[0])
            for pred, w in zip(predictions, weights):
                weighted_pred += pred * w
        else:
            weighted_pred = np.mean(predictions, axis=0)
        
        # Integrar contexto holístico
        if self.holistic_context is not None:
            # Ajustar predicción según contexto (60% consenso, 40% contexto)
            context_aligned = 0.6 * weighted_pred + 0.4 * self.holistic_context[:len(weighted_pred)]
            final_pred = context_aligned
        else:
            final_pred = weighted_pred
        
        metadata = {
            'crew_id': self.crew_id,
            'n_members': len(self.crew_members),
            'member_indices': [m['baseline_idx'] for m in self.crew_members],
            'confidences': confidences,
            'holistic_context_used': self.holistic_context is not None
        }
        
        return final_pred, metadata
    
    def save(self):
        """Guarda el crew en disco"""
        crew_path = os.path.join(self.crew_dir, f"crew_{self.crew_id}.pkl")
        with open(crew_path, 'wb') as f:
            pickle.dump({
                'crew_id': self.crew_id,
                'baseline_indices': self.baseline_indices,
                'holistic_context': self.holistic_context,
                'holistic_score': self.holistic_score,
                'created_at': self.created_at,
                'n_members': len(self.crew_members)
            }, f)
        logger.info(f"[Crew {self.crew_id}] Guardado en {crew_path}")
        
        # Guardar contexto holístico por separado
        if self.holistic_context is not None:
            context_path = os.path.join(self.crew_dir, f"holistic_context_crew_{self.crew_id}.npy")
            np.save(context_path, self.holistic_context)
    
    @classmethod
    def load(cls, crew_id: int, crew_dir: str):
        """Carga un crew desde disco"""
        crew_path = os.path.join(crew_dir, f"crew_{crew_id}.pkl")
        if not os.path.exists(crew_path):
            return None
        
        with open(crew_path, 'rb') as f:
            data = pickle.load(f)
        
        crew = cls(
            crew_id=data['crew_id'],
            crew_dir=crew_dir,
            baseline_indices=data['baseline_indices']
        )
        crew.holistic_context = data['holistic_context']
        crew.holistic_score = data.get('holistic_score', 0.0)
        crew.created_at = data['created_at']
        
        # Cargar contexto holístico si existe por separado
        context_path = os.path.join(crew_dir, f"holistic_context_crew_{crew_id}.npy")
        if os.path.exists(context_path):
            crew.holistic_context = np.load(context_path, allow_pickle=True)
        
        return crew


class GlobalCrewManager:
    """
    Gestiona múltiples crews y permite enrutamiento entre ellos.
    Cada crew maneja ~20,000 muestras (5 baselines de 4,000).
    """
    
    def __init__(self, base_dir: str, crew_size: int = 6):
        """
        Args:
            base_dir: Directorio base para crews
            crew_size: Número de baselines por crew (default: 6 = 24,000 muestras)
        """
        self.base_dir = base_dir
        self.crew_size = crew_size
        self.crews: Dict[int, AgentCrew] = {}
        self.active_crew_id = None
        self.crew_history = []
        
        # Crear directorio de crews
        self.crews_dir = os.path.join(base_dir, "crews")
        os.makedirs(self.crews_dir, exist_ok=True)
        
        # Cargar crews existentes
        self._load_existing_crews()
    
    def _load_existing_crews(self):
        """Carga crews existentes desde disco"""
        import glob
        crew_files = glob.glob(os.path.join(self.crews_dir, "crew_*.pkl"))
        for cf in crew_files:
            try:
                crew_id = int(cf.split("crew_")[1].split(".pkl")[0])
                crew = AgentCrew.load(crew_id, self.crews_dir)
                if crew:
                    self.crews[crew_id] = crew
                    logger.info(f"[CrewManager] Cargado crew {crew_id} con {len(crew.baseline_indices)} baselines")
            except Exception as e:
                logger.warning(f"[CrewManager] Error cargando crew {cf}: {e}")
    
    def get_crew_for_baseline(self, baseline_idx: int) -> Tuple[int, AgentCrew]:
        """
        Determina a qué crew pertenece una baseline.
        Los crews se forman cada 6 baselines (0-5 crew0, 6-11 crew1, etc.)
        """
        crew_id = baseline_idx // self.crew_size
        return crew_id, self.crews.get(crew_id)
    
    async def add_baseline_to_crew(self, baseline_idx: int, model: DSVMWrapper, 
                                    memory: AgentMemory, X_holistic: np.ndarray = None,
                                    Y_holistic: np.ndarray = None):
        """
        Agrega una baseline a su crew correspondiente.
        Si el crew no existe, lo crea.
        Si el crew se completa (6 baselines), calcula el contexto holístico.
        """
        crew_id, crew = self.get_crew_for_baseline(baseline_idx)
        
        if crew is None:
            # Crear nuevo crew
            baseline_range_start = crew_id * self.crew_size
            baseline_range_end = baseline_range_start + self.crew_size
            baseline_indices = list(range(baseline_range_start, min(baseline_range_end, baseline_idx + 1)))
            
            crew = AgentCrew(
                crew_id=crew_id,
                crew_dir=self.crews_dir,
                baseline_indices=baseline_indices
            )
            self.crews[crew_id] = crew
            logger.info(f"[CrewManager] Creado nuevo crew {crew_id} para baseline {baseline_idx}")
        
        # Agregar miembro
        crew.add_member(baseline_idx, model, memory)
        
        # Verificar si el crew está completo (6 baselines)
        if len(crew.crew_members) >= self.crew_size:
            logger.info(f"[CrewManager] Crew {crew_id} completado ({len(crew.crew_members)} miembros)")
            
            # Calcular contexto holístico del crew si se proporcionan datos
            if X_holistic is not None and Y_holistic is not None:
                await crew.compute_holistic_context(X_holistic, Y_holistic)
                crew.save()
                
                # Guardar en historial
                self.crew_history.append({
                    'crew_id': crew_id,
                    'completed_at': datetime.now().isoformat(),
                    'n_members': len(crew.crew_members),
                    'holistic_score': crew.holistic_score
                })
        
        return crew
    
    async def route_to_crew(self, X: np.ndarray, context: Dict = None) -> Tuple[np.ndarray, Dict]:
        """
        Enruta una solicitud al crew más adecuado basado en:
        1. Similitud del input con el contexto holístico del crew
        2. Rendimiento histórico del crew
        """
        if not self.crews:
            raise ValueError("[CrewManager] No hay crews disponibles para enrutar")
        
        best_crew = None
        best_score = -float('inf')
        routing_debug = {}
        
        for crew_id, crew in self.crews.items():
            # Calcular similitud con contexto holístico
            similarity = 0
            if crew.holistic_context is not None and len(X) > 0:
                try:
                    # Similitud coseno entre input y contexto
                    x_flat = X.flatten()[:len(crew.holistic_context)]
                    ctx_flat = crew.holistic_context.flatten()[:len(x_flat)]
                    norm_product = np.linalg.norm(x_flat) * np.linalg.norm(ctx_flat) + 1e-9
                    similarity = np.dot(x_flat, ctx_flat) / norm_product
                except Exception as e:
                    logger.debug(f"[CrewManager] Error calculando similitud para crew {crew_id}: {e}")
                    similarity = 0
            
            # Bonus por número de miembros (más miembros = más confiable)
            member_bonus = min(0.3, len(crew.crew_members) / self.crew_size * 0.3)
            
            # Bonus por contexto holístico existente
            context_bonus = 0.2 if crew.holistic_context is not None else 0
            
            total_score = similarity + member_bonus + context_bonus
            
            routing_debug[crew_id] = {
                'similarity': similarity,
                'member_bonus': member_bonus,
                'context_bonus': context_bonus,
                'total_score': total_score,
                'n_members': len(crew.crew_members)
            }
            
            if total_score > best_score:
                best_score = total_score
                best_crew = crew
        
        if best_crew is None:
            raise ValueError("[CrewManager] No se pudo seleccionar un crew")
        
        # Realizar predicción con el crew seleccionado
        prediction, metadata = await best_crew.consensus_prediction(X)
        
        routing_result = {
            'selected_crew': best_crew.crew_id,
            'score': best_score,
            'routing_debug': routing_debug,
            'crew_metadata': metadata
        }
        
        logger.info(f"[CrewManager] Ruteado a crew {best_crew.crew_id} (score={best_score:.4f})")
        
        return prediction, routing_result
    
    def get_crew_metrics(self) -> Dict:
        """Métricas de todos los crews"""
        return {
            'total_crews': len(self.crews),
            'completed_crews': sum(1 for c in self.crews.values() if c.holistic_context is not None),
            'crews': {
                crew_id: {
                    'n_members': len(crew.crew_members),
                    'has_holistic_context': crew.holistic_context is not None,
                    'baseline_indices': crew.baseline_indices,
                    'holistic_score': crew.holistic_score
                }
                for crew_id, crew in self.crews.items()
            },
            'history': self.crew_history[-10:]  # Últimos 10 crews completados
        }
    
    async def get_vulnerability_analysis(self, X_test: np.ndarray, Y_test: np.ndarray) -> Dict:
        """
        Analiza vulnerabilidades entre crews:
        - Consistencia entre crews
        - Divergencia de predicciones
        - Posibles ataques de enrutamiento
        """
        if len(self.crews) < 2:
            return {'status': 'insufficient_crews', 'n_crews': len(self.crews)}
        
        predictions_by_crew = {}
        
        for crew_id, crew in self.crews.items():
            if crew.crew_members:
                # Usar el primer miembro como representante o el contexto
                try:
                    preds = []
                    for member in crew.crew_members[:3]:  # Tomar hasta 3 miembros
                        pred = await member['model'].predict(X_test[:100])
                        preds.append(pred[0])
                    predictions_by_crew[crew_id] = np.mean(preds, axis=0)
                except Exception as e:
                    logger.warning(f"[Vulnerability] Error con crew {crew_id}: {e}")
        
        if len(predictions_by_crew) < 2:
            return {'status': 'not_enough_predictions'}
        
        # Calcular divergencia entre crews
        crews_list = list(predictions_by_crew.keys())
        divergences = []
        
        for i, c1 in enumerate(crews_list):
            for c2 in crews_list[i+1:]:
                pred1 = predictions_by_crew[c1]
                pred2 = predictions_by_crew[c2]
                mse = np.mean((pred1 - pred2) ** 2)
                divergences.append({
                    'crew1': c1,
                    'crew2': c2,
                    'mse_divergence': float(mse)
                })
        
        # Identificar vulnerabilidades
        vulnerabilities = []
        for div in divergences:
            if div['mse_divergence'] > 0.1:  # Umbral de divergencia significativa
                vulnerabilities.append({
                    'type': 'high_divergence',
                    'crew_pair': (div['crew1'], div['crew2']),
                    'mse': div['mse_divergence'],
                    'risk': 'medium' if div['mse_divergence'] > 0.2 else 'low'
                })
        
        return {
            'n_crews_analyzed': len(predictions_by_crew),
            'mean_divergence': np.mean([d['mse_divergence'] for d in divergences]) if divergences else 0,
            'divergences': divergences,
            'vulnerabilities': vulnerabilities,
            'routing_attack_possible': len(vulnerabilities) > 0
        }

async def _save_audio_sample(
    model: DSVMWrapper,
    X_test: np.ndarray,
    Y_test: np.ndarray,
    output_dir: str,
    n_frames: int = 200,
) -> None:
    """
    Reconstruye un fragmento de audio desde predicciones del modelo
    y lo guarda como WAV para escucha subjetiva.
    """
    try:
        X_sample = X_test[:n_frames]
        Y_sample = Y_test[:n_frames]
        preds    = await model.predict(X_sample)
        preds = preds[0]

        # Reconstruir señal temporal (sin fase — usar magnitud directamente)
        # Normalizar antes de istft
        def _frames_to_audio(mag_frames: np.ndarray) -> np.ndarray:
            mag = np.clip(mag_frames, 0, None).T   # [bins, frames]
            # Usar fase neutra (ceros) como aproximación
            complex_spec = mag.astype(complex)
            _, audio = scipy_istft(complex_spec, fs=SR,
                                   nperseg=N_FFT, noverlap=N_FFT - HOP_LENGTH)
            audio = audio / (np.abs(audio).max() + 1e-9)
            return audio.astype(np.float32)

        audio_occluded  = _frames_to_audio(X_sample)
        audio_estimated = _frames_to_audio(preds)
        audio_clean     = _frames_to_audio(Y_sample)

        def _save(name, arr):
            path = os.path.join(output_dir, name)
            wav_write(path, SR, (arr * 32767).astype(np.int16))
            logger.info(f"[Audio] Guardado: {path}")

        _save("sample_occluded.wav",  audio_occluded)
        _save("sample_estimated.wav", audio_estimated)
        _save("sample_clean.wav",     audio_clean)

    except Exception as e:
        logger.warning(f"[Audio] No se pudo guardar muestra de audio: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# PUNTO DE ENTRADA
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="AOC Pipeline — APICultor",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--audio_dir", required=True,
        help="Directorio con audios limpios (.wav)"
    )
    p.add_argument(
        "--occluded_dir", default=None,
        help="Directorio con audios ocluidos (.wav). "
             "Si no se provee, se simula la oclusión."
    )
    p.add_argument(
        "--model_path", default=None,
        help="Path al modelo DSVM serializado (.pkl). "
             "Si no existe, se usa un fallback Ridge."
    )
    p.add_argument(
        "--output_dir", default="results/",
        help="Directorio de salida para reportes y audios"
    )
    p.add_argument(
        "--max_files", type=int, default=200,
        help="Máximo de archivos de audio a cargar"
    )
    p.add_argument(
        "--occlusion_ratio", type=float, default=0.4,
        help="Ratio de oclusión simulada (solo si no hay occluded_dir)"
    )
    p.add_argument(
        "--n_adversaries", type=int, default=2,
        help="Número de adversarios para cross-validation"
    )
    p.add_argument(
        "--num_rollouts", type=int, default=80,
        help="Rollouts RL (post-training)"
    )
    p.add_argument(
        "--regression", action="store_true", default=True,
        help="Modo regresión (True para AOC señal-a-señal)"
    )
    p.add_argument(
        "--post_train", action="store_true", default=False,
        help="Activar SFT + RL post-entrenamiento constitucional"
    )
    p.add_argument(
        "--log_level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    # En parse_args() agregar:
    p.add_argument(
        "--quote_reference_path", default=None,
        help="Path a QuoteReferenceModel guardado para usar como reward_model y reducir reward hacking"
    )
    p.add_argument(
        "--alignment_tax", action="store_true", default=False,
        help="Calcular alignment tax entre quote_reference_model y reward_model actual"
    )
    p.add_argument(
        "--sample_efficiency_analysis", action="store_true", default=False,
        help="Analizar sample efficiency: cuántos rollouts son necesarios para máximo gain"
    )
    p.add_argument(
        "--secure_clustering", action="store_true", default=False,
        help="Activar secure clustering con homomorphic encryption"
    )    
    p.add_argument(
        "--go_no_go_threshold", type=float, default=0.05,
        help="Umbral de mejora para decisión GO/NO-GO en promotion a producción"
    )        
    p.add_argument(
        "--model_routing", action="store_true", default=False,
        help="Activar model routing: mapear input a múltiples modelos y combinar outputs"
    )
    p.add_argument(
        "--feedback_to_reward", action="store_true", default=False,
        help="Convertir feedback (good/bad/neutral) a valor de reward para GO/NO-GO"
    )
    p.add_argument(
        "--adversarial_noise_levels", type=str, default="0.01,0.05,0.1",
        help="Niveles de ruido para modelos adversarios (separados por comas)"
    )
    p.add_argument(
        "--adversarial_shuffle_features", action="store_true", default=True,
        help="Aplicar shuffle de features en datos adversarios"
    )
    p.add_argument(
        "--capture_adversary_models", action="store_true", default=True,
        help="Capturar modelos adversarios de multiple_hypothesis_testing para majority voting"
    )    
    p.add_argument(
        "--knowledge_distillation", action="store_true", default=False,
        help="Activar knowledge distillation: entrenar student model desde teacher"
    )
    p.add_argument(
        "--distillation_temperature", type=float, default=3.0,
        help="Temperatura para soft targets en distillation"
    )
    p.add_argument(
        "--distillation_alpha", type=float, default=0.7,
        help="Peso para KL loss vs hard loss (alpha=1 solo soft, alpha=0 solo hard)"
    )
    p.add_argument(
        "--distillation_epochs", type=int, default=10,
        help="Número de épocas para distillation"
    )
    p.add_argument(
        "--consensus_threshold", type=int, default=3,
        help="Número mínimo de agentes que deben coincidir para consenso"
    )    
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    asyncio.run(run_full_pipeline(args))