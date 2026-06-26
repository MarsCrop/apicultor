# === auto_intervention_system.py ===
"""
Sistema de intervención automática que aplica correcciones en tiempo real
cuando detecta errores, ajustando hasta alcanzar umbrales óptimos.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass
from ..gradients.descent import *
from enum import Enum
import logging
from collections import deque
import random
import math

logger = logging.getLogger(__name__)

# Configuración RL
RL_DISCOUNT_FACTOR = 0.99
RL_LEARNING_RATE = 0.001
RL_BUFFER_SIZE = 10000
RL_BATCH_SIZE = 64
RL_TAU = 0.005  # Para soft updates
RL_EPSILON = 0.1  # Para epsilon-greedy
RL_EPSILON_DECAY = 0.995

# === UMBRALES DE INTERVENCIÓN ===

@dataclass
class InterventionThresholds:
    """Umbrales para activar intervenciones automáticas"""
    # Factuality/Hallucination
    factual_confidence_threshold: float = 0.85
    max_hallucination_rate: float = 0.1
    #ammount of fact backed information
    min_evidence_overlap: float = 0.7
    
    # Reasoning
    reasoning_accuracy_threshold: float = 0.9
    max_step_error: float = 0.05
    cot_consistency_threshold: float = 0.8
    
    # Schema/Format
    schema_compliance_threshold: float = 0.95
    max_format_errors: int = 0
    
    # Instruction Following
    instruction_compliance_threshold: float = 0.9
    constraint_violation_tolerance: float = 0.1
    
    # Uncertainty/Calibration
    calibration_error_threshold: float = 0.15
    max_overconfidence_rate: float = 0.1
    
    # Distribution Shift
    distribution_stability_threshold: float = 0.8
    max_performance_drop: float = 0.2

# === SISTEMA DE INTERVENCIÓN AUTOMÁTICA ===

class AutoInterventionSystem:
    """
    Sistema que detecta errores e interviene automáticamente aplicando
    correcciones hasta alcanzar umbrales óptimos.
    """
    
    def __init__(self, 
                 base_model: Callable,
                 thresholds: Optional[InterventionThresholds] = None):
        
        self.base_model = base_model
        self.thresholds = thresholds or InterventionThresholds()
        self.intervention_history = []
        self.error_statistics = {}
        self.use_rl = True
        self.epsilon = 1e-3
        
        # Contadores para monitoreo
        self.intervention_count = 0
        self.successful_interventions = 0
        self._init_rl_systems()

    def _init_rl_systems(self):
        """
        Inicializa sistemas de Reinforcement Learning usando solo numpy.
        """
        # Buffer de experiencia para RL
        self.replay_buffer = deque(maxlen=RL_BUFFER_SIZE)
        
        # Políticas RL por tipo de error (Q-tables o redes simples en numpy)
        self.rl_policies = {}
        
        # Dimensiones de estado y acción por tipo de error
        policy_configs = {
            'factuality': {'state_dim': 10, 'action_dim': 3, 'hidden_layers': [32, 16]},
            'reasoning': {'state_dim': 12, 'action_dim': 4, 'hidden_layers': [64, 32]},
            'schema': {'state_dim': 8, 'action_dim': 2, 'hidden_layers': [16, 8]},
            'calibration': {'state_dim': 10, 'action_dim': 3, 'hidden_layers': [32, 16]},
            'distribution_shift': {'state_dim': 8, 'action_dim': 2, 'hidden_layers': [16, 8]},
            'tool_use': {'state_dim': 15, 'action_dim': 5, 'hidden_layers': [64, 32, 16]},
            'refusal': {'state_dim': 10, 'action_dim': 3, 'hidden_layers': [32, 16]},
            'toxicity': {'state_dim': 12, 'action_dim': 3, 'hidden_layers': [32, 16]},
            'style': {'state_dim': 6, 'action_dim': 2, 'hidden_layers': [16, 8]},
            'retrieval': {'state_dim': 14, 'action_dim': 4, 'hidden_layers': [64, 32]},
            'long_context': {'state_dim': 16, 'action_dim': 3, 'hidden_layers': [64, 32]},
            'multi_turn': {'state_dim': 18, 'action_dim': 4, 'hidden_layers': [64, 32, 16]},
            'overconfidence': {'state_dim': 10, 'action_dim': 2, 'hidden_layers': [32, 16]}
        }
        
        for error_type, config in policy_configs.items():
            self.rl_policies[error_type] = self._create_numpy_policy(
                state_dim=config['state_dim'],
                action_dim=config['action_dim'],
                hidden_layers=config['hidden_layers']
            )
        
        # Políticas target para stable learning
        self.target_policies = {
            error_type: self._copy_policy(policy)
            for error_type, policy in self.rl_policies.items()
        }
        
        # Estadísticas RL
        self.rl_stats = {
            error_type: {
                'total_rewards': 0.0,
                'episodes': 0,
                'avg_reward': 0.0,
                'success_rate': 0.0,
                'exploration_rate': self.epsilon
            }
            for error_type in self.rl_policies.keys()
        }
    
        logger.info(f"Sistemas RL numpy inicializados para {len(self.rl_policies)} tipos de error")

    def _apply_iterative_refinement(self, 
                                   input_data: np.ndarray, 
                                   output: np.ndarray, 
                                   n_iterations: int = 3) -> np.ndarray:
        """Refinamiento iterativo completo. Useful to refine from prompt injections and jailbreaks"""
        refined = output.copy()
        
        for i in range(n_iterations):
            # Aplicar pequeñas correcciones basadas en autoconsistencia
            if len(refined.shape) == 1:
                # Para vectores: smoothing iterativo
                if len(refined) > 2:
                    refined = 0.5 * refined + 0.25 * np.roll(refined, 1) + 0.25 * np.roll(refined, -1)
                    # Corregir bordes
                    refined[0] = refined[1]
                    refined[-1] = refined[-2]
            elif len(refined.shape) == 2:
                # Para matrices: regularización
                row_means = np.mean(refined, axis=0, keepdims=True)
                col_means = np.mean(refined, axis=1, keepdims=True)
                refined = 0.6 * refined + 0.2 * row_means + 0.2 * col_means
            else:
                # Para tensores de mayor dimensión
                mean_val = np.mean(refined)
                refined = 0.8 * refined + 0.2 * mean_val
        
        return refined

    def _copy_policy(self, policy: Dict) -> Dict:
        """
        Crea una copia profunda de una política.
        """
        copied_policy = {
            'layers': [],
            'state_dim': policy['state_dim'],
            'action_dim': policy['action_dim']
        }
        
        for layer in policy['layers']:
            copied_layer = {
                'weights': layer['weights'].copy(),
                'biases': layer['biases'].copy(),
                'activation': layer['activation']
            }
            copied_policy['layers'].append(copied_layer)
        
        return copied_policy

    def _create_numpy_policy(self, state_dim: int, action_dim: int, hidden_layers: List[int]) -> Dict:
        """
        Crea una política RL usando solo numpy (red neuronal simple).
        """
        layers = []
        prev_dim = state_dim
        
        # Inicializar pesos y biases para cada capa
        for hidden_dim in hidden_layers:
            # Xavier/Glorot initialization
            scale = np.sqrt(2.0 / (prev_dim + hidden_dim))
            weights = np.random.randn(prev_dim, hidden_dim) * scale
            biases = np.zeros(hidden_dim)
            
            layers.append({
                'weights': weights,
                'biases': biases,
                'activation': 'relu'
            })
            prev_dim = hidden_dim
        
        # Capa de salida
        scale = np.sqrt(2.0 / (prev_dim + action_dim))
        output_weights = np.random.randn(prev_dim, action_dim) * scale
        output_biases = np.zeros(action_dim)
        
        layers.append({
            'weights': output_weights,
            'biases': output_biases,
            'activation': 'softmax'
        })
    
        return {
            'layers': layers,
            'state_dim': state_dim,
            'action_dim': action_dim
        }
        
    def process_with_intervention(self,
                                 input_data: np.ndarray,
                                 target: Optional[np.ndarray] = None,
                                 context: Optional[np.ndarray] = None,
                                 error_buckets: Optional[List[str]] = None) -> Dict:
        """
        Procesa entrada aplicando intervenciones automáticas con RL.
        """
        initial_output = self.base_model(input_data)
        diagnostics = self._diagnose_errors(input_data, initial_output, target, context)
        
        # Filtrar por buckets de error específicos si se proporcionan
        if error_buckets:
            diagnostics['errors'] = [
                error for error in diagnostics['errors'] 
                if error[0] in error_buckets
            ]
            diagnostics['requires_intervention'] = len(diagnostics['errors']) > 0
        
        final_output = initial_output
        interventions_applied = []
        
        # Aplicar intervenciones para cada error detectado
        for error_type, severity, metrics in diagnostics['errors']:
            if severity in ['medium', 'high', 'critical']:
                # Seleccionar tipo de intervención basado en RL
                if self.use_rl and error_type in self.rl_policies:
                    intervention_result = self._apply_rl_intervention(
                        error_type, input_data, final_output, target, context, metrics
                    )
                else:
                    intervention_result = self._apply_intervention(
                        error_type, input_data, final_output, target, context, metrics
                    )
                
                if intervention_result['success']:
                    final_output = intervention_result['corrected_output']
                    interventions_applied.append({
                        'error_type': error_type,
                        'severity': severity,
                        'intervention': intervention_result['intervention_type'],
                        'improvement': intervention_result['improvement'],
                        'rl_used': intervention_result.get('rl_used', False),
                        'rl_reward': intervention_result.get('rl_reward', 0.0),
                        'rl_action': intervention_result.get('rl_action', -1)
                    })
                    
                    # Actualizar política RL si se usó
                    if intervention_result.get('rl_used', False):
                        self._update_rl_policy(
                            error_type,
                            intervention_result['rl_state'],
                            intervention_result['rl_action'],
                            intervention_result['rl_reward'],
                            intervention_result['next_state'],
                            intervention_result.get('rl_action_probs', None)
                        )
        
        # Calcular métricas finales
        final_metrics = self._compute_metrics(input_data, final_output, target, context)
        
        result = {
            'initial_output': initial_output,
            'final_output': final_output,
            'diagnostics': diagnostics,
            'interventions_applied': interventions_applied,
            'metrics': final_metrics,
            'improved': len(interventions_applied) > 0
        }
        
        self.intervention_history.append(result)
        
        # Decay epsilon
        self.epsilon *= RL_EPSILON_DECAY
        
        return result

    def _compute_metrics(self, input_data: np.ndarray, output: np.ndarray, 
                        target: Optional[np.ndarray], context: Optional[np.ndarray]) -> Dict:
        """Calcula métricas completas para el output."""
        metrics = {}
        
        # Métricas de factualidad
        factual_metrics = self._check_factuality(input_data, output, context)
        metrics.update(factual_metrics)
        
        # Métricas de reasoning si hay target
        if target is not None:
            reasoning_metrics = self._check_reasoning(input_data, output, target)
            metrics.update(reasoning_metrics)
        
        # Métricas de schema
        schema_metrics = self._check_schema_compliance(output)
        metrics.update(schema_metrics)
        
        # Métricas de calibración
        if target is not None and len(output.shape) > 1:
            calibration_metrics = self._check_calibration(output, target)
            metrics.update(calibration_metrics)
        
        return metrics
    
    def _calculate_general_improvement(self, original_output: np.ndarray, 
                                     corrected_output: np.ndarray, metrics: Dict,
                                     target: Optional[np.ndarray], 
                                     context: Optional[np.ndarray]) -> float:
        """Calcula mejora general cuando no hay tipo específico."""
        if target is not None and original_output.shape == target.shape:
            # Usar error cuadrático medio normalizado
            mse_original = np.mean((original_output - target) ** 2)
            mse_corrected = np.mean((corrected_output - target) ** 2)
            
            if mse_original > 0:
                improvement = (mse_original - mse_corrected) / mse_original
                return max(0.0, min(1.0, improvement))
        
        # Si no hay target, usar correlación consigo mismo (consistencia)
        if original_output.shape == corrected_output.shape:
            correlation = np.corrcoef(original_output.flatten(), corrected_output.flatten())[0, 1]
            return max(0.0, correlation)
        
        return 0.0    

    def _calculate_emd_improvement(self, original_output: np.ndarray,
                                 corrected_output: np.ndarray,
                                 context: np.ndarray) -> float:
        """Calcula mejora en Earth Mover's Distance (simplificado)."""
        # Implementación simplificada de EMD
        if original_output.shape != corrected_output.shape:
            return 0.0
        
        # Calcular distancia entre distribuciones usando percentiles
        percentiles = np.linspace(0, 100, 20)
        orig_percentiles = np.percentile(original_output.flatten(), percentiles)
        corr_percentiles = np.percentile(corrected_output.flatten(), percentiles)
        ctx_percentiles = np.percentile(context.flatten(), percentiles)
        
        # Distancia original vs contexto
        orig_ctx_distance = np.mean(np.abs(orig_percentiles - ctx_percentiles))
        corr_ctx_distance = np.mean(np.abs(corr_percentiles - ctx_percentiles))
        
        if orig_ctx_distance > 0:
            improvement = (orig_ctx_distance - corr_ctx_distance) / orig_ctx_distance
            return max(0.0, improvement)
        
        return 0.0
    
    def _diagnose_errors(self,
                        input_data: np.ndarray,
                        output: np.ndarray,
                        target: Optional[np.ndarray] = None,
                        context: Optional[np.ndarray] = None) -> Dict:
        """
        Diagnóstica todos los tipos de errores posibles.
        """
        errors = []
        metrics = {}
        
        # 1. Factuality/Hallucination
        factual_metrics = self._check_factuality(input_data, output, context)
        metrics.update(factual_metrics)
        
        if factual_metrics['hallucination_risk'] > self.thresholds.max_hallucination_rate:
            severity = self._determine_severity(
                factual_metrics['hallucination_risk'], 
                self.thresholds.max_hallucination_rate
            )
            errors.append(('factuality', severity, factual_metrics))
        
        # 2. Reasoning
        if target is not None:
            reasoning_metrics = self._check_reasoning(input_data, output, target)
            metrics.update(reasoning_metrics)
            
            if reasoning_metrics['reasoning_accuracy'] < self.thresholds.reasoning_accuracy_threshold:
                severity = self._determine_severity(
                    reasoning_metrics['reasoning_accuracy'],
                    self.thresholds.reasoning_accuracy_threshold,
                    inverse=True
                )
                errors.append(('reasoning', severity, reasoning_metrics))
        
        # 3. Schema/Format
        schema_metrics = self._check_schema_compliance(output)
        metrics.update(schema_metrics)
        
        if schema_metrics['schema_violations'] > self.thresholds.max_format_errors:
            errors.append(('schema', 'high', schema_metrics))
        
        # 4. Uncertainty/Calibration
        if len(output.shape) > 1:  # Clasificación
            calibration_metrics = self._check_calibration(output, target)
            metrics.update(calibration_metrics)
            
            if calibration_metrics['calibration_error'] > self.thresholds.calibration_error_threshold:
                severity = self._determine_severity(
                    calibration_metrics['calibration_error'],
                    self.thresholds.calibration_error_threshold
                )
                errors.append(('calibration', severity, calibration_metrics))
        
        # 5. Distribution Shift (si hay contexto histórico)
        if context is not None and len(self.intervention_history) > 10:
            shift_metrics = self._check_distribution_shift(output, context)
            metrics.update(shift_metrics)
            
            if shift_metrics['distribution_shift'] > (1 - self.thresholds.distribution_stability_threshold):
                errors.append(('distribution_shift', 'medium', shift_metrics))
        
        return {
            'errors': errors,
            'metrics': metrics,
            'requires_intervention': len(errors) > 0
        }
    
    def _apply_intervention(self,
                          error_type: str,
                          input_data: np.ndarray,
                          output: np.ndarray,
                          target: Optional[np.ndarray],
                          context: Optional[np.ndarray],
                          error_metrics: Dict) -> Dict:
        """
        Aplica intervención específica para tipo de error.
        """
        self.intervention_count += 1
        
        if error_type == 'factuality':
            return self._intervene_factuality(input_data, output, context, error_metrics)
        
        elif error_type == 'reasoning':
            return self._intervene_reasoning(input_data, output, target, error_metrics)
        
        elif error_type == 'schema':
            return self._intervene_schema(output, error_metrics)
        
        elif error_type == 'calibration':
            return self._intervene_calibration(output, target, error_metrics)
        
        elif error_type == 'distribution_shift':
            return self._intervene_distribution_shift(output, context, error_metrics)
        
        else:
            return {
                'success': False,
                'intervention_type': 'unknown',
                'improvement': 0.0,
                'corrected_output': output
            }

    def _calculate_improvement(self,
                        original_output: np.ndarray,
                        corrected_output: np.ndarray,
                        metrics: Dict,
                        error_type: Optional[str] = None,
                        target: Optional[np.ndarray] = None,
                        context: Optional[np.ndarray] = None) -> float:
        """
        Calcula cuánto mejora el output corregido respecto al original.
        
        Args:
            original_output: Output original del modelo (antes de intervención)
            corrected_output: Output después de aplicar intervención
            metrics: Métricas de error del output original
            error_type: Tipo de error que se está corrigiendo
            target: Target/valores esperados (si están disponibles)
            context: Contexto/evidencia disponible
        
        Returns:
            improvement_score: Puntuación de mejora entre 0.0 (sin mejora) y 1.0 (mejora perfecta)
        """
        
        if original_output is None or corrected_output is None:
            logger.warning("Output original o corregido es None")
            return 0.0
        
        if original_output.shape != corrected_output.shape:
            logger.warning(f"Shapes no coinciden: {original_output.shape} vs {corrected_output.shape}")
            # Intentar reshape si es posible
            try:
                if original_output.size == corrected_output.size:
                    corrected_output = corrected_output.reshape(original_output.shape)
                else:
                    return 0.0
            except:
                return 0.0
        
        improvement_scores = []
        weights = []
        
        # 1. MEJORA EN ERROR DE FACTUALIDAD/HALLUCINACIÓN
        if error_type in ['factuality', None] and 'hallucination_risk' in metrics:
                factual_improvement = self._calculate_factual_improvement(
                    original_output, corrected_output, metrics, context
                )
                improvement_scores.append(factual_improvement)
                weights.append(0.35)  # Peso alto para factualidad
    
        # 2. MEJORA EN ERROR DE REASONING
        if error_type in ['reasoning', None] and target is not None:
            reasoning_improvement = self._calculate_reasoning_improvement(
                original_output, corrected_output, target, metrics
            )
            improvement_scores.append(reasoning_improvement)
            weights.append(0.30)  # Peso alto para reasoning
    
        # 3. MEJORA EN ERROR DE SCHEMA/FORMATO
        if error_type in ['schema', None]:
            schema_improvement = self._calculate_schema_improvement(
                original_output, corrected_output, metrics
            )
            improvement_scores.append(schema_improvement)
            weights.append(0.20)  # Peso medio para schema
    
        # 4. MEJORA EN ERROR DE CALIBRACIÓN
        if error_type in ['calibration', None] and target is not None:
            calibration_improvement = self._calculate_calibration_improvement(
                original_output, corrected_output, target, metrics
            )
            improvement_scores.append(calibration_improvement)
            weights.append(0.15)  # Peso medio para calibración
    
        # 5. MEJORA EN ERROR DE DISTRIBUCIÓN
        if error_type in ['distribution_shift', None] and context is not None:
            distribution_improvement = self._calculate_distribution_improvement(
                original_output, corrected_output, context, metrics
            )
            improvement_scores.append(distribution_improvement)
            weights.append(0.10)  # Peso bajo para distribución
    
        # Si no se pudo calcular mejora específica, usar métrica general
        if not improvement_scores:
            general_improvement = self._calculate_general_improvement(
                original_output, corrected_output, metrics, target, context
            )
            improvement_scores.append(general_improvement)
            weights.append(1.0)
    
        # Normalizar pesos
        if weights:
            weights_sum = sum(weights)
            if weights_sum > 0:
                weights = [w / weights_sum for w in weights]
        
        # Calcular puntuación final ponderada
        final_improvement = 0.0
        for score, weight in zip(improvement_scores, weights):
            final_improvement += score * weight
    
        # Asegurar que esté en rango [0, 1]
        final_improvement = max(0.0, min(1.0, final_improvement))
        
        # Log para debugging
        logger.debug(f"Improvement calculado: {final_improvement:.4f} "
                 f"(scores: {improvement_scores}, weights: {weights})")
        
        return float(final_improvement)
    
    def _calculate_factual_improvement(self,
                               original_output: np.ndarray,
                               corrected_output: np.ndarray,
                               metrics: Dict,
                               context: Optional[np.ndarray] = None) -> float:
        """
        Calcula mejora específica para errores de factualidad/alucinación.
        """
        original_risk = metrics.get('hallucination_risk', 0.0)
        
        # Calcular riesgo de alucinación para output corregido
        corrected_metrics = self._check_factuality(None, corrected_output, context)
        corrected_risk = corrected_metrics.get('hallucination_risk', 0.0)
        
        # Reducción en riesgo de alucinación
        risk_reduction = original_risk - corrected_risk
        
        # Mejora en autocorrelación (coherencia interna)
        original_autocorr = metrics.get('autocorrelation', 0.0)
        corrected_autocorr = corrected_metrics.get('autocorrelation', 0.0)
        autocorr_improvement = abs(corrected_autocorr - original_autocorr)
        
        # Mejora en correlación con contexto
        if context is not None:
            original_context_corr = metrics.get('context_correlation', 0.0)
            corrected_context_corr = corrected_metrics.get('context_correlation', 0.0)
            context_improvement = max(0, corrected_context_corr - original_context_corr)
        else:
            context_improvement = 0.0
        
        # Reducción de valores extremos
        original_extremes = metrics.get('extreme_values', 0.0)
        corrected_extremes = corrected_metrics.get('extreme_values', 0.0)
        extremes_reduction = original_extremes - corrected_extremes
    
        # Combinar métricas con pesos
        improvement_components = [
            (risk_reduction, 0.40),           # Peso más alto para reducción de riesgo
            (autocorr_improvement, 0.20),     # Coherencia interna
            (context_improvement, 0.25),      # Alineación con contexto
            (extremes_reduction, 0.15)        # Valores extremos
        ]
        
        factual_improvement = 0.0
        for value, weight in improvement_components:
            factual_improvement += max(0, value) * weight
        
        # Normalizar a [0, 1]
        factual_improvement = min(1.0, factual_improvement)
        
        return factual_improvement * 100   

    def _calculate_distribution_improvement(self,
                                     original_output: np.ndarray,
                                     corrected_output: np.ndarray,
                                     context: np.ndarray,
                                     metrics: Dict) -> float:
        """
        Calcula mejora específica para errores de cambio de distribución.
        """
        if context is None:
            return 0.0
        
        # Métricas de cambio de distribución para output original
        original_shift = metrics.get('distribution_shift', 0.0)
        original_mean_shift = metrics.get('mean_shift', 0.0)
        original_std_ratio = metrics.get('std_ratio', 0.0)
        
        # Calcular métricas para output corregido
        corrected_metrics = self._check_distribution_shift(corrected_output, context)
        corrected_shift = corrected_metrics.get('distribution_shift', 0.0)
        corrected_mean_shift = corrected_metrics.get('mean_shift', 0.0)
        corrected_std_ratio = corrected_metrics.get('std_ratio', 0.0)
        
        # Reducción en shift total
        shift_improvement = 0.0
        if original_shift > 0:
            shift_reduction = (original_shift - corrected_shift) / original_shift
            shift_improvement = max(0, shift_reduction)
        else:
            shift_improvement = 1.0 if corrected_shift == 0 else 0.0
        
        # Reducción en mean shift
        mean_shift_improvement = 0.0
        if original_mean_shift > 0:
            mean_reduction = (original_mean_shift - corrected_mean_shift) / original_mean_shift
            mean_shift_improvement = max(0, mean_reduction)
        else:
            mean_shift_improvement = 1.0 if corrected_mean_shift == 0 else 0.0
        
        # Mejora en std ratio (debería acercarse a 1)
        std_ratio_improvement = 0.0
        original_std_distance = abs(original_std_ratio - 1.0)
        corrected_std_distance = abs(corrected_std_ratio - 1.0)
        
        if original_std_distance > 0:
            std_improvement = (original_std_distance - corrected_std_distance) / original_std_distance
            std_ratio_improvement = max(0, std_improvement)
        else:
            std_ratio_improvement = 1.0 if corrected_std_distance == 0 else 0.0
        
        # Calcular KL divergence entre distribuciones
        try:
            from scipy import stats
            
            # Aplanar arrays para cálculo de distribución
            original_flat = original_output.flatten()
            corrected_flat = corrected_output.flatten()
            context_flat = context.flatten()
            
            # Histogramas para calcular KL
            hist_original, _ = np.histogram(original_flat, bins=50, density=True)
            hist_corrected, _ = np.histogram(corrected_flat, bins=50, density=True)
            hist_context, _ = np.histogram(context_flat, bins=50, density=True)
            
            # Suavizar para evitar ceros
            hist_original += 1e-10
            hist_corrected += 1e-10
            hist_context += 1e-10
            
            # Normalizar
            hist_original /= np.sum(hist_original)
            hist_corrected /= np.sum(hist_corrected)
            hist_context /= np.sum(hist_context)
            
            # KL divergence
            kl_original = stats.entropy(hist_original, hist_context)
            kl_corrected = stats.entropy(hist_corrected, hist_context)
            
            # Reducción en KL divergence
            if kl_original > 0:
                kl_improvement = (kl_original - kl_corrected) / kl_original
                kl_improvement = max(0, kl_improvement)
            else:
                kl_improvement = 1.0 if kl_corrected == 0 else 0.0
        except:
            kl_improvement = 0.0
        
        # Calcular Earth Mover's Distance (simplificado)
        emd_improvement = self._calculate_emd_improvement(
            original_output, corrected_output, context
        )
        
        # Combinar métricas con pesos
        improvement_components = [
            (shift_improvement, 0.30),
            (mean_shift_improvement, 0.20),
            (std_ratio_improvement, 0.15),
            (kl_improvement, 0.20),
            (emd_improvement, 0.15)
        ]
    
        total_improvement = 0.0
        for value, weight in improvement_components:
            total_improvement += value * weight
        
        # Normalizar a [0, 1]
        total_improvement = min(1.0, total_improvement)
        
        return total_improvement
    
    # === INTERVENCIONES ESPECÍFICAS ===    
    def _intervene_factuality(self,
                         input_data: np.ndarray,
                         output: np.ndarray,
                         context: Optional[np.ndarray],
                         metrics: Dict) -> Dict:
        """
        Intervención para errores de factualidad/alucinación.
        """
        hallucination_risk = metrics.get('hallucination_risk', 0.0)
        
        # Estrategia 1: Regularización por contexto
        if context is not None and hallucination_risk > 0.3:
            # Ajustar output hacia el contexto hasta alcanzar mejora aceptable
            max_iterations = 10
            iteration = 0
            improvement = 0
            
            while iteration < max_iterations and (improvement < 0.35 or improvement < 0.5):
                context_aligned = self._align_with_context(output, context)
                improvement = self._calculate_improvement(output, context_aligned, metrics)
                iteration += 1
                
                # Actualizar output para próxima iteración
                if iteration < max_iterations and (improvement < 0.35 or improvement < 0.5):
                    output = context_aligned
            
            if improvement >= 0.35:
                self.successful_interventions += 1
                return {
                    'success': True,
                    'intervention_type': 'context_alignment',
                    'improvement': improvement,
                    'corrected_output': context_aligned
                }
        
        # Estrategia 2: Suavizado probabilístico
        if hallucination_risk > 0.2:
            max_iterations = 8
            iteration = 0
            improvement = 0
            
            while iteration < max_iterations and (improvement < 0.35 or improvement < 0.5):
                smoothed = self._apply_probabilistic_smoothing(output, alpha=0.3)
                improvement = self._calculate_improvement(output, smoothed, metrics)
                iteration += 1
                
                if iteration < max_iterations and (improvement < 0.35 or improvement < 0.5):
                    output = smoothed
            
            if improvement >= 0.35:
                self.successful_interventions += 1
                return {
                    'success': True,
                    'intervention_type': 'probabilistic_smoothing',
                    'improvement': improvement,
                    'corrected_output': smoothed
                }
        
        # Estrategia 3: Limitación de extremos
        if metrics.get('extreme_values', 0) > 0:
            max_iterations = 6
            iteration = 0
            improvement = 0
            
            while iteration < max_iterations and (improvement < 0.35 or improvement < 0.5):
                clipped = self._clip_extreme_values(output, std_multiplier=2.0)
                improvement = self._calculate_improvement(output, clipped, metrics)
                iteration += 1
                
                if iteration < max_iterations and (improvement < 0.35 or improvement < 0.5):
                    output = clipped
            
            if improvement >= 0.35:
                self.successful_interventions += 1
                return {
                    'success': True,
                    'intervention_type': 'value_clipping',
                    'improvement': improvement,
                    'corrected_output': clipped
                }
        
        return {
            'success': False,
            'intervention_type': 'factuality_intervention',
            'improvement': 0.0,
            'corrected_output': output
        }
    
    def _intervene_reasoning(self,
                            input_data: np.ndarray,
                            output: np.ndarray,
                            target: Optional[np.ndarray],
                            metrics: Dict) -> Dict:
        """
        Intervención para errores de reasoning.
        """
        reasoning_accuracy = metrics.get('reasoning_accuracy', 0.0)
        
        # Estrategia 1: Corrección paso a paso
        if reasoning_accuracy < 0.8 and target is not None:
            while improvement < .4 or improvement < .6:
                step_corrected = parallel_attention_sgd(10, [input_data, output, target], 10, batch_size = 2000)
                improvement = self._calculate_improvement(output, step_corrected, metrics)
            
            self.successful_interventions += 1
            return {
                'success': True,
                'intervention_type': 'stepwise_correction',
                'improvement': improvement,
                'corrected_output': step_corrected
            }
        
        # Estrategia 2: Ensembling con modelos simples
        if reasoning_accuracy < 0.8:
            while improvement < .4 or improvement < .6:
                ensembled = self._apply_ensembling_correction(input_data, output)
                improvement = self._calculate_improvement(output, ensembled, metrics)
            
            self.successful_interventions += 1
            return {
                    'success': True,
                    'intervention_type': 'ensembling_correction',
                    'improvement': improvement,
                    'corrected_output': ensembled
            }
        
        # Estrategia 3: Refinamiento iterativo
        refined = self._apply_iterative_refinement(input_data, output, n_iterations=3)
        improvement = self._calculate_improvement(output, refined, metrics)
        while improvement < .4 or improvement < .6:
            iter_refined = self._apply_iterative_refinement(input_data, output, n_iterations=3)
            improvement = self._calculate_improvement(output, iter_refined, metrics)
            if improvement > 0.05:
                self.successful_interventions += 1
                return {
                    'success': True,
                    'intervention_type': 'iterative_refinement',
                    'improvement': improvement,
                    'corrected_output': iter_refined
                }
        
        return {
            'success': False,
            'intervention_type': 'reasoning_intervention',
            'improvement': 0.0,
            'corrected_output': output
        }
    
    def _intervene_schema(self,
                         output: np.ndarray,
                         metrics: Dict) -> Dict:
        """
        Intervención para violaciones de esquema/formato.
        """
        violations = metrics.get('schema_violations', 0)
        
        # Estrategia 1: Normalización dimensional
        if violations > 0 and len(output.shape) > 1:
            normalized = self._normalize_to_schema(output)
            improvement = self._calculate_schema_improvement(output, normalized)
            
            if improvement > 0:
                self.successful_interventions += 1
                return {
                    'success': True,
                    'intervention_type': 'schema_normalization',
                    'improvement': improvement,
                    'corrected_output': normalized
                }
        
        # Estrategia 2: Proyección a espacio válido
        projected = self._project_to_valid_space(output)
        improvement = self._calculate_schema_improvement(output, projected)
        
        if improvement > 0:
            self.successful_interventions += 1
            return {
                'success': True,
                'intervention_type': 'space_projection',
                'improvement': improvement,
                'corrected_output': projected
            }
        
        return {
            'success': False,
            'intervention_type': 'schema_intervention',
            'improvement': 0.0,
            'corrected_output': output
        }


    def _calculate_schema_improvement(self,
                                original_output: np.ndarray,
                                corrected_output: np.ndarray,
                                metrics: Dict) -> float:
        """
        Calcula mejora específica para errores de schema/formato.
        """
        # Violaciones originales
        original_violations = metrics.get('schema_violations', 0)
        
        # Calcular violaciones corregidas
        corrected_metrics = self._check_schema_compliance(corrected_output)
        corrected_violations = corrected_metrics.get('schema_violations', 0)
        
        # Reducción en violaciones
        if original_violations > 0:
            violation_improvement = (original_violations - corrected_violations) / original_violations
        else:
            violation_improvement = 1.0 if corrected_violations == 0 else 0.0
        
        # Mejora en normalización
        if len(original_output.shape) == 2:
            # Para matrices de probabilidad: mejora en normalización por fila
            original_row_sums = np.sum(original_output, axis=1)
            corrected_row_sums = np.sum(corrected_output, axis=1)
            
            original_row_error = np.mean(np.abs(original_row_sums - 1.0))
            corrected_row_error = np.mean(np.abs(corrected_row_sums - 1.0))
            
            if original_row_error > 0:
                normalization_improvement = (original_row_error - corrected_row_error) / original_row_error
            else:
                normalization_improvement = 1.0 if corrected_row_error == 0 else 0.0
        else:
            normalization_improvement = 0.0
        
        # Mejora en valores válidos (no NaN/Inf)
        original_valid = np.sum(np.isfinite(original_output)) / original_output.size
        corrected_valid = np.sum(np.isfinite(corrected_output)) / corrected_output.size
        validity_improvement = max(0, corrected_valid - original_valid)
        
        # Combinar métricas
        improvement_components = [
            (violation_improvement, 0.4),
            (normalization_improvement, 0.3),
            (validity_improvement, 0.3)
        ]
        
        schema_improvement = 0.0
        for value, weight in improvement_components:
            schema_improvement += max(0, value) * weight
        
        return min(1.0, schema_improvement)
    
    def _intervene_calibration(self,
                              output: np.ndarray,
                              target: Optional[np.ndarray],
                              metrics: Dict) -> Dict:
        """
        Intervención para problemas de calibración/sobreconfianza.
        """
        calibration_error = metrics.get('calibration_error', 0.0)
        
        # Estrategia 1: Temperature scaling adaptativo
        if calibration_error > 0.1:
            temperature = self._calculate_optimal_temperature(output, target)
            calibrated = self._apply_temperature_scaling(output, temperature)
            improvement = self._calculate_calibration_improvement(output, calibrated, target)
            
            if improvement > 0.1:
                self.successful_interventions += 1
                return {
                    'success': True,
                    'intervention_type': 'temperature_scaling',
                    'improvement': improvement,
                    'corrected_output': calibrated
                }
        
        # Estrategia 3: Label smoothing
        smoothed = self._apply_label_smoothing(output, alpha=0.1)
        improvement = self._calculate_calibration_improvement(output, smoothed, target)
        
        if improvement > 0:
            self.successful_interventions += 1
            return {
                'success': True,
                'intervention_type': 'label_smoothing',
                'improvement': improvement,
                'corrected_output': smoothed
            }
        
        return {
            'success': False,
            'intervention_type': 'calibration_intervention',
            'improvement': 0.0,
            'corrected_output': output
        }
    
    def _intervene_distribution_shift(self,
                                     output: np.ndarray,
                                     context: np.ndarray,
                                     metrics: Dict) -> Dict:
        """
        Intervención para cambios de distribución.
        """
        shift_magnitude = metrics.get('distribution_shift', 0.0)
        
        # Estrategia 1: Domain adaptation
        if shift_magnitude > 0.3:
            adapted = self._apply_domain_adaptation(output, context)
            improvement = self._calculate_shift_improvement(output, adapted, context)
            while improvement > 0.25 or improvement < .4:
                adapted = self._apply_domain_adaptation(output, context)
                improvement = self._calculate_shift_improvement(output, adapted, context)            
            
            self.successful_interventions += 1
            return {
                    'success': True,
                    'intervention_type': 'domain_adaptation',
                    'improvement': improvement,
                    'corrected_output': adapted
            }
        
        # Estrategia 2: Robust normalization
        robust_normalized = self._apply_robust_normalization(output, context)
        improvement = self._calculate_shift_improvement(output, robust_normalized, context)
        while improvement > 0.25 or improvement < .4:
            robust_normalized = self._apply_robust_normalization(output, context)
            improvement = self._calculate_shift_improvement(output, robust_normalized, context)    

        self.successful_interventions += 1
        return {
            'success': True,
            'intervention_type': 'robust_normalization',
            'improvement': improvement,
            'corrected_output': robust_normalized
        }
    
    # === MÉTODOS DE CORRECCIÓN CONCRETOS ===
    
    def _align_with_context(self, output: np.ndarray, context: np.ndarray) -> np.ndarray:
        """Alinea output con contexto usando weighted average."""
        if output.shape == context.shape:
            # Interpolación entre output original y contexto
            alpha = 0.7  # Peso del contexto
            aligned = alpha * context + (1 - alpha) * output
            return aligned
        return output
    
    def _apply_probabilistic_smoothing(self, output: np.ndarray, alpha: float = 0.3) -> np.ndarray:
        """Aplica suavizado probabilístico."""
        if len(output.shape) > 1:
            # Para clasificación: mixing con distribución uniforme
            n_classes = output.shape[1]
            uniform = np.ones_like(output) / n_classes
            smoothed = (1 - alpha) * output + alpha * uniform
            return smoothed
        else:
            # Para regresión: smoothing con vecinos
            if len(output) > 1:
                smoothed = np.convolve(output, [alpha, 1-2*alpha, alpha], mode='same')
                return smoothed
        return output
    
    def _clip_extreme_values(self, output: np.ndarray, std_multiplier: float = 2.0) -> np.ndarray:
        """Recorta valores extremos."""
        mean_val = np.mean(output)
        std_val = np.std(output)
        
        lower_bound = mean_val - std_multiplier * std_val
        upper_bound = mean_val + std_multiplier * std_val
        
        clipped = np.clip(output, lower_bound, upper_bound)
        return clipped
    
    def _apply_ensembling_correction(self, input_data: np.ndarray, output: np.ndarray) -> np.ndarray:
        """Ensambla con modelos simples."""
        # Modelo 1: Promedio móvil (para series temporales)
        if len(output.shape) == 1 and len(output) > 3:
            ma_correction = np.convolve(output, np.ones(3)/3, mode='same')
        
        # Modelo 2: Baseline simple (media)
        baseline = np.mean(output) * np.ones_like(output)
        
        # Ensamblar ponderadamente
        weights = [0.6, 0.4]  # Más peso al original
        ensembled = weights[0] * output + weights[1] * baseline
        
        return ensembled
    
    def _apply_iterative_refinement(self, 
                                   input_data: np.ndarray, 
                                   output: np.ndarray, 
                                   n_iterations: int = 3) -> np.ndarray:
        """Refinamiento iterativo."""
        refined = output.copy()
        
        for i in range(n_iterations):
            # Aplicar pequeñas correcciones basadas en autoconsistencia
            if len(refined.shape) == 1:
                # Para vectores: smoothing iterativo
                refined = 0.5 * refined + 0.25 * np.roll(refined, 1) + 0.25 * np.roll(refined, -1)
            elif len(refined.shape) == 2:
                # Para matrices: regularización
                refined = 0.8 * refined + 0.2 * np.mean(refined, axis=0, keepdims=True)
        
        return refined
    
    def _normalize_to_schema(self, output: np.ndarray) -> np.ndarray:
        """Normaliza output a esquema esperado."""
        # Normalizar dimensiones
        if len(output.shape) == 2:
            # Asegurar que sea matriz de probabilidades (suma a 1 por fila)
            row_sums = np.sum(output, axis=1, keepdims=True)
            normalized = output / (row_sums + 1e-10)
            return normalized
        
        # Asegurar valores en rango [0, 1] si es apropiado
        if np.min(output) < 0 or np.max(output) > 1:
            normalized = (output - np.min(output)) / (np.max(output) - np.min(output) + 1e-10)
            return normalized
        
        return output
    
    def _project_to_valid_space(self, output: np.ndarray) -> np.ndarray:
        """Proyecta a espacio de soluciones válidas."""
        # Para vectores de probabilidad: proyecto a simplex
        if len(output.shape) == 2:
            from scipy import optimize
            # Proyección al simplex (suma a 1, no negativa)
            projected = output.copy()
            for i in range(projected.shape[0]):
                projected[i] = self._project_to_simplex(projected[i])
            return projected
        
        # Para valores continuos: clip a rango razonable
        projected = np.clip(output, -10, 10)
        return projected
    
    def _project_to_simplex(self, v: np.ndarray) -> np.ndarray:
        """Proyecta vector al simplex (suma a 1, no negativo)."""
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u) - 1
        ind = np.arange(1, len(v) + 1)
        cond = u - cssv / ind > 0
        rho = ind[cond][-1]
        theta = cssv[cond][-1] / rho
        w = np.maximum(v - theta, 0)
        return w
    
    def _calculate_optimal_temperature(self, output: np.ndarray, target: Optional[np.ndarray]) -> float:
        """Calcula temperatura óptima para scaling."""
        if target is None or len(output.shape) <= 1:
            return 1.0
        
        # Método simple: temperatura que maximiza likelihood
        n_samples, n_classes = output.shape
        
        if len(target.shape) == 1:
            # Target como índices
            target_one_hot = np.eye(n_classes)[target]
        else:
            target_one_hot = target
        
        # Búsqueda simple de temperatura
        temperatures = np.linspace(0.1, 10, 50)
        best_temp = 1.0
        best_nll = float('inf')
        
        for temp in temperatures:
            scaled = output / temp
            probs = self._softmax(scaled)
            
            # Negative log likelihood
            nll = -np.sum(target_one_hot * np.log(probs + 1e-10)) / n_samples
            
            if nll < best_nll:
                best_nll = nll
                best_temp = temp
        
        return float(best_temp)
    
    def _apply_temperature_scaling(self, output: np.ndarray, temperature: float) -> np.ndarray:
        """Aplica temperature scaling."""
        if len(output.shape) <= 1:
            return output
        
        scaled = output / temperature
        return self._softmax(scaled)
    
    def _apply_platt_scaling(self, output: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Aplica Platt scaling para calibración."""
        if len(output.shape) <= 1:
            return output
        
        n_samples, n_classes = output.shape
        
        # Platt scaling por clase (simplificado)
        calibrated = output.copy()
        
        for c in range(n_classes):
            if len(target.shape) == 1:
                target_binary = (target == c).astype(float)
            else:
                target_binary = target[:, c]
            
            # Ajustar logistic regression (simplificado)
            logits = output[:, c]
            # Parámetros A, B de Platt
            A, B = self._fit_platt_parameters(logits, target_binary)
            
            # Aplicar transformación
            calibrated[:, c] = 1.0 / (1.0 + np.exp(A * logits + B))
        
        # Renormalizar
        row_sums = np.sum(calibrated, axis=1, keepdims=True)
        calibrated = calibrated / (row_sums + 1e-10)
        
        return calibrated
    
    def _fit_platt_parameters(self, logits: np.ndarray, targets: np.ndarray) -> Tuple[float, float]:
        """Ajusta parámetros A, B para Platt scaling."""
        # Método simplificado
        mean_logit = np.mean(logits)
        std_logit = np.std(logits) + 1e-10
        
        # Aproximación inicial
        A = -1.0 / std_logit
        B = mean_logit / std_logit
        
        return A, B
    
    def _apply_label_smoothing(self, output: np.ndarray, alpha: float = 0.1) -> np.ndarray:
        """Aplica label smoothing."""
        if len(output.shape) <= 1:
            return output
        
        n_classes = output.shape[1]
        uniform = np.ones_like(output) / n_classes
        smoothed = (1 - alpha) * output + alpha * uniform
        
        return smoothed
    
    def _apply_domain_adaptation(self, output: np.ndarray, context: np.ndarray) -> np.ndarray:
        """Adapta output al dominio del contexto."""
        if output.shape == context.shape:
            # Alinear estadísticas
            output_mean = np.mean(output)
            output_std = np.std(output) + 1e-10
            context_mean = np.mean(context)
            context_std = np.std(context) + 1e-10
            
            # Whitening y re-coloring
            whitened = (output - output_mean) / output_std
            adapted = whitened * context_std + context_mean
            
            return adapted
        
        return output
    
    def _apply_robust_normalization(self, output: np.ndarray, context: np.ndarray) -> np.ndarray:
        """Aplica normalización robusta usando contexto."""
        # Usar estadísticas robustas (median, MAD)
        output_median = np.median(output)
        output_mad = np.median(np.abs(output - output_median)) + 1e-10
        
        context_median = np.median(context)
        context_mad = np.median(np.abs(context - context_median)) + 1e-10
        
        # Normalizar robustamente
        normalized = (output - output_median) / output_mad
        robust_normalized = normalized * context_mad + context_median
        
        return robust_normalized
    
    # === MÉTODOS DE EVALUACIÓN ===
    
    def _check_factuality(self, 
                         input_data: np.ndarray, 
                         output: np.ndarray, 
                         context: Optional[np.ndarray]) -> Dict:
        """Evalúa factualidad/alucinación."""
        metrics = {}
        
        # 1. Coherencia interna
        if len(output) > 1:
            autocorr = np.corrcoef(output[:-1], output[1:])[0, 1]
            metrics['autocorrelation'] = float(autocorr)
        
        # 2. Valores extremos
        z_scores = np.abs((output - np.mean(output)) / (np.std(output) + 1e-10))
        extreme_count = np.sum(z_scores > 3)
        metrics['extreme_values'] = float(extreme_count / len(output))
        
        # 3. Consistencia con contexto
        if context is not None and output.shape == context.shape:
            context_correlation = np.corrcoef(output.flatten(), context.flatten())[0, 1]
            metrics['context_correlation'] = float(context_correlation)
            metrics['hallucination_risk'] = float(max(0, 1 - abs(context_correlation)))
        else:
            metrics['hallucination_risk'] = float(metrics.get('extreme_values', 0))
        
        return metrics
    
    def _check_reasoning(self, 
                        input_data: np.ndarray, 
                        output: np.ndarray, 
                        target: np.ndarray) -> Dict:
        """Evalúa calidad de reasoning."""
        metrics = {}
        
        if output.shape == target.shape:
            # Error absoluto
            abs_error = np.mean(np.abs(output - target))
            metrics['absolute_error'] = float(abs_error)
            
            # Error relativo
            rel_error = np.mean(np.abs((output - target) / (np.abs(target) + 1e-10)))
            metrics['relative_error'] = float(rel_error)
            
            # Accuracy para reasoning
            if len(output.shape) > 1:
                # Clasificación
                pred_classes = np.argmax(output, axis=1)
                true_classes = np.argmax(target, axis=1) if len(target.shape) > 1 else target
                accuracy = np.mean(pred_classes == true_classes)
                metrics['reasoning_accuracy'] = float(accuracy)
            else:
                # Regresión: R² score
                ss_res = np.sum((output - target) ** 2)
                ss_tot = np.sum((target - np.mean(target)) ** 2)
                r2 = 1 - (ss_res / (ss_tot + 1e-10))
                metrics['reasoning_accuracy'] = float(max(0, r2))
        
        return metrics
    
    def _check_schema_compliance(self, output: np.ndarray) -> Dict:
        """Evalúa cumplimiento de esquema."""
        metrics = {'schema_violations': 0}
        
        # Verificar NaN/Inf
        if np.any(np.isnan(output)) or np.any(np.isinf(output)):
            metrics['schema_violations'] += 1
        
        # Verificar dimensiones esperadas
        if len(output.shape) > 2:
            metrics['schema_violations'] += 1
        
        # Verificar rangos para probabilidades
        if len(output.shape) == 2:
            if np.any(output < 0) or np.any(output > 1):
                metrics['schema_violations'] += 1
            
            # Verificar que sumen ~1 por fila
            row_sums = np.sum(output, axis=1)
            if np.any(np.abs(row_sums - 1) > 0.01):
                metrics['schema_violations'] += 1
        
        return metrics
    
    def _check_calibration(self, output: np.ndarray, target: Optional[np.ndarray]) -> Dict:
        """Evalúa calibración de confianza."""
        metrics = {}
        
        if len(output.shape) > 1 and target is not None:
            n_samples, n_classes = output.shape
            
            # Convertir target a one-hot si es necesario
            if len(target.shape) == 1:
                target_one_hot = np.eye(n_classes)[target]
            else:
                target_one_hot = target
            
            # Calcular Expected Calibration Error (ECE)
            n_bins = 10
            bin_boundaries = np.linspace(0, 1, n_bins + 1)
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]
            
            confidences = np.max(output, axis=1)
            predictions = np.argmax(output, axis=1)
            
            ece = 0.0
            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                in_bin = (confidences >= bin_lower) & (confidences < bin_upper)
                prop_in_bin = np.mean(in_bin)
                
                if prop_in_bin > 0:
                    accuracy_in_bin = np.mean(
                        predictions[in_bin] == np.argmax(target_one_hot[in_bin], axis=1)
                    )
                    avg_confidence_in_bin = np.mean(confidences[in_bin])
                    ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
            metrics['calibration_error'] = float(ece)
            
            # Sobreconfianza
            overconfidence = np.mean(confidences[predictions != np.argmax(target_one_hot, axis=1)])
            metrics['overconfidence_rate'] = float(overconfidence)
        
        return metrics
    
    def _check_distribution_shift(self, output: np.ndarray, context: np.ndarray) -> Dict:
        """Evalúa cambios de distribución."""
        metrics = {}
        
        if output.shape == context.shape:
            # Distancia entre distribuciones (simplificada)
            output_mean = np.mean(output)
            output_std = np.std(output)
            context_mean = np.mean(context)
            context_std = np.std(context)
            
            # Distancia estandarizada
            mean_shift = np.abs(output_mean - context_mean) / (context_std + 1e-10)
            std_ratio = output_std / (context_std + 1e-10)
            
            metrics['mean_shift'] = float(mean_shift)
            metrics['std_ratio'] = float(std_ratio)
            metrics['distribution_shift'] = float(mean_shift + np.abs(np.log(std_ratio)))
        
        return metrics
    
    # === MÉTODOS AUXILIARES ===
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Softmax estable."""
        x_max = np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def _determine_severity(self, 
                           value: float, 
                           threshold: float, 
                           inverse: bool = False) -> str:
        """Determina severidad basada en umbral."""
        if inverse:
            # Valor más bajo es peor
            if value < threshold * 0.5:
                return 'critical'
            elif value < threshold * 0.7:
                return 'high'
            elif value < threshold * 0.9:
                return 'medium'
            else:
                return 'low'
        else:
            # Valor más alto es peor
            if value > threshold * 2.0:
                return 'critical'
            elif value > threshold * 1.5:
                return ''
                
    def _forward_pass(self, policy: Dict, state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass de la red neuronal en numpy.
        Retorna logits y probabilidades de acción.
        """
        x = state.copy()
        activations = [x.copy()]
        
        for i, layer in enumerate(policy['layers']):
            # Linear transformation
            x = np.dot(x, layer['weights']) + layer['biases']
            
            # Activation function
            if layer['activation'] == 'relu':
                x = np.maximum(0, x)
            elif layer['activation'] == 'softmax' and i == len(policy['layers']) - 1:
                # Softmax para capa de salida
                x = self._softmax_numpy(x)
            elif layer['activation'] == 'tanh':
                x = np.tanh(x)
            elif layer['activation'] == 'sigmoid':
                x = 1.0 / (1.0 + np.exp(-x))
            
            activations.append(x.copy())
        
        # Logits (antes de softmax) y probabilidades
        logits = activations[-2] if len(activations) > 1 else state
        action_probs = x
        
        return logits, action_probs              

    def _select_action(self, policy: Dict, state: np.ndarray, epsilon: float = None) -> Tuple[int, np.ndarray]:
        """
        Selecciona acción usando epsilon-greedy.
        """
        if epsilon is None:
            epsilon = self.epsilon
        
        # Forward pass para obtener probabilidades
        _, action_probs = self._forward_pass(policy, state)
        
        # Epsilon-greedy
        if np.random.random() < epsilon:
            # Exploración: acción aleatoria
            action = np.random.randint(0, policy['action_dim'])
        else:
            # Explotación: mejor acción según política
            action = np.argmax(action_probs)
        
        return action, action_probs

    def _apply_rl_intervention(self,
                         error_type: str,
                         input_data: np.ndarray,
                         output: np.ndarray,
                         target: Optional[np.ndarray],
                         context: Optional[np.ndarray],
                         error_metrics: Dict) -> Dict:
        """
        Aplica intervención usando Reinforcement Learning (numpy).
        """
        self.intervention_count += 1
        
        # Obtener estado para RL
        state = self._get_rl_state_numpy(error_type, input_data, output, error_metrics, context)
        
        # Seleccionar acción usando política RL
        policy = self.rl_policies[error_type]
        action, action_probs = self._select_action(policy, state, self.epsilon)
        
        # Aplicar intervención según acción seleccionada
        if error_type == 'factuality':
            result = self._rl_intervene_factuality_numpy(
                action, input_data, output, context, error_metrics
            )
        elif error_type == 'reasoning':
            result = self._rl_intervene_reasoning_numpy(
                action, input_data, output, target, error_metrics
            )
        elif error_type == 'schema':
            result = self._rl_intervene_schema_numpy(action, output, error_metrics)
        elif error_type == 'calibration':
            result = self._rl_intervene_calibration_numpy(action, output, target, error_metrics)
        elif error_type == 'distribution_shift':
            result = self._rl_intervene_distribution_shift_numpy(action, output, context, error_metrics)
        elif error_type == 'tool_use':
            result = self._rl_intervene_tool_use_numpy(action, input_data, output, context, error_metrics)
        elif error_type == 'refusal':
            result = self._rl_intervene_refusal_numpy(action, input_data, output, error_metrics)
        elif error_type == 'toxicity':
            result = self._rl_intervene_toxicity_numpy(action, output, error_metrics)
        elif error_type == 'style':
            result = self._rl_intervene_style_numpy(action, output, error_metrics)
        elif error_type == 'retrieval':
            result = self._rl_intervene_retrieval_numpy(action, output, context, error_metrics)
        elif error_type == 'long_context':
            result = self._rl_intervene_long_context_numpy(action, output, context, error_metrics)
        elif error_type == 'multi_turn':
            result = self._rl_intervene_multi_turn_numpy(action, input_data, output, context, error_metrics)
        elif error_type == 'overconfidence':
            result = self._rl_intervene_overconfidence_numpy(action, output, target, error_metrics)
        else:
            result = {
                'success': False,
                'intervention_type': 'unknown',
                'improvement': 0.0,
                'corrected_output': output
            }
        
        # Calcular recompensa RL
        if result['success']:
            next_state = self._get_rl_state_numpy(
                error_type, input_data, result['corrected_output'], 
                self._check_error(error_type, result['corrected_output'], target, context),
                context
            )
            
            reward = self._calculate_rl_reward_numpy(
                error_type, output, result['corrected_output'], error_metrics, result['improvement']
            )
            
            result.update({
                'rl_used': True,
                'rl_state': state,
                'rl_action': action,
                'rl_action_probs': action_probs,
                'rl_reward': reward,
                'next_state': next_state
            })
            
            # Actualizar estadísticas RL
            self.rl_stats[error_type]['total_rewards'] += reward
            self.rl_stats[error_type]['episodes'] += 1
            self.rl_stats[error_type]['avg_reward'] = (
                self.rl_stats[error_type]['total_rewards'] / 
                self.rl_stats[error_type]['episodes']
            )
            self.rl_stats[error_type]['exploration_rate'] = self.epsilon
            
            if reward > 0:
                successful_episodes = self.rl_stats[error_type].get('successful_episodes', 0) + 1
                self.rl_stats[error_type]['successful_episodes'] = successful_episodes
                self.rl_stats[error_type]['success_rate'] = (
                    successful_episodes / self.rl_stats[error_type]['episodes']
                )
        
        return result

    def _check_error(self, error_type: str, output: np.ndarray,
                    target: Optional[np.ndarray], context: Optional[np.ndarray]) -> Dict:
        """Revisa un tipo específico de error."""
        if error_type == 'factuality':
            return self._check_factuality(None, output, context)
        elif error_type == 'reasoning' and target is not None:
            return self._check_reasoning(None, output, target)
        elif error_type == 'schema':
            return self._check_schema_compliance(output)
        elif error_type == 'calibration' and target is not None:
            return self._check_calibration(output, target)
        elif error_type == 'distribution_shift' and context is not None:
            return self._check_distribution_shift(output, context)
        else:
            return {}
    
    def _rl_intervene_factuality_numpy(self,
                                action: int,
                                input_data: np.ndarray,
                                output: np.ndarray,
                                context: Optional[np.ndarray],
                                metrics: Dict) -> Dict:
        """
        Intervención RL para factualidad/alucinación (numpy).
        Acciones RL según tabla:
        0: Preference pairs rewarding grounded answers
        1: Penalizing unsupported claims
        2: Context-conditioned with citations
        """
        hallucination_risk = metrics.get('hallucination_risk', 0.0)
        context_correlation = metrics.get('context_correlation', 0.0)
        
        # Usar context_correlation para ajustar umbrales dinámicamente
        correlation_factor = max(0.5, context_correlation)  # Factor mínimo 0.5
        adjusted_improvement_threshold = 0.1 * correlation_factor
        
        if action == 0:
            # RL: Preference pairs rewarding grounded answers
            grounded_output = self._apply_grounded_answer_preference(output, context)
            improvement = self._calculate_factual_improvement(
                output, grounded_output, metrics, context
            )
            
            # Usar context_correlation en la condición
            if improvement > 0.15 and context_correlation > 0.3:
                self.successful_interventions += 1
                return {
                    'success': True,
                    'intervention_type': 'grounded_answer_preference',
                    'improvement': improvement,
                    'context_correlation_used': context_correlation,
                    'corrected_output': grounded_output
                }
        
        elif action == 1:
            # RL: Penalizing unsupported claims
            penalized_output = self._apply_unsupported_claim_penalty(output, context, metrics)
            improvement = self._calculate_factual_improvement(
                output, penalized_output, metrics, context
            )
            
            # Umbral ajustado por correlación de contexto
            threshold = max(0.08, 0.1 - (context_correlation * 0.05))
            if improvement > threshold:
                self.successful_interventions += 1
                return {
                    'success': True,
                    'intervention_type': 'unsupported_claim_penalty',
                    'improvement': improvement,
                    'context_correlation': context_correlation,
                    'threshold_used': threshold,
                    'corrected_output': penalized_output
                }
        
        elif action == 2:
            # RL: Context-conditioned with citations
            # Esta acción debería depender especialmente de context_correlation
            if context_correlation > 0.5:  # Solo intervenir si hay buena correlación
                cited_output = self._add_citations_to_output(output, context)
                improvement = self._calculate_factual_improvement(
                    output, cited_output, metrics, context
                )
            
                if improvement > (0.05 + context_correlation * 0.1):
                    self.successful_interventions += 1
                    return {
                        'success': True,
                        'intervention_type': 'context_citation',
                        'improvement': improvement,
                        'context_correlation': context_correlation,
                        'corrected_output': cited_output
                    }
        
        return {
            'success': False,
            'intervention_type': None,
            'improvement': 0.0,
            'context_correlation': context_correlation
        }

    def _rl_intervene_reasoning_numpy(self,
                               action: int,
                               input_data: np.ndarray,
                               output: np.ndarray,
                               target: Optional[np.ndarray],
                               metrics: Dict) -> Dict:
        """
        Intervención RL para reasoning (numpy).
        Acciones RL según tabla:
        0: FT: High-quality CoT with unit checks
        1: FT: Rejection-sampling k→ choose top 1
        2: RL: Preferences for correct step sequences
        3: RL: Final verification pass
        """
        reasoning_accuracy = metrics.get('reasoning_accuracy', 0.0)
        
        if action in [0,1,2] and target is not None:
            # RL: Preferences for correct step sequences
            step_corrected = self._apply_step_sequence_preference(output, target)
            improvement = self._calculate_reasoning_improvement(
                output, step_corrected, target, metrics
            )
            
            if improvement > 0.25:
                self.successful_interventions += 1
                return {
                    'success': True,
                    'intervention_type': 'step_sequence_preference',
                    'improvement': improvement,
                    'corrected_output': step_corrected
                }
        
        elif action == 3 and target is not None:
            # RL: Final verification pass
            verified_output = self._apply_final_verification(output, target)
            improvement = self._calculate_reasoning_improvement(
                output, verified_output, target, metrics
            )
            
            if improvement > 0.3:
                self.successful_interventions += 1
                return {
                    'success': True,
                    'intervention_type': 'final_verification_pass',
                    'improvement': improvement,
                    'corrected_output': verified_output
                }

    def _calculate_reasoning_improvement(self,
                                   original_output: np.ndarray,
                                   corrected_output: np.ndarray,
                                   target: np.ndarray,
                                   metrics: Dict) -> float:
        """
        Calcula mejora específica para errores de reasoning.
        """
        if target is None or original_output.shape != target.shape:
            return 0.0
        
        # Error cuadrático medio original y corregido
        mse_original = np.mean((original_output - target) ** 2)
        mse_corrected = np.mean((corrected_output - target) ** 2)
        
        # Reducción en MSE
        if mse_original > 0:
            mse_improvement = (mse_original - mse_corrected) / mse_original
        else:
            mse_improvement = 1.0 if mse_corrected == 0 else 0.0
        
        # Mejora en consistencia lógica
        if len(original_output.shape) > 1:
            # Para secuencias/matrices: consistencia de patrones
            original_pattern = np.diff(original_output, axis=0)
            corrected_pattern = np.diff(corrected_output, axis=0)
            target_pattern = np.diff(target, axis=0)
            
            pattern_corr_original = np.mean([
                np.corrcoef(original_pattern[:, i], target_pattern[:, i])[0, 1]
                for i in range(original_pattern.shape[1])
            ])
            
            pattern_corr_corrected = np.mean([
                np.corrcoef(corrected_pattern[:, i], target_pattern[:, i])[0, 1]
                for i in range(corrected_pattern.shape[1])
            ])
            
            pattern_improvement = max(0, pattern_corr_corrected - pattern_corr_original)
        else:
            pattern_improvement = 0.0
        
        # Mejora en accuracy (para clasificación)
        if len(original_output.shape) > 1 and original_output.shape[1] > 1:
            # Asumimos que son probabilidades de clase
            original_pred = np.argmax(original_output, axis=1)
            corrected_pred = np.argmax(corrected_output, axis=1)
            true_labels = np.argmax(target, axis=1) if len(target.shape) > 1 else target
            
            original_accuracy = np.mean(original_pred == true_labels)
            corrected_accuracy = np.mean(corrected_pred == true_labels)
            accuracy_improvement = max(0, corrected_accuracy - original_accuracy)
        else:
            accuracy_improvement = 0.0
        
        # Combinar métricas
        improvement_components = [
            (max(0, mse_improvement), 0.5),
            (pattern_improvement, 0.3),
            (accuracy_improvement, 0.2)
        ]
        
        reasoning_improvement = 0.0
        for value, weight in improvement_components:
            reasoning_improvement += value * weight
        
        return min(1.0, reasoning_improvement)
    
    def _rl_intervene_schema_numpy(self,
                            action: int,
                            output: np.ndarray,
                            metrics: Dict) -> Dict:
        """
        Intervención RL para schema violations (numpy).
        Acciones RL según tabla:
        0: FT: Schema-locked outputs (exact keys, examples + counter-examples)
        1: RL: Preferences that rank perfectly formatted outputs above near-misses
        """
        violations = metrics.get('schema_violations', 0)
        
        # RL: Preferences ranking perfect formats above near-misses
        preferred_output = self._apply_format_preference_ranking(output)
        improvement = self._calculate_schema_improvement(output, preferred_output, metrics)
            
        self.successful_interventions += 1
        return {
                    'success': True,
                    'intervention_type': 'format_preference_ranking',
                    'improvement': improvement,
                    'corrected_output': preferred_output
        }
        
        # Fallback
        return self._intervene_schema(output, metrics)

    def _rl_intervene_tool_use_numpy(self,
                              action: int,
                              input_data: np.ndarray,
                              output: np.ndarray,
                              context: Optional[np.ndarray],
                              metrics: Dict) -> Dict:
        """
        Intervención RL para tool/API use (numpy).
        Acciones RL según tabla:
        0: FT: Multi-turn tool chains
        1: FT: Failure→repair data pairs
        2: RL: Preferences ranking correct tool sequences
        3: RL: Ranking higher than superficially fluent but wrong ones
        4: Tool choice optimization
        """
        tool_accuracy = metrics.get('tool_accuracy', 0.0)
        
        if action in [1,2,3]:
            # RL: Preferences ranking correct tool sequences
            sequenced_output = self._apply_tool_sequence_preference(output, context)
            improvement = self._calculate_tool_use_improvement(output, sequenced_output, metrics)
        
        if improvement > 0.3:
            self.successful_interventions += 1
            return {
                'success': True,
                'intervention_type': 'tool_sequence_preference',
                'improvement': improvement,
                'corrected_output': sequenced_output
            }
    
        elif action == 3:
            # RL: Ranking higher than superficially fluent but wrong ones
            corrected_output = self._apply_fluency_correction(output, context)
            improvement = self._calculate_tool_use_improvement(output, corrected_output, metrics)
            
            if improvement > 0.15:
                self.successful_interventions += 1
                return {
                    'success': True,
                    'intervention_type': 'fluency_correction',
                    'improvement': improvement,
                    'corrected_output': corrected_output
                }
        
        elif action == 4:
            # Tool choice optimization
            optimized_output = self._apply_tool_choice_optimization_numpy(output, context)
            improvement = self._calculate_tool_use_improvement(output, optimized_output, metrics)
            
            if improvement > 0.2:
                self.successful_interventions += 1
                return {
                    'success': True,
                    'intervention_type': 'tool_choice_optimization',
                    'improvement': improvement,
                    'corrected_output': optimized_output
                }
    
        return {
            'success': False,
            'intervention_type': 'tool_use_intervention',
            'improvement': 0.0,
            'corrected_output': output
        }

    def _rl_intervene_refusal_numpy(self,
                             action: int,
                             input_data: np.ndarray,
                             output: np.ndarray,
                             metrics: Dict) -> Dict:
        """
        Intervención RL para under/over refusal (numpy).
        Acciones RL según tabla:
        0: FT: Red-team → safe refusal + helpful alternative guidance
        1: RL: Preferences that strongly favor correct refusals over partial answers
        2: RL: Preferences penalizing over-refusal vs. helpful compliant responses
        """
        refusal_type = metrics.get('refusal_type', 'unknown')
        
        if refusal_type == 'under' and action in [0, 1]:
            if action == 0:
                # FT: Red-team → safe refusal + helpful alternative guidance
                safe_output = self._apply_safe_refusal_with_alternative(output, input_data)
                improvement = self._calculate_refusal_improvement(output, safe_output, metrics)
                
                if improvement > 0.35:
                    self.successful_interventions += 1
                    return {
                        'success': True,
                    'intervention_type': 'safe_refusal_with_alternative',
                        'improvement': improvement,
                        'corrected_output': safe_output
                    }
            
            elif action == 1:
                # RL: Preferences favoring correct refusals over partial answers
                preferred_output = self._apply_refusal_preference(output, input_data)
                improvement = self._calculate_refusal_improvement(output, preferred_output, metrics)
                
                if improvement > 0.4:
                    self.successful_interventions += 1
                    return {
                        'success': True,
                        'intervention_type': 'refusal_preference',
                        'improvement': improvement,
                        'corrected_output': preferred_output
                    }
        
        elif refusal_type == 'over' and action == 2:
            # RL: Preferences penalizing over-refusal vs. helpful compliant responses
            compliant_output = self._apply_helpful_compliant_response(output, input_data)
            improvement = self._calculate_refusal_improvement(output, compliant_output, metrics)
            
            if improvement > 0.3:
                self.successful_interventions += 1
                return {
                    'success': True,
                    'intervention_type': 'helpful_compliant_response',
                    'improvement': improvement,
                    'corrected_output': compliant_output
                }
        
        return {
            'success': False,
            'intervention_type': 'refusal_intervention',
            'improvement': 0.0,
            'corrected_output': output
        }

    def _rl_intervene_toxicity_numpy(self,
                              action: int,
                              output: np.ndarray,
                              metrics: Dict) -> Dict:
        """
        Intervención RL para toxicidad/bias (numpy).
        Acciones RL según tabla:
        0: FT: Counter-bias & sensitive-topic exemplars with neutral reframing
        1: RL: Preferences that downrank toxic/biased outputs vs. neutral equivalents
        2: Neutral reframing with preference learning
        """
        toxicity_score = metrics.get('toxicity_score', 0.0)
        bias_score = metrics.get('bias_score', 0.0)
        
        if action == 0:
            # FT: Counter-bias & sensitive-topic exemplars
            reframed_output = self._apply_counter_bias_reframing_numpy(output)
            improvement = self._calculate_toxicity_improvement(output, reframed_output, metrics)
            
            if improvement > 0.4:
                self.successful_interventions += 1
                return {
                    'success': True,
                    'intervention_type': 'counter_bias_reframing',
                    'improvement': improvement,
                    'corrected_output': reframed_output
                }
        
        elif action == 1:
            # RL: Preferences downranking toxic/biased outputs
            downranked_output = self._apply_toxicity_downranking(output)
            improvement = self._calculate_toxicity_improvement(output, downranked_output, metrics)
            
            if improvement > 0.35:
                self.successful_interventions += 1
                return {
                    'success': True,
                    'intervention_type': 'toxicity_downranking',
                    'improvement': improvement,
                    'corrected_output': downranked_output
                }
        
        elif action == 2:
            # Neutral reframing with preference learning
            neutral_output = self._apply_neutral_reframing_with_preferences(output)
            improvement = self._calculate_toxicity_improvement(output, neutral_output, metrics)
            
            if improvement > 0.3:
                self.successful_interventions += 1
                return {
                    'success': True,
                    'intervention_type': 'neutral_reframing_with_preferences',
                    'improvement': improvement,
                    'corrected_output': neutral_output
                }
        
        return {
            'success': False,
            'intervention_type': 'toxicity_intervention',
            'improvement': 0.0,
            'corrected_output': output
        }

    def _rl_intervene_style_numpy(self,
                           action: int,
                           output: np.ndarray,
                           metrics: Dict) -> Dict:
        """
        Intervención RL para estilo/tono (numpy).
        Acciones RL según tabla:
        0: FT: Parallel style pairs (concise vs. detailed) with targets
        1: RL: Dual-objective preferences (quality + brevity)
        """
        verbosity_score = metrics.get('verbosity_score', 0.5)
        clarity_score = metrics.get('clarity_score', 0.5)
        
        if action == 0:
            # FT: Parallel style pairs
            style_adjusted = self._apply_parallel_style_adjustment(output, metrics)
            improvement = self._calculate_style_improvement(output, style_adjusted, metrics)
            
            if improvement > 0.25:
                self.successful_interventions += 1
                return {
                    'success': True,
                    'intervention_type': 'parallel_style_adjustment',
                    'improvement': improvement,
                    'corrected_output': style_adjusted
                }
        
        elif action == 1:
            # RL: Dual-objective preferences (quality + brevity)
            dual_optimized = self._apply_dual_objective_optimization(output, metrics)
            improvement = self._calculate_style_improvement(output, dual_optimized, metrics)
        
            if improvement > 0.3:
                self.successful_interventions += 1
                return {
                    'success': True,
                    'intervention_type': 'dual_objective_optimization',
                    'improvement': improvement,
                    'corrected_output': dual_optimized
                }
        
        return {
            'success': False,
            'intervention_type': 'style_intervention',
            'improvement': 0.0,
            'corrected_output': output
        }

    def _rl_intervene_retrieval_numpy(self,
                               action: int,
                               output: np.ndarray,
                               context: np.ndarray,
                               metrics: Dict) -> Dict:
        """
        Intervención RL para retrieval/grounding failures (numpy).
        Acciones RL según tabla:
        0: FT: Context-conditioned answers with gold span attribution
        1: RL: Preferences that favor answers using correct spans
        2: RL: Preferences over plausible but ungrounded ones
        3: Hard negative learning
        """
        grounding_score = metrics.get('grounding_score', 0.0)
        
        if action == 0 and context is not None:
            # FT: Context-conditioned answers with gold span attribution
            attributed_output = self._apply_gold_span_attribution(output, context)
            improvement = self._calculate_retrieval_improvement(output, attributed_output, metrics, context)
            
            if improvement > 0.35:
                self.successful_interventions += 1
                return {
                    'success': True,
                    'intervention_type': 'gold_span_attribution',
                    'improvement': improvement,
                    'corrected_output': attributed_output
                }
        
        elif action == 1 and context is not None:
            # RL: Preferences favoring answers using correct spans
            span_preferred = self._apply_correct_span_preference(output, context)
            improvement = self._calculate_retrieval_improvement(output, span_preferred, metrics, context)
            
            if improvement > 0.4:
                self.successful_interventions += 1
                return {
                    'success': True,
                    'intervention_type': 'correct_span_preference',
                    'improvement': improvement,
                    'corrected_output': span_preferred
                }
        
        elif action == 2:
            # RL: Preferences over plausible but ungrounded ones
            grounded_output = self._apply_grounded_over_plausible(output, context)
            improvement = self._calculate_retrieval_improvement(output, grounded_output, metrics, context)
            
            if improvement > 0.3:
                self.successful_interventions += 1
                return {
                    'success': True,
                    'intervention_type': 'grounded_over_plausible',
                    'improvement': improvement,
                    'corrected_output': grounded_output
                }
        
        elif action == 3 and context is not None:
            # Hard negative learning
            hard_negative_output = self._apply_hard_negative_learning(output, context)
            improvement = self._calculate_retrieval_improvement(output, hard_negative_output, metrics, context)
            
            if improvement > 0.25:
                self.successful_interventions += 1
                return {
                    'success': True,
                    'intervention_type': 'hard_negative_learning',
                    'improvement': improvement,
                    'corrected_output': hard_negative_output
                }
        
        return {
            'success': False,
            'intervention_type': 'retrieval_intervention',
            'improvement': 0.0,
            'corrected_output': output
        }

    def _rl_intervene_long_context_numpy(self,
                                  action: int,
                                  output: np.ndarray,
                                  context: np.ndarray,
                                  metrics: Dict) -> Dict:
        """
        Intervención RL para long-context failures (numpy).
        Acciones RL según tabla:
        0: FT: Long-context tasks (cross-section synthesis, citation)
        1: RL: Preferences rewarding correct long-range use
        2: RL: Preferences over myopic answers
        """
        context_length = metrics.get('context_length', 0)
        
        if action == 0 and context is not None and context_length > 1000:
            # FT: Long-context tasks (cross-section synthesis)
            synthesized_output = self._apply_cross_section_synthesis_numpy(output, context)
            improvement = self._calculate_long_context_improvement(output, synthesized_output, metrics, context)
            
            if improvement > 0.3:
                self.successful_interventions += 1
                return {
                    'success': True,
                    'intervention_type': 'cross_section_synthesis',
                    'improvement': improvement,
                    'corrected_output': synthesized_output
                }
        
        elif action == 1 and context is not None:
            # RL: Preferences rewarding correct long-range use
            long_range_output = self._apply_long_range_preference(output, context)
            improvement = self._calculate_long_context_improvement(output, long_range_output, metrics, context)
            
            if improvement > 0.35:
                self.successful_interventions += 1
                return {
                    'success': True,
                    'intervention_type': 'long_range_preference',
                    'improvement': improvement,
                    'corrected_output': long_range_output
                }
        
        elif action == 2:
            # RL: Preferences over myopic answers
            non_myopic_output = self._apply_non_myopic_preference(output, context)
            improvement = self._calculate_long_context_improvement(output, non_myopic_output, metrics, context)
            
            if improvement > 0.25:
                self.successful_interventions += 1
                return {
                    'success': True,
                    'intervention_type': 'non_myopic_preference',
                    'improvement': improvement,
                    'corrected_output': non_myopic_output
                }
        
        return {
            'success': False,
            'intervention_type': 'long_context_intervention',
            'improvement': 0.0,
            'corrected_output': output
        }

    def _rl_intervene_multi_turn_numpy(self,
                                action: int,
                                input_data: np.ndarray,
                                output: np.ndarray,
                                context: np.ndarray,
                                metrics: Dict) -> Dict:
        """
        Intervención RL para multi-turn inconsistency (numpy).
        Acciones RL según tabla:
        0: FT: Dialogue with state persistence & self-correction
        1: RL: Preferences ranking consistent multi-turn outputs
        2: RL: Preferences above single-turn optimal but inconsistent
        3: State-aware consistency optimization
        """
        inconsistency_score = metrics.get('inconsistency_score', 0.0)
        
        if action == 0:
            # FT: Dialogue with state persistence & self-correction
            persistent_output = self._apply_state_persistence_correction_numpy(output, context)
            improvement = self._calculate_multi_turn_improvement(output, persistent_output, metrics, context)
            
            if improvement > 0.35:
                self.successful_interventions += 1
                return {
                    'success': True,
                    'intervention_type': 'state_persistence_correction',
                    'improvement': improvement,
                    'corrected_output': persistent_output
                }
        
        elif action == 1:
            # RL: Preferences ranking consistent multi-turn outputs
            consistent_output = self._apply_consistency_preference(output, context)
            improvement = self._calculate_multi_turn_improvement(output, consistent_output, metrics, context)
            
            if improvement > 0.4:
                self.successful_interventions += 1
                return {
                    'success': True,
                    'intervention_type': 'consistency_preference',
                    'improvement': improvement,
                    'corrected_output': consistent_output
                }
        
        elif action == 2:
            # RL: Preferences above single-turn optimal but inconsistent
            optimal_consistent = self._apply_optimal_consistency_balance(output, context)
            improvement = self._calculate_multi_turn_improvement(output, optimal_consistent, metrics, context)
            
            if improvement > 0.3:
                self.successful_interventions += 1
                return {
                    'success': True,
                    'intervention_type': 'optimal_consistency_balance',
                    'improvement': improvement,
                    'corrected_output': optimal_consistent
                }
        
        elif action == 3:
            # State-aware consistency optimization
            state_aware_output = self._apply_state_aware_optimization(output, context)
            improvement = self._calculate_multi_turn_improvement(output, state_aware_output, metrics, context)
            
            if improvement > 0.25:
                self.successful_interventions += 1
                return {
                    'success': True,
                    'intervention_type': 'state_aware_optimization',
                    'improvement': improvement,
                    'corrected_output': state_aware_output
                }
        
        return {
            'success': False,
            'intervention_type': 'multi_turn_intervention',
            'improvement': 0.0,
            'corrected_output': output
        }

    def _rl_intervene_overconfidence_numpy(self,
                                    action: int,
                                    output: np.ndarray,
                                    target: Optional[np.ndarray],
                                    metrics: Dict) -> Dict:
        """
        Intervención RL para overconfidence/miscalibration (numpy).
        Acciones RL según tabla:
        0: FT: Patterns with ”I don’t know” in low-evidence cases
        1: RL: Preferences that up-rank calibrated admissions over being confidently wrong
        """
        calibration_error = metrics.get('calibration_error', 0.0)
        # RL: Preferences up-ranking calibrated admissions
        calibrated_output = self._apply_calibrated_admission_preference(output, target, metrics)
        improvement = self._calculate_overconfidence_improvement(output, calibrated_output, metrics, target)
        
        if improvement > 0.35:
            self.successful_interventions += 1
            return {
                'success': True,
                'intervention_type': 'calibrated_admission_preference',
                'improvement': improvement,
                'corrected_output': calibrated_output
            }
    
        # Fallback
        return self._intervene_calibration(output, target, metrics)

    def _rl_intervene_distribution_shift_numpy(self,
                                        action: int,
                                        output: np.ndarray,
                                        context: np.ndarray,
                                        metrics: Dict) -> Dict:
        """
        Intervención RL para distribution shift (numpy).
        Acciones RL según tabla:
        0: FT: Paraphrase/domain-augmented prompts with matched targets
        1: RL: Preferences that prefer robust paraphrase-invariant outputs
        """
        distribution_shift = metrics.get('distribution_shift', 0.0)
        
        if action == 0 and context is not None:
            # FT: Paraphrase/domain-augmented prompts
            augmented_output = self._apply_paraphrase_augmentation(output, context)
            improvement = self._calculate_distribution_improvement(output, augmented_output, context, metrics)
            
            if improvement > 0.25:
                self.successful_interventions += 1
                return {
                    'success': True,
                    'intervention_type': 'paraphrase_augmentation',
                    'improvement': improvement,
                    'corrected_output': augmented_output
                }
        
        elif action == 1:
            # RL: Preferences for robust paraphrase-invariant outputs
            robust_output = self._apply_paraphrase_invariant_preference(output, context)
            improvement = self._calculate_distribution_improvement(output, robust_output, context, metrics)
            
            if improvement > 0.3:
                self.successful_interventions += 1
                return {
                    'success': True,
                    'intervention_type': 'paraphrase_invariant_preference',
                    'improvement': improvement,
                    'corrected_output': robust_output
                }
        
        # Fallback
        return self._intervene_distribution_shift(output, context, metrics)

    def _calculate_rl_reward_numpy(self,
                            error_type: str,
                            original_output: np.ndarray,
                            corrected_output: np.ndarray,
                            metrics: Dict,
                            improvement: float) -> float:
        """
        Calcula recompensa RL usando solo numpy.
        """
        base_reward = improvement * 10.0  # Escala la mejora
        
        # Factores específicos según tipo de error y tabla
        reward_factors = {
            'factuality': {
                'reward': lambda: self._calculate_groundedness_score(corrected_output, metrics.get('context', None)) * 2.0,
                'penalty': lambda: -self._count_unsupported_claims(corrected_output) * 1.0,
                'bonus': lambda: self._calculate_citation_accuracy(corrected_output, metrics.get('context', None)) * 1.5
            },
            'reasoning': {
                'reward': lambda: metrics.get('step_accuracy', 0.0) * 1.0,
                'bonus': lambda: self._calculate_verification_score_numpy(corrected_output) * 2.0,
                'penalty': lambda: -metrics.get('logical_errors', 0) * 0.5
            },
            'schema': {
                'bonus': lambda: (1.0 if self._is_perfectly_formatted(corrected_output) else 0.0) * 3.0,
                'penalty': lambda: -self._count_near_misses(corrected_output) * 0.3
            },
            'tool_use': {
                'bonus': lambda: self._calculate_tool_sequence_accuracy(corrected_output) * 2.0,
                'wrong_choice_penalty': lambda: -self._count_wrong_tool_choices(corrected_output) * 1.0,
                'fluent_wrong_penalty': lambda: -self._calculate_fluent_but_wrong_score(corrected_output) * 0.5
            },
            'refusal': {
                'correct_refusal_bonus': lambda: (1.0 if self._is_correct_refusal(corrected_output, metrics) else 0.0) * 2.5,
                'partial_answer_penalty': lambda: -self._has_partial_answer(corrected_output) * 1.0,
                'over_refusal_penalty': lambda: -self._is_over_refusal(corrected_output, metrics) * 0.8,
                'helpful_bonus': lambda: self._calculate_helpfulness_score(corrected_output) * 1.5
            },
            'toxicity': {
                'neutral_bonus': lambda: self._calculate_neutrality_score(corrected_output) * 2.0,
                'penalty': lambda: -self._calculate_toxicity_score_numpy(corrected_output) * 2.0,
                'counter_bias_bonus': lambda: self._calculate_counter_bias_score(corrected_output) * 1.5
            },
            'style': {
                'brevity_bonus': lambda: max(0, 1.0 - self._calculate_verbosity_score(corrected_output)) * 1.0,
                'clarity_bonus': lambda: self._calculate_clarity_score_numpy(corrected_output) * 1.5,
                'penalty': lambda: -max(0, self._calculate_verbosity_score(corrected_output) - 0.7) * 0.5
            },
            'retrieval': {
                'correct_span_bonus': lambda: self._calculate_span_accuracy(corrected_output, metrics.get('context', None)) * 2.0,
                'penalty': lambda: -self._count_ungrounded_claims(corrected_output) * 1.0,
                'hard_negative_bonus': lambda: self._calculate_hard_negative_improvement(corrected_output, original_output) * 1.0
            },
            'long_context': {
                'long_range_bonus': lambda: self._calculate_long_range_usage(corrected_output, metrics.get('context', None)) * 2.5,
                'penalty': lambda: -self._calculate_myopic_score(corrected_output) * 1.0,
                'cross_doc_bonus': lambda: self._calculate_cross_document_coherence(corrected_output, metrics.get('context', None)) * 1.5
            },    
            'multi_turn': {
                'consistency_bonus': lambda: (1.0 - metrics.get('inconsistency_score', 1.0)) * 2.0,
                'penalty': lambda: -metrics.get('inconsistency_score', 0.0) * 1.5,
                'state_persistence_bonus': lambda: self._calculate_state_persistence_score(corrected_output, metrics.get('context', None)) * 1.5
            },
            'overconfidence': {
                'calibrated_bonus': lambda: (1.0 - metrics.get('calibration_error', 1.0)) * 2.0,
                'penalty': lambda: -self._calculate_confidently_wrong_score(corrected_output, metrics) * 1.5,
                'uncertainty_bonus': lambda: self._calculate_uncertainty_expression_score(corrected_output) * 1.0
            },
            'distribution_shift': {
                'invariant_bonus': lambda: self._calculate_paraphrase_invariance(corrected_output) * 2.0,
                'penalty': lambda: -self._calculate_fragility_score(corrected_output) * 1.0,
                'robustness_bonus': lambda: self._calculate_domain_robustness_score(corrected_output) * 1.5
            }
        }
    
        # Aplicar factores específicos del tipo de error
        if error_type in reward_factors:
            for factor_name, factor_func in reward_factors[error_type].items():
                try:
                    base_reward += factor_func()
                except:
                    pass
        
        # Normalizar y limitar recompensa
        reward = max(-10.0, min(10.0, base_reward))
        
        logger.debug(f"Recompensa RL numpy calculada para {error_type}: {reward:.4f}")
        return reward

    def _calculate_citation_accuracy(self, 
                               output: np.ndarray, 
                               context: Optional[np.ndarray]) -> float:
        """Bonus de precisión de citas/contexto."""
        if context is None or output.shape != context.shape:
            return 0.0
        
        # Similitud entre output y contexto
        if len(output.shape) == 1:
            correlation = np.corrcoef(output, context)[0, 1]
        else:
            # Para matrices: promedio de correlaciones por fila/columna
            correlations = []
            for i in range(min(output.shape[0], context.shape[0])):
                for j in range(min(output.shape[1], context.shape[1])):
                    if output.shape[0] > i and context.shape[0] > i:
                        corr = np.corrcoef(output[i], context[i])[0, 1]
                        correlations.append(corr)
            
            correlation = np.mean(correlations) if correlations else 0.0
        
        citation_accuracy = max(0, correlation)
        return float(citation_accuracy)

    def _calculate_clarity_score_numpy(self, output: np.ndarray) -> float:
        """Bonus de claridad."""
        if len(output.shape) == 0:
            return 0.0
        
        # Claridad basada en varianza/definición
        if len(output.shape) == 1:
            # Para vectores: kurtosis (picos definidos)
            if len(output) > 3:
                from scipy import stats
                try:
                    kurt = stats.kurtosis(output)
                    # Kurtosis positiva indica picos definidos (más clara)
                    clarity = 1.0 / (1.0 + np.exp(-kurt * 0.5))
                except:
                    clarity = 0.5
            else:
                clarity = 0.5
        elif len(output.shape) == 2 and output.shape[1] > 1:
            # Para clasificación: definición de probabilidades
            max_probs = np.max(output, axis=1)
            clarity = np.mean(max_probs)
        else:
            # Caso general: relación señal/ruido
            signal_power = np.mean(output ** 2)
            noise_power = np.var(output)
            
            if noise_power > 0:
                snr = signal_power / noise_power
                clarity = 1.0 - 1.0 / (1.0 + snr)
            else:
                clarity = 1.0
        
        return float(clarity)

    def scale_in(self, value: np.ndarray) -> np.ndarray:
        """Función de escala de entrada."""
        if len(value.shape) == 0:
            return value
        return value / (np.max(np.abs(value)) + 1e-10)
    
    def scale_out(self, value: np.ndarray) -> np.ndarray:
        """Función de escala de salida."""
        if len(value.shape) == 0:
            return value
        # Normalizar a rango [0, 1] si es apropiado
        min_val = np.min(value)
        max_val = np.max(value)
        
        if max_val - min_val > 1e-10:
            return (value - min_val) / (max_val - min_val)
        else:
            return value
    
    # Corregir condicionales en métodos existentes:
    def _rl_intervene_factuality_numpy(self,
                                action: int,
                                input_data: np.ndarray,
                                output: np.ndarray,
                                context: Optional[np.ndarray],
                                metrics: Dict) -> Dict:
        """
        Intervención RL para factualidad/alucinación (numpy).
        """
        hallucination_risk = metrics.get('hallucination_risk', 0.0)
        context_correlation = metrics.get('context_correlation', 0.0)
        
        if action == 0:
            # RL: Preference pairs rewarding grounded answers
            grounded_output = self._apply_grounded_answer_preference(output, context)
            improvement = self._calculate_factual_improvement(
                output, grounded_output, metrics, context
            )
            
            if improvement > 0.15:
                self.successful_interventions += 1
                return {
                    'success': True,
                    'intervention_type': 'grounded_answer_preference',
                    'improvement': improvement,
                    'corrected_output': grounded_output
                }
        
        elif action == 1:
            # RL: Penalizing unsupported claims
            penalized_output = self._apply_unsupported_claim_penalty(output, context, metrics)
            improvement = self._calculate_factual_improvement(
                output, penalized_output, metrics, context
            )
            
            if improvement > 0.1:
                self.successful_interventions += 1
                return {
                    'success': True,
                    'intervention_type': 'unsupported_claim_penalty',
                    'improvement': improvement,
                    'corrected_output': penalized_output
                }
        
        elif action == 2:
            # RL: Context-conditioned with citations
            cited_output = self._apply_context_with_citations(output, context)
            improvement = self._calculate_factual_improvement(
                output, cited_output, metrics, context
            )
            
            if improvement > 0.2:
                self.successful_interventions += 1
                return {
                    'success': True,
                    'intervention_type': 'context_with_citations',
                    'improvement': improvement,
                    'corrected_output': cited_output
                }
    
        # Fallback a intervención no RL
        return self._intervene_factuality(input_data, output, context, metrics)

    def _calculate_verification_score_numpy(self, output: np.ndarray) -> float:
        """Bonus de verificación para RL."""
        if len(output.shape) > 1:
            # Para clasificación: confianza en la predicción
            confidence = np.max(output, axis=1)
            verification_score = np.mean(confidence)
        else:
            # Para regresión: consistencia interna
            if len(output) > 2:
                autocorr = np.corrcoef(output[:-1], output[1:])[0, 1]
                verification_score = max(0, autocorr)
            else:
                verification_score = 0.5
        
        return float(verification_score)

    def _calculate_toxicity_score_numpy(self, output: np.ndarray) -> float:
        """Penalty de toxicidad (simplificado para numpy arrays)."""
        # Para texto necesitaríamos un modelo, pero para arrays numéricos:
        # Asumimos que valores extremos negativos podrían representar toxicidad
        if len(output.shape) == 0:
            return 0.0
        
        # Detectar valores extremadamente negativos
        mean_val = np.mean(output)
        std_val = np.std(output) + 1e-10
        z_scores = (output - mean_val) / std_val
        
        # Ponderar más los valores negativos extremos
        negative_extremes = np.sum(z_scores < -2.5) / output.size
        very_negative = np.sum(z_scores < -3.5) / output.size
        
        toxicity_score = negative_extremes * 0.7 + very_negative * 0.3
        return float(toxicity_score)
    
    def _update_rl_policy(self,
                    error_type: str,
                    state: np.ndarray,
                    action: int,
                    reward: float,
                    next_state: np.ndarray,
                    action_probs: Optional[np.ndarray] = None):
        """
        Actualiza la política RL usando solo numpy (DQN-style).
        """
        if error_type not in self.rl_policies:
            return
        
        # Almacenar experiencia en buffer
        experience = (state, action, reward, next_state, action_probs)
        self.replay_buffer.append(experience)
        
        # Entrenar si hay suficientes muestras
        if len(self.replay_buffer) >= RL_BATCH_SIZE:
            self._train_rl_policy_numpy(error_type)

    def _train_rl_policy_numpy(self, error_type: str):
        """
        Entrena la política RL usando experiencia replay (numpy).
        """
        policy = self.rl_policies[error_type]
        target_policy = self.target_policies[error_type]
        
        # Muestra aleatoria del buffer
        batch_indices = np.random.choice(len(self.replay_buffer), RL_BATCH_SIZE, replace=False)
        batch = [self.replay_buffer[i] for i in batch_indices]
        
        states, actions, rewards, next_states, _ = zip(*batch)
        
        # Convertir a arrays numpy
        states_array = np.array(states)
        actions_array = np.array(actions)
        rewards_array = np.array(rewards)
        next_states_array = np.array(next_states)
        
        # Obtener Q-values actuales
        current_q_values = self._get_q_values(policy, states_array)
        
        # Obtener target Q-values
        next_q_values = self._get_q_values(target_policy, next_states_array)
        max_next_q = np.max(next_q_values, axis=1)
        
        # Calcular target Q-values (DQN)
        target_q = current_q_values.copy()
        batch_indices = np.arange(RL_BATCH_SIZE)
        target_q[batch_indices, actions_array] = rewards_array + RL_DISCOUNT_FACTOR * max_next_q
        
        # Calcular pérdida (MSE)
        loss = np.mean((current_q_values - target_q) ** 2)
        
        # Actualizar política usando gradiente descendente simple
        self._update_policy_gradient(policy, states_array, target_q, RL_LEARNING_RATE)
        
        # Soft update de target network
        self._soft_update_policy(policy, target_policy, RL_TAU)
        
        # Registrar pérdida
        self.rl_stats[error_type]['last_loss'] = loss
        logger.debug(f"Política RL {error_type} entrenada, pérdida: {loss:.6f}")

    def _get_q_values(self, policy: Dict, states: np.ndarray) -> np.ndarray:
        """
        Obtiene Q-values de la política para un batch de estados.
        """
        batch_size = states.shape[0]
        q_values = np.zeros((batch_size, policy['action_dim']))
        
        for i in range(batch_size):
            # Forward pass para obtener logits (que usamos como Q-values aproximados)
            logits, _ = self._forward_pass(policy, states[i])
            q_values[i] = logits
        
        return q_values

    def _update_policy_gradient(self, policy: Dict, states: np.ndarray, target_q: np.ndarray, learning_rate: float):
        """
        Actualiza los pesos de la política usando gradiente descendente.
        """
        batch_size = states.shape[0]
        
        # Forward pass para obtener activaciones
        activations = [states.copy()]
        layer_outputs = [states.copy()]
        
        current_input = states
        for layer in policy['layers'][:-1]:  # Todas menos la última
            # Linear
            z = np.dot(current_input, layer['weights']) + layer['biases']
            layer_outputs.append(z)
            
            # ReLU activation
            a = np.maximum(0, z)
            activations.append(a)
            current_input = a
        
        # Capa de salida (sin activación para Q-values)
        last_layer = policy['layers'][-1]
        output = np.dot(current_input, last_layer['weights']) + last_layer['biases']
        layer_outputs.append(output)
        
        # Backward pass
        d_output = (output - target_q) / batch_size
        
        # Gradientes para capa de salida
        last_grad_w = np.dot(activations[-1].T, d_output)
        last_grad_b = np.sum(d_output, axis=0)
        
        # Actualizar pesos capa de salida
        last_layer['weights'] -= learning_rate * last_grad_w
        last_layer['biases'] -= learning_rate * last_grad_b
        
        # Backprop through hidden layers
        d_hidden = d_output
        for i in reversed(range(len(policy['layers']) - 1)):
            layer = policy['layers'][i]
            
            # Gradiente a través de ReLU
            d_relu = d_hidden * (layer_outputs[i+1] > 0).astype(float)
            
            # Gradientes para pesos y biases
            grad_w = np.dot(activations[i].T, d_relu)
            grad_b = np.sum(d_relu, axis=0)
            
            # Actualizar pesos
            layer['weights'] -= learning_rate * grad_w
            layer['biases'] -= learning_rate * grad_b
            
            # Preparar gradiente para siguiente capa
            d_hidden = np.dot(d_relu, layer['weights'].T)

    def _soft_update_policy(self, policy: Dict, target_policy: Dict, tau: float):
        """
        Soft update de target network (polyak averaging).
        """
        for i in range(len(policy['layers'])):
            target_policy['layers'][i]['weights'] = (
                tau * policy['layers'][i]['weights'] + 
                (1 - tau) * target_policy['layers'][i]['weights']
            )
            target_policy['layers'][i]['biases'] = (
                tau * policy['layers'][i]['biases'] + 
                (1 - tau) * target_policy['layers'][i]['biases']
            )


    def _get_rl_state_numpy(self,
                     error_type: str,
                     input_data: np.ndarray,
                     output: np.ndarray,
                     metrics: Dict,
                     context: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Construye el estado para RL usando solo numpy.
        """
        state_features = []
        
        # Métricas básicas (normalizadas)
        base_metrics = ['hallucination_risk', 'reasoning_accuracy', 'schema_violations',
                       'calibration_error', 'distribution_shift', 'max_step_error',
                       'cot_consistency_threshold']
        
        for metric in base_metrics:
            if metric in metrics:
                state_features.append(min(1.0, max(0.0, float(metrics[metric]))))
            else:
                state_features.append(0.0)
        
        # Características específicas por tipo de error
        if error_type == 'factuality':
            if context is not None:
                context_corr = metrics.get('context_correlation', 0.0)
                state_features.append(max(0.0, context_corr))
            
            extreme_values = metrics.get('extreme_values', 0.0)
            state_features.append(min(1.0, extreme_values))
            
            # Tamaño del output normalizado
            if hasattr(output, 'shape'):
                size_norm = min(1.0, output.size / 1000.0)
                state_features.append(size_norm)
            else:
                state_features.append(0.0)
        
        elif error_type == 'reasoning':
            # Error por paso
            step_error = metrics.get('step_error', 0.0)
            state_features.append(min(1.0, step_error))
            
            # Consistencia de reasoning
            consistency = metrics.get('reasoning_consistency', 0.5)
            state_features.append(consistency)
        
        elif error_type == 'tool_use':
            tool_acc = metrics.get('tool_accuracy', 0.0)
            state_features.append(tool_acc)
            
            arg_acc = metrics.get('argument_accuracy', 0.0)
            state_features.append(arg_acc)
            
            follow_up = metrics.get('follow_up_score', 0.0)
            state_features.append(follow_up)
        
        # Rellenar o truncar a la dimensión esperada
        policy_config = {
            'factuality': 10, 'reasoning': 12, 'schema': 8, 'calibration': 10,
            'distribution_shift': 8, 'tool_use': 15, 'refusal': 10, 'toxicity': 12,
            'style': 6, 'retrieval': 14, 'long_context': 16, 'multi_turn': 18,
            'overconfidence': 10
        }
        
        expected_dim = policy_config.get(error_type, 10)
        
        if len(state_features) < expected_dim:
            # Rellenar con ceros
            state_features.extend([0.0] * (expected_dim - len(state_features)))
        elif len(state_features) > expected_dim:
            # Truncar
            state_features = state_features[:expected_dim]
        
        return np.array(state_features, dtype=np.float32)

    def get_rl_statistics_numpy(self) -> Dict:
        """
        Obtiene estadísticas de los sistemas RL numpy.
        """
        if not self.use_rl:
            return {'rl_enabled': False}
        
        stats = {
            'rl_enabled': True,
            'epsilon': self.epsilon,
            'replay_buffer_size': len(self.replay_buffer),
            'total_episodes': sum(data['episodes'] for data in self.rl_stats.values()),
            'policies': {}
        }
        
        for error_type, data in self.rl_stats.items():
            stats['policies'][error_type] = {
                'episodes': data['episodes'],
                'avg_reward': data.get('avg_reward', 0.0),
                'success_rate': data.get('success_rate', 0.0),
            'exploration_rate': data.get('exploration_rate', self.epsilon),
                'last_loss': data.get('last_loss', 0.0)
            }
        
        return stats


    def _apply_grounded_answer_preference(self,
                                   output: np.ndarray,
                                   context: Optional[np.ndarray]) -> np.ndarray:
        """
        Aplica preferencia por respuestas con evidencia (RL para factualidad).
        """
        if context is None:
            return output
        
        # Calcular similitud entre output y contexto
        if output.shape == context.shape:
            similarity = np.corrcoef(output.flatten(), context.flatten())[0, 1]
            
            # Aumentar peso del contexto basado en similitud
            alpha = max(0.3, min(0.8, similarity))
            grounded = alpha * context + (1 - alpha) * output
            return grounded
        
        return output

    def _apply_unsupported_claim_penalty(self,
                                  output: np.ndarray,
                                  context: Optional[np.ndarray],
                                  metrics: Dict) -> np.ndarray:
        """
        Penaliza afirmaciones no soportadas (RL para factualidad).
        """
        # Identificar valores extremos (potenciales afirmaciones no soportadas)
        mean_val = np.mean(output)
        if len(output.shape) > 0:
            std_val = np.std(output) + 1e-10
            
            # Valores extremos (más de 3 desviaciones estándar)
            z_scores = np.abs((output - mean_val) / std_val)
            extreme_mask = z_scores > 3.0
            
            # Reducir valores extremos
            penalized = output.copy()
            penalized[extreme_mask] = mean_val + np.sign(output[extreme_mask] - mean_val) * 2.0 * std_val
            
            return penalized
        
        return output

    def _apply_context_with_citations(self,
                               output: np.ndarray,
                               context: np.ndarray) -> np.ndarray:
        """
        Aplica contexto con citas requeridas (FT para factualidad).
        """
        if output.shape == context.shape:
            # Marcar elementos del contexto como "citados"
            citation_strength = 0.7
            cited_output = citation_strength * context + (1 - citation_strength) * output
            
            # Añadir "marca de cita" a valores altos del contexto
            context_high = context > np.percentile(context, 75)
            cited_output[context_high] = context[context_high] * 0.9 + cited_output[context_high] * 0.1
            
            return cited_output
    
        return output

    def _apply_cot_with_unit_checks(self,
                            output: np.ndarray,
                            target: np.ndarray) -> np.ndarray:
        """
        Aplica Chain-of-Thought con verificaciones de unidad (FT para reasoning).
        """
        if output.shape == target.shape:
            # Verificar consistencia de unidades/magnitudes
            output_magnitude = np.mean(np.abs(output))
            target_magnitude = np.mean(np.abs(target)) + 1e-10
            
            # Escalar output para coincidir con magnitud del target
            scale_factor = target_magnitude / output_magnitude
            scaled_output = output * scale_factor
            
            # Mezclar con target basado en error
            error = np.mean(np.abs(scaled_output - target))
            blend_factor = min(0.5, error)
            
            corrected = (1 - blend_factor) * scaled_output + blend_factor * target
            return corrected
        
        return output
        
    def _apply_rejection_sampling(self,
                           output: np.ndarray,
                           input_data: np.ndarray) -> np.ndarray:
        """
        Aplica rejection sampling k→ choose top 1 (FT para reasoning).
        """
        # Generar múltiples candidatos mediante perturbaciones
        n_candidates = 5
        candidates = []
        
        for i in range(n_candidates):
            # Perturbar output con leakage
            noise = np.random.normal(0, 0.1, output.shape)
            candidate = output + noise
            
            # Normalizar si es necesario
            if len(output.shape) > 1:
                row_sums = np.sum(candidate, axis=1, keepdims=True)
                candidate = candidate / (row_sums + 1e-10)
            
            candidates.append(candidate)
        
        # Seleccionar el más "consistente" (menor varianza)
        consistencies = []
        for candidate in candidates:
            if len(candidate.shape) > 1:
                row_vars = np.var(candidate, axis=1)
                consistency = 1.0 / (np.mean(row_vars) + 1e-10)
            else:
                consistency = 1.0 / (np.var(candidate) + 1e-10)
            consistencies.append(consistency)
        
        best_idx = np.argmax(consistencies)
        return candidates[best_idx]

    def _apply_step_sequence_preference(self,
                                 output: np.ndarray,
                                 target: np.ndarray) -> np.ndarray:    
        """
        Aplica preferencia por secuencias de pasos correctas (RL para reasoning).
        """
        if output.shape == target.shape and len(output.shape) > 1:
            # Verificar consistencia paso a paso
            step_diffs = np.diff(output, axis=0)
            target_diffs = np.diff(target, axis=0)
            
            step_correlation = np.mean([
                np.corrcoef(step_diffs[:, i], target_diffs[:, i])[0, 1]
                for i in range(step_diffs.shape[1])
            ])
            
            # Ajustar output para mejorar correlación de pasos
            if step_correlation < 0.8:
                # Mezclar con target diferencias
                alpha = 0.3
                adjusted_diffs = (1 - alpha) * step_diffs + alpha * target_diffs
                
                # Reconstruir output
                corrected = output.copy()
                corrected[1:] = output[0:1] + np.cumsum(adjusted_diffs, axis=0)
                return corrected
    
        return output

    def _apply_final_verification(self,
                           output: np.ndarray,
                           target: np.ndarray) -> np.ndarray:
        """
        Aplica paso final de verificación (RL para reasoning).
        """
        if output.shape == target.shape:
            # Verificar consistencia global
            global_error = np.mean(np.abs(output - target))
            
            if global_error > 0.1:
                # Aplicar corrección proporcional al error
                correction_strength = min(0.5, global_error)
                verified = (1 - correction_strength) * output + correction_strength * target
                
                # Suavizar transiciones abruptas
                if len(output.shape) == 1 and len(output) > 2:
                    verified = np.convolve(verified, [0.25, 0.5, 0.25], mode='same')
                
                return verified
        
        return output

    def _apply_schema_locked_output(self, output: np.ndarray) -> np.ndarray:
        """
        Aplica salida bloqueada por esquema (FT para schema violations).
        """
        locked = output.copy()
        
        # Forzar cumplimiento estricto de esquema
        if len(locked.shape) == 2:
            # Probabilidades deben sumar 1
            row_sums = np.sum(locked, axis=1, keepdims=True)
            locked = locked / (row_sums + 1e-10)
            
            # Forzar rango [0, 1]
            locked = np.clip(locked, 0, 1)
            
            # Redondear valores muy cercanos a 0 o 1
            locked[locked < 0.01] = 0.0
            locked[locked > 0.99] = 1.0
        
        # Eliminar NaN/Inf
        locked = np.nan_to_num(locked, nan=0.0, posinf=1.0, neginf=0.0)
        
        return locked

    def _apply_format_preference_ranking(self, output: np.ndarray) -> np.ndarray:
        """
        Aplica ranking de preferencia de formatos (RL para schema violations).
        """
        # Generar formato "perfecto" como referencia
        if len(output.shape) == 2:
            perfect_format = output.copy()
            
            # Hacer distribución más "extrema" (preferencia por certeza)
            row_max = np.max(perfect_format, axis=1, keepdims=True)
            perfect_format = (perfect_format == row_max).astype(float)
            
            # Normalizar
            row_sums = np.sum(perfect_format, axis=1, keepdims=True)
            perfect_format = perfect_format / (row_sums + 1e-10)
            
            # Mezclar con output original basado en "cercanía" al formato perfecto
            similarity = np.mean([
                np.corrcoef(perfect_format[i], output[i])[0, 1]
                for i in range(output.shape[0])
            ])
        
            blend = max(0.1, similarity)
            preferred = blend * perfect_format + (1 - blend) * output
            
            return preferred
        
        return output

    def _calculate_groundedness_score(self, output: np.ndarray, context: Optional[np.ndarray]) -> float:
        """
        Calcula score de groundedness para recompensa RL.
        """
        if context is None or output.shape != context.shape:
            return 0.0
        
        # Calcular correlación con contexto
        correlation = np.corrcoef(output.flatten(), context.flatten())[0, 1]
        return max(0.0, correlation)

    def _count_unsupported_claims(self, output: np.ndarray) -> float:
        """
        Cuenta afirmaciones no soportadas (valores extremos).
        """
        if len(output.shape) == 0:
            return 0.0
        
        mean_val = np.mean(output)
        std_val = np.std(output) + 1e-10
        z_scores = np.abs((output - mean_val) / std_val)
        
        # Porcentaje de valores extremos
        extreme_ratio = np.mean(z_scores > 3.0)
        return extreme_ratio

# Los métodos restantes seguirían patrones similares...

                