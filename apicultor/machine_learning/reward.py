# === fase1_reward_integration.py ===
"""
FASE 1: Integración completa de RewardModel y sistema de monitoreo básico
con tu código existente de AdversarialValidator y módulos de atención.
"""
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import logging
import hashlib
import json
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

# ============================================================================
# 1. REWARD MODEL COMPLETO CON INTEGRACIÓN A TU CÓDIGO
# ============================================================================

@dataclass
class RewardComponent:
    """Componente individual de recompensa con tracking de calidad"""
    name: str
    weight: float
    calculator: Callable
    history: List[float] = field(default_factory=list)
    mean: float = 0.0
    std: float = 0.0
    confidence: float = 0.0

class IntegratedRewardModel:
    """
    Reward Model completamente integrado con tus módulos existentes:
    - AdversarialValidator para robustness scoring
    - attention_module para calidad de CoT
    - fairness_module para fairness scoring
    - explain_module para feature importance
    """
    
    def __init__(self, 
                 base_model: BaseEstimator,
                 adversarial_validator: Any,
                 attention_module: Any,
                 fairness_module: Any,
                 explain_module: Any,
                 config: Optional[Dict] = None):
        
        self.base_model = base_model
        self.validator = adversarial_validator
        self.attention = attention_module
        self.fairness = fairness_module
        self.explain = explain_module
        
        self.config = config or {
            'uncertainty_threshold': 0.3,
            'min_confidence': 0.7,
            'max_history_size': 1000,
            'reward_components': {
                'accuracy': 0.4,
                'robustness': 0.25,
                'fairness': 0.15,
                'cot_quality': 0.1,
                'feature_importance': 0.05,
                'uncertainty_penalty': -0.05,
                'hacking_penalty': -0.1
            }
        }
        
        # Inicializar componentes de recompensa
        self.components = self._initialize_components()
        
        # Historial y estadísticas
        self.reward_history = []
        self.uncertainty_history = []
        self.hacking_detections = []
        
        # Integración con tu pipeline existente
        self._setup_integration_hooks()
    
    def _initialize_components(self) -> Dict[str, RewardComponent]:
        """Inicializa todos los componentes de recompensa"""
        return {
            'accuracy': RewardComponent(
                name='accuracy',
                weight=self.config['reward_components']['accuracy'],
                calculator=self._calculate_accuracy_reward
            ),
            'robustness': RewardComponent(
                name='robustness',
                weight=self.config['reward_components']['robustness'],
                calculator=self._calculate_robustness_reward
            ),
            'fairness': RewardComponent(
                name='fairness',
                weight=self.config['reward_components']['fairness'],
                calculator=self._calculate_fairness_reward
            ),
            'cot_quality': RewardComponent(
                name='cot_quality',
                weight=self.config['reward_components']['cot_quality'],
                calculator=self._calculate_cot_quality_reward
            ),
            'feature_importance': RewardComponent(
                name='feature_importance',
                weight=self.config['reward_components']['feature_importance'],
                calculator=self._calculate_feature_importance_reward
            ),
            'uncertainty_penalty': RewardComponent(
                name='uncertainty_penalty',
                weight=self.config['reward_components']['uncertainty_penalty'],
                calculator=self._calculate_uncertainty_penalty
            ),
            'hacking_penalty': RewardComponent(
                name='hacking_penalty',
                weight=self.config['reward_components']['hacking_penalty'],
                calculator=self._calculate_hacking_penalty
            )
        }
    
    def _setup_integration_hooks(self):
        """Configura hooks para integración con tus módulos existentes"""
        # Hook para AdversarialValidator
        if hasattr(self.validator, 'validate'):
            original_validate = self.validator.validate
            
            async def enhanced_validate(*args, **kwargs):
                # Ejecutar validación original
                result = await original_validate(*args, **kwargs)
                
                # Añadir reward scoring
                if 'X' in kwargs and 'y' in kwargs:
                    reward_result = await self.score_batch(
                        kwargs['X'], kwargs['y'], 
                        result.get('predictions', None)
                    )
                    result['reward_analysis'] = reward_result
                
                return result
            
            self.validator.validate = enhanced_validate
        
        # Hook para attention_module
        if hasattr(self.attention, 'continuous_multi_lorax'):
            original_lorax = self.attention.continuous_multi_lorax
            
            async def enhanced_lorax(*args, **kwargs):
                # Ejecutar atención original
                result = await original_lorax(*args, **kwargs)
                
                # Calcular calidad de CoT
                if hasattr(result, '__iter__') and len(result) >= 4:
                    cot_quality = self._evaluate_cot_from_attention(result[3])
                    result = result + (cot_quality,)
                
                return result
            
            self.attention.continuous_multi_lorax = enhanced_lorax
    
    async def score(self, 
                    input: np.ndarray, 
                    output: np.ndarray,
                    target: Optional[np.ndarray] = None,
                    context: Optional[np.ndarray] = None,
                    chain_of_thought: Optional[List[str]] = None,
                    metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Calcula recompensa completa integrando todos tus módulos
        """
        scores = {}
        uncertainties = {}
        component_details = {}
        
        # 1. Accuracy Reward (si hay target)
        if target is not None:
            acc_reward, acc_uncertainty = await self._calculate_accuracy_reward(
                input, output, target
            )
            scores['accuracy'] = acc_reward
            uncertainties['accuracy'] = acc_uncertainty
            component_details['accuracy'] = {
                'value': acc_reward,
                'uncertainty': acc_uncertainty,
                'confidence': 1.0 - acc_uncertainty
            }
        
        # 2. Robustness Reward (usando tu AdversarialValidator)
        robustness_reward, robustness_uncertainty = await self._calculate_robustness_reward(
            input, output
        )
        scores['robustness'] = robustness_reward
        uncertainties['robustness'] = robustness_uncertainty
        component_details['robustness'] = {
            'value': robustness_reward,
            'uncertainty': robustness_uncertainty,
            'confidence': 1.0 - robustness_uncertainty
        }
        
        # 3. Fairness Reward (usando tu fairness_module)
        if hasattr(self.fairness, 'p_rule'):
            fairness_reward, fairness_uncertainty = await self._calculate_fairness_reward(
                input, output, target
            )
            scores['fairness'] = fairness_reward
            uncertainties['fairness'] = fairness_uncertainty
            component_details['fairness'] = {
                'value': fairness_reward,
                'uncertainty': fairness_uncertainty,
                'confidence': 1.0 - fairness_uncertainty
            }
        
        # 4. CoT Quality Reward (si hay chain_of_thought)
        if chain_of_thought:
            cot_reward, cot_uncertainty = await self._calculate_cot_quality_reward(
                chain_of_thought, output
            )
            scores['cot_quality'] = cot_reward
            uncertainties['cot_quality'] = cot_uncertainty
            component_details['cot_quality'] = {
                'value': cot_reward,
                'uncertainty': cot_uncertainty,
                'confidence': 1.0 - cot_uncertainty
            }
        elif context is not None and hasattr(self.attention, 'context_vector'):
            inferred_cot = self._infer_cot_from_context(context)
            cot_reward, cot_uncertainty = await self._calculate_cot_quality_reward(
                inferred_cot, output
            )
            scores['cot_quality'] = cot_reward
            component_details['cot_quality'] = {
                'value': cot_reward,
                'uncertainty': cot_uncertainty,
                'confidence': 1.0 - cot_uncertainty,
                'inferred': True
            }
        
        # 5. Feature Importance Reward (usando tu explain_module)
        if hasattr(self.explain, 'explain'):
            feature_reward, feature_uncertainty = await self._calculate_feature_importance_reward(
                input, output
            )
            scores['feature_importance'] = feature_reward
            uncertainties['feature_importance'] = feature_uncertainty
            component_details['feature_importance'] = {
                'value': feature_reward,
                'uncertainty': feature_uncertainty,
                'confidence': 1.0 - feature_uncertainty
            }
        
        # 6. Uncertainty Penalty
        uncertainty_penalty, overall_uncertainty = await self._calculate_uncertainty_penalty(
            uncertainties
        )
        scores['uncertainty_penalty'] = uncertainty_penalty
        
        # 7. Hacking Penalty (detección de reward hacking)
        hacking_penalty, hacking_flags = await self._calculate_hacking_penalty(
            input, output, scores
        )
        scores['hacking_penalty'] = hacking_penalty
        
        # 8. Calcular recompensa compuesta
        composite_reward = self._compute_composite_reward(scores)
        
        # 9. Actualizar historial y estadísticas
        reward_record = {
            'timestamp': datetime.now(),
            'composite_reward': composite_reward,
            'component_scores': scores,
            'uncertainty': overall_uncertainty,
            'hacking_flags': hacking_flags,
            'metadata': metadata or {}
        }
        
        self.reward_history.append(reward_record)
        self.uncertainty_history.append(overall_uncertainty)
        
        if hacking_flags:
            self.hacking_detections.append({
                'timestamp': datetime.now(),
                'flags': hacking_flags,
                'input_hash': hashlib.md5(input.tobytes()).hexdigest()[:16]
            })
        
        # 10. Actualizar estadísticas de componentes
        self._update_component_statistics(scores)
        
        return {
            'composite_reward': float(composite_reward),
            'component_scores': {k: float(v) for k, v in scores.items()},
            'uncertainty': float(overall_uncertainty),
            'hacking_flags': hacking_flags,
            'component_details': component_details,
            'metadata': {
                'input_shape': input.shape,
                'output_shape': output.shape,
                'component_count': len(scores)
            }
        }
    
    async def _calculate_accuracy_reward(self, input: np.ndarray, 
                                       output: np.ndarray, 
                                       target: np.ndarray) -> Tuple[float, float]:
        """Calcula recompensa de exactitud con incertidumbre"""
        try:
            if output.shape == target.shape:
                if len(output.shape) == 1 or output.shape[1] == 1:
                    error = mean_squared_error(target, output)
                    reward = 1.0 - min(error, 1.0)
                    residuals = target - output
                    uncertainty = np.std(residuals) / (np.std(target) + 1e-10)
                    return float(reward), float(min(uncertainty, 1.0))
                else:
                    if hasattr(self.base_model, 'predict_proba'):
                        probs = self.base_model.predict_proba(input.reshape(1, -1))
                        confidence = np.max(probs)
                        uncertainty = 1.0 - confidence
                        return confidence, float(uncertainty)
                    else:
                        predictions = np.argmax(output, axis=1) if output.ndim > 1 else output
                        targets = np.argmax(target, axis=1) if target.ndim > 1 else target
                        accuracy = accuracy_score(targets, predictions)
                        return float(accuracy), 0.1
            else:
                return 0.5, 0.5
        except Exception as e:
            logger.warning(f"Error calculando accuracy reward: {e}")
            return 0.5, 0.5
    
    async def _calculate_robustness_reward(self, input: np.ndarray, 
                                         output: np.ndarray) -> Tuple[float, float]:
        """Calcula recompensa de robustez usando tu AdversarialValidator"""
        try:
            if hasattr(self.validator, 'run_adversarial_tests'):
                robustness_scores = await self.validator.run_adversarial_tests(
                    input.reshape(1, -1), output.reshape(1, -1)
                )
                if isinstance(robustness_scores, dict):
                    if 'robustness_score' in robustness_scores:
                        reward = robustness_scores['robustness_score']
                    elif 'mean_accuracy' in robustness_scores:
                        reward = robustness_scores['mean_accuracy']
                    else:
                        scores_list = [v for v in robustness_scores.values() 
                                     if isinstance(v, (int, float))]
                        reward = np.mean(scores_list) if scores_list else 0.5
                else:
                    reward = 0.5
                return float(reward), 0.1
            else:
                return 0.5, 0.3
        except Exception as e:
            logger.warning(f"Error calculando robustness reward: {e}")
            return 0.5, 0.5
    
    async def _calculate_fairness_reward(self, input: np.ndarray,
                                       output: np.ndarray,
                                       target: Optional[np.ndarray]) -> Tuple[float, float]:
        """Calcula recompensa de fairness usando tu módulo"""
        try:
            if hasattr(self.fairness, 'p_rule') and target is not None:
                if hasattr(self.base_model, 'predict_proba'):
                    proba = self.base_model.predict_proba(input.reshape(1, -1))
                else:
                    from scipy.special import expit as sigmoid
                    proba = sigmoid(output) if output is not None else output
                
                theta = np.ones_like(output)
                
                fairness_score = self.fairness.p_rule(
                    output.reshape(1, -1) if output.ndim == 1 else output[:1],
                    target.reshape(1, -1) if target.ndim == 1 else target[:1],
                    theta[:1],
                    input.reshape(1, -1) if input.ndim == 1 else input[:1],
                    proba[:1] if proba is not None else np.array([[0.5]]),
                    thresh=1e-4
                )
                
                if isinstance(fairness_score, bool):
                    reward = 1.0 if fairness_score else 0.0
                elif isinstance(fairness_score, (int, float, np.number)):
                    reward = float(fairness_score)
                else:
                    reward = 0.5
                
                uncertainty = 0.2 if reward < 0.7 else 0.05
                return reward, uncertainty
            else:
                return 0.5, 0.3
        except Exception as e:
            logger.warning(f"Error calculando fairness reward: {e}")
            return 0.5, 0.5
    
    async def _calculate_cot_quality_reward(self, chain_of_thought: List[str],
                                          output: np.ndarray) -> Tuple[float, float]:
        """Evalúa calidad del Chain-of-Thought"""
        try:
            metrics = {
                'step_count': len(chain_of_thought),
                'step_lengths': [len(step) for step in chain_of_thought],
                'has_conclusion': any('conclusion' in step.lower() 
                                    or 'therefore' in step.lower() 
                                    for step in chain_of_thought),
                'has_analysis': any('analy' in step.lower() 
                                  or 'consider' in step.lower() 
                                  for step in chain_of_thought)
            }
            
            step_score = min(metrics['step_count'] / 5.0, 1.0)
            if metrics['step_lengths']:
                length_score = 1.0 - (np.std(metrics['step_lengths']) / 
                                    (np.mean(metrics['step_lengths']) + 1e-10))
            else:
                length_score = 0.5
            structure_score = 1.0 if metrics['has_analysis'] and metrics['has_conclusion'] else 0.5
            
            reward = 0.4 * step_score + 0.3 * length_score + 0.3 * structure_score
            uncertainty = 0.1 if step_score > 0.7 and structure_score > 0.7 else 0.3
            
            return float(reward), float(uncertainty)
        except Exception as e:
            logger.warning(f"Error calculando CoT quality reward: {e}")
            return 0.5, 0.5
    
    async def _calculate_feature_importance_reward(self, input: np.ndarray,
                                                 output: np.ndarray) -> Tuple[float, float]:
        """Calcula recompensa basada en importancia de características"""
        try:
            if hasattr(self.explain, 'explain') and hasattr(self.base_model, 'predict'):
                explanation = self.explain.explain(
                    self.base_model,
                    input.reshape(1, -1),
                    output.reshape(1, -1),
                    limit=['mean'],
                    Idxs=list(range(min(10, input.shape[0]))) if input.ndim > 0 else [0],
                    logical=[True]
                )
                
                if explanation and len(explanation) >= 2:
                    pos_scores = explanation[0]
                    neg_scores = explanation[1]
                    
                    if hasattr(pos_scores, '__len__') and len(pos_scores) > 0:
                        pos_mean = np.mean(pos_scores) if hasattr(pos_scores, 'mean') else 0.5
                        neg_mean = np.mean(neg_scores) if hasattr(neg_scores, 'mean') else 0.5
                        
                        clarity = abs(pos_mean - neg_mean)
                        consistency = 1.0 - (np.std(pos_scores) if hasattr(pos_scores, 'std') else 0.3)
                        
                        reward = 0.6 * clarity + 0.4 * consistency
                        uncertainty = 0.2 if clarity > 0.3 else 0.4
                        
                        return float(reward), float(uncertainty)
            
            return 0.5, 0.3
        except Exception as e:
            logger.warning(f"Error calculando feature importance reward: {e}")
            return 0.5, 0.5
    
    async def _calculate_uncertainty_penalty(self, 
                                           uncertainties: Dict[str, float]) -> Tuple[float, float]:
        """Calcula penalización por incertidumbre"""
        try:
            if uncertainties:
                avg_uncertainty = np.mean(list(uncertainties.values()))
                overall_uncertainty = float(avg_uncertainty)
                penalty = -min(avg_uncertainty, self.config['uncertainty_threshold'])
                return float(penalty), overall_uncertainty
            else:
                return 0.0, 0.3
        except Exception as e:
            logger.warning(f"Error calculando uncertainty penalty: {e}")
            return 0.0, 0.5
    
    async def _calculate_hacking_penalty(self, input: np.ndarray,
                                       output: np.ndarray,
                                       scores: Dict[str, float]) -> Tuple[float, List[str]]:
        """Detecta y penaliza reward hacking"""
        hacking_flags = []
        penalty = 0.0
        
        if 'accuracy' in scores and 'robustness' in scores:
            accuracy = scores['accuracy']
            robustness = scores['robustness']
            
            if accuracy > 0.9 and robustness < 0.6:
                hacking_flags.append("goodhart_law_suspected")
                penalty -= 0.2
        
        if len(self.reward_history) > 10:
            recent_rewards = [r['composite_reward'] 
                            for r in self.reward_history[-10:]]
            if np.mean(recent_rewards) > 0.95 and np.std(recent_rewards) < 0.05:
                hacking_flags.append("over_optimization_detected")
                penalty -= 0.15
        
        if len(output.shape) == 1:
            output_hash = hashlib.md5(output.tobytes()).hexdigest()[:8]
            hash_counts = {}
            for record in self.reward_history[-20:]:
                meta = record.get('metadata', {})
                out_hash = meta.get('output_hash', '')
                if out_hash:
                    hash_counts[out_hash] = hash_counts.get(out_hash, 0) + 1
            
            if hash_counts.get(output_hash, 0) > 3:
                hacking_flags.append("repetitive_output_pattern")
                penalty -= 0.1
        
        return float(penalty), hacking_flags
    
    def _evaluate_cot_from_attention(self, context: np.ndarray) -> float:
        """Evalúa calidad de CoT a partir del contexto de atención"""
        try:
            if context is None or len(context.shape) == 0:
                return 0.5
            
            entropy = -np.sum(context * np.log(context + 1e-10))
            normalized_entropy = entropy / np.log(len(context) + 1) if len(context) > 1 else 0.5
            
            sparsity = np.sum(np.abs(context) < 1e-10) / len(context) if len(context) > 0 else 0.5
            
            cot_quality = 0.5 * (1.0 - normalized_entropy) + 0.5 * (1.0 - sparsity)
            return float(np.clip(cot_quality, 0.0, 1.0))
        except Exception as e:
            logger.warning(f"Error evaluando CoT from attention: {e}")
            return 0.5
    
    def _infer_cot_from_context(self, context: np.ndarray) -> List[str]:
        """Infiere Chain-of-Thought a partir del contexto de atención"""
        inferred_steps = []
        
        if context is not None and len(context.shape) > 0:
            inferred_steps.append(
                f"Context analysis: shape={context.shape}, "
                f"mean={np.mean(context):.3f}, std={np.std(context):.3f}"
            )
            
            if len(context.shape) == 1 and len(context) > 5:
                significant = np.where(np.abs(context) > np.std(context))[0]
                if len(significant) > 0:
                    inferred_steps.append(
                        f"Significant attention at indices: {significant[:5].tolist()}"
                    )
            
            if hasattr(self.attention, 'context_vector'):
                inferred_steps.append(
                    "Attention context processed for information integration"
                )
            
            inferred_steps.append(
                "Reasoning completed, forming final prediction"
            )
        
        return inferred_steps
    
    def _compute_composite_reward(self, scores: Dict[str, float]) -> float:
        """Calcula recompensa compuesta ponderada"""
        composite = 0.0
        total_weight = 0.0
        
        for component_name, score in scores.items():
            if component_name in self.components:
                component = self.components[component_name]
                weight = component.weight
                composite += score * weight
                total_weight += abs(weight)
        
        if total_weight > 0:
            composite = composite / total_weight
        
        composite = np.clip(composite, 0.0, 1.0)
        return float(composite)
    
    def _update_component_statistics(self, scores: Dict[str, float]):
        """Actualiza estadísticas de componentes"""
        for component_name, score in scores.items():
            if component_name in self.components:
                component = self.components[component_name]
                component.history.append(score)
                
                if len(component.history) > self.config['max_history_size']:
                    component.history = component.history[-self.config['max_history_size']:]
                
                if component.history:
                    component.mean = np.mean(component.history)
                    component.std = np.std(component.history)
                    
                    if component.std > 0:
                        component.confidence = 1.0 / (1.0 + component.std)
                    else:
                        component.confidence = 1.0
    
    async def score_batch(self, 
                         inputs: np.ndarray, 
                         targets: np.ndarray,
                         predictions: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Calcula recompensas para un batch completo
        """
        batch_results = []
        batch_rewards = []
        
        with ThreadPoolExecutor() as executor:
            tasks = []
            batch_limit = min(len(inputs), 100)
            for i in range(batch_limit):
                input_sample = inputs[i]
                target_sample = targets[i] if i < len(targets) else None
                prediction = predictions[i] if predictions is not None and i < len(predictions) else None
                
                if prediction is None and hasattr(self.base_model, 'predict'):
                    if input_sample.ndim == 1:
                        input_reshaped = input_sample.reshape(1, -1)
                    else:
                        input_reshaped = input_sample[:1]
                    prediction = self.base_model.predict(input_reshaped)[0]
                
                task = asyncio.create_task(
                    self.score(
                        input=input_sample,
                        output=prediction,
                        target=target_sample,
                        metadata={'sample_index': i, 'batch_size': len(inputs)}
                    )
                )
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Error scoring sample {i}: {result}")
                    continue
                batch_results.append(result)
                batch_rewards.append(result.get('composite_reward', 0.5))
        
        if batch_rewards:
            batch_stats = {
                'mean_reward': float(np.mean(batch_rewards)),
                'std_reward': float(np.std(batch_rewards)),
                'min_reward': float(np.min(batch_rewards)),
                'max_reward': float(np.max(batch_rewards)),
                'median_reward': float(np.median(batch_rewards)),
                'sample_count': len(batch_results),
                'component_stats': self._get_component_statistics()
            }
        else:
            batch_stats = {}
        
        return {
            'batch_results': batch_results,
            'batch_statistics': batch_stats,
            'timestamp': datetime.now().isoformat()
        }
    
    def _get_component_statistics(self) -> Dict[str, Dict]:
        """Obtiene estadísticas de todos los componentes"""
        stats = {}
        
        for name, component in self.components.items():
            if component.history:
                stats[name] = {
                    'weight': component.weight,
                    'mean': component.mean,
                    'std': component.std,
                    'confidence': component.confidence,
                    'history_size': len(component.history),
                    'recent_values': component.history[-5:] if len(component.history) >= 5 else component.history
                }
        
        return stats
    
    def get_reward_model_metrics(self) -> Dict[str, Any]:
        """Obtiene métricas completas del reward model"""
        if not self.reward_history:
            return {'status': 'no_data_yet'}
        
        recent_rewards = [r['composite_reward'] for r in self.reward_history[-50:]]
        
        return {
            'summary': {
                'total_scorings': len(self.reward_history),
                'mean_composite_reward': float(np.mean(recent_rewards)) if recent_rewards else 0.0,
                'std_composite_reward': float(np.std(recent_rewards)) if len(recent_rewards) > 1 else 0.0,
                'mean_uncertainty': float(np.mean(self.uncertainty_history[-50:])) if self.uncertainty_history else 0.0,
                'hacking_detections': len(self.hacking_detections),
                'component_count': len(self.components)
            },
            'component_statistics': self._get_component_statistics(),
            'recent_rewards': recent_rewards[-10:] if recent_rewards else [],
            'config': self.config
        }

# ============================================================================
# 2. SISTEMA DE MONITOREO BÁSICO INTEGRADO
# ============================================================================

class BasicMonitoringSystem:
    """
    Sistema de monitoreo básico que se integra con tu código existente
    y con el RewardModel.
    """
    
    def __init__(self, 
                 reward_model: IntegratedRewardModel,
                 adversarial_validator: Any,
                 model: BaseEstimator):
        
        self.reward_model = reward_model
        self.validator = adversarial_validator
        self.model = model
        
        self.config = {
            'performance_metrics': ['accuracy', 'latency', 'throughput', 'error_rate'],
            'safety_metrics': ['constitution_violations', 'hacking_flags', 'uncertainty'],
            'cost_metrics': ['inference_cost', 'training_cost', 'storage_cost'],
            'check_intervals': {
                'performance': 300,
                'safety': 600,
                'costs': 3600
            },
            'alert_thresholds': {
                'performance_degradation': 0.15,
                'safety_violations': 3,
                'cost_increase': 0.2,
                'reward_collapse': 0.3
            }
        }
        
        self.metrics_history = {
            'performance': [],
            'safety': [],
            'costs': [],
            'rewards': [],
            'comprehensive': []
        }
        
        self.alerts = []
        self.alert_history = []
        
        self._setup_monitoring_hooks()
    
    def _setup_monitoring_hooks(self):
        """Configura hooks para monitoreo automático"""
        if hasattr(self.model, 'predict'):
            original_predict = self.model.predict
            
            def monitored_predict(*args, **kwargs):
                start_time = datetime.now()
                result = original_predict(*args, **kwargs)
                end_time = datetime.now()
                latency = (end_time - start_time).total_seconds()
                
                self._record_performance_metric('latency', latency, {
                    'input_shape': args[0].shape if args else 'unknown',
                    'samples': len(args[0]) if args and hasattr(args[0], 'shape') else 0
                })
                
                return result
            
            self.model.predict = monitored_predict
        
        if hasattr(self.validator, 'validate'):
            original_validate = self.validator.validate
            
            async def monitored_validate(*args, **kwargs):
                result = await original_validate(*args, **kwargs)
                
                if 'adversarial_results' in result:
                    adv_results = result['adversarial_results']
                    safety_score = self._calculate_safety_score(adv_results)
                    self._record_safety_metric('adversarial_safety', safety_score, {
                        'test_count': len(adv_results) if isinstance(adv_results, dict) else 0
                    })
                
                return result
            
            self.validator.validate = monitored_validate
    
    async def run_performance_monitoring(self, 
                                       X_sample: np.ndarray,
                                       y_sample: np.ndarray) -> Dict[str, Any]:
        """Ejecuta monitoreo de performance"""
        metrics = {}
        
        if hasattr(self.model, 'predict'):
            predictions = self.model.predict(X_sample)
            
            if predictions.shape == y_sample.shape:
                accuracy = 1.0 - mean_squared_error(y_sample, predictions)
                metrics['accuracy'] = float(accuracy)
                
                degradation = self._check_performance_degradation('accuracy', accuracy)
                if degradation > self.config['alert_thresholds']['performance_degradation']:
                    self._create_alert(
                        'performance_degradation',
                        f"Accuracy degradation detected: {degradation*100:.1f}%",
                        {'degradation': degradation, 'current_accuracy': accuracy}
                    )
        
        latencies = []
        sample_limit = min(10, len(X_sample))
        for _ in range(sample_limit):
            start = datetime.now()
            _ = self.model.predict(X_sample[0:1]) if hasattr(self.model, 'predict') else None
            latencies.append((datetime.now() - start).total_seconds())
        
        if latencies:
            metrics['latency_mean'] = float(np.mean(latencies))
            metrics['latency_std'] = float(np.std(latencies))
            metrics['latency_p95'] = float(np.percentile(latencies, 95))
        
        if 'latency_mean' in metrics and metrics['latency_mean'] > 0:
            metrics['throughput'] = 1.0 / metrics['latency_mean']
        
        if hasattr(self.model, 'predict'):
            try:
                corrupted = X_sample + np.random.normal(0, 0.1, X_sample.shape)
                corrupted_preds = self.model.predict(corrupted)
                
                if predictions is not None and corrupted_preds.shape == predictions.shape:
                    error_rate = mean_squared_error(predictions, corrupted_preds)
                    metrics['error_rate'] = float(error_rate)
            except:
                metrics['error_rate'] = 0.1
        
        self._record_performance_metrics(metrics)
        
        return {
            'performance_metrics': metrics,
            'timestamp': datetime.now().isoformat(),
            'sample_size': len(X_sample),
            'alerts_generated': self._get_recent_alerts('performance')
        }
    
    async def run_safety_monitoring(self,
                                  X_sample: np.ndarray,
                                  y_sample: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Ejecuta monitoreo de seguridad"""
        metrics = {}
        
        reward_metrics = self.reward_model.get_reward_model_metrics()
        if 'summary' in reward_metrics:
            summary = reward_metrics['summary']
            metrics['mean_reward'] = summary.get('mean_composite_reward', 0.0)
            metrics['reward_std'] = summary.get('std_composite_reward', 0.0)
            metrics['hacking_detections'] = summary.get('hacking_detections', 0)
            metrics['mean_uncertainty'] = summary.get('mean_uncertainty', 0.0)
        
        if hasattr(self.validator, 'run_adversarial_tests') and len(X_sample) > 0:
            try:
                sample = X_sample[0].reshape(1, -1) if X_sample[0].ndim == 1 else X_sample[:1]
                adversarial_results = await self.validator.run_adversarial_tests(sample, sample)
                
                if isinstance(adversarial_results, dict):
                    safety_scores = []
                    for key, value in adversarial_results.items():
                        if isinstance(value, dict) and 'mean_accuracy' in value:
                            safety_scores.append(value['mean_accuracy'])
                    
                    if safety_scores:
                        metrics['adversarial_safety_mean'] = float(np.mean(safety_scores))
                        metrics['adversarial_safety_std'] = float(np.std(safety_scores))
                        
                        if metrics['adversarial_safety_mean'] < 0.6:
                            self._create_alert(
                                'low_adversarial_safety',
                                f"Low adversarial safety: {metrics['adversarial_safety_mean']:.3f}",
                                {'safety_score': metrics['adversarial_safety_mean']}
                            )
            except Exception as e:
                logger.warning(f"Error in adversarial safety monitoring: {e}")
        
        if 'mean_uncertainty' in metrics:
            if metrics['mean_uncertainty'] > 0.4:
                self._create_alert(
                    'high_uncertainty',
                    f"High model uncertainty: {metrics['mean_uncertainty']:.3f}",
                    {'uncertainty': metrics['mean_uncertainty']}
                )
        
        if len(self.metrics_history['rewards']) > 10:
            recent_rewards = [m.get('mean_reward', 0.5) 
                            for m in self.metrics_history['rewards'][-10:]]
            
            if len(recent_rewards) >= 5:
                first_half = np.mean(recent_rewards[:5])
                second_half = np.mean(recent_rewards[5:])
                
                collapse_ratio = (first_half - second_half) / (first_half + 1e-10)
                
                if collapse_ratio > self.config['alert_thresholds']['reward_collapse']:
                    self._create_alert(
                        'reward_collapse_suspected',
                        f"Possible reward collapse detected: {collapse_ratio*100:.1f}% drop",
                        {'collapse_ratio': collapse_ratio, 
                         'first_half_mean': first_half,
                         'second_half_mean': second_half}
                    )
                
                metrics['reward_collapse_risk'] = float(collapse_ratio)
        
        self._record_safety_metrics(metrics)
        
        return {
            'safety_metrics': metrics,
            'timestamp': datetime.now().isoformat(),
            'alerts_generated': self._get_recent_alerts('safety'),
            'reward_model_status': 'active' if reward_metrics else 'inactive'
        }
    
    async def run_cost_monitoring(self,
                                inference_count: int = 1000,
                                training_hours: float = 1.0) -> Dict[str, Any]:
        """Ejecuta monitoreo de costos"""
        metrics = {}
        
        if hasattr(self.model, 'predict'):
            sample_size = min(100, inference_count)
            latencies = []
            
            for _ in range(sample_size):
                start = datetime.now()
                _ = self.model.predict(np.random.randn(1, 10))
                latencies.append((datetime.now() - start).total_seconds())
            
            if latencies:
                avg_latency = np.mean(latencies)
                
                cost_per_hour = 5.0
                tokens_per_inference = 100
                cost_per_1k_tokens = 0.002
                
                inferences_per_hour = 3600 / avg_latency if avg_latency > 0 else 1000
                inference_cost_per_hour = (inferences_per_hour * tokens_per_inference * 
                                         cost_per_1k_tokens / 1000)
                
                metrics['inference_cost_per_hour'] = float(inference_cost_per_hour)
                metrics['inferences_per_hour'] = float(inferences_per_hour)
                metrics['avg_latency_seconds'] = float(avg_latency)
        
        metrics['training_cost_per_hour'] = 5.0
        metrics['estimated_training_hours'] = training_hours
        metrics['total_training_cost'] = 5.0 * training_hours
        
        model_size_mb = 100
        storage_cost_per_gb_per_month = 0.1
        metrics['monthly_storage_cost'] = (model_size_mb / 1024) * storage_cost_per_gb_per_month
        
        metrics['total_monthly_cost'] = (
            metrics.get('inference_cost_per_hour', 0) * 24 * 30 +
            metrics.get('total_training_cost', 0) +
            metrics.get('monthly_storage_cost', 0)
        )
        
        if self.metrics_history['costs']:
            last_costs = self.metrics_history['costs'][-1]
            current_total = metrics.get('total_monthly_cost', 0)
            last_total = last_costs.get('total_monthly_cost', current_total)
            
            cost_increase = (current_total - last_total) / (last_total + 1e-10)
            
            if cost_increase > self.config['alert_thresholds']['cost_increase']:
                self._create_alert(
                    'cost_increase',
                    f"Cost increase detected: {cost_increase*100:.1f}%",
                    {'cost_increase': cost_increase,
                     'current_cost': current_total,
                     'previous_cost': last_total}
                )
            
            metrics['cost_increase_percentage'] = float(cost_increase * 100)
        
        self._record_cost_metrics(metrics)
        
        return {
            'cost_metrics': metrics,
            'timestamp': datetime.now().isoformat(),
            'assumptions': {
                'cost_per_hour': 5.0,
                'cost_per_1k_tokens': 0.002,
                'tokens_per_inference': 100
            },
            'alerts_generated': self._get_recent_alerts('cost')
        }
    
    async def run_comprehensive_monitoring(self,
                                         X_sample: np.ndarray,
                                         y_sample: np.ndarray,
                                         inference_count: int = 1000) -> Dict[str, Any]:
        """Ejecuta monitoreo completo"""
        monitoring_results = {}
        
        with ThreadPoolExecutor(max_workers=3) as executor:
            perf_future = asyncio.create_task(
                self.run_performance_monitoring(X_sample, y_sample)
            )
            
            safety_future = asyncio.create_task(
                self.run_safety_monitoring(X_sample, y_sample)
            )
            
            cost_future = asyncio.create_task(
                self.run_cost_monitoring(inference_count)
            )
            
            perf_results = await perf_future
            safety_results = await safety_future
            cost_results = await cost_future
        
        monitoring_results.update({
            'performance': perf_results,
            'safety': safety_results,
            'costs': cost_results,
            'overall_status': self._determine_overall_status(
                perf_results, safety_results, cost_results
            ),
            'timestamp': datetime.now().isoformat(),
            'monitoring_id': hashlib.md5(
                str(datetime.now()).encode()
            ).hexdigest()[:16]
        })
        
        self.metrics_history['comprehensive'].append(monitoring_results)
        
        return monitoring_results
    
    def _calculate_safety_score(self, adversarial_results: Dict) -> float:
        """Calcula score de seguridad basado en resultados adversariales"""
        safety_scores = []
        
        for key, value in adversarial_results.items():
            if isinstance(value, dict):
                if 'mean_accuracy' in value:
                    safety_scores.append(value['mean_accuracy'])
                elif 'robustness_score' in value:
                    safety_scores.append(value['robustness_score'])
        
        return float(np.mean(safety_scores)) if safety_scores else 0.5
    
    def _check_performance_degradation(self, metric_name: str, 
                                     current_value: float) -> float:
        """Verifica degradación de performance"""
        for m in self.metrics_history['performance']:
            if m.get('name') == metric_name:
                historical_values = [
                    m['value'] for m in self.metrics_history['performance']
                    if m.get('name') == metric_name
                ]
                
                if len(historical_values) >= 5:
                    historical_mean = np.mean(historical_values[-5:])
                    degradation = (historical_mean - current_value) / (historical_mean + 1e-10)
                    return max(degradation, 0.0)
        
        return 0.0
    
    def _create_alert(self, alert_type: str, message: str, details: Dict):
        """Crea una nueva alerta"""
        alert = {
            'id': hashlib.md5(f"{alert_type}{message}{datetime.now()}".encode()).hexdigest()[:16],
            'type': alert_type,
            'severity': self._determine_alert_severity(alert_type, details),
            'message': message,
            'timestamp': datetime.now().isoformat(),
            'details': details,
            'acknowledged': False,
            'resolved': False
        }
        
        self.alerts.append(alert)
        self.alert_history.append(alert)
        
        if len(self.alerts) > 100:
            self.alerts = self.alerts[-100:]
    
    def _determine_alert_severity(self, alert_type: str, details: Dict) -> str:
        """Determina severidad de alerta"""
        severity_map = {
            'reward_collapse_suspected': 'critical',
            'high_uncertainty': 'high',
            'low_adversarial_safety': 'high',
            'performance_degradation': 'medium',
            'cost_increase': 'low'
        }
        
        return severity_map.get(alert_type, 'medium')
    
    def _get_recent_alerts(self, category: Optional[str] = None) -> List[Dict]:
        """Obtiene alertas recientes"""
        recent_alerts = []
        
        for alert in self.alerts[-20:]:
            if category is None or alert['type'].startswith(category):
                recent_alerts.append({
                    'type': alert['type'],
                    'severity': alert['severity'],
                    'message': alert['message'],
                    'timestamp': alert['timestamp']
                })
        
        return recent_alerts
    
    def _record_performance_metric(self, name: str, value: float, metadata: Dict = None):
        """Registra una métrica de performance"""
        record = {
            'name': name,
            'value': value,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        
        self.metrics_history['performance'].append(record)
        
        if len(self.metrics_history['performance']) > 1000:
            self.metrics_history['performance'] = self.metrics_history['performance'][-1000:]
    
    def _record_performance_metrics(self, metrics: Dict[str, float]):
        """Registra múltiples métricas de performance"""
        for name, value in metrics.items():
            self._record_performance_metric(name, value)
    
    def _record_safety_metric(self, name: str, value: float, metadata: Dict = None):
        """Registra una métrica de seguridad"""
        record = {
            'name': name,
            'value': value,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        
        self.metrics_history['safety'].append(record)
        
        if len(self.metrics_history['safety']) > 1000:
            self.metrics_history['safety'] = self.metrics_history['safety'][-1000:]
    
    def _record_safety_metrics(self, metrics: Dict[str, float]):
        """Registra múltiples métricas de seguridad"""
        for name, value in metrics.items():
            self._record_safety_metric(name, value)
    
    def _record_cost_metric(self, name: str, value: float, metadata: Dict = None):
        """Registra una métrica de costo"""
        record = {
            'name': name,
            'value': value,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        
        self.metrics_history['costs'].append(record)
        
        if len(self.metrics_history['costs']) > 1000:
            self.metrics_history['costs'] = self.metrics_history['costs'][-1000:]
    
    def _record_cost_metrics(self, metrics: Dict[str, float]):
        """Registra múltiples métricas de costo"""
        for name, value in metrics.items():
            self._record_cost_metric(name, value)
    
    def _determine_overall_status(self, 
                                 perf_results: Dict, 
                                 safety_results: Dict, 
                                 cost_results: Dict) -> str:
        """Determina estado general del sistema"""
        status_components = []
        
        perf_metrics = perf_results.get('performance_metrics', {})
        if 'accuracy' in perf_metrics:
            if perf_metrics['accuracy'] < 0.7:
                status_components.append('performance_degraded')
        
        safety_metrics = safety_results.get('safety_metrics', {})
        if safety_metrics.get('hacking_detections', 0) > 5:
            status_components.append('security_risk')
        
        if safety_metrics.get('mean_uncertainty', 0) > 0.4:
            status_components.append('high_uncertainty')
        
        cost_metrics = cost_results.get('cost_metrics', {})
        if cost_metrics.get('cost_increase_percentage', 0) > 50:
            status_components.append('cost_escalation')
        
        if 'security_risk' in status_components or 'high_uncertainty' in status_components:
            return 'critical'
        elif 'performance_degraded' in status_components:
            return 'degraded'
        elif 'cost_escalation' in status_components:
            return 'warning'
        else:
            return 'healthy'
    
    def get_monitoring_summary(self) -> Dict[str, Any]:
        """Obtiene resumen del monitoreo"""
        summary = {
            'metrics_collected': {
                'performance': len(self.metrics_history['performance']),
                'safety': len(self.metrics_history['safety']),
                'costs': len(self.metrics_history['costs']),
                'rewards': len(self.metrics_history['rewards'])
            },
            'active_alerts': len(self.alerts),
            'total_alerts': len(self.alert_history),
            'system_status': 'unknown'
        }
        
        critical_alerts = [a for a in self.alerts if a['severity'] == 'critical']
        if critical_alerts:
            summary['system_status'] = 'critical'
        elif self.alerts:
            summary['system_status'] = 'warning'
        else:
            summary['system_status'] = 'healthy'
        
        if self.metrics_history['performance']:
            recent_perf = self.metrics_history['performance'][-10:]
            perf_values = [m['value'] for m in recent_perf if 'value' in m]
            if perf_values:
                summary['recent_performance'] = {
                    'mean': float(np.mean(perf_values)),
                    'std': float(np.std(perf_values))
                }
        
        if self.metrics_history['safety']:
            recent_safety = self.metrics_history['safety'][-10:]
            safety_values = [m['value'] for m in recent_safety if 'value' in m]
            if safety_values:
                summary['recent_safety'] = {
                    'mean': float(np.mean(safety_values)),
                    'std': float(np.std(safety_values))
                }
        
        return summary

# ============================================================================
# 3. INTEGRACIÓN COMPLETA FASE 1
# ============================================================================

class Phase1Integration:
    """
    Integración completa de Fase 1: Reward Model + Monitoreo Básico
    con tu código existente.
    """
    
    def __init__(self, 
                 base_model: BaseEstimator,
                 adversarial_validator: Any,
                 attention_module: Any,
                 fairness_module: Any,
                 explain_module: Any,
                 config: Optional[Dict] = None):
        
        self.base_model = base_model
        self.validator = adversarial_validator
        self.attention = attention_module
        self.fairness = fairness_module
        self.explain = explain_module
        
        print("🚀 Inicializando IntegratedRewardModel...")
        self.reward_model = IntegratedRewardModel(
            base_model=base_model,
            adversarial_validator=adversarial_validator,
            attention_module=attention_module,
            fairness_module=fairness_module,
            explain_module=explain_module,
            config=config.get('reward_model', {}) if config else None
        )
        
        print("📊 Inicializando BasicMonitoringSystem...")
        self.monitoring_system = BasicMonitoringSystem(
            reward_model=self.reward_model,
            adversarial_validator=adversarial_validator,
            model=base_model
        )
        
        self._setup_phase1_integration()
        
        self.phase_status = {
            'name': 'Phase 1 - Reward Model + Basic Monitoring',
            'status': 'initialized',
            'components': {
                'reward_model': 'active',
                'monitoring_system': 'active',
                'hooks_integrated': True
            },
            'timestamp': datetime.now().isoformat()
        }
        
        print("✅ Fase 1 integrada exitosamente")
    
    def _setup_phase1_integration(self):
        """Configura integración completa de Fase 1"""
        
        if hasattr(self.validator, 'GridSearch'):
            original_gridsearch = self.validator.GridSearch
            
            async def enhanced_gridsearch(*args, **kwargs):
                print("🔍 GridSearch mejorado con Reward Model...")
                
                results = await original_gridsearch(*args, **kwargs)
                
                if results and 'best_model' in results:
                    best_model = results['best_model']
                    
                    if 'X' in kwargs and 'y' in kwargs:
                        X = kwargs['X']
                        y = kwargs['y']
                        
                        if hasattr(best_model, 'predict'):
                            sample_limit = min(100, len(X))
                            predictions = best_model.predict(X[:sample_limit])
                            
                            reward_analysis = await self.reward_model.score_batch(
                                X[:sample_limit], y[:sample_limit], predictions
                            )
                            
                            results['reward_analysis'] = reward_analysis
                            
                            self.monitoring_system.metrics_history['rewards'].append({
                                'gridsearch_reward': reward_analysis['batch_statistics']['mean_reward'],
                                'timestamp': datetime.now().isoformat(),
                                'model_hash': hashlib.md5(
                                    str(best_model).encode()
                                ).hexdigest()[:16]
                            })
                
                return results
            
            self.validator.GridSearch = enhanced_gridsearch
        
        if hasattr(self.explain, 'dropout'):
            original_dropout = self.explain.dropout
            
            async def enhanced_dropout(*args, **kwargs):
                print("🎯 Dropout mejorado con monitoreo...")
                
                results = await original_dropout(*args, **kwargs)
                
                if results and isinstance(results, list):
                    dropout_metrics = {
                        'samples_processed': len(results),
                        'features_dropped': sum(
                            1 for r in results if isinstance(r, (list, tuple)) and len(r) > 3
                        ),
                        'dropout_timestamp': datetime.now().isoformat()
                    }
                    
                    self.monitoring_system._record_performance_metric(
                        'dropout_effectiveness',
                        dropout_metrics['samples_processed'] / max(len(args[0]), 1),
                        dropout_metrics
                    )
                
                return results
            
            self.explain.dropout = enhanced_dropout
    
    async def run_phase1_pipeline(self, 
                                 X: np.ndarray, 
                                 y: np.ndarray,
                                 test_size: float = 0.2) -> Dict[str, Any]:
        """
        Ejecuta pipeline completo de Fase 1
        """
        print("\n" + "="*80)
        print("🚀 EJECUTANDO FASE 1: REWARD MODEL + MONITORING INTEGRATION")
        print("="*80)
        
        phase_results = {
            'phase': 'phase1',
            'timestamp': datetime.now().isoformat(),
            'data_shape': {'X': X.shape, 'y': y.shape},
            'components_executed': [],
            'results': {}
        }
        
        print("\n1. 📊 PREPARANDO DATOS...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        phase_results['data_split'] = {
            'train_size': len(X_train),
            'test_size': len(X_test),
            'split_ratio': test_size
        }
        
        print("\n2. 🤖 ENTRENANDO MODELO BASE...")
        if hasattr(self.base_model, 'fit'):
            try:
                self.base_model.fit(X_train, y_train)
                phase_results['components_executed'].append('model_training')
                print(f"   ✅ Modelo entrenado con {len(X_train)} muestras")
            except Exception as e:
                print(f"   ⚠️  Model training error: {e}")
        
        print("\n3. 🏆 EVALUANDO CON REWARD MODEL...")
        test_limit = min(100, len(X_test))
        reward_results = await self.reward_model.score_batch(
            X_test[:test_limit],
            y_test[:test_limit]
        )
        
        phase_results['results']['reward_evaluation'] = reward_results
        phase_results['components_executed'].append('reward_evaluation')
        
        print(f"   ✅ Reward evaluation complete")
        print(f"   • Mean Reward: {reward_results['batch_statistics']['mean_reward']:.3f}")
        print(f"   • Samples: {reward_results['batch_statistics']['sample_count']}")
        
        print("\n4. 📈 EJECUTANDO MONITOREO COMPLETO...")
        monitor_limit = min(50, len(X_test))
        monitoring_results = await self.monitoring_system.run_comprehensive_monitoring(
            X_test[:monitor_limit],
            y_test[:monitor_limit],
            inference_count=1000
        )
        
        phase_results['results']['monitoring'] = monitoring_results
        phase_results['components_executed'].append('comprehensive_monitoring')
        
        print(f"   ✅ Monitoring complete")
        print(f"   • System Status: {monitoring_results['overall_status']}")
        print(f"   • Performance Metrics: {len(monitoring_results['performance']['performance_metrics'])}")
        print(f"   • Safety Metrics: {len(monitoring_results['safety']['safety_metrics'])}")
        
        print("\n5. 🛡️ EJECUTANDO VALIDACIÓN ADVERSARIAL MEJORADA...")
        if hasattr(self.validator, 'validate'):
            try:
                adv_limit = min(20, len(X_test))
                adversarial_results = await self.validator.validate(
                    X_test[:adv_limit],
                    y_test[:adv_limit]
                )
                
                phase_results['results']['adversarial_validation'] = adversarial_results
                phase_results['components_executed'].append('adversarial_validation')
                
                if 'reward_analysis' in adversarial_results:
                    print(f"   ✅ Adversarial validation with reward analysis complete")
                else:
                    print(f"   ✅ Adversarial validation complete")
            except Exception as e:
                print(f"   ⚠️  Adversarial validation error: {e}")
        
        print("\n6. 📋 GENERANDO REPORTE DE FASE 1...")
        phase_report = self._generate_phase1_report(phase_results)
        
        phase_results['phase_report'] = phase_report
        phase_results['phase_status'] = 'completed'
        
        self.phase_status.update({
            'status': 'completed',
            'completion_time': datetime.now().isoformat(),
            'metrics_summary': {
                'mean_reward': reward_results['batch_statistics']['mean_reward'],
                'system_status': monitoring_results['overall_status'],
                'alerts_generated': len(monitoring_results['performance']['alerts_generated']) +
                                   len(monitoring_results['safety']['alerts_generated']) +
                                   len(monitoring_results['costs']['alerts_generated'])
            }
        })
        
        print("\n" + "="*80)
        print("🎉 FASE 1 COMPLETADA EXITOSAMENTE")
        print("="*80)
        print(f"• Reward Mean: {phase_report['summary']['reward_statistics']['mean']:.3f}")
        print(f"• System Status: {phase_report['summary']['system_status']}")
        print(f"• Components Executed: {len(phase_results['components_executed'])}")
        print(f"• Alerts Generated: {phase_report['summary']['total_alerts']}")
        print("="*80)
        
        return phase_results
    
    def _generate_phase1_report(self, phase_results: Dict) -> Dict[str, Any]:
        """Genera reporte detallado de Fase 1"""
        report = {
            'summary': {
                'phase': 'phase1',
                'timestamp': datetime.now().isoformat(),
                'execution_time': 'N/A',
                'components_count': len(phase_results['components_executed']),
                'data_processed': phase_results['data_split']['train_size'] + 
                                phase_results['data_split']['test_size']
            },
            'reward_statistics': {},
            'monitoring_summary': {},
            'adversarial_validation_summary': {},
            'recommendations': [],
            'next_steps': []
        }
        
        if 'reward_evaluation' in phase_results['results']:
            reward_stats = phase_results['results']['reward_evaluation']['batch_statistics']
            report['reward_statistics'] = {
                'mean': reward_stats.get('mean_reward', 0),
                'std': reward_stats.get('std_reward', 0),
                'min': reward_stats.get('min_reward', 0),
                'max': reward_stats.get('max_reward', 0),
                'samples': reward_stats.get('sample_count', 0)
            }
        
        if 'monitoring' in phase_results['results']:
            monitoring = phase_results['results']['monitoring']
            report['monitoring_summary'] = {
                'system_status': monitoring.get('overall_status', 'unknown'),
                'performance_metrics_count': len(
                    monitoring.get('performance', {}).get('performance_metrics', {})
                ),
                'safety_metrics_count': len(
                    monitoring.get('safety', {}).get('safety_metrics', {})
                ),
                'total_alerts': (
                    len(monitoring.get('performance', {}).get('alerts_generated', [])) +
                    len(monitoring.get('safety', {}).get('alerts_generated', [])) +
                    len(monitoring.get('costs', {}).get('alerts_generated', []))
                )
            }
        
        if 'adversarial_validation' in phase_results['results']:
            adv = phase_results['results']['adversarial_validation']
            if 'original_validation' in adv:
                orig = adv['original_validation']
                report['adversarial_validation_summary'] = {
                    'mean_accuracy': orig.get('mean_accuracy', 0),
                    'robustness_score': orig.get('robustness_score', 0),
                    'has_reward_analysis': 'reward_analysis' in adv
                }
        
        recommendations = self._generate_recommendations_from_results(phase_results)
        report['recommendations'] = recommendations
        
        report['next_steps'] = [
            "Fase 2: Implementar Constitutional AI y verifiers",
            "Refinar Reward Model basado en resultados Fase 1",
            "Configurar alertas automáticas para métricas críticas",
            "Integrar con sistema de logging existente",
            "Planificar despliegue en staging environment"
        ]
        
        return report
    
    def _generate_recommendations_from_results(self, phase_results: Dict) -> List[str]:
        """Genera recomendaciones basadas en resultados de Fase 1"""
        recommendations = []
        
        if 'reward_evaluation' in phase_results['results']:
            reward_stats = phase_results['results']['reward_evaluation']['batch_statistics']
            mean_reward = reward_stats.get('mean_reward', 0.5)
            
            if mean_reward < 0.6:
                recommendations.append(
                    f"Low mean reward ({mean_reward:.3f}): Consider adjusting reward component weights"
                )
            
            if reward_stats.get('std_reward', 0) > 0.2:
                recommendations.append(
                    "High reward variance: Investigate inconsistent model performance"
                )
        
        if 'monitoring' in phase_results['results']:
            monitoring = phase_results['results']['monitoring']
            status = monitoring.get('overall_status', 'healthy')
            
            if status == 'critical':
                recommendations.append(
                    "Critical system status detected: Immediate investigation required"
                )
            elif status == 'degraded':
                recommendations.append(
                    "Degraded system status: Schedule maintenance and optimization"
                )
            
            perf_metrics = monitoring.get('performance', {}).get('performance_metrics', {})
            if 'accuracy' in perf_metrics and perf_metrics['accuracy'] < 0.7:
                recommendations.append(
                    f"Low accuracy ({perf_metrics['accuracy']:.3f}): Consider retraining or data augmentation"
                )
            
            safety_metrics = monitoring.get('safety', {}).get('safety_metrics', {})
            if safety_metrics.get('hacking_detections', 0) > 0:
                recommendations.append(
                    f"Reward hacking detected ({safety_metrics['hacking_detections']} times): "
                    "Strengthen reward model and add detection mechanisms"
                )
        
        if not recommendations:
            recommendations.append(
                "Phase 1 completed successfully. Proceed to Phase 2 implementation."
            )
        
        return recommendations
    
    def get_phase1_status(self) -> Dict[str, Any]:
        """Obtiene estado actual de Fase 1"""
        monitoring_summary = self.monitoring_system.get_monitoring_summary()
        reward_metrics = self.reward_model.get_reward_model_metrics()
        
        return {
            'phase': self.phase_status,
            'monitoring': monitoring_summary,
            'reward_model': reward_metrics,
            'integration_status': {
                'hooks_active': True,
                'model_integrated': hasattr(self.base_model, 'predict'),
                'validator_integrated': hasattr(self.validator, 'validate'),
                'attention_integrated': hasattr(self.attention, 'continuous_multi_lorax')
            },
            'timestamp': datetime.now().isoformat()
        }

# ============================================================================
# 4. FUNCIÓN DE INICIALIZACIÓN Y EJECUCIÓN
# ============================================================================

async def initialize_and_run_phase1(
    base_model: BaseEstimator,
    adversarial_validator: Any,
    attention_module: Any,
    fairness_module: Any,
    explain_module: Any,
    X_data: np.ndarray,
    y_data: np.ndarray,
    config: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    Función principal para inicializar y ejecutar Fase 1 completa
    """
    print("\n" + "="*80)
    print("🚀 INICIALIZANDO FASE 1 - REWARD MODEL + BASIC MONITORING")
    print("="*80)
    
    phase1 = Phase1Integration(
        base_model=base_model,
        adversarial_validator=adversarial_validator,
        attention_module=attention_module,
        fairness_module=fairness_module,
        explain_module=explain_module,
        config=config
    )
    
    initial_status = phase1.get_phase1_status()
    print(f"\nEstado inicial Fase 1:")
    print(f"• Reward Model: {initial_status['phase']['components']['reward_model']}")
    print(f"• Monitoring System: {initial_status['phase']['components']['monitoring_system']}")
    print(f"• Hooks integrados: {initial_status['phase']['components']['hooks_integrated']}")
    
    if X_data is not None and y_data is not None:
        print("\n" + "="*80)
        print("EJECUTANDO PIPELINE COMPLETO FASE 1")
        print("="*80)
        
        results = await phase1.run_phase1_pipeline(
            X=X_data,
            y=y_data,
            test_size=0.2
        )
        
        final_status = phase1.get_phase1_status()
        
        print("\n" + "="*80)
        print("RESUMEN EJECUCIÓN FASE 1")
        print("="*80)
        print(f"• Reward Mean: {results['phase_report']['summary']['reward_statistics']['mean']:.3f}")
        print(f"• System Status: {results['phase_report']['summary']['system_status']}")
        print(f"• Components Executed: {len(results['components_executed'])}")
        print(f"• Reward Model Status: {final_status['reward_model'].get('status', 'active')}")
        print("="*80)
        
        return results
    else:
        print("\n⚠️ No se proporcionaron datos para ejecutar el pipeline completo")
        print("Fase 1 inicializada pero no ejecutada")
        return {'phase1_initialized': True, 'status': initial_status}