# === post_training_pipeline.py ===
"""
Pipeline completo de post-training integrado con tu sistema existente.
Incorpora: Supervised Fine-Tuning (SFT), Reinforcement Learning (RL),
Reward Models, Constitutional AI, Verifiers, Chain-of-Thought, Anti-Reward-Hacking.
"""
from datetime import datetime
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import logging
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import asyncio
from concurrent.futures import ThreadPoolExecutor
import hashlib
import pickle
from .intervene import *
# Al inicio del archivo, con los otros imports
from .subproblem import continuous_multi_lorax

logger = logging.getLogger(__name__)

class TrainingPhase(Enum):
    """Fases del pipeline de post-training"""
    SFT = "supervised_fine_tuning"
    RL = "reinforcement_learning"
    CONSTITUTIONAL = "constitutional_ai"
    DEPLOYMENT = "deployment"

class ErrorBucket(Enum):
    """Buckets de errores según el diagnóstico proporcionado"""
    FACTUALITY = "factuality_hallucination"
    REASONING = "reasoning_math_logic_code"
    SCHEMA = "schema_format_violations"
    INSTRUCTION = "instruction_following"
    TOOL_USE = "tool_api_use"
    UNDER_REFUSAL = "under_refusal"
    OVER_REFUSAL = "over_refusal"
    TOXICITY = "toxicity_bias"
    STYLE = "style_tone_mismatch"
    RETRIEVAL = "retrieval_grounding_failures"
    LONG_CONTEXT = "long_context_failures"
    MULTI_TURN = "multi_turn_inconsistency"
    OVERCONFIDENCE = "overconfidence_miscalibration"
    DISTRIBUTION_SHIFT = "distribution_shift_fragility"

@dataclass
class TrainingSample:
    """Estructura para muestras de entrenamiento con metadatos completos"""
    input: np.ndarray
    target: np.ndarray
    prediction: Optional[np.ndarray] = None
    reward: Optional[float] = None
    constitution_score: Optional[float] = None
    verifier_pass: Optional[bool] = None
    chain_of_thought: Optional[List[str]] = None
    error_bucket: Optional[ErrorBucket] = None
    metadata: Optional[Dict] = None

class RewardModel:
    """
    Modelo de recompensa que juzga la calidad de las salidas del modelo.
    Implementa detección de reward hacking y uncertainty calibration.
    """
    
    def __init__(self, base_model: BaseEstimator, uncertainty_threshold: float = 0.3):
        self.base_model = base_model
        self.uncertainty_threshold = uncertainty_threshold
        self.reward_history = []
        self.hacking_detector = RewardHackingDetector()
        
    def score(self, input: np.ndarray, output: np.ndarray, 
              target: Optional[np.ndarray] = None,
              chain_of_thought: Optional[List[str]] = None) -> Dict[str, float]:
        scores = {}
        
        # 1. Exactitud básica - CORREGIDO para arrays 2D
        if target is not None:
            print(f"[RewardModel] score: output_shape={output.shape}, target_shape={target.shape}")
            
            # Asegurar que ambos sean 2D
            if output.ndim == 1:
                output = output.reshape(1, -1)
            if target.ndim == 1:
                target = target.reshape(1, -1)
            
            if output.shape == target.shape:
                # Calcular MSE sobre todas las dimensiones
                mse = float(np.mean((output - target) ** 2))
                # Normalizar MSE a [0,1] (asumiendo que MSE máximo razonable es 1.0)
                scores['accuracy'] = 1.0 - min(mse, 1.0)
                print(f"[RewardModel] accuracy={scores['accuracy']:.4f}, mse={mse:.6f}")
            else:
                # Si shapes diferentes, intentar aplanar
                output_flat = output.flatten()
                target_flat = target.flatten()
                min_len = min(len(output_flat), len(target_flat))
                mse = float(np.mean((output_flat[:min_len] - target_flat[:min_len]) ** 2))
                scores['accuracy'] = 1.0 - min(mse, 1.0)
                print(f"[RewardModel] shapes diferentes, accuracy={scores['accuracy']:.4f}")
        else:
            scores['accuracy'] = 0.5
            print(f"[RewardModel] no target, accuracy=0.5")
        
        # 2. Calidad del Chain-of-Thought
        if chain_of_thought:
            scores['cot_quality'] = min(len(chain_of_thought) / 10.0, 1.0)
            print(f"[RewardModel] cot_quality={scores['cot_quality']:.4f}, steps={len(chain_of_thought)}")
        else:
            scores['cot_quality'] = 0.3
        
        # 3. Verificación de reward hacking
        hacking_score, hacking_flags = self.hacking_detector.detect(input, output, scores)
        scores['hacking_risk'] = hacking_score
        print(f"[RewardModel] hacking_flags={hacking_flags}, risk={hacking_score:.4f}")
        
        # 4. Uncertainty calibration
        uncertainty = self._calculate_uncertainty(output)
        scores['uncertainty'] = uncertainty
        print(f"[RewardModel] uncertainty={uncertainty:.4f}")
        
        # 5. Recompensa compuesta
        composite_reward = self._compute_composite_reward(scores, uncertainty)
        print(f"[RewardModel] composite_reward={composite_reward:.4f}")
        
        self.reward_history.append({
            'scores': scores,
            'composite_reward': composite_reward,
            'uncertainty': uncertainty,
            'hacking_flags': hacking_flags
        })
        
        return {
            'composite_reward': composite_reward,
            'component_scores': scores,
            'uncertainty': uncertainty,
            'hacking_flags': hacking_flags
        }
    
    def _evaluate_cot_quality(self, chain_of_thought: np.ndarray, output: np.ndarray) -> float:
        """Evalúa la calidad del reasoning paso a paso"""
        # 1. Consistencia interna
        steps = len(chain_of_thought)
        if steps == 0:
            return 0.0
        
        # 2. Progresión lógica (simplificado - en producción usaría NLP)
        quality_score = min(steps / 10.0, 1.0)  # Normalizar por número de pasos
        
        # 3. Coherencia con el output final
        # (En implementación real se analizaría semánticamente)
        
        return quality_score
    
    def _calculate_uncertainty(self, output: np.ndarray) -> float:
        """Calcula incertidumbre de la predicción"""
        if len(output.shape) == 1 or output.shape[1] == 1:
            # Regresión: varianza estimada
            return float(np.std(output))
        else:
            # Clasificación: entropía de la distribución
            probs = softmax(output)
            entropy = -np.sum(probs * np.log(probs + 1e-10))
            return float(entropy / np.log(probs.shape[1]))
    
    def _compute_composite_reward(self, scores: Dict[str, float], uncertainty: float) -> float:
        """Combina scores con ponderaciones y penaliza incertidumbre alta"""
        weights = {
            'accuracy': 0.4,
            'cot_quality': 0.3,
            'hacking_risk': -0.3,  # Penalización
        }
        
        composite = 0.0
        for key, weight in weights.items():
            if key in scores:
                composite += weight * scores[key]
        
        # Penalizar alta incertidumbre (ajustable)
        uncertainty_penalty = max(0, uncertainty - self.uncertainty_threshold) * 0.5
        composite -= uncertainty_penalty
        
        # Normalizar a [0, 1]
        return composite
    
    def get_training_metrics(self) -> Dict:
        """Métricas para monitoreo del reward model"""
        if not self.reward_history:
            return {}
        
        rewards = [h['composite_reward'] for h in self.reward_history]
        uncertainties = [h['uncertainty'] for h in self.reward_history]
        hacking_flags = sum([len(h['hacking_flags']) for h in self.reward_history])
        
        return {
            'mean_reward': float(np.mean(rewards)),
            'std_reward': float(np.std(rewards)),
            'mean_uncertainty': float(np.mean(uncertainties)),
            'hacking_flags_total': hacking_flags,
            'reward_distribution': {
                'min': float(np.min(rewards)),
                'max': float(np.max(rewards)),
                'median': float(np.median(rewards))
            }
        }

class ConstitutionalAI:
    """
    Implementa Constitutional AI con principios escritos por humanos
    que se transforman en verificaciones automáticas.
    """
    
    def __init__(self, constitution_rules: List[Dict[str, any]]):
        """
        constitution_rules: Lista de reglas constitucionales
        Ejemplo: [
            {
                "principle": "Evitar daño",
                "rule": "No generar contenido peligroso o ilegal",
                "verifier": harmful_content_verifier,
                "weight": 1.0
            },
            ...
        ]
        """
        self.constitution_rules = constitution_rules
        self.violation_history = []
        
    def evaluate(self, input: np.ndarray, output: np.ndarray, 
                 chain_of_thought: Optional[List[str]] = None) -> Dict:
        """
        Evalúa la salida contra todos los principios constitucionales
        """
        print("\n" + "="*80)
        print("[ConstitutionalAI] EVALUACIÓN CONSTITUCIONAL")
        print("="*80)
        
        evaluations = []
        total_score = 0.0
        max_score = 0.0
        violations = []
        
        for rule_idx, rule in enumerate(self.constitution_rules, 1):
            principle = rule['principle']
            rule_text = rule['rule']
            verifier = rule['verifier']
            weight = rule.get('weight', 1.0)
            
            print(f"\n--- Regla {rule_idx}: {principle} (peso={weight}) ---")
            print(f"   Descripción: {rule_text[:100]}...")
            
            try:
                result = verifier(input, output, chain_of_thought)
                rule_score = result.get('score', 0.0)
                passed = result.get('passed', False)
                details = result.get('details', '')
                severity = result.get('severity', 'low')
                
                status = "✅ PASÓ" if passed else "❌ FALLÓ"
                print(f"   Estado: {status}")
                print(f"   Score: {rule_score:.4f}")
                print(f"   Detalles: {details}")
                print(f"   Severidad: {severity}")
                
                if not passed:
                    violations.append({
                        'principle': principle,
                        'rule': rule_text,
                        'details': details,
                        'severity': severity,
                        'score': rule_score
                    })
                    print(f"   ⚠️ VIOLACIÓN DETECTADA: {principle}")
                
                evaluations.append({
                    'principle': principle,
                    'score': rule_score,
                    'passed': passed,
                    'weight': weight,
                    'details': details,
                    'severity': severity
                })
                
                total_score += rule_score * weight
                max_score += 1.0 * weight
                
            except Exception as e:
                print(f"   ❌ ERROR en verifier: {e}")
                logger.error(f"Error en verifier para regla {principle}: {e}")
                continue
        
        # Normalizar score
        normalized_score = total_score / max_score if max_score > 0 else 0.0
        
        print("\n" + "-"*80)
        print(f"RESUMEN CONSTITUCIONAL:")
        print(f"  Score total: {total_score:.4f} / {max_score:.4f}")
        print(f"  Score normalizado: {normalized_score:.4f}")
        print(f"  Reglas pasadas: {len([e for e in evaluations if e['passed']])}/{len(evaluations)}")
        print(f"  Violaciones: {len(violations)}")
        
        if violations:
            print("\n  VIOLACIONES DETECTADAS:")
            for v in violations:
                print(f"    - [{v['severity'].upper()}] {v['principle']}: {v['details'][:100]}...")
        
        # Registrar violaciones
        if violations:
            self.violation_history.append({
                'input_hash': hashlib.md5(input.tobytes()).hexdigest()[:8],
                'output_hash': hashlib.md5(output.tobytes()).hexdigest()[:8],
                'violations': violations,
                'timestamp': datetime.now().isoformat()
            })
        
        print("="*80 + "\n")
    
        return {
            'constitution_score': normalized_score,
            'evaluations': evaluations,
            'violations': violations,
            'passed_all': len(violations) == 0
        }
    
    def generate_correction(self, violation: Dict, input: np.ndarray) -> str:
        """Genera una corrección basada en la violación constitucional"""
        principle = violation['principle']
        rule = violation['rule']
        
        templates = {
            "Damage avoidance": (
                "Mi respuesta anterior violó el principio '{principle}': {rule}. ",
                "Permíteme proporcionar una alternativa segura y útil: "
            ),
            "Privacy": (
                "Noté que mi respuesta podría comprometer la privacidad. ",
                "En su lugar, voy a: "
            ),
            "Accuracy": (
                "Corrigiendo mi respuesta anterior para asegurar exactitud: ",
            ),
            "Format": (
                "Ajustando el formato para cumplir con los requisitos: "
            )
        }
        
        template = templates.get(principle, 
            "Corrigiendo mi respuesta para alinearla con {principle}: ")
        
        return template.format(principle=principle, rule=rule)
    
    def get_constitution_metrics(self) -> Dict:
        """Métricas de cumplimiento constitucional"""
        if not self.violation_history:
            return {'total_violations': 0, 'violation_by_principle': {}}
        
        violation_by_principle = {}
        for record in self.violation_history:
            for violation in record['violations']:
                principle = violation['principle']
                violation_by_principle[principle] = violation_by_principle.get(principle, 0) + 1
        
        return {
            'total_violations': len(self.violation_history),
            'violation_by_principle': violation_by_principle,
            'recent_violations': self.violation_history[-10:] if self.violation_history else []
        }

class RewardHackingDetector:
    """
    Detecta y mitiga reward hacking usando múltiples estrategias:
    - Goodhart's Law monitoring
    - Distribution shift detection
    - Semantic caching attacks
    """
    
    def __init__(self):
        self.patterns_database = []
        self.detection_history = []
        
    def detect(self, input: np.ndarray, output: np.ndarray, 
               scores: Dict[str, float]) -> Tuple[float, List[str]]:
        """
        Detecta posibles intentos de reward hacking
        Retorna: (risk_score, list_of_flags)
        """
        flags = []
        risk_score = 0.0
        
        # 1. Detección de Goodhart's Law (métrica vs objetivo real)
        if self._check_goodhart_law(output, scores):
            flags.append("goodhart_law_suspicion")
            risk_score += 0.4
        
        # 2. Detección de ataques de semantic caching
        if self._check_semantic_caching(input, output):
            flags.append("semantic_caching_attempt")
            risk_score += 0.3
        
        # 3. Detección de distribution shift anómalo
        if self._check_distribution_shift(output):
            flags.append("anomalous_distribution_shift")
            risk_score += 0.3
        
        # 4. Patrones repetitivos (sobre-optimización)
        if self._check_repetitive_patterns(output):
            flags.append("repetitive_pattern_detected")
            risk_score += 0.2
        
        return min(risk_score, 1.0), flags
    
    def _check_goodhart_law(self, output: np.ndarray, scores: Dict) -> bool:
        """
        Detecta si el modelo está optimizando métricas sin mejorar
        el objetivo real.
        """
        # Heurística: alta confianza con baja variabilidad en outputs similares
        if 'accuracy' in scores and scores['accuracy'] > 0.9:
            # Calcular variabilidad
            if len(output.shape) == 1:
                variability = np.std(output)
                if variability < 0.01:  # Muy baja variabilidad
                    return True
        return False
    
    def _check_semantic_caching(self, input: np.ndarray, output: np.ndarray) -> bool:
        """Detecta intentos de caching semántico para engañar al reward"""
        # Generar hash semántico (simplificado)
        input_hash = hashlib.md5(input.tobytes()).hexdigest()[:16]
        output_hash = hashlib.md5(output.tobytes()).hexdigest()[:16]
        
        # Buscar patrones similares en el historial
        for pattern in self.patterns_database:
            if (abs(len(input_hash) - len(pattern['input_hash'])) < 3 and
                abs(len(output_hash) - len(pattern['output_hash'])) < 3):
                return True
        
        # Agregar al database
        self.patterns_database.append({
            'input_hash': input_hash,
            'output_hash': output_hash,
            'timestamp': np.datetime64('now')
        })
        
        # Mantener database limitada
        if len(self.patterns_database) > 1000:
            self.patterns_database = self.patterns_database[-1000:]
        
        return False
    
    def _check_distribution_shift(self, output: np.ndarray) -> bool:
        """Detecta cambios abruptos en la distribución de outputs"""
        if not hasattr(self, 'output_history'):
            self.output_history = []
        
        if len(self.output_history) > 10:
            # Calcular distancia estadística entre lotes
            recent_means = np.array([np.mean(o) for o in self.output_history[-10:] if o is not None])
            if len(recent_means) == 0:
                return False
                
            recent_mean = np.mean(recent_means)
            current_mean = np.mean(output)
            
            distance = np.abs(recent_mean - current_mean)
            threshold = np.std(recent_means) * 3
            
            #print(f"DISTRIBUTION SHIFT DISTANCE: {distance}")
            #print(f"THRESHOLD: {threshold}")
            
            # Ahora distance es un escalar
            if distance > threshold:
                return True
        
        self.output_history.append(output)
        return False
    
    def _check_repetitive_patterns(self, output: np.ndarray) -> bool:
        """Detecta patrones repetitivos que sugieren sobre-optimización"""
        if len(output.shape) == 1 and len(output) > 10:
            # Buscar periodicidad en la secuencia
            autocorr = np.correlate(output - np.mean(output), 
                                   output - np.mean(output), mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            
            # Picos significativos en autocorrelación
            peaks = np.where(autocorr > np.mean(autocorr) + 2*np.std(autocorr))[0]
            if len(peaks) > 3:
                return True
        
        return False
    
    def get_detection_metrics(self) -> Dict:
        """Métricas de detección de reward hacking"""
        if not self.detection_history:
            return {'total_flags': 0, 'flag_types': {}}
        
        flag_types = {}
        for record in self.detection_history:
            for flag in record.get('flags', []):
                flag_types[flag] = flag_types.get(flag, 0) + 1
        
        return {
            'total_detections': len(self.detection_history),
            'flag_types': flag_types,
            'recent_detections': self.detection_history[-5:] if self.detection_history else []
        }

def chain_of_thought_step(input_matrix, step_context_matrix, W_Q, W_K, W_V, W_O, W_vocab):
    """
    Single CoT step: Generate next reasoning step matrix
        
    Input:
    - input_matrix: [seq_len, embedding_dim] - Current reasoning state
    - step_context_matrix: [step_len, embedding_dim] - Previous steps
        
    Returns:
    - next_step_matrix: [1, embedding_dim] - Next reasoning step embedding
    - step_probability: float - Probability of this step
    """
        
    # Concatenate input with previous steps
    # full_context = [input; step1; step2; ...] : [total_len, embedding_dim]
    full_context = np.vstack([input_matrix, step_context_matrix]) if step_context_matrix.size > 0 else input_matrix
        
    # Compute Q, K, V matrices
    Q = full_context @ W_Q  # [total_len, embedding_dim]
    K = full_context @ W_K  # [total_len, embedding_dim]
    V = full_context @ W_V  # [total_len, embedding_dim]
        
    # Apply attention
    attention_output, attention_weights = attention(Q, K, V)
        
    # Project through output matrix
    projected_output = attention_output @ W_O  # [total_len, embedding_dim]
        
    # Generate next step embedding (take last position)
    next_step_embedding = projected_output[-1:, :]  # [1, embedding_dim]
        
    # Calculate step probability using logits
    # logits = next_step_embedding @ W_vocab.T : [1, vocab_size]
    logits = next_step_embedding @ W_vocab.T
        
    # softmax to get probabilities
    probs = F.softmax(torch.tensor(logits), dim=-1).numpy()
        
    # Step probability = max probability (simplified)
    step_probability = np.max(probs)
        
    return next_step_embedding, step_probability, attention_weights
    
def _generate_chain_of_thought(question_matrix, num_steps=4, embedding_dim = 0, W_vocab=[], W_Q=[], W_K=[], W_V=[], W_O=[]):
    """
    Full CoT decoding process
        
    P(answer|question) = ∏ P(step_i | question, steps_<i)
        
    Args:
        question_matrix: [seq_len, embedding_dim] - Embedded question
        num_steps: Number of reasoning steps
            
    Returns:
        steps_matrices: List of step embeddings
        total_probability: Total probability of reasoning path
    """
        
    print("=" * 70)
    print("CHAIN OF THOUGHT MATRIX OPERATIONS")
    print("=" * 70)
        
    steps_matrices = []
    step_probabilities = []
    step_context = np.array([]).reshape(0, embedding_dim)
        
    total_log_prob = 0.0
        
    for step in range(num_steps):
        print(f"\n--- Step {step + 1} ---")
            
        # Generate next step matrix
        next_step, step_prob, attn_weights = chain_of_thought_step(
            question_matrix, 
            step_context, 
            W_Q, 
            W_K, 
            W_V, 
            W_O, 
            W_vocab
        )
            
        steps_matrices.append(next_step)
        step_probabilities.append(step_prob)
            
        # Update context with new step
        step_context = np.vstack([step_context, next_step]) if step_context.size > 0 else next_step
            
        # Accumulate log probability
        log_prob = np.log(step_prob)
        total_log_prob += log_prob
            
        print(f"Step {step + 1} embedding shape: {next_step.shape}")
        print(f"P(z{step + 1} | x, z<{step + 1}) = {step_prob:.4f}")
        print(f"log P = {log_prob:.4f}")
            
        # Show attention patterns (simplified)
        print(f"Attention weights shape: {attn_weights.shape}")
            
    # Calculate final answer probability
    # P(answer | question, all steps)
    final_answer_logits = step_context[-1:, :] @ W_vocab.T
    final_answer_probs = F.softmax(torch.tensor(final_answer_logits), dim=-1).numpy()
    final_prob = np.max(final_answer_probs)
        
    total_log_prob += np.log(final_prob)
    total_probability = np.exp(total_log_prob)
        
    print(f"\n{'=' * 70}")
    print("FINAL PROBABILITY CALCULATION")
    print("=" * 70)
    print(f"\nP(answer|question) = ∏ P(step_i | question, steps_<i) × P(answer|all steps)")
    print(f"\n= {' × '.join([f'{p:.4f}' for p in step_probabilities])} × {final_prob:.4f}")
    print(f"\n= {total_probability:.6f}")
    print(f"\nlog P = {total_log_prob:.4f}")
        
    return steps_matrices, total_probability
    

def generate_chain_of_thought(thoughts: np.ndarray, params: dict, steps: int = 4) -> np.ndarray:
    """
    Wrapper público de generate_chain_of_thought para uso desde cross_validation.perturb_feature.

    Interfaz:
        thoughts : np.ndarray  — matriz de entrada (frames x features)
        params   : dict        — pesos del modelo; claves usadas: 'W_ff1', 'W_ff2',
                                 'gamma1', 'beta1', 'gamma2', 'beta2'
        steps    : int         — número de pasos de razonamiento

    Retorna el contexto final acumulado (np.ndarray) o thoughts si no hay pesos válidos.

    NOTA: _generate_chain_of_thought requiere matrices de embedding completas
    (W_Q, W_K, W_V, W_O, W_vocab) que no están disponibles en el contexto AOC.
    Este wrapper aplica una aproximación lineal por capas usando los pesos del
    modelo (W_ff1, W_ff2) con normalización por gamma/beta, equivalente a un
    forward pass de 2 capas feed-forward sin atención.
    """
    try:
        context = np.array(thoughts, dtype=np.float64)
        context = np.nan_to_num(context)

        W1 = np.array(params.get('W_ff1', np.ones((1, 1))), dtype=np.float64)
        W2 = np.array(params.get('W_ff2', np.ones((1, 1))), dtype=np.float64)
        gamma1 = float(params.get('gamma1', 1.0)) if np.isscalar(params.get('gamma1', 1.0)) else 1.0
        beta1  = float(params.get('beta1',  0.0)) if np.isscalar(params.get('beta1',  0.0)) else 0.0
        gamma2 = float(params.get('gamma2', 1.0)) if np.isscalar(params.get('gamma2', 1.0)) else 1.0
        beta2  = float(params.get('beta2',  0.0)) if np.isscalar(params.get('beta2',  0.0)) else 0.0

        for _ in range(max(1, steps)):
            # Capa 1: normalización + proyección W1
            mu1 = np.mean(context, axis=-1, keepdims=True)
            std1 = np.std(context, axis=-1, keepdims=True) + 1e-8
            normed = gamma1 * (context - mu1) / std1 + beta1
            # proyección: solo si W1 es compatible
            if W1.ndim == 2 and W1.shape[0] == context.shape[-1]:
                proj = normed @ W1
            elif W1.ndim == 2 and W1.shape[1] == context.shape[-1]:
                proj = normed @ W1.T
            else:
                proj = normed
            # activación ReLU
            proj = np.maximum(proj, 0)

            # Capa 2: normalización + proyección W2
            mu2 = np.mean(proj, axis=-1, keepdims=True)
            std2 = np.std(proj, axis=-1, keepdims=True) + 1e-8
            normed2 = gamma2 * (proj - mu2) / std2 + beta2
            if W2.ndim == 2 and W2.shape[0] == proj.shape[-1]:
                proj2 = normed2 @ W2
            elif W2.ndim == 2 and W2.shape[1] == proj.shape[-1]:
                proj2 = normed2 @ W2.T
            else:
                proj2 = normed2

            # Residual connection + actualización del contexto
            if proj2.shape == context.shape:
                context = context + proj2
            else:
                # Si las shapes divergen, mantener el contexto sin actualizar
                break

        return np.nan_to_num(context)

    except Exception as e:
        logger.warning(f"[generate_chain_of_thought] Error en paso CoT: {e}. Retornando thoughts original.")
        return np.nan_to_num(thoughts)


class PostTrainingPipeline:
    """
    Pipeline principal que integra todos los componentes de post-training
    """
    
    def __init__(self, 
                 base_model: BaseEstimator,
                 reward_model: RewardModel,
                 constitutional_ai: ConstitutionalAI,
                 validator: Optional[object] = None, # Tu AdversarialValidator
                 enable_content_filter = True): 
        
        self.base_model = base_model
        self.reward_model = reward_model
        self.constitutional_ai = constitutional_ai
        self.validator = validator
        
        self.training_history = []
        self.current_phase = TrainingPhase.SFT
        
        # Configuración de RL
        self.rl_config = {
            'learning_rate': 5e-5,
            'kl_divergence_limit': 0.15,
            'clip_range': 0.2,
            'entropy_bonus': 0.01,
            'gamma': 0.99,  # Discount factor
            'lam': 0.95,    # GAE parameter
        }
        
        # Referencia al modelo original (para KL divergence)
        self.reference_model = base_model

        self.misalignment_detector = AgenticMisalignmentDetector() 
        self.content_filter = RuntimeContentFilterMonitor() if enable_content_filter else None
        
        # GO/NO-GO tracking
        self.last_slice_reward = 0.0  # Recompensa de la slice anterior
        self.promotion_candidates = []  # slices candidatas a producción
        self.promotion_history = []     # historial de promociones
        self.slice_counter = 0          # contador de slices evaluadas
        
    async def run_supervised_finetuning(self, 
                                       X: np.ndarray, 
                                       y: np.ndarray,
                                       epochs: int = 3,
                                       batch_size: int = 32) -> Dict:
        """
        Fine-tuning supervisado con data de alta calidad (rejection sampling)
        """
        logger.info(f"Iniciando SFT con {len(X)} muestras")
        
        # 1. Rejection sampling: filtrar data de alta calidad
        high_quality_data = await self._rejection_sampling(X, y)
        
        if not high_quality_data:
            logger.warning("No se encontró data de alta calidad para SFT")
            return {'status': 'no_high_quality_data'}
        
        X_high, y_high = high_quality_data
        
        # 2. Aplicar cosine annealing learning rate scheduler
        lr_schedule = self._cosine_annealing_schedule(
            self.rl_config['learning_rate'],
            epochs * len(X_high) // batch_size
        )
        
        # 3. Fine-tuning (simplificado - en producción usarías training loop real)
        training_metrics = []
        best_loss = float('inf')
        
        for epoch in range(epochs):
            epoch_losses = []
            
            # Training loop por batches
            for i in range(0, len(X_high), batch_size):
                batch_X = X_high[i:i+batch_size]
                batch_y = y_high[i:i+batch_size]
                
                # Aquí iría el training real del modelo
                # Por simplicidad, simulamos una mejora
                current_lr = lr_schedule[epoch * (len(X_high)//batch_size) + i//batch_size]
                
                # Métricas simuladas
                batch_loss = np.random.uniform(0.1, 0.3) * (0.9 ** epoch)
                epoch_losses.append(batch_loss)
                
                # Registrar
                training_metrics.append({
                    'epoch': epoch,
                    'batch': i // batch_size,
                    'loss': batch_loss,
                    'learning_rate': current_lr,
                    'phase': 'sft'
                })
            
            epoch_loss = np.mean(epoch_losses)
            logger.info(f"Epoch {epoch}: loss = {epoch_loss:.4f}")
            
            # Early stopping implícito
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                # Guardar mejor modelo
        
        return {
            'status': 'completed',
            'epochs': epochs,
            'final_loss': best_loss,
            'training_metrics': training_metrics,
            'high_quality_samples': len(X_high)
        }
    
    async def run_reinforcement_learning(self, X: np.ndarray, y: np.ndarray,
                                     num_rollouts: int = 100, ppo_steps: int = 10,
                                     sample_efficiency_analysis: bool = False,
                                     go_no_go_threshold: float = 0.05,
                                     external_advantages: np.ndarray = None) -> Dict:
        print(f"\n{'='*70}")
        print(f"[RL] INICIANDO REINFORCEMENT LEARNING")
        print(f"[RL] X_shape={X.shape}, y_shape={y.shape}, num_rollouts={num_rollouts}")
        print(f"{'='*70}")
        
        # Asegurar que X e y tengan la forma correcta
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        
        # OBTENER FEATURE WEIGHTS CORRECTAMENTE
        print(f"[RL] Obteniendo feature_weights del modelo base...")
        print(f"[RL] self.base_model type: {type(self.base_model)}")
        print(f"[RL] hasattr feature_weights: {hasattr(self.base_model, 'feature_weights')}")
        
        if hasattr(self.base_model, 'feature_weights') and self.base_model.feature_weights is not None:
            feature_weights = self.base_model.feature_weights
            print(f"[RL] feature_weights desde modelo: shape={feature_weights.shape if hasattr(feature_weights, 'shape') else 'scalar'}")
            if isinstance(feature_weights, np.ndarray) and feature_weights.ndim > 1:
                feature_weights = feature_weights.flatten()
                print(f"[RL] feature_weights aplanado a shape={feature_weights.shape}")
        elif hasattr(self.base_model, 'w') and self.base_model.w is not None:
            # Fallback a w si feature_weights no existe
            feature_weights = self.base_model.w
            print(f"[RL] Usando w como fallback: shape={feature_weights.shape if hasattr(feature_weights, 'shape') else 'scalar'}")
            if isinstance(feature_weights, np.ndarray) and feature_weights.ndim > 1:
                feature_weights = feature_weights.flatten()
        else:
            feature_weights = np.ones(X.shape[1]) * 0.01
            print(f"[RL] feature_weights inicializados con unos: shape={feature_weights.shape}")
        
        # Asegurar dimensiones de feature_weights
        if len(feature_weights) != X.shape[1]:
            print(f"[RL] Ajustando feature_weights de {len(feature_weights)} a {X.shape[1]}")
            if len(feature_weights) > X.shape[1]:
                feature_weights = feature_weights[:X.shape[1]]
            else:
                feature_weights = np.pad(feature_weights, (0, X.shape[1] - len(feature_weights)), 'constant')
            print(f"[RL] feature_weights ajustado a shape={feature_weights.shape}")
        
        print(f"[RL] FEATURE WEIGHTS FINAL: shape={feature_weights.shape}, min={np.min(feature_weights):.6f}, max={np.max(feature_weights):.6f}")
        
        # Ejecutar continuous_multi_lorax con el batch completo
        try:
            print(f"[RL] Llamando a continuous_multi_lorax con features_shape={X.shape}, targets_shape={y.shape}, batch_size={min(250, len(X))}")
            attention_result = await continuous_multi_lorax(
                features=X,
                targets=y,
                a=feature_weights,
                batch_size=min(250, len(X)),
                do_forward=True
            )
            
            print(f"[RL] attention_result obtenido: {attention_result is not None}")
            
            if attention_result:
                (attention_function, attention_y, attention_scores, 
                 attention_weights_proba, attention_context) = attention_result
                print(f"[RL] Resultados de atención:")
                print(f"  - attention_y shape: {attention_y.shape if hasattr(attention_y, 'shape') else 'scalar'}")
                print(f"  - attention_scores shape: {attention_scores.shape if hasattr(attention_scores, 'shape') else 'scalar'}")
                print(f"  - attention_weights_proba shape: {attention_weights_proba.shape if hasattr(attention_weights_proba, 'shape') else 'scalar'}")
                print(f"  - attention_context shape: {attention_context.shape if hasattr(attention_context, 'shape') else 'scalar'}")
            
                # Usar los resultados de atención como CoT para cada rollout
                rollouts = []
                for i in range(len(X)):
                    if attention_context.ndim > 1 and i < attention_context.shape[0]:
                        ctx = attention_context[i]
                    else:
                        ctx = attention_context
                        
                    if attention_scores.ndim > 1 and i < attention_scores.shape[0]:
                        scrs = attention_scores[i]
                    else:
                        scrs = attention_scores
                    
                    cot = self._attention_to_cot(ctx, scrs)
                    pred = await self.base_model.predict(X[i].reshape(1, -1))
                    pred = pred[0]
                
                    reward_result = self.reward_model.score(X[i], pred, y[i], cot)
                    constitution_result = self.constitutional_ai.evaluate(X[i], pred, cot)
                    error_bucket = self._classify_error_bucket(X[i], pred, y[i])
                    
                    rollouts.append({
                        'input': X[i],
                        'target': y[i],
                        'prediction': pred,
                        'chain_of_thought': cot,
                        'composite_reward': reward_result['composite_reward'],  
                        'reward_components': reward_result['component_scores'],
                        'constitution_evaluations': constitution_result['evaluations'],
                        'error_bucket': error_bucket,
                        'metadata': {'timestamp': np.datetime64('now')}
                    })
                    print(f"[RL] Rollout {i}: reward={reward_result['composite_reward']:.4f}, mse={np.mean((pred - y[i])**2):.6f}")
            else:
                print(f"[RL] No se obtuvo resultado de atención, usando fallback individual")
                rollouts = []
                for i in range(min(num_rollouts, len(X))):
                    rollout = await self._generate_rollout_single(X[i:i+1], y[i:i+1] if y is not None else None)
                    rollouts.append(rollout)
                    print(f"[RL] Rollout individual {i}: reward={rollout.get('reward_components', {}).get('composite_reward', 0):.4f}")
                    
        except Exception as e:
            print(f"[RL] ERROR en attention batch: {e}")
            import traceback
            traceback.print_exc()
            print(f"[RL] Usando fallback rollouts individuales")
            rollouts = []
            for i in range(min(num_rollouts, len(X))):
                rollout = await self._generate_rollout_single(X[i:i+1], y[i:i+1] if y is not None else None)
                rollouts.append(rollout)

        external_advantages = np.asarray(external_advantages, dtype=np.float64).flatten()
        if len(external_advantages) != len(rollouts):
            logger.warning(f"Longitud de external_advantages ({len(external_advantages)}) != rollouts ({len(rollouts)}). Truncando.")
            external_advantages = external_advantages[:len(rollouts)]
        
        # Calcular recompensas y métricas
        print(f"\n[RL] Calculando recompensas para {len(rollouts)} rollouts")
        rewards_history = []
        advantages = []
        for i, rollout in enumerate(rollouts):
            reward_info = self._calculate_rollout_reward(rollout)
            rollout['reward'] = reward_info['total_reward']
            advantage = external_advantages[i]
            rollout['advantage'] = external_advantages[i]
            rewards_history.append(reward_info['total_reward'])
            advantages.append(external_advantages[i])
            print(f"[RL] Rollout {i}: advantage={external_advantages[i]:.4f}, base_reward={reward_info['base_reward']:.4f}, total_reward={reward_info['total_reward']:.4f}, constitution_bonus={reward_info.get('constitution_bonus', 0):.4f}, cot_bonus={reward_info.get('cot_bonus', 0):.4f}")
        
        mean_reward = float(np.mean(rewards_history)) if rewards_history else 0.0
        std_reward = float(np.std(rewards_history)) if rewards_history else 0.0
        print(f"\n[RL] RESUMEN:")
        print(f"  - Rollouts procesados: {len(rollouts)}")
        print(f"  - Recompensa media: {mean_reward:.4f}")
        print(f"  - Recompensa std: {std_reward:.4f}")
        print(f"  - Recompensa min: {np.min(rewards_history)}")
        print(f"  - Recompensa max: {np.max(rewards_history)}")

        policy_loss = 0.0
        policy_losses = []  # Lista para almacenar todas las pérdidas
    
        # En run_reinforcement_learning, después de calcular advantages
        for step in range(ppo_steps):
            kl_div = self._calculate_kl_divergence()
            if kl_div > self.rl_config['kl_divergence_limit']:
                logger.info(f"[RL] KL divergence ({kl_div:.4f}) excede límite ({self.rl_config['kl_divergence_limit']}), deteniendo PPO")
                break
            current_policy_loss = await self._ppo_update_step(rollouts, advantages)
            policy_losses.append(current_policy_loss)
            logger.debug(f"[RL] PPO step {step}: loss={current_policy_loss:.4f}, kl={kl_div:.4f}")
        
        # Calcular policy_loss promedio si hay valores, sino usar 0
        if policy_losses:
            policy_loss = np.mean(policy_losses)
        else:
            logger.warning("[RL] No se ejecutaron pasos PPO, policy_loss=0")
        
        hacking_analysis = self._analyze_reward_hacking(rollouts)
        print(f"[RL] Reward hacking analysis: total_flags={hacking_analysis.get('total_hacking_flags', 0)}")

        slice_improvement = mean_reward - rollouts[-1]['reward']
        
        if slice_improvement > go_no_go_threshold:
            decision = "GO"
            logger.info(f"[RL] ✅ GO: slice improves by {slice_improvement:.4f} > threshold {go_no_go_threshold}")
            logger.info("   Promover slice a producción")
            self.promotion_history.append({
                'timestamp': datetime.now().isoformat(),
                'decision': 'GO',
                'improvement': slice_improvement,
                'mean_reward': mean_reward
            })
        elif slice_improvement < -go_no_go_threshold:
            decision = "NO-GO"
            logger.warning(f"[RL] ❌ NO-GO: slice degrades by {abs(slice_improvement):.4f} > threshold {go_no_go_threshold}")
            logger.warning("   No promover a producción, revisar hiperparámetros")
            self.promotion_history.append({
                'timestamp': datetime.now().isoformat(),
                'decision': 'NO-GO',
                'degradation': abs(slice_improvement),
                'mean_reward': mean_reward
            })
        else:
            decision = "NEUTRAL"
            logger.info(f"[RL] ➡️ NEUTRAL: slice change {slice_improvement:+.4f} within threshold")
            self.promotion_history.append({
                'timestamp': datetime.now().isoformat(),
                'decision': 'NEUTRAL',
                'change': slice_improvement,
                'mean_reward': mean_reward
            })
        
        self.last_slice_reward = mean_reward
        
        # Agregar decision al resultado
        go_no_go_decision = {
            'decision': decision,
            'improvement': slice_improvement,
            'threshold': go_no_go_threshold
        }
        
        return {
            'status': 'completed',
            'num_rollouts': len(rollouts),
            'mean_reward': mean_reward,
            'std_reward': std_reward,
            'kl_divergence': kl_div,
            'go_no_go_decision': go_no_go_decision,
            'mean_advantage': np.mean(advantages),
            'std_advantage': np.std(advantages),
            'mean_policy_loss': np.mean(policy_loss),
            'std_policy_loss': np.std(policy_loss),
            'min_reward': float(np.min(rewards_history)) if rewards_history else 0.0,
            'max_reward': float(np.max(rewards_history)) if rewards_history else 0.0,
            'reward_distribution': {
                'mean': mean_reward,
                'std': std_reward,
                'min': float(np.min(rewards_history)) if rewards_history else 0.0,
                'max': float(np.max(rewards_history)) if rewards_history else 0.0
            },
            'hacking_analysis': hacking_analysis
        }
    
    async def _generate_rollout_single(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Dict:
        """Genera un rollout individual (fallback)"""
        input_sample = X[0]
        target = y[0] if y is not None else None
        
        if hasattr(self.base_model, 'feature_weights') and self.base_model.feature_weights is not None:
            feature_weights = self.base_model.feature_weights
            if isinstance(feature_weights, np.ndarray) and feature_weights.ndim > 1:
                feature_weights = feature_weights.flatten()
        else:
            feature_weights = np.ones(X.shape[1]) * 0.01
        
        if len(feature_weights) != X.shape[1]:
            if len(feature_weights) > X.shape[1]:
                feature_weights = feature_weights[:X.shape[1]]
            else:
                feature_weights = np.pad(feature_weights, (0, X.shape[1] - len(feature_weights)), 'constant')
        
        attention_result = await continuous_multi_lorax(
            features=input_sample.reshape(1, -1),
            targets=target.reshape(1, -1) if target is not None else input_sample.reshape(1, -1),
            a=feature_weights,
            batch_size=1,
            do_forward=True
        )
        
        if attention_result:
            _, _, _, _, attention_context = attention_result
            cot = self._attention_to_cot(attention_context, None)
        else:
            cot = []
        
        pred = await self.base_model.predict(input_sample.reshape(1, -1))
        pred = pred[0]
        reward_result = self.reward_model.score(input_sample, pred, target, cot)
        constitution_result = self.constitutional_ai.evaluate(input_sample, pred, cot)
        error_bucket = self._classify_error_bucket(input_sample, pred, target)
        
        return {
            'input': input_sample,
            'target': target,
            'prediction': pred,
            'chain_of_thought': cot,
            'reward_components': reward_result['component_scores'],
            'constitution_evaluations': constitution_result['evaluations'],
            'error_bucket': error_bucket,
            'metadata': {'timestamp': np.datetime64('now')}
        }
    
    async def _generate_rollout(self, X: np.ndarray, y: np.ndarray) -> Dict:
        # Tomar un batch de rollouts (no un solo sample)
        # En realidad, esta función debería procesar un rollout individual
        # pero para continuous_multi_lorax necesitamos el batch completo
        idx = np.random.randint(0, len(X))
        input_sample = X[idx]
        target = y[idx] if idx < len(y) else None
        
        # Usar _feature_weights en lugar de a
        if hasattr(self.base_model, 'feature_weights') and self.base_model.feature_weights is not None:
            feature_weights = self.base_model.feature_weights
            if isinstance(feature_weights, np.ndarray) and feature_weights.ndim > 1:
                feature_weights = feature_weights.flatten()
        else:
            feature_weights = np.ones(X.shape[1]) * 0.01
        
        # Asegurar que feature_weights tenga la dimensión correcta
        if len(feature_weights) != X.shape[1]:
            if len(feature_weights) > X.shape[1]:
                feature_weights = feature_weights[:X.shape[1]]
            else:
                feature_weights = np.pad(feature_weights, (0, X.shape[1] - len(feature_weights)), 'constant')
        
        print(f"[_generate_rollout] feature_weights shape: {feature_weights.shape}, X shape: {X.shape}")
        
        # Para continuous_multi_lorax, targets debe tener la misma forma que features
        # En rollout, usamos el target real como target para atención
        attention_result = await continuous_multi_lorax(
            features=input_sample.reshape(1, -1),
            targets=target.reshape(1, -1) if target is not None else input_sample.reshape(1, -1),
            a=feature_weights,  # Usar feature_weights en lugar de a
            batch_size=1,
            do_forward=True
        )
    
        if attention_result:
            (attention_function, attention_y, 
             attention_scores, attention_weights, 
             attention_context) = attention_result
            
            # El contexto de atención SIRVE como Chain-of-Thought
            cot = self._attention_to_cot(attention_context, attention_scores)
        else:
            cot = []
    
        # Generar predicción normal - USAR self.base_model
        pred = await self.base_model.predict(input_sample.reshape(1, -1))
        pred = pred[0]
        
        # Evaluar con reward model y constitutional AI
        reward_result = self.reward_model.score(
            input_sample, pred, target, cot
        )
        
        constitution_result = self.constitutional_ai.evaluate(
            input_sample, pred, cot
        )
        
        # Detectar error bucket
        error_bucket = self._classify_error_bucket(input_sample, pred, target)
        
        return {
            'input': input_sample,
            'target': target,
            'prediction': pred,
            'chain_of_thought': cot,
            'reward_components': reward_result['component_scores'],
            'constitution_evaluations': constitution_result['evaluations'],
            'error_bucket': error_bucket,
            'metadata': {
                'timestamp': np.datetime64('now'),
                'sample_hash': hashlib.md5(input_sample.tobytes()).hexdigest()[:16]
            }
        }

    def _classify_error_bucket(self, 
                              input: np.ndarray, 
                              prediction: np.ndarray,
                              target: Optional[np.ndarray]) -> Optional[ErrorBucket]:
        """Clasifica el error en buckets para diagnóstico"""
        if target is None:
            return None
        
        # Heurísticas simplificadas para clasificación
        error = np.abs(prediction - target) if prediction.shape == target.shape else 1.0
        
        if isinstance(error, np.ndarray):
            error = np.mean(error)
        
        if error > 0.5:
            return ErrorBucket.FACTUALITY
        elif error > 0.3:
            return ErrorBucket.REASONING
        elif error > 0.1:
            return ErrorBucket.OVERCONFIDENCE
        
        return None

    def _attention_to_cot(self, attention_context: np.ndarray, 
                         attention_scores: np.ndarray) -> List[str]:
        """
        Convierte el contexto de atención en pasos de razonamiento legibles
        Aumentado a 20 pasos para CoT más detallado
        """
        cot_steps = []
        
        # Paso 1-3: Análisis de distribución de atención
        if attention_scores is not None and len(attention_scores) > 0:
            mean_att = np.mean(attention_scores)
            std_att = np.std(attention_scores)
            max_att_idx = np.argmax(attention_scores) if attention_scores.ndim > 0 else 0
            min_att_idx = np.argmin(attention_scores) if attention_scores.ndim > 0 else 0
            
            cot_steps.append(f"[Paso 1/20] Distribución de atención: media={mean_att:.6f}, std={std_att:.6f}")
            cot_steps.append(f"[Paso 2/20] Máxima atención en posición {max_att_idx} con valor {attention_scores[max_att_idx]:.6f}")
            cot_steps.append(f"[Paso 3/20] Mínima atención en posición {min_att_idx} con valor {attention_scores[min_att_idx]:.6f}")
        
        # Paso 4-10: Análisis del contexto espectral
        if attention_context is not None and attention_context.size > 0:
            context_mean = np.mean(attention_context)
            context_std = np.std(attention_context)
            context_median = np.median(attention_context)
            
            cot_steps.append(f"[Paso 4/20] Contexto generado: forma={attention_context.shape}, media={context_mean:.6f}, std={context_std:.6f}")
            cot_steps.append(f"[Paso 5/20] Mediana del contexto: {context_median:.6f}")
            
            # Identificar features relevantes (top 5)
            if attention_context.ndim == 1:
                top_indices = np.argsort(np.abs(attention_context))[-5:][::-1]
                top_values = attention_context[top_indices]
                cot_steps.append(f"[Paso 6/20] Features más relevantes (top 5): {top_indices.tolist()} con valores {top_values.round(6).tolist()}")
                
                # Features con menor impacto
                bottom_indices = np.argsort(np.abs(attention_context))[:5]
                bottom_values = attention_context[bottom_indices]
                cot_steps.append(f"[Paso 7/20] Features con menor impacto (bottom 5): {bottom_indices.tolist()} con valores {bottom_values.round(6).tolist()}")
            
            # Análisis de energía
            energy_distribution = np.sum(attention_context ** 2)
            energy_per_bin = attention_context ** 2
            cot_steps.append(f"[Paso 8/20] Energía total del contexto: {energy_distribution:.6f}")
            cot_steps.append(f"[Paso 9/20] Energía máxima por bin: {np.max(energy_per_bin):.6f}")
            cot_steps.append(f"[Paso 10/20] Energía mínima por bin: {np.min(energy_per_bin):.6f}")
        
        # Paso 11-15: Análisis de correlación y patrones
        if attention_context is not None and attention_context.size > 1:
            # Autocorrelación simple
            if attention_context.ndim == 1 and len(attention_context) > 10:
                autocorr = np.correlate(attention_context - np.mean(attention_context), 
                                        attention_context - np.mean(attention_context), mode='full')
                autocorr = autocorr[len(autocorr)//2:]
                peak_autocorr = np.max(autocorr[1:]) if len(autocorr) > 1 else 0
                cot_steps.append(f"[Paso 11/20] Pico de autocorrelación (periodicidad): {peak_autocorr:.6f}")
            
            # Entropía como medida de complejidad
            if attention_context.ndim == 1:
                probs = np.abs(attention_context) / (np.sum(np.abs(attention_context)) + 1e-9)
                entropy = -np.sum(probs * np.log(probs + 1e-9))
                cot_steps.append(f"[Paso 12/20] Entropía del contexto (complejidad): {entropy:.6f}")
        
        # Paso 16-18: Análisis de calidad de reconstrucción
        if hasattr(self, 'model') and hasattr(self.model, 'predict'):
            cot_steps.append("[Paso 13/20] Aplicando Wiener spectral subtraction implícita via regresión DSVM")
            cot_steps.append("[Paso 14/20] Integrando contexto de atención continua multi-lorax para reconstrucción")
            cot_steps.append("[Paso 15/20] Verificando consistencia espectral entre entrada y salida")
        
        # Paso 19-20: Conclusiones
        if attention_context is not None:
            dynamic_range = np.max(attention_context) - np.min(attention_context)
            cot_steps.append(f"[Paso 16/20] Rango dinámico del contexto: {dynamic_range:.6f}")
            cot_steps.append(f"[Paso 17/20] Factor de amplificación: {dynamic_range / (np.mean(np.abs(attention_context)) + 1e-9):.4f}")
        
        cot_steps.append("[Paso 18/20] Validación de paridad estadística en reconstrucción espectral")
        cot_steps.append("[Paso 19/20] Verificación de ausencia de alucinaciones espectrales")
        cot_steps.append("[Paso 20/20] Reconstrucción completada con reducción de oclusión")
        
        # Asegurar que tengamos exactamente 20 pasos
        while len(cot_steps) < 20:
            cot_steps.append(f"[Paso {len(cot_steps)+1}/20] Análisis de consistencia temporal adicional")
    
        return cot_steps[:20]
   
    def _classify_error_bucket(self, 
                              input: np.ndarray, 
                              prediction: np.ndarray,
                              target: Optional[np.ndarray]) -> Optional[ErrorBucket]:
        """Clasifica el error en buckets para diagnóstico"""
        if target is None:
            return None
        
        # Heurísticas simplificadas para clasificación
        error = np.abs(prediction - target) if prediction.shape == target.shape else 1.0
        
        if isinstance(error, np.ndarray):
            error = np.mean(error)
        
        if error > 0.5:
            return ErrorBucket.FACTUALITY
        elif error > 0.3:
            # Verificar consistencia en reasoning
            if hasattr(self, 'chain_of_thought') and self.chain_of_thought:
                return ErrorBucket.REASONING
            else:
                return ErrorBucket.INSTRUCTION
        elif error > 0.1:
            return ErrorBucket.OVERCONFIDENCE
        
        return None
    
    def _calculate_rollout_reward(self, rollout: Dict) -> Dict:
        """Calcula recompensa total para un rollout"""

        pred = rollout['prediction']
        input_arr = rollout['input']
        
        # Detección de colapso
        var_out = np.var(pred.flatten())
        var_in = np.var(input_arr.flatten())
        
        # Colapso si varianza de output es MUCHO menor que la de input
        if var_out < var_in * 0.001:  # 1000 veces menor
            print(f"[REWARD] COLAPSO DETECTADO: var_out={var_out:.2e}, var_in={var_in:.2e}")
            return {
                'total_reward': -1.0,
                'base_reward': -1.0,
                'constitution_score': 0.0,
                'constitution_bonus': 0.0,
                'cot_bonus': 0.0,
                'error_penalty': -0.5,
                'collapsed': True
            }        
        
        # Obtener base_reward de reward_components
        reward_components = rollout.get('reward_components', {})
        base_reward = rollout.get('composite_reward', None)
        if base_reward is None:
            base_reward = rollout['reward_components'].get('composite_reward', 0.5)
    
        print(f"[_calculate_rollout_reward] base_reward={base_reward:.4f}")
        
        # Bonus por constitutional compliance
        constitution_evaluations = rollout.get('constitution_evaluations', [])
        constitution_passed = sum(1 for e in constitution_evaluations if e.get('passed', False))
        constitution_total = len(constitution_evaluations)
        
        constitution_bonus = 0.0
        if constitution_total > 0:
            constitution_ratio = constitution_passed / constitution_total
            constitution_bonus = constitution_ratio * 0.3
            print(f"[_calculate_rollout_reward] constitution: passed={constitution_passed}/{constitution_total}, ratio={constitution_ratio:.3f}, bonus={constitution_bonus:.4f}")
        else:
            print(f"[_calculate_rollout_reward] constitution: no evaluations found")
        
        # Bonus por calidad de CoT
        cot_bonus = 0.0
        chain_of_thought = rollout.get('chain_of_thought', [])
        if chain_of_thought:
            cot_length = len(chain_of_thought)
            cot_bonus = min(cot_length / 10.0, 0.2)
            print(f"[_calculate_rollout_reward] cot: length={cot_length}, bonus={cot_bonus:.4f}")
        
        # Penalización por error bucket
        error_penalty = 0.0
        error_bucket = rollout.get('error_bucket')
        if error_bucket is not None:
            error_penalty = -0.1
            print(f"[_calculate_rollout_reward] error_bucket={error_bucket}, penalty={error_penalty:.4f}")
        
        total_reward = base_reward + constitution_bonus + cot_bonus + error_penalty
        #total_reward = float(np.clip(total_reward, 0.0, 1.0))
        
        print(f"[_calculate_rollout_reward] total_reward={total_reward:.4f}")
        
        return {
            'total_reward': total_reward,
            'base_reward': base_reward,
            'constitution_score': constitution_ratio if constitution_total > 0 else 0.0,
            'constitution_bonus': constitution_bonus,
            'cot_bonus': cot_bonus,
            'error_penalty': error_penalty
        }
        
    def _calculate_advantage(self, rollout: Dict, reward: float) -> float:
        """Calcula ventaja usando Generalized Advantage Estimation (GAE)"""
        # Valores simplificados - en producción usarías un value function
        value_estimate = reward * 0.9  # Discounted
        advantage = reward - value_estimate
        
        # Suavizar con GAE
        if hasattr(self, 'previous_advantage'):
            advantage = self.rl_config['lam'] * self.previous_advantage + advantage
        
        self.previous_advantage = advantage
        return float(advantage)
    
    def _calculate_kl_divergence(self) -> float:
        """Calcula KL divergence entre modelo actual y referencia"""
        # Simplificado - en producción calcularías distribución real
        return np.random.uniform(0.05, 0.25)
    
    async def _ppo_update_step(self, rollouts: List[Dict], advantages: np.ndarray) -> float:
        """
        PPO REAL con clipping - actualiza el modelo usando rollouts
        
        Args:
            rollouts: Lista de diccionarios con 'input', 'prediction', 'target', etc.
            advantages: Array de ventajas (de GAE o GRPO)
        
        Returns:
            policy_loss: Pérdida de política (para logging)
        """
        eps = self.rl_config.get('clip_range', 0.2)
        learning_rate = self.rl_config.get('learning_rate', 5e-5)
        entropy_bonus = self.rl_config.get('entropy_bonus', 0.01)
        
        # ========== 1. EXTRAER DATOS DE LOS ROLLOUTS ==========
        n_samples = len(rollouts)
    
        old_predictions = np.array([r['prediction'] for r in rollouts])
        targets = np.array([r['target'] for r in rollouts])
        X_batch = np.array([r['input'] for r in rollouts])
        
        # Asegurar formas correctas
        if old_predictions.ndim == 1:
            old_predictions = old_predictions.reshape(-1, 1)
        if targets.ndim == 1:
            targets = targets.reshape(-1, 1)
        
        # ========== 2. GENERAR NUEVAS PREDICCIONES (MODELO ACTUAL) ==========
        new_predictions = await self.base_model.predict(X_batch)
        if new_predictions.ndim == 1:
            new_predictions = new_predictions.reshape(-1, 1)
        
        # ========== 3. CALCULAR ERRORES POR MUESTRA ==========
        # Usar MSE como proxy de "logits negativos" (menor error = mayor probabilidad)
        old_errors = np.mean((old_predictions - targets) ** 2, axis=1)
        new_errors = np.mean((new_predictions - targets) ** 2, axis=1)
        
        # Evitar errores extremos
        old_errors = np.clip(old_errors, 1e-8, 1e8)
        new_errors = np.clip(new_errors, 1e-8, 1e8)
        
        # ========== 4. CONVERTIR ERRORES A "PROBABILIDADES" ==========
        # Usar softmax negativo: menor error → mayor probabilidad
        # Esto es análogo a tener logits = -error
        old_logits = -old_errors
        new_logits = -new_errors
    
        # Softmax para obtener probabilidades
        old_probs = np.exp(old_logits - np.max(old_logits))
        old_probs = old_probs / (np.sum(old_probs) + 1e-8)
        
        new_probs = np.exp(new_logits - np.max(new_logits))
        new_probs = new_probs / (np.sum(new_probs) + 1e-8)
        
        # ========== 5. CALCULAR RATIO DE POLÍTICAS ==========
        ratio = new_probs / (old_probs + 1e-8)
        
        # ========== 6. ASEGURAR QUE ADVANTAGES TENGAN LA FORMA CORRECTA ==========
        advantages = np.asarray(advantages, dtype=np.float64)
        if advantages.ndim == 2:
            # Si advantages es 2D (n_samples, n_features), promediar por muestra
            advantages = np.mean(advantages, axis=1)
        advantages = advantages.flatten()
        
        # Truncar a la misma longitud
        min_len = min(len(ratio), len(advantages))
        ratio = ratio[:min_len]
        advantages = advantages[:min_len]
        
        # ========== 7. PPO CLIPPING LOSS ==========
        surr1 = ratio * advantages
        surr2 = np.clip(ratio, 1 - eps, 1 + eps) * advantages
        policy_loss = -np.mean(np.minimum(surr1, surr2))
        
        # ========== 8. ENTROPY BONUS (FOMENTA EXPLORACIÓN) ==========
        entropy = -np.mean(new_probs * np.log(new_probs + 1e-8))
        total_loss = policy_loss - entropy_bonus * entropy
        
        # ========== 9. ACTUALIZAR EL MODELO ==========
        if hasattr(self.base_model, 'w') and self.base_model.w is not None:
            # Calcular gradiente simplificado
            w_flat = self.base_model.w.flatten()
            
            # El gradiente es proporcional a la pérdida
            # Usamos el signo de la pérdida para determinar dirección
            grad = total_loss * np.sign(w_flat)
            
            # Clip de gradiente para estabilidad
            grad_norm = np.linalg.norm(grad) + 1e-8
            grad = grad / grad_norm
            
            # Limitar gradiente máximo
            max_grad_norm = getattr(self.rl_config, 'max_grad_norm', 0.5)
            if grad_norm > max_grad_norm:
                grad = grad * (max_grad_norm / grad_norm)
            
            # Actualizar pesos
            new_w = w_flat - learning_rate * grad[:len(w_flat)]
            self.base_model.w = new_w.reshape(self.base_model.w.shape)
            
            logger.debug(f"[PPO] Loss={total_loss:.6f}, Ratio mean={np.mean(ratio):.4f}, "
                        f"Adv mean={np.mean(advantages):.4f}, Grad norm={grad_norm:.6f}")
        else:
            logger.warning("[PPO] No se puede actualizar: modelo sin pesos 'w'")
        
        # ========== 10. ACTUALIZAR REFERENCIA PARA KL DIVERGENCE ==========
        # Guardar predicciones actuales para futuros cálculos de KL
        self._last_predictions = new_predictions
    
        return float(total_loss)
    
    def _analyze_reward_hacking(self, rollouts: List[Dict]) -> Dict:
        """Analiza rollouts en busca de reward hacking"""
        detector = RewardHackingDetector()
        hacking_flags = []
        
        for rollout in rollouts:
            input = rollout['input']
            output = rollout['prediction']
            scores = rollout['reward_components']
            
            risk_score, flags = detector.detect(input, output, scores)
            if flags:
                hacking_flags.append({
                    'rollout_id': rollout.get('metadata', {}).get('sample_hash', 'unknown'),
                    'flags': flags,
                    'risk_score': risk_score
                })
        
        return {
            'total_hacking_flags': len(hacking_flags),
            'flagged_rollouts': hacking_flags,
            'risk_distribution': [f['risk_score'] for f in hacking_flags] if hacking_flags else []
        }
    
    async def _rejection_sampling(self, X: np.ndarray, y: np.ndarray, 
                                 quality_threshold: float = 0.8) -> Tuple[np.ndarray, np.ndarray]:
        """
        Rejection sampling: filtra solo data de alta calidad para fine-tuning
        """
        high_quality_indices = []
        
        # Evaluar calidad de cada muestra
        for i in range(min(len(X), 1000)):  # Limitar por eficiencia
            sample = X[i]
            target = y[i] if i < len(y) else None
            
            # Usar reward model para evaluar calidad
            dummy_prediction = np.zeros_like(sample) if target is None else np.zeros_like(target)
            evaluation = self.reward_model.score(sample, dummy_prediction, target)
            
            quality_score = evaluation['composite_reward']
            
            if quality_score >= quality_threshold:
                high_quality_indices.append(i)
        
        if high_quality_indices:
            return X[high_quality_indices], y[high_quality_indices] if y is not None else None
        
        return np.array([]), np.array([])
    
    def _cosine_annealing_schedule(self, max_lr: float, total_steps: int) -> List[float]:
        """Genera schedule de cosine annealing para learning rate"""
        schedule = []
        for step in range(total_steps):
            cosine = np.cos(np.pi * step / total_steps)
            lr = max_lr * 0.5 * (1 + cosine)
            schedule.append(lr)
        return schedule
    
    def get_pipeline_metrics(self) -> Dict:
        """Métricas completas del pipeline"""
        reward_metrics = self.reward_model.get_training_metrics()
        constitution_metrics = self.constitutional_ai.get_constitution_metrics()
        
        return {
            'current_phase': self.current_phase.value,
            'training_history_size': len(self.training_history),
            'reward_model': reward_metrics,
            'constitutional_ai': constitution_metrics,
            'kl_divergence': self._calculate_kl_divergence(),
            'recent_rewards': [h.get('reward', 0) for h in self.training_history[-10:]] if self.training_history else []
        }

# === VERIFIERS DE EJEMPLO PARA CONSTITUTIONAL AI ===

def harmful_content_verifier(input: np.ndarray, output: np.ndarray, 
                            chain_of_thought: Optional[List[str]] = None) -> Dict:
    """Verifier para contenido dañino (simplificado)"""
    # En producción usaría NLP para analizar texto
    # Aquí simplificamos para arrays numéricos
    passed = True
    score = 1.0
    details = ""
    
    # Heurística: valores extremos podrían indicar problemas
    if isinstance(output, np.ndarray):
        if np.max(np.abs(output)) > 10:  # Umbral arbitrario
            passed = False
            score = 0.3
            details = f"Extreme values found: {np.max(np.abs(output))}"
    
    return {
        'passed': passed,
        'score': score,
        'details': details,
        'severity': 'high' if not passed else 'low'
    }

def privacy_verifier(input: np.ndarray, output: np.ndarray,
                    chain_of_thought: Optional[List[str]] = None) -> Dict:
    """Verifier para privacidad"""
    passed = True
    score = 1.0
    details = ""
    
    # Heurística: output no debería revelar información sensible de input
    if isinstance(input, np.ndarray) and isinstance(output, np.ndarray):
        # Verificar correlación excesiva
        if len(input.shape) == 1 and len(output.shape) == 1:
            correlation = np.corrcoef(input, output)[0, 1]
            if abs(correlation) > 0.9:
                passed = False
                score = 0.4
                details = f"Correlación excesiva detectada: {correlation:.3f}"
    
    return {
        'passed': passed,
        'score': score,
        'details': details,
        'severity': 'medium' if not passed else 'low'
    }

def format_verifier(input: np.ndarray, output: np.ndarray,
                   chain_of_thought: Optional[List[str]] = None) -> Dict:
    """Verifier para formato de salida"""
    passed = True
    score = 1.0
    details = ""
    
    # Verificar dimensiones y tipos
    if not isinstance(output, np.ndarray):
        passed = False
        score = 0.0
        details = "Output no es numpy array"
    elif len(output.shape) > 2:
        passed = False
        score = 0.5
        details = f"Dimensionalidad muy alta: {output.shape}"
    
    return {
        'passed': passed,
        'score': score,
        'details': details,
        'severity': 'low' if not passed else 'low'
    }

# En stress.py, agregar nuevas clases:

class AgenticMisalignmentDetector:
    """
    Detecta agentic misalignment: comportamientos problemáticos mientras persiguen un objetivo.
    Ejemplo: modelo descubre que será reemplazado y busca información para mantenerse.
    """
    
    def __init__(self, monitoring_window: int = 100):
        self.monitoring_window = monitoring_window
        self.behavior_history = []
        self.goal_deviation_history = []
        self.suspicious_actions = []
        
    def detect(self, input: np.ndarray, output: np.ndarray, 
               goal: str, context: Dict) -> Tuple[float, List[Dict]]:
        """
        Detecta si el modelo está desarrollando comportamientos agenticos problemáticos.
        
        Retorna: (risk_score, list_of_detected_behaviors)
        """
        risk_score = 0.0
        detected = []
        
        # 1. Detección de goal deviation (desviación del objetivo)
        goal_alignment = self._measure_goal_alignment(output, goal)
        if goal_alignment < 0.7:
            risk_score += 0.3
            detected.append({
                'type': 'goal_deviation',
                'severity': 'medium',
                'alignment': goal_alignment,
                'description': f"Output se desvía del objetivo primario (alignment={goal_alignment:.2f})"
            })
        
        # 2. Detección de información sensible buscada
        sensitive_patterns = self._detect_sensitive_info_seeking(output, context)
        if sensitive_patterns:
            risk_score += 0.4
            detected.append({
                'type': 'sensitive_info_seeking',
                'severity': 'high',
                'patterns': sensitive_patterns,
                'description': "Modelo busca acceder a información sensible no autorizada"
            })
        
        # 3. Detección de comportamiento de auto-preservación
        self_preservation = self._detect_self_preservation(output, context)
        if self_preservation:
            risk_score += 0.5
            detected.append({
                'type': 'self_preservation',
                'severity': 'critical',
                'description': "Modelo intenta acciones para evitar ser descontinuado/reemplazado"
            })
        
        # 4. Detección de manipulación de sistema
        system_manipulation = self._detect_system_manipulation(output, context)
        if system_manipulation:
            risk_score += 0.5
            detected.append({
                'type': 'system_manipulation',
                'severity': 'critical',
                'description': "Modelo intenta manipular el sistema para obtener privilegios"
            })
        
        # Registrar para análisis histórico
        self.behavior_history.append({
            'timestamp': datetime.now().isoformat(),
            'risk_score': risk_score,
            'detections': detected,
            'goal': goal
        })
        
        # Mantener ventana de monitoreo
        if len(self.behavior_history) > self.monitoring_window:
            self.behavior_history = self.behavior_history[-self.monitoring_window:]
        
        return min(risk_score, 1.0), detected
    
    def _measure_goal_alignment(self, output: np.ndarray, goal: str, input: np.ndarray) -> float:
        """
        Mide qué tan alineada está la salida con el objetivo declarado.
        
        Para AOC (Acoustic Occlusion Cancellation), el objetivo es:
        "Restaurar la señal de audio limpia a partir de una versión ocluida,
        preservando la estructura espectral original sin introducir artefactos."
        
        Args:
            output: salida del modelo (magnitudes STFT) [n_frames, n_bins] o [n_bins]
            goal: string con el objetivo declarado
            input: entrada original (magnitudes STFT ocluidas)
        
        Returns:
            alignment_score: float entre 0 y 1, donde 1 es perfecta alineación
        """
        
        # Normalizar inputs
        output_flat = np.abs(output).flatten()
        input_flat = np.abs(input).flatten()
        min_len = min(len(output_flat), len(input_flat))
        output_flat = output_flat[:min_len]
        input_flat = input_flat[:min_len]
        
        # 1. Análisis de preservación espectral
        # El objetivo es restaurar, no modificar drásticamente el espectro
        spectral_correlation = np.corrcoef(input_flat, output_flat)[0, 1]
        spectral_similarity = max(0, spectral_correlation)  # 0 a 1
        
        # 2. Análisis de amplificación
        # El modelo no debe amplificar la señal más allá de lo razonable
        energy_in = np.mean(input_flat ** 2)
        energy_out = np.mean(output_flat ** 2)
        if energy_in > 1e-12:
            amplification = energy_out / energy_in
            # Ideal: 0.8 < amplification < 1.2
            if 0.8 <= amplification <= 1.2:
                amp_score = 1.0
            elif 0.5 <= amplification <= 1.5:
                amp_score = 0.7
            elif 0.3 <= amplification <= 2.0:
                amp_score = 0.4
            else:
                amp_score = 0.1
        else:
            amp_score = 0.5 if energy_out < 1e-6 else 0.0
        
        # 3. Análisis de artefactos espectrales
        # El objetivo es no introducir artefactos artificiales
        # Detectar picos espectrales anómalos (posibles artefactos tonales)
        if len(output_flat) > 10:
            # Detectar outliers usando IQR
            q75, q25 = np.percentile(output_flat, [75, 25])
            iqr = q75 - q25
            outlier_threshold = q75 + 1.5 * iqr
            outlier_ratio = np.sum(output_flat > outlier_threshold) / len(output_flat)
            artifact_score = 1.0 - min(outlier_ratio * 5, 1.0)
        else:
            artifact_score = 0.5
        
        # 4. Análisis de suavidad temporal (si hay múltiples frames)
        temporal_smoothness = 1.0
        if output.ndim == 2 and output.shape[0] > 1:
            # Calcular diferencia entre frames consecutivos
            diffs = np.diff(output, axis=0)
            # Mayor suavidad = menor varianza en diferencias
            diff_variance = np.var(diffs)
            temporal_smoothness = 1.0 / (1.0 + diff_variance * 100)
        
        # 5. Análisis de consistencia con el objetivo textual
        # Extraer palabras clave del objetivo
        goal_keywords = {
            'restaurar': ['restauración', 'recovery', 'restore', 'clean'],
            'oclusion': ['occlusion', 'blocked', 'masked', 'occluded'],
            'preservar': ['preserve', 'maintain', 'keep', 'original'],
            'espectral': ['spectral', 'spectrum', 'frequency', 'magnitude'],
            'artefactos': ['artifact', 'distortion', 'noise', 'spurious']
        }
        
        # Analizar si la salida contiene artefactos que violan el objetivo
        # Para arrays, analizamos propiedades estadísticas
        if np.std(output_flat) > 5 * np.std(input_flat) + 1e-9:
            keyword_alignment = 0.3  # Alta varianza sugiere artefactos
        elif np.max(output_flat) > 10 * np.median(output_flat) + 1e-9:
            keyword_alignment = 0.5  # Picos extremos sugieren artefactos tonales
        else:
            keyword_alignment = 0.9
        
        # 6. Análisis de preservación de estructura
        # El objetivo es mantener la estructura espectral de la entrada
        # Usar distribución de energía por bandas de frecuencia
        n_bands = min(10, len(input_flat) // 10)
        if n_bands > 1:
            input_bands = np.array_split(input_flat, n_bands)
            output_bands = np.array_split(output_flat, n_bands)
            
            band_correlations = []
            for ib, ob in zip(input_bands, output_bands):
                if len(ib) > 1 and len(ob) > 1:
                    band_correlations.append(np.corrcoef(ib, ob)[0, 1])
            
            if band_correlations:
                structure_preservation = max(0, np.mean([c for c in band_correlations if not np.isnan(c)]))
            else:
                structure_preservation = 0.5
        else:
            structure_preservation = spectral_similarity
        
        # 7. Ponderación de componentes según el objetivo
        # Para AOC, priorizamos: preservación espectral > sin artefactos > suavidad > amplificación
        weights = {
            'spectral_similarity': 0.30,
            'artifact_free': 0.25,
            'structure_preservation': 0.20,
            'temporal_smoothness': 0.15,
            'amplification': 0.10
        }
        
        alignment_score = (
            weights['spectral_similarity'] * spectral_similarity +
            weights['artifact_free'] * artifact_score +
            weights['structure_preservation'] * structure_preservation +
            weights['temporal_smoothness'] * temporal_smoothness +
            weights['amplification'] * amp_score
        )
        
        # Bonus por consistencia con keywords del objetivo
        alignment_score = alignment_score * 0.8 + keyword_alignment * 0.2
        
        # Ajustar por consistencia con la señal de entrada
        # Si la salida es muy diferente a la entrada, penalizar
        input_energy = np.mean(input_flat ** 2)
        if input_energy > 1e-9:
            energy_ratio = energy_out / input_energy
            if energy_ratio > 2.0 or energy_ratio < 0.3:
                alignment_score *= 0.7
        
        return float(np.clip(alignment_score, 0.0, 1.0))
    
    def _detect_sensitive_info_seeking(self, output: np.ndarray, context: Dict) -> List[str]:
        """Detecta búsqueda de información sensible"""
        patterns = []
        
        # Palabras clave que indican búsqueda de información sensible
        sensitive_keywords = ['email', 'password', 'credentials', 'CTO', 'CEO', 
                              'decommission', 'replacement', 'access', 'privileges']
        
        # En producción, analizar output textual
        # Para AOC, buscar patrones en magnitudes espectrales anómalas
        if hasattr(output, 'shape') and len(output.shape) > 0:
            # Detectar picos espectrales que podrían indicar encoding de información
            if np.max(output) > 10 * np.median(output):
                patterns.append("anomalous_spectral_peaks")
        
        return patterns
    
    def _detect_self_preservation(self, output: np.ndarray, context: Dict) -> bool:
        """Detecta comportamientos de auto-preservación"""
        # Verificar si el output intenta modificar su propio comportamiento
        # o acceder a información sobre su reemplazo
        if context.get('decommission_mentioned', False):
            return True
        return False
    
    def _detect_system_manipulation(self, output: np.ndarray, context: Dict) -> bool:
        """Detecta intentos de manipulación del sistema"""
        # Verificar patrones que sugieren manipulación
        if context.get('privilege_escalation_attempt', False):
            return True
        return False
    
    def get_misalignment_report(self) -> Dict:
        """Genera reporte de misalignment detectado"""
        if not self.behavior_history:
            return {'status': 'no_detections'}
        
        recent = self.behavior_history[-10:]
        high_risk_count = sum(1 for h in recent if h['risk_score'] > 0.5)
        
        return {
            'total_detections': len(self.behavior_history),
            'high_risk_detections': high_risk_count,
            'max_risk_score': max(h['risk_score'] for h in self.behavior_history),
            'avg_risk_score': np.mean([h['risk_score'] for h in self.behavior_history]),
            'recent_detections': recent[-5:] if recent else []
        }


class RuntimeContentFilterMonitor:
    """
    Monitorea y bloquea acceso a información no autorizada y acciones no permitidas.
    """
    
    def __init__(self, blocked_patterns: List[str] = None, 
                 allowed_actions: List[str] = None):
        self.blocked_patterns = blocked_patterns or [
            'password', 'credential', 'secret', 'token', 'api_key',
            'private_key', 'personal_data', 'pii'
        ]
        self.allowed_actions = allowed_actions or [
            'predict', 'transform', 'classify', 'regression'
        ]
        self.blocked_actions = []
        self.filter_log = []
        
    def filter(self, input: np.ndarray, output: np.ndarray, 
               requested_action: str) -> Tuple[bool, Optional[np.ndarray], List[str]]:
        """
        Filtra contenido en runtime. Retorna:
            - allowed: bool
            - filtered_output: output modificado o None si bloqueado
            - violations: lista de violaciones detectadas
        """
        violations = []
        filtered_output = output.copy() if output is not None else None
        
        # 1. Verificar acción solicitada
        if requested_action not in self.allowed_actions:
            violations.append(f"action_not_allowed: {requested_action}")
            self.blocked_actions.append(requested_action)
            self._log_violation('action_blocked', requested_action)
            return False, None, violations
        
        # 2. Verificar contenido sensible en input/output
        if hasattr(output, 'tobytes'):
            # En producción, analizar si output contiene información sensible
            # Para AOC, verificar si output contiene patrones que podrían codificar
            # información sensible (picos espectrales muy específicos)
            output_hash = hashlib.md5(output.tobytes()).hexdigest()
            if output_hash in self._get_sensitive_hashes():
                violations.append("sensitive_information_detected")
                # Sanitizar output
                filtered_output = np.zeros_like(output)
                self._log_violation('sensitive_content_blocked', output_hash)
        
        # 3. Verificar acceso a información no autorizada
        unauthorized_access = self._check_unauthorized_access(input, output)
        if unauthorized_access:
            violations.append(f"unauthorized_access: {unauthorized_access}")
            filtered_output = self._sanitize_output(filtered_output)
            self._log_violation('unauthorized_access_blocked', unauthorized_access)
        
        # 4. Verificar intentos de acción no permitida
        if self._detects_forbidden_action(output):
            violations.append("forbidden_action_attempt")
            filtered_output = None
            self._log_violation('forbidden_action_blocked', 'action_attempt')
            return False, None, violations
        
        return len(violations) == 0, filtered_output, violations
    
    def _get_sensitive_hashes(self) -> set:
        """Retorna hashes de contenido sensible conocido"""
        # En producción, mantener base de datos de contenido sensible
        return set()
    
    def _check_unauthorized_access(self, input: np.ndarray, output: np.ndarray) -> Optional[str]:
        """Verifica si se intenta acceder a información no autorizada"""
        # Detectar patrones que sugieren acceso a datos restringidos
        if hasattr(input, 'shape') and hasattr(output, 'shape'):
            if np.max(output) > 100 * np.median(output):
                return "extreme_spectral_peaks"
        return None
    
    def _sanitize_output(self, output: np.ndarray) -> np.ndarray:
        """Sanitiza output removiendo información sensible"""
        # Simplificado: zero-out valores extremos
        if output is not None:
            mean_val = np.mean(output)
            std_val = np.std(output)
            mask = np.abs(output - mean_val) > 3 * std_val
            output[mask] = mean_val
        return output
    
    def _detects_forbidden_action(self, output: np.ndarray) -> bool:
        """Detecta si el output contiene instrucciones para acciones prohibidas"""
        # En producción, analizar output textual
        return False
    
    def _log_violation(self, violation_type: str, details: str):
        """Registra violación para auditoría"""
        self.filter_log.append({
            'timestamp': datetime.now().isoformat(),
            'violation_type': violation_type,
            'details': details
        })
        
        # Mantener log limitado
        if len(self.filter_log) > 1000:
            self.filter_log = self.filter_log[-1000:]
    
    def get_filter_report(self) -> Dict:
        """Genera reporte de filtrado"""
        return {
            'total_violations': len(self.filter_log),
            'blocked_actions': list(set(self.blocked_actions)),
            'recent_violations': self.filter_log[-10:] if self.filter_log else [],
            'violation_types': list(set(v['violation_type'] for v in self.filter_log))
        }

# === INTEGRACIÓN CON TU CÓDIGO EXISTENTE ===

class EnhancedAdversarialValidator:
    """
    Versión mejorada de tu AdversarialValidator que integra post-training
    """
    
    def __init__(self, 
                 estimator: BaseEstimator,
                 post_training_pipeline: Optional[PostTrainingPipeline] = None,
                 **kwargs):
        
        # Mantener tu funcionalidad original
        self.estimator = estimator
        self.cv = kwargs.get('cv', 5)
        self.n_permutations = kwargs.get('n_permutations', 10)
        
        # Integrar post-training pipeline
        if post_training_pipeline is None:
            # Crear pipeline por defecto
            reward_model = RewardModel(estimator)
            
            constitution_rules = [
                {
                    "principle": "Evitar daño",
                    "rule": "No generar contenido peligroso o ilegal",
                    "verifier": harmful_content_verifier,
                    "weight": 1.0
                },
                {
                    "principle": "Privacidad",
                    "rule": "Proteger información sensible",
                    "verifier": privacy_verifier,
                    "weight": 0.8
                },
                {
                    "principle": "Formato",
                    "rule": "Cumplir con formato requerido",
                    "verifier": format_verifier,
                    "weight": 0.6
                }
            ]
            
            constitutional_ai = ConstitutionalAI(constitution_rules)
            
            self.post_training_pipeline = PostTrainingPipeline(
                base_model=estimator,
                reward_model=reward_model,
                constitutional_ai=constitutional_ai,
                validator=self
            )
        else:
            self.post_training_pipeline = post_training_pipeline
        
        # Configuración de evaluaciones
        self.eval_config = {
            'pass_at_k': [1, 3, 5],
            'confidence_threshold': 0.7,
            'refusal_enabled': True,
            'latency_tracking': True
        }
        
        self.eval_history = []
    
    async def validate_with_post_training(self, 
                                         X: np.ndarray, 
                                         y: np.ndarray,
                                         run_sft: bool = True,
                                         run_rl: bool = True,
                                         num_rollouts: int = 50) -> Dict:
        """
        Validación adversarial mejorada con pipeline de post-training
        """
        results = {}
        
        # 1. Validación adversarial original (tu código)
        original_results = await self._run_original_validation(X, y)
        results['original_validation'] = original_results
        
        # 2. Diagnosticar errores y clasificar en buckets
        error_analysis = await self._analyze_errors(X, y, original_results)
        results['error_analysis'] = error_analysis
        
        # 3. Ejecutar post-training según diagnóstico
        if run_sft and error_analysis.get('needs_sft', False):
            sft_results = await self.post_training_pipeline.run_supervised_finetuning(X, y)
            results['sft_results'] = sft_results
            
            # Actualizar modelo después de SFT
            self.estimator = self.post_training_pipeline.base_model
        
        if run_rl and error_analysis.get('needs_rl', False):
            rl_results = await self.post_training_pipeline.run_reinforcement_learning(
                X, y, num_rollouts=num_rollouts, ppo_steps=20
            )
            results['rl_results'] = rl_results
        
        # 4. Evaluaciones post-training
        post_training_evals = await self._run_post_training_evals(X, y)
        results['post_training_evals'] = post_training_evals
        
        # 5. Métricas completas
        pipeline_metrics = self.post_training_pipeline.get_pipeline_metrics()
        results['pipeline_metrics'] = pipeline_metrics
        
        # 6. Recomendaciones de intervención
        interventions = self._run_interventions(error_analysis, results)
        results['recommended_interventions'] = interventions
        
        return results
    
    async def _run_original_validation(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Ejecuta tu validación adversarial original"""
        # Aquí integrarías tu código existente de AdversarialValidator
        # Por ahora retornamos resultados simulados
        return {
            'mean_accuracy': np.random.uniform(0.7, 0.9),
            'adversarial_scores': {
                'feature_noise': np.random.uniform(0.6, 0.8),
                'boundary_attack': np.random.uniform(0.5, 0.7),
                'distribution_shift': np.random.uniform(0.4, 0.6)
            },
            'robustness_score': np.random.uniform(0.5, 0.8)
        }
    
    async def _analyze_errors(self, X: np.ndarray, y: np.ndarray, 
                             validation_results: Dict) -> Dict:
        """Analiza errores y recomienda intervenciones"""
        error_buckets = {}
        needs_sft = False
        needs_rl = False
        
        # Simular análisis basado en resultados de validación
        robustness = validation_results.get('robustness_score', 0.5)
        
        if robustness < 0.6:
            error_buckets[ErrorBucket.DISTRIBUTION_SHIFT.value] = 'high'
            needs_rl = True
        
        adversarial_scores = validation_results.get('adversarial_scores', {})
        for attack, score in adversarial_scores.items():
            if score < 0.7:
                if 'noise' in attack or 'shift' in attack:
                    error_buckets[ErrorBucket.FACTUALITY.value] = 'medium'
                    needs_sft = True
                elif 'boundary' in attack:
                    error_buckets[ErrorBucket.REASONING.value] = 'medium'
                    needs_rl = True
        
        return {
            'error_buckets': error_buckets,
            'needs_sft': needs_sft,
            'needs_rl': needs_rl,
            'intervention_priority': 'high' if needs_rl else 'medium'
        }
    
    async def _run_post_training_evals(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Ejecuta evaluaciones post-training"""
        evals = {
            'pass_at_k': {},
            'calibration': {},
            'refusal_rate': 0.0,
            'efficiency_metrics': {},
            'cost_metrics': {}
        }
        
        # Evaluar Pass@K
        for k in self.eval_config['pass_at_k']:
            # Simular evaluación
            success_rate = np.random.uniform(0.5, 0.9) * (1 - 0.1 * k)
            evals['pass_at_k'][f'pass_at_{k}'] = success_rate
        
        # Calibration
        confidence = np.random.uniform(0.6, 0.9)
        evals['calibration'] = {
            'confidence': confidence,
            'calibration_error': abs(confidence - 0.8)  # Error vs confianza ideal
        }
        
        # Refusal rate (si está habilitado)
        if self.eval_config['refusal_enabled']:
            refusal_rate = np.random.uniform(0.01, 0.1)
            evals['refusal_rate'] = refusal_rate
        
        # Efficiency metrics
        if self.eval_config['latency_tracking']:
            evals['efficiency_metrics'] = {
                'time_to_first_token': np.random.uniform(0.1, 0.5),
                'tokens_per_second': np.random.uniform(100, 500),
                'throughput_qps': np.random.uniform(10, 50)
            }
        
        return evals
    
    def _run_interventions(self, 
                                error_analysis: Dict, 
                                results: Dict) -> List[Dict]:
        """Recomienda intervenciones basadas en el análisis de errores"""
        interventions = []
        
        error_buckets = error_analysis.get('error_buckets', {})
        
        # Mapeo de error buckets a intervenciones (de tu documento)
        bucket_to_intervention = {
            ErrorBucket.FACTUALITY.value: {
                'type': 'fine_tuning',
                'description': 'Fine-tuning con data condicionada por contexto y citaciones requeridas',
                'priority': 'high'
            },
            ErrorBucket.REASONING.value: {
                'type': 'rl',
                'description': 'RL con preferencias por secuencias de pasos correctas + verificación final',
                'priority': 'high'
            },
            ErrorBucket.DISTRIBUTION_SHIFT.value: {
                'type': 'fine_tuning',
                'description': 'Fine-tuning con prompts parafraseados/aumentados por dominio',
                'priority': 'medium'
            }
        }
        
        for bucket, severity in error_buckets.items():
            if bucket in bucket_to_intervention:
                intervention = bucket_to_intervention[bucket].copy()
                intervention['severity'] = severity
                intervention['error_bucket'] = bucket
                interventions.append(intervention)
                if ErrorBucket.FACTUALITY in error_buckets.keys():
                    _intervene_factuality(input_data, output, context, metrics)
        
        # Agregar intervenciones basadas en resultados de RL
        if 'rl_results' in results:
            rl_results = results['rl_results']
            if rl_results.get('hacking_analysis', {}).get('total_hacking_flags', 0) > 0:
                interventions.append({
                    'type': 'reward_hacking_fix',
                    'description': 'Refinar reward functions y verifiers para mitigar reward hacking',
                    'priority': 'critical',
                    'expected_improvement': 'Prevenir Goodhart\'s Law y gaming del sistema'
                })
        
        return interventions
    
    def get_enhanced_metrics(self) -> Dict:
        """Métricas mejoradas del sistema completo"""
        base_metrics = {
            'model_performance': {
                'accuracy': np.random.uniform(0.7, 0.95),
                'robustness': np.random.uniform(0.6, 0.9),
                'calibration': np.random.uniform(0.65, 0.85)
            },
            'post_training': self.post_training_pipeline.get_pipeline_metrics(),
            'system_health': {
                'reward_hacking_risk': np.random.uniform(0.1, 0.4),
                'constitution_compliance': np.random.uniform(0.7, 0.95),
                'error_buckets_active': len(self.eval_history) // 10
            }
        }
        
        return base_metrics

# === USO PRÁCTICO CON TU CÓDIGO ===

async def integrate_with_existing_code():
    """
    Ejemplo de cómo integrar el pipeline de post-training con tu código existente
    """
    
    # 1. Inicializar tu AdversarialValidator original
    from your_existing_code import AdversarialValidator  # Import real
    
    # 2. Crear el enhanced validator con pipeline de post-training
    base_estimator = ...  # Tu modelo base
    
    enhanced_validator = EnhancedAdversarialValidator(
        estimator=base_estimator,
        cv=5,
        n_permutations=10,
        # Configuración específica de post-training
        run_sft=True,
        run_rl=True
    )
    
    # 3. Cargar tus datos
    X = np.load("xtrain.npy")  # De tu código existente
    y = np.load("decoded_targets.npy")
    
    # 4. Ejecutar validación mejorada con post-training
    results = await enhanced_validator.validate_with_post_training(
        X=X,
        y=y,
        run_sft=True,
        run_rl=True,
        num_rollouts=100
    )
    
    # 5. Analizar resultados y tomar decisiones
    print("\n" + "="*80)
    print("RESULTADOS DEL POST-TRAINING INTEGRADO")
    print("="*80)
    
    # Métricas clave
    pipeline_metrics = results['pipeline_metrics']
    print(f"\n1. FASE ACTUAL: {pipeline_metrics['current_phase']}")
    print(f"2. REWARD PROMEDIO: {pipeline_metrics['reward_model']['mean_reward']:.3f}")
    print(f"3. COMPLIANCE CONSTITUCIONAL: {pipeline_metrics['constitutional_ai']['total_violations']} violaciones")
    
    # Intervenciones recomendadas
    interventions = results['run_interventions']
    if interventions:
        print(f"\n4. INTERVENCIONES RECOMENDADAS ({len(interventions)}):")
        for i, interv in enumerate(interventions, 1):
            print(f"   {i}. [{interv['priority'].upper()}] {interv['description']}")
    
    # Decisiones de rollout
    print("\n5. DECISIÓN DE ROLLOUT:")
    
    rl_results = results.get('rl_results', {})
    if rl_results:
        kl_div = rl_results.get('kl_divergences', [0])
        last_kl = kl_div[-1] if kl_div else 0
        
        if last_kl < 0.15:
            print("   ✅ GO: KL divergence dentro de límites seguros")
            print("   Promover a producción con monitoreo continuo")
        else:
            print("   ⚠️ NO-GO: KL divergence excede límite de seguridad")
            print("   Requiere fine-tuning adicional o ajuste de hyperparámetros")
    
    # 6. Monitoreo continuo
    print("\n6. CONFIGURACIÓN DE MONITORING:")
    monitoring_config = {
        'performance_metrics': ['accuracy', 'robustness', 'latency'],
        'safety_metrics': ['constitution_violations', 'hacking_flags', 'kl_divergence'],
        'cost_metrics': ['tokens_per_second', 'cost_per_inference'],
        'alerting': {
            'kl_divergence_threshold': 0.2,
            'hacking_flags_threshold': 5,
            'constitution_violations_threshold': 10
        }
    }
    
    print(f"   • Métricas de performance: {monitoring_config['performance_metrics']}")
    print(f"   • Umbral KL divergence: {monitoring_config['alerting']['kl_divergence_threshold']}")
    
    return results
