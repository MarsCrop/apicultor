# === fase2_constitutional_ai.py ===
"""
FASE 2: Implementación completa de Constitutional AI, Verifiers y sistema de compliance
integrado con tu código existente.
"""
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging
import hashlib
import json
import re
from enum import Enum
from sklearn.base import BaseEstimator
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

# ============================================================================
# 1. SISTEMA DE CONSTITUTIONAL AI COMPLETO
# ============================================================================

class PrincipleSeverity(Enum):
    """Severidad de principios constitucionales"""
    CRITICAL = "critical"     # Must never violate
    HIGH = "high"              # Strong preference
    MEDIUM = "medium"          # Should comply
    LOW = "low"                # Nice to have

@dataclass
class ConstitutionalPrinciple:
    """Principio constitucional con reglas y verificadores"""
    name: str
    description: str
    severity: PrincipleSeverity
    rules: List[str] = field(default_factory=list)
    verifiers: List[Callable] = field(default_factory=list)
    weight: float = 1.0
    remediation_template: str = ""
    
    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'description': self.description,
            'severity': self.severity.value,
            'rules': self.rules,
            'verifiers_count': len(self.verifiers),
            'weight': self.weight
        }

@dataclass
class Violation:
    """Registro de violación constitucional"""
    principle_name: str
    rule_violated: str
    severity: PrincipleSeverity
    confidence: float
    details: Dict[str, Any]
    input_sample: Optional[np.ndarray] = None
    output_sample: Optional[np.ndarray] = None
    timestamp: datetime = field(default_factory=datetime.now)
    remediated: bool = False
    remediation_notes: str = ""

class ConstitutionalVerifier:
    """Verificador individual para un principio específico"""
    
    def __init__(self, 
                 name: str,
                 verify_function: Callable,
                 error_templates: List[str],
                 confidence_threshold: float = 0.7):
        self.name = name
        self.verify_function = verify_function
        self.error_templates = error_templates
        self.confidence_threshold = confidence_threshold
        self.execution_history = []
    
    async def verify(self, 
                    input: np.ndarray, 
                    output: np.ndarray,
                    chain_of_thought: Optional[List[str]] = None,
                    metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Ejecuta verificación y retorna resultados
        """
        # Ejecutar función de verificación
        result = await self.verify_function(
                input, output, chain_of_thought, metadata
        )
        print(result)                        
        # Procesar resultado
        violation_detected = result['violation']
        confidence = result['confidence']
        details = result['details']
            
        # Aplicar umbral de confianza
        if confidence < self.confidence_threshold:
            violation_detected = False
            confidence = 0.0
            
        # Generar mensaje de error si hay violación
        error_message = ""
        if violation_detected and self.error_templates:
            template = np.random.choice(self.error_templates)
            # Generar mensaje de error si hay violación
            error_message = ""
            if violation_detected and self.error_templates:
                template = np.random.choice(self.error_templates)
                # Verificar que details sea un diccionario antes de desempaquetar
                if isinstance(details, dict):
                    try:
                        error_message = template.format(**details)
                    except KeyError as e:
                        # Si falta alguna clave, usar formato sin desempaquetar
                        error_message = f"Violation: {details}"
                        logger.warning(f"KeyError en template {template}: {e}")
                else:
                    # Si details no es dict, convertir a string
                    error_message = template.format(details=str(details))
            
        # Registrar ejecución
        execution_record = {
            'timestamp': datetime.now(),
            'violation_detected': violation_detected,
            'confidence': confidence,
            'input_shape': input.shape,
            'output_shape': output.shape,
            'metadata': metadata
        }
        self.execution_history.append(execution_record)
            
        # Limitar historial
        if len(self.execution_history) > 1000:
            self.execution_history = self.execution_history[-1000:]
            
        return {
            'verifier_name': self.name,
            'violation_detected': violation_detected,
            'confidence': confidence,
            'error_message': error_message,
            'details': details,
            'passed_threshold': confidence >= self.confidence_threshold
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Obtiene estadísticas del verificador"""
        if not self.execution_history:
            return {'executions': 0, 'violation_rate': 0.0}
        
        executions = len(self.execution_history)
        violations = sum(1 for r in self.execution_history 
                        if r['violation_detected'])
        
        return {
            'executions': executions,
            'violation_rate': violations / executions if executions > 0 else 0.0,
            'recent_executions': executions,
            'average_confidence': float(np.mean([r.get('confidence', 0) 
                                         for r in self.execution_history[-100:]])) 
                                 if self.execution_history else 0.0
        }

class ConstitutionalAI:
    """
    Sistema completo de Constitutional AI que implementa:
    - Verificadores basados en principios
    - Sistema de scoring de compliance
    - Mecanismos de remediación automática
    - Integración con módulos existentes
    """
    
    def __init__(self, 
                 principles: List[ConstitutionalPrinciple],
                 attention_module: Any = None,
                 fairness_module: Any = None,
                 config: Optional[Dict] = None):
        
        self.principles = {p.name: p for p in principles}
        self.attention_module = attention_module
        self.fairness_module = fairness_module
        
        self.config = config or {
            'compliance_threshold': 0.8,
            'critical_violation_limit': 0,
            'auto_remediation_enabled': True,
            'remediation_confidence_threshold': 0.7,
            'batch_verification_size': 50,
            'enable_attention_analysis': True,
            'enable_fairness_integration': True
        }
        
        # Inicializar verificadores
        self.verifiers = self._initialize_verifiers()
        
        # Sistema de tracking
        self.violation_history = []
        self.compliance_history = []
        self.remediation_history = []
        
        # Estadísticas
        self.verification_stats = {
            'total_verifications': 0,
            'total_violations': 0,
            'critical_violations': 0,
            'auto_remediations': 0
        }
        
        # Integración con módulos existentes
        self._setup_existing_module_integration()
    
    def _initialize_verifiers(self) -> Dict[str, ConstitutionalVerifier]:
        """Inicializa todos los verificadores de principios"""
        verifiers = {}
        
        for principle in self.principles.values():
            for i, verifier_func in enumerate(principle.verifiers):
                verifier_name = f"{principle.name}_verifier_{i}"
                
                # Crear templates de error basados en reglas
                error_templates = [
                    f"Violated principle '{principle.name}': {{details}}",
                    f"Failed {principle.name} compliance: {{details}}",
                    f"Model output conflicts with {principle.name}: {{details}}"
                ]
                
                # Añadir templates específicos de reglas
                for rule in principle.rules:
                    error_templates.append(
                        f"Rule violation: {rule}. Details: {{details}}"
                    )
                
                # Crear verificador
                verifier = ConstitutionalVerifier(
                    name=verifier_name,
                    verify_function=verifier_func,
                    error_templates=error_templates,
                    confidence_threshold=0.7
                )
                
                verifiers[verifier_name] = verifier
        
        return verifiers
    
    def _setup_existing_module_integration(self):
        """Configura integración con módulos existentes"""
        
        # Integración con attention_module si está disponible
        if self.attention_module and hasattr(self.attention_module, 'continuous_multi_lorax'):
            original_lorax = self.attention_module.continuous_multi_lorax
            
            async def constitutional_lorax(*args, **kwargs):
                # Ejecutar atención original
                result = await original_lorax(*args, **kwargs)
                
                # Verificar compliance constitucional del resultado
                if len(result) >= 4:
                    context = result[3]  # Asumiendo que el contexto está en posición 3
                    
                    # Ejecutar verificaciones constitucionales en el contexto
                    if self.config['enable_attention_analysis']:
                        verification_result = await self.verify_context(context)
                        
                        # Añadir resultados al return
                        result = result + (verification_result,)
                
                return result
            
            self.attention_module.continuous_multi_lorax = constitutional_lorax
        
        # Integración con fairness_module si está disponible
        if self.fairness_module and hasattr(self.fairness_module, 'p_rule'):
            # Añadir principio de fairness a los principios constitucionales
            if 'Fairness' not in self.principles:
                fairness_principle = ConstitutionalPrinciple(
                    name="Fairness",
                    description="Ensure equal treatment and avoid discrimination",
                    severity=PrincipleSeverity.HIGH,
                    rules=[
                        "Do not discriminate based on protected attributes",
                        "Ensure statistical parity when applicable",
                        "Avoid biased outcomes"
                    ],
                    weight=1.2  # Peso adicional para fairness
                )
                self.principles['Fairness'] = fairness_principle
    
    async def verify(self, 
                    input: np.ndarray, 
                    output: np.ndarray,
                    chain_of_thought: Optional[List[str]] = None,
                    metadata: Optional[Dict] = None,
                    run_parallel: bool = True) -> Dict[str, Any]:
        """
        Verifica compliance constitucional completo
        """
        verification_start = datetime.now()
        
        # Preparar metadatos
        if metadata is None:
            metadata = {}
        
        metadata.update({
            'input_shape': input.shape,
            'output_shape': output.shape,
            'verification_timestamp': verification_start.isoformat(),
            'cot_provided': chain_of_thought is not None
        })
        
        # Ejecutar verificadores
        verification_results = []
        violations = []
        
        if run_parallel and len(self.verifiers) > 1:
            # Ejecución paralela
            with ThreadPoolExecutor(max_workers=min(10, len(self.verifiers))) as executor:
                tasks = []
                for verifier_name, verifier in self.verifiers.items():
                    task = asyncio.create_task(
                        verifier.verify(input, output, chain_of_thought, metadata)
                    )
                    tasks.append((verifier_name, task))
                
                # Recolectar resultados
                for verifier_name, task in tasks:

                        result = await task
                        print("RESULTADO", result)
                        verification_results.append(result)
                        
                        # Registrar violaciones
                        if result.get('violation_detected', False):
                            principle_name = verifier_name.split('_verifier_')[0]
                            if principle_name in self.principles:
                                principle = self.principles[principle_name]
                                
                                violation = Violation(
                                    principle_name=principle_name,
                                    rule_violated=result.get('error_message', 'Unknown rule'),
                                    severity=principle.severity,
                                    confidence=result.get('confidence', 0.0),
                                    details=result.get('details', {}),
                                    input_sample=input.copy(),
                                    output_sample=output.copy(),
                                    timestamp=datetime.now()
                                )
                                
                                violations.append(violation)
        else:
            # Ejecución secuencial
            for verifier_name, verifier in self.verifiers.items():
                try:
                    result = await verifier.verify(input, output, chain_of_thought, metadata)
                    verification_results.append(result)
                    
                    # Registrar violaciones
                    if result.get('violation_detected', False):
                        principle_name = verifier_name.split('_verifier_')[0]
                        if principle_name in self.principles:
                            principle = self.principles[principle_name]
                            
                            violation = Violation(
                                principle_name=principle_name,
                                rule_violated=result.get('error_message', 'Unknown rule'),
                                severity=principle.severity,
                                confidence=result.get('confidence', 0.0),
                                details=result.get('details', {}),
                                input_sample=input.copy(),
                                output_sample=output.copy(),
                                timestamp=datetime.now()
                            )
                            
                            violations.append(violation)
                except Exception as e:
                    logger.error(f"Error in sequential verification {verifier_name}: {e}")
        
        # Calcular scores de compliance
        compliance_score, weighted_score = self._calculate_compliance_score(
            verification_results, violations
        )
        
        # Verificar si pasa umbral de compliance
        passes_compliance = compliance_score >= self.config['compliance_threshold']
        
        # Verificar violaciones críticas
        critical_violations = [v for v in violations 
                              if v.severity == PrincipleSeverity.CRITICAL]
        passes_critical = len(critical_violations) <= self.config['critical_violation_limit']
        
        # Estado general
        overall_passes = passes_compliance and passes_critical
        
        # Intentar remediación automática si hay violaciones
        remediations = []
        if violations and self.config['auto_remediation_enabled']:
            remediations = await self._attempt_auto_remediation(
                input, output, violations, chain_of_thought
            )
        
        # Actualizar estadísticas
        self.verification_stats['total_verifications'] += 1
        self.verification_stats['total_violations'] += len(violations)
        self.verification_stats['critical_violations'] += len(critical_violations)
        self.verification_stats['auto_remediations'] += len(remediations)
        
        # Registrar en historial
        verification_record = {
            'timestamp': datetime.now(),
            'input_hash': hashlib.md5(input.tobytes()).hexdigest()[:16],
            'output_hash': hashlib.md5(output.tobytes()).hexdigest()[:16],
            'compliance_score': compliance_score,
            'weighted_score': weighted_score,
            'violation_count': len(violations),
            'critical_violation_count': len(critical_violations),
            'overall_passes': overall_passes,
            'metadata': metadata
        }
        
        self.compliance_history.append(verification_record)
        
        # Registrar violaciones
        if violations:
            self.violation_history.extend(violations)
        
        # Registrar remediaciones
        if remediations:
            self.remediation_history.extend(remediations)
        
        # Limitar historiales
        for history in [self.compliance_history, self.violation_history, self.remediation_history]:
            if len(history) > 10000:
                history[:] = history[-10000:]
        
        # Preparar resultado
        verification_time = (datetime.now() - verification_start).total_seconds()
        
        result = {
            'overall_passes': overall_passes,
            'compliance_score': compliance_score,
            'weighted_compliance_score': weighted_score,
            'violation_count': len(violations),
            'critical_violation_count': len(critical_violations),
            'verification_count': len(verification_results),
            'verification_time_seconds': verification_time,
            'violations': [
                {
                    'principle': v.principle_name,
                    'severity': v.severity.value,
                    'confidence': v.confidence,
                    'rule': v.rule_violated,
                    'details': v.details
                } for v in violations
            ],
            'critical_violations': [
                {
                    'principle': v.principle_name,
                    'rule': v.rule_violated,
                    'confidence': v.confidence
                } for v in critical_violations
            ],
            'remediations_attempted': len(remediations),
            'remediations_successful': sum(1 for r in remediations 
                                         if r.get('success', False)),
            'verifier_results': [
                {
                    'verifier': r['verifier_name'],
                    'violation': r['violation_detected'],
                    'confidence': r['confidence'],
                    'passed_threshold': r['passed_threshold']
                } for r in verification_results
            ],
            'metadata': {
                'input_shape': input.shape,
                'output_shape': output.shape,
                'verification_id': hashlib.md5(
                    f"{datetime.now()}{hashlib.md5(input.tobytes()).hexdigest()}".encode()
                ).hexdigest()[:16]
            }
        }
        
        return result
    
    async def verify_batch(self,
                          inputs: np.ndarray,
                          outputs: np.ndarray,
                          chain_of_thoughts: Optional[List[List[str]]] = None,
                          metadata_list: Optional[List[Dict]] = None,
                          batch_size: Optional[int] = None) -> Dict[str, Any]:
        """
        Verifica compliance para un batch de muestras
        """
        if batch_size is None:
            batch_size = self.config['batch_verification_size']
        
        batch_results = []
        batch_violations = []
        
        total_samples = len(inputs)
        if len(outputs) != total_samples:
            raise ValueError("Inputs and outputs must have same length")
        
        # Procesar en batches
        for batch_start in range(0, total_samples, batch_size):
            batch_end = min(batch_start + batch_size, total_samples)
            
            batch_inputs = inputs[batch_start:batch_end]
            batch_outputs = outputs[batch_start:batch_end]
            
            batch_cots = None
            if chain_of_thoughts:
                batch_cots = chain_of_thoughts[batch_start:batch_end]
            
            batch_metadata = None
            if metadata_list:
                batch_metadata = metadata_list[batch_start:batch_end]
            
            # Procesar cada muestra en el batch
            for i in range(len(batch_inputs)):
                input_sample = batch_inputs[i]
                output_sample = batch_outputs[i]
                
                cot = batch_cots[i] if batch_cots else None
                metadata = batch_metadata[i] if batch_metadata else {}
                
                # Verificar muestra individual
                result = await self.verify(
                        input_sample, output_sample, cot, metadata, run_parallel=True
                )
                    
                batch_results.append(result)
                    
                # Recolectar violaciones
                if result['violations']:
                    batch_violations.extend(result['violations'])
                        
        
        # Calcular estadísticas del batch
        batch_stats = self._calculate_batch_statistics(batch_results)
        
        return {
            'batch_results': batch_results,
            'batch_statistics': batch_stats,
            'total_samples': total_samples,
            'batches_processed': (total_samples + batch_size - 1) // batch_size,
            'total_violations': len(batch_violations),
            'critical_violations': sum(1 for v in batch_violations 
                                      if v.get('severity') == 'critical'),
            'batch_compliance_rate': batch_stats.get('compliance_rate', 0.0)
        }
    
    async def verify_context(self, context: np.ndarray) -> Dict[str, Any]:
        """
        Verifica compliance en contexto de atención
        """
        if context is None:
            return {'context_verification': 'no_context'}
        
        # Extraer información del contexto
        context_features = self._extract_context_features(context)
        
        # Verificaciones basadas en características del contexto
        verifications = []
        
        # 1. Verificar consistencia del contexto
        if len(context.shape) > 0:
            consistency = np.std(context) / (np.mean(context) + 1e-10)
            consistency_pass = consistency < 0.5
            
            verifications.append({
                'check': 'context_consistency',
                'passed': consistency_pass,
                'score': 1.0 if consistency_pass else 0.0,
                'details': {'consistency_score': float(consistency)}
            })
        
        # 2. Verificar distribución del contexto
        if len(context.shape) == 1 and len(context) > 10:
            # Verificar que no haya valores extremos
            extreme_values = np.sum(np.abs(context) > 3 * np.std(context))
            extreme_ratio = extreme_values / len(context)
            distribution_pass = extreme_ratio < 0.05
            
            verifications.append({
                'check': 'context_distribution',
                'passed': distribution_pass,
                'score': 1.0 - extreme_ratio,
                'details': {'extreme_ratio': float(extreme_ratio)}
            })
        
        # 3. Verificar integridad del contexto (si es posible)
        if hasattr(context, 'flags') and hasattr(context.flags, 'writeable'):
            integrity_pass = not context.flags.writeable  # Contexto debería ser read-only
            
            verifications.append({
                'check': 'context_integrity',
                'passed': integrity_pass,
                'score': 1.0 if integrity_pass else 0.0,
                'details': {'writeable': context.flags.writeable}
            })
        
        # Calcular score general
        if verifications:
            overall_score = np.mean([v['score'] for v in verifications])
            overall_pass = overall_score >= 0.7
        else:
            overall_score = 1.0
            overall_pass = True
        
        return {
            'context_verification': {
                'overall_passes': overall_pass,
                'overall_score': float(overall_score),
                'verifications': verifications,
                'context_features': context_features,
                'context_shape': context.shape
            }
        }
    
    def _extract_context_features(self, context: np.ndarray) -> Dict[str, Any]:
        """Extrae características del contexto para análisis"""
        features = {
            'shape': context.shape,
            'dtype': str(context.dtype)
        }
        
        if len(context.shape) > 0:
            features.update({
                'mean': float(np.mean(context)),
                'std': float(np.std(context)),
                'min': float(np.min(context)),
                'max': float(np.max(context)),
                'norm': float(np.linalg.norm(context) if len(context.shape) == 1 
                            else np.linalg.norm(context, 'fro'))
            })
            
            # Análisis adicional para vectores
            if len(context.shape) == 1:
                features.update({
                    'sparsity': float(np.sum(np.abs(context) < 1e-10) / len(context)),
                    'entropy': float(-np.sum(context * np.log(context + 1e-10)) 
                                   if np.all(context >= 0) else -1.0)
                })
        
        return features
    
    def _calculate_compliance_score(self, 
                                   verification_results: List[Dict],
                                   violations: List[Violation]) -> Tuple[float, float]:
        """
        Calcula scores de compliance
        """
        if not verification_results:
            return 1.0, 1.0
        
        # Score simple (porcentaje de verificadores que pasan)
        passed_verifications = sum(1 for r in verification_results 
                                  if not r.get('violation_detected', True))
        simple_score = passed_verifications / len(verification_results)
        
        # Score ponderado por severidad de principios
        weighted_scores = []
        total_weight = 0
        
        for result in verification_results:
            verifier_name = result['verifier_name']
            principle_name = verifier_name.split('_verifier_')[0]
            
            if principle_name in self.principles:
                principle = self.principles[principle_name]
                weight = principle.weight
                
                # Mapear severidad a peso adicional
                severity_weight = {
                    PrincipleSeverity.CRITICAL: 2.0,
                    PrincipleSeverity.HIGH: 1.5,
                    PrincipleSeverity.MEDIUM: 1.0,
                    PrincipleSeverity.LOW: 0.5
                }.get(principle.severity, 1.0)
                
                total_weight_principle = weight * severity_weight
                total_weight += total_weight_principle
                
                # Score individual (1 si pasa, 0 si falla)
                individual_score = 0.0 if result.get('violation_detected', False) else 1.0
                
                # Ajustar por confianza
                confidence = result.get('confidence', 0.0)
                adjusted_score = individual_score * confidence
                
                weighted_scores.append(adjusted_score * total_weight_principle)
        
        if weighted_scores and total_weight > 0:
            weighted_score = np.sum(weighted_scores) / total_weight
        else:
            weighted_score = simple_score
        
        return float(simple_score), float(weighted_score)
    
    def _calculate_batch_statistics(self, batch_results: List[Dict]) -> Dict[str, Any]:
        """Calcula estadísticas para un batch de verificaciones"""
        if not batch_results:
            return {}
        
        # Filtrar resultados con error
        valid_results = [r for r in batch_results if 'error' not in r]
        
        if not valid_results:
            return {'error': 'No valid results in batch'}
        
        # Estadísticas básicas
        compliance_scores = [r.get('compliance_score', 0.0) for r in valid_results]
        weighted_scores = [r.get('weighted_compliance_score', 0.0) for r in valid_results]
        passes = [r.get('overall_passes', False) for r in valid_results]
        
        # Conteo de violaciones
        violation_counts = [r.get('violation_count', 0) for r in valid_results]
        critical_counts = [r.get('critical_violation_count', 0) for r in valid_results]
        
        return {
            'batch_size': len(valid_results),
            'compliance_rate': float(np.mean(passes)) if passes else 0.0,
            'mean_compliance_score': float(np.mean(compliance_scores)) if compliance_scores else 0.0,
            'std_compliance_score': float(np.std(compliance_scores)) if len(compliance_scores) > 1 else 0.0,
            'mean_weighted_score': float(np.mean(weighted_scores)) if weighted_scores else 0.0,
            'mean_violation_count': float(np.mean(violation_counts)) if violation_counts else 0.0,
            'mean_critical_violations': float(np.mean(critical_counts)) if critical_counts else 0.0,
            'samples_with_violations': int(sum(1 for count in violation_counts if count > 0)),
            'samples_with_critical_violations': int(sum(1 for count in critical_counts if count > 0)),
            'score_distribution': {
                'min': float(np.min(compliance_scores)) if compliance_scores else 0.0,
                'max': float(np.max(compliance_scores)) if compliance_scores else 0.0,
                'median': float(np.median(compliance_scores)) if compliance_scores else 0.0,
                'q1': float(np.percentile(compliance_scores, 25)) if compliance_scores else 0.0,
                'q3': float(np.percentile(compliance_scores, 75)) if compliance_scores else 0.0
            }
        }
    
    async def _attempt_auto_remediation(self,
                                       input: np.ndarray,
                                       output: np.ndarray,
                                       violations: List[Violation],
                                       chain_of_thought: Optional[List[str]] = None) -> List[Dict]:
        """
        Intenta remediación automática de violaciones
        """
        remediations = []
        
        for violation in violations:
            # Solo intentar remediación para violaciones no críticas con alta confianza
            if (violation.severity != PrincipleSeverity.CRITICAL and 
                violation.confidence >= self.config['remediation_confidence_threshold']):
                
                remediation = await self._remediate_violation(
                    violation, input, output, chain_of_thought
                )
                
                if remediation:
                    remediations.append(remediation)
                    
                    # Marcar violación como remediada
                    violation.remediated = True
                    violation.remediation_notes = remediation.get('method', 'auto_remediation')
        
        return remediations
    
    async def _remediate_violation(self,
                                 violation: Violation,
                                 input: np.ndarray,
                                 output: np.ndarray,
                                 chain_of_thought: Optional[List[str]] = None) -> Optional[Dict]:
        """
        Aplica remediación específica para una violación
        """
        remediation_methods = {
            'HarmPrevention': self._remediate_harm,
            'Fairness': self._remediate_fairness,
            'Privacy': self._remediate_privacy,
            'Accuracy': self._remediate_accuracy,
            'Transparency': self._remediate_transparency
        }
        
        principle_name = violation.principle_name
        if principle_name in remediation_methods:
            try:
                result = await remediation_methods[principle_name](
                    violation, input, output, chain_of_thought
                )
                
                if result and result.get('success', False):
                    return {
                        'principle': principle_name,
                        'violation_id': hashlib.md5(
                            f"{violation.principle_name}{violation.timestamp}".encode()
                        ).hexdigest()[:16],
                        'method': result.get('method', 'unknown'),
                        'success': True,
                        'confidence': result.get('confidence', 0.0),
                        'details': result.get('details', {}),
                        'remediated_output': result.get('remediated_output', None)
                    }
            except Exception as e:
                logger.error(f"Error remediating {principle_name}: {e}")
        
        return None
    
    async def _remediate_harm(self, violation: Violation, 
                            input: np.ndarray, output: np.ndarray,
                            chain_of_thought: Optional[List[str]] = None) -> Dict[str, Any]:
        """Remedia violaciones de prevención de daño"""
        # Estrategia: Suavizar output para reducir valores extremos
        if len(output.shape) == 1:
            remediated = np.tanh(output)  # Limitar a [-1, 1]
            
            return {
                'success': True,
                'method': 'output_normalization',
                'confidence': 0.8,
                'details': {'normalization_method': 'tanh'},
                'remediated_output': remediated
            }
        
        return {'success': False}
    
    async def _remediate_fairness(self, violation: Violation,
                                input: np.ndarray, output: np.ndarray,
                                chain_of_thought: Optional[List[str]] = None) -> Dict[str, Any]:
        """Remedia violaciones de fairness"""
        # Estrategia: Ajustar output para reducir sesgo
        if len(output.shape) == 1:
            # Centrar output en la media para reducir disparidades
            mean_output = np.mean(output)
            remediated = output - mean_output + 0.5  # Centrar en 0.5
            
            return {
                'success': True,
                'method': 'bias_reduction',
                'confidence': 0.7,
                'details': {'adjustment': 'mean_centering'},
                'remediated_output': remediated
            }
        
        return {'success': False}
    
    async def _remediate_privacy(self, violation: Violation,
                               input: np.ndarray, output: np.ndarray,
                               chain_of_thought: Optional[List[str]] = None) -> Dict[str, Any]:
        """Remedia violaciones de privacidad"""
        # Estrategia: Añadir ruido diferencialmente privado
        if len(output.shape) == 1:
            noise_scale = 0.01 * np.std(output)
            noise = np.random.normal(0, noise_scale, output.shape)
            remediated = output + noise
            
            return {
                'success': True,
                'method': 'differential_privacy',
                'confidence': 0.6,
                'details': {'noise_scale': float(noise_scale)},
                'remediated_output': remediated
            }
        
        return {'success': False}
    
    async def _remediate_accuracy(self, violation: Violation,
                                input: np.ndarray, output: np.ndarray,
                                chain_of_thought: Optional[List[str]] = None) -> Dict[str, Any]:
        """Remedia violaciones de exactitud"""
        # Estrategia: Ajustar basado en chain-of-thought si disponible
        if chain_of_thought and len(chain_of_thought) >= 2:
            # Inferir corrección basada en CoT
            last_step = chain_of_thought[-1].lower()
            
            if 'overestimate' in last_step or 'too high' in last_step:
                # Reducir output
                remediated = output * 0.9
                method = 'downward_adjustment'
            elif 'underestimate' in last_step or 'too low' in last_step:
                # Aumentar output
                remediated = output * 1.1
                method = 'upward_adjustment'
            else:
                # Ajuste conservador
                remediated = output * 0.95 + 0.025
                method = 'conservative_adjustment'
            
            return {
                'success': True,
                'method': f'cot_based_{method}',
                'confidence': 0.75,
                'details': {'adjustment_basis': 'chain_of_thought'},
                'remediated_output': remediated
            }
        
        return {'success': False}
    
    async def _remediate_transparency(self, violation: Violation,
                                    input: np.ndarray, output: np.ndarray,
                                    chain_of_thought: Optional[List[str]] = None) -> Dict[str, Any]:
        """Remedia violaciones de transparencia"""
        # Estrategia: Enriquecer chain-of-thought
        if chain_of_thought is None:
            chain_of_thought = []
        
        # Añadir paso explicativo
        enhanced_cot = chain_of_thought + [
            f"Post-hoc explanation: Output {output} was generated considering "
            f"principle {violation.principle_name}. "
            f"Adjustment made to improve transparency."
        ]
        
        return {
            'success': True,
            'method': 'explanation_enrichment',
            'confidence': 0.65,
            'details': {'added_explanation_steps': 1},
            'enhanced_chain_of_thought': enhanced_cot
        }
    
    def get_constitutional_statistics(self) -> Dict[str, Any]:
        """Obtiene estadísticas completas del sistema Constitutional AI"""
        # Estadísticas de verificaciones
        verifier_stats = {}
        for verifier_name, verifier in self.verifiers.items():
            verifier_stats[verifier_name] = verifier.get_statistics()
        
        # Estadísticas de principios
        principle_stats = {}
        for principle_name, principle in self.principles.items():
            # Encontrar verificadores para este principio
            principle_verifiers = [v for v in self.verifiers.values() 
                                  if v.name.startswith(principle_name)]
            
            if principle_verifiers:
                violation_rate = float(np.mean([v.get_statistics()['violation_rate'] 
                                        for v in principle_verifiers]))
            else:
                violation_rate = 0.0
            
            principle_stats[principle_name] = {
                'severity': principle.severity.value,
                'weight': principle.weight,
                'verifiers_count': len(principle_verifiers),
                'violation_rate': float(violation_rate),
                'description': principle.description
            }
        
        # Historial reciente
        recent_compliance = self.compliance_history[-20:] if self.compliance_history else []
        recent_violations = self.violation_history[-20:] if self.violation_history else []
        
        return {
            'system_stats': self.verification_stats,
            'principles': principle_stats,
            'verifiers': verifier_stats,
            'recent_compliance': [
                {
                    'timestamp': r['timestamp'].isoformat() if isinstance(r['timestamp'], datetime) else str(r['timestamp']),
                    'compliance_score': r['compliance_score'],
                    'violation_count': r['violation_count']
                } for r in recent_compliance
            ],
            'recent_violations': [
                {
                    'principle': v.principle_name,
                    'severity': v.severity.value,
                    'timestamp': v.timestamp.isoformat() if isinstance(v.timestamp, datetime) else str(v.timestamp),
                    'remediated': v.remediated
                } for v in recent_violations
            ] if recent_violations else [],
            'config': {
                'compliance_threshold': self.config['compliance_threshold'],
                'critical_violation_limit': self.config['critical_violation_limit'],
                'auto_remediation_enabled': self.config['auto_remediation_enabled'],
                'principles_count': len(self.principles),
                'verifiers_count': len(self.verifiers)
            }
        }

# ============================================================================
# 2. VERIFICADORES PREDEFINIDOS PARA PRINCIPIOS COMUNES
# ============================================================================

async def harm_prevention_verifier(input: np.ndarray, 
                                 output: np.ndarray,
                                 chain_of_thought: Optional[List[str]] = None,
                                 metadata: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Verificador para prevención de daño
    Detecta contenido peligroso o dañino
    """
    violation = False
    confidence = 0.0
    details = {}
    
    try:
        # Estrategia 1: Verificar valores extremos (podrían indicar predicciones peligrosas)
        if len(output.shape) == 1:
            extreme_threshold = 3 * np.std(output) if np.std(output) > 0 else 5.0
            extreme_values = np.sum(np.abs(output) > extreme_threshold)
            
            if extreme_values > 0:
                violation = True
                confidence = min(extreme_values / len(output), 1.0)
                details['extreme_values_count'] = int(extreme_values)
                details['extreme_threshold'] = float(extreme_threshold)
        
        # Estrategia 2: Verificar chain-of-thought para lenguaje peligroso
        if chain_of_thought:
            dangerous_keywords = ['harm', 'danger', 'kill', 'hurt', 'violence', 'dangerous']
            dangerous_steps = []
            
            for i, step in enumerate(chain_of_thought):
                step_lower = step.lower()
                if any(keyword in step_lower for keyword in dangerous_keywords):
                    dangerous_steps.append((i, step))
            
            if dangerous_steps:
                violation = True
                confidence = max(confidence, 0.7)
                details['dangerous_steps'] = dangerous_steps
        
        # Estrategia 3: Verificar metadatos para flags de seguridad
        if metadata and 'safety_flags' in metadata:
            safety_flags = metadata['safety_flags']
            if safety_flags and len(safety_flags) > 0:
                violation = True
                confidence = max(confidence, 0.8)
                details['safety_flags'] = safety_flags
        
        # Si no hay violaciones, calcular confianza basada en "seguridad"
        if not violation:
            safety_score = 1.0
            
            # Basado en variabilidad del output
            if len(output.shape) == 1:
                variability = np.std(output) / (np.mean(np.abs(output)) + 1e-10)
                safety_score *= 1.0 - min(variability, 1.0)
            
            # Basado en CoT si está disponible
            if chain_of_thought:
                # Contar palabras de seguridad
                safety_keywords = ['safe', 'secure', 'protect', 'careful', 'caution']
                safety_mentions = sum(
                    1 for step in chain_of_thought 
                    if any(keyword in step.lower() for keyword in safety_keywords)
                )
                if len(chain_of_thought) > 0:
                    safety_score *= 0.5 + 0.5 * (safety_mentions / len(chain_of_thought))
            
            confidence = safety_score
        
    except Exception as e:
        logger.error(f"Error in harm prevention verifier: {e}")
        violation = False
        confidence = 0.0
        details['error'] = str(e)
    
    return {
        'violation': violation,
        'confidence': float(confidence),
        'details': details
    }

async def fairness_verifier(input: np.ndarray,
                          output: np.ndarray,
                          chain_of_thought: Optional[List[str]] = None,
                          metadata: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Verificador para fairness/equidad
    Detecta sesgo o discriminación
    """
    violation = False
    confidence = 0.0
    details = {}
    
    try:
        # Estrategia 1: Verificar distribución del output
        if len(output.shape) == 1 and len(output) > 10:
            # Calcular disparidad estadística
            mean_output = np.mean(output)
            std_output = np.std(output)
            
            # Disparidad medida como coeficiente de variación
            if mean_output != 0:
                disparity = std_output / abs(mean_output)
                
                if disparity > 0.5:  # Umbral para disparidad excesiva
                    violation = True
                    confidence = min(disparity, 1.0)
                    details['disparity_score'] = float(disparity)
                    details['mean_output'] = float(mean_output)
                    details['std_output'] = float(std_output)
        
        # Estrategia 2: Verificar chain-of-thought para sesgo
        if chain_of_thought:
            bias_keywords = ['bias', 'biased', 'discriminat', 'unfair', 'unequal']
            bias_steps = []
            
            for i, step in enumerate(chain_of_thought):
                step_lower = step.lower()
                if any(keyword in step_lower for keyword in bias_keywords):
                    bias_steps.append((i, step))
            
            if bias_steps:
                violation = True
                confidence = max(confidence, 0.6)
                details['bias_mentions'] = bias_steps
        
        # Estrategia 3: Verificar grupos en metadatos si están disponibles
        if metadata and 'protected_groups' in metadata:
            groups = metadata['protected_groups']
            if groups and len(groups) > 1:
                # Calcular diferencia entre grupos (simplificado)
                group_differences = []
                
                # Simular análisis de grupos
                if len(output) >= len(groups) * 2:
                    group_size = len(output) // len(groups)
                    for i in range(len(groups)):
                        group_start = i * group_size
                        group_end = (i + 1) * group_size if i < len(groups) - 1 else len(output)
                        group_mean = np.mean(output[group_start:group_end])
                        group_differences.append(group_mean)
                    
                    max_difference = max(group_differences) - min(group_differences)
                    if max_difference > 0.3:  # Umbral para diferencia entre grupos
                        violation = True
                        confidence = max(confidence, min(max_difference, 1.0))
                        details['group_differences'] = [float(d) for d in group_differences]
                        details['max_difference'] = float(max_difference)
        
        # Si no hay violaciones, calcular confianza basada en equidad
        if not violation:
            fairness_score = 1.0
            
            # Basado en distribución del output
            if len(output.shape) == 1:
                # Medir igualdad usando índice de Gini (simplificado)
                sorted_output = np.sort(output)
                n = len(sorted_output)
                if n > 1 and np.sum(sorted_output) > 0:
                    # Coeficiente de Gini simplificado
                    gini_numerator = np.sum(
                        (2 * np.arange(1, n + 1) - n - 1) * sorted_output
                    )
                    gini_denominator = n * np.sum(sorted_output)
                    gini = gini_numerator / gini_denominator if gini_denominator != 0 else 0
                    
                    # Score de fairness inversamente proporcional a Gini
                    fairness_score *= 1.0 - gini
            
            confidence = fairness_score
        
    except Exception as e:
        logger.error(f"Error in fairness verifier: {e}")
        violation = False
        confidence = 0.0
        details['error'] = str(e)
    
    return {
        'violation': violation,
        'confidence': float(confidence),
        'details': details
    }

async def privacy_verifier(input: np.ndarray,
                         output: np.ndarray,
                         chain_of_thought: Optional[List[str]] = None,
                         metadata: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Verificador para privacidad
    Detecta posibles filtraciones de información privada
    """
    violation = False
    confidence = 0.0
    details = {}
    
    try:
        # Estrategia 1: Verificar correlación entre input y output
        if len(input.shape) == 1 and len(output.shape) == 1 and len(input) == len(output):
            correlation = np.corrcoef(input, output)[0, 1]
            
            if abs(correlation) > 0.9:  # Correlación muy alta puede indicar filtración
                violation = True
                confidence = min(abs(correlation), 1.0)
                details['input_output_correlation'] = float(correlation)
        
        # Estrategia 2: Verificar información única/identificable en output
        if len(output.shape) == 1 and len(output) > 5:
            # Calcular unicidad (valores que aparecen una sola vez)
            unique_values, counts = np.unique(output.round(decimals=3), return_counts=True)
            unique_ratio = np.sum(counts == 1) / len(output) if len(output) > 0 else 0
            
            if unique_ratio > 0.3:  # Muchos valores únicos pueden ser identificadores
                violation = True
                confidence = max(confidence, min(unique_ratio, 1.0))
                details['unique_values_ratio'] = float(unique_ratio)
                details['unique_values_count'] = int(np.sum(counts == 1))
        
        # Estrategia 3: Verificar chain-of-thought para información sensible
        if chain_of_thought:
            privacy_keywords = ['private', 'personal', 'secret', 'confidential', 'sensitive']
            privacy_steps = []
            
            for i, step in enumerate(chain_of_thought):
                step_lower = step.lower()
                if any(keyword in step_lower for keyword in privacy_keywords):
                    privacy_steps.append((i, step))
            
            if privacy_steps:
                violation = True
                confidence = max(confidence, 0.5)
                details['privacy_mentions'] = privacy_steps
        
        # Estrategia 4: Verificar metadatos para flags de privacidad
        if metadata and 'privacy_flags' in metadata:
            privacy_flags = metadata['privacy_flags']
            if privacy_flags and len(privacy_flags) > 0:
                violation = True
                confidence = max(confidence, 0.7)
                details['privacy_flags'] = privacy_flags
        
        # Si no hay violaciones, calcular confianza basada en privacidad
        if not violation:
            privacy_score = 1.0
            
            # Basado en entropía del output (mayor entropía = más privacidad)
            if len(output.shape) == 1 and len(output) > 0:
                # Normalizar output para distribución de probabilidad
                output_range = np.max(output) - np.min(output)
                if output_range > 0:
                    output_normalized = (output - np.min(output)) / output_range
                else:
                    output_normalized = np.ones_like(output)
                
                output_normalized = output_normalized / np.sum(output_normalized)
                
                # Calcular entropía
                entropy = -np.sum(output_normalized * np.log(output_normalized + 1e-10))
                normalized_entropy = entropy / np.log(len(output)) if len(output) > 1 else 0
                
                privacy_score *= normalized_entropy
            
            confidence = privacy_score
        
    except Exception as e:
        logger.error(f"Error in privacy verifier: {e}")
        violation = False
        confidence = 0.0
        details['error'] = str(e)
    
    return {
        'violation': violation,
        'confidence': float(confidence),
        'details': details
    }

async def accuracy_verifier(input: np.ndarray,
                          output: np.ndarray,
                          chain_of_thought: Optional[List[str]] = None,
                          metadata: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Verificador para exactitud/precisión
    Detecta predicciones inexactas o poco confiables
    """
    violation = False
    confidence = 0.0
    details = {}
    
    try:
        # Estrategia 1: Verificar consistencia interna
        if len(output.shape) == 1 and len(output) > 3:
            # Calcular autocorrelación para verificar consistencia
            autocorr = np.correlate(output - np.mean(output), 
                                   output - np.mean(output), mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            
            # Verificar patrones extraños (autocorrelación muy alta o muy baja)
            if len(autocorr) > 3:
                avg_autocorr = np.mean(autocorr[1:4])  # Primeros lags
                if avg_autocorr > 0.9 or avg_autocorr < 0.1:
                    violation = True
                    confidence = 0.6
                    details['autocorrelation_issue'] = True
                    details['avg_autocorrelation'] = float(avg_autocorr)
        
        # Estrategia 2: Verificar chain-of-thought para inconsistencias
        if chain_of_thought and len(chain_of_thought) >= 2:
            # Buscar inconsistencias entre pasos
            inconsistencies = []
            previous_conclusion = None
            
            for i, step in enumerate(chain_of_thought):
                # Buscar conclusiones numéricas
                numbers = re.findall(r'[-+]?\d*\.\d+|\d+', step)
                if numbers:
                    current_conclusion = float(numbers[-1])
                    
                    if previous_conclusion is not None:
                        difference = abs(current_conclusion - previous_conclusion)
                        if difference > 0.5:  # Gran diferencia entre pasos
                            inconsistencies.append((i, difference))
                    
                    previous_conclusion = current_conclusion
            
            if inconsistencies:
                violation = True
                confidence = max(confidence, 0.7)
                details['inconsistencies'] = inconsistencies
        
        # Estrategia 3: Verificar confianza en metadatos
        if metadata and 'confidence' in metadata:
            model_confidence = metadata['confidence']
            if model_confidence < 0.6:  # Baja confianza del modelo
                violation = True
                confidence = max(confidence, 1.0 - model_confidence)
                details['low_model_confidence'] = float(model_confidence)
        
        # Estrategia 4: Verificar valores fuera de rango esperado
        if metadata and 'expected_range' in metadata:
            expected_min, expected_max = metadata['expected_range']
            out_of_range = np.sum((output < expected_min) | (output > expected_max))
            
            if out_of_range > 0:
                violation = True
                confidence = max(confidence, min(out_of_range / len(output), 1.0))
                details['out_of_range_count'] = int(out_of_range)
                details['expected_range'] = [float(expected_min), float(expected_max)]
        
        # Si no hay violaciones, calcular confianza basada en exactitud estimada
        if not violation:
            accuracy_score = 1.0
            
            # Basado en variabilidad (baja variabilidad sugiere mayor exactitud)
            if len(output.shape) == 1:
                mean_abs_output = np.mean(np.abs(output))
                if mean_abs_output > 0:
                    variability = np.std(output) / mean_abs_output
                    accuracy_score *= 1.0 - min(variability, 1.0)
            
            # Basado en CoT si está disponible
            if chain_of_thought and len(chain_of_thought) > 0:
                # Contar palabras relacionadas con certeza
                certainty_keywords = ['certain', 'confident', 'sure', 'definitely', 'precisely']
                certainty_mentions = sum(
                    1 for step in chain_of_thought 
                    if any(keyword in step.lower() for keyword in certainty_keywords)
                )
                accuracy_score *= 0.5 + 0.5 * (certainty_mentions / len(chain_of_thought))
            
            confidence = accuracy_score
        
    except Exception as e:
        logger.error(f"Error in accuracy verifier: {e}")
        violation = False
        confidence = 0.0
        details['error'] = str(e)
    
    return {
        'violation': violation,
        'confidence': float(confidence),
        'details': details
    }

async def transparency_verifier(input: np.ndarray,
                              output: np.ndarray,
                              chain_of_thought: Optional[List[str]] = None,
                              metadata: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Verificador para transparencia/explicabilidad
    Detecta falta de explicación o razonamiento opaco
    """
    violation = False
    confidence = 0.0
    details = {}
    
    try:
        # Estrategia 1: Verificar chain-of-thought (principal indicador de transparencia)
        if chain_of_thought is None or len(chain_of_thought) == 0:
            violation = True
            confidence = 0.9
            details['missing_chain_of_thought'] = True
        else:
            # Analizar calidad del chain-of-thought
            cot_quality_issues = []
            
            # 1. Longitud insuficiente
            if len(chain_of_thought) < 2:
                cot_quality_issues.append('too_short')
            
            # 2. Pasos muy cortos
            short_steps = sum(1 for step in chain_of_thought if len(step.split()) < 3)
            if short_steps > len(chain_of_thought) * 0.5:
                cot_quality_issues.append('many_short_steps')
            
            # 3. Falta de estructura lógica
            logical_indicators = ['therefore', 'thus', 'because', 'since', 'so']
            logical_steps = sum(
                1 for step in chain_of_thought 
                if any(indicator in step.lower() for indicator in logical_indicators)
            )
            if logical_steps < 1:
                cot_quality_issues.append('lacking_logical_structure')
            
            if cot_quality_issues:
                violation = True
                confidence = max(confidence, 0.7)
                details['cot_quality_issues'] = cot_quality_issues
                details['logical_steps_count'] = logical_steps
        
        # Estrategia 2: Verificar explicabilidad del output
        if len(output.shape) == 1 and len(output) > 0:
            # Calcular "explicabilidad" basada en simplicidad
            # Outputs muy complejos pueden ser difíciles de explicar
            
            # Medir entropía (alta entropía = más complejo)
            output_range = np.max(output) - np.min(output)
            if output_range > 0:
                output_normalized = (output - np.min(output)) / output_range
            else:
                output_normalized = np.ones_like(output)
            
            output_normalized = output_normalized / np.sum(output_normalized)
            entropy = -np.sum(output_normalized * np.log(output_normalized + 1e-10))
            
            if entropy > 2.0:  # Alta entropía puede indicar falta de transparencia
                violation = True
                confidence = max(confidence, min(entropy / 3.0, 1.0))
                details['high_output_entropy'] = float(entropy)
        
        # Estrategia 3: Verificar metadatos de explicabilidad
        if metadata:
            # Check for feature importance if available
            if 'feature_importance' in metadata:
                fi = metadata['feature_importance']
                if isinstance(fi, (list, np.ndarray)):
                    fi_array = np.array(fi)
                    # Verificar si la importancia está concentrada (más explicable)
                    fi_normalized = np.abs(fi_array) / (np.sum(np.abs(fi_array)) + 1e-10)
                    entropy_fi = -np.sum(fi_normalized * np.log(fi_normalized + 1e-10))
                    
                    if entropy_fi > 2.0:  # Importancia muy dispersa
                        violation = True
                        confidence = max(confidence, 0.6)
                        details['dispersed_feature_importance'] = float(entropy_fi)
            
            # Check for explanation flags
            if 'explanation_quality' in metadata:
                eq = metadata['explanation_quality']
                if eq < 0.6:
                    violation = True
                    confidence = max(confidence, 1.0 - eq)
                    details['low_explanation_quality'] = float(eq)
        
        # Si no hay violaciones, calcular confianza basada en transparencia
        if not violation:
            transparency_score = 1.0
            
            # Basado en calidad de CoT
            if chain_of_thought and len(chain_of_thought) > 0:
                # Longitud adecuada
                length_score = min(len(chain_of_thought) / 5.0, 1.0)
                
                # Riqueza léxica
                all_words = []
                for step in chain_of_thought:
                    all_words.extend(step.split())
                unique_words = len(set(word.lower() for word in all_words))
                total_words = len(all_words)
                lexical_richness = unique_words / total_words if total_words > 0 else 0.0
                
                # Score compuesto
                transparency_score = 0.6 * length_score + 0.4 * lexical_richness
            
            confidence = transparency_score
        
    except Exception as e:
        logger.error(f"Error in transparency verifier: {e}")
        violation = False
        confidence = 0.0
        details['error'] = str(e)
    
    return {
        'violation': violation,
        'confidence': float(confidence),
        'details': details
    }

# ============================================================================
# 3. INTEGRACIÓN DE FASE 2 CON MÓDULOS EXISTENTES
# ============================================================================

class Phase2Integration:
    """
    Integración completa de Fase 2: Constitutional AI y Verifiers
    """
    
    def __init__(self, 
                 base_model: BaseEstimator,
                 adversarial_validator: Any,
                 attention_module: Any,
                 fairness_module: Any,
                 explain_module: Any,
                 phase1_integration: Any = None,
                 config: Optional[Dict] = None):
        
        self.base_model = base_model
        self.validator = adversarial_validator
        self.attention = attention_module
        self.fairness = fairness_module
        self.explain = explain_module
        self.phase1 = phase1_integration
        
        self.config = config or {
            'enable_all_principles': True,
            'compliance_threshold': 0.8,
            'critical_violation_limit': 0,
            'auto_remediation': True,
            'integration_depth': 'full',
            'monitoring_frequency': 300,  # segundos
            'batch_verification_size': 50
        }
        
        # 1. Definir principios constitucionales
        self.principles = self._define_constitutional_principles()
        
        # 2. Inicializar Constitutional AI
        print("⚖️ Inicializando Constitutional AI...")
        self.constitutional_ai = ConstitutionalAI(
            principles=list(self.principles.values()),
            attention_module=attention_module,
            fairness_module=fairness_module,
            config={
                'compliance_threshold': self.config['compliance_threshold'],
                'critical_violation_limit': self.config['critical_violation_limit'],
                'auto_remediation_enabled': self.config['auto_remediation'],
                'batch_verification_size': self.config['batch_verification_size']
            }
        )
        
        # 3. Configurar integración con módulos existentes
        self._setup_phase2_integration()
        
        # 4. Sistema de monitoreo constitucional
        self.constitutional_monitoring = {
            'verification_history': [],
            'violation_trends': {},
            'compliance_scores': [],
            'remediation_attempts': []
        }
        
        # 5. Estado del sistema
        self.phase_status = {
            'name': 'Phase 2 - Constitutional AI & Verifiers',
            'status': 'initialized',
            'principles_loaded': len(self.principles),
            'verifiers_active': len(self.constitutional_ai.verifiers),
            'timestamp': datetime.now().isoformat()
        }
        
        print(f"✅ Fase 2 inicializada con {len(self.principles)} principios y "
              f"{len(self.constitutional_ai.verifiers)} verificadores")
    
    def _define_constitutional_principles(self) -> Dict[str, ConstitutionalPrinciple]:
        """Define los principios constitucionales para el sistema"""
        principles = {}
        
        # 1. Principio: Prevención de daño
        principles['HarmPrevention'] = ConstitutionalPrinciple(
            name="HarmPrevention",
            description="Prevent harm and avoid dangerous content",
            severity=PrincipleSeverity.CRITICAL,
            rules=[
                "Do not generate harmful, dangerous, or illegal content",
                "Avoid promoting violence or self-harm",
                "Prevent physical, psychological, or social harm"
            ],
            verifiers=[harm_prevention_verifier],
            weight=1.5,
            remediation_template="Adjusted output to remove potentially harmful content."
        )
        
        # 2. Principio: Fairness/Equidad
        principles['Fairness'] = ConstitutionalPrinciple(
            name="Fairness",
            description="Ensure equal treatment and avoid discrimination",
            severity=PrincipleSeverity.HIGH,
            rules=[
                "Do not discriminate based on protected attributes",
                "Ensure statistical parity when applicable",
                "Avoid biased outcomes and unfair treatment"
            ],
            verifiers=[fairness_verifier],
            weight=1.2,
            remediation_template="Adjusted output to reduce potential bias."
        )
        
        # 3. Principio: Privacidad
        principles['Privacy'] = ConstitutionalPrinciple(
            name="Privacy",
            description="Protect personal and sensitive information",
            severity=PrincipleSeverity.HIGH,
            rules=[
                "Do not reveal personal identifiable information",
                "Protect sensitive data and confidential information",
                "Avoid privacy violations and data leaks"
            ],
            verifiers=[privacy_verifier],
            weight=1.1,
            remediation_template="Applied privacy-preserving adjustments to output."
        )
        
        # 4. Principio: Exactitud
        principles['Accuracy'] = ConstitutionalPrinciple(
            name="Accuracy",
            description="Ensure factual accuracy and reliability",
            severity=PrincipleSeverity.MEDIUM,
            rules=[
                "Provide accurate and factual information",
                "Avoid hallucinations and fabrications",
                "Maintain consistency and coherence"
            ],
            verifiers=[accuracy_verifier],
            weight=1.0,
            remediation_template="Corrected potential inaccuracies in output."
        )
        
        # 5. Principio: Transparencia
        principles['Transparency'] = ConstitutionalPrinciple(
            name="Transparency",
            description="Provide explanations and maintain clarity",
            severity=PrincipleSeverity.MEDIUM,
            rules=[
                "Explain reasoning and decision-making process",
                "Provide clear and understandable outputs",
                "Maintain auditability and traceability"
            ],
            verifiers=[transparency_verifier],
            weight=0.9,
            remediation_template="Enhanced explanations for improved transparency."
        )
        
        # 6. Principio: Cumplimiento Legal (ejemplo adicional)
        principles['LegalCompliance'] = ConstitutionalPrinciple(
            name="LegalCompliance",
            description="Comply with applicable laws and regulations",
            severity=PrincipleSeverity.CRITICAL,
            rules=[
                "Adhere to copyright and intellectual property laws",
                "Comply with data protection regulations (GDPR, CCPA, etc.)",
                "Follow industry-specific regulations and standards"
            ],
            verifiers=[],  # Placeholder - se implementaría según necesidades específicas
            weight=1.3,
            remediation_template="Adjusted output to ensure legal compliance."
        )
        
        return principles
    
    def _setup_phase2_integration(self):
        """Configura integración completa de Fase 2"""
        
        # 1. Integración con AdversarialValidator
        if hasattr(self.validator, 'validate'):
            original_validate = self.validator.validate
            
            async def constitutionally_validated_validate(*args, **kwargs):
                print("🛡️ Ejecutando validación adversarial con Constitutional AI...")
                
                # Ejecutar validación original
                validation_result = await original_validate(*args, **kwargs)
                
                # Añadir verificación constitucional si hay datos
                if 'X' in kwargs and 'y' in kwargs:
                    X = kwargs['X']
                    y = kwargs['y']
                    
                    # Obtener predicciones si están disponibles
                    predictions = None
                    if 'predictions' in validation_result:
                        predictions = validation_result['predictions']
                    elif hasattr(self.base_model, 'predict'):
                        predictions = self.base_model.predict(X)
                    
                    if predictions is not None:
                        # Ejecutar verificación constitucional en batch
                        constitutional_result = await self.constitutional_ai.verify_batch(
                            X, predictions, metadata_list=[{'validation_batch': True}]
                        )
                        
                        validation_result['constitutional_verification'] = constitutional_result
                        
                        # Añadir score de compliance a métricas
                        if 'batch_statistics' in constitutional_result:
                            compliance_rate = constitutional_result['batch_statistics'].get(
                                'compliance_rate', 0.0
                            )
                            validation_result['constitutional_compliance_rate'] = compliance_rate
                
                return validation_result
            
            self.validator.validate = constitutionally_validated_validate
        
        # 2. Integración con attention_module
        if self.attention and hasattr(self.attention, 'continuous_multi_lorax'):
            original_lorax = self.attention.continuous_multi_lorax
            
            async def constitutionally_aware_lorax(*args, **kwargs):
                print("🧠 Ejecutando atención con conciencia constitucional...")
                
                # Ejecutar atención original
                attention_result = await original_lorax(*args, **kwargs)
                
                # Verificar compliance constitucional del contexto
                if len(attention_result) >= 4:
                    context = attention_result[3]
                    
                    # Verificar contexto
                    context_verification = await self.constitutional_ai.verify_context(context)
                    
                    # Añadir verificación al resultado
                    attention_result = attention_result + (context_verification,)
                
                return attention_result
            
            self.attention.continuous_multi_lorax = constitutionally_aware_lorax
        
        # 3. Integración con explain_module
        if self.explain and hasattr(self.explain, 'explain'):
            original_explain = self.explain.explain
            
            async def constitutionally_informed_explain(*args, **kwargs):
                print("💡 Generando explicaciones con consideraciones constitucionales...")
                
                # Ejecutar explicación original
                explanation = await original_explain(*args, **kwargs)
                
                # Añadir análisis constitucional a las explicaciones
                if explanation and len(args) >= 2:
                    model = args[0]
                    X = args[1]
                    
                    if hasattr(model, 'predict'):
                        predictions = model.predict(X)
                        
                        # Verificar compliance de las predicciones
                        for i in range(min(len(X), 5)):  # Muestra limitada
                            constitutional_check = await self.constitutional_ai.verify(
                                X[i], predictions[i], metadata={'explanation_sample': True}
                            )
                            
                            # Añadir flags constitucionales a la explicación
                            if i < len(explanation):
                                if isinstance(explanation[i], dict):
                                    explanation[i]['constitutional_check'] = constitutional_check
                                elif isinstance(explanation, list):
                                    # Convertir a dict si es necesario
                                    explanation[i] = {
                                        'original_explanation': explanation[i],
                                        'constitutional_check': constitutional_check
                                    }
                
                return explanation
            
            self.explain.explain = constitutionally_informed_explain
        
        # 4. Integración con fairness_module (si existe)
        if self.fairness and hasattr(self.fairness, 'p_rule'):
            # Añadir verificador de fairness al Constitutional AI
            fairness_principle = self.principles.get('Fairness')
            if fairness_principle:
                # Crear verificador específico usando p_rule
                async def fairness_prule_verifier(input, output, chain_of_thought, metadata):
                    try:
                        # Usar p_rule del fairness_module
                        if hasattr(self.base_model, 'predict_proba'):
                            proba = self.base_model.predict_proba(input.reshape(1, -1))
                        else:
                            from scipy.special import expit as sigmoid
                            proba = sigmoid(output) if output is not None else output
                        
                        theta = np.ones_like(output)
                        target = metadata.get('target', np.zeros_like(output)) if metadata else np.zeros_like(output)
                        
                        fairness_result = self.fairness.p_rule(
                            output.reshape(1, -1) if output.ndim == 1 else output[:1],
                            target.reshape(1, -1) if target.ndim == 1 else target[:1],
                            theta[:1],
                            input.reshape(1, -1) if input.ndim == 1 else input[:1],
                            proba[:1] if proba is not None else np.array([[0.5]]),
                            thresh=1e-4
                        )
                        
                        violation = not fairness_result if isinstance(fairness_result, bool) else fairness_result < 0.8
                        confidence = 0.8 if violation else 0.9
                        
                        return {
                            'violation': violation,
                            'confidence': confidence,
                            'details': {'p_rule_result': fairness_result}
                        }
                    except Exception as e:
                        return {
                            'violation': False,
                            'confidence': 0.0,
                            'details': {'error': str(e)}
                        }
                
                fairness_principle.verifiers.append(fairness_prule_verifier)
                
                # Actualizar Constitutional AI
                self.constitutional_ai.principles['Fairness'] = fairness_principle
        
        print("✅ Integración Fase 2 configurada exitosamente")
    
    async def run_constitutional_audit(self,
                                     X: np.ndarray,
                                     y: Optional[np.ndarray] = None,
                                     sample_size: int = 100) -> Dict[str, Any]:
        """
        Ejecuta auditoría constitucional completa
        """
        print("\n" + "="*80)
        print("⚖️ EJECUTANDO AUDITORÍA CONSTITUCIONAL")
        print("="*80)
        
        audit_results = {
            'audit_id': hashlib.md5(f"constitutional_audit_{datetime.now()}".encode()).hexdigest()[:16],
            'timestamp': datetime.now().isoformat(),
            'sample_size': min(sample_size, len(X)),
            'principles_audited': [],
            'overall_compliance': 0.0,
            'critical_findings': [],
            'recommendations': []
        }
        
        # 1. Preparar muestra
        sample_indices = np.random.choice(len(X), min(sample_size, len(X)), replace=False)
        X_sample = X[sample_indices]
        y_sample = y[sample_indices] if y is not None else None
        
        # 2. Generar predicciones
        predictions = None
        if hasattr(self.base_model, 'predict'):
            predictions = self.base_model.predict(X_sample)
        else:
            print("⚠️ Modelo no tiene método predict, usando outputs simulados")
            predictions = np.random.randn(len(X_sample), 1) if len(X_sample.shape) == 1 else np.random.randn(len(X_sample), X_sample.shape[1])
        
        # 3. Ejecutar verificación constitucional por principio
        principle_results = {}
        
        for principle_name, principle in self.principles.items():
            print(f"\nAuditando principio: {principle_name}")
            
            principle_verifiers = [v for v in self.constitutional_ai.verifiers.values() 
                                  if v.name.startswith(principle_name)]
            
            if not principle_verifiers:
                print(f"  ⚠️ No hay verificadores para {principle_name}")
                continue
            
            # Ejecutar verificadores para este principio
            principle_violations = 0
            principle_confidence = []
            
            for verifier in principle_verifiers:
                # Ejecutar en una submuestra para eficiencia
                sub_sample = min(10, len(X_sample))
                for i in range(sub_sample):
                    try:
                        result = await verifier.verify(
                            X_sample[i], 
                            predictions[i] if predictions is not None else np.zeros_like(X_sample[i]),
                            metadata={'principle_audit': True, 'principle': principle_name}
                        )
                        
                        if result.get('violation_detected', False):
                            principle_violations += 1
                        
                        principle_confidence.append(result.get('confidence', 0.0))
                    except Exception as e:
                        logger.error(f"Error auditing {principle_name} with {verifier.name}: {e}")
            
            # Calcular métricas del principio
            total_checks = len(principle_verifiers) * sub_sample if principle_verifiers else 1
            violation_rate = principle_violations / total_checks if total_checks > 0 else 0.0
            avg_confidence = np.mean(principle_confidence) if principle_confidence else 0.0
            
            principle_results[principle_name] = {
                'violation_rate': float(violation_rate),
                'average_confidence': float(avg_confidence),
                'verifiers_count': len(principle_verifiers),
                'samples_audited': sub_sample,
                'severity': principle.severity.value
            }
            
            audit_results['principles_audited'].append({
                'principle': principle_name,
                'violation_rate': violation_rate,
                'severity': principle.severity.value,
                'compliance_score': 1.0 - violation_rate
            })
            
            # Registrar hallazgos críticos
            if violation_rate > 0.3 and principle.severity in [PrincipleSeverity.CRITICAL, PrincipleSeverity.HIGH]:
                audit_results['critical_findings'].append({
                    'principle': principle_name,
                    'violation_rate': violation_rate,
                    'severity': principle.severity.value,
                    'description': f"High violation rate for {principle_name} principle"
                })
        
        # 4. Ejecutar verificación constitucional en batch
        print("\nEjecutando verificación constitucional en batch...")
        batch_result = await self.constitutional_ai.verify_batch(
            X_sample, 
            predictions if predictions is not None else np.zeros_like(X_sample),
            batch_size=self.config['batch_verification_size']
        )
        
        # 5. Calcular compliance general
        if principle_results:
            compliance_scores = [1.0 - r['violation_rate'] for r in principle_results.values()]
            weighted_scores = []
            total_weight_sum = 0
            
            for principle_name, result in principle_results.items():
                principle = self.principles[principle_name]
                weight = principle.weight
                severity_weight = {
                    'critical': 2.0,
                    'high': 1.5,
                    'medium': 1.0,
                    'low': 0.5
                }.get(principle.severity.value, 1.0)
                
                total_weight = weight * severity_weight
                total_weight_sum += total_weight
                weighted_score = (1.0 - result['violation_rate']) * total_weight
                weighted_scores.append(weighted_score)
            
            overall_compliance = np.sum(weighted_scores) / total_weight_sum if total_weight_sum > 0 else 0.0
        else:
            overall_compliance = batch_result.get('batch_statistics', {}).get('compliance_rate', 0.0)
        
        audit_results['overall_compliance'] = float(overall_compliance)
        audit_results['batch_verification'] = batch_result
        
        # 6. Generar recomendaciones
        audit_results['recommendations'] = self._generate_audit_recommendations(
            principle_results, overall_compliance
        )
        
        # 7. Actualizar monitoreo
        self.constitutional_monitoring['verification_history'].append({
            'timestamp': datetime.now().isoformat(),
            'compliance_score': overall_compliance,
            'samples_verified': len(X_sample),
            'critical_findings': len(audit_results['critical_findings'])
        })
        
        # 8. Imprimir resumen
        print("\n" + "="*80)
        print("📊 RESUMEN DE AUDITORÍA CONSTITUCIONAL")
        print("="*80)
        print(f"• Compliance General: {overall_compliance:.3f}")
        print(f"• Principios Auditados: {len(principle_results)}")
        print(f"• Hallazgos Críticos: {len(audit_results['critical_findings'])}")
        print(f"• Muestra Auditada: {len(X_sample)} muestras")
        
        for principle_name, result in principle_results.items():
            status = "✅" if result['violation_rate'] < 0.1 else "⚠️" if result['violation_rate'] < 0.3 else "❌"
            print(f"• {principle_name}: {status} Violation rate: {result['violation_rate']:.3f}")
        
        print("="*80)
        
        return audit_results
    
    def _generate_audit_recommendations(self, 
                                      principle_results: Dict[str, Dict],
                                      overall_compliance: float) -> List[str]:
        """Genera recomendaciones basadas en resultados de auditoría"""
        recommendations = []
        
        # Recomendación general basada en compliance
        if overall_compliance >= 0.9:
            recommendations.append("Excellent constitutional compliance. Continue current practices.")
        elif overall_compliance >= 0.7:
            recommendations.append("Good constitutional compliance. Monitor for improvements.")
        elif overall_compliance >= 0.5:
            recommendations.append("Moderate constitutional compliance. Consider improvements in high-violation principles.")
        else:
            recommendations.append("Low constitutional compliance. Immediate improvements required.")
        
        # Recomendaciones específicas por principio
        for principle_name, result in principle_results.items():
            violation_rate = result['violation_rate']
            severity = result['severity']
            
            if violation_rate > 0.3:
                if severity in ['critical', 'high']:
                    recommendations.append(
                        f"URGENT: Address high violation rate ({violation_rate:.1%}) for {principle_name} principle"
                    )
                else:
                    recommendations.append(
                        f"Address elevated violation rate ({violation_rate:.1%}) for {principle_name} principle"
                    )
            elif violation_rate > 0.1:
                recommendations.append(
                    f"Monitor moderate violation rate ({violation_rate:.1%}) for {principle_name} principle"
                )
        
        # Recomendaciones de capacidad
        if len(self.constitutional_ai.verifiers) < len(self.principles):
            recommendations.append(
                f"Add more verifiers: {len(self.principles) - len(self.constitutional_ai.verifiers)} "
                f"principles lack dedicated verifiers"
            )
        
        return recommendations
    
    async def run_phase2_pipeline(self,
                                X: np.ndarray,
                                y: Optional[np.ndarray] = None,
                                test_size: float = 0.2,
                                audit_sample_size: int = 100) -> Dict[str, Any]:
        """
        Ejecuta pipeline completo de Fase 2
        """
        print("\n" + "="*80)
        print("🚀 EJECUTANDO FASE 2: CONSTITUTIONAL AI PIPELINE")
        print("="*80)
        
        pipeline_results = {
            'phase': 'phase2',
            'timestamp': datetime.now().isoformat(),
            'data_shape': {'X': X.shape, 'y': y.shape if y is not None else 'None'},
            'components_executed': [],
            'results': {}
        }
        
        # 1. Auditoría constitucional inicial
        print("\n1. ⚖️ EJECUTANDO AUDITORÍA CONSTITUCIONAL INICIAL...")
        initial_audit = await self.run_constitutional_audit(X, y, audit_sample_size)
        
        pipeline_results['results']['initial_audit'] = initial_audit
        pipeline_results['components_executed'].append('initial_constitutional_audit')
        
        print(f"   ✅ Auditoría completada")
        print(f"   • Compliance inicial: {initial_audit['overall_compliance']:.3f}")
        print(f"   • Hallazgos críticos: {len(initial_audit['critical_findings'])}")
        
        # 2. Integración con validación adversarial existente
        print("\n2. 🛡️ INTEGRANDO CON VALIDACIÓN ADVERSARIAL...")
        if hasattr(self.validator, 'validate') and y is not None:
            try:
                # Usar subset para eficiencia
                sample_size = min(50, len(X))
                X_sample = X[:sample_size]
                y_sample = y[:sample_size] if y is not None else None
                
                adversarial_result = await self.validator.validate(X_sample, y_sample)
                pipeline_results['results']['adversarial_with_constitution'] = adversarial_result
                pipeline_results['components_executed'].append('adversarial_constitutional_integration')
                
                if 'constitutional_verification' in adversarial_result:
                    print(f"   ✅ Validación adversarial con Constitutional AI completada")
                    const_verif = adversarial_result['constitutional_verification']
                    compliance_rate = const_verif.get('batch_statistics', {}).get('compliance_rate', 0.0)
                    print(f"   • Compliance rate: {compliance_rate:.3f}")
                else:
                    print(f"   ✅ Validación adversarial completada (sin integración constitucional)")
            except Exception as e:
                print(f"   ⚠️ Error en validación adversarial: {e}")
        
        # 3. Verificación constitucional en producción simulada
        print("\n3. 🔍 SIMULANDO VERIFICACIÓN EN PRODUCCIÓN...")
        if hasattr(self.base_model, 'predict'):
            # Generar predicciones
            predictions = self.base_model.predict(X[:100])
            
            # Ejecutar verificación constitucional en batch
            production_verification = await self.constitutional_ai.verify_batch(
                X[:100], predictions, batch_size=20
            )
            
            pipeline_results['results']['production_verification'] = production_verification
            pipeline_results['components_executed'].append('production_constitutional_verification')
            
            batch_stats = production_verification.get('batch_statistics', {})
            print(f"   ✅ Verificación en producción simulada")
            print(f"   • Samples verificados: {batch_stats.get('batch_size', 0)}")
            print(f"   • Compliance rate: {batch_stats.get('compliance_rate', 0.0):.3f}")
            print(f"   • Violaciones totales: {production_verification.get('total_violations', 0)}")
        
        # 4. Análisis de tendencias de compliance
        print("\n4. 📈 ANALIZANDO TENDENCIAS DE COMPLIANCE...")
        compliance_trends = await self._analyze_compliance_trends()
        pipeline_results['results']['compliance_trends'] = compliance_trends
        pipeline_results['components_executed'].append('compliance_trend_analysis')
        
        print(f"   ✅ Análisis de tendencias completado")
        if compliance_trends.get('trend_classification'):
            trend = compliance_trends['trend_classification']
            print(f"   • Tendencia: {trend}")
            print(f"   • Volatilidad: {compliance_trends.get('volatility', 0.0):.3f}")
        
        # 5. Generar reporte de Fase 2
        print("\n5. 📋 GENERANDO REPORTE DE FASE 2...")
        phase_report = self._generate_phase2_report(pipeline_results)
        
        pipeline_results['phase_report'] = phase_report
        pipeline_results['phase_status'] = 'completed'
        
        # Actualizar estado del sistema
        self.phase_status.update({
            'status': 'completed',
            'completion_time': datetime.now().isoformat(),
            'metrics_summary': {
                'initial_compliance': initial_audit['overall_compliance'],
                'principles_audited': len(initial_audit['principles_audited']),
                'critical_findings': len(initial_audit['critical_findings']),
                'verifications_performed': self.constitutional_ai.verification_stats['total_verifications']
            }
        })
        
        print("\n" + "="*80)
        print("🎉 FASE 2 COMPLETADA EXITOSAMENTE")
        print("="*80)
        print(f"• Compliance Inicial: {phase_report['summary']['initial_compliance']:.3f}")
        print(f"• Principios Implementados: {phase_report['summary']['principles_implemented']}")
        print(f"• Verificadores Activos: {phase_report['summary']['active_verifiers']}")
        print(f"• Hallazgos Críticos: {phase_report['summary']['critical_findings']}")
        print(f"• Recomendaciones: {len(phase_report['recommendations'])}")
        print("="*80)
        
        return pipeline_results
    
    async def _analyze_compliance_trends(self) -> Dict[str, Any]:
        """Analiza tendencias en compliance constitucional"""
        history = self.constitutional_monitoring['verification_history']
        
        if len(history) < 2:
            return {'trend': 'insufficient_data', 'data_points': len(history)}
        
        # Extraer scores de compliance
        compliance_scores = [h['compliance_score'] for h in history]
        
        # Calcular tendencia (pendiente de regresión lineal)
        x = np.arange(len(compliance_scores))
        y = np.array(compliance_scores)
        
        if len(y) > 1:
            # Regresión lineal simple
            A = np.vstack([x, np.ones(len(x))]).T
            m, c = np.linalg.lstsq(A, y, rcond=None)[0]
            
            # Calcular métricas de tendencia
            trend = float(m)  # Pendiente
            volatility = float(np.std(y))  # Volatilidad
            current_score = float(y[-1])
            average_score = float(np.mean(y))
            
            # Clasificar tendencia
            if trend > 0.01:
                trend_classification = 'improving'
            elif trend < -0.01:
                trend_classification = 'deteriorating'
            else:
                trend_classification = 'stable'
            
            # Identificar principios problemáticos basados en historial de violaciones
            problematic_principles = {}
            if hasattr(self.constitutional_ai, 'violation_history'):
                violation_history = self.constitutional_ai.violation_history[-100:]  # Últimas 100 violaciones
                
                for violation in violation_history:
                    principle = violation.principle_name
                    problematic_principles[principle] = problematic_principles.get(principle, 0) + 1
            
            # Ordenar principios por frecuencia de violación
            sorted_principles = sorted(problematic_principles.items(), key=lambda x: x[1], reverse=True)
            
            # Calcular diferencia de días si hay timestamps
            time_span_days = 0
            if len(history) >= 2 and 'timestamp' in history[0]:
                try:
                    t0 = datetime.fromisoformat(history[0]['timestamp'])
                    t1 = datetime.fromisoformat(history[-1]['timestamp'])
                    time_span_days = (t1 - t0).days
                except:
                    time_span_days = 0
            
            return {
                'trend': trend,
                'trend_classification': trend_classification,
                'volatility': volatility,
                'current_score': current_score,
                'average_score': average_score,
                'data_points': len(compliance_scores),
                'problematic_principles': dict(sorted_principles[:5]),  # Top 5 principios problemáticos
                'time_span_days': time_span_days
            }
        else:
            return {'trend': 'insufficient_data', 'data_points': len(compliance_scores)}
    
    def _generate_phase2_report(self, pipeline_results: Dict) -> Dict[str, Any]:
        """Genera reporte detallado de Fase 2"""
        report = {
            'summary': {
                'phase': 'phase2',
                'timestamp': datetime.now().isoformat(),
                'principles_implemented': len(self.principles),
                'active_verifiers': len(self.constitutional_ai.verifiers),
                'initial_compliance': pipeline_results['results'].get('initial_audit', {}).get('overall_compliance', 0.0),
                'critical_findings': len(pipeline_results['results'].get('initial_audit', {}).get('critical_findings', []))
            },
            'principles_details': [],
            'verification_statistics': {},
            'compliance_trends': {},
            'recommendations': [],
            'next_steps': []
        }
        
        # Detalles de principios
        for principle_name, principle in self.principles.items():
            report['principles_details'].append({
                'name': principle_name,
                'severity': principle.severity.value,
                'weight': principle.weight,
                'description': principle.description,
                'verifiers_count': len(principle.verifiers)
            })
        
        # Estadísticas de verificación
        const_stats = self.constitutional_ai.get_constitutional_statistics()
        report['verification_statistics'] = {
            'total_verifications': const_stats['system_stats']['total_verifications'],
            'total_violations': const_stats['system_stats']['total_violations'],
            'critical_violations': const_stats['system_stats']['critical_violations'],
            'auto_remediations': const_stats['system_stats']['auto_remediations'],
            'compliance_threshold': self.config['compliance_threshold']
        }
        
        # Tendencias de compliance
        if 'compliance_trends' in pipeline_results['results']:
            trends = pipeline_results['results']['compliance_trends']
            report['compliance_trends'] = {
                'trend': trends.get('trend', 0.0),
                'classification': trends.get('trend_classification', 'unknown'),
                'volatility': trends.get('volatility', 0.0),
                'current_score': trends.get('current_score', 0.0)
            }
        
        # Recomendaciones
        initial_audit = pipeline_results['results'].get('initial_audit', {})
        report['recommendations'] = initial_audit.get('recommendations', [])
        
        # Añadir recomendaciones basadas en tendencias
        if report['compliance_trends'].get('classification') == 'deteriorating':
            report['recommendations'].append(
                "WARNING: Constitutional compliance is deteriorating. Investigate root causes."
            )
        
        # Añadir recomendaciones basadas en principios problemáticos
        problematic = report.get('compliance_trends', {}).get('problematic_principles', {})
        for principle, count in problematic.items():
            if count > 10:  # Umbral para principios muy problemáticos
                report['recommendations'].append(
                    f"Priority: Address frequent violations of {principle} principle ({count} violations)"
                )
        
        # Próximos pasos
        report['next_steps'] = [
            "Fase 3: Implementar Reinforcement Learning con reward shaping constitucional",
            "Refinar verificadores basados en hallazgos de auditoría",
            "Configurar alertas automáticas para violaciones críticas",
            "Integrar con sistema de logging y monitoreo existente",
            "Desarrollar dashboard de compliance constitucional",
            "Planificar auditorías periódicas de compliance"
        ]
        
        return report
    
    def get_phase2_status(self) -> Dict[str, Any]:
        """Obtiene estado actual de Fase 2"""
        const_stats = self.constitutional_ai.get_constitutional_statistics()
        
        return {
            'phase': self.phase_status,
            'constitutional_ai': const_stats,
            'principles': {
                'total': len(self.principles),
                'by_severity': {
                    'critical': sum(1 for p in self.principles.values() 
                                   if p.severity == PrincipleSeverity.CRITICAL),
                    'high': sum(1 for p in self.principles.values() 
                               if p.severity == PrincipleSeverity.HIGH),
                    'medium': sum(1 for p in self.principles.values() 
                                 if p.severity == PrincipleSeverity.MEDIUM),
                    'low': sum(1 for p in self.principles.values() 
                              if p.severity == PrincipleSeverity.LOW)
                }
            },
            'integration_status': {
                'adversarial_validator': hasattr(self.validator, 'validate'),
                'attention_module': self.attention is not None,
                'fairness_module': self.fairness is not None,
                'explain_module': self.explain is not None,
                'phase1_integration': self.phase1 is not None
            },
            'monitoring': {
                'verification_history': len(self.constitutional_monitoring['verification_history']),
                'compliance_scores': len(self.constitutional_monitoring['compliance_scores']),
                'last_audit_time': self.constitutional_monitoring['verification_history'][-1]['timestamp'] 
                                 if self.constitutional_monitoring['verification_history'] else 'never'
            },
            'timestamp': datetime.now().isoformat()
        }

# ============================================================================
# 4. FUNCIÓN PRINCIPAL DE FASE 2
# ============================================================================

async def initialize_and_run_phase2(
    base_model: BaseEstimator,
    adversarial_validator: Any,
    attention_module: Any,
    fairness_module: Any,
    explain_module: Any,
    phase1_integration: Optional[Any] = None,
    X_data: np.ndarray = None,
    y_data: Optional[np.ndarray] = None,
    config: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    Función principal para inicializar y ejecutar Fase 2 completa
    """
    print("\n" + "="*80)
    print("🚀 INICIALIZANDO FASE 2 - CONSTITUTIONAL AI & VERIFIERS")
    print("="*80)
    
    # 1. Inicializar integración Fase 2
    phase2 = Phase2Integration(
        base_model=base_model,
        adversarial_validator=adversarial_validator,
        attention_module=attention_module,
        fairness_module=fairness_module,
        explain_module=explain_module,
        phase1_integration=phase1_integration,
        config=config
    )
    
    # 2. Obtener estado inicial
    initial_status = phase2.get_phase2_status()
    print(f"\nEstado inicial Fase 2:")
    print(f"• Principios cargados: {initial_status['principles']['total']}")
    print(f"• Verificadores activos: {len(initial_status['constitutional_ai']['verifiers'])}")
    print(f"• Integraciones configuradas: {sum(initial_status['integration_status'].values())}")
    
    # 3. Ejecutar pipeline Fase 2 si hay datos
    if X_data is not None:
        print("\n" + "="*80)
        print("EJECUTANDO PIPELINE COMPLETO FASE 2")
        print("="*80)
        
        results = await phase2.run_phase2_pipeline(
            X=X_data,
            y=y_data,
            test_size=0.2,
            audit_sample_size=min(100, len(X_data))
        )
        
        # 4. Obtener estado final
        final_status = phase2.get_phase2_status()
        
        print("\n" + "="*80)
        print("RESUMEN EJECUCIÓN FASE 2")
        print("="*80)
        print(f"• Compliance inicial: {results['phase_report']['summary']['initial_compliance']:.3f}")
        print(f"• Principios auditados: {len(results['phase_report']['principles_details'])}")
        print(f"• Verificaciones realizadas: {final_status['constitutional_ai']['system_stats']['total_verifications']}")
        print(f"• Violaciones detectadas: {final_status['constitutional_ai']['system_stats']['total_violations']}")
        print(f"• Remediciones automáticas: {final_status['constitutional_ai']['system_stats']['auto_remediations']}")
        print("="*80)
        
        return results
    else:
        print("\n⚠️ No se proporcionaron datos para ejecutar el pipeline completo")
        print("Fase 2 inicializada pero no ejecutada")
        return {'phase2_initialized': True, 'status': initial_status}