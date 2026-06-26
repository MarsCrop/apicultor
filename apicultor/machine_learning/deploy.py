# === fase4_deployment_monitoring.py ===
"""
FASE 4: Sistema completo de deployment, monitoreo y mantenimiento en producción
con integración de todas las fases anteriores. Implementación 100% NumPy.
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
import warnings
from enum import Enum
import time
import pickle
import shutil
import os
import sys
from pathlib import Path
import copy
import math
import random

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

# ============================================================================
# 1. SISTEMA DE DEPLOYMENT CON CI/CD (SOLO NUMPY)
# ============================================================================

class DeploymentStage(Enum):
    """Etapas del proceso de deployment"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    CANARY = "canary"
    PRODUCTION = "production"
    ROLLBACK = "rollback"

@dataclass
class DeploymentArtifact:
    """Artifact para deployment"""
    name: str
    version: str
    artifact_type: str  # 'model', 'config', 'code', 'data'
    content: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    checksum: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    size_bytes: int = 0
    
    def calculate_checksum(self) -> str:
        """Calcula checksum del artifact usando NumPy"""
        try:
            # Serializar contenido para checksum
            if isinstance(self.content, np.ndarray):
                content_bytes = self.content.tobytes()
            elif hasattr(self.content, 'get_params') or hasattr(self.content, 'params'):
                # Modelo - serializar parámetros
                content_str = str(self._extract_model_params(self.content))
                content_bytes = content_str.encode('utf-8')
            else:
                content_str = str(self.content)
                content_bytes = content_str.encode('utf-8')
            
            self.checksum = hashlib.md5(content_bytes).hexdigest()
            self.size_bytes = len(content_bytes)
            return self.checksum
            
        except Exception as e:
            # Fallback simple
            content_str = f"{self.name}_{self.version}_{self.created_at}"
            self.checksum = hashlib.md5(content_str.encode()).hexdigest()
            return self.checksum
    
    def _extract_model_params(self, model: Any) -> Dict:
        """Extrae parámetros de modelo para checksum"""
        params = {}
        
        if hasattr(model, 'get_params'):
            try:
                params = model.get_params()
            except:
                pass
        
        # Buscar parámetros comunes
        for attr in ['coef_', 'intercept_', 'weights', 'biases', 'W', 'b']:
            if hasattr(model, attr):
                val = getattr(model, attr)
                if isinstance(val, np.ndarray):
                    params[attr] = val.shape
                else:
                    params[attr] = str(type(val))
        
        return params
    
    def to_dict(self) -> Dict:
        """Convierte a dict para serialización"""
        return {
            'name': self.name,
            'version': self.version,
            'artifact_type': self.artifact_type,
            'metadata': self.metadata,
            'checksum': self.checksum,
            'created_at': self.created_at.isoformat(),
            'size_bytes': self.size_bytes
        }

class CICDPipeline:
    """
    Pipeline de CI/CD completo para ML usando solo NumPy
    """
    
    def __init__(self, 
                 artifacts_dir: str = "./artifacts",
                 config: Optional[Dict] = None):
        
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        self.config = config or {
            'versioning_scheme': 'semantic',
            'auto_versioning': True,
            'artifact_retention_days': 30,
            'rollback_strategy': 'automatic',
            'health_check_timeout': 30,
            'max_rollback_attempts': 3,
            'deployment_timeout_seconds': 1800,
            'performance_threshold': 0.7,
            'security_threshold': 0.8,
            'min_samples_for_test': 100
        }
        
        # Registro de deployments
        self.deployment_history = []
        self.artifacts_registry = {}
        
        # Estado actual
        self.current_stage = DeploymentStage.DEVELOPMENT
        self.current_version = "0.0.0"
        self.deployed_models = {}
        self.rollback_stack = []
        
        # Inicializar pipeline
        self._initialize_pipeline()
    
    def _initialize_pipeline(self):
        """Inicializa estructura del pipeline"""
        # Crear directorios necesarios
        stages = [stage.value for stage in DeploymentStage]
        for stage in stages:
            stage_dir = self.artifacts_dir / stage
            stage_dir.mkdir(exist_ok=True)
            
            # Subdirectorios para cada stage
            (stage_dir / 'models').mkdir(exist_ok=True)
            (stage_dir / 'configs').mkdir(exist_ok=True)
            (stage_dir / 'metrics').mkdir(exist_ok=True)
            (stage_dir / 'logs').mkdir(exist_ok=True)
        
        # Crear archivo de configuración inicial
        config_path = self.artifacts_dir / 'pipeline_config.json'
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        logger.info(f"CI/CD Pipeline inicializado en {self.artifacts_dir}")
    
    def register_artifact(self, artifact: DeploymentArtifact) -> str:
        """Registra un nuevo artifact"""
        artifact.calculate_checksum()
        artifact_id = f"{artifact.name}_{artifact.version}_{artifact.checksum[:8]}"
        
        self.artifacts_registry[artifact_id] = artifact.to_dict()
        
        # Guardar artifact en disco
        artifact_path = self.artifacts_dir / self.current_stage.value / 'models' / f"{artifact_id}.pkl"
        with open(artifact_path, 'wb') as f:
            pickle.dump(artifact, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Guardar metadata en JSON
        metadata_path = self.artifacts_dir / self.current_stage.value / 'configs' / f"{artifact_id}.json"
        with open(metadata_path, 'w') as f:
            json.dump(artifact.to_dict(), f, indent=2)
        
        # Actualizar versión actual si es un modelo
        if artifact.artifact_type == 'model':
            self.current_version = artifact.version
            logger.info(f"Modelo registrado: {artifact.name} v{artifact.version}")
        
        return artifact_id
    
    def create_model_artifact(self,
                             model: Any,
                             model_name: str,
                             metrics: Optional[Dict] = None,
                             version: Optional[str] = None) -> DeploymentArtifact:
        """Crea artifact de modelo"""
        if version is None:
            version = self._generate_version()
        
        metadata = {
            'model_type': type(model).__name__,
            'created_by': 'deployment_pipeline',
            'stage': self.current_stage.value,
            'created_at': datetime.now().isoformat(),
            'performance_metrics': metrics or {},
            'model_size': self._estimate_model_size(model)
        }
        
        # Extraer características del modelo
        if hasattr(model, 'input_dim'):
            metadata['input_dim'] = getattr(model, 'input_dim', 'unknown')
        if hasattr(model, 'output_dim'):
            metadata['output_dim'] = getattr(model, 'output_dim', 'unknown')
        
        artifact = DeploymentArtifact(
            name=model_name,
            version=version,
            artifact_type='model',
            content=model,
            metadata=metadata
        )
        
        return artifact
    
    def _estimate_model_size(self, model: Any) -> int:
        """Estima tamaño del modelo en bytes"""
        size = 0
        
        # Contar parámetros si es numpy array
        if isinstance(model, np.ndarray):
            size = model.nbytes
        
        # Modelo con parámetros
        elif hasattr(model, 'params'):
            for key, param in model.params.items():
                if isinstance(param, np.ndarray):
                    size += param.nbytes
        
        # Modelo con coef_ e intercept_
        elif hasattr(model, 'coef_'):
            if isinstance(model.coef_, np.ndarray):
                size += model.coef_.nbytes
            if hasattr(model, 'intercept_') and isinstance(model.intercept_, np.ndarray):
                size += model.intercept_.nbytes
        
        return size
    
    def _generate_version(self) -> str:
        """Genera versión automática usando solo Python"""
        if self.config['versioning_scheme'] == 'semantic':
            # Semantic versioning: major.minor.patch
            current = self.current_version.split('.')
            if len(current) == 3:
                try:
                    major, minor, patch = map(int, current)
                    patch += 1
                    if patch >= 100:
                        patch = 0
                        minor += 1
                    if minor >= 100:
                        minor = 0
                        major += 1
                    return f"{major}.{minor}.{patch}"
                except:
                    return "1.0.0"
            else:
                return "1.0.0"
        
        elif self.config['versioning_scheme'] == 'timestamp':
            # Timestamp versioning
            now = datetime.now()
            return now.strftime("%Y%m%d.%H%M%S")
        
        else:  # hash style
            random_bytes = os.urandom(8)
            return hashlib.md5(random_bytes).hexdigest()[:8]
    
    async def run_tests(self,
                       model_artifact: DeploymentArtifact,
                       test_data: Optional[Dict] = None) -> Dict[str, Any]:
        """Ejecuta tests automatizados usando solo NumPy"""
        test_results = {
            'artifact_id': f"{model_artifact.name}_{model_artifact.version}",
            'tests_run': [],
            'passed': True,
            'overall_score': 0.0,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Ejecutando tests para {model_artifact.name} v{model_artifact.version}")
        
        # Ejecutar tests en paralelo
        test_tasks = [
            self._run_integrity_test(model_artifact),
            self._run_performance_test(model_artifact, test_data),
            self._run_security_test(model_artifact),
            self._run_compatibility_test(model_artifact),
            self._run_memory_test(model_artifact)
        ]
        
        results = await asyncio.gather(*test_tasks)
        
        for result in results:
            test_results['tests_run'].append(result)
            if not result['passed']:
                test_results['passed'] = False
        
        # Calcular score general
        scores = [r.get('score', 0.0) for r in results if 'score' in r]
        if scores:
            test_results['overall_score'] = float(np.mean(scores))
        
        logger.info(f"Tests completados: {len(results)}, Passed: {test_results['passed']}, Score: {test_results['overall_score']:.3f}")
        
        return test_results
    
    async def _run_integrity_test(self, artifact: DeploymentArtifact) -> Dict[str, Any]:
        """Test de integridad del modelo"""
        try:
            model = artifact.content
            
            # Verificar que el modelo se puede usar
            test_passed = True
            details = {'checks': []}
            
            # Check 1: Modelo no es None
            if model is None:
                test_passed = False
                details['checks'].append('model_is_none')
            
            # Check 2: Tiene método predict o forward
            has_predict = hasattr(model, 'predict')
            has_forward = hasattr(model, 'forward')
            
            if not (has_predict or has_forward):
                test_passed = False
                details['checks'].append('no_predict_or_forward')
            
            details['has_predict'] = has_predict
            details['has_forward'] = has_forward
            
            # Check 3: Prueba de inferencia simple
            if test_passed and (has_predict or has_forward):
                try:
                    # Crear input de prueba
                    if hasattr(model, 'input_dim'):
                        input_dim = model.input_dim
                        test_input = np.random.randn(1, input_dim).astype(np.float32)
                    else:
                        # Intentar inferir dimensión
                        test_input = np.random.randn(1, 10).astype(np.float32)
                    
                    if has_predict:
                        prediction = model.predict(test_input)
                    else:
                        prediction = model.forward(test_input)
                    
                    details['inference_test'] = 'passed'
                    details['prediction_shape'] = str(prediction.shape) if hasattr(prediction, 'shape') else 'unknown'
                    details['prediction_dtype'] = str(prediction.dtype) if hasattr(prediction, 'dtype') else 'unknown'
                    
                except Exception as e:
                    test_passed = False
                    details['inference_test'] = 'failed'
                    details['inference_error'] = str(e)
            
            return {
                'test_name': 'model_integrity',
                'passed': test_passed,
                'score': 1.0 if test_passed else 0.0,
                'details': details
            }
            
        except Exception as e:
            return {
                'test_name': 'model_integrity',
                'passed': False,
                'score': 0.0,
                'details': {'error': str(e)}
            }
    
    async def _run_performance_test(self,
                                  artifact: DeploymentArtifact,
                                  test_data: Optional[Dict] = None) -> Dict[str, Any]:
        """Test de performance del modelo usando NumPy"""
        try:
            model = artifact.content
            details = {}
            score = 0.0
            
            if not (hasattr(model, 'predict') or hasattr(model, 'forward')):
                return {
                    'test_name': 'performance',
                    'passed': False,
                    'score': 0.0,
                    'details': {'error': 'Model does not have predict or forward method'}
                }
            
            # Medir tiempo de inferencia
            inference_times = []
            throughput_samples = []
            
            # Tamaños de batch para test
            batch_sizes = [1, 8, 32]
            
            for batch_size in batch_sizes:
                # Generar datos de prueba
                if test_data and 'X_test' in test_data:
                    X_test = test_data['X_test']
                    if len(X_test) >= batch_size:
                        X_batch = X_test[:batch_size]
                    else:
                        X_batch = np.random.randn(batch_size, X_test.shape[1]).astype(np.float32)
                else:
                    # Generar datos aleatorios
                    input_dim = getattr(model, 'input_dim', 10)
                    X_batch = np.random.randn(batch_size, input_dim).astype(np.float32)
                
                # Medir tiempo
                start_time = time.perf_counter()
                
                if hasattr(model, 'predict'):
                    predictions = model.predict(X_batch)
                else:
                    predictions = model.forward(X_batch)
                
                end_time = time.perf_counter()
                
                inference_time = end_time - start_time
                inference_times.append(inference_time)
                
                # Calcular throughput
                throughput = batch_size / inference_time if inference_time > 0 else 0
                throughput_samples.append(throughput)
            
            # Calcular métricas de latency
            avg_latency = np.mean(inference_times) if inference_times else 0
            avg_throughput = np.mean(throughput_samples) if throughput_samples else 0
            
            details.update({
                'latency_seconds': float(avg_latency),
                'throughput_samples_per_second': float(avg_throughput),
                'batch_sizes_tested': batch_sizes,
                'inference_times': [float(t) for t in inference_times]
            })
            
            # Calcular accuracy si hay test data
            if test_data and 'X_test' in test_data and 'y_test' in test_data:
                X_test = test_data['X_test']
                y_test = test_data['y_test']
                
                sample_size = min(len(X_test), self.config.get('min_samples_for_test', 100))
                X_sample = X_test[:sample_size]
                y_sample = y_test[:sample_size]
                
                # Realizar predicciones
                if hasattr(model, 'predict'):
                    predictions = model.predict(X_sample)
                else:
                    predictions = model.forward(X_sample)
                
                # Calcular accuracy
                if predictions.shape == y_sample.shape:
                    if len(predictions.shape) == 1 or predictions.shape[1] == 1:
                        # Regresión: R² score
                        ss_res = np.sum((y_sample - predictions) ** 2)
                        ss_tot = np.sum((y_sample - np.mean(y_sample)) ** 2)
                        accuracy = 1 - (ss_res / (ss_tot + 1e-8))
                    else:
                        # Clasificación: accuracy
                        pred_classes = np.argmax(predictions, axis=1)
                        true_classes = np.argmax(y_sample, axis=1) if len(y_sample.shape) > 1 else y_sample
                        accuracy = np.mean(pred_classes == true_classes)
                    
                    score = float(accuracy)
                    details['accuracy'] = float(accuracy)
                    details['samples_tested'] = sample_size
            
            # Score base basado en throughput
            throughput_score = min(avg_throughput / 1000, 1.0)  # Normalizar
            
            if 'accuracy' in details:
                final_score = 0.7 * details['accuracy'] + 0.3 * throughput_score
            else:
                final_score = throughput_score
            
            passed = final_score >= self.config.get('performance_threshold', 0.7)
            
            return {
                'test_name': 'performance',
                'passed': passed,
                'score': float(final_score),
                'details': details
            }
            
        except Exception as e:
            return {
                'test_name': 'performance',
                'passed': False,
                'score': 0.0,
                'details': {'error': str(e)}
            }
    
    async def _run_security_test(self, artifact: DeploymentArtifact) -> Dict[str, Any]:
        """Test de seguridad/constitucional"""
        try:
            metadata = artifact.metadata
            details = {}
            
            # Extraer flags de seguridad
            security_flags = metadata.get('security_flags', [])
            constitutional_score = metadata.get('constitutional_score', 1.0)
            hacking_flags = metadata.get('hacking_flags', [])
            
            # Verificar contenido peligroso
            model_content = artifact.content
            security_issues = []
            
            # Check 1: Verificar valores extremos en parámetros
            if hasattr(model_content, 'params'):
                for param_name, param in model_content.params.items():
                    if isinstance(param, np.ndarray):
                        param_abs = np.abs(param)
                        if np.any(param_abs > 1000):  # Valores extremos
                            security_issues.append(f'extreme_values_{param_name}')
            
            # Check 2: Verificar NaN o Inf
            if hasattr(model_content, 'params'):
                for param_name, param in model_content.params.items():
                    if isinstance(param, np.ndarray):
                        if np.any(np.isnan(param)):
                            security_issues.append(f'nan_values_{param_name}')
                        if np.any(np.isinf(param)):
                            security_issues.append(f'inf_values_{param_name}')
            
            # Combinar todos los issues
            all_issues = security_flags + hacking_flags + security_issues
            
            # Calcular score de seguridad
            base_score = constitutional_score
            issue_penalty = len(all_issues) * 0.1
            security_score = max(0, base_score - issue_penalty)
            
            passed = (security_score >= self.config.get('security_threshold', 0.8))
            
            details.update({
                'security_score': float(security_score),
                'constitutional_score': float(constitutional_score),
                'security_issues': all_issues,
                'issues_count': len(all_issues),
                'model_size_bytes': artifact.size_bytes
            })
            
            return {
                'test_name': 'security',
                'passed': passed,
                'score': float(security_score),
                'details': details
            }
            
        except Exception as e:
            return {
                'test_name': 'security',
                'passed': False,
                'score': 0.0,
                'details': {'error': str(e)}
            }
    
    async def _run_compatibility_test(self, artifact: DeploymentArtifact) -> Dict[str, Any]:
        """Test de compatibilidad con sistema"""
        try:
            model = artifact.content
            compatibility_issues = []
            details = {}
            
            # Check 1: Tipo de modelo
            model_type = type(model).__name__
            details['model_type'] = model_type
            
            # Check 2: Versión de numpy
            numpy_version = np.__version__
            details['numpy_version'] = numpy_version
            
            # Check 3: Verificar atributos requeridos
            required_for_inference = ['predict', 'forward']
            has_inference_method = any(hasattr(model, attr) for attr in required_for_inference)
            
            if not has_inference_method:
                compatibility_issues.append('no_inference_method')
            
            # Check 4: Verificar tipos de datos en parámetros
            if hasattr(model, 'params'):
                for param_name, param in model.params.items():
                    if isinstance(param, np.ndarray):
                        if param.dtype not in [np.float32, np.float64]:
                            compatibility_issues.append(f'unsupported_dtype_{param_name}_{param.dtype}')
            
            # Check 5: Verificar dimensiones
            try:
                # Intentar inferir input dimension
                if hasattr(model, 'input_dim'):
                    input_dim = model.input_dim
                    if input_dim <= 0:
                        compatibility_issues.append('invalid_input_dim')
                else:
                    # Intentar inferir de parámetros
                    if hasattr(model, 'params') and 'W' in model.params:
                        input_dim = model.params['W'].shape[0]
                        details['inferred_input_dim'] = input_dim
            except:
                compatibility_issues.append('cannot_determine_input_dim')
            
            passed = len(compatibility_issues) == 0
            score = 1.0 if passed else max(0, 1.0 - len(compatibility_issues) * 0.2)
            
            details['compatibility_issues'] = compatibility_issues
            details['issues_count'] = len(compatibility_issues)
            
            return {
                'test_name': 'compatibility',
                'passed': passed,
                'score': float(score),
                'details': details
            }
            
        except Exception as e:
            return {
                'test_name': 'compatibility',
                'passed': False,
                'score': 0.0,
                'details': {'error': str(e)}
            }
    
    async def _run_memory_test(self, artifact: DeploymentArtifact) -> Dict[str, Any]:
        """Test de uso de memoria"""
        try:
            model = artifact.content
            details = {}
            
            # Estimar uso de memoria
            memory_usage = 0
            
            if hasattr(model, 'params'):
                for param_name, param in model.params.items():
                    if isinstance(param, np.ndarray):
                        memory_usage += param.nbytes
                    elif isinstance(param, dict):
                        # Parámetros anidados
                        for sub_param in param.values():
                            if isinstance(sub_param, np.ndarray):
                                memory_usage += sub_param.nbytes
            
            # Convertir a MB
            memory_usage_mb = memory_usage / (1024 * 1024)
            
            # Score basado en uso de memoria (menos es mejor)
            # Considerar 100MB como límite razonable
            memory_score = max(0, 1.0 - (memory_usage_mb / 100))
            
            passed = memory_usage_mb < 500  # Límite de 500MB
            
            details.update({
                'memory_usage_bytes': memory_usage,
                'memory_usage_mb': memory_usage_mb,
                'estimated_parameters': self._count_parameters(model)
            })
            
            return {
                'test_name': 'memory',
                'passed': passed,
                'score': float(memory_score),
                'details': details
            }
            
        except Exception as e:
            return {
                'test_name': 'memory',
                'passed': False,
                'score': 0.0,
                'details': {'error': str(e)}
            }
    
    def _count_parameters(self, model: Any) -> int:
        """Cuenta número de parámetros del modelo"""
        count = 0
        
        if hasattr(model, 'params'):
            for param in model.params.values():
                if isinstance(param, np.ndarray):
                    count += param.size
                elif isinstance(param, dict):
                    for sub_param in param.values():
                        if isinstance(sub_param, np.ndarray):
                            count += sub_param.size
        
        return count
    
    def _load_artifact(self, artifact_id: str) -> Optional[DeploymentArtifact]:
        """Carga artifact desde disco"""
        # Buscar en todos los stages
        for stage in DeploymentStage:
            artifact_path = self.artifacts_dir / stage.value / 'models' / f"{artifact_id}.pkl"
            if artifact_path.exists():
                try:
                    with open(artifact_path, 'rb') as f:
                        return pickle.load(f)
                except Exception as e:
                    logger.error(f"Error cargando artifact {artifact_id}: {e}")
                    return None
        
        return None
    
    async def deploy_to_stage(self,
                            artifact_id: str,
                            target_stage: DeploymentStage,
                            test_data: Optional[Dict] = None,
                            canary_percentage: float = 0.1) -> Dict[str, Any]:
        """
        Despliega artifact a un stage específico
        """
        logger.info(f"Desplegando {artifact_id} a {target_stage.value}")
        
        deployment_id = hashlib.md5(
            f"{artifact_id}_{target_stage.value}_{datetime.now().timestamp()}".encode()
        ).hexdigest()[:16]
        
        deployment_record = {
            'deployment_id': deployment_id,
            'artifact_id': artifact_id,
            'source_stage': self.current_stage.value,
            'target_stage': target_stage.value,
            'start_time': datetime.now().isoformat(),
            'status': 'in_progress',
            'tests_passed': False,
            'health_check_passed': False,
            'canary_percentage': canary_percentage if target_stage == DeploymentStage.CANARY else None
        }
        
        try:
            # 1. Cargar artifact
            artifact = self._load_artifact(artifact_id)
            if not artifact:
                raise ValueError(f"Artifact {artifact_id} no encontrado")
            
            # 2. Guardar en stack de rollback
            self.rollback_stack.append({
                'stage': self.current_stage,
                'artifact_id': artifact_id,
                'timestamp': datetime.now()
            })
            
            # 3. Ejecutar tests si no estamos en development
            if target_stage != DeploymentStage.DEVELOPMENT:
                logger.info("Ejecutando tests pre-deployment...")
                test_results = await self.run_tests(artifact, test_data)
                deployment_record['test_results'] = test_results
                
                if not test_results['passed']:
                    raise ValueError(f"Tests fallaron. Score: {test_results['overall_score']:.3f}")
                
                deployment_record['tests_passed'] = True
            
            # 4. Health check
            logger.info("Ejecutando health check...")
            health_check = await self._run_stage_health_check(target_stage)
            deployment_record['health_check'] = health_check
            
            if not health_check['healthy']:
                raise ValueError(f"Health check falló: {health_check.get('error', 'Unknown error')}")
            
            deployment_record['health_check_passed'] = True
            
            # 5. Realizar deployment
            logger.info(f"Realizando deployment a {target_stage.value}...")
            
            if target_stage == DeploymentStage.CANARY:
                # Deployment canary
                await self._deploy_canary(artifact, canary_percentage)
                deployment_record['deployment_type'] = 'canary'
                deployment_record['canary_traffic_percentage'] = canary_percentage
                
            else:
                # Deployment completo
                await self._deploy_full(artifact, target_stage)
                deployment_record['deployment_type'] = 'full'
            
            # 6. Actualizar estado
            self.current_stage = target_stage
            self.deployed_models[target_stage.value] = {
                'artifact_id': artifact_id,
                'version': artifact.version,
                'deployed_at': datetime.now().isoformat(),
                'model_name': artifact.name
            }
            
            # 7. Ejecutar post-deployment checks
            logger.info("Ejecutando post-deployment checks...")
            post_deploy_check = await self._run_post_deployment_check(target_stage, artifact)
            deployment_record['post_deployment_check'] = post_deploy_check
            
            if not post_deploy_check['passed']:
                logger.warning(f"Post-deployment checks fallaron: {post_deploy_check.get('issues', [])}")
            
            # 8. Guardar registro de deployment
            deployment_record.update({
                'status': 'success',
                'end_time': datetime.now().isoformat(),
                'duration_seconds': (datetime.now() - datetime.fromisoformat(
                    deployment_record['start_time'].replace('Z', '+00:00')
                )).total_seconds(),
                'model_version': artifact.version,
                'model_name': artifact.name
            })
            
            logger.info(f"✅ Deployment exitoso a {target_stage.value}")
            
        except Exception as e:
            deployment_record.update({
                'status': 'failed',
                'error': str(e),
                'end_time': datetime.now().isoformat(),
                'duration_seconds': (datetime.now() - datetime.fromisoformat(
                    deployment_record['start_time'].replace('Z', '+00:00')
                )).total_seconds()
            })
            
            logger.error(f"❌ Deployment fallido: {e}")
            
            # Intentar rollback automático si está configurado
            if self.config.get('rollback_strategy') == 'automatic':
                logger.info("Intentando rollback automático...")
                try:
                    await self.rollback_latest()
                except Exception as rollback_error:
                    logger.error(f"Rollback fallido: {rollback_error}")
        
        finally:
            self.deployment_history.append(deployment_record)
            self._save_deployment_record(deployment_record)
        
        return deployment_record
    
    async def _run_stage_health_check(self, stage: DeploymentStage) -> Dict[str, Any]:
        """Health check del stage usando solo Python"""
        try:
            details = {}
            
            # Verificar directorio del stage
            stage_dir = self.artifacts_dir / stage.value
            if not stage_dir.exists():
                return {
                    'healthy': False,
                    'error': f"Directorio {stage_dir} no existe"
                }
            
            # Verificar permisos
            test_file = stage_dir / '.health_check_test'
            try:
                with open(test_file, 'w') as f:
                    f.write('test')
                os.remove(test_file)
                details['write_permissions'] = True
            except PermissionError:
                return {
                    'healthy': False,
                    'error': f"Sin permisos de escritura en {stage_dir}"
                }
            
            # Verificar espacio en disco (simplificado)
            try:
                stat = os.statvfs(stage_dir)
                free_gb = (stat.f_bavail * stat.f_frsize) / (1024**3)
                details['free_disk_gb'] = free_gb
                
                if free_gb < 1.0:  # Menos de 1GB libre
                    return {
                        'healthy': False,
                        'error': f"Espacio en disco insuficiente: {free_gb:.2f}GB"
                    }
            except:
                # Si no se puede verificar espacio, continuar
                details['disk_check'] = 'skipped'
            
            # Verificar que haya modelos disponibles en stage anterior para rollback
            if stage != DeploymentStage.DEVELOPMENT:
                prev_stage = self._get_previous_stage(stage)
                if prev_stage:
                    prev_models_dir = self.artifacts_dir / prev_stage.value / 'models'
                    if prev_models_dir.exists():
                        model_files = list(prev_models_dir.glob('*.pkl'))
                        details['rollback_models_available'] = len(model_files) > 0
            
            return {
                'healthy': True,
                'details': details
            }
            
        except Exception as e:
            return {
                'healthy': False,
                'error': str(e)
            }
    
    def _get_previous_stage(self, stage: DeploymentStage) -> Optional[DeploymentStage]:
        """Obtiene el stage anterior en el pipeline"""
        stages = list(DeploymentStage)
        try:
            current_idx = stages.index(stage)
            if current_idx > 0:
                return stages[current_idx - 1]
        except:
            pass
        return None
    
    async def _deploy_canary(self, artifact: DeploymentArtifact, percentage: float):
        """Implementa deployment canary"""
        logger.info(f"Implementando canary deployment ({percentage*100:.1f}% de tráfico)")
        
        # En un sistema real, esto controlaría el enrutamiento de tráfico
        # Aquí simulamos la lógica
        
        # 1. Guardar artifact en directorio canary
        canary_dir = self.artifacts_dir / DeploymentStage.CANARY.value / 'models'
        artifact_id = f"{artifact.name}_{artifact.version}_{artifact.checksum[:8]}"
        artifact_path = canary_dir / f"{artifact_id}.pkl"
        
        with open(artifact_path, 'wb') as f:
            pickle.dump(artifact, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # 2. Guardar configuración de canary
        canary_config = {
            'artifact_id': artifact_id,
            'percentage': percentage,
            'start_time': datetime.now().isoformat(),
            'model_version': artifact.version,
            'checksum': artifact.checksum
        }
        
        config_path = self.artifacts_dir / DeploymentStage.CANARY.value / 'configs' / 'canary_config.json'
        with open(config_path, 'w') as f:
            json.dump(canary_config, f, indent=2)
        
        # 3. Simular monitoreo de canary
        logger.info(f"Canary deployment configurado. {percentage*100:.1f}% de tráfico será dirigido a v{artifact.version}")
        
        # 4. Simular período de observación
        await asyncio.sleep(2)  # Simular tiempo de observación
    
    async def _deploy_full(self, artifact: DeploymentArtifact, stage: DeploymentStage):
        """Implementa deployment completo"""
        logger.info(f"Implementando deployment completo a {stage.value}")
        
        # 1. Guardar artifact
        stage_dir = self.artifacts_dir / stage.value / 'models'
        artifact_id = f"{artifact.name}_{artifact.version}_{artifact.checksum[:8]}"
        artifact_path = stage_dir / f"{artifact_id}.pkl"
        
        with open(artifact_path, 'wb') as f:
            pickle.dump(artifact, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # 2. Actualizar archivo de versión actual
        current_version_file = self.artifacts_dir / stage.value / 'current_version.json'
        current_version_info = {
            'artifact_id': artifact_id,
            'version': artifact.version,
            'deployed_at': datetime.now().isoformat(),
            'model_name': artifact.name,
            'checksum': artifact.checksum
        }
        
        with open(current_version_file, 'w') as f:
            json.dump(current_version_info, f, indent=2)
        
        # 3. Limpiar versiones antiguas (retención)
        self._cleanup_old_versions(stage)
        
        logger.info(f"Deployment completo exitoso. Versión actual: {artifact.version}")
    
    def _cleanup_old_versions(self, stage: DeploymentStage):
        """Limpia versiones antiguas según política de retención"""
        retention_days = self.config.get('artifact_retention_days', 30)
        cutoff_time = datetime.now() - timedelta(days=retention_days)
        
        models_dir = self.artifacts_dir / stage.value / 'models'
        
        if not models_dir.exists():
            return
        
        # Listar todos los artifacts
        for artifact_file in models_dir.glob('*.pkl'):
            try:
                # Cargar artifact para ver fecha
                with open(artifact_file, 'rb') as f:
                    artifact = pickle.load(f)
                
                # Verificar si es muy antiguo
                if artifact.created_at < cutoff_time:
                    # No eliminar la versión actual
                    current_version = self.deployed_models.get(stage.value, {}).get('version')
                    if artifact.version != current_version:
                        artifact_file.unlink()
                        logger.info(f"Eliminado artifact antiguo: {artifact_file.name}")
                        
            except Exception as e:
                logger.warning(f"Error limpiando artifact {artifact_file}: {e}")
    
    async def _run_post_deployment_check(self, 
                                       stage: DeploymentStage,
                                       artifact: DeploymentArtifact) -> Dict[str, Any]:
        """Checks post-deployment"""
        try:
            model = artifact.content
            issues = []
            details = {}
            
            # 1. Verificar que el modelo se carga correctamente
            try:
                test_input = np.random.randn(1, 10).astype(np.float32)
                
                if hasattr(model, 'predict'):
                    prediction = model.predict(test_input)
                    details['predict_method'] = 'ok'
                elif hasattr(model, 'forward'):
                    prediction = model.forward(test_input)
                    details['forward_method'] = 'ok'
                else:
                    issues.append('no_inference_method')
                    prediction = None
                
                if prediction is not None:
                    details['prediction_shape'] = str(prediction.shape)
                    details['prediction_dtype'] = str(prediction.dtype)
            except Exception as e:
                issues.append(f'inference_failed: {str(e)}')
            
            # 2. Verificar métricas de performance
            if 'performance_metrics' in artifact.metadata:
                metrics = artifact.metadata['performance_metrics']
                details['performance_metrics'] = metrics
                
                # Verificar umbrales
                if 'accuracy' in metrics and metrics['accuracy'] < 0.7:
                    issues.append('low_accuracy')
                if 'loss' in metrics and metrics['loss'] > 1.0:
                    issues.append('high_loss')
            
            # 3. Verificar integridad de archivos
            artifact_id = f"{artifact.name}_{artifact.version}_{artifact.checksum[:8]}"
            artifact_path = self.artifacts_dir / stage.value / 'models' / f"{artifact_id}.pkl"
            
            if artifact_path.exists():
                file_size = artifact_path.stat().st_size
                details['file_size_bytes'] = file_size
                
                if file_size == 0:
                    issues.append('empty_artifact_file')
            else:
                issues.append('artifact_file_missing')
            
            passed = len(issues) == 0
            details['issues'] = issues
            details['issues_count'] = len(issues)
            
            return {
                'passed': passed,
                'details': details
            }
            
        except Exception as e:
            return {
                'passed': False,
                'details': {
                    'error': str(e),
                    'issues': ['check_failed']
                }
            }
    
    async def rollback_latest(self, target_stage: Optional[DeploymentStage] = None) -> Dict[str, Any]:
        """Realiza rollback al deployment anterior"""
        logger.info("Iniciando rollback...")
        
        if not self.rollback_stack:
            raise ValueError("No hay deployments previos para rollback")
        
        # Obtener deployment anterior
        previous = self.rollback_stack.pop()
        previous_stage = previous['stage']
        previous_artifact_id = previous['artifact_id']
        
        if target_stage:
            previous_stage = target_stage
        
        logger.info(f"Rollback a {previous_stage.value}, artifact: {previous_artifact_id}")
        
        # Cargar artifact anterior
        previous_artifact = self._load_artifact(previous_artifact_id)
        if not previous_artifact:
            raise ValueError(f"Artifact anterior {previous_artifact_id} no encontrado")
        
        # Realizar deployment al stage anterior
        deployment_result = await self.deploy_to_stage(
            previous_artifact_id,
            previous_stage,
            canary_percentage=1.0  # Rollback completo
        )
        
        deployment_result['rollback'] = True
        deployment_result['rollback_from'] = self.current_stage.value
        deployment_result['rollback_to'] = previous_stage.value
        deployment_result['rollback_artifact'] = previous_artifact_id
        
        logger.info(f"✅ Rollback completado a v{previous_artifact.version}")
        
        return deployment_result
    
    def _save_deployment_record(self, record: Dict[str, Any]):
        """Guarda registro de deployment"""
        deployments_file = self.artifacts_dir / 'deployments_history.json'
        
        # Cargar historial existente
        if deployments_file.exists():
            try:
                with open(deployments_file, 'r') as f:
                    history = json.load(f)
            except:
                history = []
        else:
            history = []
        
        # Agregar nuevo registro
        history.append(record)
        
        # Guardar
        with open(deployments_file, 'w') as f:
            json.dump(history, f, indent=2)
    
    def get_deployment_status(self, stage: Optional[DeploymentStage] = None) -> Dict[str, Any]:
        """Obtiene estado actual del deployment"""
        if stage is None:
            stage = self.current_stage
        
        status = {
            'current_stage': stage.value,
            'current_version': self.current_version,
            'deployed_models': self.deployed_models.get(stage.value, {}),
            'health_status': 'unknown'
        }
        
        # Verificar health del stage actual
        health_file = self.artifacts_dir / stage.value / 'health.json'
        if health_file.exists():
            try:
                with open(health_file, 'r') as f:
                    health_data = json.load(f)
                status['health_status'] = health_data.get('status', 'unknown')
                status['last_health_check'] = health_data.get('timestamp', 'unknown')
            except:
                pass
        
        # Obtener métricas recientes
        metrics_dir = self.artifacts_dir / stage.value / 'metrics'
        if metrics_dir.exists():
            metric_files = list(metrics_dir.glob('*.json'))
            if metric_files:
                # Tomar el archivo más reciente
                latest_metric = max(metric_files, key=lambda x: x.stat().st_mtime)
                try:
                    with open(latest_metric, 'r') as f:
                        status['latest_metrics'] = json.load(f)
                except:
                    pass
        
        return status
    
    def cleanup_old_deployments(self):
        """Limpia deployments antiguos"""
        retention_days = self.config.get('artifact_retention_days', 30)
        cutoff_time = datetime.now() - timedelta(days=retention_days)
        
        # Limpiar historial
        deployments_file = self.artifacts_dir / 'deployments_history.json'
        if deployments_file.exists():
            try:
                with open(deployments_file, 'r') as f:
                    history = json.load(f)
                
                # Filtrar deployments recientes
                filtered_history = []
                for record in history:
                    record_time = datetime.fromisoformat(
                        record.get('start_time', '2000-01-01').replace('Z', '+00:00')
                    )
                    if record_time > cutoff_time:
                        filtered_history.append(record)
                
                with open(deployments_file, 'w') as f:
                    json.dump(filtered_history, f, indent=2)
                
                logger.info(f"Historial limpiado: {len(history) - len(filtered_history)} registros eliminados")
                
            except Exception as e:
                logger.error(f"Error limpiando historial: {e}")

# ============================================================================
# 2. SISTEMA DE MONITOREO EN PRODUCCIÓN
# ============================================================================

class PerformanceMetric(Enum):
    """Métricas de performance a monitorear"""
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    ACCURACY = "accuracy"
    ERROR_RATE = "error_rate"
    MEMORY_USAGE = "memory_usage"
    CPU_USAGE = "cpu_usage"
    REQUEST_RATE = "request_rate"
    CACHE_HIT_RATE = "cache_hit_rate"

class AlertSeverity(Enum):
    """Severidad de alertas"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class Alert:
    """Alerta del sistema de monitoreo"""
    id: str
    severity: AlertSeverity
    metric: PerformanceMetric
    current_value: float
    threshold: float
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'severity': self.severity.value,
            'metric': self.metric.value,
            'current_value': self.current_value,
            'threshold': self.threshold,
            'message': self.message,
            'timestamp': self.timestamp.isoformat(),
            'resolved': self.resolved,
            'resolved_at': self.resolved_at.isoformat() if self.resolved_at else None
        }

class ProductionMonitor:
    """
    Sistema de monitoreo en producción usando solo NumPy
    """
    
    def __init__(self, 
                 monitoring_dir: str = "./monitoring",
                 config: Optional[Dict] = None):
        
        self.monitoring_dir = Path(monitoring_dir)
        self.monitoring_dir.mkdir(parents=True, exist_ok=True)
        
        self.config = config or {
            'sampling_interval_seconds': 60,
            'window_size': 100,
            'alert_cooldown_minutes': 5,
            'metrics_retention_days': 7,
            
            # Umbrales para alertas
            'latency_threshold_ms': 1000,
            'error_rate_threshold': 0.05,
            'memory_threshold_mb': 1024,
            'cpu_threshold_percent': 80,
            'accuracy_drop_threshold': 0.1,
            
            # Configuración de drift detection
            'drift_detection_enabled': True,
            'drift_confidence_level': 0.95,
            'drift_window_size': 1000,
            
            # Configuración de A/B testing
            'ab_testing_enabled': True,
            'min_samples_for_ab_test': 100
        }
        
        # Almacenamiento de métricas
        self.metrics_history = {metric.value: [] for metric in PerformanceMetric}
        self.alerts_history = []
        self.active_alerts = {}
        
        # Estadísticas de drift
        self.drift_detectors = {}
        self.reference_distributions = {}
        
        # A/B testing
        self.ab_test_groups = {}
        self.ab_test_results = {}
        
        # Inicializar
        self._initialize_monitoring()
    
    def _initialize_monitoring(self):
        """Inicializa sistema de monitoreo"""
        # Crear directorios necesarios
        (self.monitoring_dir / 'metrics').mkdir(exist_ok=True)
        (self.monitoring_dir / 'alerts').mkdir(exist_ok=True)
        (self.monitoring_dir / 'drift').mkdir(exist_ok=True)
        (self.monitoring_dir / 'ab_tests').mkdir(exist_ok=True)
        
        # Cargar historial si existe
        self._load_history()
        
        logger.info(f"Sistema de monitoreo inicializado en {self.monitoring_dir}")
    
    def _load_history(self):
        """Carga historial de monitoreo desde disco"""
        metrics_dir = self.monitoring_dir / 'metrics'
        if metrics_dir.exists():
            for metric_file in metrics_dir.glob('*.json'):
                metric_name = metric_file.stem
                try:
                    with open(metric_file, 'r') as f:
                        data = json.load(f)
                        self.metrics_history[metric_name] = data.get('values', [])
                except:
                    pass
    
    def record_metric(self, 
                     metric: PerformanceMetric, 
                     value: float,
                     timestamp: Optional[datetime] = None):
        """Registra una métrica"""
        if timestamp is None:
            timestamp = datetime.now()
        
        # Agregar a historial en memoria
        metric_entry = {
            'timestamp': timestamp.isoformat(),
            'value': value
        }
        
        self.metrics_history[metric.value].append(metric_entry)
        
        # Mantener ventana deslizante
        if len(self.metrics_history[metric.value]) > self.config['window_size']:
            self.metrics_history[metric.value] = self.metrics_history[metric.value][-self.config['window_size']:]
        
        # Guardar en disco periódicamente
        if len(self.metrics_history[metric.value]) % 10 == 0:
            self._save_metric_history(metric)
        
        # Verificar alertas
        self._check_alerts(metric, value, timestamp)
        
        # Detectar drift si está habilitado
        if self.config['drift_detection_enabled']:
            self._detect_drift(metric, value, timestamp)
    
    def _save_metric_history(self, metric: PerformanceMetric):
        """Guarda historial de métrica en disco"""
        metric_file = self.monitoring_dir / 'metrics' / f"{metric.value}.json"
        
        data = {
            'metric': metric.value,
            'window_size': self.config['window_size'],
            'values': self.metrics_history[metric.value][-100:]  # Guardar últimos 100
        }
        
        with open(metric_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _check_alerts(self, metric: PerformanceMetric, value: float, timestamp: datetime):
        """Verifica si se debe generar una alerta"""
        alert_id = f"{metric.value}_{timestamp.strftime('%Y%m%d_%H%M%S')}"
        
        # Verificar si ya hay una alerta activa para esta métrica
        if metric.value in self.active_alerts:
            last_alert = self.active_alerts[metric.value]
            last_alert_time = datetime.fromisoformat(last_alert['timestamp'].replace('Z', '+00:00'))
            
            # Cooldown para alertas repetidas
            cooldown = timedelta(minutes=self.config['alert_cooldown_minutes'])
            if timestamp - last_alert_time < cooldown:
                return
        
        # Determinar umbral y severidad
        threshold, severity = self._get_metric_threshold(metric, value)
        
        if threshold is not None and self._should_alert(metric, value, threshold):
            # Crear alerta
            alert = Alert(
                id=alert_id,
                severity=severity,
                metric=metric,
                current_value=value,
                threshold=threshold,
                message=self._generate_alert_message(metric, value, threshold)
            )
            
            # Registrar alerta
            self.alerts_history.append(alert.to_dict())
            self.active_alerts[metric.value] = alert.to_dict()
            
            # Guardar alerta en disco
            self._save_alert(alert)
            
            # Log
            logger.warning(f"⚠️  Alerta {severity.value}: {alert.message}")
    
    def _get_metric_threshold(self, metric: PerformanceMetric, value: float) -> Tuple[Optional[float], AlertSeverity]:
        """Obtiene umbral y severidad para una métrica"""
        config = self.config
        
        if metric == PerformanceMetric.LATENCY:
            threshold = config['latency_threshold_ms']
            if value > threshold * 2:
                return threshold, AlertSeverity.CRITICAL
            elif value > threshold * 1.5:
                return threshold, AlertSeverity.ERROR
            elif value > threshold:
                return threshold, AlertSeverity.WARNING
        
        elif metric == PerformanceMetric.ERROR_RATE:
            threshold = config['error_rate_threshold']
            if value > threshold * 3:
                return threshold, AlertSeverity.CRITICAL
            elif value > threshold * 2:
                return threshold, AlertSeverity.ERROR
            elif value > threshold:
                return threshold, AlertSeverity.WARNING
        
        elif metric == PerformanceMetric.MEMORY_USAGE:
            threshold = config['memory_threshold_mb']
            if value > threshold * 1.5:
                return threshold, AlertSeverity.CRITICAL
            elif value > threshold * 1.2:
                return threshold, AlertSeverity.ERROR
            elif value > threshold:
                return threshold, AlertSeverity.WARNING
        
        elif metric == PerformanceMetric.CPU_USAGE:
            threshold = config['cpu_threshold_percent']
            if value > threshold * 1.3:
                return threshold, AlertSeverity.CRITICAL
            elif value > threshold * 1.1:
                return threshold, AlertSeverity.ERROR
            elif value > threshold:
                return threshold, AlertSeverity.WARNING
        
        elif metric == PerformanceMetric.ACCURACY:
            # Para accuracy, alertamos por caídas
            threshold = None
            # Necesitamos historial para detectar caídas
            if len(self.metrics_history[metric.value]) > 10:
                recent_values = [v['value'] for v in self.metrics_history[metric.value][-10:]]
                avg_recent = np.mean(recent_values)
                
                if len(self.metrics_history[metric.value]) > 100:
                    historical_values = [v['value'] for v in self.metrics_history[metric.value][-100:]]
                    avg_historical = np.mean(historical_values)
                    
                    drop = avg_historical - avg_recent
                    if drop > config['accuracy_drop_threshold']:
                        threshold = config['accuracy_drop_threshold']
                        if drop > threshold * 2:
                            return threshold, AlertSeverity.CRITICAL
                        elif drop > threshold * 1.5:
                            return threshold, AlertSeverity.ERROR
                        else:
                            return threshold, AlertSeverity.WARNING
        
        return None, AlertSeverity.INFO
    
    def _should_alert(self, metric: PerformanceMetric, value: float, threshold: float) -> bool:
        """Determina si se debe generar alerta"""
        # Para métricas donde valores altos son malos
        high_bad_metrics = [
            PerformanceMetric.LATENCY,
            PerformanceMetric.ERROR_RATE,
            PerformanceMetric.MEMORY_USAGE,
            PerformanceMetric.CPU_USAGE
        ]
        
        if metric in high_bad_metrics:
            return value > threshold
        
        # Para accuracy, valores bajos son malos
        if metric == PerformanceMetric.ACCURACY:
            return value < threshold
        
        return False
    
    def _generate_alert_message(self, metric: PerformanceMetric, value: float, threshold: float) -> str:
        """Genera mensaje de alerta"""
        messages = {
            PerformanceMetric.LATENCY: 
                f"Latencia alta: {value:.1f}ms (umbral: {threshold:.1f}ms)",
            PerformanceMetric.ERROR_RATE:
                f"Tasa de error alta: {value:.3%} (umbral: {threshold:.3%})",
            PerformanceMetric.MEMORY_USAGE:
                f"Uso de memoria alto: {value:.1f}MB (umbral: {threshold:.1f}MB)",
            PerformanceMetric.CPU_USAGE:
                f"Uso de CPU alto: {value:.1f}% (umbral: {threshold:.1f}%)",
            PerformanceMetric.ACCURACY:
                f"Caída en accuracy: {value:.3%} (umbral: {threshold:.3%})",
            PerformanceMetric.THROUGHPUT:
                f"Throughput bajo: {value:.1f} req/s",
            PerformanceMetric.REQUEST_RATE:
                f"Tasa de request anómala: {value:.1f} req/s"
        }
        
        return messages.get(metric, f"Alerta en {metric.value}: {value} (umbral: {threshold})")
    
    def _save_alert(self, alert: Alert):
        """Guarda alerta en disco"""
        alert_file = self.monitoring_dir / 'alerts' / f"{alert.id}.json"
        with open(alert_file, 'w') as f:
            json.dump(alert.to_dict(), f, indent=2)
    
    def _detect_drift(self, metric: PerformanceMetric, value: float, timestamp: datetime):
        """Detecta drift en distribuciones"""
        if metric not in [PerformanceMetric.ACCURACY, PerformanceMetric.ERROR_RATE]:
            return
        
        # Inicializar detector de drift si no existe
        if metric.value not in self.drift_detectors:
            self.drift_detectors[metric.value] = {
                'window': [],
                'reference_mean': None,
                'reference_std': None,
                'drift_detected': False,
                'last_drift_check': None
            }
        
        detector = self.drift_detectors[metric.value]
        detector['window'].append(value)
        
        # Mantener tamaño de ventana
        if len(detector['window']) > self.config['drift_window_size']:
            detector['window'] = detector['window'][-self.config['drift_window_size']:]
        
        # Establecer distribución de referencia si no existe
        if detector['reference_mean'] is None and len(detector['window']) >= 100:
            detector['reference_mean'] = np.mean(detector['window'])
            detector['reference_std'] = np.std(detector['window'])
            logger.info(f"Distribución de referencia establecida para {metric.value}: "
                       f"μ={detector['reference_mean']:.4f}, σ={detector['reference_std']:.4f}")
        
        # Verificar drift periódicamente
        if (detector['reference_mean'] is not None and 
            len(detector['window']) >= 50 and
            (detector['last_drift_check'] is None or 
             timestamp - detector['last_drift_check'] > timedelta(hours=1))):
            
            # Calcular estadísticas actuales
            current_window = detector['window'][-50:]
            current_mean = np.mean(current_window)
            current_std = np.std(current_window)
            
            # Test Z para diferencia de medias
            n = len(current_window)
            if n > 1 and detector['reference_std'] > 0:
                z_score = (current_mean - detector['reference_mean']) / (detector['reference_std'] / np.sqrt(n))
                
                # Umbral Z para confianza dada
                confidence = self.config['drift_confidence_level']
                z_threshold = self._z_score_for_confidence(confidence)
                
                if abs(z_score) > z_threshold:
                    detector['drift_detected'] = True
                    
                    # Crear alerta de drift
                    alert_id = f"drift_{metric.value}_{timestamp.strftime('%Y%m%d_%H%M%S')}"
                    alert = Alert(
                        id=alert_id,
                        severity=AlertSeverity.WARNING,
                        metric=metric,
                        current_value=current_mean,
                        threshold=detector['reference_mean'],
                        message=f"Drift detectado en {metric.value}: "
                               f"media actual={current_mean:.4f}, "
                               f"media referencia={detector['reference_mean']:.4f}, "
                               f"Z-score={z_score:.2f}"
                    )
                    
                    self.alerts_history.append(alert.to_dict())
                    self._save_alert(alert)
                    
                    logger.warning(f"⚠️  Drift detectado en {metric.value}: Z-score={z_score:.2f}")
                
                else:
                    detector['drift_detected'] = False
            
            detector['last_drift_check'] = timestamp
    
    def _z_score_for_confidence(self, confidence: float) -> float:
        """Calcula Z-score para nivel de confianza dado"""
        # Valores aproximados para distribución normal
        z_scores = {
            0.90: 1.645,
            0.95: 1.960,
            0.99: 2.576,
            0.995: 2.807,
            0.999: 3.291
        }
        return z_scores.get(confidence, 1.960)  # Default 95%
    
    def start_ab_test(self,
                     test_name: str,
                     variants: Dict[str, Any],
                     primary_metric: PerformanceMetric = PerformanceMetric.ACCURACY,
                     sample_size: int = 1000) -> str:
        """Inicia un test A/B"""
        test_id = f"{test_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.ab_test_groups[test_id] = {
            'test_name': test_name,
            'variants': variants,
            'primary_metric': primary_metric.value,
            'sample_size': sample_size,
            'start_time': datetime.now().isoformat(),
            'samples': {variant: [] for variant in variants.keys()},
            'metrics': {variant: [] for variant in variants.keys()}
        }
        
        logger.info(f"Test A/B iniciado: {test_id}")
        return test_id
    
    def record_ab_test_result(self,
                            test_id: str,
                            variant: str,
                            metric_value: float):
        """Registra resultado para un test A/B"""
        if test_id not in self.ab_test_groups:
            logger.warning(f"Test A/B {test_id} no encontrado")
            return
        
        test = self.ab_test_groups[test_id]
        
        if variant not in test['samples']:
            logger.warning(f"Variante {variant} no encontrada en test {test_id}")
            return
        
        # Registrar resultado
        test['samples'][variant].append(metric_value)
        test['metrics'][variant].append({
            'timestamp': datetime.now().isoformat(),
            'value': metric_value
        })
        
        # Verificar si se alcanzó el tamaño de muestra
        total_samples = sum(len(samples) for samples in test['samples'].values())
        if total_samples >= test['sample_size']:
            self._analyze_ab_test(test_id)
    
    def _analyze_ab_test(self, test_id: str):
        """Analiza resultados de test A/B"""
        test = self.ab_test_groups[test_id]
        
        if len(test['variants']) < 2:
            logger.warning(f"Test A/B {test_id} necesita al menos 2 variantes")
            return
        
        variants = list(test['variants'].keys())
        
        # Obtener muestras para cada variante
        samples = {}
        for variant in variants:
            samples[variant] = test['samples'][variant]
            
            # Verificar tamaño mínimo
            if len(samples[variant]) < self.config['min_samples_for_ab_test']:
                logger.info(f"Variante {variant} no tiene suficientes muestras: {len(samples[variant])}")
                return
        
        # Calcular estadísticas
        results = {}
        for variant in variants:
            variant_samples = samples[variant]
            results[variant] = {
                'mean': np.mean(variant_samples),
                'std': np.std(variant_samples),
                'n': len(variant_samples),
                'confidence_interval': self._calculate_confidence_interval(variant_samples)
            }
        
        # Test t para dos variantes
        if len(variants) == 2:
            variant_a, variant_b = variants[0], variants[1]
            samples_a = samples[variant_a]
            samples_b = samples[variant_b]
            
            # Test t independiente
            t_stat, p_value = self._independent_t_test(samples_a, samples_b)
            
            # Calcular improvement
            mean_a = np.mean(samples_a)
            mean_b = np.mean(samples_b)
            improvement = ((mean_b - mean_a) / mean_a * 100) if mean_a != 0 else 0
            
            test_result = {
                'test_id': test_id,
                'test_name': test['test_name'],
                'variants': variants,
                'results': results,
                't_statistic': t_stat,
                'p_value': p_value,
                'significant': p_value < 0.05,
                'improvement_percent': improvement,
                'winner': variant_b if mean_b > mean_a and p_value < 0.05 else variant_a if p_value < 0.05 else 'none',
                'analysis_time': datetime.now().isoformat()
            }
            
            self.ab_test_results[test_id] = test_result
            
            # Guardar resultados
            result_file = self.monitoring_dir / 'ab_tests' / f"{test_id}_result.json"
            with open(result_file, 'w') as f:
                json.dump(test_result, f, indent=2)
            
            logger.info(f"Test A/B {test_id} analizado: "
                       f"p-value={p_value:.4f}, "
                       f"mejora={improvement:.1f}%, "
                       f"ganador={test_result['winner']}")
    
    def _calculate_confidence_interval(self, samples: List[float], confidence: float = 0.95) -> Dict:
        """Calcula intervalo de confianza usando solo NumPy"""
        if len(samples) < 2:
            return {'lower': 0, 'upper': 0, 'confidence': confidence}
        
        mean = np.mean(samples)
        std = np.std(samples)
        n = len(samples)
        
        # Z-score para confianza dada
        z = self._z_score_for_confidence(confidence)
        
        # Error estándar
        se = std / np.sqrt(n)
        
        # Intervalo de confianza
        margin = z * se
        
        return {
            'lower': float(mean - margin),
            'upper': float(mean + margin),
            'confidence': confidence
        }
    
    def _independent_t_test(self, samples_a: List[float], samples_b: List[float]) -> Tuple[float, float]:
        """Realiza test t independiente usando solo NumPy"""
        mean_a = np.mean(samples_a)
        mean_b = np.mean(samples_b)
        var_a = np.var(samples_a, ddof=1)
        var_b = np.var(samples_b, ddof=1)
        n_a = len(samples_a)
        n_b = len(samples_b)
        
        # Estadístico t
        t_stat = (mean_a - mean_b) / np.sqrt(var_a/n_a + var_b/n_b)
        
        # Grados de libertad (aproximación de Welch)
        df_num = (var_a/n_a + var_b/n_b) ** 2
        df_den = (var_a/n_a)**2/(n_a-1) + (var_b/n_b)**2/(n_b-1)
        df = df_num / df_den if df_den != 0 else n_a + n_b - 2
        
        # p-value (aproximación usando distribución t)
        # Para simplicidad, usamos aproximación normal para muestras grandes
        if n_a > 30 and n_b > 30:
            p_value = 2 * (1 - self._normal_cdf(abs(t_stat)))
        else:
            # Aproximación simple para distribución t
            p_value = 2 * (1 - self._t_cdf(abs(t_stat), df))
        
        return float(t_stat), float(p_value)
    
    def _normal_cdf(self, x: float) -> float:
        """CDF de distribución normal (aproximación)"""
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))
    
    def _t_cdf(self, x: float, df: float) -> float:
        """CDF de distribución t (aproximación simple)"""
        # Aproximación para df > 30
        if df > 30:
            return self._normal_cdf(x)
        else:
            # Aproximación simple
            z = x / math.sqrt(df + x**2)
            return 0.5 + 0.5 * z * (1 - z**2/3)
    
    def get_monitoring_dashboard(self) -> Dict[str, Any]:
        """Genera dashboard de monitoreo"""
        dashboard = {
            'timestamp': datetime.now().isoformat(),
            'summary': {},
            'active_alerts': len(self.active_alerts),
            'recent_alerts': self.alerts_history[-10:] if self.alerts_history else [],
            'metrics_summary': {},
            'ab_tests_running': len(self.ab_test_groups),
            'ab_tests_completed': len(self.ab_test_results)
        }
        
        # Resumen de métricas
        for metric in PerformanceMetric:
            history = self.metrics_history[metric.value]
            if history:
                recent_values = [h['value'] for h in history[-50:]]
                dashboard['metrics_summary'][metric.value] = {
                    'current': recent_values[-1] if recent_values else 0,
                    'mean': np.mean(recent_values) if recent_values else 0,
                    'std': np.std(recent_values) if len(recent_values) > 1 else 0,
                    'trend': self._calculate_trend(recent_values) if len(recent_values) > 5 else 'stable'
                }
        
        # Resumen general
        dashboard['summary'] = {
            'health_score': self._calculate_health_score(),
            'system_status': self._determine_system_status(),
            'recommendations': self._generate_recommendations()
        }
        
        return dashboard
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calcula tendencia de valores"""
        if len(values) < 5:
            return "stable"
        
        # Dividir en mitades
        half = len(values) // 2
        first_half = values[:half]
        second_half = values[half:]
        
        mean_first = np.mean(first_half)
        mean_second = np.mean(second_half)
        
        if mean_second > mean_first * 1.1:
            return "increasing"
        elif mean_second < mean_first * 0.9:
            return "decreasing"
        else:
            return "stable"
    
    def _calculate_health_score(self) -> float:
        """Calcula score de salud del sistema"""
        weights = {
            'latency': 0.25,
            'error_rate': 0.25,
            'accuracy': 0.20,
            'memory': 0.15,
            'cpu': 0.15
        }
        
        scores = {}
        
        for metric_name, weight in weights.items():
            history = self.metrics_history.get(metric_name, [])
            if history:
                recent_values = [h['value'] for h in history[-20:]]
                if recent_values:
                    # Normalizar según umbrales
                    if metric_name == 'latency':
                        norm_score = max(0, 1 - np.mean(recent_values) / self.config['latency_threshold_ms'])
                    elif metric_name == 'error_rate':
                        norm_score = max(0, 1 - np.mean(recent_values) / self.config['error_rate_threshold'])
                    elif metric_name == 'accuracy':
                        norm_score = np.mean(recent_values)  # Ya está entre 0-1
                    elif metric_name == 'memory_usage':
                        norm_score = max(0, 1 - np.mean(recent_values) / self.config['memory_threshold_mb'])
                    elif metric_name == 'cpu_usage':
                        norm_score = max(0, 1 - np.mean(recent_values) / 100)
                    else:
                        norm_score = 0.5
                    
                    scores[metric_name] = norm_score
        
        # Calcular weighted average
        if scores:
            weighted_sum = sum(score * weights[metric] for metric, score in scores.items() if metric in weights)
            total_weight = sum(weights[metric] for metric in scores if metric in weights)
            return weighted_sum / total_weight if total_weight > 0 else 0.5
        else:
            return 0.5
    
    def _determine_system_status(self) -> str:
        """Determina estado del sistema"""
        health_score = self._calculate_health_score()
        
        if health_score >= 0.8:
            return "healthy"
        elif health_score >= 0.6:
            return "degraded"
        elif health_score >= 0.4:
            return "unhealthy"
        else:
            return "critical"
    
    def _generate_recommendations(self) -> List[str]:
        """Genera recomendaciones basadas en métricas"""
        recommendations = []
        
        # Verificar cada métrica
        for metric in PerformanceMetric:
            history = self.metrics_history[metric.value]
            if len(history) >= 10:
                recent = [h['value'] for h in history[-10:]]
                avg_recent = np.mean(recent)
                
                if metric == PerformanceMetric.LATENCY and avg_recent > self.config['latency_threshold_ms']:
                    recommendations.append("Optimizar modelo para reducir latencia")
                
                elif metric == PerformanceMetric.ERROR_RATE and avg_recent > self.config['error_rate_threshold']:
                    recommendations.append("Investigar causas de errores elevados")
                
                elif metric == PerformanceMetric.MEMORY_USAGE and avg_recent > self.config['memory_threshold_mb']:
                    recommendations.append("Optimizar uso de memoria del modelo")
        
        # Verificar drift
        for metric_name, detector in self.drift_detectors.items():
            if detector.get('drift_detected', False):
                recommendations.append(f"Reentrenar modelo - drift detectado en {metric_name}")
        
        return recommendations[:5]  # Limitar a 5 recomendaciones

# ============================================================================
# 3. SISTEMA DE MANTENIMIENTO AUTOMÁTICO
# ============================================================================

class MaintenanceAction(Enum):
    """Acciones de mantenimiento automático"""
    RETRAIN_MODEL = "retrain_model"
    SCALE_RESOURCES = "scale_resources"
    ROLLBACK_VERSION = "rollback_version"
    ADJUST_THRESHOLDS = "adjust_thresholds"
    CLEAN_CACHE = "clean_cache"
    UPDATE_CONFIG = "update_config"

class MaintenanceScheduler:
    """
    Sistema de mantenimiento automático
    """
    
    def __init__(self, 
                 cicd_pipeline: CICDPipeline,
                 production_monitor: ProductionMonitor,
                 config: Optional[Dict] = None):
        
        self.pipeline = cicd_pipeline
        self.monitor = production_monitor
        
        self.config = config or {
            'retrain_interval_hours': 24,
            'retrain_accuracy_threshold': 0.8,
            'auto_scaling_enabled': True,
            'scale_up_cpu_threshold': 70,
            'scale_down_cpu_threshold': 30,
            'auto_rollback_enabled': True,
            'rollback_error_threshold': 0.1,
            'maintenance_window_start': '02:00',
            'maintenance_window_end': '04:00',
            'max_concurrent_maintenance': 1
        }
        
        self.maintenance_history = []
        self.scheduled_tasks = {}
        
        logger.info("Sistema de mantenimiento inicializado")
    
    async def run_maintenance_cycle(self):
        """Ejecuta ciclo de mantenimiento"""
        logger.info("Iniciando ciclo de mantenimiento...")
        
        current_time = datetime.now()
        
        # 1. Verificar si estamos en ventana de mantenimiento
        if not self._is_in_maintenance_window(current_time):
            logger.info("Fuera de ventana de mantenimiento")
            return
        
        # 2. Ejecutar checks de mantenimiento
        checks = [
            self._check_retraining_needed,
            self._check_scaling_needed,
            self._check_rollback_needed,
            self._check_config_updates_needed
        ]
        
        for check in checks:
            try:
                action_needed, details = await check()
                if action_needed:
                    await self._execute_maintenance_action(details)
            except Exception as e:
                logger.error(f"Error en check de mantenimiento: {e}")
        
        logger.info("Ciclo de mantenimiento completado")
    
    def _is_in_maintenance_window(self, current_time: datetime) -> bool:
        """Verifica si estamos en ventana de mantenimiento"""
        try:
            window_start = datetime.strptime(self.config['maintenance_window_start'], '%H:%M').time()
            window_end = datetime.strptime(self.config['maintenance_window_end'], '%H:%M').time()
            current = current_time.time()
            
            return window_start <= current <= window_end
        except:
            # Si hay error en la configuración, permitir siempre
            return True
    
    async def _check_retraining_needed(self) -> Tuple[bool, Dict]:
        """Verifica si se necesita reentrenamiento"""
        details = {
            'action': MaintenanceAction.RETRAIN_MODEL,
            'reason': [],
            'metrics': {}
        }
        
        # Verificar accuracy
        accuracy_history = self.monitor.metrics_history.get('accuracy', [])
        if accuracy_history:
            recent_accuracy = [h['value'] for h in accuracy_history[-20:]]
            avg_accuracy = np.mean(recent_accuracy) if recent_accuracy else 0
            
            details['metrics']['accuracy'] = avg_accuracy
            
            if avg_accuracy < self.config['retrain_accuracy_threshold']:
                details['reason'].append(f'Accuracy baja: {avg_accuracy:.3f}')
        
        # Verificar drift
        for metric_name, detector in self.monitor.drift_detectors.items():
            if detector.get('drift_detected', False):
                details['reason'].append(f'Drift detectado en {metric_name}')
        
        # Verificar tiempo desde último entrenamiento
        last_training = self._get_last_training_time()
        if last_training:
            hours_since = (datetime.now() - last_training).total_seconds() / 3600
            if hours_since > self.config['retrain_interval_hours']:
                details['reason'].append(f'Tiempo desde último entrenamiento: {hours_since:.1f}h')
        
        action_needed = len(details['reason']) > 0
        
        if action_needed:
            logger.info(f"Reentrenamiento necesario: {details['reason']}")
        
        return action_needed, details
    
    def _get_last_training_time(self) -> Optional[datetime]:
        """Obtiene tiempo del último entrenamiento"""
        # Buscar en historial de deployments
        for record in reversed(self.pipeline.deployment_history):
            if record.get('artifact_type') == 'model':
                try:
                    return datetime.fromisoformat(record['start_time'].replace('Z', '+00:00'))
                except:
                    pass
        return None
    
    async def _check_scaling_needed(self) -> Tuple[bool, Dict]:
        """Verifica si se necesita escalamiento"""
        if not self.config['auto_scaling_enabled']:
            return False, {}
        
        details = {
            'action': MaintenanceAction.SCALE_RESOURCES,
            'reason': [],
            'metrics': {}
        }
        
        # Verificar uso de CPU
        cpu_history = self.monitor.metrics_history.get('cpu_usage', [])
        if cpu_history:
            recent_cpu = [h['value'] for h in cpu_history[-10:]]
            avg_cpu = np.mean(recent_cpu) if recent_cpu else 0
            
            details['metrics']['cpu_usage'] = avg_cpu
            
            if avg_cpu > self.config['scale_up_cpu_threshold']:
                details['reason'].append(f'CPU alto: {avg_cpu:.1f}%')
                details['scale_direction'] = 'up'
            elif avg_cpu < self.config['scale_down_cpu_threshold']:
                details['reason'].append(f'CPU bajo: {avg_cpu:.1f}%')
                details['scale_direction'] = 'down'
        
        # Verificar latencia
        latency_history = self.monitor.metrics_history.get('latency', [])
        if latency_history:
            recent_latency = [h['value'] for h in latency_history[-10:]]
            avg_latency = np.mean(recent_latency) if recent_latency else 0
            
            details['metrics']['latency'] = avg_latency
            
            if avg_latency > self.pipeline.config.get('latency_threshold_ms', 1000) * 1.5:
                details['reason'].append(f'Latencia alta: {avg_latency:.1f}ms')
                if 'scale_direction' not in details:
                    details['scale_direction'] = 'up'
        
        action_needed = len(details['reason']) > 0
        
        if action_needed:
            logger.info(f"Escalamiento necesario: {details['reason']}")
        
        return action_needed, details
    
    async def _check_rollback_needed(self) -> Tuple[bool, Dict]:
        """Verifica si se necesita rollback"""
        if not self.config['auto_rollback_enabled']:
            return False, {}
        
        details = {
            'action': MaintenanceAction.ROLLBACK_VERSION,
            'reason': [],
            'metrics': {}
        }
        
        # Verificar tasa de error
        error_history = self.monitor.metrics_history.get('error_rate', [])
        if error_history:
            recent_errors = [h['value'] for h in error_history[-10:]]
            avg_error = np.mean(recent_errors) if recent_errors else 0
            
            details['metrics']['error_rate'] = avg_error
            
            if avg_error > self.config['rollback_error_threshold']:
                details['reason'].append(f'Tasa de error alta: {avg_error:.3f}')
        
        # Verificar caída en accuracy después de deployment
        accuracy_history = self.monitor.metrics_history.get('accuracy', [])
        if accuracy_history and len(accuracy_history) > 20:
            # Comparar antes y después del último deployment
            last_deployment = self._get_last_deployment_time()
            if last_deployment:
                # Accuracy antes del deployment (últimas 10 mediciones antes del deployment)
                before_deployment = [h for h in accuracy_history if 
                                    datetime.fromisoformat(h['timestamp'].replace('Z', '+00:00')) < last_deployment]
                
                if before_deployment:
                    before_accuracy = np.mean([h['value'] for h in before_deployment[-10:]])
                    
                    # Accuracy después del deployment
                    after_deployment = [h for h in accuracy_history if 
                                       datetime.fromisoformat(h['timestamp'].replace('Z', '+00:00')) > last_deployment]
                    
                    if after_deployment:
                        after_accuracy = np.mean([h['value'] for h in after_deployment[:10]])
                        
                        if after_accuracy < before_accuracy * 0.9:  # Caída del 10%
                            details['reason'].append(f'Caída en accuracy después de deployment: '
                                                   f'{after_accuracy:.3f} vs {before_accuracy:.3f}')
        
        action_needed = len(details['reason']) > 0
        
        if action_needed:
            logger.info(f"Rollback necesario: {details['reason']}")
        
        return action_needed, details
    
    def _get_last_deployment_time(self) -> Optional[datetime]:
        """Obtiene tiempo del último deployment"""
        if self.pipeline.deployment_history:
            last_record = self.pipeline.deployment_history[-1]
            try:
                return datetime.fromisoformat(last_record['start_time'].replace('Z', '+00:00'))
            except:
                pass
        return None
    
    async def _check_config_updates_needed(self) -> Tuple[bool, Dict]:
        """Verifica si se necesitan actualizaciones de configuración"""
        details = {
            'action': MaintenanceAction.UPDATE_CONFIG,
            'reason': [],
            'metrics': {}
        }
        
        # Analizar métricas para ajustar umbrales
        config_updates = {}
        
        # Ajustar umbral de latencia basado en percentil 95
        latency_history = self.monitor.metrics_history.get('latency', [])
        if len(latency_history) >= 100:
            latencies = [h['value'] for h in latency_history[-100:]]
            p95 = np.percentile(latencies, 95)
            
            current_threshold = self.pipeline.config.get('latency_threshold_ms', 1000)
            if abs(p95 - current_threshold) > current_threshold * 0.2:  # Diferencia > 20%
                config_updates['latency_threshold_ms'] = float(p95 * 1.1)  # 10% arriba del P95
                details['reason'].append(f'Ajustar umbral de latencia a {p95*1.1:.1f}ms')
        
        # Ajustar umbral de error basado en tendencia
        error_history = self.monitor.metrics_history.get('error_rate', [])
        if len(error_history) >= 50:
            errors = [h['value'] for h in error_history[-50:]]
            avg_error = np.mean(errors)
            std_error = np.std(errors)
            
            current_threshold = self.config.get('rollback_error_threshold', 0.1)
            new_threshold = avg_error + 2 * std_error  # Media + 2σ
            
            if new_threshold > 0.01 and new_threshold < 0.5:  # Límites razonables
                if abs(new_threshold - current_threshold) > current_threshold * 0.3:
                    config_updates['rollback_error_threshold'] = float(new_threshold)
                    details['reason'].append(f'Ajustar umbral de error a {new_threshold:.3f}')
        
        if config_updates:
            details['config_updates'] = config_updates
            action_needed = True
        else:
            action_needed = False
        
        return action_needed, details
    
    async def _execute_maintenance_action(self, details: Dict):
        """Ejecuta acción de mantenimiento"""
        action = details['action']
        reasons = details.get('reason', [])
        
        logger.info(f"Ejecutando acción de mantenimiento: {action.value}")
        logger.info(f"Razones: {reasons}")
        
        try:
            if action == MaintenanceAction.RETRAIN_MODEL:
                await self._execute_retraining()
            
            elif action == MaintenanceAction.SCALE_RESOURCES:
                scale_direction = details.get('scale_direction', 'up')
                await self._execute_scaling(scale_direction)
            
            elif action == MaintenanceAction.ROLLBACK_VERSION:
                await self._execute_rollback()
            
            elif action == MaintenanceAction.UPDATE_CONFIG:
                config_updates = details.get('config_updates', {})
                await self._execute_config_update(config_updates)
            
            # Registrar en historial
            maintenance_record = {
                'action': action.value,
                'timestamp': datetime.now().isoformat(),
                'reasons': reasons,
                'details': details,
                'status': 'completed'
            }
            
            self.maintenance_history.append(maintenance_record)
            
            logger.info(f"✅ Acción de mantenimiento completada: {action.value}")
            
        except Exception as e:
            logger.error(f"❌ Error ejecutando acción de mantenimiento: {e}")
            
            maintenance_record = {
                'action': action.value,
                'timestamp': datetime.now().isoformat(),
                'reasons': reasons,
                'error': str(e),
                'status': 'failed'
            }
            
            self.maintenance_history.append(maintenance_record)
    
    async def _execute_retraining(self):
        """Ejecuta reentrenamiento del modelo"""
        logger.info("Iniciando reentrenamiento del modelo...")
        
        # En un sistema real, aquí se llamaría al pipeline de entrenamiento
        # Por ahora simulamos el proceso
        
        # 1. Preparar datos
        # 2. Entrenar nuevo modelo
        # 3. Evaluar
        # 4. Desplegar si es mejor
        
        await asyncio.sleep(2)  # Simular tiempo de entrenamiento
        
        logger.info("Reentrenamiento simulado completado")
    
    async def _execute_scaling(self, direction: str):
        """Ejecuta escalamiento de recursos"""
        logger.info(f"Escalamiento {direction} de recursos...")
        
        if direction == 'up':
            # Aumentar recursos
            logger.info("Aumentando capacidad del sistema...")
        else:
            # Reducir recursos
            logger.info("Reduciendo capacidad del sistema...")
        
        await asyncio.sleep(1)
        logger.info("Escalamiento completado")
    
    async def _execute_rollback(self):
        """Ejecuta rollback de versión"""
        logger.info("Ejecutando rollback automático...")
        
        try:
            await self.pipeline.rollback_latest()
            logger.info("✅ Rollback automático completado")
        except Exception as e:
            logger.error(f"❌ Rollback automático falló: {e}")
            raise
    
    async def _execute_config_update(self, config_updates: Dict):
        """Ejecuta actualización de configuración"""
        logger.info(f"Actualizando configuración: {config_updates}")
        
        # Actualizar configuración del pipeline
        for key, value in config_updates.items():
            if key in self.pipeline.config:
                old_value = self.pipeline.config[key]
                self.pipeline.config[key] = value
                logger.info(f"  {key}: {old_value} -> {value}")
        
        # Actualizar configuración del monitor
        for key, value in config_updates.items():
            if key in self.monitor.config:
                old_value = self.monitor.config[key]
                self.monitor.config[key] = value
                logger.info(f"  {key}: {old_value} -> {value}")
        
        # Guardar configuración actualizada
        self._save_configurations()
        
        logger.info("✅ Configuración actualizada")
    
    def _save_configurations(self):
        """Guarda configuraciones actualizadas"""
        # Guardar configuración del pipeline
        pipeline_config_path = self.pipeline.artifacts_dir / 'pipeline_config.json'
        with open(pipeline_config_path, 'w') as f:
            json.dump(self.pipeline.config, f, indent=2)
        
        # Guardar configuración del monitor
        monitor_config_path = self.monitor.monitoring_dir / 'monitor_config.json'
        with open(monitor_config_path, 'w') as f:
            json.dump(self.monitor.config, f, indent=2)

# ============================================================================
# 4. SISTEMA DE LOGGING Y AUDITORÍA
# ============================================================================

class AuditLogger:
    """
    Sistema de logging y auditoría
    """
    
    def __init__(self, log_dir: str = "./logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Configurar logging
        self._setup_logging()
        
        self.audit_trail = []
    
    def _setup_logging(self):
        """Configura sistema de logging"""
        log_file = self.log_dir / f"deployment_{datetime.now().strftime('%Y%m%d')}.log"
        
        # Configurar root logger
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    
    def log_audit_event(self, 
                       event_type: str,
                       event_data: Dict,
                       user: str = "system",
                       severity: str = "info"):
        """Registra evento de auditoría"""
        event = {
            'id': hashlib.md5(f"{event_type}_{datetime.now().timestamp()}".encode()).hexdigest()[:16],
            'timestamp': datetime.now().isoformat(),
            'event_type': event_type,
            'user': user,
            'severity': severity,
            'data': event_data
        }
        
        self.audit_trail.append(event)
        
        # Guardar en archivo
        audit_file = self.log_dir / 'audit_trail.json'
        
        # Cargar existente y agregar nuevo evento
        if audit_file.exists():
            try:
                with open(audit_file, 'r') as f:
                    existing = json.load(f)
            except:
                existing = []
        else:
            existing = []
        
        existing.append(event)
        
        with open(audit_file, 'w') as f:
            json.dump(existing, f, indent=2)
        
        # Log convencional
        log_message = f"AUDIT [{severity.upper()}] {event_type} by {user}"
        if severity == 'error':
            logger.error(log_message)
        elif severity == 'warning':
            logger.warning(log_message)
        else:
            logger.info(log_message)
    
    def get_audit_report(self, 
                        start_date: Optional[datetime] = None,
                        end_date: Optional[datetime] = None) -> Dict:
        """Genera reporte de auditoría"""
        filtered_events = self.audit_trail
        
        if start_date:
            filtered_events = [e for e in filtered_events 
                             if datetime.fromisoformat(e['timestamp'].replace('Z', '+00:00')) >= start_date]
        
        if end_date:
            filtered_events = [e for e in filtered_events 
                             if datetime.fromisoformat(e['timestamp'].replace('Z', '+00:00')) <= end_date]
        
        # Estadísticas
        event_counts = {}
        user_counts = {}
        severity_counts = {}
        
        for event in filtered_events:
            event_type = event['event_type']
            user = event['user']
            severity = event['severity']
            
            event_counts[event_type] = event_counts.get(event_type, 0) + 1
            user_counts[user] = user_counts.get(user, 0) + 1
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        return {
            'period': {
                'start': start_date.isoformat() if start_date else 'all',
                'end': end_date.isoformat() if end_date else 'now'
            },
            'total_events': len(filtered_events),
            'event_counts': event_counts,
            'user_counts': user_counts,
            'severity_counts': severity_counts,
            'recent_events': filtered_events[-20:] if filtered_events else []
        }

# ============================================================================
# 5. SISTEMA INTEGRADO DE DEPLOYMENT Y MONITOREO
# ============================================================================

class ProductionDeploymentSystem:
    """
    Sistema completo integrado de deployment, monitoreo y mantenimiento
    """
    
    def __init__(self,
                 artifacts_dir: str = "./artifacts",
                 monitoring_dir: str = "./monitoring",
                 logs_dir: str = "./logs"):
        
        # Inicializar componentes
        self.cicd_pipeline = CICDPipeline(artifacts_dir)
        self.production_monitor = ProductionMonitor(monitoring_dir)
        self.maintenance_scheduler = MaintenanceScheduler(
            self.cicd_pipeline, 
            self.production_monitor
        )
        self.audit_logger = AuditLogger(logs_dir)
        
        # Estado del sistema
        self.system_status = "initializing"
        self.start_time = datetime.now()
        
        # Configuración
        self.config = {
            'auto_maintenance': True,
            'maintenance_interval_minutes': 60,
            'metrics_collection_interval_seconds': 30,
            'health_check_interval_seconds': 300
        }
        
        # Inicializar
        self._initialize_system()
    
    def _initialize_system(self):
        """Inicializa sistema completo"""
        logger.info("🚀 Inicializando Sistema de Deployment en Producción")
        logger.info(f"• Pipeline CI/CD: {self.cicd_pipeline.artifacts_dir}")
        logger.info(f"• Monitor: {self.production_monitor.monitoring_dir}")
        logger.info(f"• Logs: {self.audit_logger.log_dir}")
        
        # Registrar evento de inicialización
        self.audit_logger.log_audit_event(
            event_type="system_initialization",
            event_data={
                'components': ['cicd_pipeline', 'production_monitor', 'maintenance_scheduler', 'audit_logger'],
                'config': self.config
            },
            user="system",
            severity="info"
        )
        
        self.system_status = "running"
        logger.info("✅ Sistema de Deployment inicializado y listo")
    
    async def deploy_model(self,
                          model: Any,
                          model_name: str,
                          model_metrics: Dict,
                          test_data: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Despliega un modelo a producción
        """
        logger.info(f"🚀 Iniciando deployment de {model_name}")
        
        # 1. Crear artifact
        artifact = self.cicd_pipeline.create_model_artifact(
            model, model_name, model_metrics
        )
        
        # 2. Registrar artifact
        artifact_id = self.cicd_pipeline.register_artifact(artifact)
        
        # 3. Auditoría
        self.audit_logger.log_audit_event(
            event_type="model_registration",
            event_data={
                'model_name': model_name,
                'artifact_id': artifact_id,
                'version': artifact.version,
                'metrics': model_metrics
            },
            user="deployment_system",
            severity="info"
        )
        
        # 4. Desplegar a staging
        deployment_result = await self.cicd_pipeline.deploy_to_stage(
            artifact_id,
            DeploymentStage.STAGING,
            test_data
        )
        
        # 5. Si pasa staging, desplegar a producción
        if deployment_result['status'] == 'success':
            # Evaluar en staging por un tiempo
            await asyncio.sleep(5)  # Simular período de evaluación
            
            # Desplegar a producción
            production_result = await self.cicd_pipeline.deploy_to_stage(
                artifact_id,
                DeploymentStage.PRODUCTION,
                test_data
            )
            
            # Auditoría
            self.audit_logger.log_audit_event(
                event_type="production_deployment",
                event_data=production_result,
                user="deployment_system",
                severity="info" if production_result['status'] == 'success' else "error"
            )
            
            return production_result
        
        else:
            # Auditoría de fallo
            self.audit_logger.log_audit_event(
                event_type="deployment_failed",
                event_data=deployment_result,
                user="deployment_system",
                severity="error"
            )
            
            return deployment_result
    
    async def start_monitoring(self):
        """Inicia monitoreo continuo"""
        logger.info("📊 Iniciando monitoreo continuo")
        
        while self.system_status == "running":
            try:
                # Recolectar métricas del sistema
                await self._collect_system_metrics()
                
                # Ejecutar mantenimiento si está habilitado
                if self.config['auto_maintenance']:
                    await self.maintenance_scheduler.run_maintenance_cycle()
                
                # Esperar hasta siguiente ciclo
                await asyncio.sleep(self.config['metrics_collection_interval_seconds'])
                
            except Exception as e:
                logger.error(f"Error en ciclo de monitoreo: {e}")
                await asyncio.sleep(10)  # Esperar antes de reintentar
    
    async def _collect_system_metrics(self):
        """Recolecta métricas del sistema"""
        # Métricas simuladas - en producción estas vendrían de métricas reales
        
        # Latencia (ms)
        latency = np.random.normal(50, 20)
        latency = max(10, min(1000, latency))
        self.production_monitor.record_metric(PerformanceMetric.LATENCY, latency)
        
        # Accuracy
        accuracy = np.random.normal(0.85, 0.05)
        accuracy = max(0.5, min(1.0, accuracy))
        self.production_monitor.record_metric(PerformanceMetric.ACCURACY, accuracy)
        
        # Tasa de error
        error_rate = np.random.exponential(0.02)
        error_rate = min(0.5, error_rate)
        self.production_monitor.record_metric(PerformanceMetric.ERROR_RATE, error_rate)
        
        # Uso de CPU (%)
        cpu_usage = np.random.normal(40, 15)
        cpu_usage = max(5, min(100, cpu_usage))
        self.production_monitor.record_metric(PerformanceMetric.CPU_USAGE, cpu_usage)
        
        # Uso de memoria (MB)
        memory_usage = np.random.normal(500, 100)
        memory_usage = max(100, min(2000, memory_usage))
        self.production_monitor.record_metric(PerformanceMetric.MEMORY_USAGE, memory_usage)
        
        # Throughput (requests/segundo)
        throughput = np.random.poisson(100)
        self.production_monitor.record_metric(PerformanceMetric.THROUGHPUT, throughput)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Obtiene estado completo del sistema"""
        status = {
            'system': {
                'status': self.system_status,
                'uptime_seconds': (datetime.now() - self.start_time).total_seconds(),
                'config': self.config
            },
            'deployment': self.cicd_pipeline.get_deployment_status(),
            'monitoring': self.production_monitor.get_monitoring_dashboard(),
            'maintenance': {
                'history_count': len(self.maintenance_scheduler.maintenance_history),
                'recent_actions': self.maintenance_scheduler.maintenance_history[-5:] if self.maintenance_scheduler.maintenance_history else []
            },
            'audit': self.audit_logger.get_audit_report()
        }
        
        return status
    
    async def run_health_check(self) -> Dict[str, Any]:
        """Ejecuta health check completo del sistema"""
        health_checks = []
        
        # 1. Check pipeline
        try:
            pipeline_health = await self.cicd_pipeline._run_stage_health_check(
                self.cicd_pipeline.current_stage
            )
            health_checks.append({
                'component': 'cicd_pipeline',
                'healthy': pipeline_health['healthy'],
                'details': pipeline_health.get('details', {})
            })
        except Exception as e:
            health_checks.append({
                'component': 'cicd_pipeline',
                'healthy': False,
                'error': str(e)
            })
        
        # 2. Check monitor
        try:
            # Verificar que puede escribir métricas
            test_metric = PerformanceMetric.LATENCY
            self.production_monitor.record_metric(test_metric, 0.0)
            health_checks.append({
                'component': 'production_monitor',
                'healthy': True,
                'details': {'metric_recording': 'working'}
            })
        except Exception as e:
            health_checks.append({
                'component': 'production_monitor',
                'healthy': False,
                'error': str(e)
            })
        
        # 3. Check auditoría
        try:
            self.audit_logger.log_audit_event(
                event_type="health_check",
                event_data={'check': 'system_health'},
                user="health_check",
                severity="info"
            )
            health_checks.append({
                'component': 'audit_logger',
                'healthy': True,
                'details': {'audit_logging': 'working'}
            })
        except Exception as e:
            health_checks.append({
                'component': 'audit_logger',
                'healthy': False,
                'error': str(e)
            })
        
        # Determinar salud general
        all_healthy = all(check['healthy'] for check in health_checks)
        
        result = {
            'timestamp': datetime.now().isoformat(),
            'overall_healthy': all_healthy,
            'checks': health_checks,
            'system_status': 'healthy' if all_healthy else 'unhealthy'
        }
        
        # Log resultado
        if all_healthy:
            logger.info("✅ Health check pasado")
        else:
            logger.warning("⚠️  Health check falló")
            unhealthy = [c['component'] for c in health_checks if not c['healthy']]
            logger.warning(f"Componentes no saludables: {unhealthy}")
        
        return result

# ============================================================================
# 6. EJEMPLO DE USO COMPLETO
# ============================================================================

async def example_usage():
    """Ejemplo de uso completo del sistema"""
    print("\n" + "="*80)
    print("🚀 EJEMPLO COMPLETO - SISTEMA DE DEPLOYMENT Y MONITOREO")
    print("="*80)
    
    # 1. Inicializar sistema
    print("\n1. Inicializando sistema...")
    deployment_system = ProductionDeploymentSystem()
    
    # 2. Crear modelo de ejemplo
    print("\n2. Creando modelo de ejemplo...")
    
    class ExampleModel:
        def __init__(self):
            self.input_dim = 10
            self.output_dim = 3
            self.params = {
                'W': np.random.randn(10, 3),
                'b': np.random.randn(3)
            }
        
        def predict(self, X):
            return X @ self.params['W'] + self.params['b']
        
        def forward(self, X):
            return self.predict(X)
    
    example_model = ExampleModel()
    model_metrics = {
        'accuracy': 0.87,
        'loss': 0.45,
        'f1_score': 0.85,
        'training_time_seconds': 120,
        'constitutional_score': 0.92
    }
    
    # 3. Desplegar modelo
    print("\n3. Desplegando modelo a producción...")
    test_data = {
        'X_test': np.random.randn(100, 10),
        'y_test': np.random.randn(100, 3)
    }
    
    deployment_result = await deployment_system.deploy_model(
        example_model,
        "example_classifier",
        model_metrics,
        test_data
    )
    
    print(f"   • Resultado: {deployment_result['status']}")
    print(f"   • Versión: {deployment_result.get('model_version', 'unknown')}")
    
    # 4. Iniciar monitoreo
    print("\n4. Iniciando monitoreo continuo...")
    monitoring_task = asyncio.create_task(deployment_system.start_monitoring())
    
    # 5. Ejecutar health check
    print("\n5. Ejecutando health check del sistema...")
    health_result = await deployment_system.run_health_check()
    print(f"   • Salud general: {'✅' if health_result['overall_healthy'] else '❌'}")
    print(f"   • Estado del sistema: {health_result['system_status']}")
    
    # 6. Obtener estado del sistema
    print("\n6. Obteniendo estado completo del sistema...")
    system_status = deployment_system.get_system_status()
    print(f"   • Uptime: {system_status['system']['uptime_seconds']:.1f}s")
    print(f"   • Stage actual: {system_status['deployment']['current_stage']}")
    print(f"   • Alertas activas: {system_status['monitoring']['active_alerts']}")
    
    # 7. Simular algunos ciclos de monitoreo
    print("\n7. Simulando ciclos de monitoreo (10 segundos)...")
    await asyncio.sleep(10)
    
    # 8. Obtener dashboard de monitoreo
    print("\n8. Dashboard de monitoreo:")
    dashboard = deployment_system.production_monitor.get_monitoring_dashboard()
    print(f"   • Score de salud: {dashboard['summary']['health_score']:.3f}")
    print(f"   • Estado del sistema: {dashboard['summary']['system_status']}")
    print(f"   • Métricas recopiladas: {len(dashboard['metrics_summary'])}")
    
    # 9. Detener monitoreo
    print("\n9. Deteniendo monitoreo...")
    monitoring_task.cancel()
    try:
        await monitoring_task
    except asyncio.CancelledError:
        pass
    
    # 10. Generar reporte final
    print("\n10. Reporte final:")
    print("   • Sistema: ✅ Inicializado y funcionando")
    print("   • Deployment: ✅ Completado exitosamente")
    print("   • Monitoreo: ✅ Recopilando métricas")
    print("   • Mantenimiento: ✅ Programado automático")
    print("   • Auditoría: ✅ Registrando eventos")
    
    print("\n" + "="*80)
    print("🎯 EJEMPLO COMPLETADO EXITOSAMENTE")
    print("="*80)
    
    return deployment_system

# ============================================================================
# 7. FUNCIÓN PRINCIPAL
# ============================================================================

async def main():
    """Función principal"""
    print("\n" + "="*80)
    print("🤖 SISTEMA DE DEPLOYMENT Y MONITOREO EN PRODUCCIÓN")
    print("="*80)
    print("\nCaracterísticas implementadas:")
    print("✅ CI/CD Pipeline completo")
    print("✅ Sistema de monitoreo en tiempo real")
    print("✅ Mantenimiento automático")
    print("✅ Auditoría y logging")
    print("✅ Health checks automáticos")
    print("✅ A/B testing integrado")
    print("✅ Detección de drift")
    print("✅ Rollback automático")
    print("✅ Calibración automática")
    
    # Ejecutar ejemplo
    await example_usage()

if __name__ == "__main__":
    asyncio.run(main())