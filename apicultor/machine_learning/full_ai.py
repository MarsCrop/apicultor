# === integrated_module.py ===
"""
Módulo integrador que importa y conecta TODOS tus módulos existentes.
NO crea funciones nuevas - solo importa y organiza lo que ya tienes.
"""
import numpy as np
from typing import Dict, List, Any, Optional
import asyncio
from datetime import datetime
import logging

# ============================================================================
# 1. IMPORTAR TODOS TUS MÓDULOS EXISTENTES POR NOMBRE
# ============================================================================

# Tu sistema de monitoreo existente
from monitor import PostTrainingMonitor, Alert, AlertSeverity

# Tu sistema de Reward Model existente  
# (Asumiendo que tienes reward.py con una clase RewardModel)
try:
    from reward import RewardModel
    REWARD_MODEL_AVAILABLE = True
except ImportError:
    REWARD_MODEL_AVAILABLE = False
    print("⚠️  reward.py no encontrado. Algunas funciones no estarán disponibles.")

# Tu sistema de intervenciones existente (intervene.py)
try:
    from intervene import (
        generate_imputed_data,
        define_adversary_data, 
        scan_data_leakage,
        DiscriminationError,
        productionize,
        define_weight,
        forward_perturbation,
        perturb_feature,
        GridSearch
    )
    INTERVENE_MODULE_AVAILABLE = True
except ImportError:
    INTERVENE_MODULE_AVAILABLE = False
    print("⚠️  intervene.py no encontrado. Algunas funciones no estarán disponibles.")

# Tu sistema de cross-validation existente (cross_validation.py)
try:
    from cross_validation import (
        generate_imputed_data as cv_generate_imputed_data,
        define_adversary_data as cv_define_adversary_data,
        scan_data_leakage as cv_scan_data_leakage,
        GridSearch as cv_GridSearch
    )
    CROSS_VALIDATION_AVAILABLE = True
except ImportError:
    CROSS_VALIDATION_AVAILABLE = False
    print("⚠️  cross_validation.py no encontrado. Algunas funciones no estarán disponibles.")

# Tu sistema de stress testing existente (stress.py)
try:
    from stress import (
        generate_imputed_data as stress_generate_imputed_data,
        define_adversary_data as stress_define_adversary_data,
        scan_data_leakage as stress_scan_data_leakage,
        GridSearch as stress_GridSearch
    )
    STRESS_TESTING_AVAILABLE = True
except ImportError:
    STRESS_TESTING_AVAILABLE = False
    print("⚠️  stress.py no encontrado. Algunas funciones no estarán disponibles.")

# Tu sistema de métricas existente (metrics.py)
try:
    from metrics import (
        generate_imputed_data as metrics_generate_imputed_data,
        define_adversary_data as metrics_define_adversary_data,
        scan_data_leakage as metrics_scan_data_leakage,
        GridSearch as metrics_GridSearch
    )
    METRICS_MODULE_AVAILABLE = True
except ImportError:
    METRICS_MODULE_AVAILABLE = False
    print("⚠️  metrics.py no encontrado. Algunas funciones no estarán disponibles.")

# Tu sistema de deployment existente (deploy.py)
try:
    from deploy import (
        generate_imputed_data as deploy_generate_imputed_data,
        define_adversary_data as deploy_define_adversary_data,
        scan_data_leakage as deploy_scan_data_leakage,
        GridSearch as deploy_GridSearch
    )
    DEPLOY_MODULE_AVAILABLE = True
except ImportError:
    DEPLOY_MODULE_AVAILABLE = False
    print("⚠️  deploy.py no encontrado. Algunas funciones no estarán disponibles.")

# Tu sistema de integración existente (integration_layer.py)
try:
    from integration_layer import PostTrainingIntegration, DeploymentManager
    INTEGRATION_LAYER_AVAILABLE = True
except ImportError:
    INTEGRATION_LAYER_AVAILABLE = False
    print("⚠️  integration_layer.py no encontrado. Algunas funciones no estarán disponibles.")

# Tus módulos de fairness, explain, attention, etc.
try:
    from fairness import p_rule, ind_fairness, BEC, BTC
    from explain import explain
    from attention import continuous_decode, continuous_multi_lorax
    from dependency import dropout, gather_layers_outputs
    FAIRNESS_MODULES_AVAILABLE = True
except ImportError:
    FAIRNESS_MODULES_AVAILABLE = False
    print("⚠️  Módulos de fairness/explain/attention no encontrados.")

# ============================================================================
# 2. SISTEMA DE CONSTITUTIONAL AI BASADO EN TUS MÓDULOS EXISTENTES
# ============================================================================

class InterventionBasedConstitutionalAI:
    """
    Constitutional AI BASADO EN TUS MÓDULOS EXISTENTES.
    Usa funciones de tus archivos intervene.py, cross_validation.py, etc.
    NO crea nuevas funciones - solo usa las que ya tienes.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {
            'compliance_threshold': 0.8,
            'critical_failure_threshold': 0.5,
            'enable_auto_remediation': True,
            'max_intervention_history': 1000,
            'evaluation_timeout_seconds': 30
        }
        
        # Historial de evaluaciones usando TUS funciones
        self.intervention_history = []
        self.evaluation_history = []
        
        # Cargar funciones de TUS módulos
        self._load_existing_functions()
    
    def _load_existing_functions(self):
        """Carga funciones de tus módulos existentes"""
        self.available_functions = {
            'intervene': INTERVENE_MODULE_AVAILABLE,
            'cross_validation': CROSS_VALIDATION_AVAILABLE,
            'stress_testing': STRESS_TESTING_AVAILABLE,
            'metrics': METRICS_MODULE_AVAILABLE,
            'deploy': DEPLOY_MODULE_AVAILABLE,
            'fairness': FAIRNESS_MODULES_AVAILABLE
        }
    
    async def evaluate_with_existing_modules(self, 
                                           model, 
                                           X: np.ndarray, 
                                           y: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Ejecuta evaluación usando EXCLUSIVAMENTE tus módulos existentes
        """
        evaluation_results = {
            'timestamp': datetime.now().isoformat(),
            'modules_used': [],
            'results': {},
            'overall_status': 'unknown'
        }
        
        print(f"\n{'='*80}")
        print("CONSTITUTIONAL AI EVALUATION (USANDO TUS MÓDULOS EXISTENTES)")
        print(f"{'='*80}")
        
        # 1. Usar GridSearch de tu módulo cross_validation.py
        if self.available_functions['cross_validation']:
            print("\n1. 📊 EJECUTANDO GridSearch DE cross_validation.py...")
            try:
                # Usar TU función GridSearch exactamente como está
                grid_context = await cv_GridSearch(
                    model=model,
                    features=X,
                    targets=y if y is not None else np.zeros(len(X)),
                    Cs=[0.1, 1.0, 10.0],
                    reg_params=[0.01, 0.1, 1.0],
                    kernel_configs=[('linear', 1), ('rbf', 0.5)],
                    last=True,
                    criteria='default',
                    intersects=[0, 1],
                    logical=[True, False],
                    regression=y is not None and len(y.shape) > 1,
                    model_name='constitutional_eval'
                )
                evaluation_results['results']['grid_search'] = {
                    'context_shape': grid_context.shape if hasattr(grid_context, 'shape') else str(type(grid_context)),
                    'success': True
                }
                evaluation_results['modules_used'].append('cross_validation.GridSearch')
                print(f"   ✅ GridSearch completado")
            except Exception as e:
                evaluation_results['results']['grid_search'] = {
                    'error': str(e),
                    'success': False
                }
                print(f"   ❌ GridSearch error: {e}")
        
        # 2. Usar scan_data_leakage de tu módulo intervene.py
        if self.available_functions['intervene'] and y is not None:
            print("\n2. 🔍 EJECUTANDO scan_data_leakage DE intervene.py...")
            try:
                # Usar TU función scan_data_leakage exactamente como está
                if len(y) > 0:
                    wpr, correctly_decoded, leaking_targets, leaking_idxs, error = scan_data_leakage(
                        y[:100] if len(y) > 100 else y,
                        y[:100] if len(y) > 100 else y,
                        wrong_yrate=0.2,
                        error_threshold=0.1
                    )
                    evaluation_results['results']['data_leakage'] = {
                        'wrong_prediction_rate': float(wpr),
                        'correctly_decoded_samples': len(correctly_decoded) if hasattr(correctly_decoded, '__len__') else 0,
                        'leaking_samples': len(leaking_targets) if hasattr(leaking_targets, '__len__') else 0,
                        'success': True
                    }
                    evaluation_results['modules_used'].append('intervene.scan_data_leakage')
                    print(f"   ✅ Data leakage check: {wpr*100:.1f}% wrong prediction rate")
            except Exception as e:
                evaluation_results['results']['data_leakage'] = {
                    'error': str(e),
                    'success': False
                }
                print(f"   ❌ Data leakage check error: {e}")
        
        # 3. Usar define_adversary_data de tu módulo intervene.py
        if self.available_functions['intervene'] and y is not None:
            print("\n3. ⚔️ EJECUTANDO define_adversary_data DE intervene.py...")
            try:
                # Usar TU función define_adversary_data exactamente como está
                if len(y) > 0:
                    adversary_data, adversary_targets, prev_w, prev_b, adv_w, adv_b = define_adversary_data(
                        dataset=X[:100] if len(X) > 100 else X,
                        y=y[:100] if len(y) > 100 else y,
                        unequal_treatment_factor=0.2,
                        hyp='mean',
                        impute=False,
                        regression=y is not None and len(y.shape) > 1
                    )
                    evaluation_results['results']['adversary_data'] = {
                        'adversary_data_shape': adversary_data.shape if hasattr(adversary_data, 'shape') else str(type(adversary_data)),
                        'adversary_targets_shape': adversary_targets.shape if hasattr(adversary_targets, 'shape') else str(type(adversary_targets)),
                        'success': True
                    }
                    evaluation_results['modules_used'].append('intervene.define_adversary_data')
                    print(f"   ✅ Adversary data generado")
            except Exception as e:
                evaluation_results['results']['adversary_data'] = {
                    'error': str(e),
                    'success': False
                }
                print(f"   ❌ Adversary data error: {e}")
        
        # 4. Usar p_rule de tu módulo fairness.py
        if self.available_functions['fairness'] and y is not None and hasattr(model, 'predict'):
            print("\n4. ⚖️ EJECUTANDO p_rule DE fairness.py...")
            try:
                # Usar TU función p_rule exactamente como está
                if len(y) > 0:
                    predictions = model.predict(X[:100]) if hasattr(model, 'predict') else np.zeros(100)
                    p_rule_value = p_rule(
                        predictions,
                        y[:100] if len(y) > 100 else y,
                        model.w if hasattr(model, 'w') else np.ones_like(predictions),
                        X[:100] if len(X) > 100 else X,
                        predictions,  # Usando predictions como proba
                        thresh=1e-4
                    )
                    evaluation_results['results']['fairness_p_rule'] = {
                        'p_rule_value': float(p_rule_value) if isinstance(p_rule_value, (int, float, np.number)) else str(p_rule_value),
                        'success': True
                    }
                    evaluation_results['modules_used'].append('fairness.p_rule')
                    print(f"   ✅ p_rule calculado: {p_rule_value}")
            except Exception as e:
                evaluation_results['results']['fairness_p_rule'] = {
                    'error': str(e),
                    'success': False
                }
                print(f"   ❌ p_rule error: {e}")
        
        # 5. Usar explain de tu módulo explain.py
        if self.available_functions['fairness'] and y is not None:
            print("\n5. 🧠 EJECUTANDO explain DE explain.py...")
            try:
                # Usar TU función explain exactamente como está
                if len(y) > 10:
                    pex, cex, vis = explain(
                        model,
                        X[:10] if len(X) > 10 else X,
                        y[:10] if len(y) > 10 else y,
                        criteria='default',
                        intersects=[0, 1],
                        logical=[True, False]
                    )
                    evaluation_results['results']['explain'] = {
                        'parent_explanation': str(pex)[:100] if pex else 'None',
                        'child_explanation': str(cex)[:100] if cex else 'None',
                        'visualization': 'Available' if vis else 'None',
                        'success': True
                    }
                    evaluation_results['modules_used'].append('explain.explain')
                    print(f"   ✅ Explain completado")
            except Exception as e:
                evaluation_results['results']['explain'] = {
                    'error': str(e),
                    'success': False
                }
                print(f"   ❌ Explain error: {e}")
        
        # 6. Determinar estado general basado en resultados
        successful_checks = sum(1 for r in evaluation_results['results'].values() if r.get('success', False))
        total_checks = len(evaluation_results['results'])
        
        if total_checks > 0:
            success_rate = successful_checks / total_checks
            if success_rate >= self.config['compliance_threshold']:
                evaluation_results['overall_status'] = 'compliant'
            elif success_rate >= self.config['critical_failure_threshold']:
                evaluation_results['overall_status'] = 'needs_improvement'
            else:
                evaluation_results['overall_status'] = 'non_compliant'
            
            evaluation_results['success_rate'] = float(success_rate)
            evaluation_results['successful_checks'] = successful_checks
            evaluation_results['total_checks'] = total_checks
        
        # Guardar en historial
        self.evaluation_history.append(evaluation_results)
        if len(self.evaluation_history) > self.config['max_intervention_history']:
            self.evaluation_history = self.evaluation_history[-self.config['max_intervention_history']:]
        
        print(f"\n{'='*80}")
        print("RESUMEN DE EVALUACIÓN CONSTITUCIONAL")
        print(f"{'='*80}")
        print(f"• Módulos usados: {len(evaluation_results['modules_used'])}")
        print(f"• Checks exitosos: {successful_checks}/{total_checks}")
        print(f"• Tasa de éxito: {evaluation_results.get('success_rate', 0)*100:.1f}%")
        print(f"• Estado general: {evaluation_results['overall_status']}")
        print(f"{'='*80}")
        
        return evaluation_results

# ============================================================================
# 3. SISTEMA INTEGRADO QUE USA TUS MÓDULOS EXISTENTES
# ============================================================================

class IntegratedAISystem:
    """
    Sistema AI que INTEGRA TODOS tus módulos existentes.
    NO crea nuevas funciones - solo organiza y usa las que ya tienes.
    """
    
    def __init__(self, 
                 base_model: Any,
                 model_name: str = "IntegratedAISystem",
                 config: Optional[Dict] = None):
        
        self.base_model = base_model
        self.model_name = model_name
        
        # Inicializar todos tus sistemas existentes
        print(f"\n{'='*80}")
        print(f"INICIALIZANDO SISTEMA INTEGRADO CON TUS MÓDULOS")
        print(f"{'='*80}")
        
        # 1. Tu sistema de monitoreo
        print("\n1. 📊 CARGANDO PostTrainingMonitor DE monitor.py...")
        self.monitor = PostTrainingMonitor(model_name=model_name)
        print(f"   ✅ Sistema de monitoreo cargado")
        
        # 2. Tu sistema de Reward Model (si está disponible)
        if REWARD_MODEL_AVAILABLE:
            print("\n2. 🏆 CARGANDO RewardModel DE reward.py...")
            self.reward_model = RewardModel()
            print(f"   ✅ Reward Model cargado")
        else:
            self.reward_model = None
            print(f"   ⚠️  Reward Model NO disponible")
        
        # 3. Constitutional AI basado en tus módulos
        print("\n3. ⚖️ CARGANDO Constitutional AI BASADO EN TUS MÓDULOS...")
        self.constitutional_ai = InterventionBasedConstitutionalAI(config=config)
        print(f"   ✅ Constitutional AI cargado")
        
        # 4. Tu sistema de integración (si está disponible)
        if INTEGRATION_LAYER_AVAILABLE:
            print("\n4. 🔗 CARGANDO PostTrainingIntegration DE integration_layer.py...")
            # Necesitarías pasar tus módulos existentes aquí
            self.integrator = PostTrainingIntegration(existing_modules={})
            print(f"   ✅ Integration layer cargada")
        else:
            self.integrator = None
        
        print(f"\n{'='*80}")
        print("✅ SISTEMA INTEGRADO INICIALIZADO")
        print(f"{'='*80}")
        print(f"• Modelo base: {model_name}")
        print(f"• Monitoreo: ✅ Cargado")
        print(f"• Reward Model: {'✅ Cargado' if REWARD_MODEL_AVAILABLE else '⚠️ No disponible'}")
        print(f"• Constitutional AI: ✅ Cargado")
        print(f"• Integration Layer: {'✅ Cargado' if INTEGRATION_LAYER_AVAILABLE else '⚠️ No disponible'}")
        print(f"• Módulos de fairness/explain: {'✅ Cargados' if FAIRNESS_MODULES_AVAILABLE else '⚠️ No disponibles'}")
        print(f"{'='*80}")
    
    async def run_complete_pipeline(self,
                                   X: np.ndarray,
                                   y: Optional[np.ndarray] = None,
                                   sample_size: int = 100) -> Dict[str, Any]:
        """
        Ejecuta el pipeline completo usando TUS módulos existentes
        """
        pipeline_results = {
            'pipeline_id': f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'timestamp': datetime.now().isoformat(),
            'model_name': self.model_name,
            'stages': {},
            'overall_status': 'unknown'
        }
        
        print(f"\n{'='*80}")
        print(f"🚀 EJECUTANDO PIPELINE COMPLETO CON TUS MÓDULOS")
        print(f"{'='*80}")
        
        # 1. Monitoreo con TU sistema de monitor.py
        print("\n1. 📊 EJECUTANDO MONITOREO (PostTrainingMonitor DE monitor.py)...")
        try:
            monitoring_result = await self.monitor.monitor_model(
                model_pipeline={'base_model': self.base_model},
                X_sample=X[:sample_size],
                y_sample=y[:sample_size] if y is not None else None,
                inference_count=100
            )
            pipeline_results['stages']['monitoring'] = monitoring_result
            print(f"   ✅ Monitoreo completado")
        except Exception as e:
            pipeline_results['stages']['monitoring'] = {'error': str(e)}
            print(f"   ❌ Error en monitoreo: {e}")
        
        # 2. Evaluación Constitutional con TUS módulos
        print("\n2. ⚖️ EJECUTANDO EVALUACIÓN CONSTITUCIONAL (TUS MÓDULOS)...")
        constitutional_result = await self.constitutional_ai.evaluate_with_existing_modules(
            model=self.base_model,
            X=X[:sample_size],
            y=y[:sample_size] if y is not None else None
        )
        pipeline_results['stages']['constitutional_evaluation'] = constitutional_result
        print(f"   ✅ Evaluación constitucional completada")
        
        # 3. Reward Model (si está disponible)
        if self.reward_model is not None:
            print("\n3. 🏆 EJECUTANDO REWARD MODEL (reward.py)...")
            try:
                # Esto depende de cómo sea TU RewardModel
                reward_result = await self.reward_model.evaluate(
                    X_sample=X[:sample_size],
                    y_sample=y[:sample_size] if y is not None else None
                )
                pipeline_results['stages']['reward_evaluation'] = reward_result
                print(f"   ✅ Reward evaluation completado")
            except Exception as e:
                pipeline_results['stages']['reward_evaluation'] = {'error': str(e)}
                print(f"   ❌ Error en reward model: {e}")
        
        # 4. Determinar estado general
        monitoring_ok = 'error' not in pipeline_results['stages'].get('monitoring', {})
        constitutional_ok = pipeline_results['stages'].get('constitutional_evaluation', {}).get('overall_status') in ['compliant', 'needs_improvement']
        
        if monitoring_ok and constitutional_ok:
            pipeline_results['overall_status'] = 'healthy'
        elif not constitutional_ok:
            pipeline_results['overall_status'] = 'constitutional_issues'
        elif not monitoring_ok:
            pipeline_results['overall_status'] = 'monitoring_issues'
        else:
            pipeline_results['overall_status'] = 'unknown'
        
        print(f"\n{'='*80}")
        print("🎉 PIPELINE COMPLETADO")
        print(f"{'='*80}")
        print(f"• Model: {self.model_name}")
        print(f"• Samples: {sample_size}")
        print(f"• Overall status: {pipeline_results['overall_status']}")
        print(f"• Constitutional status: {constitutional_result.get('overall_status', 'unknown')}")
        print(f"• Modules used: {len(constitutional_result.get('modules_used', []))}")
        print(f"{'='*80}")
        
        return pipeline_results
    
    def get_system_summary(self) -> Dict[str, Any]:
        """Obtiene resumen del sistema integrado"""
        return {
            'model_name': self.model_name,
            'base_model_type': str(type(self.base_model)),
            'modules_available': {
                'monitor': True,  # Siempre disponible (es la base)
                'reward_model': REWARD_MODEL_AVAILABLE,
                'intervene': INTERVENE_MODULE_AVAILABLE,
                'cross_validation': CROSS_VALIDATION_AVAILABLE,
                'stress_testing': STRESS_TESTING_AVAILABLE,
                'metrics': METRICS_MODULE_AVAILABLE,
                'deploy': DEPLOY_MODULE_AVAILABLE,
                'integration_layer': INTEGRATION_LAYER_AVAILABLE,
                'fairness_modules': FAIRNESS_MODULES_AVAILABLE
            },
            'constitutional_history': {
                'total_evaluations': len(self.constitutional_ai.evaluation_history),
                'recent_status': self.constitutional_ai.evaluation_history[-1]['overall_status'] if self.constitutional_ai.evaluation_history else 'none'
            },
            'timestamp': datetime.now().isoformat()
        }

# ============================================================================
# 4. FUNCIÓN DE CONVENIENCIA PARA USAR EL SISTEMA INTEGRADO
# ============================================================================

async def run_integrated_system_with_existing_modules(
    base_model: Any,
    X_data: np.ndarray,
    y_data: Optional[np.ndarray] = None,
    model_name: str = "MyAIModel",
    config: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    Función de conveniencia para ejecutar tu sistema integrado.
    SOLO usa tus módulos existentes - no crea nada nuevo.
    """
    print(f"\n{'='*80}")
    print(f"EJECUTANDO SISTEMA INTEGRADO CON TUS MÓDULOS EXISTENTES")
    print(f"Modelo: {model_name}")
    print(f"Datos: X={X_data.shape}, y={y_data.shape if y_data is not None else 'None'}")
    print(f"{'='*80}")
    
    # 1. Crear sistema integrado
    integrated_system = IntegratedAISystem(
        base_model=base_model,
        model_name=model_name,
        config=config
    )
    
    # 2. Obtener resumen del sistema
    system_summary = integrated_system.get_system_summary()
    print(f"\n📋 RESUMEN DEL SISTEMA:")
    for key, value in system_summary.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for subkey, subvalue in value.items():
                print(f"    {subkey}: {subvalue}")
        else:
            print(f"  {key}: {value}")
    
    # 3. Ejecutar pipeline completo
    print(f"\n{'='*80}")
    print("INICIANDO PIPELINE COMPLETO...")
    print(f"{'='*80}")
    
    pipeline_results = await integrated_system.run_complete_pipeline(
        X=X_data,
        y=y_data,
        sample_size=min(100, len(X_data))
    )
    
    # 4. Mostrar resultados finales
    print(f"\n{'='*80}")
    print("RESULTADOS FINALES DEL PIPELINE")
    print(f"{'='*80}")
    
    for stage_name, stage_result in pipeline_results['stages'].items():
        print(f"\n{stage_name.upper()}:")
        
        if 'error' in stage_result:
            print(f"  ❌ Error: {stage_result['error']}")
        elif stage_name == 'constitutional_evaluation':
            print(f"  • Estado: {stage_result.get('overall_status', 'unknown')}")
            print(f"  • Módulos usados: {len(stage_result.get('modules_used', []))}")
            print(f"  • Checks exitosos: {stage_result.get('successful_checks', 0)}/{stage_result.get('total_checks', 0)}")
        elif stage_name == 'monitoring':
            if 'performance' in stage_result:
                perf = stage_result['performance']
                print(f"  • Accuracy: {perf.get('accuracy', 0):.3f}")
                print(f"  • Latency: {perf.get('latency', 0)*1000:.1f}ms")
    
    print(f"\n{'='*80}")
    print(f"ESTADO FINAL: {pipeline_results['overall_status'].upper()}")
    print(f"{'='*80}")
    
    return {
        'system_summary': system_summary,
        'pipeline_results': pipeline_results
    }

# ============================================================================
# 5. EJEMPLO DE USO (USA TUS MÓDULOS, NO CREA NUEVOS)
# ============================================================================

async def example_usage_existing_modules():
    """
    Ejemplo que usa EXCLUSIVAMENTE tus módulos existentes.
    Si falta algún módulo, el sistema seguirá funcionando con lo que tenga.
    """
    import numpy as np
    from sklearn.linear_model import LinearRegression
    
    print("\n" + "="*80)
    print("EJEMPLO: USANDO TUS MÓDULOS EXISTENTES")
    print("="*80)
    
    # 1. Crear datos de ejemplo
    X = np.random.randn(100, 10)
    y = np.random.randn(100)
    
    # 2. Crear modelo simple (solo para demostración)
    model = LinearRegression()
    model.fit(X, y)
    
    # 3. Ejecutar sistema integrado con TUS módulos
    results = await run_integrated_system_with_existing_modules(
        base_model=model,
        X_data=X,
        y_data=y,
        model_name="ExampleLinearModel",
        config={'compliance_threshold': 0.7}
    )
    
    return results

# ============================================================================
# 6. PUNTO DE ENTRADA PRINCIPAL
# ============================================================================

if __name__ == "__main__":
    # Ejecutar ejemplo
    import asyncio
    
    print("\n" + "="*80)
    print("INTEGRATED AI SYSTEM - USANDO TUS MÓDULOS EXISTENTES")
    print("="*80)
    
    # Verificar qué módulos están disponibles
    print("\n🔍 MÓDULOS DISPONIBLES:")
    print(f"• monitor.py: ✅")
    print(f"• reward.py: {'✅' if REWARD_MODEL_AVAILABLE else '❌'}")
    print(f"• intervene.py: {'✅' if INTERVENE_MODULE_AVAILABLE else '❌'}")
    print(f"• cross_validation.py: {'✅' if CROSS_VALIDATION_AVAILABLE else '❌'}")
    print(f"• stress.py: {'✅' if STRESS_TESTING_AVAILABLE else '❌'}")
    print(f"• metrics.py: {'✅' if METRICS_MODULE_AVAILABLE else '❌'}")
    print(f"• deploy.py: {'✅' if DEPLOY_MODULE_AVAILABLE else '❌'}")
    print(f"• integration_layer.py: {'✅' if INTEGRATION_LAYER_AVAILABLE else '❌'}")
    print(f"• fairness/explain/attention: {'✅' if FAIRNESS_MODULES_AVAILABLE else '❌'}")
    
    # Ejecutar ejemplo
    asyncio.run(example_usage_existing_modules())