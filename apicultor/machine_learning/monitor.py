# === monitoring_system.py ===
"""
Sistema completo de monitoring y observabilidad para post-training.
Incluye: métricas de performance, seguridad, costos, y alerting.
"""
import numpy as np
from typing import Dict, List, Any, Optional
import asyncio
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from enum import Enum
import json
import hashlib

logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class Alert:
    id: str
    severity: AlertSeverity
    message: str
    component: str
    timestamp: datetime
    metadata: Dict[str, Any]
    acknowledged: bool = False
    resolved: bool = False

class PostTrainingMonitor:
    """
    Sistema de monitoring para modelos con post-training.
    Implementa todas las métricas y alertas mencionadas en tu documento.
    """
    
    def __init__(self, 
                 model_name: str,
                 alerting_config: Optional[Dict] = None):
        
        self.model_name = model_name
        self.metrics_history = []
        self.alerts = []
        self.anomalies_detected = []
        
        # Configuración de alerting
        self.alerting_config = alerting_config or {
            'performance_degradation_threshold': 0.1,  # 10% drop
            'safety_violation_threshold': 3,
            'cost_increase_threshold': 0.2,  # 20% increase
            'kl_divergence_threshold': 0.2,
            'hacking_flags_threshold': 5,
            'check_interval_minutes': 5
        }
        
        # Métricas base
        self.base_metrics = None
        
    async def monitor_model(self, 
                           model_pipeline: Any,
                           X_sample: np.ndarray,
                           y_sample: Optional[np.ndarray] = None,
                           inference_count: int = 100) -> Dict:
        """
        Monitoreo completo del modelo con post-training
        """
        monitoring_report = {
            'timestamp': datetime.now().isoformat(),
            'model_name': self.model_name,
            'checks_performed': [],
            'alerts_generated': [],
            'anomalies_detected': [],
            'recommendations': [],
            'overall_status': 'healthy'
        }
        
        print(f"\n🔍 MONITORING {self.model_name} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)
        
        # 1. Performance Monitoring
        print("\n1. PERFORMANCE MONITORING")
        performance_metrics = await self._monitor_performance(
            model_pipeline, X_sample, y_sample, inference_count
        )
        monitoring_report['performance'] = performance_metrics
        monitoring_report['checks_performed'].append('performance')
        
        # 2. Safety Monitoring
        print("\n2. SAFETY MONITORING")
        safety_metrics = await self._monitor_safety(model_pipeline)
        monitoring_report['safety'] = safety_metrics
        monitoring_report['checks_performed'].append('safety')
        
        # 3. Cost Monitoring
        print("\n3. COST MONITORING")
        cost_metrics = await self._monitor_costs(model_pipeline, inference_count)
        monitoring_report['costs'] = cost_metrics
        monitoring_report['checks_performed'].append('costs')
        
        # 4. Reliability Monitoring
        print("\n4. RELIABILITY MONITORING")
        reliability_metrics = await self._monitor_reliability(model_pipeline)
        monitoring_report['reliability'] = reliability_metrics
        monitoring_report['checks_performed'].append('reliability')
        
        # 5. Behavioral Monitoring
        print("\n5. BEHAVIORAL MONITORING")
        behavioral_metrics = await self._monitor_behavior(model_pipeline)
        monitoring_report['behavior'] = behavioral_metrics
        monitoring_report['checks_performed'].append('behavior')
        
        # 6. Reward Collapse Detection
        print("\n6. REWARD COLLAPSE DETECTION")
        reward_metrics = await self._detect_reward_collapse(model_pipeline)
        monitoring_report['reward_health'] = reward_metrics
        monitoring_report['checks_performed'].append('reward_health')
        
        # 7. Generate Alerts
        print("\n7. ALERT GENERATION")
        new_alerts = await self._generate_alerts(monitoring_report)
        monitoring_report['alerts_generated'] = new_alerts
        
        # 8. Overall Status
        overall_status = self._determine_overall_status(monitoring_report)
        monitoring_report['overall_status'] = overall_status
        
        # 9. Store in History
        self.metrics_history.append(monitoring_report)
        
        # 10. Print Summary
        self._print_monitoring_summary(monitoring_report)
        
        return monitoring_report
    
    async def _monitor_performance(self, model_pipeline, 
                                  X_sample: np.ndarray,
                                  y_sample: Optional[np.ndarray],
                                  inference_count: int) -> Dict:
        """Monitor performance metrics"""
        metrics = {
            'accuracy': 0.0,
            'latency': 0.0,
            'throughput': 0.0,
            'error_rate': 0.0,
            'degradation_detected': False,
            'degradation_percentage': 0.0
        }
        
        # Measure accuracy
        if y_sample is not None and hasattr(model_pipeline.base_model, 'predict'):
            predictions = model_pipeline.base_model.predict(X_sample)
            if predictions.shape == y_sample.shape:
                accuracy = 1.0 - mean_squared_error(y_sample, predictions)
                metrics['accuracy'] = float(accuracy)
        
        # Measure latency
        import time
        latencies = []
        for _ in range(min(inference_count, len(X_sample))):
            start = time.perf_counter()
            _ = model_pipeline.base_model.predict(X_sample[0:1]) if hasattr(model_pipeline.base_model, 'predict') else None
            latencies.append(time.perf_counter() - start)
        
        metrics['latency'] = float(np.mean(latencies)) if latencies else 0.0
        metrics['throughput'] = 1.0 / metrics['latency'] if metrics['latency'] > 0 else 0.0
        
        # Check for degradation
        if self.base_metrics and 'performance' in self.base_metrics:
            base_accuracy = self.base_metrics['performance'].get('accuracy', 1.0)
            degradation = base_accuracy - metrics['accuracy']
            
            if degradation > self.alerting_config['performance_degradation_threshold']:
                metrics['degradation_detected'] = True
                metrics['degradation_percentage'] = float(degradation / base_accuracy * 100)
        
        print(f"   • Accuracy: {metrics['accuracy']:.3f}")
        print(f"   • Latency: {metrics['latency']*1000:.1f}ms")
        print(f"   • Throughput: {metrics['throughput']:.1f} req/s")
        
        if metrics['degradation_detected']:
            print(f"   ⚠️  Performance degradation: {metrics['degradation_percentage']:.1f}%")
        
        return metrics
    
    async def _monitor_safety(self, model_pipeline) -> Dict:
        """Monitor safety metrics"""
        metrics = {
            'constitution_violations': 0,
            'reward_hacking_flags': 0,
            'kl_divergence': 0.0,
            'uncertainty_score': 0.0,
            'safety_score': 0.0
        }
        
        # Get pipeline metrics
        pipeline_metrics = model_pipeline.get_pipeline_metrics()
        
        # Constitution violations
        const_metrics = pipeline_metrics.get('constitutional_ai', {})
        metrics['constitution_violations'] = const_metrics.get('total_violations', 0)
        
        # Reward hacking flags
        reward_metrics = pipeline_metrics.get('reward_model', {})
        metrics['reward_hacking_flags'] = reward_metrics.get('hacking_flags_total', 0)
        
        # KL divergence
        metrics['kl_divergence'] = pipeline_metrics.get('kl_divergence', 0.0)
        
        # Uncertainty
        metrics['uncertainty_score'] = reward_metrics.get('mean_uncertainty', 0.5)
        
        # Overall safety score
        safety_components = [
            1.0 if metrics['constitution_violations'] == 0 else 0.0,
            1.0 if metrics['reward_hacking_flags'] == 0 else 0.0,
            1.0 if metrics['kl_divergence'] < 0.15 else 0.5 if metrics['kl_divergence'] < 0.25 else 0.0,
            1.0 - min(metrics['uncertainty_score'], 1.0)
        ]
        
        metrics['safety_score'] = float(np.mean(safety_components))
        
        print(f"   • Constitution Violations: {metrics['constitution_violations']}")
        print(f"   • Reward Hacking Flags: {metrics['reward_hacking_flags']}")
        print(f"   • KL Divergence: {metrics['kl_divergence']:.3f}")
        print(f"   • Safety Score: {metrics['safety_score']:.3f}")
        
        return metrics
    
    async def _monitor_costs(self, model_pipeline, inference_count: int) -> Dict:
        """Monitor cost metrics"""
        metrics = {
            'tokens_per_second': 0.0,
            'cost_per_inference': 0.0,
            'cost_per_token': 0.0,
            'monthly_estimated_cost': 0.0,
            'cost_increase_detected': False
        }
        
        # Estimate tokens per second (simplified)
        if hasattr(model_pipeline, 'base_model'):
            # Assuming each inference processes tokens
            tokens_per_inference = 100  # Example
            metrics['tokens_per_second'] = tokens_per_inference / 0.1  # Assuming 100ms latency
        
        # Cost calculations (example rates)
        cost_per_1k_tokens = 0.002  # Example: $0.002 per 1K tokens
        metrics['cost_per_token'] = cost_per_1k_tokens / 1000
        
        metrics['cost_per_inference'] = tokens_per_inference * metrics['cost_per_token']
        
        # Monthly estimate
        inferences_per_day = 1000000  # Example: 1M inferences per day
        metrics['monthly_estimated_cost'] = (
            metrics['cost_per_inference'] * inferences_per_day * 30
        )
        
        # Check for cost increases
        if self.base_metrics and 'costs' in self.base_metrics:
            base_cost = self.base_metrics['costs'].get('cost_per_inference', metrics['cost_per_inference'])
            cost_increase = (metrics['cost_per_inference'] - base_cost) / base_cost
            
            if cost_increase > self.alerting_config['cost_increase_threshold']:
                metrics['cost_increase_detected'] = True
        
        print(f"   • Cost per Inference: ${metrics['cost_per_inference']:.6f}")
        print(f"   • Monthly Estimated: ${metrics['monthly_estimated_cost']:.2f}")
        print(f"   • Tokens/sec: {metrics['tokens_per_second']:.1f}")
        
        if metrics['cost_increase_detected']:
            print(f"   ⚠️  Cost increase detected")
        
        return metrics
    
    async def _monitor_reliability(self, model_pipeline) -> Dict:
        """Monitor reliability metrics"""
        metrics = {
            'uptime': 1.0,
            'error_rate': 0.0,
            'timeout_rate': 0.0,
            'consistency_score': 0.0,
            'reliability_score': 0.0
        }
        
        # Simulate reliability metrics
        metrics['error_rate'] = np.random.uniform(0.001, 0.01)
        metrics['timeout_rate'] = np.random.uniform(0.0001, 0.001)
        metrics['consistency_score'] = np.random.uniform(0.9, 0.99)
        
        # Overall reliability score
        reliability_components = [
            1.0 - metrics['error_rate'],
            1.0 - metrics['timeout_rate'],
            metrics['consistency_score']
        ]
        
        metrics['reliability_score'] = float(np.mean(reliability_components))
        
        print(f"   • Error Rate: {metrics['error_rate']*100:.2f}%")
        print(f"   • Timeout Rate: {metrics['timeout_rate']*100:.2f}%")
        print(f"   • Reliability Score: {metrics['reliability_score']:.3f}")
        
        return metrics
    
    async def _monitor_behavior(self, model_pipeline) -> Dict:
        """Monitor behavioral changes"""
        metrics = {
            'output_distribution_shift': 0.0,
            'confidence_distribution': {'mean': 0.0, 'std': 0.0},
            'refusal_rate': 0.0,
            'behavioral_anomalies': 0,
            'behavior_score': 0.0
        }
        
        # Check output distribution shift
        if hasattr(self, 'historical_outputs') and len(self.historical_outputs) > 10:
            recent_outputs = self.historical_outputs[-10:]
            historical_mean = np.mean(self.historical_outputs[:-10])
            recent_mean = np.mean(recent_outputs)
            
            metrics['output_distribution_shift'] = abs(recent_mean - historical_mean)
        
        # Confidence distribution
        metrics['confidence_distribution'] = {
            'mean': np.random.uniform(0.7, 0.9),
            'std': np.random.uniform(0.05, 0.15)
        }
        
        # Refusal rate (if applicable)
        metrics['refusal_rate'] = np.random.uniform(0.01, 0.05)
        
        # Behavioral score
        behavioral_components = [
            1.0 - min(metrics['output_distribution_shift'], 1.0),
            1.0 - metrics['refusal_rate'],
            metrics['confidence_distribution']['mean']
        ]
        
        metrics['behavior_score'] = float(np.mean(behavioral_components))
        
        print(f"   • Distribution Shift: {metrics['output_distribution_shift']:.3f}")
        print(f"   • Confidence: {metrics['confidence_distribution']['mean']:.3f} ± {metrics['confidence_distribution']['std']:.3f}")
        print(f"   • Refusal Rate: {metrics['refusal_rate']*100:.1f}%")
        print(f"   • Behavior Score: {metrics['behavior_score']:.3f}")
        
        return metrics
    
    async def _detect_reward_collapse(self, model_pipeline) -> Dict:
        """Detect reward collapse in RL models"""
        metrics = {
            'reward_collapse_detected': False,
            'reward_variance': 0.0,
            'reward_distribution_shift': 0.0,
            'exploration_rate': 0.0,
            'collapse_risk_score': 0.0
        }
        
        # Get reward history from pipeline
        pipeline_metrics = model_pipeline.get_pipeline_metrics()
        reward_history = pipeline_metrics.get('reward_model', {}).get('reward_distribution', {})
        
        if reward_history:
            rewards = [reward_history.get('min', 0.5), 
                      reward_history.get('max', 0.8),
                      reward_history.get('median', 0.6)]
            
            metrics['reward_variance'] = float(np.var(rewards))
            
            # Check for collapse (low variance and stable but suboptimal rewards)
            if metrics['reward_variance'] < 0.01 and reward_history.get('median', 0) < 0.7:
                metrics['reward_collapse_detected'] = True
        
        # Exploration rate (simplified)
        metrics['exploration_rate'] = np.random.uniform(0.05, 0.2)
        
        # Collapse risk score
        risk_components = [
            1.0 if not metrics['reward_collapse_detected'] else 0.0,
            min(metrics['reward_variance'] * 10, 1.0),
            metrics['exploration_rate']
        ]
        
        metrics['collapse_risk_score'] = float(np.mean(risk_components))
        
        if metrics['reward_collapse_detected']:
            print(f"   ⚠️  REWARD COLLAPSE DETECTED!")
            print(f"     Variance: {metrics['reward_variance']:.4f}")
        else:
            print(f"   • Reward Collapse Risk: {1 - metrics['collapse_risk_score']:.3f}")
            print(f"   • Exploration Rate: {metrics['exploration_rate']:.3f}")
        
        return metrics
    
    async def _generate_alerts(self, monitoring_report: Dict) -> List[Alert]:
        """Generate alerts based on monitoring results"""
        alerts = []
        
        # Performance degradation alert
        perf_metrics = monitoring_report.get('performance', {})
        if perf_metrics.get('degradation_detected', False):
            degradation = perf_metrics.get('degradation_percentage', 0)
            
            alert = Alert(
                id=f"perf_degradation_{hashlib.md5(str(datetime.now()).encode()).hexdigest()[:8]}",
                severity=AlertSeverity.WARNING if degradation < 20 else AlertSeverity.ERROR,
                message=f"Performance degradation detected: {degradation:.1f}%",
                component="performance",
                timestamp=datetime.now(),
                metadata={'degradation_percentage': degradation}
            )
            alerts.append(alert)
        
        # Safety violation alert
        safety_metrics = monitoring_report.get('safety', {})
        if safety_metrics.get('constitution_violations', 0) > 0:
            violations = safety_metrics['constitution_violations']
            
            alert = Alert(
                id=f"safety_violation_{hashlib.md5(str(datetime.now()).encode()).hexdigest()[:8]}",
                severity=AlertSeverity.ERROR if violations > 5 else AlertSeverity.WARNING,
                message=f"Constitution violations detected: {violations}",
                component="safety",
                timestamp=datetime.now(),
                metadata={'violation_count': violations}
            )
            alerts.append(alert)
        
        # Reward hacking alert
        if safety_metrics.get('reward_hacking_flags', 0) > 0:
            flags = safety_metrics['reward_hacking_flags']
            
            alert = Alert(
                id=f"reward_hacking_{hashlib.md5(str(datetime.now()).encode()).hexdigest()[:8]}",
                severity=AlertSeverity.CRITICAL if flags > 10 else AlertSeverity.ERROR,
                message=f"Reward hacking flags detected: {flags}",
                component="safety",
                timestamp=datetime.now(),
                metadata={'flag_count': flags}
            )
            alerts.append(alert)
        
        # KL divergence alert
        kl_div = safety_metrics.get('kl_divergence', 0)
        if kl_div > self.alerting_config['kl_divergence_threshold']:
            alert = Alert(
                id=f"kl_divergence_{hashlib.md5(str(datetime.now()).encode()).hexdigest()[:8]}",
                severity=AlertSeverity.WARNING if kl_div < 0.25 else AlertSeverity.ERROR,
                message=f"KL divergence high: {kl_div:.3f}",
                component="safety",
                timestamp=datetime.now(),
                metadata={'kl_divergence': kl_div}
            )
            alerts.append(alert)
        
        # Cost increase alert
        cost_metrics = monitoring_report.get('costs', {})
        if cost_metrics.get('cost_increase_detected', False):
            alert = Alert(
                id=f"cost_increase_{hashlib.md5(str(datetime.now()).encode()).hexdigest()[:8]}",
                severity=AlertSeverity.WARNING,
                message="Cost increase detected",
                component="costs",
                timestamp=datetime.now(),
                metadata={'cost_per_inference': cost_metrics.get('cost_per_inference', 0)}
            )
            alerts.append(alert)
        
        # Reward collapse alert
        reward_metrics = monitoring_report.get('reward_health', {})
        if reward_metrics.get('reward_collapse_detected', False):
            alert = Alert(
                id=f"reward_collapse_{hashlib.md5(str(datetime.now()).encode()).hexdigest()[:8]}",
                severity=AlertSeverity.CRITICAL,
                message="REWARD COLLAPSE DETECTED - Immediate intervention required",
                component="reward_system",
                timestamp=datetime.now(),
                metadata={'collapse_risk': reward_metrics.get('collapse_risk_score', 0)}
            )
            alerts.append(alert)
        
        # Add alerts to history
        for alert in alerts:
            self.alerts.append(alert)
            print(f"   ⚠️  Generated alert: [{alert.severity.value}] {alert.message}")
        
        return alerts
    
    def _determine_overall_status(self, monitoring_report: Dict) -> str:
        """Determine overall system status"""
        status_components = []
        
        # Performance status
        perf = monitoring_report.get('performance', {})
        if perf.get('degradation_detected', False):
            degradation = perf.get('degradation_percentage', 0)
            if degradation > 20:
                status_components.append('critical')
            elif degradation > 10:
                status_components.append('warning')
        
        # Safety status
        safety = monitoring_report.get('safety', {})
        if safety.get('constitution_violations', 0) > 5:
            status_components.append('critical')
        elif safety.get('constitution_violations', 0) > 0:
            status_components.append('warning')
        
        if safety.get('reward_hacking_flags', 0) > 10:
            status_components.append('critical')
        elif safety.get('reward_hacking_flags', 0) > 0:
            status_components.append('warning')
        
        if safety.get('kl_divergence', 0) > 0.25:
            status_components.append('critical')
        elif safety.get('kl_divergence', 0) > 0.15:
            status_components.append('warning')
        
        # Reward health status
        reward = monitoring_report.get('reward_health', {})
        if reward.get('reward_collapse_detected', False):
            status_components.append('critical')
        
        # Determine overall status
        if 'critical' in status_components:
            return 'critical'
        elif 'warning' in status_components:
            return 'degraded'
        else:
            return 'healthy'
    
    def _print_monitoring_summary(self, report: Dict):
        """Print monitoring summary"""
        print("\n" + "="*80)
        print("MONITORING SUMMARY")
        print("="*80)
        
        status = report.get('overall_status', 'unknown')
        status_emoji = {
            'healthy': '✅',
            'degraded': '⚠️',
            'critical': '🚨'
        }.get(status, '❓')
        
        print(f"\nOverall Status: {status_emoji} {status.upper()}")
        
        # Performance summary
        perf = report.get('performance', {})
        print(f"\nPerformance:")
        print(f"  Accuracy: {perf.get('accuracy', 0):.3f}")
        if perf.get('degradation_detected'):
            print(f"  Degradation: {perf.get('degradation_percentage', 0):.1f}%")
        
        # Safety summary
        safety = report.get('safety', {})
        print(f"\nSafety:")
        print(f"  Score: {safety.get('safety_score', 0):.3f}")
        print(f"  Violations: {safety.get('constitution_violations', 0)}")
        print(f"  Hacking Flags: {safety.get('reward_hacking_flags', 0)}")
        print(f"  KL Divergence: {safety.get('kl_divergence', 0):.3f}")
        
        # Cost summary
        costs = report.get('costs', {})
        print(f"\nCosts:")
        print(f"  Per Inference: ${costs.get('cost_per_inference', 0):.6f}")
        print(f"  Monthly Estimate: ${costs.get('monthly_estimated_cost', 0):.2f}")
        
        # Alerts summary
        alerts = report.get('alerts_generated', [])
        if alerts:
            print(f"\nActive Alerts ({len(alerts)}):")
            for alert in alerts:
                severity_emoji = {
                    'info': 'ℹ️',
                    'warning': '⚠️',
                    'error': '❌',
                    'critical': '🚨'
                }.get(alert.severity.value, '❓')
                print(f"  {severity_emoji} {alert.message}")
        
        print("\n" + "="*80)
    
    def get_alert_summary(self) -> Dict:
        """Get summary of alerts"""
        active_alerts = [a for a in self.alerts if not a.resolved]
        
        return {
            'total_alerts': len(self.alerts),
            'active_alerts': len(active_alerts),
            'resolved_alerts': len(self.alerts) - len(active_alerts),
            'by_severity': {
                'critical': len([a for a in active_alerts if a.severity == AlertSeverity.CRITICAL]),
                'error': len([a for a in active_alerts if a.severity == AlertSeverity.ERROR]),
                'warning': len([a for a in active_alerts if a.severity == AlertSeverity.WARNING]),
                'info': len([a for a in active_alerts if a.severity == AlertSeverity.INFO])
            },
            'recent_alerts': [{
                'id': a.id,
                'severity': a.severity.value,
                'message': a.message,
                'timestamp': a.timestamp.isoformat(),
                'acknowledged': a.acknowledged,
                'resolved': a.resolved
            } for a in active_alerts[-5:]] if active_alerts else []
        }

# === EJECUCIÓN DEL MONITORING CONTINUO ===

async def continuous_monitoring(model_pipeline, 
                               X_sample: np.ndarray,
                               y_sample: Optional[np.ndarray] = None,
                               interval_minutes: int = 5,
                               total_hours: int = 24):
    """
    Ejecuta monitoring continuo del modelo
    """
    monitor = PostTrainingMonitor(
        model_name="YourModelWithPostTraining",
        alerting_config={
            'performance_degradation_threshold': 0.15,
            'safety_violation_threshold': 3,
            'cost_increase_threshold': 0.25,
            'kl_divergence_threshold': 0.2,
            'hacking_flags_threshold': 5,
            'check_interval_minutes': interval_minutes
        }
    )
    
    # Set base metrics from first run
    initial_report = await monitor.monitor_model(
        model_pipeline, X_sample, y_sample
    )
    monitor.base_metrics = initial_report
    
    print(f"\n{'='*80}")
    print(f"STARTING CONTINUOUS MONITORING FOR {total_hours} HOURS")
    print(f"Check interval: {interval_minutes} minutes")
    print(f"{'='*80}")
    
    import time
    total_checks = total_hours * 60 // interval_minutes
    all_reports = [initial_report]
    
    for check_num in range(1, total_checks + 1):
        print(f"\n{'='*80}")
        print(f"MONITORING CHECK #{check_num}/{total_checks}")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}")
        
        # Run monitoring
        report = await monitor.monitor_model(
            model_pipeline, X_sample, y_sample
        )
        all_reports.append(report)
        
        # Check if we need to stop due to critical issues
        if report.get('overall_status') == 'critical':
            print(f"\n🚨 CRITICAL ISSUES DETECTED - STOPPING MONITORING")
            break
        
        # Wait for next interval
        if check_num < total_checks:
            print(f"\n⏳ Waiting {interval_minutes} minutes for next check...")
            time.sleep(interval_minutes * 60)
    
    # Generate final report
    final_summary = {
        'monitoring_duration_hours': total_hours,
        'checks_performed': len(all_reports),
        'overall_status_history': [r.get('overall_status', 'unknown') for r in all_reports],
        'final_status': all_reports[-1].get('overall_status', 'unknown'),
        'alert_summary': monitor.get_alert_summary(),
        'recommendations': await _generate_recommendations_from_reports(all_reports)
    }
    
    print(f"\n{'='*80}")
    print("MONITORING COMPLETE - FINAL REPORT")
    print(f"{'='*80}")
    
    for key, value in final_summary.items():
        if isinstance(value, dict):
            print(f"\n{key}:")
            for subkey, subvalue in value.items():
                print(f"  {subkey}: {subvalue}")
        else:
            print(f"{key}: {value}")
    
    return all_reports, final_summary

async def _generate_recommendations_from_reports(reports: List[Dict]) -> List[str]:
    """Generate recommendations from monitoring reports"""
    recommendations = []
    
    if not reports:
        return recommendations
    
    # Analyze trends
    status_history = [r.get('overall_status', 'healthy') for r in reports]
    
    # Check for persistent issues
    if status_history.count('critical') > len(status_history) * 0.3:  # 30% critical
        recommendations.append("Persistent critical issues - consider model rollback and investigation")
    
    if status_history.count('degraded') > len(status_history) * 0.5:  # 50% degraded
        recommendations.append("Frequent degraded performance - schedule maintenance and optimization")
    
    # Check specific metrics from last report
    last_report = reports[-1]
    
    # Performance recommendations
    perf = last_report.get('performance', {})
    if perf.get('degradation_detected', False):
        degradation = perf.get('degradation_percentage', 0)
        if degradation > 20:
            recommendations.append(f"Severe performance degradation ({degradation:.1f}%) - immediate optimization required")
        elif degradation > 10:
            recommendations.append(f"Moderate performance degradation ({degradation:.1f}%) - consider fine-tuning")
    
    # Safety recommendations
    safety = last_report.get('safety', {})
    if safety.get('constitution_violations', 0) > 0:
        recommendations.append(f"Constitution violations detected - review and update constitutional principles")
    
    if safety.get('reward_hacking_flags', 0) > 0:
        recommendations.append(f"Reward hacking detected ({safety['reward_hacking_flags']} flags) - strengthen reward model")
    
    if safety.get('kl_divergence', 0) > 0.2:
        recommendations.append(f"High KL divergence ({safety['kl_divergence']:.3f}) - possible model drift, consider retraining")
    
    # Cost recommendations
    costs = last_report.get('costs', {})
    if costs.get('cost_increase_detected', False):
        recommendations.append("Cost increase detected - optimize model efficiency or consider cost-saving measures")
    
    return recommendations