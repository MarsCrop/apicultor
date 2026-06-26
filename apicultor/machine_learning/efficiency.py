import time
import psutil
import tracemalloc
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from collections import deque

def compute_snr(clean: np.ndarray, estimated: np.ndarray, eps: float = 1e-9) -> float:
    """Signal-to-Noise Ratio en dB sobre magnitudes STFT."""
    noise   = clean - estimated
    snr_val = 10 * np.log10(
        (np.sum(clean ** 2) + eps) / (np.sum(noise ** 2) + eps)
    )
    return float(snr_val)

@dataclass
class EfficiencyMetrics:
    """Métricas de eficiencia del modelo"""
    # Latencia
    time_to_first_token_ms: float = 0.0
    time_per_input_token_ms: float = 0.0
    total_latency_ms: float = 0.0
    
    # Throughput
    tokens_per_second: float = 0.0
    samples_per_second: float = 0.0
    
    # Costo
    cost_per_token: float = 0.0
    cost_per_sample: float = 0.0
    
    # Token efficiency (verbosidad)
    output_tokens_count: int = 0
    input_tokens_count: int = 0
    verbosity_ratio: float = 0.0  # output/input
    
    # Memoria
    peak_memory_mb: float = 0.0
    avg_memory_mb: float = 0.0
    
    # Holístico
    domain_scores: Dict[str, float] = None


class EfficiencyMonitor:
    """
    Monitorea eficiencia del modelo en tiempo real.
    Incluye métricas de latencia, throughput, costo y token efficiency.
    """
    
    def __init__(self, cost_per_token_usd: float = 0.001):
        self.cost_per_token_usd = cost_per_token_usd
        self.metrics_history = deque(maxlen=1000)
        self.traces = []
        
    def start_trace(self, trace_name: str):
        """Inicia seguimiento de una operación"""
        tracemalloc.start()
        self.traces.append({
            'name': trace_name,
            'start_time': time.perf_counter(),
            'start_memory': tracemalloc.get_traced_memory()[0]
        })
        
    def end_trace(self) -> Dict:
        """Termina seguimiento y retorna métricas"""
        if not self.traces:
            return {}
        
        trace = self.traces.pop()
        end_time = time.perf_counter()
        end_memory = tracemalloc.get_traced_memory()[0]
        tracemalloc.stop()
        
        return {
            'latency_ms': (end_time - trace['start_time']) * 1000,
            'memory_delta_mb': (end_memory - trace['start_memory']) / 1024 / 1024
        }
    
    async def measure_inference(
        self,
        model_fn,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        domain: str = "general"
    ) -> EfficiencyMetrics:
        """
        Mide eficiencia de inferencia completa.
        
        Args:
            model_fn: Función de predicción del modelo
            X: Input samples
            y: Targets (opcional, para calcular accuracy)
            domain: Dominio de aplicación (military, text, music, medicine, etc.)
        """
        metrics = EfficiencyMetrics(domain_scores={})
        
        # 1. Time to first token (TTFT)
        start_time = time.perf_counter()
        first_pred = await model_fn(X[0:1])  # Primer sample
        ttft_ms = (time.perf_counter() - start_time) * 1000
        metrics.time_to_first_token_ms = ttft_ms
        
        # 2. Time per input token (latencia total)
        total_start = time.perf_counter()
        all_preds = await model_fn(X)
        total_latency_ms = (time.perf_counter() - total_start) * 1000
        metrics.total_latency_ms = total_latency_ms
        
        # 3. Tokens per second
        n_samples = len(X)
        n_tokens = X.shape[1] if len(X.shape) > 1 else 1
        total_tokens = n_samples * n_tokens
        metrics.tokens_per_second = total_tokens / (total_latency_ms / 1000)
        metrics.samples_per_second = n_samples / (total_latency_ms / 1000)
        
        # 4. Time per input token
        metrics.time_per_input_token_ms = total_latency_ms / total_tokens if total_tokens > 0 else 0
        
        # 5. Token efficiency (verbosidad)
        output_tokens = all_preds.shape[1] if len(all_preds.shape) > 1 else 1
        metrics.output_tokens_count = output_tokens * n_samples
        metrics.input_tokens_count = total_tokens
        metrics.verbosity_ratio = metrics.output_tokens_count / (metrics.input_tokens_count + 1e-9)
        
        # 6. Costo
        metrics.cost_per_token = self.cost_per_token_usd
        metrics.cost_per_sample = (metrics.cost_per_token * metrics.input_tokens_count) / n_samples
        
        # 7. Memoria
        process = psutil.Process()
        try:
            mem_info = process.memory_info()
            if hasattr(mem_info, 'peak_wset'):
                metrics.peak_memory_mb = mem_info.peak_wset / 1024 / 1024
            elif hasattr(mem_info, 'rss'):
                # En Linux no hay peak_wset, usar rss como aproximación
                metrics.peak_memory_mb = mem_info.rss / 1024 / 1024
            else:
                metrics.peak_memory_mb = 0.0
        except Exception:
            metrics.peak_memory_mb = 0.0
        metrics.avg_memory_mb = process.memory_info().rss / 1024 / 1024
        
        # 8. Evaluación holística por dominio
        if y is not None:
            domain_score = self._evaluate_domain_performance(all_preds, y, domain)
            metrics.domain_scores[domain] = domain_score
        
        self.metrics_history.append(metrics)
        return metrics
    
    def _evaluate_domain_performance(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        domain: str
    ) -> float:
        """
        Evalúa rendimiento específico por dominio.
        Diferentes métricas según el dominio.
        """
        mse = np.mean((predictions - targets) ** 2)
        
        if domain == "military":
            # Precisión en detección de señales
            return 1.0 - min(mse, 1.0)
        elif domain == "text":
            # Coherencia semántica (placeholder)
            return 1.0 - min(mse, 1.0)
        elif domain == "music":
            # Preservación armónica
            return 1.0 - min(mse, 1.0)
        elif domain == "medicine":
            # Sensibilidad a anomalías
            return 1.0 - min(mse, 1.0)
        else:
            return 1.0 - min(mse, 1.0)
    
    def get_summary(self) -> Dict:
        """Retorna resumen de métricas agregadas"""
        if not self.metrics_history:
            return {}
        
        recent = list(self.metrics_history)[-100:]
        
        return {
            'latency': {
                'avg_ttft_ms': np.mean([m.time_to_first_token_ms for m in recent]),
                'avg_time_per_token_ms': np.mean([m.time_per_input_token_ms for m in recent]),
                'p95_ttft_ms': np.percentile([m.time_to_first_token_ms for m in recent], 95)
            },
            'throughput': {
                'avg_tokens_per_sec': np.mean([m.tokens_per_second for m in recent]),
                'avg_samples_per_sec': np.mean([m.samples_per_second for m in recent])
            },
            'cost': {
                'avg_cost_per_sample_usd': np.mean([m.cost_per_sample for m in recent]),
                'total_cost_usd': sum([m.cost_per_sample for m in self.metrics_history])
            },
            'token_efficiency': {
                'avg_verbosity_ratio': np.mean([m.verbosity_ratio for m in recent]),
                'avg_output_tokens': np.mean([m.output_tokens_count for m in recent])
            },
            'memory': {
                'peak_memory_mb': max([m.peak_memory_mb for m in recent]),
                'avg_memory_mb': np.mean([m.avg_memory_mb for m in recent])
            },
            'holistic_domains': {
                domain: np.mean([m.domain_scores.get(domain, 0) for m in recent])
                for domain in set([d for m in recent for d in m.domain_scores.keys()])
            }
        }


class HolisticEvaluator:
    """
    Evaluación holística de modelos a través de múltiples dominios.
    """
    
    def __init__(self, domains: List[str] = None):
        self.domains = domains or ['military', 'text', 'music', 'medicine']
        self.domain_results = {}
        
    async def evaluate(
        self,
        model,
        domain_datasets: Dict[str, Tuple[np.ndarray, np.ndarray]]
    ) -> Dict:
        """
        Evalúa modelo en múltiples dominios.
        
        Args:
            model: Modelo a evaluar
            domain_datasets: Dict con {dominio: (X, y)}
        """
        results = {}
        
        for domain, (X, y) in domain_datasets.items():
            if domain not in self.domains:
                continue
                
            # Predicciones
            start_time = time.time()
            predictions = await model.predict(X)
            inference_time = time.time() - start_time
            
            # Métricas específicas por dominio
            metrics = self._domain_specific_metrics(domain, predictions, y)
            metrics['inference_time_ms'] = inference_time * 1000
            metrics['samples_processed'] = len(X)
            
            results[domain] = metrics
            
        self.domain_results = results
        return results
    
    def _domain_specific_metrics(
        self,
        domain: str,
        predictions: np.ndarray,
        targets: np.ndarray
    ) -> Dict:
        """Métricas específicas por dominio"""
        
        mse = np.mean((predictions - targets) ** 2)
        snr = compute_snr(targets, predictions)
        
        base_metrics = {
            'mse': float(mse),
            'snr_db': float(snr),
            'correlation': float(np.corrcoef(predictions.flatten(), targets.flatten())[0, 1])
        }
        
        if domain == "military":
            # Detección de señales, falsos positivos
            threshold = np.percentile(targets, 90)
            tp = np.sum((predictions > threshold) & (targets > threshold))
            fp = np.sum((predictions > threshold) & (targets <= threshold))
            base_metrics['precision'] = tp / (tp + fp + 1e-9)
            base_metrics['recall'] = tp / (np.sum(targets > threshold) + 1e-9)
            
        elif domain == "text":
            # Coherencia, entropía
            pred_entropy = -np.sum(predictions * np.log(predictions + 1e-9))
            base_metrics['entropy'] = float(pred_entropy)
            
        elif domain == "music":
            # Conservación espectral
            spectral_diff = np.mean(np.abs(np.fft.rfft(predictions) - np.fft.rfft(targets)))
            base_metrics['spectral_difference'] = float(spectral_diff)
            
        elif domain == "medicine":
            # Sensibilidad a anomalías
            anomaly_mask = targets > np.percentile(targets, 95)
            if np.any(anomaly_mask):
                anomaly_mse = np.mean((predictions[anomaly_mask] - targets[anomaly_mask]) ** 2)
                base_metrics['anomaly_mse'] = float(anomaly_mse)
        
        return base_metrics
    
    def get_holistic_score(self) -> float:
        """Calcula puntaje holístico promedio sobre todos los dominios"""
        if not self.domain_results:
            return 0.0
        
        scores = []
        for domain, metrics in self.domain_results.items():
            # Normalizar SNR a score 0-1 (asumiendo SNR máximo 30dB)
            snr_score = min(1.0, metrics.get('snr_db', 0) / 30.0)
            mse_score = 1.0 - min(1.0, metrics.get('mse', 1.0))
            scores.append((snr_score + mse_score) / 2)
        
        return float(np.mean(scores)) if scores else 0.0