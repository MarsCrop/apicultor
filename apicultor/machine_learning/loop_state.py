import os
import json
import glob
from typing import Dict, List, Optional, Tuple
from datetime import datetime

class LoopStateManager:
    """
    Gestiona el estado de los loops para poder reanudar ejecuciones interrumpidas.
    Permite saltar loops ya completados y continuar desde el último loop exitoso.
    """
    
    def __init__(self, output_dir: str, state_file: str = "pipeline_state.json"):
        self.output_dir = output_dir
        self.state_file = os.path.join(output_dir, state_file)
        self.state = self._load_state()
    
    def _load_state(self) -> Dict:
        """Carga el estado guardado o crea uno nuevo"""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                print(f"✅ Estado cargado desde {self.state_file}")
                return state
            except Exception as e:
                print(f"⚠️ Error cargando estado: {e}")
                return self._create_new_state()
        else:
            return self._create_new_state()
    
    def _create_new_state(self) -> Dict:
        """Crea un nuevo estado"""
        return {
            'created_at': datetime.now().isoformat(),
            'last_loop_completed': 0,
            'completed_loops': [],
            'failed_loops': [],
            'in_progress_loops': [],
            'total_expected_loops': None,
            'max_loop_counter': 0,
            'status': 'initialized'
        }
    
    def save(self):
        """Guarda el estado actual"""
        self.state['updated_at'] = datetime.now().isoformat()
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)
        print(f"💾 Estado guardado: loop {self.state['last_loop_completed']}")
    
    def get_next_loop(self, total_loops: Optional[int] = None) -> Optional[int]:
        """
        Determina el próximo loop a ejecutar.
        
        Args:
            total_loops: Número total de loops esperados (opcional)
        
        Returns:
            Número del próximo loop, o None si todos están completados
        """
        if total_loops:
            self.state['total_expected_loops'] = total_loops
        
        # Verificar si todos los loops están completados
        if self.state['total_expected_loops']:
            completed = len(self.state['completed_loops'])
            if completed >= self.state['total_expected_loops']:
                print(f"✅ Todos los loops completados ({completed}/{self.state['total_expected_loops']})")
                return None
        
        # Buscar el siguiente loop no completado
        next_loop = self.state['last_loop_completed'] + 1
        
        # Verificar si ya fue completado (por si acaso)
        while next_loop in self.state['completed_loops']:
            next_loop += 1
        
        # Verificar si excede el total esperado
        if self.state['total_expected_loops'] and next_loop > self.state['total_expected_loops']:
            print(f"✅ Límite alcanzado: {next_loop} > {self.state['total_expected_loops']}")
            return None
        
        return next_loop
    
    def mark_loop_started(self, loop_counter: int):
        """Marca un loop como iniciado"""
        if loop_counter not in self.state['in_progress_loops']:
            self.state['in_progress_loops'].append(loop_counter)
        self.state['status'] = f'running_loop_{loop_counter}'
        self.save()
    
    def mark_loop_completed(self, loop_counter: int, results: Dict = None):
        """Marca un loop como completado exitosamente"""
        if loop_counter not in self.state['completed_loops']:
            self.state['completed_loops'].append(loop_counter)
        if loop_counter in self.state['in_progress_loops']:
            self.state['in_progress_loops'].remove(loop_counter)
        self.state['last_loop_completed'] = max(self.state['last_loop_completed'], loop_counter)
        self.state['status'] = 'running'
        if results:
            self.state[f'loop_{loop_counter}_summary'] = {
                'completed_at': datetime.now().isoformat(),
                'snr_db': results.get('aoc_final', {}).get('snr_db'),
                'mse': results.get('aoc_final', {}).get('mse')
            }
        self.save()
    
    def mark_loop_failed(self, loop_counter: int, error: str = None):
        """Marca un loop como fallido"""
        if loop_counter not in self.state['failed_loops']:
            self.state['failed_loops'].append({
                'loop': loop_counter,
                'error': error,
                'timestamp': datetime.now().isoformat()
            })
        if loop_counter in self.state['in_progress_loops']:
            self.state['in_progress_loops'].remove(loop_counter)
        self.state['status'] = 'has_failures'
        self.save()
    
    def is_loop_completed(self, loop_counter: int) -> bool:
        """Verifica si un loop específico ya fue completado"""
        return loop_counter in self.state['completed_loops']
    
    def is_loop_failed(self, loop_counter: int) -> bool:
        """Verifica si un loop específico falló"""
        return any(f['loop'] == loop_counter for f in self.state['failed_loops'])
    
    def get_completed_loops(self) -> List[int]:
        """Retorna lista de loops completados"""
        return sorted(self.state['completed_loops'])
    
    def get_failed_loops(self) -> List[Dict]:
        """Retorna lista de loops fallidos"""
        return self.state['failed_loops']
    
    def get_summary(self) -> str:
        """Retorna un resumen del estado"""
        summary = f"""
        📊 RESUMEN DE EJECUCIÓN
        {'='*40}
        Estado: {self.state['status']}
        Loops completados: {len(self.state['completed_loops'])}
        Loops fallidos: {len(self.state['failed_loops'])}
        Último loop completado: {self.state['last_loop_completed']}
        Loops en progreso: {self.state['in_progress_loops']}
        """
        if self.state['total_expected_loops']:
            summary += f"\nTotal esperado: {self.state['total_expected_loops']}"
            progress = len(self.state['completed_loops']) / self.state['total_expected_loops'] * 100
            summary += f"\nProgreso: {progress:.1f}%"
        return summary


def find_existing_checkpoints(output_dir: str, loop_counter: int) -> Dict[str, bool]:
    """
    Encuentra qué archivos de checkpoint existen para un loop específico.
    
    Returns:
        Diccionario con flags indicando qué archivos ya existen
    """
    checkpoints = {
        'cv_results': os.path.join(output_dir, f"cv_results_loop_{loop_counter}.pkl"),
        'model': os.path.join(output_dir, f"aoc_dsvm_loop_{loop_counter}.pkl"),
        'memory': os.path.join(output_dir, f"memory_report_loop_{loop_counter}.json"),
        'adversary_collection': os.path.join(output_dir, f"adversary_models_collection_loop_{loop_counter}.json"),
        'quote_report': os.path.join(output_dir, f"quote_reference_report_loop_{loop_counter}.txt"),
        'evals': os.path.join(output_dir, f"evals_loop_{loop_counter}.txt"),
        'rl_summary': os.path.join(output_dir, "rl_summary.txt"),  # Sobrescribe, no por loop
    }
    
    exists = {}
    for name, path in checkpoints.items():
        exists[name] = os.path.exists(path)
    
    return exists


def should_skip_loop(output_dir: str, loop_counter: int, force_rerun: bool = False) -> Tuple[bool, str]:
    """
    Determina si un loop debe ser saltado porque ya fue completado.
    
    Args:
        output_dir: Directorio de salida
        loop_counter: Número del loop a verificar
        force_rerun: Si True, fuerza re-ejecución aunque existan checkpoints
    
    Returns:
        (skip, reason): Tupla con booleano y razón
    """
    if force_rerun:
        return False, "force_rerun=True"
    
    checkpoints = find_existing_checkpoints(output_dir, loop_counter)
    
    # Criterio: loop completado si existe cv_results y modelo
    if checkpoints['cv_results'] and checkpoints['model']:
        # Verificar que los archivos no estén vacíos/corruptos
        try:
            if checkpoints['cv_results']:
                import pickle
                with open(checkpoints['cv_results'], 'rb') as f:
                    cv_data = pickle.load(f)
                    if cv_data and isinstance(cv_data, dict):
                        # Verificar que tenga los datos mínimos esperados
                        if 'exp_data' in cv_data or 'context' in cv_data:
                            return True, f"Loop {loop_counter} ya completado (cv_results y modelo existen)"
        except Exception as e:
            print(f"⚠️ Archivo corrupto para loop {loop_counter}: {e}")
            return False, f"Archivo corrupto: {e}"
    
    # Si solo existe cv_results pero no modelo, puede ser completado parcialmente
    if checkpoints['cv_results'] and not checkpoints['model']:
        return False, "CV completado pero falta modelo - re-ejecutar"
    
    return False, "Loop no completado"


def get_next_available_loop(output_dir: str, start_from: int = 1, max_loops: int = None) -> Optional[int]:
    """
    Encuentra el próximo loop disponible para ejecutar.
    
    Args:
        output_dir: Directorio de salida
        start_from: Loop desde el cual empezar a buscar
        max_loops: Límite máximo de loops a considerar
    
    Returns:
        Número del próximo loop disponible, o None si no hay
    """
    loop_counter = start_from
    
    while True:
        if max_loops and loop_counter > max_loops:
            return None
        
        skip, reason = should_skip_loop(output_dir, loop_counter)
        if not skip:
            print(f"📌 Próximo loop disponible: {loop_counter} ({reason})")
            return loop_counter
        
        print(f"⏭️ Saltando loop {loop_counter}: {reason}")
        loop_counter += 1


class ResumableLoopIterator:
    """
    Iterador reanudable para loops. Permite pausar y reanudar ejecuciones.
    """
    
    def __init__(self, output_dir: str, total_loops: int = None, start_from: int = 1):
        self.output_dir = output_dir
        self.total_loops = total_loops
        self.start_from = start_from
        self.state_manager = LoopStateManager(output_dir)
        self.current_loop = None
        
        # Si hay estado guardado, usarlo
        if self.state_manager.state['last_loop_completed'] > 0:
            self.start_from = self.state_manager.state['last_loop_completed'] + 1
            print(f"🔄 Reanudando desde loop {self.start_from}")
    
    def __iter__(self):
        self.current_loop = self.start_from
        return self
    
    def __next__(self) -> int:
        if self.total_loops and self.current_loop > self.total_loops:
            raise StopIteration
        
        # Verificar si este loop ya fue completado
        if self.state_manager.is_loop_completed(self.current_loop):
            self.current_loop += 1
            return self.__next__()
        
        # Verificar si este loop debe ser saltado por checkpoints existentes
        skip, reason = should_skip_loop(self.output_dir, self.current_loop)
        if skip:
            print(f"⏭️ Saltando loop {self.current_loop}: {reason}")
            self.state_manager.mark_loop_completed(self.current_loop)
            self.current_loop += 1
            return self.__next__()
        
        loop = self.current_loop
        self.current_loop += 1
        return loop
    
    def mark_success(self, loop: int, results: Dict = None):
        """Marca un loop como exitoso"""
        self.state_manager.mark_loop_completed(loop, results)
    
    def mark_failure(self, loop: int, error: str = None):
        """Marca un loop como fallido"""
        self.state_manager.mark_loop_failed(loop, error)
    
    def get_progress(self) -> Dict:
        """Obtiene el progreso actual"""
        completed = len(self.state_manager.get_completed_loops())
        if self.total_loops:
            return {
                'completed': completed,
                'total': self.total_loops,
                'percentage': (completed / self.total_loops) * 100,
                'remaining': self.total_loops - completed
            }
        return {
            'completed': completed,
            'total': 'unknown',
            'percentage': 0,
            'remaining': 'unknown'
        }


# ============================================================================
# INTEGRACIÓN CON EL PIPELINE EXISTENTE
# ============================================================================

async def run_pipeline_with_resume(args):
    """
    Función principal que integra la funcionalidad de reanudación.
    """
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Calcular número total de loops necesarios
    total_samples = sum(1 for _ in Path(args.audio_dir).glob("*.wav")) * 100  # Estimado
    samples_per_baseline = 4000
    total_loops = (total_samples + samples_per_baseline - 1) // samples_per_baseline
    total_loops = min(total_loops, 50)  # Límite máximo
    
    print(f"""
    {'='*60}
    🚀 PIPELINE AOC CON REANUDACIÓN
    {'='*60}
    Directorio: {output_dir}
    Total loops esperados: {total_loops}
    Muestras por loop: {samples_per_baseline}
    {'='*60}
    """)
    
    # Crear iterador reanudable
    loop_iterator = ResumableLoopIterator(
        output_dir=output_dir,
        total_loops=total_loops,
        start_from=args.start_loop if hasattr(args, 'start_loop') else 1
    )
    
    all_results = []
    
    for loop_counter in loop_iterator:
        print(f"\n{'='*60}")
        print(f"🔄 EJECUTANDO LOOP {loop_counter}/{total_loops}")
        print(f"{'='*60}")
        
        # Verificar checkpoints existentes nuevamente (por si acaso)
        skip, reason = should_skip_loop(output_dir, loop_counter)
        if skip:
            print(f"⏭️ Loop {loop_counter} ya completado, saltando...")
            loop_iterator.mark_success(loop_counter)
            continue
        
        loop_iterator.state_manager.mark_loop_started(loop_counter)
        
        try:
            # Ejecutar el pipeline para este loop
            results = await run_full_pipeline_for_loop(
                args=args,
                loop_counter=loop_counter,
                output_dir=output_dir
            )
            
            # Marcar como exitoso
            loop_iterator.mark_success(loop_counter, results)
            all_results.append(results)
            
            # Mostrar progreso
            progress = loop_iterator.get_progress()
            print(f"\n📊 Progreso: {progress['percentage']:.1f}% "
                  f"({progress['completed']}/{progress['total']})")
            
        except Exception as e:
            print(f"❌ Loop {loop_counter} falló: {e}")
            loop_iterator.mark_failure(loop_counter, str(e))
            
            # Decidir si continuar o detenerse
            if args.stop_on_failure:
                print("🛑 Deteniendo por --stop_on_failure")
                break
            else:
                print("⚠️ Continuando con siguiente loop...")
                continue
    
    # Resumen final
    print(f"\n{'='*60}")
    print("📊 RESUMEN FINAL")
    print(f"{'='*60}")
    print(loop_iterator.state_manager.get_summary())
    
    # Guardar resultados combinados
    final_results = {
        'completed_loops': loop_iterator.state_manager.get_completed_loops(),
        'failed_loops': loop_iterator.state_manager.get_failed_loops(),
        'total_loops': total_loops,
        'all_results': all_results,
        'final_state': loop_iterator.state_manager.state
    }
    
    final_path = os.path.join(output_dir, "pipeline_complete_results.json")
    with open(final_path, 'w') as f:
        json.dump(final_results, f, indent=2, default=str)
    
    print(f"\n💾 Resultados finales guardados en: {final_path}")
    
    return final_results


async def run_full_pipeline_for_loop(args, loop_counter: int, output_dir: str) -> Dict:
    """
    Ejecuta un loop específico del pipeline.
    """
    # Crear directorio para este loop
    loop_dir = os.path.join(output_dir, f"loop_{loop_counter:04d}")
    os.makedirs(loop_dir, exist_ok=True)
    
    # Configurar args específicos del loop
    args.output_dir = loop_dir
    args.loop_counter = loop_counter
    
    # Ejecutar el pipeline original
    results = await run_full_pipeline(args)
    
    return results


# ============================================================================
# ARGUMENTOS ADICIONALES PARA EL PARSER
# ============================================================================

def add_resume_arguments(parser):
    """Agrega argumentos para la funcionalidad de reanudación"""
    parser.add_argument(
        "--resume", 
        action="store_true", 
        default=False,
        help="Reanudar desde el último loop completado"
    )
    parser.add_argument(
        "--start_loop", 
        type=int, 
        default=1,
        help="Loop desde el cual comenzar (si no se usa --resume)"
    )
    parser.add_argument(
        "--max_loops", 
        type=int, 
        default=None,
        help="Número máximo de loops a ejecutar"
    )
    parser.add_argument(
        "--stop_on_failure", 
        action="store_true", 
        default=False,
        help="Detener ejecución si un loop falla"
    )
    parser.add_argument(
        "--force_rerun", 
        action="store_true", 
        default=False,
        help="Forzar re-ejecución aunque existan checkpoints"
    )
    return parser