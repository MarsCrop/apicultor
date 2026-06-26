import numpy as np
from typing import Any, Dict, List, Tuple
from collections import defaultdict
import pickle
import json

def inspect_exp_data_structure(
    exp_data: Any,
    max_depth: int = 5,
    max_items: int = 3,
    verbose: bool = True
) -> Dict:
    """
    Inspecciona la estructura completa de exp_data y reporta shapes/types de cada componente.
    
    Args:
        exp_data: Datos de experimento (puede ser lista, dict, o nested)
        max_depth: Profundidad máxima de recursión
        max_items: Máximo de items a mostrar en listas/dicts
        verbose: Si True, imprime el reporte
    
    Returns:
        Diccionario con la estructura encontrada
    """
    
    def get_shape_info(obj: Any) -> Dict:
        """Obtiene información de forma/tipo de un objeto"""
        if isinstance(obj, np.ndarray):
            return {
                'type': 'numpy_array',
                'shape': obj.shape,
                'dtype': str(obj.dtype),
                'size': obj.size,
                'min': float(np.min(obj)) if obj.size > 0 else None,
                'max': float(np.max(obj)) if obj.size > 0 else None,
                'mean': float(np.mean(obj)) if obj.size > 0 else None
            }
        elif isinstance(obj, list):
            return {
                'type': 'list',
                'length': len(obj),
                'sample_types': list(set(type(item).__name__ for item in obj[:max_items]))
            }
        elif isinstance(obj, dict):
            return {
                'type': 'dict',
                'keys': list(obj.keys())[:max_items],
                'num_keys': len(obj.keys())
            }
        elif isinstance(obj, tuple):
            return {
                'type': 'tuple',
                'length': len(obj)
            }
        else:
            return {
                'type': type(obj).__name__,
                'value': str(obj)[:100] if obj else None
            }
    
    def recursive_inspect(
        data: Any,
        path: str,
        depth: int = 0,
        structure: Dict = None
    ) -> Dict:
        if structure is None:
            structure = {}
        
        if depth > max_depth:
            structure[path] = {'type': 'max_depth_reached', 'truncated': True}
            return structure
        
        shape_info = get_shape_info(data)
        
        if isinstance(data, dict):
            structure[path] = {
                'type': 'dict',
                'num_keys': len(data),
                'keys': list(data.keys())[:max_items],
                'shape_info': shape_info
            }
            # Recursivamente inspeccionar cada clave
            for key, value in list(data.items())[:max_items]:
                new_path = f"{path}.{key}" if path else key
                recursive_inspect(value, new_path, depth + 1, structure)
        
        elif isinstance(data, (list, tuple)):
            structure[path] = {
                'type': 'list' if isinstance(data, list) else 'tuple',
                'length': len(data),
                'shape_info': shape_info
            }
            # Inspeccionar primeros items
            for i, item in enumerate(data[:max_items]):
                new_path = f"{path}[{i}]"
                recursive_inspect(item, new_path, depth + 1, structure)
        
        elif isinstance(data, np.ndarray):
            structure[path] = {
                'type': 'numpy_array',
                'shape': data.shape,
                'dtype': str(data.dtype),
                'stats': {
                    'min': float(np.min(data)),
                    'max': float(np.max(data)),
                    'mean': float(np.mean(data)),
                    'std': float(np.std(data))
                } if data.size > 0 else None,
                'shape_info': shape_info
            }
        
        else:
            structure[path] = shape_info
        
        return structure
    
    def flatten_structure(structure: Dict, prefix: str = "") -> List[Tuple[str, Dict]]:
        """Aplana la estructura para fácil visualización"""
        flat = []
        for key, value in structure.items():
            full_key = f"{prefix}.{key}" if prefix else key
            flat.append((full_key, value))
            if isinstance(value, dict) and 'type' in value and value['type'] in ['dict', 'list', 'tuple']:
                # Buscar subkeys
                for subkey, subvalue in value.items():
                    if subkey not in ['type', 'num_keys', 'keys', 'length', 'shape_info']:
                        if isinstance(subvalue, dict):
                            flat.extend(flatten_structure({subkey: subvalue}, full_key))
        return flat
    
    # Inspeccionar estructura
    structure = recursive_inspect(exp_data, "")
    
    # Aplanar para reporte
    flat_items = flatten_structure(structure)
    
    # Crear reporte
    report = {
        'root_type': type(exp_data).__name__,
        'structure': structure,
        'flat_view': flat_items,
        'summary': {
            'total_nodes': len(flat_items),
            'has_numpy_arrays': any(
                item[1].get('type') == 'numpy_array' 
                for item in flat_items
            ),
            'max_depth': max_depth,
            'items_shown': max_items
        }
    }
    
    # Buscar entradas con features_test (lo que necesita compute_feature_importance)
    feature_entries = []
    for path, info in flat_items:
        if info.get('type') == 'numpy_array' and 'shape' in info:
            shape = info['shape']
            if len(shape) == 2 and shape[1] > 0:  # Matriz 2D
                feature_entries.append({
                    'path': path,
                    'shape': shape,
                    'dtype': info.get('dtype'),
                    'stats': info.get('stats')
                })
    
    report['feature_entries_found'] = feature_entries
    
    if verbose:
        print_report(report, exp_data)
    
    return report


def print_report(report: Dict, exp_data: Any = None):
    """Imprime el reporte de estructura de forma legible"""
    print("\n" + "="*80)
    print("📊 INSPECCIÓN DE exp_data")
    print("="*80)
    
    print(f"\n📁 Tipo raíz: {report['root_type']}")
    print(f"📏 Nodos totales: {report['summary']['total_nodes']}")
    print(f"🔢 Arrays NumPy encontrados: {report['summary']['has_numpy_arrays']}")
    
    print("\n" + "-"*80)
    print("📋 ESTRUCTURA COMPLETA:")
    print("-"*80)
    
    for path, info in report['flat_view']:
        indent = "  " * (path.count('.') + path.count('['))
        if info.get('type') == 'numpy_array':
            shape = info.get('shape', 'unknown')
            stats = info.get('stats', {})
            print(f"{indent}📍 {path}")
            print(f"{indent}   └─ 📐 shape: {shape}")
            print(f"{indent}   └─ 📊 stats: min={stats.get('min', 'N/A'):.4f}, "
                  f"max={stats.get('max', 'N/A'):.4f}, "
                  f"mean={stats.get('mean', 'N/A'):.4f}")
        elif info.get('type') in ['dict', 'list', 'tuple']:
            length = info.get('num_keys', info.get('length', '?'))
            keys = info.get('keys', [])
            print(f"{indent}📍 {path} [{info['type']}, {length} elementos]")
            if keys:
                print(f"{indent}   └─ 🔑 keys: {keys[:3]}")
        else:
            print(f"{indent}📍 {path}: {info.get('type')} = {info.get('value', 'N/A')}")
    
    print("\n" + "-"*80)
    print("🎯 ENTRADAS CON FEATURES (para compute_feature_importance):")
    print("-"*80)
    
    if report['feature_entries_found']:
        for entry in report['feature_entries_found']:
            print(f"  📍 {entry['path']}")
            print(f"     └─ shape: {entry['shape']}")
            print(f"     └─ dtype: {entry['dtype']}")
            if entry.get('stats'):
                print(f"     └─ mean: {entry['stats']['mean']:.4f}")
    else:
        print("  ⚠️ No se encontraron entradas con features 2D")
    
    print("\n" + "-"*80)
    print("💡 RECOMENDACIONES:")
    print("-"*80)
    
    if not report['feature_entries_found']:
        print("  ⚠️ exp_data no contiene arrays 2D válidos para feature importance")
        print("  → Verificar que GridSearch haya generado exp_data correctamente")
        print("  → Revisar si cv_results_loop_*.pkl está corrupto")
    
    if report['root_type'] == 'NoneType':
        print("  ❌ exp_data es None - no se generaron datos")
    elif report['root_type'] == 'list' and report['flat_view'][0][1].get('length', 0) == 0:
        print("  ❌ exp_data es una lista vacía")
    elif report['root_type'] == 'dict' and report['structure'].get('', {}).get('num_keys', 0) == 0:
        print("  ❌ exp_data es un diccionario vacío")
    else:
        print("  ✅ exp_data tiene estructura válida")
        print("  → Se puede proceder con compute_feature_importance")
    
    print("="*80 + "\n")


def load_and_inspect_exp_data(
    output_dir: str,
    loop_counter: int = None
) -> Dict:
    """
    Carga el archivo cv_results_loop_*.pkl y analiza exp_data.
    
    Args:
        output_dir: Directorio donde están los resultados
        loop_counter: Número de loop específico (si None, busca el más reciente)
    
    Returns:
        Diccionario con estructura y contenido de exp_data
    """
    import glob
    
    if loop_counter is None:
        # Buscar el archivo más reciente
        pattern = os.path.join(output_dir, "cv_results_loop_*.pkl")
        files = glob.glob(pattern)
        if not files:
            print(f"❌ No se encontraron archivos cv_results_loop_*.pkl en {output_dir}")
            return {'error': 'no_files_found'}
        latest = max(files, key=os.path.getctime)
        print(f"📂 Usando archivo más reciente: {latest}")
    else:
        latest = os.path.join(output_dir, f"cv_results_loop_{loop_counter}.pkl")
        if not os.path.exists(latest):
            print(f"❌ No existe el archivo: {latest}")
            return {'error': 'file_not_found'}
    
    # Cargar el archivo
    try:
        with open(latest, 'rb') as f:
            cv_results = pickle.load(f)
        print(f"✅ Archivo cargado: {latest}")
        print(f"🔑 Keys en cv_results: {list(cv_results.keys())}")
    except Exception as e:
        print(f"❌ Error cargando archivo: {e}")
        return {'error': 'load_failed', 'exception': str(e)}
    
    # Extraer exp_data
    exp_data = cv_results.get('exp_data')
    
    if exp_data is None:
        print("⚠️ cv_results no contiene 'exp_data'")
        # Buscar otras posibles ubicaciones
        for key, value in cv_results.items():
            if 'exp' in key.lower() or 'data' in key.lower():
                print(f"  → Posible alternativa: '{key}' de tipo {type(value).__name__}")
                exp_data = value
                break
    
    # Inspeccionar estructura
    if exp_data is not None:
        report = inspect_exp_data_structure(exp_data, verbose=True)
        report['source_file'] = latest
        report['cv_results_keys'] = list(cv_results.keys())
        return report
    else:
        print("❌ exp_data es None - no se encontraron datos de experimento")
        return {
            'error': 'exp_data_none',
            'source_file': latest,
            'cv_results_keys': list(cv_results.keys())
        }


def extract_valid_entries_for_feature_importance(exp_data: Any) -> List:
    """
    Extrae entradas válidas para compute_feature_importance desde exp_data.
    """
    valid_entries = []
    
    def recursive_extract(data, depth=0):
        if depth > 10:
            return
        
        if isinstance(data, dict):
            # Buscar patrones conocidos
            if 'features_test' in data and isinstance(data['features_test'], np.ndarray):
                if data['features_test'].ndim == 2 and data['features_test'].shape[0] > 0:
                    entry = [
                        data.get('mse_val', 0),
                        data['features_test'],
                        data.get('scores', np.array([])),
                        data.get('weights', np.array([])),
                        data.get('hyp', 'unknown'),
                        data.get('passed', True),
                        data.get('mse_val', 0),
                        data.get('diff', 0)
                    ]
                    valid_entries.append(entry)
                    print(f"✅ Extraído entry con features_test shape={data['features_test'].shape}")
            
            # Recursión en valores
            for value in data.values():
                recursive_extract(value, depth + 1)
        
        elif isinstance(data, (list, tuple)):
            for item in data:
                recursive_extract(item, depth + 1)
                
            # Si es lista de tuplas con estructura conocida (de multiple_hypothesis_testing)
            if len(data) > 0 and isinstance(data[0], (list, tuple)) and len(data[0]) >= 8:
                for item in data:
                    if len(item) >= 2 and isinstance(item[1], np.ndarray):
                        if item[1].ndim == 2 and item[1].shape[0] > 0:
                            # Asegurar que tenga la longitud correcta
                            while len(item) < 8:
                                item = list(item) + [0]
                            valid_entries.append(list(item))
    
    recursive_extract(exp_data)
    
    print(f"\n📊 Total entradas válidas encontradas: {len(valid_entries)}")
    return valid_entries


# EJEMPLO DE USO
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="results/")
    parser.add_argument("--loop", type=int, default=None)
    args = parser.parse_args()
    
    # Inspeccionar exp_data del archivo
    report = load_and_inspect_exp_data(args.output_dir, args.loop)
    
    if not report.get('error') and 'exp_data' in report:
        # Extraer entradas válidas
        exp_data = report.get('exp_data')
        if exp_data:
            valid_entries = extract_valid_entries_for_feature_importance(exp_data)
            print(f"\n✅ {len(valid_entries)} entradas listas para compute_feature_importance")