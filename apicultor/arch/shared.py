import multiprocessing as mp
import numpy as np
import hashlib
import pickle
import os
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class SharedMemoryStore:
    """
    Almacenamiento de datos en memoria compartida entre procesos.
    Usa multiprocessing.shared_memory para compartir arrays numpy.
    """
    
    def __init__(self, name: str = None, max_entries: int = 10000, entry_size: int = 2048):
        self.name = name or f"shared_memory_{uuid.uuid4().hex[:8]}"
        self.max_entries = max_entries
        self.entry_size = entry_size
        self._initialized = False
        self._shm = None
        self._lock = mp.Lock()
        
        # Metadata en archivo compartido
        self.metadata_path = os.path.join("/tmp", f"shared_memory_meta_{self.name}.pkl")
        self._init_metadata()
    
    def _init_metadata(self):
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, 'rb') as f:
                self._metadata = pickle.load(f)
        else:
            self._metadata = {
                'n_entries': 0,
                'entry_hashes': [],
                'entry_metadata': [],
                'created_at': datetime.now().isoformat(),
                'name': self.name
            }
            self._save_metadata()
    
    def _save_metadata(self):
        with open(self.metadata_path, 'wb') as f:
            pickle.dump(self._metadata, f)
    
    def _ensure_initialized(self):
        if self._initialized:
            return
        try:
            self._shm = shared_memory.SharedMemory(name=self.name)
            self._initialized = True
        except FileNotFoundError:
            total_size = self.max_entries * self.entry_size * 8
            self._shm = shared_memory.SharedMemory(
                name=self.name,
                create=True,
                size=total_size
            )
            arr = np.ndarray((self.max_entries, self.entry_size), dtype=np.float64, buffer=self._shm.buf)
            arr.fill(0)
            self._initialized = True
    
    def add_entry(self, data: np.ndarray, metadata: Dict = None) -> bool:
        with self._lock:
            self._ensure_initialized()
            
            data_hash = hashlib.md5(data.tobytes()).hexdigest()
            if data_hash in self._metadata['entry_hashes']:
                return False
            
            data_flat = data.flatten()
            if len(data_flat) > self.entry_size:
                data_flat = data_flat[:self.entry_size]
            elif len(data_flat) < self.entry_size:
                data_flat = np.pad(data_flat, (0, self.entry_size - len(data_flat)))
            
            idx = self._metadata['n_entries']
            
            if idx >= self.max_entries:
                arr = np.ndarray((self.max_entries, self.entry_size), dtype=np.float64, buffer=self._shm.buf)
                arr[:-1] = arr[1:]
                idx = self.max_entries - 1
                if self._metadata['entry_hashes']:
                    self._metadata['entry_hashes'].pop(0)
                    self._metadata['entry_metadata'].pop(0)
                self._metadata['n_entries'] = self.max_entries
            
            arr = np.ndarray((self.max_entries, self.entry_size), dtype=np.float64, buffer=self._shm.buf)
            arr[idx] = data_flat
            
            self._metadata['entry_hashes'].append(data_hash)
            self._metadata['entry_metadata'].append({
                'timestamp': datetime.now().isoformat(),
                'metadata': metadata or {},
                'hash': data_hash
            })
            self._metadata['n_entries'] = min(self._metadata['n_entries'] + 1, self.max_entries)
            
            self._save_metadata()
            return True
    
    def get_entries(self, top_k: int = None) -> List[np.ndarray]:
        with self._lock:
            self._ensure_initialized()
            n = min(self._metadata['n_entries'], self.max_entries)
            if n == 0:
                return []
            if top_k is not None:
                n = min(n, top_k)
            arr = np.ndarray((self.max_entries, self.entry_size), dtype=np.float64, buffer=self._shm.buf)
            return [arr[i].copy() for i in range(n)]
    
    def get_entries_with_metadata(self, top_k: int = None) -> List[Tuple[np.ndarray, Dict]]:
        with self._lock:
            self._ensure_initialized()
            n = min(self._metadata['n_entries'], self.max_entries)
            if n == 0:
                return []
            if top_k is not None:
                n = min(n, top_k)
            arr = np.ndarray((self.max_entries, self.entry_size), dtype=np.float64, buffer=self._shm.buf)
            return [(arr[i].copy(), self._metadata['entry_metadata'][i] if i < len(self._metadata['entry_metadata']) else {}) for i in range(n)]
    
    def get_metadata(self) -> Dict:
        with self._lock:
            return {
                'name': self.name,
                'n_entries': self._metadata['n_entries'],
                'max_entries': self.max_entries,
                'entry_size': self.entry_size,
                'created_at': self._metadata.get('created_at'),
                'unique_hashes': len(set(self._metadata.get('entry_hashes', [])))
            }
    
    def clear(self):
        with self._lock:
            self._ensure_initialized()
            arr = np.ndarray((self.max_entries, self.entry_size), dtype=np.float64, buffer=self._shm.buf)
            arr.fill(0)
            self._metadata['n_entries'] = 0
            self._metadata['entry_hashes'] = []
            self._metadata['entry_metadata'] = []
            self._save_metadata()
    
    def close(self):
        if self._shm is not None:
            self._shm.close()
            self._initialized = False
    
    def unlink(self):
        self.close()
        if self._shm is not None:
            self._shm.unlink()
            self._shm = None
        if os.path.exists(self.metadata_path):
            os.remove(self.metadata_path)
        self._initialized = False
