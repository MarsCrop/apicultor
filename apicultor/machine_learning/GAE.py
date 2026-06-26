# Agregar después de los imports existentes
from typing import Tuple, Dict, Any
import numpy as np
import os
import sys
import asyncio
import argparse
import logging
import hashlib
import json
import pickle
import warnings
from apicultor.machine_learning.GAE import *
from collections import defaultdict
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from scipy.signal import stft as scipy_stft, istft as scipy_istft
from scipy.io.wavfile import write as wav_write
from soundfile import read as sf_read
