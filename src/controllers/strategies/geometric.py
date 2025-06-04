from src.models.base.sia import SIA
from src.models.core.solution import Solution
from src.constants.models import GEOMETRIC_LABEL
import numpy as np
import time
from src.funcs.base import ABECEDARY
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

class Geometric(SIA):
    def __init__(self, gestor):
        super().__init__(gestor)
        self.tabla_costos = {}  # Dict[str][int][int]
    
    def aplicar_estrategia(self, condicion, alcance, mecanismo):
        