from src.models.base.sia import SIA
from src.models.core.solution import Solution
from src.constants.models import GEOMETRIC_LABEL
import numpy as np
import time
from src.funcs.base import ABECEDARY
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
import itertools
import src.controllers.strategies.TablaSecuencialCostos as  TablaSecuencialCostos

class Geometric(SIA):
    def __init__(self, gestor):
        super().__init__(gestor)
        self.tabla_costos = None

    def aplicar_estrategia(self, condicion: str, alcance: str, mecanismo: str):
        """
        Aplica la estrategia geométrica para encontrar la mejor bipartición del sistema,
        usando la estructura del hipercubo y la distancia de Hamming.
        """
        
        self.sia_cargar_tpm()
        
        
        # Paso 1: preparar el subsistema usando el método de SIA
        self.sia_preparar_subsistema(condicion, alcance, mecanismo)

        # Paso 2: obtener la dimensión del sistema (número de variables)
        n = len(self.sia_gestor.estado_inicial)

        # Paso 3: crear la tabla de costos (si aplica en tu lógica)
        self.tabla_costos = TablaSecuencialCostos(n=n, max_distance=7)

        # Paso 4: generar biparticiones posibles (solo hasta la mitad por simetría)
        indices = list(range(n))
        mejor_particion = None
        mejor_costo = float('inf')

        for k in range(1, n // 2 + 1):
            for grupoA in itertools.combinations(indices, k):
                grupoB = tuple(i for i in indices if i not in grupoA)
                costo = self.evaluar_particion(grupoA, grupoB)
                if costo < mejor_costo:
                    mejor_costo = costo
                    mejor_particion = (grupoA, grupoB)

        # Paso 5: retornar la solución
        return Solution(
            label=GEOMETRIC_LABEL,
            particion=mejor_particion,
            tiempo_ejecucion=time.time() - self.sia_tiempo_inicio
        )

    def evaluar_particion(self, grupoA, grupoB):
        """
        Cuenta cuántas transiciones elementales (cambio de un bit) cruzan la bipartición.
        Los estados se representan como cadenas binarias de tamaño n.
        """
        n = len(self.sia_gestor.estado_inicial)
        grupoA = set(grupoA)
        grupoB = set(grupoB)
        costo = 0

        # Genera todos los estados posibles como cadenas binarias de tamaño n
        for estado_int in range(2 ** n):
            estado_bin = format(estado_int, f'0{n}b')  # Ej: '0101'
            for i in range(n):
                # Cambia el bit i
                nuevo_estado_bin = list(estado_bin)
                nuevo_estado_bin[i] = '1' if estado_bin[i] == '0' else '0'
                nuevo_estado_bin = ''.join(nuevo_estado_bin)
                nuevo_estado_int = int(nuevo_estado_bin, 2)
                if nuevo_estado_int > estado_int:  # Evita contar dos veces cada transición
                    # Si el bit cambiado pertenece a grupoA y el resto a grupoB, o viceversa
                    if (i in grupoA and any(j in grupoB for j in range(n) if estado_bin[j] != nuevo_estado_bin[j])) or \
                       (i in grupoB and any(j in grupoA for j in range(n) if estado_bin[j] != nuevo_estado_bin[j])):
                        costo += 1
        return costo
    
    

        
        