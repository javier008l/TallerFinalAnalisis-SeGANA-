from src.models.base.sia import SIA
from src.models.core.system import System
import numpy as np
from typing import Tuple
from src.models.core.solution import Solution
from src.funcs.format import fmt_biparte_q
from src.constants.models import GEOMETRIC_LABEL
from src.constants.base import ACTUAL, EFECTO
import time


class GeometricSIA(SIA):
    def __init__(self, gestor):
        super().__init__(gestor)
        self.tensors = None
        self.transiciones = {}

    def aplicar_estrategia(self, condicion: str, alcance_mask: str, mecanismo_mask: str):
        """
        Implementación de la estrategia geométrica.
        """
        self.sia_preparar_subsistema(condicion, alcance_mask, mecanismo_mask)
        subsistema: System = self.sia_subsistema
        print(f"ya hice el Subsistema preparado")
        self.tensors = self.descomponer_en_tensores(subsistema)
        print(f"ya hice la descomposición en tensores")
        self.calcular_tabla_transiciones(subsistema)
        print(f"ya hice la tabla de transiciones")
        candidatos = self.identificar_biparticiones_candidatas()
        print(f"Candidatos identificados: {candidatos}")
        alcance, mecanismo, mejor_phi, mejor_distribucion = self.evaluar_biparticiones(candidatos, subsistema)

        # Formatear partición como en QNodes
        fmt_mip = fmt_biparte_q(
            list((EFECTO, i) for i in alcance),
            list((ACTUAL, i) for i in mecanismo)
        )

        return Solution(
            estrategia=GEOMETRIC_LABEL,
            perdida=mejor_phi,
            distribucion_subsistema=subsistema.distribucion_marginal(),
            distribucion_particion=mejor_distribucion,
            tiempo_total=time.time() - self.sia_tiempo_inicio,
            particion=fmt_mip
        )

    
    def descomponer_en_tensores(self, sistema: System):
        return {cube.indice: cube.data for cube in sistema.ncubos}
    
    def calcular_tabla_transiciones(self, sistema: System):
        n = sistema.ncubos[0].data.ndim
        for idx, tensor in self.tensors.items():
            tabla = np.zeros((2**n, 2**n), dtype=np.float32)
            cache = {}  # Memoización local por tensor

            for i in range(2**n):
                for j in range(2**n):
                    tabla[i, j] = self.costo_transicion(i, j, tensor, cache, idx)
            self.transiciones[idx] = tabla
            print(f"Tabla de transiciones {tabla} calculada.")

    def costo_transicion(self, i: int, j: int, tensor: np.ndarray, cache: dict, tensor_id: int) -> float:
        """
        Calcula el costo de transición t(i, j) entre estados binarios del tensor.
        Usa memoización explícita para evitar recursión infinita.
        """
        clave = (i, j, tensor_id)
        if clave in cache:
            return cache[clave]

        d = bin(i ^ j).count("1")
        if d == 0:
            cache[clave] = 0.0
            return 0.0

        gamma = 2 ** -d
        xi = tensor.flat[i]
        xj = tensor.flat[j]
        costo_directo = abs(xi - xj)

         # Vecinos de i a un bit de distancia que estén más cerca de j
        vecinos = [
            k for k in range(2 ** tensor.ndim)
            if bin(i ^ k).count("1") == 1 and bin(k ^ j).count("1") < d
        ]

        suma_vecinos = sum(
            self.costo_transicion(k, j, tensor, cache, tensor_id) for k in vecinos
        )

        total = gamma * (costo_directo + suma_vecinos)
        cache[clave] = total
        return total
    
    def identificar_biparticiones_candidatas(self) -> list[Tuple[np.ndarray, np.ndarray]]:
            n = len(self.tensors)
            candidatos = []
            for i in range(n):
                alcance = np.array([i], dtype=np.int8)
                mecanismo = np.setdiff1d(np.arange(n), alcance)
                candidatos.append((alcance, mecanismo))
            return candidatos
    
    def evaluar_biparticiones(self, candidatos, sistema: System):
        mejor = None
        mejor_phi = float("inf")
        mejor_distribucion = None

        for alcance, mecanismo in candidatos:
            if mecanismo.size == 0:
                continue
            bipartido = sistema.bipartir(alcance, mecanismo)
            distribucion = bipartido.distribucion_marginal()
            phi = np.sum(distribucion)
            if phi < mejor_phi:
                mejor_phi = phi
                mejor = (alcance, mecanismo)
                mejor_distribucion = distribucion
        alcance, mecanismo = mejor
        return alcance, mecanismo, mejor_phi, mejor_distribucion
