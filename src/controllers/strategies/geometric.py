#from src.controllers.strategies.ScalableCostTable import example_large_system
from src.models.base.sia import SIA
from src.models.core.system import System
import numpy as np
from typing import Tuple
from src.models.core.solution import Solution
from src.funcs.format import fmt_biparte_q
from src.constants.models import GEOMETRIC_LABEL
from src.constants.base import ACTUAL, EFECTO
import time
from itertools import product
from functools import lru_cache


class GeometricSIA(SIA):
    def __init__(self, gestor):
        super().__init__(gestor)
        self.tensors = None
        self.transiciones = {}
        self.memoria = dict()
        self.X_v_global = dict()  # Almacena X_v globalmente para evitar recalcularlo
        self.TablaCostos = {}
    def aplicar_estrategia(self, condicion: str, alcance_mask: str, mecanismo_mask: str):
        """
        Implementación de la estrategia geométrica.
        """
        self.sia_preparar_subsistema(condicion, alcance_mask, mecanismo_mask)
        subsistema: System = self.sia_subsistema        
        candidatos = self.identificar_biparticiones_candidatas(3)
        alcance, mecanismo, mejor_phi, mejor_distribucion = self.evaluar_biparticiones(candidatos, subsistema)
        # Obtener índices de presente (t) y futuro (t+1)
        self.indices_alcance = self.sia_subsistema.indices_ncubos
        self.indices_mecanismo = self.sia_subsistema.dims_ncubos
        print(f"Indices alcance: {self.indices_alcance}, Indices mecanismo: {self.indices_mecanismo}")
        tpm = self.sia_cargar_tpm()
        origen =  mecanismo_mask
        destino = alcance_mask
        # Usar hash de la TPM como clave para X_v_global
        tpm_hash = hash(tpm.tobytes())
        if tpm_hash not in self.X_v_global:
            self.X_v_global[tpm_hash] = self.construir_X_v_desde_tpm(tpm)
        self.X_v = self.X_v_global[tpm_hash]

        self.TablaCostos[(origen, destino)] = self.calcular_costo_simple(origen, destino, self.X_v)
        print(f"Tabla costo: ({origen}), ({destino}):  {self.TablaCostos[(origen, destino)]}")


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
    
    def identificar_biparticiones_candidatas(self,n) -> list[Tuple[np.ndarray, np.ndarray]]:
            
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
        
    def construir_X_v_desde_tpm(self, tpm: np.ndarray, estados_relevantes=None) -> dict:
        n = int(np.log2(tpm.shape[0]))
        X_v = {}
        if estados_relevantes is None:
            for j in range(tpm.shape[0]):
                estado_bin = format(j, f'0{n}b')
                valor = tpm[j]
                X_v[estado_bin] = tuple(valor) if isinstance(valor, np.ndarray) else valor
        else:
            for estado_bin in estados_relevantes:
                j = int(estado_bin, 2)
                valor = tpm[j]
                X_v[estado_bin] = tuple(valor) if isinstance(valor, np.ndarray) else valor
        return X_v


    def hamming(self, a, b):
        return sum(x != y for x, y in zip(a, b))

    def vecinos_hamming_1(self, estado):
        """Genera todos los vecinos a distancia Hamming 1"""
        vecinos = []
        for i in range(len(estado)):
            nuevo = estado[:i] + ('0' if estado[i] == '1' else '1') + estado[i+1:]
            vecinos.append(nuevo)
        return vecinos

    def generar_pares_hamming_1(self, n):
        """
        Genera pares (a, b) de estados binarios de n bits que solo difieren en un bit.
        """
        estados = [''.join(bits) for bits in product('01', repeat=n)]
        pares = set()

        for estado in estados:
            for vecino in self.vecinos_hamming_1(estado):
                # Evitar duplicados como ('000', '001') y ('001', '000')
                if (vecino, estado) not in pares:
                    pares.add((estado, vecino))

        return sorted(pares)

    @lru_cache(maxsize=None)
    def calcular_costo_ruta_especifica(self, origen, destino, X_v_tuple):
        """
        Calcula el costo de una ruta específica por variable, usando memoización.
        Retorna una lista con el costo por cada variable.
        """
        X_v = dict(X_v_tuple)
        d = self.hamming(origen, destino)
        n = len(origen)
        # Caso base: si son iguales, costo cero para todas las variables
        if d == 0:
            return [0.0] * n

        gamma = 2 ** (-d)
        # Costo directo por variable
        costo_directo = [abs(X_v[origen][i] - X_v[destino][i]) for i in range(n)]

        # Acumulado por variable (suma de listas)
        acumulado = [0.0] * n
        for vecino in self.vecinos_hamming_1(origen):
            if self.hamming(vecino, destino) < d:
                costo_vecino = self.calcular_costo_ruta_especifica(vecino, destino, X_v_tuple)
                acumulado = [a + b for a, b in zip(acumulado, costo_vecino)]

        # Costo total por variable
        costo_total = [gamma * (cd + ac) for cd, ac in zip(costo_directo, acumulado)]
        return costo_total
    # @lru_cache(maxsize=None)
    # def calcular_costo_ruta_especifica(self, origen, destino, X_v_tuple):
    #     """
    #     Calcula el costo de una ruta específica usando memoización.
    #     X_v_tuple debe ser una tupla de (estado, valor) para permitir hashing.
    #     """
    #     # Convertir tupla de vuelta a diccionario para cálculos
    #     X_v = dict(X_v_tuple)
    #     #print(f"Calculando costo entre {origen} y {destino} con X_v: {X_v}")
    #     d = self.hamming(origen, destino)
    #     costo_directo_total = []
    #     # Caso base
    #     if d == 0:
    #         return 0.0
        
    #     for i in range(len(origen)):            
    #         # Cálculo del costo
    #         gamma = 2 ** (-d)
    #         costo_directo = (abs(X_v[origen][i] - X_v[destino][i]))
    #         #print(f"Calculando costo directo entre {origen} y {destino}: {costo_directo} (gamma: {gamma})")
    #         # Calcular acumulado de vecinos intermedios
    #         acumulado = 0.0
    #         for vecino in self.vecinos_hamming_1(origen):
    #             if self.hamming(vecino, destino) < d:  # vecino está en camino óptimo
    #                 acumulado += self.calcular_costo_ruta_especifica(vecino, destino, X_v_tuple)
    #         costo_directo_total.append(gamma * (costo_directo + acumulado))
    #     return costo_directo_total
        


    def calcular_costo_simple(self, origen, destino, X_v):
        """
        Interfaz simple para calcular el costo entre dos estados específicos
        """
        # Convierte los valores de X_v a tuplas si son arrays
        #X_v_hashable = {k: tuple(v) if isinstance(v, np.ndarray) else v for k, v in X_v.items()}
        # Convertir X_v a tupla para permitir memoización
        X_v_tuple = tuple(sorted(X_v.items()))

        return self.calcular_costo_ruta_especifica(origen, destino, X_v_tuple)
