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
        
        
        print(f"ya hice el Subsistema preparado")
        print(f"ya hice la tabla de transiciones")
        candidatos = self.identificar_biparticiones_candidatas(3)
        print(f"Candidatos identificados: {candidatos}")
        alcance, mecanismo, mejor_phi, mejor_distribucion = self.evaluar_biparticiones(candidatos, subsistema)
        
        tpm = self.sia_cargar_tpm()

        origen =  mecanismo_mask
        destino = alcance_mask
        print(f"Origen: {origen}, Destino: {destino}")
        
        self.X_v = self.construir_X_v_desde_tpm(tpm)  # o con estados relevantes si quieres



        # Construir X_v desde la TPM

        print(f"X_v construido: {self.X_v_global}")
        pares = self.generar_pares_hamming_1(len(origen))  # -1 porque el último bit es el de paridad
       
        for a, b in pares:
            self.TablaCostos[(a, b)] = self.calcular_costo_simple(a, b, self.X_v)


        print(self.TablaCostos)
        
        
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
        """
        Construye un diccionario X_v solo para los estados relevantes.
        Si estados_relevantes es None, construye para todos los estados.
        """
        n = int(np.log2(tpm.shape[0]))
        X_v = {}
        if estados_relevantes is None:
            for j in range(tpm.shape[0]):
                estado_bin = format(j, f'0{n}b')
                X_v[estado_bin] = tpm[j]
        else:
            for estado_bin in estados_relevantes:
                j = int(estado_bin, 2)
                X_v[estado_bin] = tpm[j]
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
        Calcula el costo de una ruta específica usando memoización.
        X_v_tuple debe ser una tupla de (estado, valor) para permitir hashing.
        """
        # Convertir tupla de vuelta a diccionario para cálculos
        X_v = dict(X_v_tuple)
        #print(f"Calculando costo entre {origen} y {destino} con X_v: {X_v}")
        d = self.hamming(origen, destino)
        costo_directo_total = []
        # Caso base
        if d == 0:
            return 0.0
        
        for i in range(len(origen)):            
            # Cálculo del costo
            gamma = 2 ** (-d)
            costo_directo = (abs(X_v[origen][i] - X_v[destino][i]))
            #print(f"Calculando costo directo entre {origen} y {destino}: {costo_directo} (gamma: {gamma})")
            # Calcular acumulado de vecinos intermedios
            acumulado = 0.0
            for vecino in self.vecinos_hamming_1(origen):
                if self.hamming(vecino, destino) < d:  # vecino está en camino óptimo
                    acumulado += self.calcular_costo_ruta_especifica(vecino, destino, X_v_tuple)
            costo_directo_total.append(gamma * (costo_directo + acumulado))
        return costo_directo_total
        


    def calcular_costo_simple(self, origen, destino, X_v):
        """
        Interfaz simple para calcular el costo entre dos estados específicos
        """
        # Convierte los valores de X_v a tuplas si son arrays
        X_v_hashable = {k: tuple(v) if isinstance(v, np.ndarray) else v for k, v in X_v.items()}
        # Convertir X_v a tupla para permitir memoización
        X_v_tuple = tuple(sorted(X_v_hashable.items()))

        return self.calcular_costo_ruta_especifica(origen, destino, X_v_tuple)
