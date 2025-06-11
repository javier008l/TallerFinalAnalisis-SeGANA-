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
import pandas as pd
from itertools import permutations


class GeometricSIA(SIA):
    def __init__(self, gestor):
        super().__init__(gestor)
        self.tpm = self.sia_cargar_tpm()
        self.n_states, self.n_vars = self.tpm.shape
        self.cache: dict[Tuple[int, int, int], float] = {}  # (i, j, var_idx) => costo

    def aplicar_estrategia(self, condicion: str, alcance: str, mecanismo: str):
        """
        Implementación de la estrategia geométrica.
        """
        self.sia_preparar_subsistema(condicion, alcance, mecanismo)

        tabla = self.construir_tabla_costos_variable(mecanismo)  # Ejemplo para variable 0


        caminos = self.caminos_hamming_1(mecanismo, alcance)
        print(f"Caminos Hamming 1 desde {mecanismo} a {alcance}: {caminos}")

        # print(f"Subsistema preparado: {subsistema}")
        # #print(f"tpm shape: {self.tpm.shape}")
        # costo = self.costo(0, 1, 0)  # Llamada de prueba para inicializar el cache
        # print(f"Costo entre estados 0 y 1 para la variable 0: {costo}")
        # vecinos = self.vecinos_en_ruta(000, 101)
        # print(f"Vecinos en ruta de 000 a 101: {vecinos}")
        # tabla_costos = self.construir_tabla_costos_variable(2)
        # print(f"Tabla de costos para la variable 001:\n{tabla_costos}")
    """
        # self.tensors = self.descomponer_en_tensores(subsistema)
        # print(f"ya hice la descomposición en tensores")
        # self.calcular_tabla_transiciones(subsistema)
        # print(f"ya hice la tabla de transiciones")
        # candidatos = self.identificar_biparticiones_candidatas()
        # print(f"Candidatos identificados: {candidatos}")
        # alcance, mecanismo, mejor_phi, mejor_distribucion = self.evaluar_biparticiones(candidatos, subsistema)
        
        # tpm = self.sia_cargar_tpm()
        
        # origen =  mecanismo
        # destino = alcance
        # # Construir X_v desde la TPM
        # X_v = self.construir_X_v_desde_tpm(tpm)
        # print(f"X_v construido: {X_v}")
        # #costo_total = self.calcular_costo_simple(origen, destino, X_v)
        
        # # Formatear partición como en QNodes
        # fmt_mip = fmt_biparte_q(
        #     list((EFECTO, i) for i in alcance),
        #     list((ACTUAL, i) for i in mecanismo)
        # )

        # return Solution(
        #     estrategia=GEOMETRIC_LABEL,
        #     perdida=mejor_phi,
        #     distribucion_subsistema=subsistema.distribucion_marginal(),
        #     distribucion_particion=mejor_distribucion,
        #     tiempo_total=time.time() - self.sia_tiempo_inicio,
        #     particion=fmt_mip
        # )

    """
    
    
    def hamming(self, a, b):
        if isinstance(a, int):
            a_bin = format(a, f'0{self.n_vars}b')
        else:
            # Si es string, asegurar la longitud correcta
            a_bin = a.zfill(self.n_vars)
            
        if isinstance(b, int):
            b_bin = format(b, f'0{self.n_vars}b')
        else:
            # Si es string, asegurar la longitud correcta
            b_bin = b.zfill(self.n_vars)
        # Asegurar que ambas cadenas tengan la misma longitud
        if len(a_bin) != len(b_bin):
            raise ValueError(f"Las cadenas deben tener la misma longitud: {a_bin} vs {b_bin}")
            
        return sum(x != y for x, y in zip(a_bin, b_bin))

    def vecinos_hamming_1(self, estado):
        """Genera todos los vecinos a distancia Hamming 1"""
        vecinos = []
        for i in range(len(estado)):
            nuevo = estado[:i] + ('0' if estado[i] == '1' else '1') + estado[i+1:]
            vecinos.append(nuevo)
        return vecinos
    
    

    def caminos_hamming_1(self, origen: str, destino: str) -> list[list[str]]:
        """
        Genera todos los caminos binarios de Hamming 1 desde `origen` hasta `destino`.
        """
        n = len(origen)
        indices_diferentes = [i for i in range(n) if origen[i] != destino[i]]

        caminos = []
        for orden in permutations(indices_diferentes):
            actual = list(origen)
            camino = ["".join(actual)]

            for idx in orden:
                actual[idx] = destino[idx]
                camino.append("".join(actual))

            caminos.append(camino)

        return caminos

    
    def calcular_costo_ruta_especifica(self, origen, destino, x_v_tuple):
        """
        Calcula el costo de una ruta específica usando memoización.
        X_v_tuple debe ser una tupla de (estado, valor) para permitir hashing.
        """
        # Convertir tupla de vuelta a diccionario para cálculos
        x_v = np.array(x_v_tuple)
        x_v_hashable = tuple(x_v_tuple) 
        # Convertir origen y destino a enteros si son strings
        origen_idx = int(origen, 2) if isinstance(origen, str) else origen
        destino_idx = int(destino, 2) if isinstance(destino, str) else destino
        d = self.hamming(origen, destino)
       # print(f"Calculando costo de ruta de {origen} a {destino} con distancia Hamming {d}")

        clave = (origen, destino, x_v_hashable)
        if clave in self.cache:
            #print(f"Cache hit for {self.cache[clave]}")
            return self.cache[clave]

        # Caso base
        if d == 0:
            self.cache[clave] = 0.0
            return 0.0

        # Cálculo del costo
        gamma = 2 ** (-d)
        
        costo_directo = abs(x_v[origen_idx] - x_v[destino_idx])

        # Calcular acumulado de vecinos intermedios
        acumulado = 0.0
        for vecinos in self.vecinos_hamming_1(origen):
            vecino = vecinos  # El último vecino es el destino
            if self.hamming(vecino, destino) <= d:  # vecino está en camino óptimo
                acumulado += self.calcular_costo_ruta_especifica(vecino, destino, x_v_tuple)

        return gamma * (costo_directo + acumulado)
    
    def construir_tabla_costos_variable(self, origen) -> pd.DataFrame:
        """Construye la tabla de costos para una variable específica en formato de DataFrame."""
        print(f"n_states: {self.n_states}, n_vars: {self.n_vars}")
    
        data = []
        
        for k in range(len(origen)):
            for _ in range(1):
                for j in range(self.n_states):
                    j_bin = format(j, f'0{self.n_vars}b')
                    hamming_distance = self.hamming(origen, j_bin)

                    if hamming_distance >= 1:
                        costo = self.calcular_costo_ruta_especifica(origen, j_bin, x_v_tuple=self.tpm[:, k])
                        transicion = f"t({origen},{j_bin})"
                        data.append([transicion, costo])
                df = pd.DataFrame(data, columns=["Transición", "Variable "])
        
        return df
    

