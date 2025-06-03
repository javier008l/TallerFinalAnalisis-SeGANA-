from src.models.base.sia import SIA
from src.models.core.solution import Solution
from src.constants.models import GEOMETRIC_LABEL
import numpy as np
import time
from src.funcs.base import ABECEDARY
from functools import lru_cache

class Geometric(SIA):
    def __init__(self, gestor):
        super().__init__(gestor)
        self.tabla_costos = {}  # Dict[str][int][int]
    
    def aplicar_estrategia(self, condicion, alcance, mecanismo):
        self.sia_preparar_subsistema(condicion, alcance, mecanismo)

        self.tensores = self._descomponer_en_tensores()
        self._calcular_tabla_costos()

        self.mostrar_tabla_costos()

        candidatos = self._identificar_candidatos()
        (parte1, parte2), costo = self._evaluar_candidatos(candidatos)

        return Solution(
            estrategia=GEOMETRIC_LABEL,
            perdida=costo,
            distribucion_subsistema=self.sia_dists_marginales,
            distribucion_particion={},  # Opcional, si quieres incluirla
            tiempo_total=time.time() - self.sia_tiempo_inicio,
            particion={"parte_1": list(parte1), "parte_2": list(parte2)}
        )

    def _calcular_tabla_costos(self):
        n = len(next(iter(self.tensores.values())))  # Número de estados: 2^n
        estados = list(range(n))  # Los estados van de 0 a 2^n - 1

        for var, tensor in self.tensores.items():
            self.tabla_costos[var] = np.zeros((n, n), dtype=np.float32)

            for i in estados:
                for j in estados:
                    self.tabla_costos[var][i][j] = self._calcular_transicion_costo(i, j, tensor)
    def mostrar_tabla_costos(self):
        print("\n📊 TABLA DE COSTOS POR VARIABLE")
        for var, matriz in self.tabla_costos.items():
            print(f"\n🔹 Variable: {var}")
            filas = matriz.shape[0]
            header = "     " + " ".join([f"{j:>6}" for j in range(filas)])
            print(header)
            for i in range(filas):
                fila_str = " ".join([f"{matriz[i][j]:6.3f}" for j in range(filas)])
                print(f"{i:>3}: {fila_str}")



    def _descomponer_en_tensores(self) -> dict[str, np.ndarray]:
        """
        Extrae una representación tipo-TPM desde los NCubes del subsistema.

        Devuelve:
            dict[str, np.ndarray]: Diccionario {nombre_variable: vector plano de 2^n valores}
        """
        dims = self.sia_subsistema.dims_ncubos  # e.g., [0,1,2]
        n = len(dims)
        longitud_esperada = 2 ** n

        tensores = {}

        for cube in self.sia_subsistema.ncubos:
            var_name = ABECEDARY[cube.indice]

            if cube.dims.size != n:
                raise ValueError(
                    f"El NCube {cube.indice} no tiene todas las dimensiones del subsistema. "
                    f"Esperado {n}, tiene {cube.dims.size}. No se puede vectorizar correctamente."
                )

            tensor = cube.data.flatten()

            if tensor.size != longitud_esperada:
                raise ValueError(
                    f"Tensor de {var_name} no tiene tamaño esperado: {tensor.size} vs {longitud_esperada}"
                )

            tensores[var_name] = tensor.copy()

        return tensores


    def _calcular_transicion_costo(self, i, j, X, nivel=0, max_nivel=5):
        if i == j:
            return 0.0

        dh = self._distancia_hamming(i, j)
        gamma = 2 ** (-dh)
        costo = abs(X[i] - X[j])

        if dh > 1 and nivel < max_nivel:
            visitados = set([i])
            cola = [i]
            while cola:
                siguiente = []
                for u in cola:
                    vecinos = self._vecinos_hamming(u, j)
                    for v in vecinos:
                        if v not in visitados:
                            costo += self._calcular_transicion_costo(i, v, X, nivel + 1, max_nivel)
                            visitados.add(v)
                            siguiente.append(v)
                cola = siguiente

        return gamma * costo


    def _distancia_hamming(self, a: int, b: int) -> int:
        return bin(a ^ b).count('1')

    def _vecinos_hamming(self, estado: int, destino: int) -> list:
        """Vecinos de 'estado' que se acerquen a 'destino'"""
        vecinos = []
        n_bits = int(np.log2(len(next(iter(self.tensores.values())))))
        for i in range(n_bits):
            candidato = estado ^ (1 << i)  # Cambia un bit
            if self._distancia_hamming(candidato, destino) < self._distancia_hamming(estado, destino):
                vecinos.append(candidato)
        return vecinos
    
    def _identificar_candidatos(self):
        """
        Genera biparticiones candidatas basadas en los patrones de costo mínimo en la tabla T.
        Devuelve una lista de pares (set_1, set_2) donde set_1 ∪ set_2 = variables.
        """
        variables = list(self.tensores.keys())
        candidatos = []

        # Agrupamos variables que muestran patrones similares de bajo costo
        for i in range(len(variables)):
            for j in range(i + 1, len(variables)):
                var_i, var_j = variables[i], variables[j]
                tabla_i = self.tabla_costos[var_i]
                tabla_j = self.tabla_costos[var_j]

                # Heurística simple: sumamos los costos entre mismos estados
                costo_total = np.sum(np.abs(tabla_i - tabla_j))

                if costo_total < 1e-2:  # Umbral de similitud
                    candidatos.append(({var_i, var_j}, set(variables) - {var_i, var_j}))

        # Si no se identifican candidatos claros, devolvemos biparticiones simples
        if not candidatos:
            for i in range(1, len(variables)):
                candidatos.append((set(variables[:i]), set(variables[i:])))
        
        return candidatos

    def _evaluar_candidatos(self, candidatos):
        mejor_biparticion = None
        menor_costo = float("inf")
        n = self.sia_subsistema.dims_ncubos.size
        estados = list(range(2 ** n))  # Todos los estados posibles

        for parte1, parte2 in candidatos:
            costo_total = 0.0

            for var in parte1:
                T = self.tabla_costos[var]
                for i in estados:
                    for j in estados:
                        if i == j:
                            continue
                        # Solo penalizar si el cambio involucra una variable fuera del grupo
                        if self._cambio_cruzado(i, j, parte1, parte2):
                            costo_total += T[i][j]

            if costo_total < menor_costo:
                menor_costo = costo_total
                mejor_biparticion = (parte1, parte2)

        return mejor_biparticion, menor_costo
