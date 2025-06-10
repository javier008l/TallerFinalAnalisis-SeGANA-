import time
import random
from typing import Union
import numpy as np
from src.middlewares.slogger import SafeLogger
from src.funcs.base import emd_efecto, ABECEDARY
from src.middlewares.profile import profiler_manager, profile
from src.funcs.format import fmt_biparte_q
from src.controllers.manager import Manager
from src.models.base.sia import SIA
from src.models.core.solution import Solution

from src.constants.models import (
    FASTSPLIT_ANALYSIS_TAG,  # Crea esta constante si no existe
    FASTSPLIT_LABEL,         # Ej: "FastSplit"
    FASTSPLIT_STRAREGY_TAG,  # Ej: "estrategia-fastsplit"
)
from src.constants.base import TYPE_TAG, NET_LABEL, EFECTO, ACTUAL

class FastSplit(SIA):
    """
    Estrategia basada en exploración aleatoria y mutación local para encontrar
    particiones con baja pérdida (EMD) de forma eficiente y rápida.
    """
    
    def __init__(self, gestor: Manager):
        super().__init__(gestor)
        profiler_manager.start_session(
            f"{NET_LABEL}{len(gestor.estado_inicial)}{gestor.pagina}"
        )
        self.logger = SafeLogger(FASTSPLIT_STRAREGY_TAG)

        self.indices_alcance: np.ndarray
        self.indices_mecanismo: np.ndarray
        self.tiempos: tuple[np.ndarray, np.ndarray]
        self.vertices: list[tuple[int, int]]

    @profile(context={TYPE_TAG: FASTSPLIT_ANALYSIS_TAG})
    def aplicar_estrategia(
        self,
        condicion: str,
        alcance: str,
        mecanismo: str,
        tiempo_limite: float = 2.0,
        muestras_iniciales: int = 10,
    ) -> Solution:
        self.sia_preparar_subsistema(condicion, alcance, mecanismo)

        # Obtener índices de presente (t) y futuro (t+1)
        self.indices_alcance = self.sia_subsistema.indices_ncubos
        self.indices_mecanismo = self.sia_subsistema.dims_ncubos

        self.tiempos = (
            np.zeros(self.indices_mecanismo.size, dtype=np.int8),
            np.zeros(self.indices_alcance.size, dtype=np.int8),
        )

        # Crear lista de vértices: (tiempo, índice)
        futuro = [(EFECTO, idx) for idx in self.indices_alcance]
        presente = [(ACTUAL, idx) for idx in self.indices_mecanismo]
        self.vertices = presente + futuro

        # Guardar distribución base del sistema original
        self.dist_referencia = self.sia_dists_marginales

        # Lanzar búsqueda rápida
        mejor_perdida, mejor_particion = self.buscar_buena_particion(
            tiempo_limite=tiempo_limite,
            muestras_iniciales=muestras_iniciales,
        )

        # Formatear resultado
        fmt_mip = fmt_biparte_q(list(mejor_particion), self.nodes_complement(mejor_particion))

        return Solution(
            estrategia=FASTSPLIT_LABEL,
            perdida=mejor_perdida,
            distribucion_subsistema=self.dist_referencia,
            distribucion_particion=None,  # Solo calculamos EMD
            tiempo_total=time.time() - self.sia_tiempo_inicio,
            particion=fmt_mip,
        )
    
    def buscar_buena_particion(
        self,
        tiempo_limite: float,
        muestras_iniciales: int = 10,
        max_mutaciones: int = 20,
    ) -> tuple[float, list[tuple[int, int]]]:

        mejor_perdida = float("inf")
        mejor_particion = None

        inicio = time.time()
        tiempo_actual = lambda: time.time() - inicio

        # 🔹 1. Evaluar particiones aleatorias iniciales
        for _ in range(muestras_iniciales):
            particion = self.generar_particion_aleatoria()
            perdida = self.evaluar_particion(particion)
            if perdida < mejor_perdida:
                mejor_perdida = perdida
                mejor_particion = particion
            if mejor_perdida == 0.0:
                break

        # 🔸 2. Mutaciones locales desde la mejor encontrada
        mutaciones_sin_mejora = 0
        while tiempo_actual() < tiempo_limite and mutaciones_sin_mejora < max_mutaciones:
            mutada = self.mutar_particion(mejor_particion)
            perdida_mutada = self.evaluar_particion(mutada)
            if perdida_mutada < mejor_perdida:
                mejor_perdida = perdida_mutada
                mejor_particion = mutada
                mutaciones_sin_mejora = 0
            else:
                mutaciones_sin_mejora += 1

        return mejor_perdida, mejor_particion
    
    def generar_particion_aleatoria(self) -> list[tuple[int, int]]:
        """
        Devuelve una partición aleatoria válida:
        - Ambos grupos no están vacíos en ambos tiempos simultáneamente
        """
        while True:
            k = random.randint(1, len(self.vertices) - 1)
            seleccion = set(random.sample(self.vertices, k))
            complemento = set(self.vertices) - seleccion

            if not seleccion or not complemento:
                continue  # Evita conjuntos totalmente vacíos

            # Chequeo de validez: cada grupo debe tener al menos un tiempo activo
            tiempos_sel = {t for (t, _) in seleccion}
            tiempos_comp = {t for (t, _) in complemento}

            if (0 in tiempos_sel or 1 in tiempos_sel) and (0 in tiempos_comp or 1 in tiempos_comp):
                return list(seleccion)

    def evaluar_particion(self, grupo: list[tuple[int, int]]) -> float:
        """
        Calcula la pérdida EMD de una partición representada por un subconjunto de vértices.
        """
        # Separar en t=0 (actual) y t=1 (efecto)
        t0 = [idx for (t, idx) in grupo if t == ACTUAL]
        t1 = [idx for (t, idx) in grupo if t == EFECTO]

        # Convertir a np.array (requerido por bipartir)
        arr_t0 = np.array(t0, dtype=np.int8)
        arr_t1 = np.array(t1, dtype=np.int8)

        # Bipartir subsistema según esta división
        particion = self.sia_subsistema.bipartir(arr_t1, arr_t0)

        # Calcular distribución marginal de la partición
        dist = particion.distribucion_marginal()

        # Calcular EMD con respecto al sistema original
        return emd_efecto(dist, self.dist_referencia)

    def mutar_particion(self, particion: list[tuple[int, int]]) -> list[tuple[int, int]]:
        """
        Crea una mutación válida de la partición dada:
        - Mueve un vértice entre grupos
        - Asegura que ningún grupo quede vacío en ambos tiempos simultáneamente
        """
        conjunto = set(particion)
        complemento = set(self.vertices) - conjunto

        intentos = 0
        while intentos < 10:
            if len(conjunto) > 1 and (len(complemento) == 0 or random.random() < 0.5):
                mover_de = conjunto
                mover_a = complemento
            else:
                mover_de = complemento
                mover_a = conjunto

            if not mover_de:
                return list(conjunto)

            nodo = random.choice(list(mover_de))
            mover_de.remove(nodo)
            mover_a.add(nodo)

            # Validar: cada grupo debe tener al menos un tiempo (t o t+1)
            tiempos_1 = {t for (t, _) in mover_a}
            tiempos_2 = {t for (t, _) in mover_de}
            if (0 in tiempos_1 or 1 in tiempos_1) and (0 in tiempos_2 or 1 in tiempos_2):
                return list(mover_a)

            # Deshacer si no fue válida
            mover_a.remove(nodo)
            mover_de.add(nodo)
            intentos += 1

        return list(conjunto)
        
    def nodes_complement(self, nodes: list[tuple[int, int]]) -> list[tuple[int, int]]:
        """
        Devuelve el complemento del conjunto de nodos respecto a todos los vértices.
        """
        return list(set(self.vertices) - set(nodes))

