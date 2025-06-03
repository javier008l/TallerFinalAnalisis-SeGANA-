from src.middlewares.slogger import SafeLogger
from src.controllers.manager import Manager
from src.controllers.strategies.q_nodes import QNodes

import time

def iniciar():
    start_time = time.time() 
    """Punto de entrada principal"""
    estado_inicial = "100000000000000"
    condiciones =    "111111111111111"
    """Punto de entrada principal"""
    pruebas = [
        ('111111111111111', '111111111111111'),
        ('111111111111111', '111111111111110'),
        ('111111111111111', '011111111111111'),
        ('111111111111111', '011111111111110'),
        ('111111111111111', '101010101010101'),
    ]
    
    logger = SafeLogger("PruebasQ15reiniciandomemoYCero")

    gestor_sistema = Manager(estado_inicial)
    analizador_fb = QNodes(gestor_sistema)

    # Realizar todas las pruebas
    for i, (alcance, mecanismo) in enumerate(pruebas, 1):
        print(f"Prueba {i}: Alcance = {alcance}, Mecanismo = {mecanismo}")
        sia_uno = analizador_fb.aplicar_estrategia(condiciones, alcance, mecanismo)
        logger.critic(f"Resultado: {sia_uno}\n")
    
    end_time = time.time() 
    print(f"\nTiempo total de ejecución: {end_time - start_time:.2f} segundos")
