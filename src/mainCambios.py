import pandas as pd
from openpyxl import load_workbook
import os
from src.controllers.manager import Manager
from src.controllers.strategies.q_nodes import QNodes
from src.controllers.strategies.phi import Phi
from src.controllers.strategies.geometric import Geometric

def iniciar():
    """Punto de entrada principal"""

    estado_inicio = "100000"
    condiciones =    "111111"
    # Lista de pruebas a realizar
    casos = [
    ('011111', '111111'),  # BCDEFt+1 | ABCDEFt
    ('101101', '111111'),  # ACDFt+1 | ABCDEFt
    ('101110', '111111'),  # ACDEt+1 | ABCDEFt
    ('101111', '011111'),  # ACDEFt+1 | BCDEFt
    ('101111', '101111'),  # ACDEFt+1 | ACDEFt
    ('101111', '110111'),  # ACDEFt+1 | ABDEFt
    ('101111', '111011'),  # ACDEFt+1 | ABCEFt
    ('101111', '111101'),  # ACDEFt+1 | ABCDFt

    # ('101111', '111110'),  # ACDEFt+1 | ABCDEt
    # ('110011', '111111'),  # ABEFt+1 | ABCDEFt
    # ('110111', '111111'),  # ABDFt+1 | ABCDEFt
    # ('110110', '111111'),  # ABDEt+1 | ABCDEFt
    # ('110111', '011111'),  # ABDEFt+1 | BCDEFt
    # ('110111', '101111'),  # ABDEFt+1 | ACDEFt
    # ('110111', '110111'),  # ABDEFt+1 | ABDEFt
    # ('110111', '111011'),  # ABDEFt+1 | ABCEFt
    
    # ('110111', '111101'),  # ABDEFt+1 | ABCDFt
    # ('110111', '111110'),  # ABDEFt+1 | ABCDEt
    # ('111011', '111111'),  # ABCFt+1 | ABCDEFt
    # ('111010', '111111'),  # ABCEt+1 | ABCDEFt
    # ('111011', '011111'),  # ABCEFt+1 | BCDEFt
    # ('111011', '101111'),  # ABCEFt+1 | ACDEFt
    # ('101011', '111111'),  # ACEFt+1 | ABCDEFt 
    # ('101110', '111111'),  # ADEFt+1 | ABCDEFt
    
    # ('111011', '111011'),  # ABCEFt+1 | ABCEFt
    # ('111110', '111101'),  # BCDEFt+1 | ABCDFt
    # ('111111', '011111'),  # ABCDEFt+1 | BCDEFt
    # ('111111', '101111'),  # ABCDEFt+1 | ACDEFt
    # ('111111', '101111'),  # ABCDEFt+1 | ACDEFt
    # ('110111', '111111'),  # ABDEFt+1 | ABCDEFt
    # ('111111', '111011'),  # ABCDEFt+1 | ABCEFt
    # ('111110', '111111'),  # ABCDEt+1 | ABCDEFt

    # ('111101', '111111'),  # ABCDFt+1 | ABCDEFt
    # ('101111', '111111'),  # ACDEFt+1 | ABCDEFt
    # ('110111', '111111'),  # ABDEFt+1 | ABCDEFt
    # ('111011', '111111'),  # ABCEFt | ABCDEFt+1
    # ('111110', '111111'),  # ABCDEt+1 | ABCDEFt
    # ('011111', '111111'),  # BCDEFt+1 | ABCDEFt
    # ('101111', '111111'),  # ACDEFt+1 | ABCDEFt
    # ('101111', '111111'),  # ACDEFt+1 | ABCDEFt

    # ('110111', '111111'),  # ABDEFt+1 | ABCDEFt
    # ('111011', '111111'),  # ABCEFt+1 | ABCDEFt
    # ('111110', '111111'),  # ABCDEt+1 | ABCDEFt
    # ('111111', '111101'),  # ABCDEFt+1 | ABCDFt
    # ('111111', '101111'),  # ABCDEFt+1 | ACDEFt
    # ('111111', '110111'),  # ABCDEFt+1 | ABDEFt
    # ('111111', '111011'),  # ABCDEFt+1 | ABCEFt
    # ('111111', '111110'),  # ABCDEFt+1 | ABCDEt

#     ('111111', '011111'),  # ABCDEFt+1 | BCDEFt
#     ('111111', '111111'),  # ABCDEFt+1 | ABCDEFt
]
    
    # estado_inicio = "1000000000"  # Nunca cambia
    # condiciones = "1111111111"    # Nunca cambia
    
    # casos = [
        # ("1111111111", "1111111111"),
        # ("1111111111", "1111111110"),
        # ("1111111111", "0111111111"),
        # ("1111111111", "0111111110"),
        # ("1111111111", "1010101010"),
        # ("1111111111", "0101010101"),
        # ("1111111111", "1101101101"),
        # ("1111111110", "1111111111"),

        # ("1111111110", "1111111110"),
        # ("1111111110", "0111111111"),
        # ("1111111110", "0111111110"),
        # ("1111111110", "1010101010"),
        # ("1111111110", "0101010101"),
        # ("1111111110", "1101101101"),
        # ("0111111111", "1111111111"),
        # ("0111111111", "1111111110"),

        # ("0111111111", "0111111111"),
        # ("0111111111", "0111111110"),
        # ("0111111111", "1010101010"),
        # ("0111111111", "0101010101"),
        # ("0111111111", "1101101101"),
        # ("0111111110", "1111111111"),
        # ("0111111110", "1111111110"),
        # ("0111111110", "0111111111"),

        # ("0111111110", "0111111110"),
        # ("0111111110", "1010101010"),
        # ("0111111110", "0101010101"),
        # ("0111111110", "1101101101"),
        # ("1010101010", "1111111111"),
        # ("1010101010", "1111111110"),
        # ("1010101010", "0111111111"),
        # ("1010101010", "0111111110"),

        # ("1010101010", "1010101010"),
        # ("1010101010", "0101010101"),
        # ("1010101010", "1101101101"),
        # ("0101010101", "1111111111"),
        # ("0101010101", "1111111110"),
        # ("0101010101", "0111111111"),
        # ("0101010101", "0111111110"),
        # ("0101010101", "1010101010"),

        # ("0101010101", "0101010101"),
        # ("0101010101", "1101101101"),
        # ("1101101101", "1111111111"),
        # ("1101101101", "1111111110"),
        # ("1101101101", "0111111111"),
        # ("1101101101", "0111111110"),
        # ("1101101101", "1010101010"),
        # ("1101101101", "0101010101"),

    #     ("1101101101", "1101101101"),
    #     ("0111111001", "0111111111"),  
    # ]
    
    # estado_inicio = "100000000000000"  # Nunca cambia
    # condiciones   = "111111111111111"    # Nunca cambia
    
    # casos = [
        # ("111111111111111", "111111111111111"),
        # ("111111111111111", "111111111111110"),
        # ("111111111111111", "011111111111111"),
        # ("111111111111111", "011111111111110"),
        # ("111111111111111", "101010101010101"),
        # ("111111111111111", "010101010101010"),
        # ("111111111111111", "110110110110110"),
        
        # ("111111111111110", "111111111111111"),
        # ("111111111111110", "111111111111110"),
        # ("111111111111110", "011111111111111"),
        # ("111111111111110", "011111111111110"),
        # ("111111111111110", "101010101010101"),
        # ("111111111111110", "010101010101010"),
        # ("111111111111110", "110110110110110"),
        
        # ("011111111111111", "111111111111111"),
        # ("011111111111111", "111111111111110"),
        # ("011111111111111", "011111111111111"),
        # ("011111111111111", "011111111111110"),
        # ("011111111111111", "101010101010101"),
        # ("011111111111111", "010101010101010"),
        # ("011111111111111", "110110110110110"),
        
        # ("011111111111110", "111111111111111"),
        # ("011111111111110", "111111111111110"),
        # ("011111111111110", "011111111111111"),
        # ("011111111111110", "011111111111110"),
        # ("011111111111110", "101010101010101"),
        # ("011111111111110", "010101010101010"),
        # ("011111111111110", "110110110110110"),
        
        # ("101010101010101", "111111111111111"),
        # ("101010101010101", "111111111111110"),
        # ("101010101010101", "011111111111111"),
        # ("101010101010101", "011111111111110"),
        # ("101010101010101", "101010101010101"),
        # ("101010101010101", "010101010101010"),
        # ("101010101010101", "110110110110110"),
        
        # ("010101010101010", "111111111111111"),
        # ("010101010101010", "111111111111110"),
        # ("010101010101010", "011111111111111"),
        # ("010101010101010", "011111111111110"),
        # ("010101010101010", "101010101010101"),
        # ("010101010101010", "010101010101010"),
        # ("010101010101010", "110110110110110"),
        
        # ("110110110110110", "111111111111111"),
        # ("110110110110110", "111111111111110"),
        # ("110110110110110", "011111111111111"),
        # ("110110110110110", "011111111111110"),
        # ("110110110110110", "101010101010101"),
        # ("110110110110110", "010101010101010"),
        # ("110110110110110", "110110110110110"),
        
    #     ("011111100111111", "011111111111111"),
    # ]
    
    # estado_inicio = "10000000000000000000"  # Nunca cambia
    # condiciones   = "11111111111111111111"    # Nunca cambia
    
    # casos = [
    # ("11111111111111111111", "11111111111111111111"),
    # ("11111111111111111111", "11111111111111111110"),
    # ("11111111111111111111", "01111111111111111111"),
    # ("11111111111111111111", "01111111111111111110"),
    # ("11111111111111111111", "10101010101010101010"),

#     ("11111111111111111111", "01010101010101010101"),
#     ("11111111111111111111", "11011011011011011011"),
#     ("11111111111111111110", "11111111111111111111"),
#     ("11111111111111111110", "11111111111111111110"),
#     ("11111111111111111110", "01111111111111111111"),

#     ("11111111111111111110", "01111111111111111110"),
#     ("11111111111111111110", "10101010101010101010"),
#     ("11111111111111111110", "01010101010101010101"),
#     ("11111111111111111110", "11011011011011011011"),
#     ("01111111111111111111", "11111111111111111111"),

#     ("01111111111111111111", "11111111111111111110"),
#     ("01111111111111111111", "01111111111111111111"),
#     ("01111111111111111111", "01111111111111111110"),
#     ("01111111111111111111", "10101010101010101010"),
#     ("01111111111111111111", "01010101010101010101"),

#     ("01111111111111111111", "11011011011011011011"),
#     ("01111111111111111110", "11111111111111111111"),
#     ("01111111111111111110", "11111111111111111110"),
#     ("01111111111111111110", "01111111111111111111"),
#     ("01111111111111111110", "01111111111111111110"),

#     ("01111111111111111110", "10101010101010101010"),
#     ("01111111111111111110", "01010101010101010101"),
#     ("01111111111111111110", "11011011011011011011"),
#     ("10101010101010101010", "11111111111111111111"),
#     ("10101010101010101010", "11111111111111111110"),

#     ("10101010101010101010", "01111111111111111111"),
#     ("10101010101010101010", "01111111111111111110"),
#     ("10101010101010101010", "10101010101010101010"),
#     ("10101010101010101010", "01010101010101010101"),
#     ("10101010101010101010", "11011011011011011011"),

#     ("01010101010101010101", "11111111111111111111"),
#     ("01010101010101010101", "11111111111111111110"),
#     ("01010101010101010101", "01111111111111111111"),
#     ("01010101010101010101", "01111111111111111110"),
#     ("01010101010101010101", "10101010101010101010"),

#     ("01010101010101010101", "01010101010101010101"),
#     ("01010101010101010101", "11011011011011011011"),
#     ("11011011011011011011", "11111111111111111111"),
#     ("11011011011011011011", "11111111111111111110"),
#     ("11011011011011011011", "01111111111111111111"),

#     ("11011011011011011011", "01111111111111111110"),
#     ("11011011011011011011", "10101010101010101010"),
#     ("11011011011011011011", "01010101010101010101"),
#     ("11011011011011011011", "11011011011011011011"),
#     ("10111111111111111111", "10111111111111111111")        
# ]
    
    # Lista para almacenar los resultados
    #resultados = []
    
    # Creacion de la red TPM
    #Manager.generar_red( self=Manager, dimensiones=20, datos_discretos=True)
    
    # Nombre del archivo de resultados
    archivo_excel = "PruebasTallerFinal.xlsx"

    # Si el archivo no existe, crearlo con las columnas
    if not os.path.exists(archivo_excel):
        df_vacio = pd.DataFrame(columns=["Alcance", "Mecanismo", "Estrategia", "Perdida",
                                        "Tiempo Total", "Particion"])
        df_vacio.to_excel(archivo_excel, index=False)
        
    # Iterar sobre cada caso
    for alcance, mecanismo in casos:
        config_sistema = Manager(estado_inicial=estado_inicio)
        
        ## Ejemplo de solución mediante módulo de fuerza bruta (QNodes) ###
        analizador_fb = Geometric(config_sistema)
        resultado = analizador_fb.aplicar_estrategia(condiciones, alcance, mecanismo)
        
        # Extraer los datos de la solución
        nueva_fila = {
            "Alcance": alcance,
            "Mecanismo": mecanismo,
            "Estrategia": resultado.estrategia,
            "Perdida": resultado.perdida,
            "Tiempo Total": resultado.tiempo_ejecucion,
            "Particion": resultado.particion
        }
        hoja_destino = "6 nodos"
        
        try:
            # Intenta leer el archivo Excel existente
            df_existente = pd.read_excel(archivo_excel, engine="openpyxl", sheet_name=hoja_destino)
        except FileNotFoundError:
            # Si el archivo no existe, crea un DataFrame con la nueva fila
            df_existente = pd.DataFrame() # se inicializa un DataFrame vacio en caso de no existir el archivo.

        # Concatena el DataFrame existente con la nueva fila
        df_nuevo = pd.concat([df_existente, pd.DataFrame([nueva_fila])], ignore_index=True)

        # Guarda el DataFrame actualizado en el archivo Excel, sobrescribiendo la hoja existente
        with pd.ExcelWriter(archivo_excel, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer: # se cambia el mode a 'w'
            df_nuevo.to_excel(writer, sheet_name=hoja_destino, index=False)

        print(f"Resultado agregado para {alcance} - {mecanismo}")

    print(f"Resultados guardados en '{archivo_excel}' correctamente.")



    # ## Ejemplo de solución mediante Pyphi ###
    # analizador_fi = Phi(config_sistema)
    # sia_dos = analizador_fi.aplicar_estrategia(condiciones, alcance, mecanismo)
    # print(sia_dos)
    
    # # Iterar sobre cada caso
    # for alcance, mecanismo in casos:
    #     config_sistema = Manager(estado_inicial=estado_inicio)
        
    #     ## Ejemplo de solución mediante módulo de fuerza bruta (Phi) ###
    #     analizador_fi = Phi(config_sistema)
    #     resultado = analizador_fi.aplicar_estrategia(condiciones, alcance, mecanismo)
        
    #     # Extraer los datos de la solución
    #     nueva_fila = {
    #         "Alcance": alcance,
    #         "Mecanismo": mecanismo,
    #         "Estrategia": resultado.estrategia,
    #         "Perdida": resultado.perdida,
    #         "Distribucion Subsistema": resultado.distribucion_subsistema,
    #         "Distribucion Particion": resultado.distribucion_particion,
    #         "Tiempo Total": resultado.tiempo_ejecucion,
    #         "Particion": resultado.particion
    #     }

    #     # Cargar el archivo existente sin sobrescribir
    #     with pd.ExcelWriter(archivo_excel, engine="openpyxl", mode="a", if_sheet_exists="overlay") as writer:
    #         df_existente = pd.read_excel(archivo_excel)
    #         df_nuevo = pd.concat([df_existente, pd.DataFrame([nueva_fila])], ignore_index=True)
    #         df_nuevo.to_excel(writer, index=False)

    #     print(f"Resultado agregado para {alcance} - {mecanismo}")

    # print(f"Resultados guardados en '{archivo_excel}' correctamente.")
