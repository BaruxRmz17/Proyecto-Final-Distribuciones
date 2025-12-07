# ==============================================================================
# PROYECTO FINAL DE PROBABILIDAD: SIMULACIÓN DE DISTRIBUCIONES
# AUTOR: Baruc Azael Ramirez Romo (ID: 000039)
# OBJETIVO: Generar muestras, comparar estadísticos y visualizar PDF/Histogramas.
# ==============================================================================

# 1. IMPORTACIÓN DE LIBRERÍAS
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import expon, uniform

# Configuraciones para mejor visualización
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (14, 6)
N_MUESTRAS = 100000 # Gran tamaño para asegurar la convergencia

print("=====================================================================")
print("  INICIO DE LA SIMULACIÓN: VALIDACIÓN DE DISTRIBUCIONES TEÓRICAS")
print("=====================================================================")


# ==============================================================================
# A. DISTRIBUCIÓN EXPONENCIAL (PUNTOS 1, 2, 3)
# ==============================================================================
print("\n--- A. ANÁLISIS DE LA DISTRIBUCIÓN EXPONENCIAL ---")

# Parámetros Teóricos:
LAMBDA_EX = 0.5        # Tasa (lambda) = 0.5 eventos/unidad_tiempo
ESCALA_EX = 1 / LAMBDA_EX # Escala = Media Teórica = 2.0

# 1. Generación de muestras aleatorias
muestras_exp = expon.rvs(scale=ESCALA_EX, size=N_MUESTRAS)

# 3. Comparación de Estadísticos Muestrales vs. Teóricos
media_teorica_exp = ESCALA_EX
varianza_teorica_exp = ESCALA_EX**2

media_muestral_exp = np.mean(muestras_exp)
varianza_muestral_exp = np.var(muestras_exp)

print(f"\n[Estadísticos Exponenciales]")
print(f"  > Media Teórica:   {media_teorica_exp:.4f}")
print(f"  > Media Muestral:  {media_muestral_exp:.4f}")
print(f"  > Varianza Teórica: {varianza_teorica_exp:.4f}")
print(f"  > Varianza Muestral: {varianza_muestral_exp:.4f}")

# 2. Visualización: Histograma vs. Función de Densidad Teórica (PDF)
plt.figure(figsize=(14, 6))
plt.hist(muestras_exp, bins=50, density=True, alpha=0.7, color='skyblue', label='Frecuencia Muestral')

x_exp = np.linspace(0, muestras_exp.max() * 0.9, 100)
pdf_exp = expon.pdf(x_exp, scale=ESCALA_EX)
plt.plot(x_exp, pdf_exp, 'r--', linewidth=2.5, label='PDF Teórica ($f(x; \lambda)$)')

plt.title(f'Distribución Exponencial ($\lambda={LAMBDA_EX}$): Histograma vs. PDF Teórica')
plt.xlabel('Tiempo de Espera ($x$)')
plt.ylabel('Densidad de Probabilidad')
plt.legend()
plt.show()


# ==============================================================================
# B. DISTRIBUCIÓN UNIFORME CONTINUA (PUNTOS 1, 2, 3)
# ==============================================================================
print("\n--- B. ANÁLISIS DE LA DISTRIBUCIÓN UNIFORME CONTINUA ---")

# Parámetros Teóricos:
A = 2              # Límite Inferior (a)
B = 8              # Límite Superior (b)
LOC_UN = A         # Ubicación (start)
SCALE_UN = B - A   # Escala (amplitud b-a)

# 1. Generación de muestras aleatorias
muestras_unif = uniform.rvs(loc=LOC_UN, scale=SCALE_UN, size=N_MUESTRAS)

# 3. Comparación de Estadísticos Muestrales vs. Teóricos
media_teorica_unif = (A + B) / 2
varianza_teorica_unif = (B - A)**2 / 12

media_muestral_unif = np.mean(muestras_unif)
varianza_muestral_unif = np.var(muestras_unif)

print(f"\n[Estadísticos Uniformes]")
print(f"  > Media Teórica:   {media_teorica_unif:.4f}")
print(f"  > Media Muestral:  {media_muestral_unif:.4f}")
print(f"  > Varianza Teórica: {varianza_teorica_unif:.4f}")
print(f"  > Varianza Muestral: {varianza_muestral_unif:.4f}")

# 2. Visualización: Histograma vs. Función de Densidad Teórica (PDF)
plt.figure(figsize=(14, 6))
plt.hist(muestras_unif, bins=15, density=True, alpha=0.7, color='lightgreen', label='Frecuencia Muestral')

x_unif = np.linspace(A - 1, B + 1, 100)
pdf_unif = uniform.pdf(x_unif, loc=LOC_UN, scale=SCALE_UN)
plt.plot(x_unif, pdf_unif, 'b-', linewidth=2.5, label='PDF Teórica ($f(x; a, b)$)')

plt.title(f'Distribución Uniforme: Histograma vs. PDF Teórica ($a={A}, b={B}$)')
plt.xlabel('Valor ($x$)')
plt.ylabel('Densidad de Probabilidad')
plt.xlim(A - 0.5, B + 0.5)
plt.legend()
plt.show()


# ==============================================================================
# C. SIMULACIÓN APLICADA (PUNTO 4): TIEMPO EN SEGMENTO DE MTB
# ==============================================================================
print("\n--- C. SIMULACIÓN APLICADA: TIEMPO EN SEGMENTO DE CICLISMO ---")

# Caso: Carrera de Mountain Bike en Rincón de Romos.
# Fenómeno: El tiempo que toma un ciclista promedio en completar un segmento técnico (en minutos).
# Distribución: Uniforme Continua (se asume equiprobabilidad entre el tiempo min y max).

# Parámetros del caso real:
TIEMPO_MIN_A = 15  # Minutos (ciclista élite/experto)
TIEMPO_MAX_B = 25  # Minutos (ciclista recreativo/lento)
LOC_MTB = TIEMPO_MIN_A         # Ubicación (start/a)
SCALE_MTB = TIEMPO_MAX_B - TIEMPO_MIN_A # Escala (amplitud b-a) = 10 min

# Pregunta práctica: ¿Cuál es la probabilidad de que un ciclista complete el segmento
# en menos de 17 minutos (un tiempo considerado excelente)?
UMBRAL_TIEMPO = 17 # minutos

# 1. Generar 5000 tiempos simulados de ciclistas
muestras_mtb = uniform.rvs(loc=LOC_MTB, scale=SCALE_MTB, size=5000)

# 2. Calcular la probabilidad teórica (fórmula del área)
# P(X < 17) = (17 - 15) / (25 - 15) = 2 / 10 = 0.2
probabilidad_teorica_mtb = (UMBRAL_TIEMPO - TIEMPO_MIN_A) / SCALE_MTB

# 3. Calcular la probabilidad simulada (contando cuántos datos cumplen la condición)
ciclistas_rapidos = np.sum(muestras_mtb < UMBRAL_TIEMPO)
probabilidad_simulada_mtb = ciclistas_rapidos / len(muestras_mtb)


print(f"\n[Resultados de Simulación MTB Rincón de Romos]")
print(f"  > Media de tiempo TEÓRICA del segmento: {(TIEMPO_MIN_A + TIEMPO_MAX_B) / 2:.2f} minutos.")
print(f"  > Media de tiempo SIMULADA del segmento: {np.mean(muestras_mtb):.2f} minutos.")
print("-" * 50)
print(f"  > Probabilidad TEÓRICA de terminar en < {UMBRAL_TIEMPO} min: {probabilidad_teorica_mtb:.4f}")
print(f"  > Probabilidad SIMULADA de terminar en < {UMBRAL_TIEMPO} min: {probabilidad_simulada_mtb:.4f}")

plt.figure(figsize=(14, 6))
plt.hist(muestras_mtb, bins=30, density=True, color='darkviolet', alpha=0.7, label='Tiempos Simulados (Ciclistas)')
plt.axvline(x=UMBRAL_TIEMPO, color='red', linestyle='--', linewidth=2, label=f'Umbral de Tiempo Excelente ({UMBRAL_TIEMPO} min)')
plt.title(f'Simulación Aplicada: Tiempos del Segmento MTB en Rincón de Romos ($U[{TIEMPO_MIN_A}, {TIEMPO_MAX_B}]$)')
plt.xlabel('Tiempo de Segmento (Minutos)')
plt.ylabel('Densidad')
plt.xlim(TIEMPO_MIN_A - 1, TIEMPO_MAX_B + 1)
plt.legend()
plt.show()

print("\n=====================================================================")
print("  FIN DEL PROYECTO")
print("=====================================================================")