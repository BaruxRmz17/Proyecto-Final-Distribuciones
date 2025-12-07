# ==============================================================================
# 1. CONFIGURACIÓN E IMPORTACIÓN DE LIBRERÍAS
# ==============================================================================
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import expon, uniform

# Configuraciones para mejor visualización de gráficas
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (14, 6)
N_MUESTRAS = 100000 # Un gran número para asegurar una buena convergencia

print("=====================================================================")
print("  INICIO DE LA SIMULACIÓN: VALIDACIÓN DE DISTRIBUCIONES TEÓRICAS")
print("=====================================================================")


# ==============================================================================
# A. DISTRIBUCIÓN EXPONENCIAL
# ==============================================================================
print("\n--- A. ANÁLISIS DE LA DISTRIBUCIÓN EXPONENCIAL ---")

# Parámetros: Tasa de eventos por unidad de tiempo
LAMBDA = 0.5  # Tasa (lambda) = 0.5 eventos/unidad_tiempo
ESCALA_EX = 1 / LAMBDA # Escala es el inverso de lambda, y es la Media Teórica

# 1. Generación de muestras aleatorias
muestras_exp = expon.rvs(scale=ESCALA_EX, size=N_MUESTRAS)

# 2. Visualización: Histograma vs. Función de Densidad Teórica (PDF)
plt.figure(figsize=(14, 6))
plt.hist(muestras_exp, bins=50, density=True, alpha=0.7, color='skyblue', label='Frecuencia Muestral')

# Definir el rango x para la curva teórica
x_exp = np.linspace(0, muestras_exp.max() * 0.9, 100)
# Dibujar la función de densidad de probabilidad (PDF) teórica
pdf_exp = expon.pdf(x_exp, scale=ESCALA_EX)
plt.plot(x_exp, pdf_exp, 'r--', linewidth=2.5, label='PDF Teórica ($f(x; \lambda)$)')

plt.title(f'Distribución Exponencial ($\lambda={LAMBDA}$): Histograma vs. PDF Teórica')
plt.xlabel('Tiempo de Espera ($x$)')
plt.ylabel('Densidad de Probabilidad')
plt.legend()
plt.show()

# 3. Comparación de Estadísticos Muestrales vs. Teóricos
media_teorica_exp = ESCALA_EX
varianza_teorica_exp = ESCALA_EX**2

media_muestral_exp = np.mean(muestras_exp)
varianza_muestral_exp = np.var(muestras_exp)

print(f"\n[3. Comparación de Estadísticos Exponenciales]")
print(f"  > Media Teórica:   {media_teorica_exp:.4f}")
print(f"  > Media Muestral:  {media_muestral_exp:.4f}")
print(f"  > Varianza Teórica: {varianza_teorica_exp:.4f}")
print(f"  > Varianza Muestral: {varianza_muestral_exp:.4f}")


# ==============================================================================
# B. DISTRIBUCIÓN UNIFORME CONTINUA
# ==============================================================================
print("\n--- B. ANÁLISIS DE LA DISTRIBUCIÓN UNIFORME ---")

# Parámetros: Límites del intervalo [a, b]
A = 2
B = 8
LOC_UN = A       # Ubicación (start/a)
SCALE_UN = B - A # Escala (b-a) o amplitud

# 1. Generación de muestras aleatorias
muestras_unif = uniform.rvs(loc=LOC_UN, scale=SCALE_UN, size=N_MUESTRAS)

# 2. Visualización: Histograma vs. Función de Densidad Teórica (PDF)
plt.figure(figsize=(14, 6))
# El histograma debe tener un alto uniforme dentro del rango [A, B]
plt.hist(muestras_unif, bins=15, density=True, alpha=0.7, color='lightgreen', label='Frecuencia Muestral')

# Definir el rango x para la curva teórica
x_unif = np.linspace(A - 1, B + 1, 100)
# Dibujar la función de densidad de probabilidad (PDF) teórica
pdf_unif = uniform.pdf(x_unif, loc=LOC_UN, scale=SCALE_UN)
plt.plot(x_unif, pdf_unif, 'b-', linewidth=2.5, label='PDF Teórica ($f(x; a, b)$)')

plt.title(f'Distribución Uniforme: Histograma vs. PDF Teórica ($a={A}, b={B}$)')
plt.xlabel('Valor ($x$)')
plt.ylabel('Densidad de Probabilidad')
plt.xlim(A - 0.5, B + 0.5) # Ajustar límites para claridad
plt.legend()
plt.show()

# 3. Comparación de Estadísticos Muestrales vs. Teóricos
media_teorica_unif = (A + B) / 2
varianza_teorica_unif = (B - A)**2 / 12

media_muestral_unif = np.mean(muestras_unif)
varianza_muestral_unif = np.var(muestras_unif)

print(f"\n[3. Comparación de Estadísticos Uniformes]")
print(f"  > Media Teórica:   {media_teorica_unif:.4f}")
print(f"  > Media Muestral:  {media_muestral_unif:.4f}")
print(f"  > Varianza Teórica: {varianza_teorica_unif:.4f}")
print(f"  > Varianza Muestral: {varianza_muestral_unif:.4f}")


# ==============================================================================
# C. SIMULACIÓN APLICADA (EJEMPLO DE CASO REAL)
# ==============================================================================
print("\n--- C. SIMULACIÓN APLICADA: TIEMPO DE ESPERA EN SERVICIO ---")

# Caso: Tiempos de servicio de un cajero automático (Exponencial)
# Tasa de servicio: El cajero atiende 3 clientes cada 10 minutos.
# LAMBDA_SERVICIO = 3/10 = 0.3 clientes/minuto.
LAMBDA_SERVICIO = 0.3
ESCALA_SERVICIO = 1 / LAMBDA_SERVICIO # Media = 3.33 min/cliente

# Generar 1000 tiempos de servicio
tiempos_servicio = expon.rvs(scale=ESCALA_SERVICIO, size=1000)

# Calcular la probabilidad de que el servicio dure más de 5 minutos
probabilidad_teorica = expon.sf(5, scale=ESCALA_SERVICIO) # sf = 1 - CDF
probabilidad_simulada = np.sum(tiempos_servicio > 5) / len(tiempos_servicio)

print(f"\n[Simulación de Servicio (Exponencial)]")
print(f"  > Tiempo promedio de servicio teórico: {ESCALA_SERVICIO:.2f} minutos.")
print(f"  > Probabilidad Teórica de que el servicio dure más de 5 min: {probabilidad_teorica:.4f}")
print(f"  > Probabilidad Simulada de que el servicio dure más de 5 min: {probabilidad_simulada:.4f}")

plt.figure(figsize=(14, 6))
plt.hist(tiempos_servicio, bins=30, density=True, color='purple', alpha=0.6, label='Tiempos de Servicio Simulados')
plt.axvline(x=5, color='r', linestyle='--', linewidth=2, label='Umbral de 5 minutos')
plt.title('Simulación de Tiempos de Servicio (Minutos)')
plt.xlabel('Tiempo de Servicio')
plt.ylabel('Densidad')
plt.legend()
plt.show()

print("\n=====================================================================")
print("  FIN DE LA SIMULACIÓN")
print("=====================================================================")