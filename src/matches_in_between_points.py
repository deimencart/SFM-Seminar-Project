import numpy as np

"""Configuration parameters for the pipeline"""
DATA_PATH = "D:/Dev_Space/python/Gloabal_proyect/src/auxiliary_files/"
MATCHES_1_2 = DATA_PATH + "und_seminario_new_04_und_seminario_new_01_matches.npz"
MATCHES_OLD_1 = DATA_PATH + "seminario1_fixed_und_seminario_new_04_matches.npz"

npz12 = np.load(MATCHES_1_2)
npzOld1 = np.load(MATCHES_OLD_1)

kp1 = npz12['keypoints0']      # new_04 (607 kpts)
kp2 = npz12['keypoints1']      # new_01
kp1_b = npzOld1['keypoints1']  # new_04 (607 kpts) - idéntico a kp1
kpOld = npzOld1['keypoints0']  # old

matches12 = npz12['matches']       # 607 elementos
matchesOld1 = npzOld1['matches']   # 626 elementos

print("=== VERIFICACIÓN ===")
print(f"kp1 == kp1_b: {np.allclose(kp1, kp1_b)}")
print(f"matches12: {matches12.shape}")
print(f"matchesOld1: {matchesOld1.shape}")

# SOLUCIÓN: Truncar matchesOld1 a la longitud correcta
matchesOld1 = matchesOld1[:len(kp1_b)]  # Tomar solo los primeros 607
print(f"matchesOld1 (truncado): {matchesOld1.shape}")

# Ahora construir tracks (código simplificado porque kp1 == kp1_b)
mask12 = matches12 >= 0       # Matches válidos new_04 → new_01
maskOld1 = matchesOld1 >= 0   # Matches válidos old → new_04
mask123 = mask12 & maskOld1   # Puntos visibles en las 3 imágenes

# Índices de keypoints en new_04 que tienen tracks completos
idx_new04 = np.where(mask123)[0]

# Construir matriz de tracks
tracks = np.vstack([
    idx_new04,                 # Índice en new_04 (imagen base)
    matches12[idx_new04],      # Índice en new_01
    matchesOld1[idx_new04]     # Índice en old
])

# Extraer coordenadas
x1 = kp1[tracks[0]]        # Coordenadas en new_04
x2 = kp2[tracks[1]]        # Coordenadas en new_01
x3 = kpOld[tracks[2]]      # Coordenadas en old

print("\n=== RESULTADOS ===")
print(f"Matches válidos:")
print(f"  new_04 → new_01: {np.sum(mask12)} matches")
print(f"  old → new_04: {np.sum(maskOld1)} matches")
print(f"  Tracks 3-view: {tracks.shape[1]} tracks")
print(f"\nx1 (new_04): {x1.shape}")
print(f"x2 (new_01): {x2.shape}")
print(f"x3 (old): {x3.shape}")