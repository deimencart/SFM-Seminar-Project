# Reconstrucción 3D SfM: Real Seminario de Zaragoza

> **Reconstrucción 3D Tridimensional a partir de Fotografía Histórica (c. 1950) y Moderna**  
> *Marco Académico: Visión por Computador y Geometría Multivista*  
> *Métodos: Calibración Intrínseca, Emparejamiento SuperGlue, Geometría Epipolar SVD, Triangulación DLT, Homografía RANSAC y VGGT*

---

> [!IMPORTANT]
> **Aviso de Proyecto Académico (Sin fines de plagio)**  
> Este repositorio contiene el desarrollo y los resultados de un proyecto académico experimental en visión por computador. Las formulaciones teóricas y algoritmos base (SuperGlue, LightGlue, Hartley-Zisserman y VGGT) pertenecen y son atribuidos a sus respectivos autores. Este trabajo se publica con fines puramente educativos y de demostración técnica.

---

## 📌 ¿Qué problema resuelve este proyecto?

El objetivo es reconstruir la geometría tridimensional de la fachada del **Real Seminario de Minería de Zaragoza** combinando dos fuentes visuales con una brecha temporal y tecnológica extrema:
1. Una **fotografía histórica de archivo en blanco y negro** (tomada aproximadamente en 1950, sin metadatos de cámara, con grano analógico y resolución distinta).
2. **Fotografías digitales modernas** tomadas desde diferentes ángulos en el campus actual con una cámara calibrada.

### Desafíos Superados
* **Brecha temporal de más de 70 años:** Cambios drásticos en la vegetación (árboles crecidos), presencia de vehículos modernos y desgaste de la piedra.
* **Cámara no calibrada en la foto histórica:** Se requiere estimar la relación geométrica entre una cámara analógica desconocida y la cámara moderna para recuperar las poses relativas $[R \mid \mathbf{t}]$ en el espacio.

---

## 🛠️ ¿Qué implementamos en este repositorio?

El pipeline de Structure from Motion (SfM) implementado orquesta desde la calibración hasta la triangulación 3D:

```
[ Foto Histórica (1950) ]              [ Fotos Modernas (Digital) ]
            |                                       |
            |                                       v
            |                         [ Calibración Intrínseca K ]
            |                         (Tablero ajedrez + desdistorsión)
            |                                       |
            +-------------------+-------------------+
                                |
                                v
               [ Emparejamiento SuperGlue (GNN) ]
              (Extracción de puntos de interés)
                                |
                                v
             [ Filtrado de Outliers con RANSAC ]
                (71 inliers / 33 descartados)
                                |
                                v
             [ Estimación Epipolar Normalizada ]
            (Hartley + Matriz Esencial por SVD)
                                |
                                v
               [ Triangulación 3D (DLT) ]
             (Cálculo de puntos 3D + RMSE 2.11 px)
                                |
                                v
             [ Reconstrucción Densa Neural VGGT ]
```

### 1. Calibración y Preparación de Imágenes
* **`src/camera_calibration.py`:** Calibra la cámara moderna mediante un patrón de tablero de ajedrez ($9 \times 6$), calculando la matriz intrínseca $K$ y los coeficientes de distorsión radial ($k_1, k_2$) para desdistorsionar las imágenes.
* **`src/cutting_camera.py`:** Normaliza las dimensiones de las fotos históricas y modernas a un tamaño común ($682 \times 384\,\text{px}$) forzando el centrado del punto principal óptico.

### 2. Emparejamiento y Filtrado de Puntos
* **`src/matches_in_between_points.py`:** Procesa las correspondencias de SuperGlue, descartando falsos positivos provocados por el movimiento de hojas de árboles, sombras y reflejos en automóviles.
* **`src/homographyRANSAC.py`:** Calcula la homografía plana con RANSAC para verificar la coplanaridad de las ventanas y molduras de la fachada.

### 3. Pipeline Epipolar y Triangulación 3D
* **`src/global_assignment.py` & `src/sfm_epipolar_pipeline.py`:**
  * Aplica la normalización isotrópica de Hartley a los puntos 2D para estabilizar el cálculo numérico.
  * Resuelve la matriz fundamental $F$ y la matriz esencial $E$ mediante descomposición en valores singulares (SVD).
  * Descompone $E$ en la rotación $R$ y traslación $\mathbf{t}$ relativas entre las dos cámaras.
  * Triangula los puntos correspondientes mediante DLT (Direct Linear Transform) para obtener las coordenadas 3D en el espacio.

### 4. Reconstrucción Neuronal 3D (VGGT)
* Se integró el modelo de transformer geométrico **VGGT** (*Visual Geometry Grounded Transformer*) para generar una reconstrucción densa y continua de la fachada a partir de las vistas emparejadas.

---

## 📊 Resultados y Métricas Cuantitativas

* **Puntos emparejados iniciales:** 104 correspondencias.
* **Inliers geométricos confirmados:** **71 puntos (68.3%)**.
* **Outliers descartados:** **33 puntos (31.7%)** (eliminación de falsos positivos en vegetación y vehículos).
* **Error de reproyección (Camera 1):**
  * **RMSE:** **2.110 píxeles**.
  * **Error Medio:** **1.969 píxeles**.

| Reconstrucción 3D y Poses de Cámara | Reconstrucción Densa Neural (VGGT) |
| :---: | :---: |
| ![3D Reconstruction](docs/images/reconstruction_3d_cameras_pointcloud.png) | ![VGGT](docs/images/seminario_vggt_reconstruction.png) |

| Correspondencias Filtradas SuperGlue | Verificación de Error de Reproyección |
| :---: | :---: |
| ![SuperGlue](docs/images/superglue_matches_global.png) | ![Reprojection Error](docs/images/reprojection_error_visualization.png) |

| Vista Cenital de Cámaras | Histograma de Error de Reproyección |
| :---: | :---: |
| ![Top View](docs/images/reconstruction_3d_topview.png) | ![RMSE](docs/images/rmse_distribution_camera.png) |

---

## 📁 Estructura del Repositorio

```text
seminario_zaragoza_sfm/
├── README.md                                 <- Documentación general
├── .gitignore                                <- Exclusiones de Git
├── docs/
│   ├── Seminario_1.pptx                      <- Presentación ejecutiva completa (28 slides)
│   ├── CVIS_Course_Assignment_2025_26.pdf    <- Enunciado y requerimientos
│   ├── Pipeline.drawio                       <- Diagrama de flujo de datos
│   ├── 1_Inputs.drawio                       <- Diagrama de entradas
│   └── images/                               <- Renders, nubes 3D y gráficas de error
├── src/
│   ├── camera_calibration.py                 <- Calibración con tablero y desdistorsión
│   ├── global_assignment.py                  <- Pipeline principal SfM (SVD, DLT)
│   ├── global_assignment_2.py                <- Pipeline SfM multivistal extendido
│   ├── homographyRANSAC.py                   <- Homografía plana robusta
│   ├── cutting_camera.py                     <- Ajuste dimensional y centrado de FOV
│   ├── matches_in_between_points.py          <- Depuración de correspondencias SuperGlue
│   ├── sfm_epipolar_pipeline.py              <- Verificación epipolar y normalización Hartley
│   ├── utilities.py                          <- Ensamblado SE(3) y graficación 3D
│   └── utilities2.py                         <- Utilidades adicionales
├── data/
│   ├── calibration/                          <- Matriz K y parámetros intrínsecos
│   ├── raw_images/                           <- Fotografía histórica (1950) y fotos modernas
│   ├── undistorted/                          <- Fotos corregidas por desdistorsión
│   └── matches/                              <- Arrays .npz con emparejamientos
└── results/                                  <- Salidas numéricas
```

---

## 🚀 Guía Rápida de Ejecución

### 1. Dependencias
```bash
pip install numpy opencv-python matplotlib scipy
```

### 2. Calibrar la Cámara
```bash
python src/camera_calibration.py
```

### 3. Ejecutar la Reconstrucción 3D (SfM)
Calcula las poses de las dos cámaras y triangula la nube de puntos 3D:
```bash
python src/global_assignment.py
```

### 4. Estimar Homografías Planas
```bash
python src/homographyRANSAC.py
```

---

## 📄 Presentación Técnica Completa
En la carpeta [`docs/Seminario_1.pptx`](docs/Seminario_1.pptx) se encuentra la presentación técnica de 28 diapositivas con el detalle de cada paso del algoritmo, las matrices numéricas obtenidas y la visualización de la nube de puntos.
