import matplotlib.pyplot as plt
import numpy as np
import cv2
import random
import scipy.linalg as scAlg
from pathlib import Path

def plotLabeledImagePoints(x, labels, strColor,offset):
    """
        Plot indexes of points on a 2D image.
         -input:
             x: Points coordinates.
             strColor: Color of the text.
             offset: Offset from the point to the text.
         -output: None
         """
    for k in range(x.shape[1]):
        plt.text(x[0, k]+offset[0], x[1, k]+offset[1], labels[k], color=strColor)


def plotNumberedImagePoints(x,strColor,offset):
    """
        Plot indexes of points on a 2D image.
         -input:
             x: Points coordinates.
             strColor: Color of the text.
             offset: Offset from the point to the text.
         -output: None
         """
    for k in range(x.shape[1]):
        plt.text(x[0, k]+offset[0], x[1, k]+offset[1], str(k), color=strColor)


def plotLabelled3DPoints(ax, X, labels, strColor, offset):
    """
        Plot indexes of points on a 3D plot.
         -input:
             ax: axis handle
             X: Points coordinates.
             strColor: Color of the text.
             offset: Offset from the point to the text.
         -output: None
         """
    for k in range(X.shape[1]):
        ax.text(X[0, k]+offset[0], X[1, k]+offset[1], X[2,k]+offset[2], labels[k], color=strColor)

def plotNumbered3DPoints(ax, X,strColor, offset):
    """
        Plot indexes of points on a 3D plot.
         -input:
             ax: axis handle
             X: Points coordinates.
             strColor: Color of the text.
             offset: Offset from the point to the text.
         -output: None
         """
    for k in range(X.shape[1]):
        ax.text(X[0, k]+offset[0], X[1, k]+offset[1], X[2,k]+offset[2], str(k), color=strColor)

def draw3DLine(ax, xIni, xEnd, strStyle, lColor, lWidth):
    """
    Draw a segment in a 3D plot
    -input:
        ax: axis handle
        xIni: Initial 3D point.
        xEnd: Final 3D point.
        strStyle: Line style.
        lColor: Line color.
        lWidth: Line width.
    """
    ax.plot([np.squeeze(xIni[0]), np.squeeze(xEnd[0])], [np.squeeze(xIni[1]), np.squeeze(xEnd[1])], [np.squeeze(xIni[2]), np.squeeze(xEnd[2])],
            strStyle, color=lColor, linewidth=lWidth)

def drawRefSystem(ax, T_w_c, strStyle, nameStr):
    """
        Draw a reference system in a 3D plot: Red for X axis, Green for Y axis, and Blue for Z axis
    -input:
        ax: axis handle
        T_w_c: (4x4 matrix) Reference system C seen from W.
        strStyle: lines style.
        nameStr: Name of the reference system.
    """
    draw3DLine(ax, T_w_c[0:3, 3:4], T_w_c[0:3, 3:4] + T_w_c[0:3, 0:1], strStyle, 'r', 1)
    draw3DLine(ax, T_w_c[0:3, 3:4], T_w_c[0:3, 3:4] + T_w_c[0:3, 1:2], strStyle, 'g', 1)
    draw3DLine(ax, T_w_c[0:3, 3:4], T_w_c[0:3, 3:4] + T_w_c[0:3, 2:3], strStyle, 'b', 1)
    ax.text(np.squeeze( T_w_c[0, 3]+0.1), np.squeeze( T_w_c[1, 3]+0.1), np.squeeze( T_w_c[2, 3]+0.1), nameStr)

def drawLine(l,strFormat,lWidth):
    """
    Draw a line
    -input:
      l: image line in homogenous coordinates
      strFormat: line format
      lWidth: line width
    -output: None
    """
    # p_l_y is the intersection of the line with the axis Y (x=0)
    p_l_y = np.hstack((0, -l[2] / l[1]))
    # p_l_x is the intersection point of the line with the axis X (y=0)
    p_l_x = np.hstack((-l[2] / l[0], 0))
    # Draw the line segment p_l_x to  p_l_y
    plt.plot([p_l_y[0], p_l_x[0]], [p_l_y[1], p_l_x[1]], strFormat, linewidth=lWidth)


def save_homography_txt(H, path="H_best.txt"):
    H = np.asarray(H, dtype=np.float64)
    np.savetxt(
        Path(path),
        H,
        fmt="%.8f",              # precisión
        delimiter="\t",          # separador (usa '\t' o ' ')
        header="Homography 3x3 (row-major). Normalized so H[2,2]=1"
    )
    # H_loaded = np.loadtxt("H_best.txt")

def save_fundamental_txt(F, path="F_best.txt"):
    F = np.asarray(F, dtype=np.float64)
    np.savetxt(
        Path(path),
        F,
        fmt="%.8f",              # precisión
        delimiter="\t",          # separador (usa '\t' o ' ')
        header="Fundamental 3x3 (row-major). ||F||=1"
    )
    # F_loaded = np.loadtxt("F_best.txt")

def ensure_2xN(X):
    X = np.asarray(X, dtype=float)
    return X if X.shape[0] == 2 else X.T

def normalizePoints2D(x):
    x = ensure_2xN(x)
    cx, cy = np.mean(x, axis=1)
    x_shift = x.copy()
    x_shift[0] -= cx
    x_shift[1] -= cy
    mean_dist = np.mean(np.sqrt(x_shift[0]**2 + x_shift[1]**2)) + 1e-12
    s = np.sqrt(2) / mean_dist
    T = np.array([[s, 0, -s*cx],
                  [0, s, -s*cy],
                  [0, 0,     1 ]], dtype=float)
    x_n = (T @ np.vstack([x, np.ones((1, x.shape[1]))]))[:2]
    return x_n, T

def summarize(name, arr):
    arr = np.asarray(arr) 
    print(f"{name}: mean={arr.mean():.3f}, median={np.median(arr):.3f}, max={arr.max():.3f}, p90={np.percentile(arr,90):.3f}, rmse={np.sqrt((arr**2).mean()):.3f}")


def transfer_error(H, x1, x2):
    """
    e_i = || π(H x1_i) - x2_i || (px). Devuelve errores y máscara válida (w!=0)
    Asume x1 es 3xN (homogéneo). x2 puede ser 2xN o 3xN.
    """
    # sanity checks opcionales
    assert x1.shape[0] == 3, f"x1 debe ser 3xN, got {x1.shape}"
    assert x2.shape[0] in (2,3), f"x2 debe ser 2xN o 3xN, got {x2.shape}"

    # proyectar
    X2h = H @ x1                        # 3xN
    w = X2h[2, :]                       # (N,)
    ok = np.abs(w) > 1e-12              # máscara de puntos finitos

    # normalizar homogéneas solo donde es válido
    X2h[:, ok] /= w[ok]

    # tomar solo las dos primeras filas de x2 (u,v)
    x2_2d = x2[:2, :] if x2.shape[0] == 3 else x2  # 2xN

    # error euclídeo por punto
    err = np.full(x1.shape[1], np.nan)
    err[ok] = np.linalg.norm(X2h[:2, ok] - x2_2d[:, ok], axis=0)
    return err, ok

def symmetric_transfer_error(H, x1, x2):
    """
    Error simétrico, que calcula el promedio del error de proyección hacia "adelante" y
    hacia "atrás" (usando H^{-1}).
    err[i] = 0.5 * ( ||π(H x1_i) - x2_i|| + ||π(H^{-1} x2_i) - x1_i|| )
    ok array de booleanos de puntos validos  
    Asume x1 y x2 es 3xN (homogéneo)
    """
    # adelante
    e12, ok12 = transfer_error(H, x1, x2)
    # atrás
    Hinv = np.linalg.inv(H)
    e21, ok21 = transfer_error(Hinv, x2, x1)
    # simétrico (media cuando hay ambos)
    err = np.nanmean(np.vstack([e12, e21]), axis=0)
    ok = ok12 & ok21
    return err, ok

def evaluate_homography(candidates, x1, x2):
    """
    Parámetros: 
    -----------
    candidates: todas las Homografías
    x1, x2: Puntos en ambas imagenes

    Devuelve: lista de (H, e, oks) donde
      e   -> errores por punto (array 1xN, píxeles)
      oks -> máscara booleana de puntos válidos (N,)
    """
    x1 = ensure_2xN(x1)
    x1 = np.vstack([x1, np.ones((1, x1.shape[1]))])  # 3xN homogéneo
    x2 = ensure_2xN(x2)                              
    x2 = np.vstack([x2, np.ones((1, x2.shape[1]))])  # 3xN homogéneo

    evaluation = []
    for candidate in candidates:  
        H = candidate

        e, oks = symmetric_transfer_error(H, x1, x2)     
        evaluation.append((H, e, oks))
    return evaluation

def select_best_Homography(candidatesH, x1, x2, agg='median', tau = 3.0):
    """
    Selecciona la mejor homografía en 'candidatesH' usando el error euclidio.

    Parámetros
    ----------
    candidatesH, x1, x2 : Homografía y puntos en img 1 y 2
    agg : str in {'median','mean','max','rmse','p90'}
    tau : tolerancia de inliers

    Devuelve
    --------
    best_H : np.ndarray (3x3)
        Homografía con menor error agregado y normalizada.
    best_score : float
        Valor del error agregado (píxeles).
    best_inliers : float
        Número de inliers de best_H
    
    """
    evals = evaluate_homography(candidatesH, x1, x2) # H, e, oks

    def aggregate(errors, how):
        if errors.size == 0: return np.inf
        if how == 'median': return float(np.median(errors))
        if how == 'mean': return float(np.mean(errors))
        if how == 'max': return float(np.max(errors))
        if how == 'rmse': return float(np.sqrt(np.mean(errors**2)))
        if how == 'p90': return float(np.percentile(errors, 90))
        raise ValueError(f"Aggregador no soportado: {how}")

    best_H, best_score = None, np.inf
    best_inliers = -1

    for (H, e, oks) in evals:
        # Nos quedamos con errores válidos e inliers (oks) y finitos
        valid = oks & np.isfinite(e)
        e_valid = e[valid]
        score = aggregate(e_valid, agg)

        # Criterio principal: menor score. Desempate: más inliers válidos.
        n_inl = int(np.sum(e[valid] < tau))
        #  |------------------| or |------------------------------------------------------|
        if (score < best_score) or (np.isclose(score, best_score) and n_inl > best_inliers):
            best_H, best_score, best_inliers = H, score, n_inl

    # Normaliza para evitar escalas arbitrarias (típico en homografías):
    if abs(best_H[2, 2]) > 1e-12:
        best_H = best_H / best_H[2, 2]

    return best_H, best_score, best_inliers

def RANSAC_H (n_hypotesis, x1, x2, agg='median', tau=3.0):
    """
    Calcula mediante RANSAC la mejor Homografía.
    Mediante 4 puntos aleatorios.

    Parámetros
    ----------
    n_hypotesis: Número de Homografias que se van a calcular.
    x1, x2: Puntos correspondientes a ambas imagenes en formato (N, 2).
    agg : str in {'median','mean','max','rmse','p90'}.
    tau: tolerancia de inliers.

    Devuelve
    --------
    H: La mejor homografía calculada y normalizada.
    e: El valor agregado de dicha homografía.
    inliers: Número de inliers de H
    """

    candidatesH = []

    for iters in range(n_hypotesis):
        x1_r = np.empty((0,2))  # --> (N,2)
        x2_r = np.empty((0,2))  # --> (N,2)

        idx = np.random.choice(x1.shape[0]-1, 4, replace=False)  # 4 índices únicos
        x1_r = x1[idx, :]                                        # (4,2)
        x2_r = x2[idx, :]                                        # (4,2)

        # # Normalizar puntos
        # x1n, T1 = normalizePoints2D(x1_sp[:2, :])  
        # x2n, T2 = normalizePoints2D(x2_sp[:2, :])
        # u1, v1 = x1n[:,0], x1n[:,1]
        # u2, v2 = x2n[:,0], x2n[:,1]

        u1, v1 = x1_r[:,0], x1_r[:,1]
        u2, v2 = x2_r[:,0], x2_r[:,1]

        # 2) Construir A (2N x 9), vectorizado
        N = x1_r.shape[0]
        Z = np.zeros(N); O = np.ones(N)
        A_top = np.column_stack([-u1, -v1, -O,  Z,  Z,  Z,  u2*u1, u2*v1, u2])
        A_bot = np.column_stack([  Z,  Z,  Z, -u1, -v1, -O,  v2*u1, v2*v1, v2])
        A = np.vstack([A_top, A_bot])

        # 3) SVD(A) -> último vector singular
        _, _, Vt = np.linalg.svd(A)
        Hn = Vt[-1].reshape(3,3)  # homografía en espacio normalizado

        # # 4) Des-normalizar: H = T2^{-1} Hn T1
        # H = np.linalg.inv(T2) @ Hn @ T1
        # # print(np.linalg.matrix_rank(H))  # rango 3

        candidatesH.append(Hn)   

    H, e, inliers = select_best_Homography(candidatesH, x1, x2, agg, tau)
    save_homography_txt(H)

    # print(H)
    # H_loaded = np.loadtxt("H_best.txt")
    # print(H_loaded)
    # print("Error: ", e)
    # print("Inliers: ", inliers)

    return H, e, inliers

def project_points(H, x):  # x: (N,2)
    xh = np.vstack([x.T, np.ones((1, x.shape[0]))])   # 3xN
    y  = H @ xh
    y /= y[2,:]
    return y[:2,:].T  # (N,2)


def evaluate_Fundamental(candidates, x1, x2):
    """
    Calcula la distancia simétrica punto-línea en píxeles.

    Parámetros: 
    -----------
    candidates: Todas las F a evaluar
    x1, x2: Todos los puntos de ambas imagenes

    Devuelve: lista de (F, e) 
    """
    x1 = ensure_2xN(x1)
    x2 = ensure_2xN(x2)

    x1h = np.vstack([x1, np.ones((1, x1.shape[1]))])  # 3xN
    x2h = np.vstack([x2, np.ones((1, x2.shape[1]))])  # 3xN

    evaluation = []
    for candidate in candidates:
        F = candidate

        l2 = F @ x1h                 # (3,N) líneas en img2 
        l1 = F.T @ x2h               # (3,N) líneas en img1

        # Distance between a point and the corresponding epipolar line in pixels.
        # -----------------------------------------------------------------------
        # dist punto-línea: |a*x + b*y + c| / sqrt(a^2+b^2)

        num2 = np.abs(l2[0,:]*x2[0,:] + l2[1,:]*x2[1,:] + l2[2,:])
        den2 = np.sqrt(l2[0,:]**2 + l2[1,:]**2) + 1e-12
        d2 = num2 / den2

        num1 = np.abs(l1[0,:]*x1[0,:] + l1[1,:]*x1[1,:] + l1[2,:])
        den1 = np.sqrt(l1[0,:]**2 + l1[1,:]**2) + 1e-12
        d1 = num1 / den1
        
        e = 0.5*(d1 + d2)

        # Sampson error
        # -------------
        # Fx1  = l2
        # Ftx2 = l1
        # # r = x2.T·F·x1 (escalar) --> solo buscamos la diagonal (residuo epipolar)
        # r = np.sum(x2h * Fx1, axis=0)**2
        # # (||F·x1||^2 + ||F^T·x2||^2)
        # den = Fx1[0]**2 + Fx1[1]**2 + Ftx2[0]**2 + Ftx2[1]**2 + 1e-12

        # e = np.sqrt(r / den) # Obtener resultado en px y poder usar tau con el mismo valor predeterminado

        evaluation.append((F, e))

    return evaluation



def aggregate(errors, how):
    if errors.size == 0: return np.inf
    if how == 'median': return float(np.median(errors))
    if how == 'mean': return float(np.mean(errors))
    if how == 'max': return float(np.max(errors))
    if how == 'rmse': return float(np.sqrt(np.mean(errors**2)))
    if how == 'p90': return float(np.percentile(errors, 90))
    raise ValueError(f"Aggregador no soportado: {how}")

def select_best_Fundamental(candidatesF, x1, x2, agg='median', tau = 3.0):
    """
    Selecciona la mejor homografía en 'candidatesF' usando la 
    distancia simétrica punto-línea en píxeles.

    Parámetros
    ----------
    candidatesF : todas las F
    x1, x2: Todos los puntos de ambas imagenes

    Devuelve
    --------
    best_F : np.ndarray (3x3)
        Matriz Fundamental con menor error agregado y normalizada.
    best_score : float
        Valor del error agregado (píxeles).
    best_inliers : float
        Número de inliers de best_F
    """
    evals = evaluate_Fundamental(candidatesF, x1, x2) # F, e


    best_F, best_score = None, np.inf
    best_inliers = -1

    for (F, e) in evals:
        score = aggregate(e, agg)
        # Criterio principal: más inliers válidos. Desempate: menor score.
        n_inl = int(np.sum(e < tau))
        #  |--------------------| or |------------------------------------------------|
        if (n_inl > best_inliers):# or ((n_inl == best_inliers) and (score < best_score)):
            best_F, best_score, best_inliers = F, score, n_inl


    return best_F, best_score, best_inliers

def RANSAC_F (n_hypotesis, x1, x2, agg='median', tau=1.0):
    """
    Calcula mediante RANSAC la mejor matriz Fundamental.
    Mediante 8 puntos aleatorios.

    Parámetros
    ----------
    n_hypotesis: Número de F que se van a calcular.
    x1, x2: Puntos correspondientes a ambas imagenes en formato (N, 2).
    agg : str in {'median','mean','max','rmse','p90'}.
    tau: tolerancia de inliers.

    Devuelve
    --------
    F: La mejor matriz F calculada y normalizada.
    e: El valor agregado de dicha F.
    inliers: Número de inliers de F.
    """
    
    candidatesF = []

    for iters in range(n_hypotesis):
        x1_r = np.empty((0,2))  # --> (N,2)
        x2_r = np.empty((0,2))  # --> (N,2)

        idx = np.random.choice(x1.shape[0]-1, 8, replace=False)  # 8 índices únicos
        x1_r = x1[idx, :]                                        # (8,2)
        x2_r = x2[idx, :]                                        # (8,2)

        u1, v1 = x1_r[:,0], x1_r[:,1]
        u2, v2 = x2_r[:,0], x2_r[:,1]

        # --- Normalización Hartley ---
        # x1n, T1 = normalizePoints2D(x1_r)
        # x2n, T2 = normalizePoints2D(x2_r)

        # u1, v1 = x1n[:,0], x1n[:,1]
        # u2, v2 = x2n[:,0], x2n[:,1]

        A = np.column_stack([
            u2*u1, u2*v1, u2,
            v2*u1, v2*v1, v2,
            u1,    v1,    np.ones_like(u1)
        ])
        # print("A shape:", A.shape)   # (N, 9) con N>=8

        # 3) SVD(A) y obtener F:
        _, _, Vt = np.linalg.svd(A)
        F_r3_n = Vt[-1].reshape(3,3)
        # print("rank(F) (numérico):", np.linalg.matrix_rank(F_r3_n)) # Por ruido debe ser rango 3

        # 4) Forzar rango 2 a F y desnormalizar:
        U, S, Vt_F = np.linalg.svd(F_r3_n)
        S[-1] = 0.0                         # anula el menor singular
        F = U @ np.diag(S) @ Vt_F
        # F = T2.T @ F @ T1              # desnormalizar

        candidatesF.append(F)    

    F, e, inliers = select_best_Fundamental(candidatesF, x1, x2, agg, tau)
    save_fundamental_txt(F)

    # print(F)
    # F_loaded = np.loadtxt("F_best.txt")
    # print(F_loaded)
    # print("Error: ", e)
    # print("Inliers: ", inliers)

    return F, e, inliers

def new_RANSAC_F(hypotesis, x1, x2, tau = 1.0):
    candidatesF = []
    for i in range(hypotesis):
        x1_r = np.empty((0,2))  # --> (N,2)
        x2_r = np.empty((0,2))  # --> (N,2)

        idx = np.random.choice(x1.shape[0]-1, 8, replace=False)  # 8 índices únicos
        x1_r = x1[idx, :]                                        # (8,2)
        x2_r = x2[idx, :]                                        # (8,2)

        u1, v1 = x1_r[:, 0], x1_r[:,1]
        u2, v2 = x2_r[:, 0], x2_r[:,1]


        #𝑥𝑥0𝑥𝑥1 𝑠𝑠0𝑥𝑥1 𝑤𝑤0𝑥𝑥1 𝑥𝑥0𝑠𝑠1 𝑠𝑠0𝑠𝑠1 𝑤𝑤0𝑠𝑠1 𝑥𝑥0𝑤𝑤1 𝑠𝑠0𝑤𝑤1 𝑤𝑤0𝑤𝑤1
        A = np.column_stack([u1*u2, v1*u2, u2, u1*v2, v1*v2, v2, u1, v1, np.ones_like(u1)])
        _, _, Vt = np.linalg.svd(A)
        F_r3_n = Vt[-1].reshape(3,3)
        # print("rank(F) (numérico):", np.linalg.matrix_rank(F_r3_n)) # Por ruido debe ser rango 3

        # 4) Forzar rango 2 a F y desnormalizar:
        U, S, Vt_F = np.linalg.svd(F_r3_n)
        S[-1] = 0.0                         # anula el menor singular
        F = U @ np.diag(S) @ Vt_F
        # F = T2.T @ F @ T1              # desnormalizar
        
        candidatesF.append(F)    
    F, e, inliers = select_best_Fundamental(candidatesF, x1, x2, "mean", tau)
    return F, e, inliers




if __name__ == '__main__':

    img1 = cv2.cvtColor(cv2.imread('image1.png'), cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(cv2.imread('image2.png'), cv2.COLOR_BGR2RGB)

    #path = './image1_image2_matches.npz'
    #path = './image1_image2_SIFT_matches.npz'
    path = './image1_image2_SIFT_guided_matches.npz'
    npz = np.load(path) # Load dictionary with super point
    # Create a boolean mask with True for keypoints with a good match, and False for the rest
    mask = npz['matches'] > -1       
    # Using the boolean mask, select the indexes of matched keypoints from image 2
    idxs = npz['matches'][mask]  
    # Using the boolean mask, select the keypoints from image 1 with a good match     
    x1_sp = npz['keypoints0'][mask]  # --> (N,2)
    # Using the indexes, select matched keypoints from image 2 
    x2_sp = npz['keypoints1'][idxs]  # --> (N,2)
    print(x1_sp.shape)
    print(x2_sp.shape)

    # # 4. RANSAC homography estimation
    # # ################################

    # n_hypotesis = 2000

    # H, e, _ = RANSAC_H(n_hypotesis, x1_sp, x2_sp)

    # # Identify the scene plane that defines the homography 
    # # ----------------------------------------------------
    # # La H solo es válida para definir puntos que se encuentran en una imagen plana
    # # donde solo hay traslación o en un entorno 3D con solo rtoación. 

    # # Por lo tanto, hay que encontrar que superficie real de la escena está siendo
    # # bien explicada por la H. Para esto, lo que hacemos es transferir los puntos de
    # # una imagen a la otra con li H y ver qué región queda alineada con un error pequeño.

    # x2_hat = project_points(H, x1_sp)               # proyección a img2
    # err = np.linalg.norm(x2_hat - x2_sp, axis=1)    # error por punto (distancia euclidia)
    # inliers = err < 3.0                             # umbral (px)
    # # Dibuja x2_hat[inliers] en la imagen 2 con cv2.circle para ver el plano    

    # plt.figure(1)
    # plt.imshow(img2, cmap='gray', vmin=0, vmax=255)
    # plt.plot(x2_sp[:, 0], x2_sp[:, 1],'rx', markersize=2)
    # plt.plot(x2_hat[inliers, 0], x2_hat[inliers, 1],'bo', markersize=2)
    # plt.title('Imagen2: Inliers')
    # plt.draw()  # We update the figure display
    # print('Click in the image to continue...')
    # plt.waitforbuttonpress()

    # h2, w2 = img2.shape[:2]
    # I1_warp = cv2.warpPerspective(img1, H, (w2, h2), flags=cv2.INTER_LINEAR) # Aplica la H a toda la imagen 1
    # blend = cv2.addWeighted(img2, 0.8, I1_warp, 0.5, 0.0) # Mezcla ambas imagenes
    
    # plt.figure(2)
    # plt.imshow(blend)
    # plt.plot(x2_hat[inliers, 0], x2_hat[inliers, 1], 'yo', markersize=2, label='proy inliers')
    # plt.title('Overlay: img1 warpeada sobre img2 (plano de la homografía)')
    # plt.legend(loc='lower right')
    # plt.show()


    # 5. RANSAC fundamental matrix estimation
    # #######################################

    n_hypotesis = 30000

    new_RANSAC_F(n_hypotesis, x1_sp, x2_sp, 3.0)

    F, e, inliers = new_RANSAC_F(n_hypotesis, x1_sp, x2_sp)

    # # Mostrar lineas epipolares    
    x1 = ensure_2xN(x1_sp)                       # asegura 2xN

    img2 = cv2.cvtColor(cv2.imread('image2.png'), cv2.COLOR_BGR2RGB)
    plt.figure(3)
    plt.imshow(img2, cmap='gray', vmin=0, vmax=255)

    for i in range(5):
        u, v = x1[:, i]                     # solo un punto 2D en imagen 1 (píxeles) 
        x1_h = np.array([u, v, 1.0])        # homogéneo 3x1

        l2 = F @ x1_h                     # recta en imagen 2: a,b,c
        drawLine(l2, '-', 2)

    plt.plot(x2_sp[:5, 0], x2_sp[:5, 1],'bx', markersize=8)
    plt.title('Epipolar Lines')
    plt.draw()  # We update the figure display
    plt.show()


