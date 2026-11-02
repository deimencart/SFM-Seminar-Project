from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt
import numpy as np
import cv2


# Ensamble T matrix
def ensamble_T(R_w_c, t_w_c) -> np.array:
    """
    Ensamble the a SE(3) matrix with the rotation matrix and translation vector.
    """
    T_w_c = np.zeros((4, 4))
    T_w_c[0:3, 0:3] = R_w_c
    T_w_c[0:3, 3] = t_w_c
    T_w_c[3, 3] = 1
    return T_w_c


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




def ensure_2xN(X):
    X = np.asarray(X, dtype=float)
    return X if X.shape[0] == 2 else X.T

def hat(v):
    x, y, z = np.asarray(v).ravel()
    return np.array([[ 0, -z,  y],
                     [ z,  0, -x],
                     [-y,  x,  0]], dtype=float)

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

def projection_matrices_from_Rt(K, R, t):
    P1 = K @ np.hstack([np.eye(3), np.zeros((3,1))])    # K [I|0]
    P2 = K @ np.hstack([R, t.reshape(3,1)])             # K [R|t]
    return P1, P2

def reprojection_errors(P, X, x):
    """
    P: 3x4 ; X: 3xN ; x: 2xN  (píxeles)
    return: errores euclídeos por punto (N,)
    """
    Xh = np.vstack([X, np.ones((1, X.shape[1]))])
    x_hat = P @ Xh
    x_hat = x_hat[:2] / x_hat[2]
    return np.linalg.norm(x_hat - x, axis=0)

def triangulate_dlt(P1, P2, x1, x2):
    # Igual que el 1.2 

    N = x1.shape[1]
    X = np.zeros((3,N))
    for i in range(N):
        u1, v1 = x1[:, i]
        u2, v2 = x2[:, i]
        A = np.vstack([
            u1*P1[2] - P1[0],
            v1*P1[2] - P1[1],
            u2*P2[2] - P2[0],
            v2*P2[2] - P2[1],
        ])
        _,_,Vt = np.linalg.svd(A)
        Xh = Vt[-1]
        X[:, i] = Xh[:3]/Xh[3]
    return X

def cheirality_count(R, t, K, x1, x2, z_eps=1e-6):
    P1, P2 = projection_matrices_from_Rt(K, R, t)
    X = triangulate_dlt(P1, P2, x1, x2)                 # 3xN en marco cam1
    # Profundidades
    z1 = X[2, :]                                        # en cam1: z = Z
    X2 = (R @ X) + t.reshape(3,1)
    z2 = X2[2, :]                                       # en cam2
    ok = (z1 > z_eps) & (z2 > z_eps)
    # Errores de reproyección (para desempatar si hace falta)
    e1 = reprojection_errors(P1, X[:, ok], x1[:, ok]) if np.any(ok) else np.array([np.inf])
    e2 = reprojection_errors(P2, X[:, ok], x2[:, ok]) if np.any(ok) else np.array([np.inf])
    return ok.sum(), (e1.mean() + e2.mean())/2.0, X, P1, P2

def summarize(name, arr):
    arr = np.asarray(arr)
    print(f"{name}: mean={arr.mean():.3f}, median={np.median(arr):.3f}, max={arr.max():.3f}, p90={np.percentile(arr,90):.3f}, rmse={np.sqrt((arr**2).mean()):.3f}")


def sampson_error(F, x1, x2):
    
    x1 = ensure_2xN(x1)
    X1 = np.vstack([x1, np.ones((1, x1.shape[1]))])  # 3xN homogéneo
    x2 = ensure_2xN(x2)
    X2 = np.vstack([x2, np.ones((1, x2.shape[1]))])  # 3xN homogéneo

    Fx1  = F @ X1
    Ftx2 = F.T @ X2

    # r = x2.T·F·x1 (escalar) --> solo buscamos la diagonal (residuo epipolar)
    r = np.sum(X2 * Fx1, axis=0)**2

    # (||F·x1||^2 + ||F^T·x2||^2)
    den = Fx1[0]**2 + Fx1[1]**2 + Ftx2[0]**2 + Ftx2[1]**2 + 1e-12

    return r / den  # N valores

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



if __name__ == '__main__':
    np.set_printoptions(precision=4,linewidth=1024,suppress=True)


    # Load ground truth
    T_w_c1 = np.loadtxt('T_w_c1.txt')
    T_w_c2 = np.loadtxt('T_w_c2.txt')

    K_c = np.loadtxt('K_c.txt')
    X_w = np.loadtxt('X_w.txt')

    x1 = np.loadtxt('x1Data.txt')
    x2 = np.loadtxt('x2Data.txt')

    F21_test = np.loadtxt("F_21_test.txt")



    # 1.1 Projection Matrices:
    # ########################    
    # [sx, sy, s]^T = P · [X, Y, Z, w]^T     

    P1 = K_c @ np.eye(3,4) @ np.linalg.inv(T_w_c1) # mundo a Img1
    P2 = K_c @ np.eye(3,4) @ np.linalg.inv(T_w_c2) # mundo a Img2



    # 1.2 Triangulate:
    # ################
    # Cada punto 2D x1 y x2 en las imagenes definen un rayo q sale del centro óptico de cada camara
    # AX = 0 --> X es el punto 3D origniado por la intersección de los 2 rayos, coincide con la posicion de la otra camara.
    
    N = x1.shape[1]                 # nº de correspondencias (asumimos x1, x2 de forma 2xN)
    X = np.zeros((3,N))             # guardaremos los 3D en coordenadas euclídeas (mundo)

    for i in range(N):

        u1, v1 = x1[:, i]           # (u,v) de la vista 1 para el punto i
        u2, v2 = x2[:, i]           # (u,v) de la vista 2 para el punto i
        
        # ---- Obtener A del sistema A X_h = 0
        # Cada observación en una cámara aporta 2 filas:
        #   (u * P[2,:] - P[0,:]) X_h = 0
        #   (v * P[2,:] - P[1,:]) X_h = 0
        A = np.vstack([
          u1*P1[2]-P1[0],
          v1*P1[2]-P1[1],
          u2*P2[2]-P2[0],
          v2*P2[2]-P2[1]
        ])
        
        # ----- Resolver A X_h = 0 por SVD -----
        # SVD: A = U S V^T. El vector en V^T asociado al menor singular (última fila)
        # es la solución (en sentido de mínimos cuadrados) del sistema homogéneo.
        _, _, Vt = np.linalg.svd(A)
        Xh = Vt[-1]                  # X_h = [X, Y, Z, W] (homogéneo)
        # Selecciona último vector

        # Pasar a coordenadas euclídeas dividiendo por W
        X[:, i] = Xh[:3] / Xh[3]

    fig3D = plt.figure()
    ax = plt.axes(projection='3d', adjustable='box')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    drawRefSystem(ax, np.eye(4), '-', 'W')
    drawRefSystem(ax, T_w_c1, '-', 'C1')
    drawRefSystem(ax, T_w_c2, '-', 'C2')

    ax.scatter(X[0, :], X[1, :], X[2, :], marker='.')
    # plotNumbered3DPoints(ax, X, 'r', (0.1, 0.1, 0.1))
    ax.scatter(X_w[0, :], X_w[1, :], X_w[2, :], marker='.')
    plt.show(block=False)
    plt.waitforbuttonpress()



    # 2.1 Epipolar lines visualization 
    # ################################
    # l_2 = F_21 x_1
    # x_1 = [u, v, 1]^T
    
    # F21: 3x3 (de imagen 1 a 2)
    # x1 : (u,v) en píxeles (imagen 1)
    # (a,b,c) de la recta en imagen 2: a*x + b*y + c = 0
    
    F21_test = np.loadtxt("F_21_test.txt")

    x1 = ensure_2xN(x1)                     # asegura 2xN

    img2 = cv2.cvtColor(cv2.imread('image2.png'), cv2.COLOR_BGR2RGB)
    plt.figure(1)
    plt.imshow(img2, cmap='gray', vmin=0, vmax=255)

    for i in range(5):
        u, v = x1[:, i]                     # solo un punto 2D en imagen 1 (píxeles) 
        x1_h = np.array([u, v, 1.0])        # homogéneo 3x1

        l2 = F21_test @ x1_h                     # recta en imagen 2: a,b,c
        drawLine(l2, '-', 2)

        # # ---- Resolver ax+by+c=0 ----
        # xe1 = 0; ye1 = -l2[2] / l2[1]
        # ye2 = 0; xe2 = -l2[2] / l2[0]
        # # No hace falta resolver esto pq ya tenemos la funcion drawLine q la dibuja directamente

    plt.plot(x2[0, 0:5], x2[1, 0:5],'rx', markersize=10)
    plt.title('Epipolar Line 2.1')
    plt.draw()  # We update the figure display
    print('Click in the image to continue...')
    plt.waitforbuttonpress()



    # 2.2 Fundamental matrix definition 
    # #################################
    # F = K^(-T) E K^(-1)
    # E = [t]x R

    # 1) Obtener T_c2_c1
    T_c1_w = np.linalg.inv(T_w_c1)
    T_c2_w = np.linalg.inv(T_w_c2)
    T_c2_c1 = T_c2_w @ T_w_c1
    # 2) Obtener R y t
    R, t = T_c2_c1[:3,:3], T_c2_c1[:3,3]
    # print("R:\n", R)
    # 3) Obtener E y F
    E_21 = hat(t) @ R
    F_21 = np.linalg.inv(K_c).T @ E_21 @ np.linalg.inv(K_c)
    
    x1 = ensure_2xN(x1)                       # asegura 2xN

    img2 = cv2.cvtColor(cv2.imread('image2.png'), cv2.COLOR_BGR2RGB)
    plt.figure(2)
    plt.imshow(img2, cmap='gray', vmin=0, vmax=255)

    for i in range(5):
        u, v = x1[:, i]                     # solo un punto 2D en imagen 1 (píxeles) 
        x1_h = np.array([u, v, 1.0])        # homogéneo 3x1

        l2 = F_21 @ x1_h                    # recta en imagen 2: a,b,c
        drawLine(l2, '-', 2)

    plt.plot(x2[0, 0:5], x2[1, 0:5],'rx', markersize=10)
    plt.title('Epipolar Line 2.2')
    plt.draw()  # We update the figure display
    print('Click in the image to continue...')
    plt.waitforbuttonpress()



    # 2.3 Fundamental matrix linear estimation with eight point solution
    # ##################################################################
    # x2^T · F · x1 = 0
    # x1 = [u1, v1, 1]^T       x2 = [u2, v2, 1]^T
    # l2 = F · x1              l1 = F^T · x2
    # A · f = 0     ----->     f es el vector columna de F

    # Con exactamente 8 puntos, A tiene rango 8
    # A = U · S · V^T
    # f es la última fila de V^T (V[:,9])  --> F = f.reshape(3,3)
    # Por ruido, F no tiene rango 2, se fuerza poniendo el menor valor singular a 0

    # El algoritmo lineral original es numéricamente inestable, se mejora normalizando los puntos
    # antes de calcular A, por lo que para cada imagen construimos una transformación de similitud T
    # donde (u,v) es la media de los puntos y s = sqrt(2)/d, siendo d la media de las distancias de los
    # puntos a (u,v). Por tanto, (u,v) es el centro de los puntos y s un factor de escala que hace que
    # la media de las distancias al centro sea sqrt(2)
    #       [s  0 -s·u]        1. Aplicar xn = T·x y construir A con xn
    #   T = [0  s -s·v]        2. Estimar Fn y forzar rango 2
    #       [0  0   1 ]        3. Desnormalizar F = T2^T · Fn · T1

    # 1) Normalizar puntos:
    x1 = np.loadtxt('x1Data.txt')
    x2 = np.loadtxt('x2Data.txt')
    x1n, T1 = normalizePoints2D(x1)
    x2n, T2 = normalizePoints2D(x2)

    # print("Dist media tras normalizar x1:", np.mean(np.sqrt(x1n[0]**2 + x1n[1]**2))) # debe ser sqrt(2)
    # print("Dist media tras normalizar x2:", np.mean(np.sqrt(x2n[0]**2 + x2n[1]**2))) # es decir, 1.4142

    # 2) Construir A con los puntos normalizados:
    u1, v1 = x1n[0], x1n[1]
    u2, v2 = x2n[0], x2n[1]

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
    F_r2_n = U @ np.diag(S) @ Vt_F
    F = T2.T @ F_r2_n @ T1              # desnormalizar
    # print("F:\n", F)
    # print("rank(F) (numérico):", np.linalg.matrix_rank(F)) # rango 2

    # 5) Mostrar lineas epipolares    
    x1 = ensure_2xN(x1)                       # asegura 2xN

    img2 = cv2.cvtColor(cv2.imread('image2.png'), cv2.COLOR_BGR2RGB)
    plt.figure(3)
    plt.imshow(img2, cmap='gray', vmin=0, vmax=255)

    for i in range(5):
        u, v = x1[:, i]                     # solo un punto 2D en imagen 1 (píxeles) 
        x1_h = np.array([u, v, 1.0])        # homogéneo 3x1

        l2 = F @ x1_h                       # recta en imagen 2: a,b,c
        drawLine(l2, '-', 2)

    plt.plot(x2[0, 0:5], x2[1, 0:5],'rx', markersize=10)
    plt.title('Epipolar Line 2.3')
    plt.draw()  # We update the figure display
    print('Click in the image to continue...')
    plt.waitforbuttonpress()



    # 2.4 Pose estimation from two views
    # ##################################

    # 1) Calcular E a partir de F
    # E = K^T · F · K   --->   rank(E) = 2, con los primeros valores singulares, tal que, (1, 1, 0)
    _E_21 = K_c.T @ F21_test @ K_c
    # Forzar forma ideal: U diag(1,1,0) V^T
    U, S, Vt = np.linalg.svd(_E_21)
    # Asegura rotación propia en U@Vt (det>0)
    if np.linalg.det(U @ Vt) < 0:
        Vt = -Vt
    E_21 = U @ np.diag([1.0, 1.0, 0.0]) @ Vt

    # print("E_21:\n", E_21)
    # print("rank(E_21) (numérico):", np.linalg.matrix_rank(E_21))

    # 2) Calcular las 4 posibles soluciones de (R,t) a partir de E
    # E = U · diag(1,1,0) · V^T, define:
    #
    #     [0, -1, 0]          { 4 soluciones: (R1,  t), (R1, -t), (R2,  t), (R2, -t)
    # W = [1,  0, 0]   --->   { R1 = U·W·V^T   o   R2 = U·W^T·V^T
    #     [0,  0, 1]          { t = u3  (tercera columna de U)
    #
    # Las 4 soluciones son equiprobables a priori, para saber cual es la correcta
    # se triangulan los puntos y se elige la solución que da más puntos delante de las cámaras.

    U, S, Vt = np.linalg.svd(E_21)
    # Asegurar U,V de rotación propia
    if np.linalg.det(U @ Vt) < 0:
        Vt = -Vt
    W = np.array([[0, -1,  0],
                  [1,  0,  0],
                  [0,  0,  1]], dtype=float)

    R1 = U @ W  @ Vt
    R2 = U @ W.T @ Vt
    t  = U[:, 2]            

    # Garantiza det(R)=+1
    if np.linalg.det(R1) < 0: R1 = -R1; t = -t
    if np.linalg.det(R2) < 0: R2 = -R2; t = -t

    # Cuatro combinaciones: (R1, ±t), (R2, ±t)
    combinaciones = [(R1,  t), (R1, -t), (R2,  t), (R2, -t)]

    # print("Combinaciones (R,t) posibles:")
    # for i, (R, t) in enumerate(combinaciones):
    #     print(f"Solución {i+1}:")
    #     print("R:\n", R)
    #     print("t:\n", t)
    #     print("det(R):", np.linalg.det(R))
    #     print()

    # 3) Construir matrices de proyección P1 y P2 para comprobar las 4 soluciones
    # Camara 1 como referencia: P1 = K [I|0]
    # Camara 2: P2 = K [R|t]
    # Nota: Es posible trabajar sin K, pero los puntos 2D deben estar en coordenadas
    # normalizadas, por tanto, hay que ser coherente, si se trabaja con K, se usan pixeles,
    # si no, puntos normalizados.

    # ---> def projection_matrices_From_Rt(K, R, t):   --->   devuelve P1 y P2 
    
    # 4) Triangulación y criterio de quiralidad (cheirality)
    # Para cada (R, t) se triangulan todos los pares (x1, x2) con P1, P2.
    # Obteniendo X en el marco de la cámara 1.
    # Profundidad en camara 1: Z1 = Xz (tercer componente de X)
    # Profundidad en camara 2: X2 = R·X + t  --->  Z2 = (X2)z (tercer componente de X2)
    # Se cuenta el nº de puntos con Z1>0 y Z2>0, la solución correcta es la que maximice este conteo.
    # Por ruido, usar un umbral pequeño: Z>1e-6. Si hay empate, se usa la mediana del error de reproyección.

    # ---> def cheirality_count(R, t, K, x1, x2, z_eps=1e-6):
    #      return ---> ok.sum(), (e1.mean() + e2.mean())/2.0, X, P1, P2

    # ---> def triangulate_dlt(P1, P2, x1, x2):
    #      return ---> X # 3xN en coordenadas euclídeas

    # ---> def reprojection_errors(P, X, x):
    #      return ---> errores euclídeos por punto (N,) 

    x1  = ensure_2xN(np.loadtxt("x1Data.txt"))
    x2  = ensure_2xN(np.loadtxt("x2Data.txt"))

    # Hacer cheriality count para las 4 combinaciones
    scores = []
    for idx, (R, t) in enumerate(combinaciones):
        n_ok, mean_rep, X, P1, P2 = cheirality_count(R, t, K_c, x1, x2)
        scores.append((n_ok, mean_rep, idx, R, t))
        # print(f"Hipótesis {idx+1}: delante en ambas = {n_ok},  reproy_mean(px) = {mean_rep:.3f}")

    # Selección en base al criterio: máximo nº de puntos delante, y si empata, menor error reproyección
    scores.sort(key=lambda s: (-s[0], s[1]))
    best_n_ok, best_err, best_idx, R_best, t_best = scores[0]

    print("\nSelección:")
    print(f"Elegida hipótesis {best_idx+1}  (delante={best_n_ok}, reproy_mean={best_err:.3f})")
    T21 = np.eye(4)
    T21[:3,:3] = R_best
    T21[:3, 3] = t_best
    print("T_21 =\n", T21) # aprox = T_c2_w @ T_w_c1 (2.2)



    # 2.5 Results presentation
    # ########################

    x1  = np.loadtxt("x1Data.txt") # (2,N)
    x2  = np.loadtxt("x2Data.txt") # (2,N)

    # 1) Sampson error:
    # Evalua F, partiendo de la restricción epipolar ideal: r(x1, x2) = x2^T · F · x1 = 0 
    # Para r != 0, El Sampson error es una aproximación de primer orden del error geométrico mínimo
    # Aproximación de Gauss-Newton: del error geometrico mínimo: r^2 / (||F·x1||^2 + ||F^T·x2||^2)

    # Ejemplo con F21_test
    Sampson_err = sampson_error(F21_test, x1, x2) 
    summarize("Sampson err (px^2)", Sampson_err)

    # Interpretación: Valores 0-1 px^2 suele indicar buen ajuste; si es alto, hay outliers o F esta mal.

    # 2) Reprojection error:
    # Evalura (R, t) + triangulación
    # Mide la distancias en pixeles entre el punto observado en la imagen y el punto proyectado desde el 3D
    # Para un punto correspondiente a (x1, x2) y su 3D X, el error de reproyección es:
    # Sea Xh = [X, Y, Z, 1]^T
    #              [y1]
    # y = P · Xh = [y2]   --->   u_ = y1/y3, v_ = y2/y3    
    #              [y3]
    # e = sqrt((u-u_)^2 + (v-v_)^2)

    # Ejemplo con las P del 1.1 y la triangulación del 1.2
    P1 = K_c @ np.eye(3,4) @ np.linalg.inv(T_w_c1) 
    P2 = K_c @ np.eye(3,4) @ np.linalg.inv(T_w_c2) 

    X = triangulate_dlt(P1, P2, x1, x2)
    err1 = reprojection_errors(P1, X, x1)
    err2 = reprojection_errors(P2, X, x2)
    summarize("Reproj err view1 (px)", err1)
    summarize("Reproj err view2 (px)", err2)

    # Interpretación:
    #   Mediana < 1-2px suele indicar buen ajuste (con datos reales suele ser 1-3px si hay ruido)
    #   RMSE y p90 te dicen la cola del error (sensibles a outliers)



    # 3.1 Homography Definition
    # #########################
    # x2 = H_21·x1   --->   x1 y x2 son coordenadas homogéneas en píxeles (3x1)
    # T_w_i contiene Ri y ti   --->   Xi = Ri·Xw + ti  
    # Dato:   
    #   Pi_1 = (Pi_w.T @ T_w_1).T = [[0.0149,0.9483,0.3171,-1.7257]].T 
    #                               
    #                                       [ 0.0149 ]
    #                                       [ 0.9483 ]   --->   n1 es la nomral del plano en cam1 
    #   Plano en la cámara 1 = [n1, d1]^T = [ 0.3171 ]     
    #                                       [-1.7257 ]   --->   d1 es la distancia del plano al origen de cam1
    #   --->   n1^T·X1 + d1 = 0
    # 
    # Homografía en espacio normalizado y en píxeles para K1 = K2 = K:
    #   
    #   --->   H_21 = K· (R - (t·n1^T)/d1) ·K^(-1)   --->   R y t de T_21

    # 1) Obtener T_c2_c1
    T_c1_w = np.linalg.inv(T_w_c1)
    T_c2_w = np.linalg.inv(T_w_c2)
    T_c2_c1 = T_c2_w @ T_w_c1

    # 2) Obtener R y t
    R, t = T_c2_c1[:3,:3], T_c2_c1[:3,3]

    # 3) Calculamos H_21
    n1 = np.array([0.0149, 0.9483, 0.3171], dtype=float)   # shape (3,)
    d1 = -1.7257
    H_21 = K_c @ (R - np.outer(t, n1)/d1) @ np.linalg.inv(K_c)    
    # np.outer transforma el producto vectorial en un 3x3 para poder realizar a resta con la matriz R 

    # 4) Comprobar la homografía con los datos de suelo
    # Cargar datos
    x1 = np.loadtxt("x1FloorData.txt")   # Ya son homogeneas 3xN
    x2 = np.loadtxt("x2FloorData.txt")   # Ya son homogeneas 3xN

    # Proyección por homografía
    x2_hat = H_21 @ x1          

    # Normaliza homogéneas de forma vectorizada (evita división por cero (w = 0))
    den = x2_hat[2, :]
    eps = 1e-12
    valid = np.abs(den) > eps
    x2_hat[:, valid] /= den[valid]

    print("\nH_21:\n", H_21)
    print("\nEjemplo de proyección por homografía (píxeles):")
    for i in range(5):
        if valid[i]:
            print(f"x1 = {x1[:, i]}  --->  "
                f"x2 real = {x2[:, i]}   "
                f"x2_homog = {x2_hat[:, i]}")
        else:
            print(f"x1 = {x1[:, i]}  --->  x2_homog no definida (punto en el infinito)")



    # 3.2 Point transfer visualization
    # ################################

    fig3D = plt.figure()
    ax = plt.axes(projection='3d', adjustable='box')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    drawRefSystem(ax, np.eye(4), '-', 'W')
    ax.scatter(x2_hat[0, :], x2_hat[1, :], x2_hat[2, :], marker='.')
    # plotNumbered3DPoints(ax, X, 'r', (0.1, 0.1, 0.1))
    ax.scatter(x2[0, :], x2[1, :], x2[2, :], marker='x')
    plt.show(block=False)
    plt.waitforbuttonpress()


    # 3.3 Homography linear estimation from matches 
    # #############################################
    # x2 = H_21·x1   --->   x1 y x2 son coordenadas homogéneas en píxeles (3x1)   --->   xi = [ui, vi, 1]^T
    # 
    # Cada correspondencia aporta 2 filas a A   --->   A·h = 0   --->   h es el vector columna de H (9x1)
    # Por lo que se construye A con mínimo 4 puntos para que sea 8x9 y de rango 8
    # 
    # Después, se resuleve con SVD: A = U·S·V^T  --->  h es la última fila de V^T
    # 
    # Es necesarion normalizar con Hartley, igual que en el punto 2.3, con una transformación de similitud T
    # y, después, desnormalizar H = T2^(-1)·Hn·T1

    # 1) Normalizar puntos
    x1 = np.loadtxt("x1FloorData.txt")   # Ya son homogeneas 3xN
    x2 = np.loadtxt("x2FloorData.txt")   # Ya son homogeneas 3xN
    
    x1n, T1 = normalizePoints2D(x1[:2, :])  # Necesita 2xN
    x2n, T2 = normalizePoints2D(x2[:2, :])
    
    u1, v1 = x1n[0], x1n[1]
    u2, v2 = x2n[0], x2n[1]

    # 2) Construir A (2N x 9), vectorizado
    N = x1.shape[1]
    Z = np.zeros(N); O = np.ones(N)
    A_top = np.column_stack([-u1, -v1, -O,  Z,  Z,  Z,  u2*u1, u2*v1, u2])
    A_bot = np.column_stack([  Z,  Z,  Z, -u1, -v1, -O,  v2*u1, v2*v1, v2])
    A = np.vstack([A_top, A_bot])

    # 3) SVD(A) -> último vector singular
    _, _, Vt = np.linalg.svd(A)
    Hn = Vt[-1].reshape(3,3)  # homografía en espacio normalizado

    # 4) Des-normalizar: H = T2^{-1} Hn T1
    H = np.linalg.inv(T2) @ Hn @ T1
    # print(np.linalg.matrix_rank(H))  # rango 3
    

    # 5) Evaluación:
    # Transfer error:
    #   --->   Mide para cada correspondencia, cuanto se equivoca la H al transferir puntos de la imagen 1 a la imagen 2
    #   Proyecta: x2_hat = H·x1
    #   Normaliza homogéneas: u/w y v/w
    #   Error euclídeo entre x2_hat y x2:
    #       err   --->   e_i = || x2_hat_i - x2_i || (px). 
    #       ok    --->   máscara booleana de puntos válidos (w != 0) 

    e12, ok12 = transfer_error(H, x1, x2)
    summarize("Transfer error x1->x2", e12[ok12])

    # Symmetric transfer error:
    #   --->   Es una forma equilibrada de medir cuánto explica la H las correspondencias del mimso plano en ambas imágenes,
    #          evitando sesgos de medir solo una dirección (x1->x2).
    #   Para cada correspondendia (x1 <-> x2) en homogéneas:
    #      1. Adelante (1->2): e12, ok12 = transfer_error(H, x1, x2)
    #      2. Atrás   (2->1): e21, ok21 = transfer_error(H^{-1}, x2, x1)
    #      3. Error simétrico: e_i = (e12_i + e21_i) / 2   (cuando ambos son válidos)
    #      4. ok = ok12 & ok21
    
    esym, oks = symmetric_transfer_error(H, x1, x2)
    summarize("Symmetric transfer error", esym[oks])