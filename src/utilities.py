import numpy as np 
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import cv2

# ---------------------------------
# Redimension
#--------------------------------
def dlt_camera_matrix(x_3D, x_2D):
    """
    Docstring for dlt_camera_matrix
    
    param x_3D: Description (N, 3) 3D 
    param x_2D: Description (N, 2) 2D en pixeles 
    retorna P: (3,4) tal que x ~ P X

    """
    N = x_3D.shape[0]
    x_h = np.hstack([x_3D, np.ones((N,1))]) #(N, 4)


    A = []
    for i in range(N): 
        X_i = x_h[i]
        u, v = x_2D[i]
        A.append(np.hstack([np.zeros(4), -X_i,  v*X_i]))
        A.append(np.hstack([X_i, np.zeros(4), -u*X_i]))
    A = np.asarray(A)  # (2N, 12)

    _, _, Vt = np.linalg.svd(A)
    P = Vt[-1].reshape(3,4)
    return P

def project_points(P, X3D):
    X_h = np.hstack((X3D, np.ones((X3D.shape[0], 1))))
    x_proj = (P @ X_h.T).T
    x_proj = x_proj[:, :2] / x_proj[:, 2][:, None]
    return x_proj 



def resize_max_dim(img, max_dim=752):
    h, w = img.shape[:2]
    s = max_dim / max(h, w)
    if s >= 1.0:
        return img, (1.0, 1.0)  # no resize
    new_w = int(round(w * s))
    new_h = int(round(h * s))
    img_r = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return img_r, (s, s)  # sx, sy


def dlt_camera_matrix(X3D, x2D):
    """
    DLT for camera matrix estimation.
    
    Parameters
    ----------
    X3D : (N,3) array
        3D points in world coordinates.
    x2D : (N,2) array
        Corresponding 2D points in image coordinates (pixels).

    Returns
    -------
    P : (3,4) array
        Camera projection matrix such that x ~ P X.
    """
    assert X3D.shape[0] >= 6, "DLT needs at least 6 correspondences"
    assert X3D.shape[0] == x2D.shape[0], "3D–2D count mismatch"

    N = X3D.shape[0]
    A = []

    for i in range(N):
        X, Y, Z = X3D[i]
        u, v = x2D[i]

        # Two equations per correspondence
        A.append([ X, Y, Z, 1,  0, 0, 0, 0, -u*X, -u*Y, -u*Z, -u ])
        A.append([ 0, 0, 0, 0,  X, Y, Z, 1, -v*X, -v*Y, -v*Z, -v ])

    A = np.asarray(A)

    # Solve A p = 0 via SVD
    _, _, Vt = np.linalg.svd(A)
    p = Vt[-1]                 # smallest singular value
    P = p.reshape(3, 4)

    # Optional: scale normalization (safe)
    P = P / np.linalg.norm(P)

    return P


def unscale_points_to_original(x_resized, s): 
    """
    Docstring for unscale_points_to_original
    
    :param x_resized: (N, 2) points on reescaled image
    :param s: Description (scaled number used for superglue)
    """

    x_resized = np.asanyarray(x_resized, dtype=np.float64)
    return x_resized/ s

def ensure_2xN(X):
    X = np.asarray(X, dtype=float)
    return X if X.shape[0] == 2 else X.T

def aggregate(errors, how):
    if errors.size == 0: return np.inf
    if how == 'median': return float(np.median(errors))
    if how == 'mean': return float(np.mean(errors))
    if how == 'max': return float(np.max(errors))
    if how == 'rmse': return float(np.sqrt(np.mean(errors**2)))
    if how == 'p90': return float(np.percentile(errors, 90))
    raise ValueError(f"Aggregador no soportado: {how}")


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


def extract_matches (npz_path, min_score = None): 
    npz = np.load(npz_path)

    matches = npz ["matches"]
    keypoints0 = npz["keypoints0"]
    keypoints1 = npz["keypoints1"]

    mask = matches > -1
    idx = matches[mask]

    x1 = keypoints0[mask].astype (np.float32)
    x2 = keypoints1[idx].astype (np.float32)

    # 2) score/confidence (si existe)
    score = None
    for key in ["matching_scores0", "scores", "mconf"]:
        if key in npz.files:
            score = npz[key][mask].astype(np.float32)
            break

    # 3) filtro opcional por score
    if (min_score is not None) and (score is not None):
        keep = score >= min_score
        x1, x2 = x1[keep], x2[keep]
        score = score[keep]

    return x1, x2, score

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
    best_mask = None

    for (F, e) in evals:
        mask = (e < tau)
        n_inl = int(np.sum(mask))
        score = aggregate(e[mask], agg) if n_inl > 0 else np.inf
        # Criterio principal: más inliers válidos. Desempate: menor score.
        
        
        #  |--------------------| or |------------------------------------------------|
        if (n_inl > best_inliers):# or ((n_inl == best_inliers) and (score < best_score)):
            best_F, best_score, best_inliers, best_mask = F, score, n_inl, mask


    return best_F, best_score, best_inliers , best_mask



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

        idx = np.random.choice(x1.shape[0], 8, replace=False)  # 8 índices únicos
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

    F, e, inliers, mask = select_best_Fundamental(candidatesF, x1, x2, agg, tau)
    return F, e, inliers, mask

# For E matrix 

def essentialMatrix(fundamentalMatrix, K_intrinsict):
    e_Matrix = K_intrinsict.T @ fundamentalMatrix @ K_intrinsict
    U, S, Vt = np.linalg.svd(e_Matrix)
   
    S = [1,1,0]
    e_Matrix = U @ np.diag(S) @ Vt
    return e_Matrix

def decomposeEssentialMatrix(essentialMatrix): 
    U, _, Vt = np.linalg.svd(essentialMatrix)

    # Ensure a proper rotation (det(R) = +1)
    if np.linalg.det(U @ Vt) < 0:
        Vt = -Vt

    W = np.array([[0, -1, 0],
                  [1,  0, 0],
                  [0,  0, 1]])

    R1 = U @ W @ Vt
    R2 = U @ W.T @ Vt
    t = U[:, 2]

    # 4 possible solutions
    solutions = [
        (R1,  t),
        (R1, -t),
        (R2,  t),
        (R2, -t)
    ]
    return solutions

def triangulatePoints (x_1,x_2,P1,P2): 
    """
    Triangulate corresponding points between two images.
    Returns Nx3 array of 3D points (in homogeneous coordinates normalized).
    """
    n = x_1.shape[0]
    X = np.zeros((n, 3))

    for i in range(n):
        u1, v1 = x_1[i]
        u2, v2 = x_2[i]

        A = np.array([
            u1 * P1[2, :] - P1[0, :],
            v1 * P1[2, :] - P1[1, :],
            u2 * P2[2, :] - P2[0, :],
            v2 * P2[2, :] - P2[1, :]
        ])

        _, _, Vt = np.linalg.svd(A)
        X_h = Vt[-1]
        X_h /= X_h[3]
        X[i] = X_h[:3]
    return X

def selectCorrectPose(K, x_1, x_2, possible_solutions):
    """
    Evaluate the 4 (R,t) combinations and return the physically correct one.
    Criterion: Most triangulated points have positive depth (Z>0) in both cameras.
    """
    # First camera at origin
    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))

    best_count = 0
    best_pose = None
    best_points = None

    for i, (R, t) in enumerate(possible_solutions):
        P2 = K @ np.hstack((R, t.reshape(3, 1)))
        X = triangulatePoints(x_1, x_2, P1, P2)
        #X = triangulate_point(x_1, x_2, P1, P2)        

        # Check positive depth
        X_h = np.hstack((X, np.ones((X.shape[0], 1))))
        X_cam2 = (R @ X.T + t.reshape(3, 1)).T

        count = np.sum((X[:, 2] > 0) & (X_cam2[:, 2] > 0))
        print(f"Solution {i+1}: {count} points with positive depth")

        if count > best_count:
            best_count = count
            best_pose = (R, t)
            best_points = X

    print(f"\n Best solution: {best_count} points in front of both cameras.")
    return best_pose, best_points




#------- Plotting utilities -------

#####################################################################################
#
# MRGCV Unizar - Computer vision - Laboratory 2
#
# Title: Homography, Fundamental Matrix and Two View SfM
#
# Date: 5 September 2024
#
#####################################################################################
#
# Authors: Jesus Bermudez, Richard Elvira, Jose Lamarca, JMM Montiel
#
# Version: 2.0
#
#####################################################################################



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

if __name__ == '__main__':
    np.set_printoptions(precision=4,linewidth=1024,suppress=True)


    # Load ground truth
    T_w_c1 = np.loadtxt('D:/Dev_Space/python/Lab2/T_w_c1.txt')
    T_w_c2 = np.loadtxt('D:/Dev_Space/python/Lab2/T_w_c2.txt')

    K_c = np.loadtxt('D:/Dev_Space/python/Lab2/K_c.txt')
    X_w = np.loadtxt('D:/Dev_Space/python/Lab2/X_w.txt')

    x1 = np.loadtxt('D:/Dev_Space/python/Lab2/x1Data.txt')
    x2 = np.loadtxt('D:/Dev_Space/python/Lab2/x2Data.txt')


    ##Plot the 3D cameras and the 3D points
    fig3D = plt.figure(3)

    ax = plt.axes(projection='3d', adjustable='box')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    drawRefSystem(ax, np.eye(4, 4), '-', 'W')
    drawRefSystem(ax, T_w_c1, '-', 'C1')
    drawRefSystem(ax, T_w_c2, '-', 'C2')

    ax.scatter(X_w[0, :], X_w[1, :], X_w[2, :], marker='.')
    #plotNumbered3DPoints(ax, X_w, 'r', (0.1, 0.1, 0.1)) # For plotting with numbers (choose one of the both options)

    #Matplotlib does not correctly manage the axis('equal')
    xFakeBoundingBox = np.linspace(0, 4, 2)
    yFakeBoundingBox = np.linspace(0, 4, 2)
    zFakeBoundingBox = np.linspace(0, 4, 2)
    plt.plot(xFakeBoundingBox, yFakeBoundingBox, zFakeBoundingBox, 'w.')
    print('Close the figure to continue. Left button for orbit, right button for zoom.')
    plt.show()

    ## 2D plotting example
    img1 = cv2.cvtColor(cv2.imread('D:/Dev_Space/python/Lab2/image1.png'), cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(cv2.imread('D:/Dev_Space/python/Lab2/image2.png'), cv2.COLOR_BGR2RGB)


    plt.figure(1)
    plt.imshow(img1, cmap='gray', vmin=0, vmax=255)
    plt.plot(x1[0, :], x1[1, :],'rx', markersize=10)
    plotNumberedImagePoints(x1, 'r', (10,0)) # For plotting with numbers (choose one of the both options)
    plt.title('Image 1')
    plt.draw()  # We update the figure display
    print('Click in the image to continue...')
    plt.waitforbuttonpress()

    plt.figure(2)
    plt.imshow(img2, cmap='gray', vmin=0, vmax=255)
    plt.plot(x2[0, :], x2[1, :],'rx', markersize=10)
    plotNumberedImagePoints(x2, 'r', (10,0)) # For plotting with numbers (choose one of the both options)
    plt.title('Image 2')
    plt.draw()  # We update the figure display
    print('Click in the image to continue...')
    plt.waitforbuttonpress()



    