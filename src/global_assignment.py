
import utilities
import numpy as np
import cv2
import matplotlib.pyplot as plt

#--------
#imports
#--------

# Paths
aux_path = "D:/Dev_Space/python/Gloabal_proyect/src/auxiliary_files/"

#Camera properties
k_camera = np.loadtxt (aux_path + "k_camera.txt")  # Matriz de cámara intrínseca

#Camera_images
image1 = aux_path + "und_seminario_new_01.jpg"  # Imagen de la cámara A
image2 = aux_path + "und_seminario_new_02.jpg"

#Points and matches loading
#Puntos entre la camara A y camara B con 19.5 grados de parallax
matches_4_ab = aux_path + "und_seminario_new_01_und_seminario_new_02_matches.npz"
matches_4_old_a = aux_path + "seminario1_und_seminario_new_01_matches.npz"


def P_to_Twc_with_opencv(P):
    """
    Convierte P (3x4) a T_w_c usando OpenCV:
    P ~ K [R | t], y decomposeProjectionMatrix devuelve C (centro de cámara).
    """
    K, R, t_h, _, _, _, _ = cv2.decomposeProjectionMatrix(P)

    # Normaliza K para lectura humana (K[2,2]=1)
    K = K / K[2, 2]

    # Centro de cámara en mundo (3,)
    C = (t_h[:3] / t_h[3]).reshape(3)

    # Traslación extrínseca correcta: t = -R*C
    t = -R @ C

    T_w_c = utilities.ensamble_T(R, t)
    return T_w_c, K, R, t, C

def plot_3_cameras_and_points(X3D, best_R, best_t, P_old=None):
    X3D = np.asarray(X3D)

    # --- C1 como mundo/origen
    T_w_c1 = np.eye(4)

    # --- C2: tú tienes (R,t) como C1 -> C2 (x_cam2 = R x_cam1 + t)
    # Pose de la cámara (frame) en mundo: T_w_c2 = inv(T_c2_c1)
    T_c2_c1 = utilities.ensamble_T(best_R, best_t)
    T_w_c2 = np.linalg.inv(T_c2_c1)

    # --- Cold desde P_old si lo das
    T_w_cold = None
    if P_old is not None:
        T_w_cold, K_old, R_old, t_old, C_old = P_to_Twc_with_opencv(P_old)
        print("K_old (normalizada):\n", K_old)
        print("C_old (centro cam en mundo):", C_old)
        print("R_old:\n", R_old)
        print("t_old (extrínseca):", t_old)

    # --- Plot
    fig = plt.figure(figsize=(20,20))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_title("Scene + C1, C2, Cold (up to scale)")

    # frames
    utilities.drawRefSystem(ax, T_w_c1, '-', 'C1/W')
    utilities.drawRefSystem(ax, T_w_c2, '-', 'C2')

    if T_w_cold is not None:
        utilities.drawRefSystem(ax, T_w_cold, '-', 'Cold')

    # puntos
    ax.scatter(X3D[:,0], X3D[:,1], X3D[:,2], s=5)

    # ejes “igualados”
    mins = np.min(X3D, axis=0)
    maxs = np.max(X3D, axis=0)
    span = np.max(maxs - mins)
    mid = (maxs + mins) / 2
    ax.set_xlim(mid[0]-span/2, mid[0]+span/2)
    ax.set_ylim(mid[1]-span/2, mid[1]+span/2)
    ax.set_zlim(mid[2]-span/2, mid[2]+span/2)

    plt.show()


def plot_scene_with_cameras(X3D, R_best, t_best, axis_len=0.2):
    X3D = np.asarray(X3D)

    # Cámara 1 como mundo/origen:
    T_w_c1 = np.eye(4)

    # OJO: aquí asumimos que tu [R_best|t_best] es P2 = K [R|t]
    # y por tanto X_cam2 = R X_cam1 + t (como tú lo estás usando en cheirality).
    # Eso significa que (R,t) es transform de C1 -> C2.
    # Entonces la pose de la cámara 2 "en mundo (C1)" sería:
    # T_c2_c1 = [R|t]
    # T_w_c2 = inv(T_c2_c1)
    T_c2_c1 = utilities.ensamble_T(R_best, t_best)
    T_w_c2 = np.linalg.inv(T_c2_c1)

    fig = plt.figure(figsize=(8,7))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_title("Two-view reconstruction (up to scale)")

    # Dibuja frames
    utilities.drawRefSystem(ax, T_w_c1, '-', 'C1/W')
    utilities.drawRefSystem(ax, T_w_c2, '-', 'C2')

    # Nube de puntos
    ax.scatter(X3D[:,0], X3D[:,1], X3D[:,2], s=5)

    # “Equal-ish” para que no se aplaste
    mins = np.min(X3D, axis=0)
    maxs = np.max(X3D, axis=0)
    span = np.max(maxs - mins)
    mid = (maxs + mins) / 2
    ax.set_xlim(mid[0]-span/2, mid[0]+span/2)
    ax.set_ylim(mid[1]-span/2, mid[1]+span/2)
    ax.set_zlim(mid[2]-span/2, mid[2]+span/2)

    plt.show()

# uso:
# plot_scene_with_cameras(X3D, R_best, t_best)



if __name__ == "__main__":
   # Check of matching points extract
    x1, x2, score = utilities.extract_matches (matches_4_ab, min_score=0.5)
    x1_old_new, x2_old_new , _ = utilities.extract_matches(matches_4_old_a, min_score= 0.5)
    print("x1:", x1.shape, "x2:", x2.shape, "score:", None if score is None else score.shape)

    #Ransac F estimation
    np.random.seed(42)
    n_hyp = 50000
    tau_px = 3.0
    F_best, best_score, n_inliers, inlier_mask = utilities.RANSAC_F(n_hyp, x1, x2, agg='median', tau=tau_px)
    _, _, n_inliers_old_new, inlier_mask_old_new = utilities.RANSAC_F(n_hyp, x1_old_new, x2_old_new, agg='median', tau=tau_px)

    print ("\n=== Ranscac Fundamental Matrix Estimation ===")
    print("Best Fundamental Matrix:\n", F_best)
    print("Number of inliers:", n_inliers)
    
    #print("Best score (px):", best_score)
    
    #print("Inlier mask:", inlier_mask)

    #Filter points inliers 

    x1_inliers = x1[inlier_mask]
    x2_inliers = x2[inlier_mask]

    x1_inliers_old_new = x1_old_new[inlier_mask_old_new]
    x2_inliers_old_new = x2_old_new[inlier_mask_old_new]




    #print("x1 inliers:", x1_inliers.shape, "x2 inliers:", x2_inliers.shape)

    #img1SR = cv2.cvtColor(cv2.imread(aux_path + "und_seminario_new_01.jpg"), cv2.COLOR_BGR2RGB)
    #img2SR = cv2.cvtColor(cv2.imread(aux_path + "und_seminario_new_02.jpg"), cv2.COLOR_BGR2RGB)

    img1= cv2.cvtColor(cv2.imread(aux_path + "und_seminario_new_01.jpg"), cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(cv2.imread(aux_path + "und_seminario_new_02.jpg"), cv2.COLOR_BGR2RGB)
    

    h1, w1 = img1.shape[:2]

    print (h1)

#752 because of the rescale factor
    s1 = 752 / max(h1, w1)
    s1 = min(1.0, s1)

    h2, w2 = img2.shape[:2]
    s2 = 752 / max(h2, w2)
    s2 = min(1.0, s2)

    # 3) Pasa puntos a coordenadas originales (para usar K original)
    x1_inliers_orig = utilities.unscale_points_to_original(x1_inliers, s1)
    x2_inliers_orig = utilities.unscale_points_to_original(x2_inliers, s2)

    x2_inliers_old_new_original = utilities.unscale_points_to_original(x2_inliers_old_new, s1)


    # 4) Visualiza en imagen ORIGINAL (sin resize)
    fig, ax = plt.subplots(1,2, figsize=(12,5))
    ax[0].imshow(img1); ax[0].scatter(x1[:,0], x1[:,1], s=20, marker='x'); ax[0].axis("off")
    ax[1].imshow(img2); ax[1].scatter(x2[:,0], x2[:,1], s=20, marker='x'); ax[1].axis("off")
    plt.show()

    fig, ax = plt.subplots(1,2, figsize=(12,5))
    ax[0].imshow(img1); ax[0].scatter(x1_inliers_orig[:,0], x1_inliers_orig[:,1], s=20, marker='x'); ax[0].axis("off")
    ax[1].imshow(img2); ax[1].scatter(x2_inliers_orig[:,0], x2_inliers_orig[:,1], s=20, marker='x'); ax[1].axis("off")
    plt.show()


    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    ax[0].imshow(img1)
    ax[0].scatter(x2_inliers_old_new_original[:,0], x2_inliers_old_new_original[:,1], s=20, marker='x')
    ax[0].set_title(f"Inliers - A")
    ax[0].axis("off")
    plt.show()
    
    ## Calculamos E con X resized: 
    n_hyp = 2000
    F_best, _, _, _ = utilities.RANSAC_F(n_hyp, x1= x1_inliers_orig, x2 = x2_inliers_orig, agg='median', tau=1.0)

    E = utilities.essentialMatrix(F_best, k_camera)
    print ("E: \n", E)

    solutions = utilities.decomposeEssentialMatrix(E)

    for i, (R, t) in enumerate(solutions, 1):
        print(f"\nCandidate {i}")
        print("det(R):", np.linalg.det(R))
        print("R:\n", R)
        print("t:\n", t)

    x1_use, x2_use = x1_inliers_orig, x2_inliers_orig   # o usa x1_inliers_orig, x2_inliers_orig si prefieres

    (best_R, best_t), X3D = utilities.selectCorrectPose(k_camera, x1_use, x2_use, solutions)

    print("\n=== Step 7.3: Best pose ===")
    print("R_best:\n", best_R)
    print("t_best:\n", best_t)

    # Matrices de proyección finales (en pixeles, con K original)
    P1 = k_camera @ np.hstack((np.eye(3), np.zeros((3,1))))
    P2 = k_camera @ np.hstack((best_R, best_t.reshape(3,1)))




    print("\nP1:\n", P1)
    print("\nP2:\n", P2)
    print("\nTriangulated X3D:", X3D.shape)

    print("img1 shape:", img1.shape, "img2 shape:", img2.shape)
    print("K:\n", k_camera)
    plot_scene_with_cameras(X3D, best_R, best_t)

    X3D_h = np.hstack((X3D, np.ones((X3D.shape[0],1))))
    proj1_h = (P1 @ X3D_h.T).T
    x1_3Dproj = proj1_h[:, :2]/proj1_h[:, 2][:, None]
    print("x1_3Dproj shape:", x1_3Dproj.shape)

    x1_proj = utilities.project_points(P1, X3D)
    
    # Using RMSe 
    X3D_pnp = []
    x_old_pnp = []

    threshold_px = 4.0  # tolerancia geométrica

    for i in range(x2_inliers_old_new_original.shape[0]):
        x_base = x2_inliers_old_new_original[i]   # punto BASE (old-base)

        # Distancia reproyectada
        dists = np.linalg.norm(x1_proj - x_base, axis=1)
        idx = np.argmin(dists)

        if dists[idx] < threshold_px:
            X3D_pnp.append(X3D[idx])
            x_old_pnp.append(x1_inliers_old_new[i])  # punto OLD

    X3D_pnp =np.array(X3D_pnp)
    x_old_pnp = np.array(x_old_pnp)
    
    print("PNP correspondance ", X3D_pnp.shape)
    P = utilities.dlt_camera_matrix(X3D_pnp, x_old_pnp)
    print("P old matrix: ", P)

    K_old, R_old, t_old, _, _, _, _ = cv2.decomposeProjectionMatrix(P)
    K_old = K_old/K_old[2,2]
    t_old = (t_old[:3] / t_old[3]).reshape(3)
    X_h = np.hstack((X3D_pnp, np.ones((X3D_pnp.shape[0],1))))
    proj = (P @ X_h.T).T
    x_proj = proj[:, :2] / proj[:, 2][:, None]

    rmse = np.sqrt(np.mean(np.sum((x_proj - x_old_pnp)**2, axis=1)))
    print("Reprojection RMSE (px):", rmse)
    print ("K old:", K_old)
    print ("R old:", R_old)
    print ("T old:", t_old)
    #plot_scene_with_cameras(X3D, R_old, t_old)
    plot_3_cameras_and_points(X3D, best_R, best_t, P_old=P)




   
   
   
   
    #img1, (sx1, sy1) = utilities.resize_max_dim(img1SR, max_dim=752)
    #img2, (sx2, sy2) = utilities.resize_max_dim(img2SR, max_dim=752)
"""
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))



    ax[0].imshow(img1)
    ax[0].scatter(x1_inliers[:,0], x1_inliers[:,1], s=20, marker='x')
    ax[0].set_title(f"Inliers - A")
    ax[0].axis("off")

    ax[1].imshow(img2)
    ax[1].scatter(x2_inliers[:,0], x2_inliers[:,1], s=20, marker='x')
    ax[1].set_title(f"Inliers- B")
    ax[1].axis("off")

    plt.tight_layout()
    plt.show()

    
    
    
    
    
    
    
    

    
    
    d = x2_inliers - x1_inliers          # (N,2)
    dist = np.linalg.norm(d, axis=1)     # desplazamiento por match (px)
    print("mean |dx|:", np.mean(np.abs(d[:,0])))
    print("mean |dy|:", np.mean(np.abs(d[:,1])))
    print("mean disp:", np.mean(dist))
    print("median disp:", np.median(dist))
    print("max disp:", np.max(dist))

    plt.figure(figsize=(6,4))
    plt.hist(dist, bins=30)
    plt.title("Distribución de desplazamiento |x2 - x1| (px)")
    plt.xlabel("px"); plt.ylabel("count")
    plt.show()


    d = x2_inliers - x1_inliers
    plt.figure(figsize=(6,6))
    plt.quiver(x1_inliers[:,0], x1_inliers[:,1], d[:,0], d[:,1], angles='xy', scale_units='xy', scale=1)
    plt.gca().invert_yaxis()
    plt.axis('equal')
    plt.title("Vectores: x1 -> x2 (inliers)")
    plt.xlabel("u (px)"); plt.ylabel("v (px)")
    plt.show()

    x1_plot = x1_inliers
    x2_plot = x2_inliers
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    ax[0].scatter(x1_plot[:,0], x1_plot[:,1], marker='x')
    ax[0].set_title("Points in A")
    ax[0].set_xlabel("u (px)")
    ax[0].set_ylabel("v (px)")
    ax[0].axis("equal")

    ax[1].scatter(x2_plot[:,0], x2_plot[:,1], marker='x')
    ax[1].set_title("Points in B")
    ax[1].set_xlabel("u (px)")
    ax[1].set_ylabel("v (px)")
    ax[1].axis("equal")

    plt.tight_layout()
    plt.show()

    """