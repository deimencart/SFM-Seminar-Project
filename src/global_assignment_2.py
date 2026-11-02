"""
Clean SfM Pipeline - Building Use Case
Computer Vision Course Assignment

Author: [Your Name]
Date: December 2024

Pipeline Steps:
1. Load SuperGlue matches and visualize RAW (no transformations)
2. RANSAC for Fundamental Matrix with inliers/outliers visualization

3. Essential Matrix and pose recovery
4. Triangulation and 3D reconstruction
5. PnP for old camera localization


Ransac for P
Bundleadjustemn aqui
Recuerda que hay que rescalar la imagen como te dijjon el profesor para evitar probelmas, hazlom con opencv
"""

import numpy as np
import cv2
import utilities2


# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """Configuration parameters for the pipeline"""
    
    # Paths
    DATA_PATH = "D:/Dev_Space/python/Gloabal_proyect/src/auxiliary_files/"
    
    # Images
    IMG1 = DATA_PATH + "und_seminario_new_04.jpg"
    IMG2 = DATA_PATH + "und_seminario_new_01.jpg"
    IMG_OLD = DATA_PATH + "seminario1_fixed.jpg"  
    
    # Matches (generados por SuperGlue)
    MATCHES_1_2 = DATA_PATH + "und_seminario_new_04_und_seminario_new_01_matches.npz"
    MATCHES_OLD_1 = DATA_PATH + "seminario1_fixed_und_seminario_new_04_matches.npz"
    
    # Camera calibration
    K_CAMERA = DATA_PATH + "k_camera.txt"
    
    # SuperGlue parameters (los que usaste al correr SuperGlue)
    SUPERGLUE_RESIZE = None  # Sin resize - full resolution
    
    # RANSAC parameters
    RANSAC_ITERATIONS = 100000
    RANSAC_THRESHOLD = 1.5  # pixels
    RANSAC_MIN_SCORE = 0.7 #_minimum SuperGlue score
    
    # Random seed for reproducibility
    RANDOM_SEED = 42

def main():
    """Main pipeline execution"""
    
    print("\n" + "="*70)
    print("Computer Vision Course Assignment")
    print("="*70)
    
    # Load configuration
    cfg = Config()
    
    # =========================================================================
    # STEP 1: LOAD IMAGES AND MATCHES
    # =========================================================================
    print("\n" + "="*70)
    print("LOADING DATA")
    print("="*70)
    
    # Load images
    img1 = cv2.cvtColor(cv2.imread(cfg.IMG1), cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(cv2.imread(cfg.IMG2), cv2.COLOR_BGR2RGB)

    imgOld = cv2.cvtColor(cv2.imread(cfg.IMG_OLD), cv2.COLOR_BGR2RGB)
    
    print(f"Image 1: {cfg.IMG1.split('/')[-1]} - Shape: {img1.shape}")
    print(f"Image 2: {cfg.IMG2.split('/')[-1]} - Shape: {img2.shape}")
    print(f"Image Old: {cfg.IMG_OLD.split('/')[-1]} - Shape: {imgOld.shape}")
    
    # Load camera calibration
    K = np.loadtxt(cfg.K_CAMERA)
    print(f"\nCamera Calibration Matrix K:")
    print(K)
    
    # Load SuperGlue matches (RAW - exactly as detected)
    kpts1, kpts2, scores = utilities2.load_superglue_matches(
        cfg.MATCHES_1_2, 
        min_score=cfg.RANSAC_MIN_SCORE
    )

    kptsOld1, kptsOld2, scoresOld = utilities2.load_superglue_matches(
        cfg.MATCHES_OLD_1, 
        min_score=cfg.RANSAC_MIN_SCORE
    )


    
    # Visualize RAW matches (no transformations)
    print(f"\n{'='*70}")
    print("VISUALIZING RAW MATCHES (Step 1)")
    print(f"{'='*70}")
    utilities2.visualize_raw_matches(kpts1, kpts2, img1, img2, scores, 
                          max_display=160, 
                          title="Step 1: RAW Matches from SuperGlue (No Transformations)")
    

    utilities2.visualize_raw_matches(kptsOld1, kptsOld2, imgOld, img1, scores, 
                          max_display=160, 
                          title="Step 1: RAW Matches from SuperGlue (OLD WITH NEW ONE)")
    

    # =========================================================================
    # STEP 2: RANSAC FOR FUNDAMENTAL MATRIX
    # =========================================================================
    print("\n" + "="*70)
    print("RANSAC FOR FUNDAMENTAL MATRIX")
    print("="*70)
    
    F_best, inlier_mask = utilities2.ransac_fundamental(
        kpts1, kpts2,
        n_iterations=cfg.RANSAC_ITERATIONS,
        threshold=cfg.RANSAC_THRESHOLD,
        seed=cfg.RANDOM_SEED
    )
    
    # Visualize inliers vs outliers
    print(f"\n{'='*70}")
    print("VISUALIZING INLIERS vs OUTLIERS (Step 2)")
    print(f"{'='*70}")
    utilities2.visualize_inliers_outliers(kpts1, kpts2, inlier_mask, img1, img2,
                                max_display_inliers=200,
                                max_display_outliers=100,
                                show_numbers=True,  
                                n_numbered=100)
    
    

  # =========================================================================
    # ESSENTIAL MATRIX AND POSE RECOVERY
    # =========================================================================
    print("\n" + "="*70)
    print("ESSENTIAL MATRIX AND POSE RECOVERY")
    print("="*70)
    
    # Get inlier points only
    x1_inliers = kpts1[inlier_mask]
    x2_inliers = kpts2[inlier_mask]
    
    # Compute Essential matrix from F and K
    E = utilities2.fundamental_to_essential(F_best, K)
    print("\nEssential Matrix E:")
    print(E)
    print(f"Rank of E: {np.linalg.matrix_rank(E)}")
    
    # Decompose E into 4 possible solutions
    solutions = utilities2.decompose_essential_matrix(E)
    
    print("\n4 Possible (R, t) solutions:")
    for i, (R, t) in enumerate(solutions, 1):
        print(f"\nSolution {i}:")
        print(f"  det(R) = {np.linalg.det(R):.6f} (should be +1)")
        print(f"  ||t|| = {np.linalg.norm(t):.6f} (should be 1)")
    
    # Select correct pose by checking cheirality
    (R_best, t_best), X3D = utilities2.select_correct_pose(K, x1_inliers, x2_inliers, solutions)
    
    print("\nBest pose selected:")
    print(f"R:\n{R_best}")
    print(f"t: {t_best}")
    print(f"Triangulated {len(X3D)} 3D points")
    
    #=========================================================================
    # COORDINATE FRAME CONVENTION
    # =========================================================================
    # Camera 1: WORLD ORIGIN (anchored at identity)
    #   - All 3D points are expressed in Camera 1's coordinate frame
    #   - T_w_c1 = I (identity)
    #   - P1 = K [I | 0]
    #
    # Camera 2: Relative pose to Camera 1
    #   - (R, t) represents transformation FROM C1 TO C2
    #   - x_c2 = R @ x_c1 + t
    #   - P2 = K [R | t]
    # =========================================================================
    
    # Build projection matrices
    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))  # Camera 1 at origin
    P2 = K @ np.hstack((R_best, t_best.reshape(3, 1)))  # Camera 2 relative to C1
    
    print(f"\nCoordinate frame setup:")
    print(f"  Camera 1: Fixed at world origin [I | 0]")
    print(f"  Camera 2: Relative pose (R, t) from Camera 1")
    print(f"  3D points: Expressed in Camera 1 frame")
    
    print(f"\nProjection matrices:")
    print(f"P1 = K [I | 0]:\n{P1}")
    print(f"\nP2 = K [R | t]:\n{P2}")
    
    # Compute reprojection errors
    errors1, x1_reproj = utilities2.compute_reprojection_error(X3D, x1_inliers, P1)
    errors2, x2_reproj = utilities2.compute_reprojection_error(X3D, x2_inliers, P2)
    
    print(f"\n{'='*70}")
    print("REPROJECTION ERROR ANALYSIS")
    print(f"{'='*70}")
    print(f"\nCamera 1:")
    print(f"  RMSE:   {np.sqrt(np.mean(errors1**2)):.3f} px")
    print(f"  Mean:   {np.mean(errors1):.3f} px")
    print(f"  Median: {np.median(errors1):.3f} px")
    print(f"  Max:    {np.max(errors1):.3f} px")
    
    print(f"\nCamera 2:")
    print(f"  RMSE:   {np.sqrt(np.mean(errors2**2)):.3f} px")
    print(f"  Mean:   {np.mean(errors2):.3f} px")
    print(f"  Median: {np.median(errors2):.3f} px")
    print(f"  Max:    {np.max(errors2):.3f} px")
    
    # Depth statistics
    depths = X3D[:, 2]
    print(f"\n{'='*70}")
    print("3D POINT STATISTICS")
    print(f"{'='*70}")
    print(f"Depth (Z coordinate):")
    print(f"  Mean:   {np.mean(depths):.3f}")
    print(f"  Median: {np.median(depths):.3f}")
    print(f"  Min:    {np.min(depths):.3f}")
    print(f"  Max:    {np.max(depths):.3f}")
    print(f"  Std:    {np.std(depths):.3f}")
    print(f"\nPoints behind camera 1: {np.sum(depths < 0)}")
    
    # Transform to camera 2 and check
    X_cam2 = (R_best @ X3D.T + t_best.reshape(3, 1)).T
    depths2 = X_cam2[:, 2]
    print(f"Points behind camera 2: {np.sum(depths2 < 0)}")
    
    # Visualize reprojection
    print(f"\n{'='*70}")
    print("VISUALIZING REPROJECTION ERRORS")
    print(f"{'='*70}")
    utilities2.plot_reprojection_comparison(img1, img2, x1_inliers, x2_inliers, 
                                 x1_reproj, x2_reproj, max_display=100)
    
    
    # Visualize 3D reconstruction
    print(f"\n{'='*70}")
    print("VISUALIZING 3D RECONSTRUCTION")
    print(f"{'='*70}")
    utilities2.plot_3d_reconstruction(X3D, R_best, t_best, 
                          title="Step 4: 3D Reconstruction from Two Views")
    
    print("\nReconstruction summary:")
    print(f"  - Total matches: {len(kpts1)}")
    print(f"  - Inliers: {np.sum(inlier_mask)} ({100*np.sum(inlier_mask)/len(kpts1):.1f}%)")
    print(f"  - 3D points: {len(X3D)}")
    print(f"  - Mean reprojection error: {(np.mean(errors1) + np.mean(errors2))/2:.3f} px")

    # =========================================================================
    # STEP 5: PnP FOR OLD CAMERA LOCALIZATION WITH RANSAC
    # =========================================================================
    print("\n" + "="*70)
    print("PnP FOR OLD CAMERA (DLT + RANSAC)")
    print("="*70)
    
    # 5.1 - Find 2D-3D correspondences
    print("\nFinding 2D-3D correspondences...")
    indices_old, indices_3D = utilities2.find_common_points_2D_3D(
        kptsOld2,     # Points in new_04 from old-new match
        x1_inliers,   # Inlier points in new_04 from reconstruction
        tolerance=1.0
    )
    
    print(f"Found {len(indices_old)} common 2D-3D correspondences")
    
    if len(indices_old) < 6:
        print("ERROR: Need at least 6 correspondences for PnP")
        return
    
    # Extract corresponding points
    x2D_old = kptsOld1[indices_old]  # 2D in OLD image
    X3D_common = X3D[indices_3D]      # Corresponding 3D points
    
    print(f"  2D points in old image: {x2D_old.shape}")
    print(f"  3D points: {X3D_common.shape}")
    
    # 5.2 - RANSAC for Projection Matrix
    P_old, inlier_mask_pnp = utilities2.ransac_projection_matrix(
        X3D_common,
        x2D_old,
        n_iterations=cfg.RANSAC_ITERATIONS,
        threshold=5.0,
        seed=cfg.RANDOM_SEED
    )
    
    # 5.3 - Decompose P → K, R, t
    print(f"\n{'='*70}")
    print("DECOMPOSING PROJECTION MATRIX")
    print(f"{'='*70}")
    
    K_old, R_old, t_old = utilities2.decompose_projection(P_old)
    
    print(f"\nRecovered Camera Parameters:")
    print(f"\nIntrinsic Matrix K_old:")
    print(K_old)
    print(f"\nRotation Matrix R_old:")
    print(R_old)
    print(f"  det(R) = {np.linalg.det(R_old):.6f} (should be +1)")
    print(f"\nTranslation t_old:")
    print(t_old)
    
    # Compare with ground truth K
    print(f"\n{'='*70}")
    print("COMPARISON WITH GROUND TRUTH K")
    print(f"{'='*70}")
    print(f"\nGround truth K:")
    print(K)
    print(f"\nRecovered K_old:")
    print(K_old)
    print(f"\nDifference: {np.linalg.norm(K - K_old):.3f}")
    
    # 5.4 - Reprojection errors
    errors_pnp, x2D_reproj = utilities2.compute_reprojection_error(
        X3D_common, x2D_old, P_old
    )
    
    print(f"\n{'='*70}")
    print("PnP REPROJECTION ERROR")
    print(f"{'='*70}")
    print(f"All points:")
    print(f"  RMSE:   {np.sqrt(np.mean(errors_pnp**2)):.3f} px")
    print(f"  Mean:   {np.mean(errors_pnp):.3f} px")
    print(f"  Median: {np.median(errors_pnp):.3f} px")
    print(f"  Max:    {np.max(errors_pnp):.3f} px")
    
    inlier_errors = errors_pnp[inlier_mask_pnp]
    print(f"\nInliers only:")
    print(f"  RMSE:   {np.sqrt(np.mean(inlier_errors**2)):.3f} px")
    print(f"  Mean:   {np.mean(inlier_errors):.3f} px")
    print(f"  Median: {np.median(inlier_errors):.3f} px")
    print(f"  Max:    {np.max(inlier_errors):.3f} px")
    
    # 5.5 - Visualize PnP
    print(f"\n{'='*70}")
    print("VISUALIZING PnP RESULTS")
    print(f"{'='*70}")
    
    utilities2.visualize_pnp_reprojection(
        imgOld, x2D_old, x2D_reproj, inlier_mask_pnp,
        max_display=100,
        title="Step 5: PnP - Old Camera (DLT + RANSAC)"
    )
    
    # 5.6 - 3D visualization with 3 cameras
    print(f"\n{'='*70}")
    print("3D RECONSTRUCTION - ALL 3 CAMERAS")
    print(f"{'='*70}")
    
    X3D_inliers_pnp = X3D_common[inlier_mask_pnp]
    utilities2.plot_3d_reconstruction_three_cameras(
        X3D_inliers_pnp, R_best, t_best, R_old, t_old,
        title="Complete 3D Reconstruction: 3 Cameras"
    )
# 5.3 - Decompose P → K, R, t
    print(f"\n{'='*70}")
    print("DECOMPOSING PROJECTION MATRIX")
    print(f"{'='*70}")
    
    K_old, R_old_raw, t_old_raw = utilities2.decompose_projection(P_old)
    
    print(f"\nRecovered Camera Parameters:")
    print(f"\nIntrinsic Matrix K_old:")
    print(K_old)
    
    print(f"\nRotation Matrix R_old:")
    print(R_old_raw)
    print(f"  det(R) = {np.linalg.det(R_old_raw):.6f}")
    
    print(f"\nTranslation t_old (world→camera):")
    print(t_old_raw)
    
    # =========================================================================
    # CRITICAL VERIFICATION: Check camera position
    # =========================================================================
    print(f"\n{'='*70}")
    print("CAMERA POSITION VERIFICATION")
    print(f"{'='*70}")
    
    # Invert to get camera position in world
    t_old_world = -R_old_raw.T @ t_old_raw
    
    print(f"\nCamera position in world (Camera 1 frame):")
    print(f"  t_world = {t_old_world}")
    print(f"  Distance from origin: {np.linalg.norm(t_old_world):.3f}")
    
    # Compare with Camera 2 position
    t2_world = -R_best.T @ t_best
    print(f"\nFor reference, Camera 2 position:")
    print(f"  t_world = {t2_world}")
    print(f"  Distance from origin: {np.linalg.norm(t2_world):.3f}")
    
    # =========================================================================
    # SANITY CHECKS
    # =========================================================================
    print(f"\n{'='*70}")
    print("SANITY CHECKS")
    print(f"{'='*70}")
    
    # 1. Check if points are in front of camera
    X_sample = X3D_common[inlier_mask_pnp][0]
    X_in_old = R_old_raw @ X_sample + t_old_raw
    
    print(f"\n1. Point depth check:")
    print(f"   Sample 3D point: {X_sample}")
    print(f"   In OLD camera frame: {X_in_old}")
    print(f"   Depth (Z): {X_in_old[2]:.3f}")
    
    if X_in_old[2] > 0:
        print(f"  Point is IN FRONT of camera")
    else:
        print(f"  Point is BEHIND camera - POSE IS WRONG!")
    
    # 2. Check all inlier depths
    X3D_inliers = X3D_common[inlier_mask_pnp]
    X_in_old_all = (R_old_raw @ X3D_inliers.T + t_old_raw.reshape(3, 1)).T
    depths = X_in_old_all[:, 2]
    n_positive = np.sum(depths > 0)
    
    print(f"\n2. All inliers depth check:")
    print(f"   Points in front: {n_positive}/{len(depths)} ({100*n_positive/len(depths):.1f}%)")
    
    if n_positive == len(depths):
        print(f" ALL points in front - POSE IS CORRECT")
    elif n_positive > 0.9 * len(depths):
        print(f" Most points in front - pose is mostly correct")
    else:
        print(f" Many points behind - POSE IS WRONG!")
    
    # 3. Check reprojection makes sense
    print(f"\n3. Reprojection check:")
    errors_inliers = errors_pnp[inlier_mask_pnp]
    print(f"   RMSE: {np.sqrt(np.mean(errors_inliers**2)):.2f} px")

    
    # 4. Check camera positions make geometric sense
    print(f"\n4. Geometric configuration:")
    
    # Distance between cameras
    dist_12 = np.linalg.norm(t2_world)
    dist_13 = np.linalg.norm(t_old_world)
    dist_23 = np.linalg.norm(t2_world - t_old_world)
    
    print(f"   Distance C1↔C2: {dist_12:.3f}")
    print(f"   Distance C1↔C3: {dist_13:.3f}")
    print(f"   Distance C2↔C3: {dist_23:.3f}")
    

    print(f"\nSummary:")
    print(f"  Camera 1 (NEW_04): World origin")
    print(f"  Camera 2 (NEW_01): {np.sum(inlier_mask)} inliers")
    print(f"  Camera 3 (OLD): {np.sum(inlier_mask_pnp)} inliers")
    print(f"  Total 3D points: {len(X3D_inliers_pnp)}")

# =========================================================================
# STEP 6: BUNDLE ADJUSTMENT
# =========================================================================

    print("\n" + "="*70)
    print("BUNDLE ADJUSTMENT")
    print("="*70)

    # Preparar datos
    common_indices = indices_3D[inlier_mask_pnp]
    x3_ba = x2D_old[inlier_mask_pnp]
    X3D_ba = X3D[common_indices]

    # BA - Solo OLD
    R_old_opt, t_old_opt = utilities2.bundle_adjustment_camera_old_only(
        x3_ba, X3D_ba, K_old, R_old_raw, t_old_raw
    )

    # Visualizar
    utilities2.plot_3d_reconstruction_three_cameras(
        X3D_inliers_pnp,
        R_best, t_best,
        R_old_opt, t_old_opt,
        title="After Bundle Adjustment (OLD refined)"
    )
    kpts_old_22 = x2D_old[inlier_mask_pnp]

    # Visualizar
    utilities2.visualize_old_with_inliers(imgOld, kpts_old_22, save_path="D:/Dev_Space/python/Gloabal_proyect/src/old_with_inliers.png" )
    utilities2.visualize_new_with_inliers(img1,kpts1, save_path="D:/Dev_Space/python/Gloabal_proyect/src/new_with_inliers.png" )

    print("\n" + "="*70)
    print("PIPELINE COMPLETED!")
    print("="*70)
    
    return {
        'K': K,
        'img1': img1,
        'img2': img2,
        'kpts1': kpts1,
        'kpts2': kpts2,
        'scores': scores,
        'F': F_best,
        'E': E,
        'inlier_mask': inlier_mask,
        'x1_inliers': x1_inliers,
        'x2_inliers': x2_inliers,
        'R': R_best,
        't': t_best,
        'X3D': X3D,
        'P1': P1,
        'P2': P2
    }

if __name__ == "__main__":
    main()