import numpy as np 
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
import cv2
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rot


def load_superglue_matches(npz_path, min_score=0.5, return_descriptors=False):
    """
    Load matches from SuperGlue .npz file
    
    Parameters:
    -----------
    npz_path : str
        Path to .npz file
    min_score : float
        Minimum confidence score to keep match
    return_descriptors : bool
        If True, also return descriptors for matched keypoints
    
    Returns:
    --------
    kpts0 : np.ndarray (N, 2)
        Keypoints in image 0
    kpts1 : np.ndarray (N, 2)
        Corresponding keypoints in image 1
    scores : np.ndarray (N,)
        Confidence scores
    desc0 : np.ndarray (N, 256) [optional]
        Descriptors for keypoints in image 0 (only if return_descriptors=True)
    desc1 : np.ndarray (N, 256) [optional]
        Descriptors for keypoints in image 1 (only if return_descriptors=True)
    """
    # Load file
    data = np.load(npz_path)
    
    # Extract data
    matches = data["matches"]  # (M,) array, matches[i] = j means kp0[i] matches kp1[j]
    keypoints0 = data["keypoints0"]  # (M, 2)
    keypoints1 = data["keypoints1"]  # (K, 2)
    
    # Extract descriptors if they exist
    descriptors0 = data.get("descriptors0", None)  # (M, 256) or None
    descriptors1 = data.get("descriptors1", None)  # (K, 256) or None
    
    # Get scores (different possible names)
    scores = None
    for key in ["matching_scores0", "scores", "mconf"]:
        if key in data.files:
            scores = data[key]
            break
    
    # Filter valid matches
    valid_mask = matches > -1
    
    # Apply score threshold if available
    if scores is not None:
        valid_mask = valid_mask & (scores >= min_score)
    
    # Get matched indices
    matched_indices = matches[valid_mask]
    
    # Extract matched keypoints
    kpts0 = keypoints0[valid_mask]
    kpts1 = keypoints1[matched_indices]
    match_scores = scores[valid_mask] if scores is not None else None
    
    # Extract matched descriptors (if requested and available)
    desc0 = None
    desc1 = None
    if return_descriptors:
        if descriptors0 is not None and descriptors1 is not None:
            desc0 = descriptors0[valid_mask]  # Descriptors from image 0
            desc1 = descriptors1[matched_indices]  # Corresponding descriptors from image 1
        else:
            print("Warning: Descriptors not found in .npz file")
    
    # Print statistics
    print(f"\n{'='*70}")
    print(f"LOADED MATCHES: {npz_path.split('/')[-1]}")
    print(f"{'='*70}")
    print(f"Total keypoints in image 0: {len(keypoints0)}")
    print(f"Total keypoints in image 1: {len(keypoints1)}")
    print(f"Valid matches (score >= {min_score}): {len(kpts0)}")
    if match_scores is not None:
        print(f"Score range: [{match_scores.min():.3f}, {match_scores.max():.3f}]")
        print(f"Mean score: {match_scores.mean():.3f}")
    if return_descriptors and desc0 is not None:
        print(f"Descriptor dimensionality: {desc0.shape[1]}")
        print(f"Descriptor norm (mean): {np.linalg.norm(desc0, axis=1).mean():.3f}")
    
    if return_descriptors:
        return kpts0, kpts1, match_scores, desc0, desc1
    else:
        return kpts0, kpts1, match_scores
def visualize_raw_matches(kpts0, kpts1, img0, img1, scores=None, 
                          max_display=150, title="Raw Matches"):
    """
    Visualize matches side by side with connection lines.
    IMPORTANT: This shows the EXACT coordinates from SuperGlue.
    
    Parameters:
    -----------
    kpts0, kpts1 : np.ndarray (N, 2)
        Matched keypoints
    img0, img1 : np.ndarray
        Images (RGB)
    scores : np.ndarray (N,) or None
        Match confidence scores
    max_display : int
        Maximum number of matches to display
    title : str
        Plot title
    """
    # Limit number of matches to display
    n_matches = len(kpts0)
    if n_matches > max_display:
        indices = np.random.choice(n_matches, max_display, replace=False)
        kpts0_vis = kpts0[indices]
        kpts1_vis = kpts1[indices]
        scores_vis = scores[indices] if scores is not None else None
    else:
        kpts0_vis = kpts0
        kpts1_vis = kpts1
        scores_vis = scores
    
    # Create figure
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Display images
    ax0.imshow(img0)
    ax0.set_title(f'Image 1 - {n_matches} matches', fontsize=14)
    ax0.axis('off')
    
    ax1.imshow(img1)
    ax1.set_title(f'Image 2 - {n_matches} matches', fontsize=14)
    ax1.axis('off')
    
    # Plot keypoints
    ax0.scatter(kpts0_vis[:, 0], kpts0_vis[:, 1], 
                c='lime', s=40, marker='o', edgecolors='red', linewidths=2, alpha=0.8)
    ax1.scatter(kpts1_vis[:, 0], kpts1_vis[:, 1], 
                c='lime', s=40, marker='o', edgecolors='red', linewidths=2, alpha=0.8)
    
    # Draw connection lines
    for i in range(len(kpts0_vis)):
        # Color based on score if available
        if scores_vis is not None:
            color = plt.cm.viridis(scores_vis[i])
        else:
            color = 'yellow'
        
        # Create connection line
        con = ConnectionPatch(
            xyA=kpts0_vis[i], coordsA=ax0.transData,
            xyB=kpts1_vis[i], coordsB=ax1.transData,
            color=color, alpha=0.4, linewidth=1.5
        )
        fig.add_artist(con)
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    print(f"\n{'='*70}")
    print(f"VISUALIZATION STATISTICS")
    print(f"{'='*70}")
    print(f"Displaying {len(kpts0_vis)} out of {n_matches} matches")
    print(f"Image 0 shape: {img0.shape}")
    print(f"Image 1 shape: {img1.shape}")
    print(f"Keypoints 0 range: X[{kpts0[:, 0].min():.1f}, {kpts0[:, 0].max():.1f}], "
          f"Y[{kpts0[:, 1].min():.1f}, {kpts0[:, 1].max():.1f}]")
    print(f"Keypoints 1 range: X[{kpts1[:, 0].min():.1f}, {kpts1[:, 0].max():.1f}], "
          f"Y[{kpts1[:, 1].min():.1f}, {kpts1[:, 1].max():.1f}]")

def compute_fundamental_8point(x1, x2):
    """
    Compute Fundamental matrix from 8 point correspondences.
    Simple direct method without normalization.
    
    Parameters:
    -----------
    x1, x2 : np.ndarray (N, 2) where N >= 8
        Point correspondences
    
    Returns:
    --------
    F : np.ndarray (3, 3)
        Fundamental matrix (rank 2)
    """
    n = x1.shape[0]
    
    # Build constraint matrix A
    # Each correspondence gives one equation: [x'x, x'y, x', y'x, y'y, y', x, y, 1] · f = 0
    u1, v1 = x1[:, 0], x1[:, 1]
    u2, v2 = x2[:, 0], x2[:, 1]
    
    A = np.column_stack([
        u2*u1, u2*v1, u2,
        v2*u1, v2*v1, v2,
        u1,    v1,    np.ones(n)
    ])
    
    # Solve using SVD: A·f = 0
    _, _, Vt = np.linalg.svd(A)
    F = Vt[-1].reshape(3, 3)
    
    # Enforce rank-2 constraint (fundamental matrix must have rank 2)
    U, S, Vt = np.linalg.svd(F)
    S[-1] = 0.0  # Set smallest singular value to zero
    F = U @ np.diag(S) @ Vt
    
    # Normalize (optional, for numerical stability)
    F = F / np.linalg.norm(F)
    
    return F


def evaluate_fundamental(F, x1, x2):
    """
    Evaluate Fundamental matrix using symmetric epipolar distance.
    This is the GEOMETRIC error: distance from point to epipolar line.
    
    Parameters:
    -----------
    F : np.ndarray (3, 3)
        Fundamental matrix
    x1, x2 : np.ndarray (N, 2)
        Point correspondences
    
    Returns:
    --------
    errors : np.ndarray (N,)
        Symmetric epipolar distance for each point (in pixels)
        
    Note:
    -----
    For a perfect match, x2^T · F · x1 = 0 (epipolar constraint)
    Error measures how far we are from satisfying this constraint.
    """
    # Convert to homogeneous coordinates
    x1_h = np.column_stack([x1, np.ones(len(x1))])  # (N, 3)
    x2_h = np.column_stack([x2, np.ones(len(x2))])  # (N, 3)
    
    # Compute epipolar lines
    # l2 = F · x1  →  line in image 2 corresponding to point x1
    # l1 = F^T · x2  →  line in image 1 corresponding to point x2
    Fx1 = (F @ x1_h.T).T      # (N, 3) - lines in image 2
    Ftx2 = (F.T @ x2_h.T).T   # (N, 3) - lines in image 1
    
    # Epipolar constraint: x2^T · F · x1 (should be ~0 for perfect match)
    x2tFx1 = np.sum(x2_h * Fx1, axis=1)  # (N,)
    
    # Geometric distance: point to line
    # For line l = [a, b, c], distance from point p = [x, y, 1] is:
    # d = |ax + by + c| / sqrt(a² + b²)
    
    # Distance from x2 to its epipolar line l2 in image 2
    d2 = np.abs(x2tFx1) / (np.sqrt(Fx1[:, 0]**2 + Fx1[:, 1]**2) + 1e-10)
    
    # Distance from x1 to its epipolar line l1 in image 1
    d1 = np.abs(x2tFx1) / (np.sqrt(Ftx2[:, 0]**2 + Ftx2[:, 1]**2) + 1e-10)
    
    # Symmetric epipolar distance (average of both distances)
    errors = 0.5 * (d1 + d2)
    
    return errors


def ransac_fundamental(x1, x2, n_iterations=50000, threshold=3.0, seed=42):
    """
    RANSAC for Fundamental matrix estimation.
    
    Parameters:
    -----------
    x1, x2 : np.ndarray (N, 2)
        Point correspondences
    n_iterations : int
        Number of RANSAC iterations
    threshold : float
        Inlier threshold in pixels
    seed : int
        Random seed for reproducibility
    
    Returns:
    --------
    F_best : np.ndarray (3, 3)
        Best Fundamental matrix
    inlier_mask : np.ndarray (N,)
        Boolean mask of inliers
    """
    np.random.seed(seed)
    
    n_points = len(x1)
    best_inliers = 0
    best_F = None
    best_mask = None
    
    print(f"\n{'='*70}")
    print(f"RANSAC FOR FUNDAMENTAL MATRIX")
    print(f"{'='*70}")
    print(f"Number of points: {n_points}")
    print(f"RANSAC iterations: {n_iterations}")
    print(f"Inlier threshold: {threshold} px")
    print(f"Running RANSAC...")
    
    for i in range(n_iterations):
        # Sample 8 points
        indices = np.random.choice(n_points, 8, replace=False)
        x1_sample = x1[indices]
        x2_sample = x2[indices]
        
        # Compute F
        F = compute_fundamental_8point(x1_sample, x2_sample)
        
        # Evaluate on all points
        errors = evaluate_fundamental(F, x1, x2)
        
        # Count inliers
        inlier_mask = errors < threshold
        n_inliers = np.sum(inlier_mask)
        
        # Update best
        if n_inliers > best_inliers:
            best_inliers = n_inliers
            best_F = F
            best_mask = inlier_mask
        
        # Progress
        if (i + 1) % 10000 == 0:
            print(f"  Iteration {i+1}/{n_iterations}, best inliers: {best_inliers}")
    
    print(f"\n{'='*70}")
    print(f"RANSAC RESULTS")
    print(f"{'='*70}")
    print(f"Total points: {n_points}")
    print(f"Inliers: {best_inliers} ({100*best_inliers/n_points:.1f}%)")
    print(f"Outliers: {n_points - best_inliers} ({100*(n_points-best_inliers)/n_points:.1f}%)")
    print(f"\nBest Fundamental Matrix:")
    print(best_F)
    
    # Compute errors for inliers
    errors = evaluate_fundamental(best_F, x1, x2)
    inlier_errors = errors[best_mask]
    print(f"\nInlier errors (px):")
    print(f"  Mean: {inlier_errors.mean():.3f}")
    print(f"  Median: {np.median(inlier_errors):.3f}")
    print(f"  Std: {inlier_errors.std():.3f}")
    print(f"  Max: {inlier_errors.max():.3f}")
    
    return best_F, best_mask

def visualize_inliers_outliers(x1, x2, inlier_mask, img1, img2, 
                                max_display_inliers=200, max_display_outliers=100,
                                show_numbers=False, n_numbered=20):
    """
    Visualize inliers and outliers separately.
    
    Parameters:
    -----------
    x1, x2 : np.ndarray (N, 2)
        All matched points
    inlier_mask : np.ndarray (N,)
        Boolean mask (True = inlier)
    img1, img2 : np.ndarray
        Images
    max_display_inliers : int
        Max inliers to display
    max_display_outliers : int
        Max outliers to display
    show_numbers : bool
        If True, add numbers to some points
    n_numbered : int
        Number of points to label with numbers (if show_numbers=True)
    """
    # Split into inliers and outliers
    x1_in = x1[inlier_mask]
    x2_in = x2[inlier_mask]
    x1_out = x1[~inlier_mask]
    x2_out = x2[~inlier_mask]
    
    # Sample for display
    n_in = len(x1_in)
    n_out = len(x1_out)
    
    if n_in > max_display_inliers:
        idx_in = np.random.choice(n_in, max_display_inliers, replace=False)
        x1_in_vis = x1_in[idx_in]
        x2_in_vis = x2_in[idx_in]
    else:
        idx_in = np.arange(n_in)
        x1_in_vis = x1_in
        x2_in_vis = x2_in
    
    if n_out > max_display_outliers:
        idx_out = np.random.choice(n_out, max_display_outliers, replace=False)
        x1_out_vis = x1_out[idx_out]
        x2_out_vis = x2_out[idx_out]
    else:
        x1_out_vis = x1_out
        x2_out_vis = x2_out
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    
    # Top row: Inliers
    axes[0, 0].imshow(img1)
    axes[0, 0].scatter(x1_in_vis[:, 0], x1_in_vis[:, 1], 
                       c='lime', s=30, marker='o', edgecolors='darkgreen', 
                       linewidths=2, alpha=0.8, label='Inliers')
    
    # Add numbers to some inliers
    if show_numbers and n_in > 0:
        n_to_label = min(n_numbered, len(x1_in_vis))
        for i in range(n_to_label):
            axes[0, 0].text(x1_in_vis[i, 0] + 5, x1_in_vis[i, 1] - 1, 
                           str(i), color='white', fontsize=9, fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='green', alpha=0.8))
    
    axes[0, 0].set_title(f'Image 1 - INLIERS ({n_in} total, showing {len(x1_in_vis)})', 
                         fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    axes[0, 0].legend(loc='upper right', fontsize=12)
    
    axes[0, 1].imshow(img2)
    axes[0, 1].scatter(x2_in_vis[:, 0], x2_in_vis[:, 1], 
                       c='lime', s=30, marker='o', edgecolors='darkgreen', 
                       linewidths=2, alpha=0.8, label='Inliers')
    
    # Add numbers to corresponding inliers
    if show_numbers and n_in > 0:
        n_to_label = min(n_numbered, len(x2_in_vis))
        for i in range(n_to_label):
            axes[0, 1].text(x2_in_vis[i, 0] + 5, x2_in_vis[i, 1] - 1, 
                           str(i), color='white', fontsize=9, fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='green', alpha=0.8))
    
    axes[0, 1].set_title(f'Image 2 - INLIERS ({n_in} total, showing {len(x2_in_vis)})', 
                         fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    axes[0, 1].legend(loc='upper right', fontsize=12)
    
    # Bottom row: Outliers
    axes[1, 0].imshow(img1)
    if len(x1_out_vis) > 0:
        axes[1, 0].scatter(x1_out_vis[:, 0], x1_out_vis[:, 1], 
                           c='red', s=30, marker='x', linewidths=2, 
                           alpha=0.8, label='Outliers')
    axes[1, 0].set_title(f'Image 1 - OUTLIERS ({n_out} total, showing {len(x1_out_vis)})', 
                         fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')
    axes[1, 0].legend(loc='upper right', fontsize=12)
    
    axes[1, 1].imshow(img2)
    if len(x2_out_vis) > 0:
        axes[1, 1].scatter(x2_out_vis[:, 0], x2_out_vis[:, 1], 
                           c='red', s=30, marker='x', linewidths=2, 
                           alpha=0.8, label='Outliers')
    axes[1, 1].set_title(f'Image 2 - OUTLIERS ({n_out} total, showing {len(x2_out_vis)})', 
                         fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')
    axes[1, 1].legend(loc='upper right', fontsize=12)
    
    plt.suptitle('RANSAC Results: Inliers vs Outliers', fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.show()

def fundamental_to_essential(F, K):
    """
    Convert Fundamental matrix to Essential matrix.
    
    E = K^T · F · K
    
    Parameters:
    -----------
    F : np.ndarray (3, 3)
        Fundamental matrix
    K : np.ndarray (3, 3)
        Camera calibration matrix
    
    Returns:
    --------
    E : np.ndarray (3, 3)
        Essential matrix (rank 2, with singular values [1, 1, 0])
    """
    # Compute E
    E = K.T @ F @ K
    
    # Enforce E properties: singular values should be [σ, σ, 0]
    U, S, Vt = np.linalg.svd(E)
    
    # Force singular values to be [1, 1, 0]
    S_corrected = np.array([1.0, 1.0, 0.0])
    E = U @ np.diag(S_corrected) @ Vt
    
    return E


def decompose_essential_matrix(E):
    """
    Decompose Essential matrix into 4 possible (R, t) solutions.
    
    Given E, there are 4 possible camera poses:
    - (R1, +t)
    - (R1, -t)
    - (R2, +t)
    - (R2, -t)
    
    Only one will have points in front of both cameras (positive depth).
    
    Parameters:
    -----------
    E : np.ndarray (3, 3)
        Essential matrix
    
    Returns:
    --------
    solutions : list of 4 tuples (R, t)
        R : np.ndarray (3, 3) - Rotation matrix
        t : np.ndarray (3,) - Translation vector (unit norm)
    """
    U, S, Vt = np.linalg.svd(E)
    
    # Ensure proper rotation (det(U·Vt) = +1)
    if np.linalg.det(U @ Vt) < 0:
        Vt = -Vt
    
    # W matrix for extracting rotations
    W = np.array([
        [0, -1, 0],
        [1,  0, 0],
        [0,  0, 1]
    ])
    
    # Two possible rotations
    R1 = U @ W @ Vt
    R2 = U @ W.T @ Vt
    
    # Translation (last column of U, unit vector)
    t = U[:, 2]
    
    # 4 possible solutions
    solutions = [
        (R1,  t),
        (R1, -t),
        (R2,  t),
        (R2, -t)
    ]
    
    return solutions


def triangulate_points(x1, x2, P1, P2):
    """
    Triangulate 3D points from correspondences and projection matrices.
    Uses DLT (Direct Linear Transform) method.
    
    Parameters:
    -----------
    x1, x2 : np.ndarray (N, 2)
        Corresponding points in image 1 and 2
    P1, P2 : np.ndarray (3, 4)
        Projection matrices for camera 1 and 2
    
    Returns:
    --------
    X : np.ndarray (N, 3)
        Triangulated 3D points in world coordinates (camera 1 frame)
    """
    N = x1.shape[0]
    X = np.zeros((N, 3))
    
    for i in range(N):
        u1, v1 = x1[i]
        u2, v2 = x2[i]
        
        # Build linear system A·X = 0
        # Each point gives 2 equations in each image (4 total)
        A = np.array([
            u1 * P1[2, :] - P1[0, :],  # u1(p3^T X) - p1^T X = 0
            v1 * P1[2, :] - P1[1, :],  # v1(p3^T X) - p2^T X = 0
            u2 * P2[2, :] - P2[0, :],
            v2 * P2[2, :] - P2[1, :]
        ])
        
        # Solve using SVD
        _, _, Vt = np.linalg.svd(A)
        X_h = Vt[-1]  # Last row of V (homogeneous coordinates)
        
        # Convert to 3D (normalize by w)
        X_h = X_h / X_h[3]
        X[i] = X_h[:3]
    
    return X


def check_cheirality(X, R, t):
    """
    Check cheirality constraint: points must be in front of both cameras.
    
    Parameters:
    -----------
    X : np.ndarray (N, 3)
        3D points in camera 1 frame
    R : np.ndarray (3, 3)
        Rotation from camera 1 to camera 2
    t : np.ndarray (3,)
        Translation from camera 1 to camera 2
    
    Returns:
    --------
    n_good : int
        Number of points with positive depth in both cameras
    """
    # Depth in camera 1 (just Z coordinate)
    depth1 = X[:, 2]
    
    # Transform to camera 2 frame
    X_cam2 = (R @ X.T + t.reshape(3, 1)).T
    depth2 = X_cam2[:, 2]
    
    # Count points with positive depth in BOTH cameras
    n_good = np.sum((depth1 > 0) & (depth2 > 0))
    
    return n_good


def select_correct_pose(K, x1, x2, solutions):
    """
    Select the correct (R, t) from 4 possible solutions.
    
    Criterion: The correct pose is the one that puts most points
    in front of BOTH cameras (positive depth).
    
    Parameters:
    -----------
    K : np.ndarray (3, 3)
        Camera calibration matrix
    x1, x2 : np.ndarray (N, 2)
        Inlier correspondences
    solutions : list of 4 tuples (R, t)
        Possible camera poses
    
    Returns:
    --------
    best_pose : tuple (R, t)
        The correct pose
    X3D : np.ndarray (N, 3)
        Triangulated 3D points using the best pose
    """
    # Camera 1 at origin
    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
    
    best_count = 0
    best_pose = None
    best_X3D = None
    
    print(f"\n{'='*70}")
    print("TESTING 4 POSSIBLE POSES")
    print(f"{'='*70}")
    
    for i, (R, t) in enumerate(solutions, 1):
        # Build projection matrix for camera 2
        P2 = K @ np.hstack((R, t.reshape(3, 1)))
        
        # Triangulate points
        X = triangulate_points(x1, x2, P1, P2)
        
        # Check cheirality
        n_good = check_cheirality(X, R, t)
        
        print(f"Solution {i}: {n_good}/{len(x1)} points in front of both cameras "
              f"({100*n_good/len(x1):.1f}%)")
        
        # Keep best solution
        if n_good > best_count:
            best_count = n_good
            best_pose = (R, t)
            best_X3D = X
    
    print(f"\nSelected solution with {best_count}/{len(x1)} valid points")
    
    return best_pose, best_X3D


def compute_reprojection_error(X3D, x_observed, P):
    """
    Compute reprojection error for 3D points.
    
    Parameters:
    -----------
    X3D : np.ndarray (N, 3)
        3D points
    x_observed : np.ndarray (N, 2)
        Observed 2D points
    P : np.ndarray (3, 4)
        Projection matrix
    
    Returns:
    --------
    errors : np.ndarray (N,)
        Reprojection error for each point (in pixels)
    x_projected : np.ndarray (N, 2)
        Projected 2D points
    """
    # Project 3D points to 2D
    X_h = np.column_stack([X3D, np.ones(len(X3D))])  # Homogeneous
    x_proj_h = (P @ X_h.T).T  # Project
    x_projected = x_proj_h[:, :2] / x_proj_h[:, 2:3]  # Normalize
    
    # Compute Euclidean distance
    errors = np.linalg.norm(x_observed - x_projected, axis=1)
    
    return errors, x_projected

def ensamble_T(R_w_c, t_w_c):
    """
    Ensamble a SE(3) matrix with rotation matrix and translation vector.
    From lab utilities.
    """
    T_w_c = np.zeros((4, 4))
    T_w_c[0:3, 0:3] = R_w_c
    T_w_c[0:3, 3] = t_w_c
    T_w_c[3, 3] = 1
    return T_w_c


def draw3DLine(ax, xIni, xEnd, strStyle, lColor, lWidth):
    """
    Draw a segment in a 3D plot.
    From lab utilities.
    """
    ax.plot([np.squeeze(xIni[0]), np.squeeze(xEnd[0])], 
            [np.squeeze(xIni[1]), np.squeeze(xEnd[1])], 
            [np.squeeze(xIni[2]), np.squeeze(xEnd[2])],
            strStyle, color=lColor, linewidth=lWidth)


def drawRefSystem(ax, T_w_c, strStyle, nameStr):
    """
    Draw a reference system in a 3D plot: Red for X axis, Green for Y axis, Blue for Z axis.
    From lab utilities.
    
    Parameters:
    -----------
    ax : matplotlib 3D axis
    T_w_c : np.ndarray (4, 4)
        Reference system C seen from W
    strStyle : str
        Line style
    nameStr : str
        Name of the reference system
    """
    draw3DLine(ax, T_w_c[0:3, 3:4], T_w_c[0:3, 3:4] + T_w_c[0:3, 0:1], strStyle, 'r', 1)
    draw3DLine(ax, T_w_c[0:3, 3:4], T_w_c[0:3, 3:4] + T_w_c[0:3, 1:2], strStyle, 'g', 1)
    draw3DLine(ax, T_w_c[0:3, 3:4], T_w_c[0:3, 3:4] + T_w_c[0:3, 2:3], strStyle, 'b', 1)
    ax.text(np.squeeze(T_w_c[0, 3]+0.1), 
            np.squeeze(T_w_c[1, 3]+0.1), 
            np.squeeze(T_w_c[2, 3]+0.1), nameStr)


def plotNumbered3DPoints(ax, X, strColor, offset):
    """
    Plot indexes of points on a 3D plot.
    From lab utilities.
    
    Parameters:
    -----------
    ax : matplotlib 3D axis
    X : np.ndarray (3, N)
        Points coordinates
    strColor : str
        Color of the text
    offset : tuple
        Offset from the point to the text
    """
    for k in range(X.shape[1]):
        ax.text(X[0, k]+offset[0], X[1, k]+offset[1], X[2, k]+offset[2], 
                str(k), color=strColor)


def plot_3d_reconstruction(X3D, R, t, title="3D Reconstruction"):
    """
    Visualize 3D reconstruction with camera poses using lab-style visualization.
    
    Parameters:
    -----------
    X3D : np.ndarray (N, 3)
        3D points
    R : np.ndarray (3, 3)
        Rotation of camera 2
    t : np.ndarray (3,)
        Translation of camera 2
    title : str
        Plot title
    """
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Camera 1: World frame (at origin)
    T_w_c1 = np.eye(4)
    drawRefSystem(ax, T_w_c1, '-', 'C1')
    
    # Camera 2: Build transformation matrix
    T_w_c2 = np.linalg.inv(ensamble_T(R, t))
    drawRefSystem(ax, T_w_c2, '-', 'C2')
    
    # Plot 3D points
    # Convert to (3, N) format for lab utilities
    X3D_T = X3D.T  # (3, N)
    ax.scatter(X3D_T[0, :], X3D_T[1, :], X3D_T[2, :], 
              c='blue', marker='o', s=10, alpha=0.6, label='3D Points')
    
    # Optionally add point numbers (for first 20 points to avoid clutter)
    # Uncomment if you want to see point IDs:
    # plotNumbered3DPoints(ax, X3D_T[:, :20], 'r', (0.1, 0.1, 0.1))
    
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.legend()
    
    # Equal aspect ratio trick
    max_range = np.array([
        X3D[:, 0].max() - X3D[:, 0].min(),
        X3D[:, 1].max() - X3D[:, 1].min(),
        X3D[:, 2].max() - X3D[:, 2].min()
    ]).max() / 2.0
    
    mid_x = (X3D[:, 0].max() + X3D[:, 0].min()) * 0.5
    mid_y = (X3D[:, 1].max() + X3D[:, 1].min()) * 0.5
    mid_z = (X3D[:, 2].max() + X3D[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # Fake bounding box for equal aspect (lab trick)
    xFakeBoundingBox = np.linspace(mid_x - max_range, mid_x + max_range, 2)
    yFakeBoundingBox = np.linspace(mid_y - max_range, mid_y + max_range, 2)
    zFakeBoundingBox = np.linspace(mid_z - max_range, mid_z + max_range, 2)
    ax.plot(xFakeBoundingBox, yFakeBoundingBox, zFakeBoundingBox, 'w.')
    
    plt.tight_layout()
    print('\nClose the 3D figure to continue. Left button for orbit, right button for zoom.')
    plt.show()


def plot_reprojection_comparison(img1, img2, x1, x2, x1_proj, x2_proj, 
                                 max_display=100):
    """
    Visualize observed vs reprojected points.
    Shows how well the 3D reconstruction matches the original observations.
    
    Green circles = Observed points (ground truth)
    Red X's = Reprojected points (from 3D reconstruction)
    Yellow lines = Reprojection error
    
    Parameters:
    -----------
    img1, img2 : np.ndarray
        Images (RGB)
    x1, x2 : np.ndarray (N, 2)
        Observed points
    x1_proj, x2_proj : np.ndarray (N, 2)
        Reprojected points
    max_display : int
        Maximum points to display (for clarity)
    """
    n = len(x1)
    
    # Sample points if too many
    if n > max_display:
        indices = np.random.choice(n, max_display, replace=False)
        x1_vis = x1[indices]
        x2_vis = x2[indices]
        x1_proj_vis = x1_proj[indices]
        x2_proj_vis = x2_proj[indices]
    else:
        x1_vis = x1
        x2_vis = x2
        x1_proj_vis = x1_proj
        x2_proj_vis = x2_proj
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # =========================================================================
    # Image 1
    # =========================================================================
    ax1.imshow(img1)
    
    # Observed points (green circles)
    ax1.scatter(x1_vis[:, 0], x1_vis[:, 1], 
               c='lime', s=80, marker='o', edgecolors='darkgreen',
               linewidths=2, label='Observed', alpha=0.8, zorder=3)
    
    # Reprojected points (red X's)
    ax1.scatter(x1_proj_vis[:, 0], x1_proj_vis[:, 1], 
               c='red', s=100, marker='x', linewidths=3,
               label='Reprojected', alpha=0.9, zorder=4)
    
    # Draw error lines (yellow)
    for i in range(len(x1_vis)):
        ax1.plot([x1_vis[i, 0], x1_proj_vis[i, 0]], 
                [x1_vis[i, 1], x1_proj_vis[i, 1]], 
                'yellow', alpha=0.5, linewidth=2, zorder=2)
    
    # Compute stats for this view
    errors1 = np.linalg.norm(x1_vis - x1_proj_vis, axis=1)
    ax1.set_title(f'Image 1: Reprojection\n'
                 f'Mean error: {np.mean(errors1):.2f} px, '
                 f'Max: {np.max(errors1):.2f} px', 
                 fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=12)
    ax1.axis('off')
    
    # =========================================================================
    # Image 2
    # =========================================================================
    ax2.imshow(img2)
    
    # Observed points (green circles)
    ax2.scatter(x2_vis[:, 0], x2_vis[:, 1], 
               c='lime', s=80, marker='o', edgecolors='darkgreen',
               linewidths=2, label='Observed', alpha=0.8, zorder=3)
    
    # Reprojected points (red X's)
    ax2.scatter(x2_proj_vis[:, 0], x2_proj_vis[:, 1], 
               c='red', s=100, marker='x', linewidths=3,
               label='Reprojected', alpha=0.9, zorder=4)
    
    # Draw error lines (yellow)
    for i in range(len(x2_vis)):
        ax2.plot([x2_vis[i, 0], x2_proj_vis[i, 0]], 
                [x2_vis[i, 1], x2_proj_vis[i, 1]], 
                'yellow', alpha=0.5, linewidth=2, zorder=2)
    
    # Compute stats for this view
    errors2 = np.linalg.norm(x2_vis - x2_proj_vis, axis=1)
    ax2.set_title(f'Image 2: Reprojection\n'
                 f'Mean error: {np.mean(errors2):.2f} px, '
                 f'Max: {np.max(errors2):.2f} px', 
                 fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=12)
    ax2.axis('off')
    
    plt.suptitle('Reprojection Quality Check: Observed vs Reprojected Points', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # Print explanation
    print(f"\nReprojection visualization:")
    print(f"  Green circles = Observed keypoints (SuperGlue detections)")
    print(f"  Red X's = Reprojected from 3D (where triangulated points project)")
    print(f"  Yellow lines = Reprojection error (shorter = better)")
    print(f"\nDisplaying {len(x1_vis)}/{n} points")


    # =============================================================================



def find_common_points_2D_3D(kpts_old_in_new, kpts_new_inliers, tolerance=1.0):
    """
    Find 2D-3D correspondences by matching keypoints in the NEW image.
    
    Strategy:
    - kpts_old_in_new: Points in NEW_04 that match with OLD image (from SuperGlue)
    - kpts_new_inliers: Inlier points in NEW_04 from 3D reconstruction
    - Find which keypoints are the SAME (spatially close) in NEW_04
    
    Parameters:
    -----------
    kpts_old_in_new : np.ndarray (M, 2)
        Keypoints in new_04 that match with old image
    kpts_new_inliers : np.ndarray (N, 2)
        Inlier keypoints in new_04 from reconstruction
    tolerance : float
        Maximum pixel distance to consider a match
    
    Returns:
    --------
    indices_old : np.ndarray
        Indices into the OLD image keypoints
    indices_3D : np.ndarray
        Corresponding indices into X3D array
    """
    indices_old = []
    indices_3D = []
    
    for i, kpt_old in enumerate(kpts_old_in_new):
        # Find closest point in new image inliers
        distances = np.linalg.norm(kpts_new_inliers - kpt_old, axis=1)
        min_dist = np.min(distances)
        min_idx = np.argmin(distances)
        
        if min_dist < tolerance:
            indices_old.append(i)
            indices_3D.append(min_idx)
    
    return np.array(indices_old), np.array(indices_3D)


def compute_projection_DLT(X3D, x2D):
    """
    DLT for P matrix from N≥6 correspondences WITH DATA NORMALIZATION.
    
    CRITICAL: Normalization improves numerical stability dramatically.
    
    Parameters:
    -----------
    X3D : (N, 3) 3D points
    x2D : (N, 2) 2D points
    
    Returns:
    --------
    P : (3, 4) Projection matrix
    """
    N = len(X3D)
    
    
    # Normalizar puntos 3D
    mean_3D = np.mean(X3D, axis=0)
    std_3D = np.std(X3D)
    
    T_3D = np.array([
        [1/std_3D,        0,        0, -mean_3D[0]/std_3D],
        [       0, 1/std_3D,        0, -mean_3D[1]/std_3D],
        [       0,        0, 1/std_3D, -mean_3D[2]/std_3D],
        [       0,        0,        0,                  1]
    ])
    
    X3D_hom = np.column_stack([X3D, np.ones(N)])
    X3D_norm = (T_3D @ X3D_hom.T).T[:, :3]
    
    # Normalizar puntos 2D
    mean_2D = np.mean(x2D, axis=0)
    std_2D = np.std(x2D)
    
    T_2D = np.array([
        [1/std_2D,        0, -mean_2D[0]/std_2D],
        [       0, 1/std_2D, -mean_2D[1]/std_2D],
        [       0,        0,                  1]
    ])
    
    x2D_hom = np.column_stack([x2D, np.ones(N)])
    x2D_norm = (T_2D @ x2D_hom.T).T[:, :2]
    
    # =========================================================================
    # DLT con datos normalizados
    # =========================================================================
    
    A = []
    for i in range(N):
        X, Y, Z = X3D_norm[i]
        u, v = x2D_norm[i]
        
        A.append([X, Y, Z, 1,  0, 0, 0, 0, -u*X, -u*Y, -u*Z, -u])
        A.append([0, 0, 0, 0,  X, Y, Z, 1, -v*X, -v*Y, -v*Z, -v])
    
    A = np.array(A)
    
    # Resolver con SVD
    _, _, Vt = np.linalg.svd(A)
    P_norm = Vt[-1].reshape(3, 4)
    
    # =========================================================================
    # DENORMALIZAR
    # =========================================================================
    
    P = np.linalg.inv(T_2D) @ P_norm @ T_3D
    
    # Normalizar para que P[2,3] tenga escala razonable
    # (esto ayuda a la descomposición posterior)
    if abs(P[2, 3]) > 1e-10:
        P = P / P[2, 3]
    
    return P

def evaluate_projection_matrix(P, X3D, x2D):
    """
    Evaluate Projection Matrix using reprojection error.
    
    Error: Euclidean distance between observed 2D points and 
           projected 3D points (in pixels).
    
    Parameters:
    -----------
    P : np.ndarray (3, 4)
        Projection matrix
    X3D : np.ndarray (N, 3)
        3D points
    x2D : np.ndarray (N, 2)
        Observed 2D points
    
    Returns:
    --------
    errors : np.ndarray (N,)
        Reprojection error for each point (pixels)
    """
    # Convert 3D to homogeneous
    X3D_h = np.column_stack([X3D, np.ones(len(X3D))])  # (N, 4)
    
    # Project: x = P @ X3D
    x_proj_h = (P @ X3D_h.T).T  # (N, 3)
    
    # Normalize to get pixel coordinates
    x_proj = x_proj_h[:, :2] / x_proj_h[:, 2:3]  # (N, 2)
    
    # Euclidean distance
    errors = np.linalg.norm(x2D - x_proj, axis=1)
    
    return errors


def ransac_projection_matrix(X3D, x2D, n_iterations=50000, threshold=5.0, seed=42):
    """
    RANSAC for robust Projection Matrix estimation using DLT.
    
    Similar to ransac_fundamental(), but for P instead of F.
    
    Minimal solution: 6 points (DLT)
    Evaluation: Reprojection error (pixels)
    
    Parameters:
    -----------
    X3D : np.ndarray (N, 3)
        3D points in world frame (Camera 1 coordinates)
    x2D : np.ndarray (N, 2)
        Corresponding 2D points in OLD image
    n_iterations : int
        Number of RANSAC iterations
    threshold : float
        Inlier threshold in pixels
    seed : int
        Random seed
    
    Returns:
    --------
    P_best : np.ndarray (3, 4)
        Best projection matrix
    inlier_mask : np.ndarray (N,)
        Boolean mask of inliers
    """
    np.random.seed(seed)
    
    N = len(X3D)
    best_inliers = 0
    best_P = None
    best_mask = None
    
    print(f"\n{'='*70}")
    print(f"RANSAC FOR PROJECTION MATRIX (OLD CAMERA)")
    print(f"{'='*70}")
    print(f"Number of 2D-3D correspondences: {N}")
    print(f"RANSAC iterations: {n_iterations}")
    print(f"Inlier threshold: {threshold} px")
    print(f"Running RANSAC...")
    
    for i in range(n_iterations):
        # 1. Sample 6 points (minimum for DLT)
        indices = np.random.choice(N, 6, replace=False)
        X3D_sample = X3D[indices]
        x2D_sample = x2D[indices]
        
        # 2. Compute P using DLT
        try:
            P = compute_projection_DLT(X3D_sample, x2D_sample)
        except:
            continue  # Skip if DLT fails
        
        # 3. Evaluate on all points
        errors = evaluate_projection_matrix(P, X3D, x2D)
        
        # 4. Count inliers
        inlier_mask = errors < threshold
        n_inliers = np.sum(inlier_mask)
        
        # 5. Update best
        if n_inliers > best_inliers:
            best_inliers = n_inliers
            best_P = P
            best_mask = inlier_mask
        
        # Progress
        if (i + 1) % 10000 == 0:
            print(f"  Iteration {i+1}/{n_iterations}, best inliers: {best_inliers}")
    
    print(f"\n{'='*70}")
    print(f"RANSAC RESULTS (PROJECTION MATRIX)")
    print(f"{'='*70}")
    print(f"Total correspondences: {N}")
    print(f"Inliers: {best_inliers} ({100*best_inliers/N:.1f}%)")
    print(f"Outliers: {N - best_inliers} ({100*(N-best_inliers)/N:.1f}%)")
    
    # Compute errors for inliers
    errors = evaluate_projection_matrix(best_P, X3D, x2D)
    inlier_errors = errors[best_mask]
    print(f"\nInlier reprojection errors (px):")
    print(f"  Mean: {inlier_errors.mean():.3f}")
    print(f"  Median: {np.median(inlier_errors):.3f}")
    print(f"  Std: {inlier_errors.std():.3f}")
    print(f"  Max: {inlier_errors.max():.3f}")
    
    print(f"\nBest Projection Matrix P:")
    print(best_P)
    
    return best_P, best_mask

def decompose_projection(P):
    """
    Decompose P using OpenCV method (slide step 6)
    K̄, R_cw, t_wc = cv.decomposeProjectionMatrix(sign(det(M)) * P)
    """
    
    print(f"\n{'='*70}")
    print("DECOMPOSING P")
    print(f"{'='*70}")
    
    # Extract M and compute sign(det(M))
    M = P[:, 0:3]
    det_M = np.linalg.det(M)
    sign_det_M = np.sign(det_M)
    
    print(f"\nM = P[:, 0:3]")
    print(f"det(M) = {det_M:.3f}")
    print(f"sign(det(M)) = {sign_det_M:.0f}")
    
    # Apply sign correction to P
    P_signed = sign_det_M * P
    
    print(f"\nP_signed = sign(det(M)) * P")
    
    # OpenCV decomposition
    K_raw, R_raw, t_wc_h, _, _, _, _ = cv2.decomposeProjectionMatrix(P_signed)
    
    # Normalize K
    K = K_raw / K_raw[2, 2]
    
    # Extract t_wc from homogeneous
    t_wc = t_wc_h[:3, 0] / t_wc_h[3, 0]
    
    # Build D for sign correction
    D = np.diag([
        np.sign(K[0, 0]),
        np.sign(K[1, 1]),
        np.sign(K[2, 2])
    ])
    
    print(f"\nD = diag(sign(k11), sign(k22), sign(k33))")
    print(f"D = {np.diag(D)}")
    
    # Apply D
    K = D @ K
    R_cw = D @ R_raw
    
    # Remove skew
    K[0, 1] = 0.0
    
    # Ensure det(R_cw) = +1
    if np.linalg.det(R_cw) < 0:
        R_cw = -R_cw
        K = -K
        K = K / K[2, 2]
    
    # Force orthogonality
    U, _, Vt = np.linalg.svd(R_cw)
    R_cw = U @ Vt
    if np.linalg.det(R_cw) < 0:
        Vt[-1, :] *= -1
        R_cw = U @ Vt
    
    print(f"\nFinal results:")
    print(f"  K:")
    print(f"    fx = {K[0,0]:.2f}, fy = {K[1,1]:.2f}")
    print(f"    cx = {K[0,2]:.2f}, cy = {K[1,2]:.2f}")
    print(f"  det(R_cw) = {np.linalg.det(R_cw):.6f}")
    print(f"  t_wc = {t_wc}")
    
    # Verify
    P_recon = K @ np.hstack([R_cw, (-R_cw @ t_wc).reshape(3, 1)])
    error = np.linalg.norm(P / P[2,3] - P_recon / P_recon[2,3])
    print(f"\nVerification: error = {error:.6e}")
    
    assert K[0,0] > 0 and K[1,1] > 0, "K diagonal must be positive"
    assert abs(np.linalg.det(R_cw) - 1.0) < 0.01, "det(R) must be +1"
    
    return K, R_cw, t_wc

def visualize_pnp_reprojection(img_old, x2D_old, x2D_reproj, inlier_mask,
                                max_display=100, title="PnP Reprojection"):
    """
    Visualize PnP results: observed vs reprojected points.
    
    Similar to your plot_reprojection_comparison() but for single image.
    
    Parameters:
    -----------
    img_old : np.ndarray
        Old image (RGB)
    x2D_old : np.ndarray (N, 2)
        Observed 2D points
    x2D_reproj : np.ndarray (N, 2)
        Reprojected points from 3D
    inlier_mask : np.ndarray (N,)
        Boolean mask of inliers
    max_display : int
        Max points to display
    title : str
        Plot title
    """
    # Split inliers/outliers
    x_in = x2D_old[inlier_mask]
    x_in_proj = x2D_reproj[inlier_mask]
    x_out = x2D_old[~inlier_mask]
    x_out_proj = x2D_reproj[~inlier_mask]
    
    n_in = len(x_in)
    n_out = len(x_out)
    
    # Sample if too many
    if n_in > max_display:
        idx = np.random.choice(n_in, max_display, replace=False)
        x_in = x_in[idx]
        x_in_proj = x_in_proj[idx]
    
    if n_out > max_display // 2:
        idx = np.random.choice(n_out, max_display // 2, replace=False)
        x_out = x_out[idx]
        x_out_proj = x_out_proj[idx]
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.imshow(img_old)
    
    # INLIERS
    ax.scatter(x_in[:, 0], x_in[:, 1],
              c='lime', s=80, marker='o', edgecolors='darkgreen',
              linewidths=2, label='Inliers (observed)', alpha=0.8, zorder=3)
    ax.scatter(x_in_proj[:, 0], x_in_proj[:, 1],
              c='red', s=100, marker='x', linewidths=3,
              label='Inliers (reprojected)', alpha=0.9, zorder=4)
    
    # Error lines
    for i in range(len(x_in)):
        ax.plot([x_in[i, 0], x_in_proj[i, 0]],
               [x_in[i, 1], x_in_proj[i, 1]],
               'yellow', alpha=0.5, linewidth=2, zorder=2)
    
    # OUTLIERS
    if len(x_out) > 0:
        ax.scatter(x_out[:, 0], x_out[:, 1],
                  c='orange', s=60, marker='s', edgecolors='red',
                  linewidths=2, label='Outliers', alpha=0.6, zorder=1)
    
    # Stats
    errors_in = np.linalg.norm(x_in - x_in_proj, axis=1)
    ax.set_title(f'{title}\n'
                f'Inliers: {n_in} | Mean error: {np.mean(errors_in):.2f} px | '
                f'Max: {np.max(errors_in):.2f} px',
                fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=12)
    ax.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print(f"\nPnP Reprojection visualization:")
    print(f"  Green circles = Observed keypoints in old image")
    print(f"  Red X's = Reprojected from 3D")
    print(f"  Yellow lines = Reprojection error")
    print(f"  Orange squares = Outliers (rejected by RANSAC)")


def plot_3d_reconstruction_three_cameras(X3D, R2, t2, R_old, t_old,
                                         
                                         title="3D Reconstruction - 3 Cameras"):
    """
    Visualize 3D reconstruction with THREE camera poses.
    Uses your existing drawRefSystem() style.
    
    Parameters:
    -----------
    X3D : np.ndarray (N, 3)
        3D points (inliers only recommended)
    R2, t2 : Camera 2 rotation and translation
    R_old, t_old : Old camera rotation and translation
    title : str
    """
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Camera 1: World origin
    T_w_c1 = np.eye(4)
    drawRefSystem(ax, T_w_c1, '-', 'C1 (NEW_04)')
    
    # Camera 2
    T_w_c2 = np.linalg.inv(ensamble_T(R2, t2))
    drawRefSystem(ax, T_w_c2, '-', 'C2 (NEW_01)')
    
    # Old camera
    T_w_cold = np.linalg.inv(ensamble_T(R_old, t_old))
    drawRefSystem(ax, T_w_cold, '-', 'C_OLD')
    
    # Plot 3D points
    X3D_T = X3D.T  # (3, N)
    ax.scatter(X3D_T[0, :], X3D_T[1, :], X3D_T[2, :],
              c='blue', marker='o', s=20, alpha=0.6, label='3D Points')
    
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.legend()
    
    # Equal aspect ratio
    max_range = np.array([
        X3D[:, 0].max() - X3D[:, 0].min(),
        X3D[:, 1].max() - X3D[:, 1].min(),
        X3D[:, 2].max() - X3D[:, 2].min()
    ]).max() / 2.0
    
    mid_x = (X3D[:, 0].max() + X3D[:, 0].min()) * 0.5
    mid_y = (X3D[:, 1].max() + X3D[:, 1].min()) * 0.5
    mid_z = (X3D[:, 2].max() + X3D[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # Fake bounding box
    xFakeBoundingBox = np.linspace(mid_x - max_range, mid_x + max_range, 2)
    yFakeBoundingBox = np.linspace(mid_y - max_range, mid_y + max_range, 2)
    zFakeBoundingBox = np.linspace(mid_z - max_range, mid_z + max_range, 2)
    ax.plot(xFakeBoundingBox, yFakeBoundingBox, zFakeBoundingBox, 'w.')
    
    plt.tight_layout()
    print('\nClose the 3D figure to continue.')
    plt.show()

def bundle_adjustment_camera_old_only(x3_obs, X3D, K3, R3_init, t3_init):
    """
    Bundle Adjustment: Refina SOLO R3 y t3 (camera OLD).
    """
    
    def pack(t, R):
        rv = Rot.from_matrix(R).as_rotvec()
        return np.concatenate([t, rv])
    
    def unpack(params):
        t = params[0:3]
        rv = params[3:6]
        R = Rot.from_rotvec(rv).as_matrix()
        return R, t
    
    def residuals(params):
        R, t = unpack(params)
        X_cam = (R @ X3D.T + t.reshape(3, 1)).T
        x_proj = (K3 @ X_cam.T).T
        x_proj = x_proj[:, :2] / x_proj[:, 2:3]
        return (x3_obs - x_proj).flatten()
    
    params_init = pack(t3_init, R3_init)
    rmse_init = np.sqrt(np.mean(residuals(params_init)**2))
    
    result = least_squares(residuals, params_init, method='lm', verbose=0)
    
    R3_opt, t3_opt = unpack(result.x)
    rmse_final = np.sqrt(np.mean(result.fun**2))
    
    print(f"\nBundle Adjustment (Camera OLD):")
    print(f"  RMSE: {rmse_init:.3f} px → {rmse_final:.3f} px (Δ{rmse_init-rmse_final:.3f} px)")
    
    return R3_opt, t3_opt

def visualize_old_with_inliers(img_old, kpts_old_22, save_path=None):
    """
    Visualiza imagen OLD con los 22 inliers marcados.
    
    Parameters:
    -----------
    img_old : Imagen Old
    kpts_old_22 : (22, 2) - Los 22 puntos validados O los que sean necesarios
    save_path : str - Ruta para guardar (ej: "old_with_inliers.png")
    """
    
    print(f"\n{'='*70}")
    print("VISUALIZING OLD IMAGE WITH INLIERS")
    print(f"{'='*70}")
    print(f"Original image size: {img_old.shape}")
    print(f"Marking {len(kpts_old_22)} validated points")
    
    # Copiar imagen - MANTIENE TAMAÑO ORIGINAL
    img_marked = img_old.copy()
    
    # Dibujar puntos
    for i, pt in enumerate(kpts_old_22):
        x, y = int(pt[0]), int(pt[1])
        
        # Círculo relleno verde
        cv2.circle(img_marked, (x, y), 4, (0, 255, 0), -1)
        
    
    print(f"Marked image size: {img_marked.shape}")
    print(f"Size unchanged: {img_old.shape == img_marked.shape}")
    
    # GUARDAR si se especifica ruta
    if save_path:
        # Convertir RGB → BGR para OpenCV
        img_save = cv2.cvtColor(img_marked, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, img_save)
        print(f"\nSaved to: {save_path}")
    
    # Mostrar
    plt.figure(figsize=(12, 8))
    plt.imshow(img_marked)
    plt.title(f"OLD- {len(kpts_old_22)} Validated Inliers", 
              fontsize=16, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    return img_marked
def visualize_new_with_inliers(img_new, kpts_new, save_path=None):
    """
    Visualiza imagen New with new inliers
    
    Parameters:
    -----------
    img_new : Imagen new
    kpts_new : (N, 2) - Los aprox 71 puntos validados O los que sean necesarios
    save_path : str - Ruta para guardar (ej: "old_with_inliers.png")
    """
    
    print(f"\n{'='*70}")
    print("VISUALIZING NEW IMAGE WITH INLIERS")
    print(f"{'='*70}")
    print(f"Original image size: {img_new.shape}")
    print(f"Marking {len(kpts_new)} validated points")
    
    # Copiar imagen - MANTIENE TAMAÑO ORIGINAL
    img_marked = img_new.copy()
    
    # Dibujar puntos
    for i, pt in enumerate(kpts_new):
        x, y = int(pt[0]), int(pt[1])
        
        # Círculo relleno verde
        cv2.circle(img_marked, (x, y), 4, (0, 200, 200), -1)
        
    
    print(f"Marked image size: {img_marked.shape}")
    print(f"Size unchanged: {img_new.shape == img_marked.shape}")
    
    # GUARDAR si se especifica ruta
    if save_path:
        # Convertir RGB → BGR para OpenCV
        img_save = cv2.cvtColor(img_marked, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, img_save)
        print(f"\nSaved to: {save_path}")
    
    # Mostrar
    plt.figure(figsize=(12, 8))
    plt.imshow(img_marked)
    plt.title(f"NEW 2025- {len(kpts_new)} Validated Inliers", 
              fontsize=16, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    return img_marked