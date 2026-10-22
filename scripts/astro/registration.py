"""Star detection and frame-to-reference registration.

Everything here runs on the half-resolution, static-pattern-cleaned grayscale
images.  A fixed camera looking at the rotating sky is *not* an affine mapping:
the image of a pure 3-D rotation is the projective homography K R K^-1, and on a
wide field the perspective terms grow with the rotation angle (several pixels
over a few minutes).  We therefore fit a homography, refine it on star centroids
to sub-pixel accuracy, and validate the result with an independent star-lock
score before a frame is allowed in.
"""
import cv2
import numpy as np


def _as_homography(M):
    M = np.asarray(M, np.float32)
    if M.shape == (2, 3):
        M = np.vstack([M, [0, 0, 1]]).astype(np.float32)
    return M


def homography_metrics(M, W, H):
    """Local rotation (deg), centre displacement, local scale and handedness of
    a homography, evaluated at the image centre."""
    M = _as_homography(M).astype(np.float64)
    c = np.array([W / 2.0, H / 2.0, 1.0])
    p = M @ c
    p /= p[2]
    den = M[2] @ c
    u = M[0] @ c
    v = M[1] @ c
    J = np.array([
        [M[0, 0] * den - u * M[2, 0], M[0, 1] * den - u * M[2, 1]],
        [M[1, 0] * den - v * M[2, 0], M[1, 1] * den - v * M[2, 1]],
    ]) / (den * den)
    rot = np.degrees(np.arctan2(J[1, 0] - J[0, 1], J[0, 0] + J[1, 1]))
    det = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
    scale = float(np.sqrt(abs(det)))
    return float(rot), float(p[0] - c[0]), float(p[1] - c[1]), scale, float(det)


def make_detector():
    return cv2.ORB_create(nfeatures=12000, scaleFactor=1.2, nlevels=8,
                          fastThreshold=5, edgeThreshold=19)


def build_static_map(grays, pct=75):
    """High percentile across frames -> the static pattern (hot pixels / fixed
    fixed-pattern noise).  A real star is bright in only a frame or two because
    the sky moves, so it never survives the percentile; a hot pixel does."""
    if len(grays) >= 5:
        return np.percentile(np.stack(grays), pct, axis=0)
    h, w = grays[0].shape
    return np.zeros((h, w), np.float32)


def enhance_for_registration(gray_half, pre_blur=1.0, hp_sigma=20.0):
    """Suppress single-pixel hot pixels, then high-pass so real stars become
    clean local features for ORB (a pre-blur is essential: without it ORB
    matches static hot pixels and the fit collapses to identity)."""
    g = cv2.GaussianBlur(gray_half, (0, 0), pre_blur)
    bg = cv2.GaussianBlur(g, (0, 0), hp_sigma)
    e = np.clip(g - bg, 0, None)
    e = cv2.normalize(e, None, 0, 255, cv2.NORM_MINMAX)
    return e.astype(np.uint8)


def highpass_float(gray_half, pre_blur=1.0, hp_sigma=20.0):
    g = cv2.GaussianBlur(gray_half, (0, 0), pre_blur)
    return np.clip(g - cv2.GaussianBlur(g, (0, 0), hp_sigma), 0, None)


def detect_stars(hpf, nsigma=5.0, minarea=3, maxarea=400, valid=None):
    """Connected-component star centroids.  minarea>=3 rejects isolated
    1-pixel hot pixels, which are the main source of false matches.  `valid`
    optionally restricts detections to the sky (the static foreground has
    textured edges that otherwise masquerade as stars)."""
    m, s = hpf.mean(), hpf.std()
    mask = (hpf > m + nsigma * s).astype(np.uint8)
    if valid is not None:
        mask &= (valid > 0).astype(np.uint8)
    n, lab, stats, cent = cv2.connectedComponentsWithStats(mask, 8)
    pts = [cent[i] for i in range(1, n)
           if minarea <= stats[i, cv2.CC_STAT_AREA] <= maxarea]
    return np.array(pts, np.float32)


def refine_transform(M, ref_stars, mov_stars, tols=(8.0, 4.0, 2.0, 1.5)):
    """Iterative nearest-neighbour star matching + RANSAC homography fit.
    Turns the coarse ORB estimate into a sub-pixel transform."""
    M = np.asarray(M, np.float32).copy()
    if M.shape == (2, 3):
        M = np.vstack([M, [0, 0, 1]]).astype(np.float32)
    if len(ref_stars) < 8 or len(mov_stars) < 8:
        return M
    for tol in tols:
        m2 = cv2.perspectiveTransform(mov_stars.reshape(-1, 1, 2), M).reshape(-1, 2)
        d2 = ((m2[:, None, :] - ref_stars[None, :, :]) ** 2).sum(-1)
        j = d2.argmin(1)
        dist = np.sqrt(d2[np.arange(len(m2)), j])
        keep = dist < tol
        if keep.sum() < 10:
            break
        src = mov_stars[keep].reshape(-1, 1, 2)
        dst = ref_stars[j[keep]].reshape(-1, 1, 2)
        Mn, _ = cv2.findHomography(src, dst, method=cv2.RANSAC,
                                   ransacReprojThreshold=tol,
                                   maxIters=20000, confidence=0.999)
        if Mn is not None:
            M = Mn.astype(np.float32)
    return M


def star_score(ref_enh, mov_enh, M, frac=0.55):
    """Normalized correlation of the high-pass fields over the sky region.
    >0.6 means the transform really locks the stars; ~0.1..0.3 means failure."""
    H, W = ref_enh.shape
    M = _as_homography(M)
    w = cv2.warpPerspective(mov_enh.astype(np.float32), M, (W, H),
                            flags=cv2.INTER_LINEAR)
    reg = slice(0, int(H * frac))
    a = ref_enh[reg].astype(np.float32).ravel()
    b = w[reg].ravel()
    a -= a.mean(); b -= b.mean()
    return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def star_residual(ref_stars, mov_stars, M, tol=1.5):
    """Median distance from warped moving stars to the nearest reference star,
    and the fraction of them within tol.  Small median = a true star lock."""
    if len(ref_stars) < 4 or len(mov_stars) < 4:
        return 1e9, 0.0
    M = _as_homography(M)
    m2 = cv2.perspectiveTransform(mov_stars.reshape(-1, 1, 2), M).reshape(-1, 2)
    d2 = ((m2[:, None, :] - ref_stars[None, :, :]) ** 2).sum(-1)
    d = np.sqrt(d2.min(1))
    return float(np.median(d)), float((d < tol).mean())


def transform_is_plausible(M, W, H, inliers, sc, resid_med, resid_frac):
    rot, tx, ty, scale, det = homography_metrics(M, W, H)
    return (abs(rot) < 15 and abs(tx) < 0.15 * W and abs(ty) < 0.15 * H
            and 0.95 <= scale <= 1.05 and det > 0
            and inliers >= 25 and sc >= 0.55 and resid_frac >= 0.20
            and resid_med < 3.0)


def register(ref_desc, ref_kp, mob_desc, mob_kp):
    """Return 3x3 homography M mapping moving coords -> reference coords."""
    if ref_desc is None or mob_desc is None:
        return None, 0, 0
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    knn = bf.knnMatch(mob_desc, ref_desc, k=2)
    good = [m for m, n in (p for p in knn if len(p) == 2) if m.distance < 0.75 * n.distance]
    if len(good) < 8:
        return None, len(good), 0
    src = np.float32([mob_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([ref_kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    M, inl = cv2.findHomography(src, dst, method=cv2.RANSAC,
                                ransacReprojThreshold=3.0, maxIters=20000,
                                confidence=0.999)
    ninl = int(inl.sum()) if inl is not None else 0
    return (None if M is None else M.astype(np.float32)), len(good), ninl


def scale_matrix(M, s):
    """M was fit at 1/s resolution; return equivalent matrix at full res."""
    M = _as_homography(M)
    S = np.diag([s, s, 1.0]).astype(np.float32)
    return (S @ M @ np.linalg.inv(S)).astype(np.float32)
