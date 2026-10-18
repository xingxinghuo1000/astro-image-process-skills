"""Star detection and frame-to-reference registration.

Everything here runs on the half-resolution, static-pattern-cleaned grayscale
images.  ORB gives a coarse similarity estimate (rotation + translation, scale
~1) that is refined on star centroids to sub-pixel accuracy; the result is then
validated by an independent star-lock score before a frame is allowed in.
"""
import cv2
import numpy as np


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


def detect_stars(hpf, nsigma=5.0, minarea=3, maxarea=400):
    """Connected-component star centroids.  minarea>=3 rejects isolated
    1-pixel hot pixels, which are the main source of false matches."""
    m, s = hpf.mean(), hpf.std()
    mask = (hpf > m + nsigma * s).astype(np.uint8)
    n, lab, stats, cent = cv2.connectedComponentsWithStats(mask, 8)
    pts = [cent[i] for i in range(1, n)
           if minarea <= stats[i, cv2.CC_STAT_AREA] <= maxarea]
    return np.array(pts, np.float32)


def refine_transform(M, ref_stars, mov_stars, tols=(8.0, 4.0, 2.0, 1.2, 0.8)):
    """Iterative nearest-neighbour star matching + RANSAC similarity fit.
    Turns the coarse ORB estimate into a sub-pixel transform."""
    M = M.copy()
    if len(ref_stars) < 8 or len(mov_stars) < 8:
        return M
    for tol in tols:
        m2 = cv2.transform(mov_stars.reshape(-1, 1, 2), M).reshape(-1, 2)
        d2 = ((m2[:, None, :] - ref_stars[None, :, :]) ** 2).sum(-1)
        j = d2.argmin(1)
        dist = np.sqrt(d2[np.arange(len(m2)), j])
        keep = dist < tol
        if keep.sum() < 10:
            break
        src = mov_stars[keep].reshape(-1, 1, 2)
        dst = ref_stars[j[keep]].reshape(-1, 1, 2)
        Mn, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC,
                                            ransacReprojThreshold=tol,
                                            maxIters=20000, confidence=0.999)
        if Mn is not None:
            M = Mn
    return M


def star_score(ref_enh, mov_enh, M, frac=0.55):
    """Normalized correlation of the high-pass fields over the sky region.
    >0.6 means the transform really locks the stars; ~0.1..0.3 means failure."""
    H, W = ref_enh.shape
    w = cv2.warpAffine(mov_enh.astype(np.float32), M, (W, H),
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
    m2 = cv2.transform(mov_stars.reshape(-1, 1, 2), M).reshape(-1, 2)
    d2 = ((m2[:, None, :] - ref_stars[None, :, :]) ** 2).sum(-1)
    d = np.sqrt(d2.min(1))
    return float(np.median(d)), float((d < tol).mean())


def transform_is_plausible(M, W, H, inliers, sc, resid_med, resid_frac):
    rot = np.degrees(np.arctan2(M[1, 0], M[0, 0]))
    tx, ty = M[0, 2], M[1, 2]
    return (abs(rot) < 15 and abs(tx) < 0.15 * W and abs(ty) < 0.15 * H
            and inliers >= 25 and sc >= 0.55 and resid_frac >= 0.20
            and resid_med < 3.0)


def register(ref_desc, ref_kp, mob_desc, mob_kp):
    """Return 2x3 similarity M mapping moving coords -> reference coords."""
    if ref_desc is None or mob_desc is None:
        return None, 0, 0
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    knn = bf.knnMatch(mob_desc, ref_desc, k=2)
    good = [m for m, n in (p for p in knn if len(p) == 2) if m.distance < 0.75 * n.distance]
    if len(good) < 8:
        return None, len(good), 0
    src = np.float32([mob_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([ref_kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    M, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC,
                                         ransacReprojThreshold=3.0, maxIters=8000,
                                         confidence=0.999)
    ninl = int(inl.sum()) if inl is not None else 0
    return M, len(good), ninl


def scale_matrix(M, s):
    """M was fit at 1/s resolution; return equivalent matrix at full res."""
    t = M[:, 2] * s
    return np.column_stack([M[:, 0:2], t]).astype(np.float32)
