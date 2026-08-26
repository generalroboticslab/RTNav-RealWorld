"""Point-cloud accumulator: voxels with temporal confirmation and decay."""
import numpy as np
from scipy import ndimage

VOXEL_SIZE_M = 0.05
MAX_VOXELS = 500_000
MIN_HITS = 5              # frames a voxel must be seen in before it is drawn
MAX_HITS = MIN_HITS + 10  # cap, or decay can never win
MISS_PENALTY = 1
FREE_MARGIN_M = 0.10      # how far in front of the surface counts as seen-through
MIN_CLUSTER_VOXELS = 50   # drop disconnected floating groups

MIN_Z_M = -0.2            # height band kept: drops floor and ceiling
MAX_Z_M = 1.8
EDGE_M_PER_PX = 0.1   # reject depth discontinuities: flying pixels
ERODE_PX = 1          # then clear the one-pixel fringe they leave
MIN_NORMAL_RAY_COS = 0.30  # reject surfaces viewed above ~73 degrees incidence

_OFFSET = 1 << 20
_MASK = (1 << 21) - 1


def reject_flying_pixels(cam):
    """Keep-mask over points_world rows for reliable stereo depth."""
    d = np.asarray(cam.depth, np.float32)
    valid = (np.isfinite(d) & (d > max(cam.min_depth_m, 0.0) + 1e-3)
             & (d < cam.max_depth_m - 1e-3))
    gx = np.zeros_like(d)
    gy = np.zeros_like(d)
    gx[:, 1:-1] = 0.5 * (d[:, 2:] - d[:, :-2])
    gy[1:-1, :] = 0.5 * (d[2:, :] - d[:-2, :])
    keep = valid & (np.hypot(gx, gy) < EDGE_M_PER_PX)

    K = np.asarray(cam.intrinsics, np.float32)
    v, u = np.indices(d.shape, dtype=np.float32)
    xyz = np.stack(((u - K[0, 2]) * d / K[0, 0],
                    (v - K[1, 2]) * d / K[1, 1], d), axis=-1)
    dx = xyz[1:-1, 2:] - xyz[1:-1, :-2]
    dy = xyz[2:, 1:-1] - xyz[:-2, 1:-1]
    normal = np.cross(dx, dy)
    ray = xyz[1:-1, 1:-1]
    denom = np.linalg.norm(normal, axis=2) * np.linalg.norm(ray, axis=2)
    incidence = np.zeros_like(valid)
    with np.errstate(divide="ignore", invalid="ignore"):
        cosine = np.abs(np.sum(normal * ray, axis=2)) / denom
    neighbors_valid = (valid[1:-1, 1:-1]
                       & valid[1:-1, 2:] & valid[1:-1, :-2]
                       & valid[2:, 1:-1] & valid[:-2, 1:-1])
    incidence[1:-1, 1:-1] = neighbors_valid & (cosine >= MIN_NORMAL_RAY_COS)
    keep &= incidence
    return ndimage.binary_erosion(keep, iterations=ERODE_PX)[valid]


def keep_in_height_band(pts):
    """Keep-mask for points inside [MIN_Z_M, MAX_Z_M]: drops ceiling and floor."""
    return (pts[:, 2] >= MIN_Z_M) & (pts[:, 2] <= MAX_Z_M)


def colors_by_projection(pts, cam):
    """Sample RGB by projecting world points back into the image.

    Independent of row order, so it survives any upstream point filtering.
    """
    K = np.asarray(cam.intrinsics, np.float64)
    p = (np.linalg.inv(np.asarray(cam.T_world_cam, np.float64))
         @ np.c_[pts, np.ones(len(pts))].T).T[:, :3]
    z = np.where(p[:, 2] > 1e-6, p[:, 2], 1e-6)
    u = np.clip(K[0, 0] * p[:, 0] / z + K[0, 2], 0, cam.rgb.shape[1] - 1)
    v = np.clip(K[1, 1] * p[:, 1] / z + K[1, 2], 0, cam.rgb.shape[0] - 1)
    return cam.rgb[v.astype(np.int64), u.astype(np.int64)]


class VoxelGrid:
    """Voxels keyed by a packed int64 id, newest colour wins.

    A voxel is drawn once seen in MIN_HITS frames, and loses hits whenever the
    camera sees through it, so noise and stale geometry expire instead of
    accumulating forever.
    """

    def __init__(self):
        self.ids = np.empty(0, np.int64)
        self.cols = np.empty((0, 3), np.uint8)
        self.age = np.empty(0, np.int64)
        self.hits = np.empty(0, np.int64)
        self.frame = 0

    def add(self, pts, cols, cam):
        self.frame += 1
        q = np.floor(np.asarray(pts) / VOXEL_SIZE_M).astype(np.int64) + _OFFSET
        ids = (q[:, 0] << 42) | (q[:, 1] << 21) | q[:, 2]

        # collapse this frame first: many points in one voxel are one vote
        ids, first_in_frame = np.unique(ids, return_index=True)
        cols = np.asarray(cols, np.uint8)[first_in_frame]

        # new first, so np.unique's first-occurrence keeps this frame's colour
        all_ids = np.concatenate([ids, self.ids])
        all_cols = np.concatenate([cols, self.cols])
        all_age = np.concatenate([np.full(len(ids), self.frame), self.age])
        all_hits = np.concatenate([np.ones(len(ids), np.int64), self.hits])

        self.ids, first, inverse = np.unique(
            all_ids, return_index=True, return_inverse=True)
        self.cols, self.age = all_cols[first], all_age[first]
        self.hits = np.bincount(inverse, weights=all_hits).astype(np.int64)
        np.minimum(self.hits, MAX_HITS, out=self.hits)

        self.clear_seen_through(cam)
        if len(self.ids) > MAX_VOXELS:
            keep = np.argpartition(self.age, -MAX_VOXELS)[-MAX_VOXELS:]
            self._select(keep)

        keep = self.confirmed() & self.in_large_cluster()
        return self.centers()[keep], self.cols[keep]

    def confirmed(self):
        """Mask of voxels seen in at least MIN_HITS frames — rejects flicker."""
        return self.hits >= MIN_HITS

    def clear_seen_through(self, cam):
        """Decay voxels in the frustum that sit in front of the measured surface.

        The camera looked through that space and found nothing. Voxels behind
        the surface are occluded, carry no information, and are left alone.
        """
        K = np.asarray(cam.intrinsics, np.float64)
        depth = np.asarray(cam.depth, np.float32)
        h, w = depth.shape[:2]

        centers = self.centers()
        p = (np.linalg.inv(np.asarray(cam.T_world_cam, np.float64))
             @ np.c_[centers, np.ones(len(centers))].T).T[:, :3]
        z = p[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = np.round(K[0, 0] * p[:, 0] / z + K[0, 2])
            v = np.round(K[1, 1] * p[:, 1] / z + K[1, 2])

        in_view = ((z > cam.min_depth_m) & (z < cam.max_depth_m)
                   & (u >= 0) & (u < w) & (v >= 0) & (v < h))
        surface = np.zeros(len(z))
        surface[in_view] = depth[v[in_view].astype(np.int64),
                                 u[in_view].astype(np.int64)]

        seen_through = (in_view & (surface > 0) & (z < surface - FREE_MARGIN_M)
                        & (self.age != self.frame))
        self.hits[seen_through] -= MISS_PENALTY
        self._select(self.hits > 0)

    def in_large_cluster(self):
        """Mask of voxels in a 26-connected component of at least MIN_CLUSTER_VOXELS.

        Drops small floating blobs of noise; real surfaces are one big component.
        """
        q = np.stack([self.ids >> 42, (self.ids >> 21) & _MASK, self.ids & _MASK], 1)
        lo = q.min(0)
        vol = np.zeros(q.max(0) - lo + 1, bool)
        idx = (q[:, 0] - lo[0], q[:, 1] - lo[1], q[:, 2] - lo[2])
        vol[idx] = True
        labels, _ = ndimage.label(vol, structure=np.ones((3, 3, 3)))
        return np.bincount(labels.ravel())[labels[idx]] >= MIN_CLUSTER_VOXELS

    def centers(self):
        q = np.stack([self.ids >> 42, (self.ids >> 21) & _MASK, self.ids & _MASK], 1)
        return (q - _OFFSET + 0.5) * VOXEL_SIZE_M

    def _select(self, keep):
        self.ids, self.cols = self.ids[keep], self.cols[keep]
        self.age, self.hits = self.age[keep], self.hits[keep]
