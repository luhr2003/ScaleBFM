import numpy as np


class RollingHeightMap:
    """Robot-centric rolling ground height grid in the estimator frame.

    Adapted from MagicLoco's `sim2real/perception/pyscan/fusion.py` (the elevation map behind its terrain policy). A forward
    looking depth camera sees the ground ahead; the cells under the feet were observed seconds earlier, so the map is the
    memory that lets the estimator ask "how high is the ground where this foot is planted".

    Snap-and-shift: the window origin is quantised to the cell size and the arrays are rolled when the robot crosses a cell
    boundary, with the wrapped edge invalidated, so edges are never resampled and smeared. Per cell the median of one
    frame's points is slewed toward the cell (a riser first resolved at range is noisy and must not snap the map).
    """

    def __init__(self, extent=4.0, cell=0.05, min_hits=2, step_limit=0.10, memory_frames=50):
        self.cell = float(cell)
        self.n = int(round(extent / cell))
        self.min_hits = int(min_hits)
        self.step_limit = float(step_limit)
        # Each cell averages the frames that saw it (gain 1 / frames, floored at 1 / memory_frames). The map is built with the
        # estimator's own pose, so a map that keeps following new frames would follow the estimate's drift and feed it back.
        self.memory_frames = int(memory_frames)
        self.frames = np.zeros((self.n, self.n), dtype=np.int32)
        self.z = np.zeros((self.n, self.n), dtype=np.float32)
        self.hits = np.zeros((self.n, self.n), dtype=np.int32)
        self._ox = 0
        self._oy = 0
        self._init = False

    def _cell_of(self, x, y):
        return np.floor(x / self.cell).astype(np.int64), np.floor(y / self.cell).astype(np.int64)

    def recenter(self, cx, cy):
        ix, iy = int(np.floor(cx / self.cell)), int(np.floor(cy / self.cell))
        nx, ny = ix - self.n // 2, iy - self.n // 2
        if not self._init:
            self._ox, self._oy, self._init = nx, ny, True
            return
        dx, dy = nx - self._ox, ny - self._oy
        if dx == 0 and dy == 0:
            return
        if abs(dx) >= self.n or abs(dy) >= self.n:
            self.z[:], self.hits[:], self.frames[:] = 0.0, 0, 0
        else:
            self.z = np.roll(self.z, (-dx, -dy), axis=(0, 1))
            self.hits = np.roll(self.hits, (-dx, -dy), axis=(0, 1))
            self.frames = np.roll(self.frames, (-dx, -dy), axis=(0, 1))
            if dx > 0:
                self.z[-dx:, :], self.hits[-dx:, :], self.frames[-dx:, :] = 0.0, 0, 0
            elif dx < 0:
                self.z[:-dx, :], self.hits[:-dx, :], self.frames[:-dx, :] = 0.0, 0, 0
            if dy > 0:
                self.z[:, -dy:], self.hits[:, -dy:], self.frames[:, -dy:] = 0.0, 0, 0
            elif dy < 0:
                self.z[:, :-dy], self.hits[:, :-dy], self.frames[:, :-dy] = 0.0, 0, 0
        self._ox, self._oy = nx, ny

    def clear(self):
        self.z[:], self.hits[:], self.frames[:] = 0.0, 0, 0
        self._init = False

    def update(self, points_world):
        if points_world.shape[0] == 0:
            return
        ix, iy = self._cell_of(points_world[:, 0], points_world[:, 1])
        gx, gy = ix - self._ox, iy - self._oy
        ok = (gx >= 0) & (gx < self.n) & (gy >= 0) & (gy < self.n)
        if not ok.any():
            return
        gx, gy, pz = gx[ok], gy[ok], points_world[ok, 2]
        flat = gx * self.n + gy
        order = np.lexsort((pz, flat))  # median per cell: depth noise is salt-and-pepper at range
        flat_s, pz_s = flat[order], pz[order]
        starts = np.flatnonzero(np.r_[True, flat_s[1:] != flat_s[:-1]])
        counts = np.diff(np.r_[starts, flat_s.size])
        med = pz_s[starts + counts // 2]
        cells = flat_s[starts]
        cgx, cgy = cells // self.n, cells % self.n
        seen = self.hits[cgx, cgy] > 0
        cur = self.z[cgx, cgy]
        frames = np.minimum(self.frames[cgx, cgy] + 1, self.memory_frames)
        delta = np.clip((med - cur) / frames, -self.step_limit, self.step_limit)
        self.z[cgx, cgy] = np.where(seen, cur + delta, med)
        self.frames[cgx, cgy] = frames
        self.hits[cgx, cgy] = np.minimum(self.hits[cgx, cgy] + counts, 1_000_000)

    def seed(self, x, y, z, radius):
        """Declare the ground inside a disc to be at height z (e.g. the flat patch under the robot at calibration)."""
        ix0, iy0 = int(np.floor((x - radius) / self.cell)) - self._ox, int(np.floor((y - radius) / self.cell)) - self._oy
        span = int(np.ceil(2 * radius / self.cell)) + 1
        for gx in range(max(ix0, 0), min(ix0 + span, self.n)):
            for gy in range(max(iy0, 0), min(iy0 + span, self.n)):
                cx, cy = (gx + self._ox + 0.5) * self.cell, (gy + self._oy + 0.5) * self.cell
                if (cx - x) ** 2 + (cy - y) ** 2 <= radius ** 2:
                    self.z[gx, gy] = z
                    self.hits[gx, gy] = max(self.hits[gx, gy], self.min_hits)
                    self.frames[gx, gy] = self.memory_frames  # a calibration patch is a trusted reference

    def query(self, x, y, r=1):
        """Median ground height of the trusted cells around (x, y), or None if there are none."""
        ix, iy = int(np.floor(x / self.cell)), int(np.floor(y / self.cell))
        gx, gy = ix - self._ox, iy - self._oy
        x0, x1 = max(gx - r, 0), min(gx + r + 1, self.n)
        y0, y1 = max(gy - r, 0), min(gy + r + 1, self.n)
        if x0 >= x1 or y0 >= y1:
            return None
        good = self.hits[x0:x1, y0:y1] >= self.min_hits
        if not good.any():
            return None
        return float(np.median(self.z[x0:x1, y0:y1][good]))
