"""Images for its optical nodes: decoding, what its sight node sees in them, and the glyphs its reader reads.

Decoding needs nothing but numpy: PNG (all colour types, 8 and 16 bit, not interlaced), PPM/PGM, BMP and baseline
JPEG (what the Hugging Face dataset viewer serves). With Pillow installed it also reads progressive JPEG, GIF, WebP
and interlaced PNG, and JPEG goes to Pillow (it is faster); it is optional, as everything beyond numpy.

    features(rgb)  -> what the sight node takes in: a small grey thumbnail, a colour histogram, a histogram of edge
                      directions in a grid (the shapes in it), and a few overall measures
    lines(rgb)     -> the text lines in an image, each a list of glyphs (characters, found by the ink in each column)
                      with the gaps between words
    glyph_text()   -> a glyph as the text a node learns from: "glyph:<16x16 bits>:<height>:<top>:<aspect>:<bottom>",
                      the last four relative to its line, so a node can tell "o" from "O" and "," from "'"
"""
from __future__ import annotations

import io
import struct
import zlib

import numpy as np

GLYPH = 16  # glyphs are read at 16 x 16


class ImageError(ValueError):
    pass


# ============================================================================ decoding
def sniff(data: bytes) -> str:
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if data[:3] == b"\xff\xd8\xff":
        return "jpeg"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    if data[:2] == b"BM":
        return "bmp"
    if data[:2] in (b"P5", b"P6"):
        return "pnm"
    return ""


def decode(data: bytes) -> np.ndarray:
    """Image bytes -> an H x W x 3 uint8 RGB array (transparency laid on white)."""
    kind = sniff(data)
    try:
        if kind == "png":
            return _png(data)
        if kind == "pnm":
            return _pnm(data)
        if kind == "bmp":
            return _bmp(data)
    except ImageError as exc:
        if "interlaced" not in str(exc):
            raise
    try:
        from PIL import Image  # optional
    except ImportError:
        if kind == "jpeg":
            return _jpeg(data)
        raise ImageError(f"{kind or 'this'} images need Pillow (pip install pillow); PNG, BMP, PPM and baseline JPEG work "
                         "without it")
    try:
        im = Image.open(io.BytesIO(data))
        im.load()
    except Exception as exc:
        raise ImageError(f"not an image it can read ({exc})")
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        im = Image.alpha_composite(bg, im)
    return np.asarray(im.convert("RGB"), dtype=np.uint8).copy()


def _paeth(a: int, b: int, c: int) -> int:
    p = a + b - c
    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    return a if pa <= pb and pa <= pc else b if pb <= pc else c


def _png(data: bytes) -> np.ndarray:
    pos, idat, plte, trns = 8, [], None, None
    w = h = depth = ctype = interlace = None
    while pos + 8 <= len(data):
        length, tag = struct.unpack(">I4s", data[pos:pos + 8])
        chunk = data[pos + 8:pos + 8 + length]
        pos += 12 + length
        if tag == b"IHDR":
            w, h, depth, ctype, _, _, interlace = struct.unpack(">IIBBBBB", chunk)
        elif tag == b"PLTE":
            plte = np.frombuffer(chunk, dtype=np.uint8).reshape(-1, 3)
        elif tag == b"tRNS":
            trns = chunk
        elif tag == b"IDAT":
            idat.append(chunk)
        elif tag == b"IEND":
            break
    if w is None:
        raise ImageError("a PNG with no header")
    if interlace:
        raise ImageError("interlaced PNG")
    if w * h > 40_000_000:
        raise ImageError("too large")
    channels = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}[ctype]
    bits = depth * channels
    bpp = max(1, bits // 8)
    stride = (w * bits + 7) // 8
    raw = zlib.decompress(b"".join(idat))
    out = np.zeros((h, stride), dtype=np.uint8)
    prev = np.zeros(stride, dtype=np.int32)
    for y in range(h):
        f = raw[y * (stride + 1)]
        line = np.frombuffer(raw, dtype=np.uint8, count=stride, offset=y * (stride + 1) + 1).astype(np.int32)
        if f == 0:
            cur = line
        elif f == 1:  # Sub: a running sum per channel
            cur = line.copy()
            for i in range(bpp):
                cur[i::bpp] = np.cumsum(line[i::bpp]) & 0xFF
        elif f == 2:
            cur = (line + prev) & 0xFF
        elif f == 3:
            cur, lv, pv = line.tolist(), line.tolist(), prev.tolist()
            for i in range(stride):
                left = cur[i - bpp] if i >= bpp else 0
                cur[i] = (lv[i] + ((left + pv[i]) >> 1)) & 0xFF
            cur = np.array(cur, dtype=np.int32)
        elif f == 4:
            cur, lv, pv = line.tolist(), line.tolist(), prev.tolist()
            for i in range(stride):
                if i >= bpp:
                    cur[i] = (lv[i] + _paeth(cur[i - bpp], pv[i], pv[i - bpp])) & 0xFF
                else:
                    cur[i] = (lv[i] + pv[i]) & 0xFF
            cur = np.array(cur, dtype=np.int32)
        else:
            raise ImageError(f"unknown PNG filter {f}")
        out[y] = cur
        prev = cur
    if depth == 16:
        px = out.reshape(h, w, channels * 2)[:, :, 0::2]
    elif depth == 8:
        px = out.reshape(h, w, channels)
    else:  # 1, 2 or 4 bits per sample (grey or palette)
        per = 8 // depth
        unpacked = np.zeros((h, stride * per), dtype=np.uint8)
        for k in range(per):
            unpacked[:, k::per] = (out >> (8 - depth * (k + 1))) & ((1 << depth) - 1)
        px = unpacked[:, :w, None]
        if ctype == 0:
            px = (px.astype(np.int32) * 255 // ((1 << depth) - 1)).astype(np.uint8)
    if ctype == 3:
        if plte is None:
            raise ImageError("a palette PNG with no palette")
        idx = px[:, :, 0]
        rgb = plte[np.clip(idx, 0, len(plte) - 1)]
        if trns:
            alpha = np.full(len(plte), 255, dtype=np.uint8)
            alpha[:len(trns)] = np.frombuffer(trns, dtype=np.uint8)[:len(plte)]
            return _on_white(rgb, alpha[idx])
        return rgb.astype(np.uint8)
    if ctype == 0:
        return np.repeat(px, 3, axis=2).astype(np.uint8)
    if ctype == 4:
        return _on_white(np.repeat(px[:, :, :1], 3, axis=2), px[:, :, 1])
    if ctype == 6:
        return _on_white(px[:, :, :3], px[:, :, 3])
    return px[:, :, :3].astype(np.uint8)


def _on_white(rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    a = alpha.astype(np.float32)[:, :, None] / 255.0
    return (rgb.astype(np.float32) * a + 255.0 * (1 - a)).round().astype(np.uint8)


def _pnm(data: bytes) -> np.ndarray:
    parts, pos = [], 2
    while len(parts) < 3:
        while data[pos:pos + 1].isspace():
            pos += 1
        if data[pos:pos + 1] == b"#":
            pos = data.index(b"\n", pos) + 1
            continue
        end = pos
        while not data[end:end + 1].isspace():
            end += 1
        parts.append(int(data[pos:end]))
        pos = end
    w, h, maxval = parts
    pos += 1
    ch = 3 if data[:2] == b"P6" else 1
    dtype = np.uint8 if maxval < 256 else np.dtype(">u2")
    px = np.frombuffer(data, dtype=dtype, count=w * h * ch, offset=pos).reshape(h, w, ch)
    if maxval >= 256:
        px = (px.astype(np.int32) * 255 // maxval).astype(np.uint8)
    return np.repeat(px, 3, axis=2) if ch == 1 else px.astype(np.uint8)


def _bmp(data: bytes) -> np.ndarray:
    offset = struct.unpack("<I", data[10:14])[0]
    w, h = struct.unpack("<ii", data[18:26])
    bits, comp = struct.unpack("<H", data[28:30])[0], struct.unpack("<I", data[30:34])[0]
    if bits not in (24, 32) or comp not in (0, 3):
        raise ImageError("only 24 and 32 bit uncompressed BMP without Pillow")
    bpp = bits // 8
    stride = (w * bpp + 3) & ~3
    rows = np.frombuffer(data, dtype=np.uint8, count=stride * abs(h), offset=offset).reshape(abs(h), stride)
    px = rows[:, :w * bpp].reshape(abs(h), w, bpp)[:, :, 2::-1]
    return (px[::-1] if h > 0 else px).copy()


# ---------------------------------------------------------------------------- baseline JPEG
_ZIGZAG = np.array([0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6,
                    7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59, 52, 45, 38, 31,
                    39, 46, 53, 60, 61, 54, 47, 55, 62, 63])  # the k-th coefficient in a scan -> its place in the block
_IDCT = np.array([[(np.sqrt(0.5) if u == 0 else 1.0) / 2 * np.cos((2 * x + 1) * u * np.pi / 16) for u in range(8)]
                  for x in range(8)])


def _huffman(counts: bytes, symbols: bytes) -> tuple[list, list]:
    """A Huffman table as two lookups by the next 16 bits: the symbol, and how many bits its code took."""
    sym = np.zeros(65536, dtype=np.int32)
    length = np.zeros(65536, dtype=np.int32)
    code, k = 0, 0
    for n in range(1, 17):
        for _ in range(counts[n - 1]):
            lo, hi = code << (16 - n), (code + 1) << (16 - n)
            sym[lo:hi], length[lo:hi] = symbols[k], n
            code += 1
            k += 1
        code <<= 1
    return sym.tolist(), length.tolist()


def _windows(seg: bytes) -> list:
    """For every bit of an entropy-coded segment (stuffing removed), the 16 bits from there on (padded with ones)."""
    bits = np.unpackbits(np.frombuffer(seg.replace(b"\xff\x00", b"\xff"), dtype=np.uint8))
    bits = np.concatenate([bits, np.ones(32, dtype=np.uint8)])
    w = np.lib.stride_tricks.sliding_window_view(bits, 16).astype(np.int64) @ (1 << np.arange(15, -1, -1, dtype=np.int64))
    return w.tolist()


def _extend(v: int, s: int) -> int:
    return v - (1 << s) + 1 if s and v < (1 << (s - 1)) else v


def _jpeg(data: bytes) -> np.ndarray:
    """A baseline (sequential, Huffman) JPEG: 8-bit, grey or YCbCr, any sampling, restart markers. Progressive and
    arithmetic-coded JPEG need Pillow."""
    qt: dict[int, np.ndarray] = {}
    dc_t: dict[int, tuple] = {}
    ac_t: dict[int, tuple] = {}
    frame: dict | None = None
    restart, pos, n = 0, 2, len(data)
    if data[:2] != b"\xff\xd8":
        raise ImageError("not a JPEG")
    while pos < n - 1:
        if data[pos] != 0xFF:
            raise ImageError("a broken JPEG (no marker where one belongs)")
        marker = data[pos + 1]
        pos += 2
        if marker == 0xFF:
            pos -= 1
            continue
        if marker == 0xD9:
            break
        if 0xD0 <= marker <= 0xD7 or marker in (0x01, 0xD8):
            continue
        if pos + 2 > n:
            break
        size = struct.unpack(">H", data[pos:pos + 2])[0]
        seg = data[pos + 2:pos + size]
        pos += size
        if marker == 0xDB:
            i = 0
            while i < len(seg):
                pq, tq = seg[i] >> 4, seg[i] & 15
                i += 1
                vals = struct.unpack(">64H", seg[i:i + 128]) if pq else tuple(seg[i:i + 64])
                i += 128 if pq else 64
                q = np.zeros(64)
                q[_ZIGZAG] = vals
                qt[tq] = q
        elif marker == 0xC4:
            i = 0
            while i + 17 <= len(seg):
                tc, th = seg[i] >> 4, seg[i] & 15
                counts = seg[i + 1:i + 17]
                total = sum(counts)
                (ac_t if tc else dc_t)[th] = _huffman(counts, seg[i + 17:i + 17 + total])
                i += 17 + total
        elif marker in (0xC0, 0xC1):
            if seg[0] != 8:
                raise ImageError(f"{seg[0]}-bit JPEG needs Pillow")
            h, w = struct.unpack(">HH", seg[1:5])
            comps = [{"id": seg[6 + 3 * k], "h": seg[7 + 3 * k] >> 4, "v": seg[7 + 3 * k] & 15, "q": seg[8 + 3 * k]}
                     for k in range(seg[5])]
            if not h or not w or not comps or any(not c["h"] or not c["v"] for c in comps):
                raise ImageError("a JPEG with no size")
            if len(comps) not in (1, 3):
                raise ImageError("CMYK JPEG needs Pillow")
            hmax, vmax = max(c["h"] for c in comps), max(c["v"] for c in comps)
            mx, my = -(-w // (8 * hmax)), -(-h // (8 * vmax))
            for c in comps:
                c["bw"], c["bh"] = mx * c["h"], my * c["v"]
                c["coef"] = np.zeros((c["bh"], c["bw"], 64), dtype=np.float64)
            frame = {"h": h, "w": w, "comps": comps, "hmax": hmax, "vmax": vmax, "mx": mx, "my": my}
        elif 0xC2 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
            raise ImageError("progressive or arithmetic-coded JPEG needs Pillow (pip install pillow)")
        elif marker == 0xDD:
            restart = struct.unpack(">H", seg[:2])[0]
        elif marker == 0xDA:
            if frame is None:
                raise ImageError("a JPEG scan before its frame")
            ids = {c["id"]: c for c in frame["comps"]}
            scomps = []
            for k in range(seg[0]):
                c = ids.get(seg[1 + 2 * k])
                if c is None:
                    raise ImageError("a JPEG scan of a component it does not have")
                scomps.append((c, seg[2 + 2 * k] >> 4, seg[2 + 2 * k] & 15))
            end = pos  # the entropy-coded data runs to the next marker that is not stuffing or a restart
            while True:
                end = data.find(b"\xff", end)
                if end < 0 or end + 1 >= n:
                    end = n
                    break
                if data[end + 1] == 0 or 0xD0 <= data[end + 1] <= 0xD7:
                    end += 2
                    continue
                break
            _scan(data[pos:end], frame, scomps, dc_t, ac_t, restart)
            pos = end
    if frame is None:
        raise ImageError("a JPEG with no frame")
    planes = []
    for c in frame["comps"]:
        if c["q"] not in qt:
            raise ImageError("a JPEG without its quantisation table")
        blocks = (c["coef"] * qt[c["q"]]).reshape(c["bh"], c["bw"], 8, 8)
        px = np.einsum("xu,abuv,yv->abxy", _IDCT, blocks, _IDCT) + 128.0
        plane = px.transpose(0, 2, 1, 3).reshape(c["bh"] * 8, c["bw"] * 8)
        plane = _upsample(_upsample(plane, frame["vmax"] // c["v"], 0), frame["hmax"] // c["h"], 1)
        planes.append(plane[:frame["h"], :frame["w"]])
    if len(planes) == 1:
        return np.repeat(np.clip(planes[0], 0, 255).round().astype(np.uint8)[:, :, None], 3, axis=2)
    y, cb, cr = planes[0], planes[1] - 128.0, planes[2] - 128.0
    rgb = np.stack([y + 1.402 * cr, y - 0.344136 * cb - 0.714136 * cr, y + 1.772 * cb], axis=2)
    return np.clip(rgb, 0, 255).round().astype(np.uint8)


def _upsample(plane: np.ndarray, f: int, axis: int) -> np.ndarray:
    """Chroma at full size: twice as many samples, each 3/4 its own and 1/4 its neighbour (as libjpeg does), or repeated."""
    if f <= 1:
        return plane
    if f != 2:
        return np.repeat(plane, f, axis=axis)
    a = np.moveaxis(plane, axis, 0)
    before = np.concatenate([a[:1], a[:-1]])
    after = np.concatenate([a[1:], a[-1:]])
    out = np.empty((a.shape[0] * 2,) + a.shape[1:])
    out[0::2] = 0.75 * a + 0.25 * before
    out[1::2] = 0.75 * a + 0.25 * after
    return np.moveaxis(out, 0, axis)


def _scan(raw: bytes, frame: dict, scomps: list, dc_t: dict, ac_t: dict, restart: int) -> None:
    """Decode one baseline scan's coefficients into its components (zig-zag order undone)."""
    zz = _ZIGZAG.tolist()
    if len(scomps) == 1:  # one component alone: its blocks in rows, one block an MCU
        c = scomps[0][0]
        cols = -(-(-(-frame["w"] * c["h"] // frame["hmax"])) // 8)
        rows = -(-(-(-frame["h"] * c["v"] // frame["vmax"])) // 8)
        units = [[(scomps[0], by, bx)] for by in range(rows) for bx in range(cols)]
    else:
        units = [[(sc, my * sc[0]["v"] + y, mx * sc[0]["h"] + x) for sc in scomps for y in range(sc[0]["v"]) for x in range(sc[0]["h"])]
                 for my in range(frame["my"]) for mx in range(frame["mx"])]
    parts, i = [], 0
    while True:  # the segments between restart markers
        j = raw.find(b"\xff", i)
        while j >= 0 and j + 1 < len(raw) and raw[j + 1] == 0:
            j = raw.find(b"\xff", j + 2)
        if j < 0 or j + 1 >= len(raw):
            parts.append(raw[i:])
            break
        parts.append(raw[i:j])
        i = j + 2
    per = restart or len(units)
    for k, seg in enumerate(parts):
        todo = units[k * per:(k + 1) * per]
        if not todo:
            break
        win = _windows(seg)
        p = 0
        pred = {id(sc[0]): 0 for sc in scomps}
        for unit in todo:
            for (c, td, ta), by, bx in unit:
                dsym, dlen = dc_t[td]
                asym, alen = ac_t[ta]
                coef = c["coef"][by, bx] if by < c["bh"] and bx < c["bw"] else np.zeros(64)
                v = win[p]
                t = dsym[v]
                p += dlen[v]
                diff = _extend(win[p] >> (16 - t), t) if t else 0
                p += t
                pred[id(c)] += diff
                coef[0] = pred[id(c)]
                kk = 1
                while kk < 64:
                    v = win[p]
                    rs = asym[v]
                    p += alen[v]
                    r, s = rs >> 4, rs & 15
                    if not s:
                        if r != 15:
                            break
                        kk += 16
                        continue
                    kk += r
                    if kk > 63:
                        break
                    coef[zz[kk]] = _extend(win[p] >> (16 - s), s)
                    p += s
                    kk += 1


def encode_png(rgb: np.ndarray) -> bytes:
    """An RGB (or grey) array as PNG bytes (for thumbnails and tests)."""
    a = np.asarray(rgb, dtype=np.uint8)
    if a.ndim == 2:
        a = np.repeat(a[:, :, None], 3, axis=2)
    h, w = a.shape[:2]
    raw = b"".join(b"\x00" + a[y].tobytes() for y in range(h))

    def chunk(tag: bytes, body: bytes) -> bytes:
        return struct.pack(">I", len(body)) + tag + body + struct.pack(">I", zlib.crc32(tag + body) & 0xFFFFFFFF)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


# ============================================================================ what it sees
def gray(rgb: np.ndarray) -> np.ndarray:
    return rgb[:, :, :3].astype(np.float32) @ np.array([0.299, 0.587, 0.114], dtype=np.float32)


def shrink(a: np.ndarray, h: int, w: int) -> np.ndarray:
    """Area-average resize of a 2-D (or 3-D) array to h x w."""
    H, W = a.shape[:2]
    ys = (np.arange(h + 1) * H / h).astype(int)
    xs = (np.arange(w + 1) * W / w).astype(int)
    ys[-1], xs[-1] = H, W
    c = np.cumsum(np.cumsum(np.pad(a.astype(np.float64), ((1, 0), (1, 0)) + ((0, 0),) * (a.ndim - 2)), 0), 1)
    y0, y1 = ys[:-1], np.maximum(ys[1:], ys[:-1] + 1)
    x0, x1 = xs[:-1], np.maximum(xs[1:], xs[:-1] + 1)
    y1, x1 = np.minimum(y1, H), np.minimum(x1, W)
    s = c[y1][:, x1] - c[y0][:, x1] - c[y1][:, x0] + c[y0][:, x0]
    area = ((y1 - y0)[:, None] * (x1 - x0)[None, :]).astype(np.float64)
    if a.ndim == 3:
        area = area[:, :, None]
    return (s / area).astype(np.float32)


def features(rgb: np.ndarray) -> np.ndarray:
    """What the sight node takes in (415 numbers): a 16 x 16 grey thumbnail, a colour histogram, edge directions
    in a 4 x 4 grid, and overall measures."""
    img = rgb[:, :, :3].astype(np.float32) / 255.0
    g = gray(rgb) / 255.0
    thumb = shrink(g, 16, 16).ravel()
    thumb = (thumb - thumb.mean()) / (thumb.std() + 1e-6)
    mx, mn = img.max(axis=2), img.min(axis=2)
    sat = np.where(mx > 0, (mx - mn) / (mx + 1e-6), 0.0)
    r, gg, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    d = mx - mn + 1e-6
    hue = np.where(mx == r, ((gg - b) / d) % 6, np.where(mx == gg, (b - r) / d + 2, (r - gg) / d + 4)) / 6.0
    hue_hist = np.bincount(np.minimum((hue * 12).astype(int), 11).ravel(), weights=sat.ravel(), minlength=12)
    val_hist = np.bincount(np.minimum((mx * 4).astype(int), 3).ravel(), minlength=4).astype(np.float64)
    sat_hist = np.bincount(np.minimum((sat * 4).astype(int), 3).ravel(), minlength=4).astype(np.float64)
    n = float(g.size)
    colour = np.concatenate([hue_hist / n, val_hist / n, sat_hist / n])
    small = shrink(g, 64, 64)
    gx = np.zeros_like(small)
    gy = np.zeros_like(small)
    gx[:, 1:-1] = small[:, 2:] - small[:, :-2]
    gy[1:-1, :] = small[2:, :] - small[:-2, :]
    mag = np.hypot(gx, gy)
    ang = (np.arctan2(gy, gx) % np.pi) / np.pi
    bins = np.minimum((ang * 8).astype(int), 7)
    hog = np.zeros((4, 4, 8), dtype=np.float64)
    for cy in range(4):
        for cx in range(4):
            sl = (slice(cy * 16, cy * 16 + 16), slice(cx * 16, cx * 16 + 16))
            hog[cy, cx] = np.bincount(bins[sl].ravel(), weights=mag[sl].ravel(), minlength=8)
    hog = hog.ravel()
    hog = hog / (np.linalg.norm(hog) + 1e-6)
    h, w = g.shape
    overall = np.array([*img.reshape(-1, 3).mean(0), *img.reshape(-1, 3).std(0), float((mag > 0.1).mean()),
                        np.log(w / max(h, 1)), float((g > 0.92).mean()), float((g < 0.08).mean()), float(sat.mean())])
    return np.concatenate([thumb * 0.25, colour * 2.0, hog, overall]).astype(np.float32)


def fold(vec: np.ndarray, dim: int) -> np.ndarray:
    """A feature vector laid into the encoder's width, unit length (what every node takes in)."""
    out = np.zeros(dim, dtype=np.float32)
    idx = np.arange(len(vec)) % dim
    np.add.at(out, idx, vec)
    norm = float(np.linalg.norm(out))
    return out / norm if norm > 0 else out


# ============================================================================ reading
def binarize(rgb: np.ndarray) -> np.ndarray:
    """Ink (True) and paper: Otsu's threshold, the minority side is ink."""
    g = gray(rgb)
    hist = np.bincount(g.astype(np.uint8).ravel(), minlength=256).astype(np.float64)
    total = hist.sum()
    cum, mean_cum = np.cumsum(hist), np.cumsum(hist * np.arange(256))
    w0, w1 = cum, total - cum
    m0 = np.where(w0 > 0, mean_cum / np.maximum(w0, 1), 0)
    m1 = np.where(w1 > 0, (mean_cum[-1] - mean_cum) / np.maximum(w1, 1), 0)
    t = int(np.argmax(w0 * w1 * (m0 - m1) ** 2))
    dark = g <= t
    return dark if dark.mean() <= 0.5 else ~dark


def _runs(mask: np.ndarray, min_gap: int = 1) -> list[tuple[int, int]]:
    """[start, end) runs where mask is true, joining runs closer than min_gap."""
    out, start = [], None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            out.append([start, i])
            start = None
    if start is not None:
        out.append([start, len(mask)])
    merged = []
    for r in out:
        if merged and r[0] - merged[-1][1] < min_gap:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    return [tuple(r) for r in merged]


def lines(rgb: np.ndarray, ink: np.ndarray | None = None) -> list[dict]:
    """Text lines: [{"top", "bottom", "glyphs": [{"x0", "x1", "y0", "y1", "space_before"}]}]."""
    ink = binarize(rgb) if ink is None else ink
    H, W = ink.shape
    rows = ink.sum(axis=1)
    out = []
    for top, bottom in _runs(rows > max(1, 0.002 * W), min_gap=2):
        lh = bottom - top
        if lh < 6 or lh > 0.6 * H and H > 60:
            continue
        band = ink[top:bottom]
        boxes = _split_wide(band, components(band), lh)
        if not boxes:
            continue
        gaps = [max(0, b[0] - a[1]) for a, b in zip(boxes, boxes[1:])]
        word_gap = max(0.28 * lh, 2.0 * float(np.median(gaps)) if gaps else 0.0)
        glyphs = []
        for i, (x0, x1, y0, y1) in enumerate(boxes):
            glyphs.append({"x0": int(x0), "x1": int(x1), "y0": int(top + y0), "y1": int(top + y1),
                           "space_before": bool(i and gaps[i - 1] > word_gap)})
        out.append({"top": int(top), "bottom": int(bottom), "glyphs": glyphs})
    return out


def components(band: np.ndarray) -> list[tuple[int, int, int, int]]:
    """The connected pieces of ink in a line (8-connected, found run by run), as (x0, x1, y0, y1) boxes, with
    the pieces of one character joined: the dot of an i, the bars of =, the parts of % and :."""
    parent: list[int] = []

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    runs: list[tuple[int, int, int]] = []  # (y, x0, x1)
    prev: list[int] = []
    for y in range(band.shape[0]):
        row = band[y]
        if not row.any():
            prev = []
            continue
        edges = np.flatnonzero(np.diff(np.concatenate(([0], row.view(np.int8) if row.dtype == bool else row.astype(np.int8), [0]))))
        cur = []
        for x0, x1 in zip(edges[0::2], edges[1::2]):
            k = len(runs)
            runs.append((y, int(x0), int(x1)))
            parent.append(k)
            for j in prev:
                _, a0, a1 = runs[j]
                if a0 <= x1 and x0 <= a1:  # touching, diagonals included
                    ra, rb = find(j), find(k)
                    if ra != rb:
                        parent[rb] = ra
            cur.append(k)
        prev = cur
    boxes: dict[int, list[int]] = {}
    for k, (y, x0, x1) in enumerate(runs):
        r = find(k)
        b = boxes.setdefault(r, [x0, x1, y, y + 1])
        b[0], b[1], b[2], b[3] = min(b[0], x0), max(b[1], x1), min(b[2], y), max(b[3], y + 1)
    items = sorted(boxes.values(), key=lambda b: (b[0], b[2]))
    merged: list[list[int]] = []
    for b in items:  # pieces stacked over each other (not side by side) are one character
        for m in merged[::-1][:3]:
            overlap = min(m[1], b[1]) - max(m[0], b[0])
            stacked = min(m[3], b[3]) - max(m[2], b[2]) <= 1
            if stacked and overlap > 0.5 * min(m[1] - m[0], b[1] - b[0]):
                m[0], m[1], m[2], m[3] = min(m[0], b[0]), max(m[1], b[1]), min(m[2], b[2]), max(m[3], b[3])
                break
        else:
            merged.append(list(b))
    return sorted((tuple(m) for m in merged), key=lambda m: m[0])


def _split_wide(band: np.ndarray, boxes: list, lh: int) -> list:
    """Characters that touch make one wide piece: split it where the least ink is."""
    out = []
    for x0, x1, y0, y1 in boxes:
        stack = [(x0, x1)]
        while stack:
            a, b = stack.pop()
            if b - a > 1.5 * lh and b - a >= 10:
                prof = band[y0:y1, a:b].sum(axis=0)
                lo, hi = int((b - a) * 0.3), int((b - a) * 0.7)
                cut = a + lo + int(np.argmin(prof[lo:hi])) if hi > lo else (a + b) // 2
                stack += [(cut, b), (a, cut)]
            else:
                ys = np.flatnonzero(band[y0:y1, a:b].any(axis=1))
                if len(ys):
                    out.append((a, b, y0 + int(ys[0]), y0 + int(ys[-1]) + 1))
    return sorted(out)


def glyph_bits(ink: np.ndarray, g: dict) -> np.ndarray:
    """A glyph's ink, scaled into a 16 x 16 square (aspect kept, centred)."""
    crop = ink[g["y0"]:g["y1"], g["x0"]:g["x1"]].astype(np.float32)
    h, w = crop.shape
    side = max(h, w)
    sq = np.zeros((side, side), dtype=np.float32)
    oy, ox = (side - h) // 2, (side - w) // 2
    sq[oy:oy + h, ox:ox + w] = crop
    if side < GLYPH:  # small glyphs: blow them up by repeating pixels first
        k = int(np.ceil(GLYPH / side))
        sq = np.kron(sq, np.ones((k, k), dtype=np.float32))
    return shrink(sq, GLYPH, GLYPH) >= 0.35


def glyph_text(ink: np.ndarray, g: dict, line: dict) -> str:
    bits = glyph_bits(ink, g)
    lh = max(1, line["bottom"] - line["top"])
    packed = np.packbits(bits.ravel()).tobytes().hex()
    h, w = g["y1"] - g["y0"], g["x1"] - g["x0"]
    return (f"glyph:{packed}:{h / lh:.2f}:{(g['y0'] - line['top']) / lh:.2f}:{w / max(h, 1):.2f}:"
            f"{(line['bottom'] - g['y1']) / lh:.2f}")


def glyph_vector(text: str, dim: int) -> np.ndarray:
    """A glyph's text back as numbers: its 256 bits and its four relative measures (weighted up)."""
    try:
        _, hexbits, h, top, aspect, bottom = text.split(":")
        bits = np.unpackbits(np.frombuffer(bytes.fromhex(hexbits), dtype=np.uint8)).astype(np.float32)
        extra = np.array([float(h), float(top), min(float(aspect), 4.0) / 2, float(bottom)], dtype=np.float32)
    except Exception:
        return np.zeros(dim, dtype=np.float32)
    vec = np.concatenate([bits / 8.0, np.repeat(extra, 6)])
    return fold(vec, dim)


def render_text(text: str, font_path: str, size: int, rng) -> np.ndarray | None:
    """A line of text drawn in a font (with Pillow), dark on light with some jitter - practice for its reader."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return None
    try:
        font = ImageFont.truetype(font_path, size)
    except Exception:
        return None
    left, top, right, bottom = font.getbbox(text)
    W, H = right - left + 2 * size, bottom - top + size
    shade = int(rng.integers(0, 70))
    paper = int(rng.integers(200, 256))
    im = Image.new("L", (max(W, 8), max(H, 8)), paper)
    ImageDraw.Draw(im).text((size - left, size // 2 - top), text, fill=shade, font=font)
    a = np.asarray(im, dtype=np.uint8)
    return np.repeat(a[:, :, None], 3, axis=2)
