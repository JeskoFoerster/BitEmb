"""Python wrapper for the optional native packed backend."""

from __future__ import annotations

import importlib
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from bitemb.efficiency import PackedTurboQuantIndex


class NativeBackendUnavailableError(RuntimeError):
    """Raised when the optional native CFFI module is not built."""


def _load_native_module():
    try:
        return importlib.import_module("bitemb.native._bitemb_native")
    except ImportError as exc:
        raise NativeBackendUnavailableError(
            "Native backend is not built. Run `python -m bitemb.native.build_native` "
            "after installing a C compiler."
        ) from exc


def native_backend_available() -> bool:
    try:
        _load_native_module()
    except NativeBackendUnavailableError:
        return False
    return True


@dataclass
class NativePackedBackend:
    """Thin CFFI wrapper around the native packed kernels."""

    def __post_init__(self) -> None:
        mod = _load_native_module()
        self.ffi = mod.ffi
        self.lib = mod.lib

    def _ptr(self, arr: NDArray, ctype: str):
        contiguous = np.ascontiguousarray(arr)
        return contiguous, self.ffi.cast(ctype, contiguous.ctypes.data)

    def cosine_distance_pairs(
        self, embeddings: NDArray[np.float32], pairs: NDArray[np.int64]
    ) -> NDArray[np.float64]:
        embs = np.ascontiguousarray(embeddings, dtype=np.float32)
        p = np.ascontiguousarray(pairs, dtype=np.int64)
        out = np.empty(p.shape[0], dtype=np.float64)
        rc = self.lib.bitemb_cosine_distance_pairs_f32(
            self.ffi.cast("const float *", embs.ctypes.data),
            self.ffi.cast("const int64_t *", p.ctypes.data),
            p.shape[0],
            embs.shape[1],
            self.ffi.cast("double *", out.ctypes.data),
        )
        if rc != 0:
            raise RuntimeError(f"native cosine_distance_pairs failed with code {rc}")
        return out

    def hamming_distance_pairs(
        self, packed: NDArray[np.uint8], pairs: NDArray[np.int64]
    ) -> NDArray[np.float64]:
        codes = np.ascontiguousarray(packed, dtype=np.uint8)
        p = np.ascontiguousarray(pairs, dtype=np.int64)
        out = np.empty(p.shape[0], dtype=np.float64)
        rc = self.lib.bitemb_hamming_distance_pairs_u8(
            self.ffi.cast("const uint8_t *", codes.ctypes.data),
            self.ffi.cast("const int64_t *", p.ctypes.data),
            p.shape[0],
            codes.shape[1],
            self.ffi.cast("double *", out.ctypes.data),
        )
        if rc != 0:
            raise RuntimeError(f"native hamming_distance_pairs failed with code {rc}")
        return out

    def turboquant_distance_pairs(
        self, index: PackedTurboQuantIndex, pairs: NDArray[np.int64]
    ) -> NDArray[np.float64]:
        codes = np.ascontiguousarray(index.packed_codes, dtype=np.uint8)
        col_min = np.ascontiguousarray(index.col_min, dtype=np.float64)
        col_max = np.ascontiguousarray(index.col_max, dtype=np.float64)
        p = np.ascontiguousarray(pairs, dtype=np.int64)
        out = np.empty(p.shape[0], dtype=np.float64)
        rc = self.lib.bitemb_tq_distance_pairs_packed(
            self.ffi.cast("const uint8_t *", codes.ctypes.data),
            self.ffi.cast("const double *", col_min.ctypes.data),
            self.ffi.cast("const double *", col_max.ctypes.data),
            self.ffi.cast("const int64_t *", p.ctypes.data),
            p.shape[0],
            index.dim,
            codes.shape[1],
            index.bits,
            self.ffi.cast("double *", out.ctypes.data),
        )
        if rc != 0:
            raise RuntimeError(f"native turboquant_distance_pairs failed with code {rc}")
        return out

    def knn_cosine(self, embeddings: NDArray[np.float32], k: int) -> NDArray[np.int64]:
        embs = np.ascontiguousarray(embeddings, dtype=np.float32)
        out = np.empty((embs.shape[0], k), dtype=np.int64)
        rc = self.lib.bitemb_knn_cosine_f32(
            self.ffi.cast("const float *", embs.ctypes.data),
            embs.shape[0],
            embs.shape[1],
            k,
            self.ffi.cast("int64_t *", out.ctypes.data),
        )
        if rc != 0:
            raise RuntimeError(f"native knn_cosine failed with code {rc}")
        return out

    def knn_hamming(self, packed: NDArray[np.uint8], k: int) -> NDArray[np.int64]:
        codes = np.ascontiguousarray(packed, dtype=np.uint8)
        out = np.empty((codes.shape[0], k), dtype=np.int64)
        rc = self.lib.bitemb_knn_hamming_u8(
            self.ffi.cast("const uint8_t *", codes.ctypes.data),
            codes.shape[0],
            codes.shape[1],
            k,
            self.ffi.cast("int64_t *", out.ctypes.data),
        )
        if rc != 0:
            raise RuntimeError(f"native knn_hamming failed with code {rc}")
        return out
    def knn_turboquant(self, index: PackedTurboQuantIndex, k: int) -> NDArray[np.int64]:
        codes = np.ascontiguousarray(index.packed_codes, dtype=np.uint8)
        col_min = np.ascontiguousarray(index.col_min, dtype=np.float64)
        col_max = np.ascontiguousarray(index.col_max, dtype=np.float64)
        out = np.empty((index.n, k), dtype=np.int64)
        rc = self.lib.bitemb_knn_tq_packed(
            self.ffi.cast("const uint8_t *", codes.ctypes.data),
            self.ffi.cast("const double *", col_min.ctypes.data),
            self.ffi.cast("const double *", col_max.ctypes.data),
            index.n,
            index.dim,
            codes.shape[1],
            index.bits,
            k,
            self.ffi.cast("int64_t *", out.ctypes.data),
        )
        if rc != 0:
            raise RuntimeError(f"native knn_turboquant failed with code {rc}")
        return out



