#include "bitemb_native.h"

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>

static int popcount_u8(uint8_t x) {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_popcount((unsigned int)x);
#else
    int c = 0;
    while (x) {
        x &= (uint8_t)(x - 1);
        c++;
    }
    return c;
#endif
}

static uint8_t get_packed_code(const uint8_t *row, int64_t dim_idx, int bits) {
    if (bits == 4) {
        uint8_t byte = row[dim_idx >> 1];
        return (dim_idx & 1) ? (uint8_t)((byte >> 4) & 0x0F) : (uint8_t)(byte & 0x0F);
    }
    if (bits == 2) {
        uint8_t byte = row[dim_idx >> 2];
        int shift = (int)((dim_idx & 3) << 1);
        return (uint8_t)((byte >> shift) & 0x03);
    }
    return 0;
}

int bitemb_cosine_distance_pairs_f32(
    const float *embs,
    const int64_t *pairs,
    int64_t n_pairs,
    int64_t dim,
    double *out
) {
    if (!embs || !pairs || !out || n_pairs < 0 || dim <= 0) return -1;
    for (int64_t p = 0; p < n_pairs; p++) {
        int64_t ia = pairs[p * 2];
        int64_t ib = pairs[p * 2 + 1];
        const float *a = embs + ia * dim;
        const float *b = embs + ib * dim;
        double dot = 0.0;
        for (int64_t d = 0; d < dim; d++) {
            dot += (double)a[d] * (double)b[d];
        }
        out[p] = 1.0 - dot;
    }
    return 0;
}

int bitemb_hamming_distance_pairs_u8(
    const uint8_t *packed,
    const int64_t *pairs,
    int64_t n_pairs,
    int64_t packed_dim,
    double *out
) {
    if (!packed || !pairs || !out || n_pairs < 0 || packed_dim <= 0) return -1;
    for (int64_t p = 0; p < n_pairs; p++) {
        int64_t ia = pairs[p * 2];
        int64_t ib = pairs[p * 2 + 1];
        const uint8_t *a = packed + ia * packed_dim;
        const uint8_t *b = packed + ib * packed_dim;
        int64_t dist = 0;
        for (int64_t d = 0; d < packed_dim; d++) {
            dist += popcount_u8((uint8_t)(a[d] ^ b[d]));
        }
        out[p] = (double)dist;
    }
    return 0;
}

int bitemb_tq_distance_pairs_packed(
    const uint8_t *packed_codes,
    const double *col_min,
    const double *col_max,
    const int64_t *pairs,
    int64_t n_pairs,
    int64_t dim,
    int64_t packed_dim,
    int bits,
    double *out
) {
    if (!packed_codes || !col_min || !col_max || !pairs || !out) return -1;
    if (n_pairs < 0 || dim <= 0 || packed_dim <= 0 || (bits != 2 && bits != 4)) return -2;
    double levels = (double)((1 << bits) - 1);
    for (int64_t p = 0; p < n_pairs; p++) {
        int64_t ia = pairs[p * 2];
        int64_t ib = pairs[p * 2 + 1];
        const uint8_t *a = packed_codes + ia * packed_dim;
        const uint8_t *b = packed_codes + ib * packed_dim;
        double sum = 0.0;
        for (int64_t d = 0; d < dim; d++) {
            double span = col_max[d] - col_min[d];
            double scale = span / levels;
            double diff_code = (double)get_packed_code(a, d, bits) - (double)get_packed_code(b, d, bits);
            double diff = diff_code * scale;
            sum += diff * diff;
        }
        out[p] = sum;
    }
    return 0;
}

static void insert_topk(double dist, int64_t idx, double *best_dist, int64_t *best_idx, int64_t k) {
    int64_t worst = 0;
    for (int64_t r = 1; r < k; r++) {
        if (best_dist[r] > best_dist[worst]) worst = r;
    }
    if (dist >= best_dist[worst]) return;
    best_dist[worst] = dist;
    best_idx[worst] = idx;
}

static void sort_topk(double *best_dist, int64_t *best_idx, int64_t k) {
    for (int64_t i = 0; i < k; i++) {
        for (int64_t j = i + 1; j < k; j++) {
            if (best_dist[j] < best_dist[i]) {
                double td = best_dist[i];
                best_dist[i] = best_dist[j];
                best_dist[j] = td;
                int64_t ti = best_idx[i];
                best_idx[i] = best_idx[j];
                best_idx[j] = ti;
            }
        }
    }
}

int bitemb_knn_cosine_f32(
    const float *embs,
    int64_t n,
    int64_t dim,
    int64_t k,
    int64_t *out_indices
) {
    if (!embs || !out_indices || n <= 0 || dim <= 0 || k <= 0 || k >= n) return -1;
    double *best_dist = (double *)malloc((size_t)k * sizeof(double));
    int64_t *best_idx = (int64_t *)malloc((size_t)k * sizeof(int64_t));
    if (!best_dist || !best_idx) {
        free(best_dist); free(best_idx); return -2;
    }
    for (int64_t q = 0; q < n; q++) {
        for (int64_t r = 0; r < k; r++) { best_dist[r] = DBL_MAX; best_idx[r] = -1; }
        const float *query = embs + q * dim;
        for (int64_t i = 0; i < n; i++) {
            if (i == q) continue;
            const float *cand = embs + i * dim;
            double dot = 0.0;
            for (int64_t d = 0; d < dim; d++) dot += (double)query[d] * (double)cand[d];
            insert_topk(1.0 - dot, i, best_dist, best_idx, k);
        }
        sort_topk(best_dist, best_idx, k);
        for (int64_t r = 0; r < k; r++) out_indices[q * k + r] = best_idx[r];
    }
    free(best_dist); free(best_idx);
    return 0;
}

int bitemb_knn_hamming_u8(
    const uint8_t *packed,
    int64_t n,
    int64_t packed_dim,
    int64_t k,
    int64_t *out_indices
) {
    if (!packed || !out_indices || n <= 0 || packed_dim <= 0 || k <= 0 || k >= n) return -1;
    double *best_dist = (double *)malloc((size_t)k * sizeof(double));
    int64_t *best_idx = (int64_t *)malloc((size_t)k * sizeof(int64_t));
    if (!best_dist || !best_idx) {
        free(best_dist); free(best_idx); return -2;
    }
    for (int64_t q = 0; q < n; q++) {
        for (int64_t r = 0; r < k; r++) { best_dist[r] = DBL_MAX; best_idx[r] = -1; }
        const uint8_t *query = packed + q * packed_dim;
        for (int64_t i = 0; i < n; i++) {
            if (i == q) continue;
            const uint8_t *cand = packed + i * packed_dim;
            int64_t dist = 0;
            for (int64_t d = 0; d < packed_dim; d++) dist += popcount_u8((uint8_t)(query[d] ^ cand[d]));
            insert_topk((double)dist, i, best_dist, best_idx, k);
        }
        sort_topk(best_dist, best_idx, k);
        for (int64_t r = 0; r < k; r++) out_indices[q * k + r] = best_idx[r];
    }
    free(best_dist); free(best_idx);
    return 0;
}

int bitemb_knn_tq_packed(
    const uint8_t *packed_codes,
    const double *col_min,
    const double *col_max,
    int64_t n,
    int64_t dim,
    int64_t packed_dim,
    int bits,
    int64_t k,
    int64_t *out_indices
) {
    if (!packed_codes || !col_min || !col_max || !out_indices) return -1;
    if (n <= 0 || dim <= 0 || packed_dim <= 0 || k <= 0 || k >= n || (bits != 2 && bits != 4)) return -2;
    double *best_dist = (double *)malloc((size_t)k * sizeof(double));
    int64_t *best_idx = (int64_t *)malloc((size_t)k * sizeof(int64_t));
    if (!best_dist || !best_idx) {
        free(best_dist); free(best_idx); return -3;
    }
    double levels = (double)((1 << bits) - 1);
    for (int64_t q = 0; q < n; q++) {
        for (int64_t r = 0; r < k; r++) { best_dist[r] = DBL_MAX; best_idx[r] = -1; }
        const uint8_t *query = packed_codes + q * packed_dim;
        for (int64_t i = 0; i < n; i++) {
            if (i == q) continue;
            const uint8_t *cand = packed_codes + i * packed_dim;
            double sum = 0.0;
            for (int64_t d = 0; d < dim; d++) {
                double span = col_max[d] - col_min[d];
                double scale = span / levels;
                double diff_code = (double)get_packed_code(query, d, bits) - (double)get_packed_code(cand, d, bits);
                double diff = diff_code * scale;
                sum += diff * diff;
            }
            insert_topk(sum, i, best_dist, best_idx, k);
        }
        sort_topk(best_dist, best_idx, k);
        for (int64_t r = 0; r < k; r++) out_indices[q * k + r] = best_idx[r];
    }
    free(best_dist); free(best_idx);
    return 0;
}
