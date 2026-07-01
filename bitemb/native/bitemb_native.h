#ifndef BITEMB_NATIVE_H
#define BITEMB_NATIVE_H

#include <stdint.h>

int bitemb_cosine_distance_pairs_f32(
    const float *embs,
    const int64_t *pairs,
    int64_t n_pairs,
    int64_t dim,
    double *out
);

int bitemb_hamming_distance_pairs_u8(
    const uint8_t *packed,
    const int64_t *pairs,
    int64_t n_pairs,
    int64_t packed_dim,
    double *out
);

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
);

int bitemb_knn_cosine_f32(
    const float *embs,
    int64_t n,
    int64_t dim,
    int64_t k,
    int64_t *out_indices
);

int bitemb_knn_hamming_u8(
    const uint8_t *packed,
    int64_t n,
    int64_t packed_dim,
    int64_t k,
    int64_t *out_indices
);

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
);

#endif

