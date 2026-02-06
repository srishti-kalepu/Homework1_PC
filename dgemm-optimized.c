const char *dgemm_desc = "Simple blocked dgemm.";

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 64
#endif

#define min(a, b) (((a) < (b)) ? (a) : (b))

/*
 * This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N.
 */
#include <immintrin.h>
#include <omp.h>

static void do_block_avx512_6x16(int lda, int M, int N, int K,
                                 const double *A,
                                 const double *B,
                                 double *C)
{
    int i = 0;
    // Process 6 rows at a time
    for (; i + 5 < M; i += 6) {
        int j = 0;
        // Process 16 columns at a time (2 x 8-double vectors)
        for (; j < N; j += 16) {
            
            int rem_j = N - j;
            // Masks for the two 8-double segments
            __mmask8 mask0 = (__mmask8)((1U << (rem_j < 8 ? (rem_j < 0 ? 0 : rem_j) : 8)) - 1);
            __mmask8 mask1 = (__mmask8)((1U << (rem_j < 16 ? (rem_j < 8 ? 0 : rem_j - 8) : 8)) - 1);

            // C pointers for 6 rows
            double *cp0 = C + (i + 0) * lda + j;
            double *cp1 = C + (i + 1) * lda + j;
            double *cp2 = C + (i + 2) * lda + j;
            double *cp3 = C + (i + 3) * lda + j;
            double *cp4 = C + (i + 4) * lda + j;
            double *cp5 = C + (i + 5) * lda + j;

            // Load C (12 registers: 6 rows x 2 vectors)
            __m512d c00 = _mm512_maskz_loadu_pd(mask0, cp0);     __m512d c01 = _mm512_maskz_loadu_pd(mask1, cp0 + 8);
            __m512d c10 = _mm512_maskz_loadu_pd(mask0, cp1);     __m512d c11 = _mm512_maskz_loadu_pd(mask1, cp1 + 8);
            __m512d c20 = _mm512_maskz_loadu_pd(mask0, cp2);     __m512d c21 = _mm512_maskz_loadu_pd(mask1, cp2 + 8);
            __m512d c30 = _mm512_maskz_loadu_pd(mask0, cp3);     __m512d c31 = _mm512_maskz_loadu_pd(mask1, cp3 + 8);
            __m512d c40 = _mm512_maskz_loadu_pd(mask0, cp4);     __m512d c41 = _mm512_maskz_loadu_pd(mask1, cp4 + 8);
            __m512d c50 = _mm512_maskz_loadu_pd(mask0, cp5);     __m512d c51 = _mm512_maskz_loadu_pd(mask1, cp5 + 8);

            for (int k = 0; k < K; ++k) {
                // Load B row (2 vectors)
                const double *b_row = B + k * lda + j;
                __m512d b0 = _mm512_maskz_loadu_pd(mask0, b_row);
                __m512d b1 = _mm512_maskz_loadu_pd(mask1, b_row + 8);

                // Broadcast A elements for 6 rows
                __m512d a0 = _mm512_set1_pd(*(A + (i + 0) * lda + k));
                __m512d a1 = _mm512_set1_pd(*(A + (i + 1) * lda + k));
                __m512d a2 = _mm512_set1_pd(*(A + (i + 2) * lda + k));
                __m512d a3 = _mm512_set1_pd(*(A + (i + 3) * lda + k));
                __m512d a4 = _mm512_set1_pd(*(A + (i + 4) * lda + k));
                __m512d a5 = _mm512_set1_pd(*(A + (i + 5) * lda + k));

                // FMA: C = A*B + C
                c00 = _mm512_fmadd_pd(a0, b0, c00); c01 = _mm512_fmadd_pd(a0, b1, c01);
                c10 = _mm512_fmadd_pd(a1, b0, c10); c11 = _mm512_fmadd_pd(a1, b1, c11);
                c20 = _mm512_fmadd_pd(a2, b0, c20); c21 = _mm512_fmadd_pd(a2, b1, c21);
                c30 = _mm512_fmadd_pd(a3, b0, c30); c31 = _mm512_fmadd_pd(a3, b1, c31);
                c40 = _mm512_fmadd_pd(a4, b0, c40); c41 = _mm512_fmadd_pd(a4, b1, c41);
                c50 = _mm512_fmadd_pd(a5, b0, c50); c51 = _mm512_fmadd_pd(a5, b1, c51);
            }

            // Store results
            _mm512_mask_storeu_pd(cp0, mask0, c00); _mm512_mask_storeu_pd(cp0 + 8, mask1, c01);
            _mm512_mask_storeu_pd(cp1, mask0, c10); _mm512_mask_storeu_pd(cp1 + 8, mask1, c11);
            _mm512_mask_storeu_pd(cp2, mask0, c20); _mm512_mask_storeu_pd(cp2 + 8, mask1, c21);
            _mm512_mask_storeu_pd(cp3, mask0, c30); _mm512_mask_storeu_pd(cp3 + 8, mask1, c31);
            _mm512_mask_storeu_pd(cp4, mask0, c40); _mm512_mask_storeu_pd(cp4 + 8, mask1, c41);
            _mm512_mask_storeu_pd(cp5, mask0, c50); _mm512_mask_storeu_pd(cp5 + 8, mask1, c51);
        }
    }

    // Row cleanup (scalar)
    for (; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            double cij = C[i * lda + j];
            for (int k = 0; k < K; ++k)
                cij += A[i * lda + k] * B[k * lda + j];
            C[i * lda + j] = cij;
        }
    }
}
/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are n-by-n matrices stored in row-major format.
 * On exit, A and B maintain their input values. */

void square_dgemm(int n, double *A, double *B, double *C)
{
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < n; i += BLOCK_SIZE) {
        for (int j = 0; j < n; j += BLOCK_SIZE) {
            for (int k = 0; k < n; k += BLOCK_SIZE) {

                int M = min(BLOCK_SIZE, n - i);
                int N = min(BLOCK_SIZE, n - j);
                int K = min(BLOCK_SIZE, n - k);

                do_block_avx512_6x16(
                    n,
                    M, N, K,
                    A + i * n + k,
                    B + k * n + j,
                    C + i * n + j
                );
            }
        }
    }
}