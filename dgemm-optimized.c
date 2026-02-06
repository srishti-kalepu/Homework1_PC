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

static void do_block_avx512(int lda, int M, int N, int K,
                            const double *A,
                            const double *B,
                            double *C)
{
    for (int i = 0; i < M; ++i) {
        int j = 0;
        // Vectorized loop: 8 columns at a time
        for (; j + 7 < N; j += 8) {
            __m512d c = _mm512_loadu_pd(&C[i * lda + j]);
            for (int k = 0; k < K; ++k) {
                __m512d a = _mm512_set1_pd(A[i * lda + k]);
                __m512d b = _mm512_loadu_pd(&B[k * lda + j]);
                c = _mm512_fmadd_pd(a, b, c);
            }
            _mm512_storeu_pd(&C[i * lda + j], c);
        }
        // Remainder columns (scalar cleanup, only for columns not handled by vectorized loop)
        for (; j < N; ++j) {
            double cij = C[i * lda + j];
            for (int k = 0; k < K; ++k) {
                cij += A[i * lda + k] * B[k * lda + j];
            }
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
    for (int i = 0; i < n; i += BLOCK_SIZE) {
        for (int j = 0; j < n; j += BLOCK_SIZE) {
            for (int k = 0; k < n; k += BLOCK_SIZE) {

                int M = min(BLOCK_SIZE, n - i);
                int N = min(BLOCK_SIZE, n - j);
                int K = min(BLOCK_SIZE, n - k);

                do_block_avx512(
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