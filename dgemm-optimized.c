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
#include <stdlib.h>
#include <string.h>

#define roundup8(x) (((x) + 7) & ~7)

/* ============================================================
   4x8 AVX-512 microkernel (NO MASKS, padding required)
   ============================================================ */
static void do_block_avx512_4x8(
    int lda, int M, int N, int K,
    const double *A,
    const double *B,
    double *C)
{
    for (int i = 0; i < M; i += 4) {
        for (int j = 0; j < N; j += 8) {

            __m512d c0 = _mm512_loadu_pd(&C[(i+0)*lda + j]);
            __m512d c1 = _mm512_loadu_pd(&C[(i+1)*lda + j]);
            __m512d c2 = _mm512_loadu_pd(&C[(i+2)*lda + j]);
            __m512d c3 = _mm512_loadu_pd(&C[(i+3)*lda + j]);

            for (int k = 0; k < K; ++k) {
                __m512d b  = _mm512_loadu_pd(&B[k*lda + j]);

                __m512d a0 = _mm512_set1_pd(A[(i+0)*lda + k]);
                __m512d a1 = _mm512_set1_pd(A[(i+1)*lda + k]);
                __m512d a2 = _mm512_set1_pd(A[(i+2)*lda + k]);
                __m512d a3 = _mm512_set1_pd(A[(i+3)*lda + k]);

                c0 = _mm512_fmadd_pd(a0, b, c0);
                c1 = _mm512_fmadd_pd(a1, b, c1);
                c2 = _mm512_fmadd_pd(a2, b, c2);
                c3 = _mm512_fmadd_pd(a3, b, c3);
            }

            _mm512_storeu_pd(&C[(i+0)*lda + j], c0);
            _mm512_storeu_pd(&C[(i+1)*lda + j], c1);
            _mm512_storeu_pd(&C[(i+2)*lda + j], c2);
            _mm512_storeu_pd(&C[(i+3)*lda + j], c3);
        }
    }
}


/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are n-by-n matrices stored in row-major format.
 * On exit, A and B maintain their input values. */

void square_dgemm(int n, double *A, double *B, double *C)
{
    /* Small-matrix fast path (no blocking, no OpenMP) */
    if (n <= 96) {
        int n_pad = roundup8(n);

        double *Ap, *Bp, *Cp;
        posix_memalign((void**)&Ap, 64, n_pad*n_pad*sizeof(double));
        posix_memalign((void**)&Bp, 64, n_pad*n_pad*sizeof(double));
        posix_memalign((void**)&Cp, 64, n_pad*n_pad*sizeof(double));

        memset(Ap, 0, n_pad*n_pad*sizeof(double));
        memset(Bp, 0, n_pad*n_pad*sizeof(double));
        memset(Cp, 0, n_pad*n_pad*sizeof(double));

        for (int i = 0; i < n; i++) {
            memcpy(&Ap[i*n_pad], &A[i*n], n*sizeof(double));
            memcpy(&Bp[i*n_pad], &B[i*n], n*sizeof(double));
            memcpy(&Cp[i*n_pad], &C[i*n], n*sizeof(double));
        }

        do_block_avx512_4x8(n_pad, n_pad, n_pad, n_pad, Ap, Bp, Cp);

        for (int i = 0; i < n; i++)
            memcpy(&C[i*n], &Cp[i*n_pad], n*sizeof(double));

        free(Ap); free(Bp); free(Cp);
        return;
    }

    /* General case: padded + blocked + OpenMP */
    int n_pad = roundup8(n);

    double *Ap, *Bp, *Cp;
    posix_memalign((void**)&Ap, 64, n_pad*n_pad*sizeof(double));
    posix_memalign((void**)&Bp, 64, n_pad*n_pad*sizeof(double));
    posix_memalign((void**)&Cp, 64, n_pad*n_pad*sizeof(double));

    memset(Ap, 0, n_pad*n_pad*sizeof(double));
    memset(Bp, 0, n_pad*n_pad*sizeof(double));
    memset(Cp, 0, n_pad*n_pad*sizeof(double));

    for (int i = 0; i < n; i++) {
        memcpy(&Ap[i*n_pad], &A[i*n], n*sizeof(double));
        memcpy(&Bp[i*n_pad], &B[i*n], n*sizeof(double));
        memcpy(&Cp[i*n_pad], &C[i*n], n*sizeof(double));
    }

#pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < n_pad; i += BLOCK_SIZE) {
        for (int j = 0; j < n_pad; j += BLOCK_SIZE) {
            for (int k = 0; k < n_pad; k += BLOCK_SIZE) {

                do_block_avx512_4x8(
                    n_pad,
                    min(BLOCK_SIZE, n_pad - i),
                    min(BLOCK_SIZE, n_pad - j),
                    min(BLOCK_SIZE, n_pad - k),
                    Ap + i*n_pad + k,
                    Bp + k*n_pad + j,
                    Cp + i*n_pad + j);
            }
        }
    }

    for (int i = 0; i < n; i++)
        memcpy(&C[i*n], &Cp[i*n_pad], n*sizeof(double));

    free(Ap); free(Bp); free(Cp);
}
