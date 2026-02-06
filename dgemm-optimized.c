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

static void do_block_avx512_4x8(int lda, int M, int N, int K,
                                const double *A,
                                const double *B,
                                double *C)
{
    int i = 0;
    for (; i + 3 < M; i += 4) {
        int j = 0;
        for (; j + 7 < N; j += 8) {
            
            // Pointer setup for the C block (4 rows)
            double *c_ptr0 = C + (i + 0) * lda + j;
            double *c_ptr1 = C + (i + 1) * lda + j;
            double *c_ptr2 = C + (i + 2) * lda + j;
            double *c_ptr3 = C + (i + 3) * lda + j;

            __m512d c0 = _mm512_loadu_pd(c_ptr0);
            __m512d c1 = _mm512_loadu_pd(c_ptr1);
            __m512d c2 = _mm512_loadu_pd(c_ptr2);
            __m512d c3 = _mm512_loadu_pd(c_ptr3);

            for (int k = 0; k < K; ++k) {

                // Prefetch the next row of B or the next part of current row
                // 64 bytes is the size of one AVX-512 register (8 doubles)
                _mm_prefetch((const char*)(B + (k + 1) * lda + j), _MM_HINT_T0);
                
                // Optional: Prefetch the next elements of A
                _mm_prefetch((const char*)(A + (i + 0) * lda + k + 8), _MM_HINT_T0);

                
                // Load 8 doubles from B (row k)
                __m512d b = _mm512_loadu_pd(B + k * lda + j);

                // Broadcast A[i][k] values
                __m512d a0 = _mm512_set1_pd(*(A + (i + 0) * lda + k));
                __m512d a1 = _mm512_set1_pd(*(A + (i + 1) * lda + k));
                __m512d a2 = _mm512_set1_pd(*(A + (i + 2) * lda + k));
                __m512d a3 = _mm512_set1_pd(*(A + (i + 3) * lda + k));

                c0 = _mm512_fmadd_pd(a0, b, c0);
                c1 = _mm512_fmadd_pd(a1, b, c1);
                c2 = _mm512_fmadd_pd(a2, b, c2);
                c3 = _mm512_fmadd_pd(a3, b, c3);
            }

            _mm512_storeu_pd(c_ptr0, c0);
            _mm512_storeu_pd(c_ptr1, c1);
            _mm512_storeu_pd(c_ptr2, c2);
            _mm512_storeu_pd(c_ptr3, c3);
        }

        // Scalar cleanup for remaining columns
        for (; j < N; ++j) {
            for (int ii = 0; ii < 4; ++ii) {
                double *cij_ptr = C + (i + ii) * lda + j;
                double cij = *cij_ptr;
                for (int k = 0; k < K; ++k)
                    cij += *(A + (i + ii) * lda + k) * *(B + k * lda + j);
                *cij_ptr = cij;
            }
        }
    }

    // Scalar cleanup for remaining rows
    for (; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            double *cij_ptr = C + i * lda + j;
            double cij = *cij_ptr;
            for (int k = 0; k < K; ++k)
                cij += *(A + i * lda + k) * *(B + k * lda + j);
            *cij_ptr = cij;
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

                do_block_avx512_4x8(
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