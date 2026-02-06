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


static void do_block_avx512_8x16(
    int lda, int M, int N, int K,
    const double *A, const double *B, double *C)
{
    // i-step: 8, j-step: 16
    for (int i = 0; i < M; i += 8) {
        for (int j = 0; j < N; j += 16) {
            
            // Accumulators: 8 rows, 2 ZMMs per row (total 16 registers)
            __m512d c00 = _mm512_loadu_pd(&C[(i+0)*lda + j]);
            __m512d c01 = _mm512_loadu_pd(&C[(i+0)*lda + j + 8]);
            __m512d c10 = _mm512_loadu_pd(&C[(i+1)*lda + j]);
            __m512d c11 = _mm512_loadu_pd(&C[(i+1)*lda + j + 8]);
            __m512d c20 = _mm512_loadu_pd(&C[(i+2)*lda + j]);
            __m512d c21 = _mm512_loadu_pd(&C[(i+2)*lda + j + 8]);
            __m512d c30 = _mm512_loadu_pd(&C[(i+3)*lda + j]);
            __m512d c31 = _mm512_loadu_pd(&C[(i+3)*lda + j + 8]);
            __m512d c40 = _mm512_loadu_pd(&C[(i+4)*lda + j]);
            __m512d c41 = _mm512_loadu_pd(&C[(i+4)*lda + j + 8]);
            __m512d c50 = _mm512_loadu_pd(&C[(i+5)*lda + j]);
            __m512d c51 = _mm512_loadu_pd(&C[(i+5)*lda + j + 8]);
            __m512d c60 = _mm512_loadu_pd(&C[(i+6)*lda + j]);
            __m512d c61 = _mm512_loadu_pd(&C[(i+6)*lda + j + 8]);
            __m512d c70 = _mm512_loadu_pd(&C[(i+7)*lda + j]);
            __m512d c71 = _mm512_loadu_pd(&C[(i+7)*lda + j + 8]);

            for (int k = 0; k < K; ++k) {
                // Load 16 elements from row k of B (2 loads)
                __m512d b0 = _mm512_loadu_pd(&B[k*lda + j]);
                __m512d b1 = _mm512_loadu_pd(&B[k*lda + j + 8]);

                // Broadcast A[i...i+7, k] and FMA
                // Compilers are usually smart enough to map these to registers
                c00 = _mm512_fmadd_pd(_mm512_set1_pd(A[(i+0)*lda + k]), b0, c00);
                c01 = _mm512_fmadd_pd(_mm512_set1_pd(A[(i+0)*lda + k]), b1, c01);
                
                c10 = _mm512_fmadd_pd(_mm512_set1_pd(A[(i+1)*lda + k]), b0, c10);
                c11 = _mm512_fmadd_pd(_mm512_set1_pd(A[(i+1)*lda + k]), b1, c11);
                
                c20 = _mm512_fmadd_pd(_mm512_set1_pd(A[(i+2)*lda + k]), b0, c20);
                c21 = _mm512_fmadd_pd(_mm512_set1_pd(A[(i+2)*lda + k]), b1, c21);
                
                c30 = _mm512_fmadd_pd(_mm512_set1_pd(A[(i+3)*lda + k]), b0, c30);
                c31 = _mm512_fmadd_pd(_mm512_set1_pd(A[(i+3)*lda + k]), b1, c31);
                
                c40 = _mm512_fmadd_pd(_mm512_set1_pd(A[(i+4)*lda + k]), b0, c40);
                c41 = _mm512_fmadd_pd(_mm512_set1_pd(A[(i+4)*lda + k]), b1, c41);
                
                c50 = _mm512_fmadd_pd(_mm512_set1_pd(A[(i+5)*lda + k]), b0, c50);
                c51 = _mm512_fmadd_pd(_mm512_set1_pd(A[(i+5)*lda + k]), b1, c51);
                
                c60 = _mm512_fmadd_pd(_mm512_set1_pd(A[(i+6)*lda + k]), b0, c60);
                c61 = _mm512_fmadd_pd(_mm512_set1_pd(A[(i+6)*lda + k]), b1, c61);
                
                c70 = _mm512_fmadd_pd(_mm512_set1_pd(A[(i+7)*lda + k]), b0, c70);
                c71 = _mm512_fmadd_pd(_mm512_set1_pd(A[(i+7)*lda + k]), b1, c71);
            }

            // Store results back
            _mm512_storeu_pd(&C[(i+0)*lda + j], c00);
            _mm512_storeu_pd(&C[(i+0)*lda + j + 8], c01);
            _mm512_storeu_pd(&C[(i+1)*lda + j], c10);
            _mm512_storeu_pd(&C[(i+1)*lda + j + 8], c11);
            // ... (repeat for all 8 rows)
            _mm512_storeu_pd(&C[(i+7)*lda + j], c70);
            _mm512_storeu_pd(&C[(i+7)*lda + j + 8], c71);
        }
    }
}


/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are n-by-n matrices stored in row-major format.
 * On exit, A and B maintain their input values. */

void square_dgemm(int n, double *A, double *B, double *C) {
    // 1. If N is very small (e.g., < 16), use a simple scalar loop.
    // The overhead of setting up SIMD masks is more than the math itself.
    if (n < 16) {
        // Simple ijk loop here...
        return;
    }

    __mmask8 final_mask = get_mask(n % 8);
    
    #pragma omp parallel for collapse(2) if(n > 128)
    for (int i = 0; i < n; i += BLOCK_SIZE) {
        for (int j = 0; j < n; j += BLOCK_SIZE) {
            for (int k = 0; k < n; k += BLOCK_SIZE) {
                
                int M_block = min(BLOCK_SIZE, n - i);
                int N_block = min(BLOCK_SIZE, n - j);
                int K_block = min(BLOCK_SIZE, n - k);

                // Call the masked microkernel directly on the original pointers
                // Use 'n' as the lda (stride)
                do_block_avx512_masked(
                    n, M_block, N_block, K_block,
                    A + i*n + k, 
                    B + k*n + j, 
                    C + i*n + j,
                    final_mask, (N_block % 8 != 0)
                );
            }
        }
    }
}