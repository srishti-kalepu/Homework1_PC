const char *dgemm_desc = "Simple blocked dgemm.";

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 181
#endif

#define min(a, b) (((a) < (b)) ? (a) : (b))

/*
 * This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N.
 */
static void do_block(int lda, int M, int N, int K, double *A, double *B,
                     double *C) {
  // Register blocking: 2x2 micro-kernel
  #include <immintrin.h>
  int i = 0, j = 0, k = 0;
  for (i = 0; i <= M - 4; i += 4) {
    for (j = 0; j <= N - 4; j += 4) {
      // Load C block into AVX registers
      __m256d c0 = _mm256_loadu_pd(&C[(i + 0) * lda + j]);
      __m256d c1 = _mm256_loadu_pd(&C[(i + 1) * lda + j]);
      __m256d c2 = _mm256_loadu_pd(&C[(i + 2) * lda + j]);
      __m256d c3 = _mm256_loadu_pd(&C[(i + 3) * lda + j]);

      for (k = 0; k < K; ++k) {
        // Load a column of A
        __m256d a = _mm256_set_pd(A[(i + 3) * lda + k], A[(i + 2) * lda + k], A[(i + 1) * lda + k], A[(i + 0) * lda + k]);
        // Load a row of B
        __m256d b0 = _mm256_broadcast_sd(&B[k * lda + (j + 0)]);
        __m256d b1 = _mm256_broadcast_sd(&B[k * lda + (j + 1)]);
        __m256d b2 = _mm256_broadcast_sd(&B[k * lda + (j + 2)]);
        __m256d b3 = _mm256_broadcast_sd(&B[k * lda + (j + 3)]);

        // Fused multiply-add for each column
        c0 = _mm256_fmadd_pd(a, b0, c0);
        c1 = _mm256_fmadd_pd(a, b1, c1);
        c2 = _mm256_fmadd_pd(a, b2, c2);
        c3 = _mm256_fmadd_pd(a, b3, c3);
      }

      // Store results back to C
      _mm256_storeu_pd(&C[(i + 0) * lda + j], c0);
      _mm256_storeu_pd(&C[(i + 1) * lda + j], c1);
      _mm256_storeu_pd(&C[(i + 2) * lda + j], c2);
      _mm256_storeu_pd(&C[(i + 3) * lda + j], c3);
    }
    // Handle remaining columns (N not divisible by 4)
    for (j = j; j < N; ++j) {
      for (int ii = 0; ii < 4; ++ii) {
        double cij = C[(i + ii) * lda + j];
        for (k = 0; k < K; ++k) {
          cij += A[(i + ii) * lda + k] * B[k * lda + j];
        }
        C[(i + ii) * lda + j] = cij;
      }
    }
  }
  // Handle remaining rows (M not divisible by 4)
  for (i = i; i < M; ++i) {
    for (j = 0; j < N; ++j) {
      double cij = C[i * lda + j];
      for (k = 0; k < K; ++k) {
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
void square_dgemm(int n, double *A, double *B, double *C) {
  // For each block-row of A
  for (int i = 0; i < n; i += BLOCK_SIZE) {
    // For each block-column of B
    for (int j = 0; j < n; j += BLOCK_SIZE) {
      // Accumulate block dgemms into block of C
      for (int k = 0; k < n; k += BLOCK_SIZE) {
        // Correct block dimensions if block "goes off edge of" the matrix
        int M = min(BLOCK_SIZE, n - i);
        int N = min(BLOCK_SIZE, n - j);
        int K = min(BLOCK_SIZE, n - k);
        // Perform individual block dgemm
        do_block(n, M, N, K, A + i * n + k, B + k * n + j, C + i * n + j);
      }
    }
  }
}