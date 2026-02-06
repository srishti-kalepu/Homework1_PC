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
  int i = 0, j = 0, k = 0;
  for (i = 0; i <= M - 2; i += 2) {
    for (j = 0; j <= N - 2; j += 2) {
      // Register block for C
      double c00 = C[(i + 0) * lda + (j + 0)];
      double c01 = C[(i + 0) * lda + (j + 1)];
      double c10 = C[(i + 1) * lda + (j + 0)];
      double c11 = C[(i + 1) * lda + (j + 1)];

      for (k = 0; k < K; ++k) {
        double a0 = A[(i + 0) * lda + k];
        double a1 = A[(i + 1) * lda + k];
        double b0 = B[k * lda + (j + 0)];
        double b1 = B[k * lda + (j + 1)];

        c00 += a0 * b0;
        c01 += a0 * b1;
        c10 += a1 * b0;
        c11 += a1 * b1;
      }

      C[(i + 0) * lda + (j + 0)] = c00;
      C[(i + 0) * lda + (j + 1)] = c01;
      C[(i + 1) * lda + (j + 0)] = c10;
      C[(i + 1) * lda + (j + 1)] = c11;
    }
    // Handle remaining columns (N not divisible by 2)
    for (j = j; j < N; ++j) {
      for (int ii = 0; ii < 2; ++ii) {
        double cij = C[(i + ii) * lda + j];
        for (k = 0; k < K; ++k) {
          cij += A[(i + ii) * lda + k] * B[k * lda + j];
        }
        C[(i + ii) * lda + j] = cij;
      }
    }
  }
  // Handle remaining rows (M not divisible by 2)
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