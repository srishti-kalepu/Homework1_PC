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
  for (i = 0; i <= M - 4; i += 4) {
    for (j = 0; j <= N - 4; j += 4) {
      // Register block for C
      double c00 = C[(i + 0) * lda + (j + 0)];
      double c01 = C[(i + 0) * lda + (j + 1)];
      double c02 = C[(i + 0) * lda + (j + 2)];
      double c03 = C[(i + 0) * lda + (j + 3)];
      double c10 = C[(i + 1) * lda + (j + 0)];
      double c11 = C[(i + 1) * lda + (j + 1)];
      double c12 = C[(i + 1) * lda + (j + 2)];
      double c13 = C[(i + 1) * lda + (j + 3)];
      double c20 = C[(i + 2) * lda + (j + 0)];
      double c21 = C[(i + 2) * lda + (j + 1)];
      double c22 = C[(i + 2) * lda + (j + 2)];
      double c23 = C[(i + 2) * lda + (j + 3)];
      double c30 = C[(i + 3) * lda + (j + 0)];
      double c31 = C[(i + 3) * lda + (j + 1)];
      double c32 = C[(i + 3) * lda + (j + 2)];
      double c33 = C[(i + 3) * lda + (j + 3)];

      for (k = 0; k < K; ++k) {
        double a0 = A[(i + 0) * lda + k];
        double a1 = A[(i + 1) * lda + k];
        double a2 = A[(i + 2) * lda + k];
        double a3 = A[(i + 3) * lda + k];
        double b0 = B[k * lda + (j + 0)];
        double b1 = B[k * lda + (j + 1)];
        double b2 = B[k * lda + (j + 2)];
        double b3 = B[k * lda + (j + 3)];

        c00 += a0 * b0;
        c01 += a0 * b1;
        c02 += a0 * b2;
        c03 += a0 * b3;

        c10 += a1 * b0;
        c11 += a1 * b1;
        c12 += a1 * b2;
        c13 += a1 * b3;

        c20 += a2 * b0;
        c21 += a2 * b1;
        c22 += a2 * b2;
        c23 += a2 * b3;

        c30 += a3 * b0;
        c31 += a3 * b1;
        c32 += a3 * b2;
        c33 += a3 * b3;
      }

      C[(i + 0) * lda + (j + 0)] = c00;
      C[(i + 0) * lda + (j + 1)] = c01;
      C[(i + 0) * lda + (j + 2)] = c02;
      C[(i + 0) * lda + (j + 3)] = c03;
      C[(i + 1) * lda + (j + 0)] = c10;
      C[(i + 1) * lda + (j + 1)] = c11;
      C[(i + 1) * lda + (j + 2)] = c12;
      C[(i + 1) * lda + (j + 3)] = c13;
      C[(i + 2) * lda + (j + 0)] = c20;
      C[(i + 2) * lda + (j + 1)] = c21;
      C[(i + 2) * lda + (j + 2)] = c22;
      C[(i + 2) * lda + (j + 3)] = c23;
      C[(i + 3) * lda + (j + 0)] = c30;
      C[(i + 3) * lda + (j + 1)] = c31;
      C[(i + 3) * lda + (j + 2)] = c32;
      C[(i + 3) * lda + (j + 3)] = c33;
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