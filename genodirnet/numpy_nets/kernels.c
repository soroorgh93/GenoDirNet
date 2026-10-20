/* Small C kernels for the processor only training framework (channels last layout, float32).
   Built by cpunet/__init__.py with gcc or clang: -O3 -fopenmp -ffast-math.                                          */
#include <stddef.h>
#include <string.h>
#include <math.h>
#ifdef _OPENMP
#include <omp.h>
#endif

/* depthwise convolution, x[B,H,W,C], w[kh,kw,C], out[B,Ho,Wo,C], stride s, padding p (same padding for s=1) */
void dw_fwd(const float *x, const float *w, float *out, int B, int H, int W, int C, int kh, int kw, int s, int ph, int pw, int Ho, int Wo) {
    #pragma omp parallel for collapse(2) schedule(static)
    for (int b = 0; b < B; b++)
    for (int oh = 0; oh < Ho; oh++) {
        for (int ow = 0; ow < Wo; ow++) {
            float *o = out + (((size_t)b * Ho + oh) * Wo + ow) * C;
            memset(o, 0, sizeof(float) * C);
            for (int i = 0; i < kh; i++) {
                int ih = oh * s - ph + i; if (ih < 0 || ih >= H) continue;
                for (int j = 0; j < kw; j++) {
                    int iw = ow * s - pw + j; if (iw < 0 || iw >= W) continue;
                    const float *xi = x + (((size_t)b * H + ih) * W + iw) * C;
                    const float *wi = w + ((size_t)i * kw + j) * C;
                    for (int c = 0; c < C; c++) o[c] += xi[c] * wi[c];
                }
            }
        }
    }
}

/* gradient with respect to the input: dx[B,H,W,C] from dy[B,Ho,Wo,C] */
void dw_bwd_x(const float *dy, const float *w, float *dx, int B, int H, int W, int C, int kh, int kw, int s, int ph, int pw, int Ho, int Wo) {
    #pragma omp parallel for collapse(2) schedule(static)
    for (int b = 0; b < B; b++)
    for (int ih = 0; ih < H; ih++) {
        for (int iw = 0; iw < W; iw++) {
            float *d = dx + (((size_t)b * H + ih) * W + iw) * C;
            memset(d, 0, sizeof(float) * C);
            for (int i = 0; i < kh; i++) {
                int t = ih + ph - i; if (t < 0 || t % s) continue; int oh = t / s; if (oh >= Ho) continue;
                for (int j = 0; j < kw; j++) {
                    int u = iw + pw - j; if (u < 0 || u % s) continue; int ow = u / s; if (ow >= Wo) continue;
                    const float *g = dy + (((size_t)b * Ho + oh) * Wo + ow) * C;
                    const float *wi = w + ((size_t)i * kw + j) * C;
                    for (int c = 0; c < C; c++) d[c] += g[c] * wi[c];
                }
            }
        }
    }
}

/* gradient with respect to the kernel: dw[kh,kw,C] */
void dw_bwd_w(const float *x, const float *dy, float *dw, int B, int H, int W, int C, int kh, int kw, int s, int ph, int pw, int Ho, int Wo) {
    int nth = 1;
#ifdef _OPENMP
    nth = omp_get_max_threads();
#endif
    size_t kk = (size_t)kh * kw * C;
    float *acc = (float *)calloc((size_t)nth * kk, sizeof(float));
    #pragma omp parallel
    {
        int t = 0;
#ifdef _OPENMP
        t = omp_get_thread_num();
#endif
        float *a = acc + (size_t)t * kk;
        #pragma omp for collapse(2) schedule(static)
        for (int b = 0; b < B; b++)
        for (int oh = 0; oh < Ho; oh++) {
            for (int ow = 0; ow < Wo; ow++) {
                const float *g = dy + (((size_t)b * Ho + oh) * Wo + ow) * C;
                for (int i = 0; i < kh; i++) {
                    int ih = oh * s - ph + i; if (ih < 0 || ih >= H) continue;
                    for (int j = 0; j < kw; j++) {
                        int iw = ow * s - pw + j; if (iw < 0 || iw >= W) continue;
                        const float *xi = x + (((size_t)b * H + ih) * W + iw) * C;
                        float *ai = a + ((size_t)i * kw + j) * C;
                        for (int c = 0; c < C; c++) ai[c] += xi[c] * g[c];
                    }
                }
            }
        }
    }
    memset(dw, 0, sizeof(float) * kk);
    for (int t = 0; t < nth; t++) for (size_t i = 0; i < kk; i++) dw[i] += acc[(size_t)t * kk + i];
    free(acc);
}

/* im2col for a dense convolution, channels last: cols[B,Ho,Wo,kh*kw*C] */
void im2col(const float *x, float *cols, int B, int H, int W, int C, int kh, int kw, int s, int ph, int pw, int Ho, int Wo) {
    #pragma omp parallel for collapse(2) schedule(static)
    for (int b = 0; b < B; b++)
    for (int oh = 0; oh < Ho; oh++) {
        for (int ow = 0; ow < Wo; ow++) {
            float *o = cols + (((size_t)b * Ho + oh) * Wo + ow) * (size_t)kh * kw * C;
            for (int i = 0; i < kh; i++) {
                int ih = oh * s - ph + i;
                for (int j = 0; j < kw; j++) {
                    int iw = ow * s - pw + j;
                    float *oo = o + ((size_t)i * kw + j) * C;
                    if (ih < 0 || ih >= H || iw < 0 || iw >= W) memset(oo, 0, sizeof(float) * C);
                    else memcpy(oo, x + (((size_t)b * H + ih) * W + iw) * C, sizeof(float) * C);
                }
            }
        }
    }
}

/* col2im: dx[B,H,W,C] accumulated from dcols[B,Ho,Wo,kh*kw*C] */
void col2im(const float *dcols, float *dx, int B, int H, int W, int C, int kh, int kw, int s, int ph, int pw, int Ho, int Wo) {
    #pragma omp parallel for collapse(2) schedule(static)
    for (int b = 0; b < B; b++)
    for (int ih = 0; ih < H; ih++) {
        for (int iw = 0; iw < W; iw++) {
            float *d = dx + (((size_t)b * H + ih) * W + iw) * C;
            memset(d, 0, sizeof(float) * C);
            for (int i = 0; i < kh; i++) {
                int t = ih + ph - i; if (t < 0 || t % s) continue; int oh = t / s; if (oh >= Ho) continue;
                for (int j = 0; j < kw; j++) {
                    int u = iw + pw - j; if (u < 0 || u % s) continue; int ow = u / s; if (ow >= Wo) continue;
                    const float *g = dcols + (((size_t)b * Ho + oh) * Wo + ow) * (size_t)kh * kw * C + ((size_t)i * kw + j) * C;
                    for (int c = 0; c < C; c++) d[c] += g[c];
                }
            }
        }
    }
}

/* 2 by 2 max pooling with stride 2 on channels last (used by the ResNet style reference); arg stores the winner index 0..3 */
void maxpool2_fwd(const float *x, float *out, unsigned char *arg, int B, int H, int W, int C) {
    int Ho = H / 2, Wo = W / 2;
    #pragma omp parallel for collapse(2) schedule(static)
    for (int b = 0; b < B; b++)
    for (int oh = 0; oh < Ho; oh++)
        for (int ow = 0; ow < Wo; ow++)
            for (int c = 0; c < C; c++) {
                float best = -INFINITY; unsigned char bi = 0;
                for (int i = 0; i < 2; i++) for (int j = 0; j < 2; j++) {
                    float v = x[(((size_t)b * H + 2 * oh + i) * W + 2 * ow + j) * C + c];
                    if (v > best) { best = v; bi = (unsigned char)(i * 2 + j); }
                }
                size_t o = (((size_t)b * Ho + oh) * Wo + ow) * C + c;
                out[o] = best; arg[o] = bi;
            }
}

void maxpool2_bwd(const float *dy, const unsigned char *arg, float *dx, int B, int H, int W, int C) {
    int Ho = H / 2, Wo = W / 2;
    memset(dx, 0, sizeof(float) * (size_t)B * H * W * C);
    #pragma omp parallel for collapse(2) schedule(static)
    for (int b = 0; b < B; b++)
    for (int oh = 0; oh < Ho; oh++)
        for (int ow = 0; ow < Wo; ow++)
            for (int c = 0; c < C; c++) {
                size_t o = (((size_t)b * Ho + oh) * Wo + ow) * C + c;
                int i = arg[o] / 2, j = arg[o] % 2;
                dx[(((size_t)b * H + 2 * oh + i) * W + 2 * ow + j) * C + c] += dy[o];
            }
}

/* batch normalisation over the leading dimension of x[N,C]: statistics, normalised output (xhat kept for the backward pass) */
void bn_fwd(const float *x, const float *g, const float *b, float *xhat, float *y, float *mean, float *inv, int N, int C, int training, float eps) {
    if (training) {
        int nth = 1;
#ifdef _OPENMP
        nth = omp_get_max_threads();
#endif
        double *acc = (double *)calloc((size_t)nth * 2 * C, sizeof(double));
        #pragma omp parallel
        {
            int t = 0;
#ifdef _OPENMP
            t = omp_get_thread_num();
#endif
            double *s = acc + (size_t)t * 2 * C, *q = s + C;
            #pragma omp for schedule(static)
            for (int n = 0; n < N; n++) {
                const float *xi = x + (size_t)n * C;
                for (int c = 0; c < C; c++) { s[c] += xi[c]; q[c] += (double)xi[c] * xi[c]; }
            }
        }
        for (int c = 0; c < C; c++) {
            double s = 0, q = 0;
            for (int t = 0; t < nth; t++) { s += acc[(size_t)t * 2 * C + c]; q += acc[(size_t)t * 2 * C + C + c]; }
            double m = s / N, v = q / N - m * m; if (v < 0) v = 0;
            mean[c] = (float)m; inv[c] = (float)(1.0 / sqrt(v + eps));
        }
        free(acc);
    }
    #pragma omp parallel for schedule(static)
    for (int n = 0; n < N; n++) {
        const float *xi = x + (size_t)n * C; float *h = xhat + (size_t)n * C, *o = y + (size_t)n * C;
        for (int c = 0; c < C; c++) { float z = (xi[c] - mean[c]) * inv[c]; h[c] = z; o[c] = z * g[c] + b[c]; }
    }
}

/* backward pass of batch normalisation: dx, dg, db from dy and xhat */
void bn_bwd(const float *dy, const float *xhat, const float *g, const float *inv, float *dx, float *dg, float *db, int N, int C, int training) {
    int nth = 1;
#ifdef _OPENMP
    nth = omp_get_max_threads();
#endif
    double *acc = (double *)calloc((size_t)nth * 2 * C, sizeof(double));
    #pragma omp parallel
    {
        int t = 0;
#ifdef _OPENMP
        t = omp_get_thread_num();
#endif
        double *s = acc + (size_t)t * 2 * C, *q = s + C;
        #pragma omp for schedule(static)
        for (int n = 0; n < N; n++) {
            const float *d = dy + (size_t)n * C, *h = xhat + (size_t)n * C;
            for (int c = 0; c < C; c++) { s[c] += d[c]; q[c] += (double)d[c] * h[c]; }
        }
    }
    float *mdy = (float *)malloc(sizeof(float) * C), *mdyh = (float *)malloc(sizeof(float) * C);
    for (int c = 0; c < C; c++) {
        double s = 0, q = 0;
        for (int t = 0; t < nth; t++) { s += acc[(size_t)t * 2 * C + c]; q += acc[(size_t)t * 2 * C + C + c]; }
        db[c] = (float)s; dg[c] = (float)q; mdy[c] = (float)(s / N); mdyh[c] = (float)(q / N);
    }
    #pragma omp parallel for schedule(static)
    for (int n = 0; n < N; n++) {
        const float *d = dy + (size_t)n * C, *h = xhat + (size_t)n * C; float *o = dx + (size_t)n * C;
        if (training) for (int c = 0; c < C; c++) o[c] = inv[c] * g[c] * (d[c] - mdy[c] - h[c] * mdyh[c]);
        else for (int c = 0; c < C; c++) o[c] = inv[c] * g[c] * d[c];
    }
    free(acc); free(mdy); free(mdyh);
}

void relu_fwd(const float *x, float *y, unsigned char *mask, size_t n) {
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; i++) { float v = x[i]; int m = v > 0; y[i] = m ? v : 0.0f; mask[i] = (unsigned char)m; }
}

void relu_bwd(const float *dy, const unsigned char *mask, float *dx, size_t n) {
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; i++) dx[i] = mask[i] ? dy[i] : 0.0f;
}

/* fused batch normalisation followed by ReLU: y = relu(xhat * g + b); mask keeps which units were active */
void bn_relu_fwd(const float *x, const float *g, const float *b, float *xhat, float *y, unsigned char *mask, float *mean, float *inv, int N, int C, int training, float eps) {
    bn_fwd(x, g, b, xhat, y, mean, inv, N, C, training, eps);
    size_t n = (size_t)N * C;
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; i++) { int m = y[i] > 0; mask[i] = (unsigned char)m; if (!m) y[i] = 0.0f; }
}
