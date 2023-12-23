#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cstring>
#include "quants.hpp"
#include "matmul.hpp"

#if defined(__ARM_NEON)
    #include <arm_neon.h>
#endif

void matmulF32(MatmulThreadInfo* a) {
    const float* input = (float*)a->input;
    float* w = (float*)a->weights;
    int d;

#if defined(__ARM_NEON)
    float32x4_t q;
    float32x4_t p;
    float32x4_t z;
    for (d = a->ds; d < a->de; d++) {
        z = vmovq_n_f32(0);
        for (int j = 0; j < a->n; j += 4) {
            q = vld1q_f32(&input[j]);
            p = vld1q_f32(&w[d * a->n + j]);
            z = vfmaq_f32(z, q, p);
        }
        a->output[d] = vaddvq_f32(z);
    }
#else
    for (d = a->ds; d < a->de; d++) {
        float val = 0.0f;
        for (int j = 0; j < a->n; j++) {
            val += w[d * a->n + j] * a->input[j];
        }
        a->output[d] = val;
    }
#endif
}

void matmulF16(MatmulThreadInfo* a) {
    const float* input = (float*)a->input;
    uint16_t* w = (uint16_t*)a->weights;
    int d;
    for (d = a->ds; d < a->de; d++) {
        float val = 0.0f;
        for (int j = 0; j < a->n; j++) {
            float ww = convertF16ToF32(w[d * a->n + j]);
            val += ww * input[j];
        }
        a->output[d] = val;
    }
}

void matmulQ40vQ80(MatmulThreadInfo* a) {
    const BlockQ40* w = (BlockQ40*)a->weights;
    const BlockQ80* input = (BlockQ80*)a->input;
    assert(a->n % QK40 == 0);
    const int n = a->n / QK40;

#if defined(__ARM_NEON)
    float32x4_t sumv0;
    float32x4_t sumv1;
    for (int d = a->ds; d < a->de; d++) {
        sumv0 = vmovq_n_f32(0);
        sumv1 = vmovq_n_f32(0);
        for (int j = 0; j < n; j += 2) {
            const BlockQ40* x0 = &w[d * n + j];
            const BlockQ40* x1 = &w[d * n + j + 1];
            const BlockQ80* y0 = &input[j];
            const BlockQ80* y1 = &input[j + 1];

            const uint8x16_t m4b = vdupq_n_u8(0x0F);
            const int8x16_t  s8b = vdupq_n_s8(0x8);

            const uint8x16_t v0_0 = vld1q_u8(x0->qs);
            const uint8x16_t v0_1 = vld1q_u8(x1->qs);

            // 4-bit -> 8-bit
            const int8x16_t v0_0l = vreinterpretq_s8_u8(vandq_u8  (v0_0, m4b));
            const int8x16_t v0_0h = vreinterpretq_s8_u8(vshrq_n_u8(v0_0, 4));
            const int8x16_t v0_1l = vreinterpretq_s8_u8(vandq_u8  (v0_1, m4b));
            const int8x16_t v0_1h = vreinterpretq_s8_u8(vshrq_n_u8(v0_1, 4));

            // sub 8
            const int8x16_t v0_0ls = vsubq_s8(v0_0l, s8b);
            const int8x16_t v0_0hs = vsubq_s8(v0_0h, s8b);
            const int8x16_t v0_1ls = vsubq_s8(v0_1l, s8b);
            const int8x16_t v0_1hs = vsubq_s8(v0_1h, s8b);

            // load y
            const int8x16_t v1_0l = vld1q_s8(y0->qs);
            const int8x16_t v1_0h = vld1q_s8(y0->qs + 16);
            const int8x16_t v1_1l = vld1q_s8(y1->qs);
            const int8x16_t v1_1h = vld1q_s8(y1->qs + 16);


#if defined(__ARM_FEATURE_DOTPROD)
            const int32x4_t p_0 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), v0_0ls, v1_0l), v0_0hs, v1_0h);
            const int32x4_t p_1 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), v0_1ls, v1_1l), v0_1hs, v1_1h);

            sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(p_0), convertF16ToF32(x0->d)*convertF16ToF32(y0->d));
            sumv1 = vmlaq_n_f32(sumv1, vcvtq_f32_s32(p_1), convertF16ToF32(x1->d)*convertF16ToF32(y1->d));
#else
            const int16x8_t pl0l = vmull_s8(vget_low_s8 (v0_0ls), vget_low_s8 (v1_0l));
            const int16x8_t pl0h = vmull_s8(vget_high_s8(v0_0ls), vget_high_s8(v1_0l));
            const int16x8_t ph0l = vmull_s8(vget_low_s8 (v0_0hs), vget_low_s8 (v1_0h));
            const int16x8_t ph0h = vmull_s8(vget_high_s8(v0_0hs), vget_high_s8(v1_0h));

            const int16x8_t pl1l = vmull_s8(vget_low_s8 (v0_1ls), vget_low_s8 (v1_1l));
            const int16x8_t pl1h = vmull_s8(vget_high_s8(v0_1ls), vget_high_s8(v1_1l));
            const int16x8_t ph1l = vmull_s8(vget_low_s8 (v0_1hs), vget_low_s8 (v1_1h));
            const int16x8_t ph1h = vmull_s8(vget_high_s8(v0_1hs), vget_high_s8(v1_1h));

            const int32x4_t pl0 = vaddq_s32(vpaddlq_s16(pl0l), vpaddlq_s16(pl0h));
            const int32x4_t ph0 = vaddq_s32(vpaddlq_s16(ph0l), vpaddlq_s16(ph0h));
            const int32x4_t pl1 = vaddq_s32(vpaddlq_s16(pl1l), vpaddlq_s16(pl1h));
            const int32x4_t ph1 = vaddq_s32(vpaddlq_s16(ph1l), vpaddlq_s16(ph1h));

            sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(vaddq_s32(pl0, ph0)), convertF16ToF32(x0->d) * convertF16ToF32(y0->d));
            sumv1 = vmlaq_n_f32(sumv1, vcvtq_f32_s32(vaddq_s32(pl1, ph1)), convertF16ToF32(x1->d) * convertF16ToF32(y1->d));
#endif
        }
        a->output[d] = vaddvq_f32(sumv0) + vaddvq_f32(sumv1);
    }
#else
    for (int d = a->ds; d < a->de; d++) {
        float val = 0.0f;
        for (int j = 0; j < n; j++) {
            dequantizeQ40Row(&w[d * n * blocksPerRow + j * blocksPerRow], group, k);
            for (int z = 0; z < k; z++) {
                val += group[z] * a->input[j * k + z];
            }
        }
        a->output[d] = val;
    }
#endif
}

void* matmulThread(void* arg) {
    MatmulThreadInfo* a = (MatmulThreadInfo*)arg;
    for (;;)
    {
        if (pthread_mutex_lock(&a->mutex)) {
            printf("pthread_mutex_lock failed\n");
            exit(EXIT_FAILURE);
        }
        while (!a->hasTask) {
            if (pthread_cond_wait(&a->cond, &a->mutex) != 0) {
                printf("pthread_cond_wait failed\n");
                exit(EXIT_FAILURE);
            }
        }
        a->hasTask = false;
        if (pthread_mutex_unlock(&a->mutex) != 0) {
            printf("pthread_mutex_unlock failed\n");
            exit(EXIT_FAILURE);
        }

        switch (a->type)
        {
            case F32:
                matmulF32(a);
                break;
            case F16:
                matmulF16(a);
                break;
            case Q40:
                matmulQ40vQ80(a);
                break;
            default:
                printf("Unknown float type %d\n", a->type);
                exit(EXIT_FAILURE);
        }

        if (pthread_mutex_lock(&a->mutex)) {
            printf("pthread_mutex_lock failed\n");
            exit(EXIT_FAILURE);
        }
        a->hasResult = true;
        if (pthread_cond_signal(&a->cond) != 0) {
            printf("pthread_mutex_lock failed\n");
            exit(EXIT_FAILURE);
        }
        if (pthread_mutex_unlock(&a->mutex) != 0) {
            printf("pthread_mutex_unlock failed\n");
            exit(EXIT_FAILURE);
        }
    }
    return 0;
}

//     weights      input    output
//   ___________     ___      ___
//   |         |     | |      | |
// d |         | *   | |  = d | |
//   |_________|   n | |      |_|
//        n          |_|       1
//                    1
Matmul::Matmul(int nThread) {
    this->nThread = nThread;
    threads = new MatmulThreadInfo[nThread];
    for (int i = 0; i < nThread; i++) {
        MatmulThreadInfo* thread = &threads[i];
        thread->hasTask = false;
        thread->hasResult = false;
        thread->index = i;

        if (pthread_mutex_init(&thread->mutex, NULL) != 0) {
            printf("pthread_mutex_init failed\n");
            exit(EXIT_FAILURE);
        }
        if (pthread_cond_init(&thread->cond, NULL) != 0) {
            printf("pthread_cond_init failed\n");
            exit(EXIT_FAILURE);
        }
        if (pthread_create(&thread->handler, NULL, matmulThread, (void*)thread) != 0) {
            printf("pthread_create failed\n");
            exit(EXIT_FAILURE);
        }

    }
}

void Matmul::mul(FloatType type, float* output, float* input, void* weights, int n, int d) {
    if (type == Q40) {
        BlockQ80* bq80 = new BlockQ80[n / QK80];
        quantizeQ80Row(input, bq80, n);
        input = (float*)bq80;
    }

    int i;
    for (i = 0; i < nThread; i++) {
        MatmulThreadInfo* s = &threads[i];

        if (pthread_mutex_lock(&s->mutex) != 0) {
            printf("pthread_mutex_lock failed\n");
            exit(EXIT_FAILURE);
        }

        s->output = output;
        s->input = input;
        s->weights = weights;
        s->type = type;
        s->n = n;
        s->ds = i * d / nThread;
        s->de = (i + 1) * d / nThread;
        s->hasTask = true;

        if (pthread_cond_signal(&s->cond) != 0) {
            printf("pthread_cond_signal failed\n");
            exit(EXIT_FAILURE);
        }
        if (pthread_mutex_unlock(&s->mutex) != 0) {
            printf("pthread_mutex_unlock failed\n");
            exit(EXIT_FAILURE);
        }
    }
    for (i = 0; i < nThread; i++) {
        MatmulThreadInfo* thread = &threads[i];

        if (pthread_mutex_lock(&thread->mutex) != 0) {
            printf("pthread_mutex_lock failed\n");
            exit(EXIT_FAILURE);
        }
        while (!thread->hasResult) {
            if (pthread_cond_wait(&thread->cond, &thread->mutex) != 0) {
                printf("pthread_cond_wait failed\n");
                exit(EXIT_FAILURE);
            }
        }
        thread->hasResult = false;
        if (pthread_mutex_unlock(&thread->mutex) != 0) {
            printf("pthread_mutex_unlock failed\n");
            exit(EXIT_FAILURE);
        }
    }

    if (type == Q40) {
        delete[] input;
    }
}

MatMulSlice::MatMulSlice(FloatType type, int sliceCount, int n, int d) {
    assert(d % sliceCount == 0);

    this->type = type;
    this->sliceCount = sliceCount;
    this->d0 = d / sliceCount;
    this->n = n;
    this->weights0Bytes = getBatchBytes(type, this->n, this->d0);
}

long MatMulSlice::splitWeights(int sliceIndex, char* weights, char* weights0) {
    int numbersPerBatch = getNumbersPerBatch(this->type);
    int batchBytes = getBatchBytes(this->type, numbersPerBatch, 1);

    int n = this->n / numbersPerBatch;
    long offset = this->d0 * sliceIndex * n * batchBytes;

    for (int d = 0; d < this->d0; d++) {
        for (int j = 0; j < n; j++) {
            long o = (d * n + j) * batchBytes;

            memcpy(weights0 + o, weights + offset + o, batchBytes);
        }
    }
    return offset; // offset in bytes
}

long MatMulSlice::mergeOutputs(int sliceIndex, float* output, float* output0) {
    long offset = this->d0 * sliceIndex;
    for (int i = 0; i < this->d0; i++) {
        output[offset + i] = output0[i];
    }
    return offset; // offset in floats
}
