#include <pthread.h>
#include "quants.hpp"

#ifndef matmul_hpp
#define matmul_hpp

struct MatmulThreadInfo {
    pthread_t handler;
    pthread_cond_t cond;
    pthread_mutex_t mutex;
    int index;
    bool hasTask;
    bool hasResult;
    bool isStopped;

    float* output;
    void* input;
    void* weights;
    FloatType type;
    int n;
    int ds;
    int de;
};

class Matmul {
private:
    int nThread;
    MatmulThreadInfo* threads;
public:
    Matmul(int nThread);

    void mul(FloatType type, float* output, float* input, void* weights, int n, int d);
};

class MatMulSlice {
public:
    FloatType type;
    int sliceCount;
    int d0;
    int n;
    size_t weights0Bytes;

    MatMulSlice(FloatType type, int sliceCount, int n, int d);
    long splitWeights(int sliceIndex, char* weights, char* weights0);
    long mergeOutputs(int sliceIndex, float* output, float* output0);
};

#endif
