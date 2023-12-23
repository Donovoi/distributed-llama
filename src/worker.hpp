#include "transformer.hpp"

#ifndef worker_hpp
#define worker_hpp

#define ACTION_HELLO 0
#define ACTION_CREATE_FRAGMENT 1
#define ACTION_FORWARD_FRAGMENT 2
#define ACTION_SEND_BUFFER 3

class WorkerRemoteClient: public RemoteClient {
private:
    int sliceCount;
    int* clientSockets;
    long* waitBufferTime;
    long* transferBufferTime;
public:

    WorkerRemoteClient(TransformerSpec* spec, char** hosts, int* ports);
    ~WorkerRemoteClient();
    void createFragment(uint8_t sliceIndex, uint8_t layerIndex, uint8_t type, char* weights, size_t bytes);
    void forwardFragment(uint8_t sliceIndex, uint8_t layerIndex, uint8_t type);
    void sendBuffer(uint8_t sliceIndex, uint8_t bufferIndex, void* data, size_t bytes);
    void readBuffer(uint8_t sliceIndex, uint8_t bufferIndex, void* data, size_t bytes);
    void dumpStatistics();
private:
    void sendBytes(uint8_t sliceIndex, void* data, size_t bytes);
    void readBytes(uint8_t sliceIndex, void* data, size_t bytes);
};

struct WorkerLayer {
    NativeTransformerBlockQkv* qkv;
    NativeTransformerBlockAtt* att;
    NativeTransformerBlockFfn* ffn;
    NativeTransformerBlockFfn2* ffn2;
};

class Worker {
public:
    static void serve(TransformerConfig* config, int port);

private:
    int clientSocket;
    SharedBuffer* buffer;
    TransformerState* state;
    WorkerLayer* layers;
    TransformerConfig* config;
    TransformerSpec spec;
    uint8_t sliceIndex;

public:
    Worker(TransformerConfig* config, int clientSocket);
    void readSocket(void* data, size_t bytes);
    void writeSocket(void* data, size_t bytes);
    void listen();
    void handleHello();
    void handleCreateFragment();
    void handleSendBuffer();
    void handleForwardFragment();
};

class WorkerTransformerState: public TransformerState {
private:
    SharedBuffer* buffer;
    Worker* worker;
public:
    WorkerTransformerState(SharedBuffer* buffer, Worker* worker, TransformerConfig* config);
    char* getSlicedBuffer(uint8_t bufferIndex, uint8_t sliceIndex);
    char* getUnitBuffer(uint8_t bufferIndex);
    void readSlicedBuffer(uint8_t bufferIndex, uint8_t sliceIndex);
    void sendSlicedBuffer(uint8_t bufferIndex, uint8_t sliceIndex);
    void sendUnitBuffer(uint8_t bufferIndex, uint8_t sliceIndex);
};

#endif
