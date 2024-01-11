#ifndef SOCKET_HPP
#define SOCKET_HPP

#include <cstddef>
#include <netinet/in.h>

enum SocketType {
    TCP,
    UDP
};

class SocketPool {
private:
    int* sockets;
    unsigned int sentBytes;
    unsigned int recvBytes;

public:
    static SocketPool connect(SocketType type, unsigned int nSockets, char** hosts, int* ports);

    SocketType type;
    unsigned int nSockets;
    struct sockaddr_in* addrs;

    SocketPool(SocketType type, unsigned int nSockets, int* sockets, struct sockaddr_in* addrs);
    ~SocketPool();

    void enableTurbo();
    void write(unsigned int socketIndex, const char* data, size_t size);
    void read(unsigned int socketIndex, char* data, size_t size);
    void getStats(size_t* sentBytes, size_t* recvBytes);
};

class Socket {
private:
    SocketType type;
    int socket;
    struct sockaddr_in addr;

public:
    static Socket accept(SocketType type, int port);

    Socket(SocketType type, int socket, struct sockaddr_in addr);
    ~Socket();

    void enableTurbo();
    void write(const char* data, size_t size);
    void read(char* data, size_t size);
};

#endif
