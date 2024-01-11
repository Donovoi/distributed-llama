#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <arpa/inet.h>
#include <netinet/tcp.h>
#include <errno.h>
#include <string.h>
#include <fcntl.h>
#include "socket.hpp"

#define SOCKET_LAST_ERRCODE errno
#define SOCKET_LAST_ERROR strerror(errno)

#define HELLO_BYTE 0x18

#define UDP_MAX_BYTES 512

static inline void setTcpNotBlocking(int socket) {
    int status = fcntl(socket, F_SETFL, fcntl(status, F_GETFL, 0) | O_NONBLOCK);
    if (status == -1) {
        printf("Error setting socket to non-blocking\n");
        exit(EXIT_FAILURE);
    }
}

static inline void setTcpNoDelay(int socket) {
    int flag = 1;
    int status = setsockopt(socket, IPPROTO_TCP, TCP_NODELAY, (char*)&flag, sizeof(int));
    if (status == -1) {
        printf("Error setting socket to no-delay\n");
        exit(EXIT_FAILURE);
    }
}

static inline void writeTcpSocket(int socket, const char* data, size_t size) {
    while (size > 0) {
        int s = send(socket, (char*)data, size, 0);
        if (s <= 0) {
            if (SOCKET_LAST_ERRCODE == EAGAIN) {
                continue;
            }
            printf("Error sending TCP data %d (%s)\n", SOCKET_LAST_ERRCODE, SOCKET_LAST_ERROR);
            exit(EXIT_FAILURE);
        }
        size -= s;
        data = data + s;
    }
}

static inline void readTcpSocket(int socket, char* data, size_t size) {
    while (size > 0) {
        int r = recv(socket, (char*)data, size, 0);
        if (r <= 0) {
            if (SOCKET_LAST_ERRCODE == EAGAIN) {
                continue;
            }
            printf("Error receiving UDP data %d (%s)\n", SOCKET_LAST_ERRCODE, SOCKET_LAST_ERROR);
            exit(EXIT_FAILURE);
        }
        data = data + r;
        size -= r;
    }
}

static inline void writeUdpSocket(int socket, struct sockaddr_in* addr, const char* data, size_t size) {
    while (size > 0) {
        int packageSize = size > UDP_MAX_BYTES ? UDP_MAX_BYTES : size;
        int s = sendto(socket, data, packageSize, 0, (struct sockaddr*)addr, sizeof(struct sockaddr_in));
        if (s <= 0) {
            printf("Error sending TCP data %d (%s)\n", SOCKET_LAST_ERRCODE, SOCKET_LAST_ERROR);
            exit(EXIT_FAILURE);
        }
        size -= s;
        data = (char*)data + s;
    }
}

static inline void readUdpSocket(int socket, struct sockaddr_in* addr, char* data, size_t size) {
    struct sockaddr_in addr1;
    socklen_t addrSize = sizeof(addr1);
    int nPackets = size / UDP_MAX_BYTES;
    while (size > 0) {
        int packageSize = size > UDP_MAX_BYTES ? UDP_MAX_BYTES : size;
        int r = recvfrom(socket, data, packageSize, 0, (struct sockaddr*)addr, &addrSize);
        if (r <= 0) {
            printf("Error receiving UDP data %d (%s)\n", SOCKET_LAST_ERRCODE, SOCKET_LAST_ERROR);
            exit(EXIT_FAILURE);
        }
        data = (char*)data + r;
        size -= r;
    }
}

SocketPool SocketPool::connect(SocketType type, unsigned int nSockets, char** hosts, int* ports) {
    int* sockets = new int[nSockets];
    struct sockaddr_in* addrs = new struct sockaddr_in[nSockets];
    socklen_t addrSize = sizeof(struct sockaddr_in);
    int clientSocket;

    for (unsigned int i = 0; i < nSockets; i++) {
        memset(&addrs[i], 0, addrSize);
        addrs[i].sin_family = AF_INET;
        addrs[i].sin_addr.s_addr = inet_addr(hosts[i]);
        addrs[i].sin_port = htons(ports[i]);

        if (type == TCP) {
            clientSocket = ::socket(AF_INET, SOCK_STREAM, 0);
            if (clientSocket < 0) {
                printf("Error creating socket\n");
                exit(EXIT_FAILURE);
            }

            int connectResult = ::connect(clientSocket, (struct sockaddr*)&addrs[i], addrSize);
            if (connectResult != 0) {
                printf("Cannot connect to %s:%d (%s)\n", hosts[i], ports[i], SOCKET_LAST_ERROR);
                exit(EXIT_FAILURE);
            }
        } else if (type == UDP) {
            clientSocket = ::socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
            if (clientSocket < 0) {
                printf("Error creating socket\n");
                exit(EXIT_FAILURE);
            }

            char hello = HELLO_BYTE;
            if (sendto(clientSocket, (void*)&hello, sizeof(char), 0, (struct sockaddr*)&addrs[i], addrSize) <= 0) {
                printf("Unable to send message\n");
                exit(EXIT_FAILURE);
            }
        } else {
            printf("Unknown socket type\n");
            exit(EXIT_FAILURE);
        }

        sockets[i] = clientSocket;
    }
    return SocketPool(type, nSockets, sockets, addrs);
}

SocketPool::SocketPool(SocketType type, unsigned int nSockets, int* sockets, struct sockaddr_in* addrs) {
    this->type = type;
    this->nSockets = nSockets;
    this->sockets = sockets;
    this->addrs = addrs;
    this->sentBytes = 0;
    this->recvBytes = 0;
}

SocketPool::~SocketPool() {
    for (unsigned int i = 0; i < nSockets; i++) {
        shutdown(sockets[i], 2);
    }
    delete[] sockets;
    delete[] addrs;
}

void SocketPool::enableTurbo() {
    if (type == TCP) {
        for (unsigned int i = 0; i < nSockets; i++) {
            setTcpNotBlocking(sockets[i]);
            setTcpNoDelay(sockets[i]);
        }
    }
}

void SocketPool::write(unsigned int socketIndex, const char* data, size_t size) {
    assert(socketIndex >= 0 && socketIndex < nSockets);
    sentBytes += size;
    if (type == TCP) {
        writeTcpSocket(sockets[socketIndex], data, size);
    } else if (type == UDP) {
        writeUdpSocket(sockets[socketIndex], &addrs[socketIndex], data, size);
    }
}

void SocketPool::read(unsigned int socketIndex, char* data, size_t size) {
    assert(socketIndex >= 0 && socketIndex < nSockets);
    recvBytes += size;
    if (type == TCP) {
        readTcpSocket(sockets[socketIndex], data, size);
    } else if (type == UDP) {
        readUdpSocket(sockets[socketIndex], &addrs[socketIndex], data, size);
    }
}

void SocketPool::getStats(size_t* sentBytes, size_t* recvBytes) {
    *sentBytes = this->sentBytes;
    *recvBytes = this->recvBytes;
    this->sentBytes = 0;
    this->recvBytes = 0;
}

Socket Socket::accept(SocketType type, int port) {
    const char* host = "0.0.0.0";
    struct sockaddr_in serverAddr;
    struct sockaddr_in clientAddr;
    socklen_t clientAddrSize = sizeof(clientAddr);
    int clientSocket;

    memset(&serverAddr, 0, sizeof(struct sockaddr_in));
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_port = htons(port);
    serverAddr.sin_addr.s_addr = inet_addr(host);

    if (type == TCP) {
        int serverSocket = ::socket(AF_INET, SOCK_STREAM, 0);
        if (serverSocket < 0) {
            printf("Error creating socket\n");
            exit(EXIT_FAILURE);
        }

        int bindResult = bind(serverSocket, (struct sockaddr*)&serverAddr, sizeof(serverAddr));
        if (bindResult < 0) {
            printf("Cannot bind %s:%d\n", host, port);
            exit(EXIT_FAILURE);
        }

        int listenResult = listen(serverSocket, 1);
        if (listenResult != 0) {
            printf("Cannot listen %s:%d\n", host, port);
            exit(EXIT_FAILURE);
        }
        printf("Listening on TCP %s:%d...\n", host, port);

        clientSocket = ::accept(serverSocket, (struct sockaddr*)&clientAddr, &clientAddrSize);
        if (clientSocket < 0) {
            printf("Error accepting connection\n");
            exit(EXIT_FAILURE);
        }

        shutdown(serverSocket, 2);
    } else if (type == UDP) {
        clientSocket = ::socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
        if (clientSocket < 0) {
            printf("Error creating socket\n");
            exit(EXIT_FAILURE);
        }

        if (bind(clientSocket, (struct sockaddr*)&serverAddr, sizeof(serverAddr)) < 0){
            printf("Cannot bind %s:%d\n", host, port);
            exit(EXIT_FAILURE);
        }

        printf("Listening on UDP %s:%d...\n", host, port);

        char hello;
        int recvStatus = recvfrom(clientSocket, (void*)&hello, sizeof(char), 0, (struct sockaddr*)&clientAddr, &clientAddrSize);
        if (recvStatus <= 0){
            printf("Couldn't receive\n");
            exit(EXIT_FAILURE);
        }
        if (hello != HELLO_BYTE) {
            printf("Invalid hello byte\n");
            exit(EXIT_FAILURE);
        }
        printf("recvStatus: %d\n", recvStatus);
    } else {
        printf("Unknown socket type\n");
        exit(EXIT_FAILURE);
    }

    printf("Client connected\n");

    return Socket(type, clientSocket, clientAddr);
}

Socket::Socket(SocketType type, int socket, struct sockaddr_in addr) {
    this->type = type;
    this->socket = socket;
    this->addr = addr;
}

Socket::~Socket() {
    shutdown(socket, 2);
}

void Socket::enableTurbo() {
    if (type == TCP) {
        setTcpNotBlocking(socket);
        setTcpNoDelay(socket);
    }
}

void Socket::write(const char* data, size_t size) {
    if (type == TCP) {
        writeTcpSocket(socket, data, size);
    } else if (type == UDP) {
        writeUdpSocket(socket, &addr, data, size);
    }
}

void Socket::read(char* data, size_t size) {
    if (type == TCP) {
        readTcpSocket(socket, data, size);
    } else if (type == UDP) {
        readUdpSocket(socket, &addr, data, size);
    }
}
