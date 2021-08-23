import socket


# ------------------------------------------------------------------------------------------------

class Socket:
    def __init__(self, host='', port=50001):
        self._host = host
        self._port = port
        self._conn = None
        self._sock = None


# ------------------------------------------------------------------------------------------------

    def __del__(self):
        if self._conn:
            self._conn.close()

        if self._sock:
            self._sock.close()


# ------------------------------------------------------------------------------------------------

    def accept(self):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.bind((self._host, self._port))
        self._sock.listen(1)

        print('Accepting connections on port {}'.format(self._port))
        self._conn, addr = self._sock.accept()
        print('Received connection from {}'.format(addr))


# ------------------------------------------------------------------------------------------------

    def connect(self):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.connect((self._host, self._port))
        self._conn = self._sock


# ------------------------------------------------------------------------------------------------

    def send(self, packed):
        self._conn.sendall(packed)


# ------------------------------------------------------------------------------------------------

    def receive(self, n_bytes):
        # return self._conn.recv(n_bytes)
        all_bytes = b''
        while n_bytes > 0:
            crt_bytes = self._conn.recv(n_bytes)
            all_bytes += crt_bytes
            n_bytes -= len(crt_bytes)
        return all_bytes


# ------------------------------------------------------------------------------------------------
