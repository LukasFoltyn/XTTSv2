import socket
import sounddevice as sd
import numpy as np

HOST = '127.0.0.1'
PORT = 12345
FS = 24000

def start_server():
    stream = sd.OutputStream(samplerate=FS, channels=1, dtype='float32')
    stream.start()

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.bind((HOST, PORT))
        server_socket.listen(1)
        print(f"Server listening on {HOST}:{PORT}...")

        conn, addr = server_socket.accept()
        with conn:
            print(f"Connected by {addr}")
            while True:
                data = conn.recv(4096)

                try:
                    if data.decode() == "!exit":
                        print("Closing connection...")
                        break
                except UnicodeDecodeError:
                    pass

                stream.write(np.frombuffer(data, dtype=np.float32))

    stream.stop()
    stream.close()


if __name__ == "__main__":
    start_server()
