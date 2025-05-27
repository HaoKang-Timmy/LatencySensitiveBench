import socket
class SocketConfig():
    def __init__(self,Host,Port,max_client):
        self.Host = Host
        self.Port = Port
        self.max_client = max_client
    def listen(self):
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.bind((self.Host, self.Port))
            ###
            self.socket.listen(self.max_client)
            print(f"[+] Socket is listening on port {self.Port}")  # Print port number
            self.conn, addr = self.socket.accept()
            print(f"[+] Connected to client at {addr}")  # Print client address
    def __del__(self):
        self.conn.close()
        self.socket.close()
        print(f"[-] Socket on port {self.Port} closed.")
    
        