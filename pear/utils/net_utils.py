import socket


def free_port(preferred: int) -> int:
    """Returns ``preferred`` if it is free, else a port picked by the OS."""
    for candidate in (preferred, 0):  # 0: the OS picks a free port
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("0.0.0.0", candidate))
                return sock.getsockname()[1]
            except OSError:
                continue

    raise RuntimeError("no free TCP port")
