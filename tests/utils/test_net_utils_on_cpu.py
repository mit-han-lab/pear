import socket

from pear.utils.net_utils import free_port


def test_a_port_that_is_taken_is_replaced_by_one_the_os_picks():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as taken_socket:
        taken_socket.bind(("0.0.0.0", 0))
        taken_port = taken_socket.getsockname()[1]

        assert free_port(taken_port) != taken_port

    assert free_port(taken_port) == taken_port
