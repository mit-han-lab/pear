import threading


class OperatorPaused(RuntimeError):
    """Raised when the operator revoked the running operation, which is then never a success that was executed."""

    @staticmethod
    def check_cancelled(cancellation: threading.Event | None) -> None:
        """Raises OperatorPaused once the operator has set ``cancellation``."""
        if cancellation is not None and cancellation.is_set():
            raise OperatorPaused("Operation interrupted by the operator")
