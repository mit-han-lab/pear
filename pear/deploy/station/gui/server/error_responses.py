from http import HTTPStatus

from pear.utils.logging_utils import get_logger

logger = get_logger(__name__, default_level="INFO")


class ErrorResponses:
    """The answer the page receives for the exception that ended a route or a command, the exception logged with its
    traceback first."""

    OPERATION_ERRORS = (ValueError, RuntimeError)  # what the console raises for an operation it refuses
    REFUSED_STATUS = HTTPStatus.CONFLICT  # of a refused operation, whose answer gives the reason
    FAILED_STATUS = HTTPStatus.INTERNAL_SERVER_ERROR  # of a failure of any other kind
    UNEXPECTED_ERROR_MESSAGE = "Unexpected error; see the service log"  # the answer of a failure of any other kind

    @classmethod
    def answer(cls, error: Exception, failed_call: str) -> tuple[dict[str, str], HTTPStatus]:
        """Returns the body and the status that answer ``error``, which ended ``failed_call`` (a route or a command),
        once it is logged."""
        if isinstance(error, cls.OPERATION_ERRORS):
            logger.warning("%s was refused", failed_call, exc_info=error)
            return {"error": str(error)}, cls.REFUSED_STATUS

        logger.error("%s failed unexpectedly", failed_call, exc_info=error)
        return {"error": cls.UNEXPECTED_ERROR_MESSAGE}, cls.FAILED_STATUS
