import functools
import logging
from logging.config import dictConfig

log_config = {
    "version": 1,
    "formatters": {
        "detailed": {
            "format": "[{asctime} {levelname}] [{process: d} {thread: d} {module} {funcName} {lineno: d}]: {message}",
            "datefmt": "%m/%d/%Y %I:%M:%S %p",
            "style": "{",
        },
        "simple": {
            "format": "[{asctime} {levelname}] [{module} {lineno: d}]: {message}",
            "datefmt": "%m/%d/%Y %I:%M:%S %p",
            "style": "{",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "WARNING",
            "formatter": "simple",
        },
        "logfile": {
            "class": "logging.FileHandler",
            "filename": "selftune.log",
            "level": "DEBUG",
            "formatter": "detailed",
        },
    },
    "loggers": {
        "selftune_core.backends.bluefin.bluefin": {
            "level": "DEBUG",
            "handlers": [
                "console",
                "logfile",
            ],
            "propagate": False,
        }
    },
}

dictConfig(log_config)


def get_logger(logger_name: str) -> logging.Logger:
    return logging.getLogger(name=logger_name)


def log_args_and_return(_func=None, logger=None):
    def decorator_log(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if logger is None:
                _logger = logging.getLogger(func.__name__)
            else:
                _logger = logger
            args_repr = [repr(a) for a in args]
            kwargs_repr = [f"{k}={v}" for k, v in kwargs.items()]
            signature = ", ".join(args_repr + kwargs_repr)
            _logger.info("Function %s called with args %s" % (func.__name__, signature))
            try:
                result = func(*args, **kwargs)
                _logger.info("Function %s returns %s" % (func.__name__, result))
                return result
            except Exception as e:
                _logger.critical("Exception raised in %s. exception: %s", func.__name__, str(e))
                raise e

        return wrapper

    if _func is None:
        return decorator_log
    else:
        return decorator_log(_func)
