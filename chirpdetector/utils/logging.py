"""Logging utilities used in all modules."""

from typing import Self
from contextlib import ContextDecorator
import time

import logging
import pathlib
from rich.console import Console


class Timer(ContextDecorator):
    """A simple timer class to time the execution of a block of code."""

    def __init__(
            self: Self, console: Console, message: str, verbosity: int = 1
        ) -> None:
        """Initialize the timer."""
        self.console = console
        self.message = message
        self.verbosity = verbosity

    def __enter__(self: Self) -> Self:
        """Start the timer."""
        self.start_time = time.time()
        return self

    def __exit__(
            self: Self, exc_type: None, exc_value: None, traceback: None
        ) -> None:
        """Stop the timer and log the elapsed time."""
        elapsed_time = time.time() - self.start_time
        msg = (
            f"[bold green]Execution time :[/bold green] {elapsed_time:.4f} s :"
            f" {self.message}"
        )
        if self.verbosity > 0:
            self.console.log(msg)

        if exc_type is not None:
            msg = (
                f"[bold red]Exception:[/bold red] "
                f"{exc_type.__name__}: {exc_value}"
            )
            self.console.log(msg)
            msg = (
                f"[bold red]Traceback:[/bold red] {traceback}"
            )
            self.console.log(msg)

def make_logger(name: str, logfile: pathlib.Path) -> logging.Logger:
    """Create a logger for the script.

    Parameters
    ----------
    - `name`: `str`
        Name of the logger.
    - `logfile`: `pathlib.Path`
        Path to the log file.

    Returns
    -------
    - `logging.Logger`
        Logger object.
    """
    # create logger formats for file and terminal
    file_formatter = logging.Formatter(
        """[ %(levelname)s ] ~ %(asctime)s ~ %(name)s.%(funcName)s:%(lineno)d:
        %(message)s""",
    )
    console_formatter = logging.Formatter(
        "[ %(levelname)s ] in %(name)s.%(funcName)s:%(lineno)d: %(message)s",
    )

    # create stream handler for terminal output
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(logging.INFO)

    # create stream handler for file output
    file_handler = logging.FileHandler(logfile)
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.INFO)

    # create script specific logger
    logger = logging.getLogger(name)
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)

    return logger
