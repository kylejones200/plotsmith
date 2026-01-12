"""Custom exceptions for PlotSmith."""


class PlotSmithError(Exception):
    """Base exception for all PlotSmith errors."""

    pass


class ValidationError(PlotSmithError):
    """Raised when data validation fails.

    Attributes:
        message: Error message describing the validation failure.
        context: Optional dictionary with additional context.
    """

    def __init__(self, message: str, context: dict | None = None) -> None:
        """Initialize validation error.

        Args:
            message: Error message.
            context: Optional context dictionary.
        """
        super().__init__(message)
        self.message = message
        self.context = context or {}


class PlotError(PlotSmithError):
    """Raised when plotting/drawing operations fail.

    Attributes:
        message: Error message describing the plot failure.
        plot_type: Optional plot type that failed.
    """

    def __init__(self, message: str, plot_type: str | None = None) -> None:
        """Initialize plot error.

        Args:
            message: Error message.
            plot_type: Optional plot type.
        """
        super().__init__(message)
        self.message = message
        self.plot_type = plot_type


class TaskError(PlotSmithError):
    """Raised when task execution fails.

    Attributes:
        message: Error message describing the task failure.
        task_name: Optional task name that failed.
    """

    def __init__(self, message: str, task_name: str | None = None) -> None:
        """Initialize task error.

        Args:
            message: Error message.
            task_name: Optional task name.
        """
        super().__init__(message)
        self.message = message
        self.task_name = task_name

