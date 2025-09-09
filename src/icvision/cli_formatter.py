"""
CLI output formatting utilities for ICVision.

Provides colored, formatted output for errors, warnings, success messages,
and progress indicators using only standard Python (no external dependencies).
"""

import sys
from typing import Optional


class CLIFormatter:
    """ANSI color codes and formatting utilities for CLI output."""

    # Color codes
    RED = "\033[91m"
    YELLOW = "\033[93m"
    GREEN = "\033[92m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    GRAY = "\033[90m"

    # Formatting
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"

    # Symbols
    ERROR_SYMBOL = "✗"
    WARNING_SYMBOL = "⚠"
    SUCCESS_SYMBOL = "✓"
    INFO_SYMBOL = "ℹ"
    ARROW = "→"

    @classmethod
    def _supports_color(cls) -> bool:
        """Check if terminal supports ANSI colors."""
        return (
            hasattr(sys.stdout, "isatty")
            and sys.stdout.isatty()
            and sys.platform != "win32"  # Basic check for color support
        )

    @classmethod
    def _colorize(cls, text: str, color: str) -> str:
        """Apply color to text if terminal supports it."""
        if cls._supports_color():
            return f"{color}{text}{cls.RESET}"
        return text

    @classmethod
    def print_header(cls, title: str) -> None:
        """Print a formatted section header."""
        separator = "=" * min(60, len(title) + 20)
        print(f"\n{cls._colorize(separator, cls.CYAN)}")
        print(f"{cls._colorize(cls.BOLD + title + cls.RESET, cls.CYAN)}")
        print(f"{cls._colorize(separator, cls.CYAN)}\n")

    @classmethod
    def print_subheader(cls, title: str) -> None:
        """Print a formatted subsection header."""
        separator = "-" * min(50, len(title) + 10)
        print(f"\n{cls._colorize(title, cls.WHITE + cls.BOLD)}")
        print(f"{cls._colorize(separator, cls.GRAY)}")

    @classmethod
    def print_error(cls, message: str, details: Optional[str] = None, suggestion: Optional[str] = None) -> None:
        """Print a formatted error message."""
        print(f"\n{cls._colorize(cls.BOLD + cls.ERROR_SYMBOL + ' ERROR:', cls.RED)} {message}\n")

        if details:
            print(f"{cls._colorize('Details:', cls.YELLOW)} {details}\n")

        if suggestion:
            print(f"{cls._colorize('How to fix:', cls.GREEN)}")
            for line in suggestion.split("\n"):
                if line.strip():
                    print(f"  {line}")
            print()

        cls._print_usage_hint()

    @classmethod
    def print_warning(cls, message: str, details: Optional[str] = None) -> None:
        """Print a formatted warning message."""
        print(f"\n{cls._colorize(cls.BOLD + cls.WARNING_SYMBOL + ' WARNING:', cls.YELLOW)} {message}")

        if details:
            print(f"{cls._colorize('Details:', cls.YELLOW)} {details}")
        print()

    @classmethod
    def print_success(cls, message: str, details: Optional[str] = None) -> None:
        """Print a formatted success message."""
        print(f"\n{cls._colorize(cls.BOLD + cls.SUCCESS_SYMBOL + ' SUCCESS:', cls.GREEN)} {message}")

        if details:
            print(f"{cls._colorize('Details:', cls.GREEN)} {details}")
        print()

    @classmethod
    def print_info(cls, message: str, details: Optional[str] = None) -> None:
        """Print a formatted info message."""
        print(f"\n{cls._colorize(cls.INFO_SYMBOL + ' INFO:', cls.BLUE)} {message}")

        if details:
            print(f"{cls._colorize('Details:', cls.BLUE)} {details}")
        print()

    @classmethod
    def print_progress(cls, step: int, total: int, message: str) -> None:
        """Print a progress indicator."""
        progress_bar = "█" * (step * 20 // total) + "░" * (20 - (step * 20 // total))
        percentage = (step * 100) // total
        print(f"\r{cls._colorize(f'[{progress_bar}]', cls.CYAN)} {percentage:3d}% - {message}", end="", flush=True)
        if step == total:
            print()  # New line when complete

    @classmethod
    def print_list(cls, title: str, items: list, color: str = None) -> None:
        """Print a formatted list."""
        if color is None:
            color = cls.WHITE

        print(f"{cls._colorize(title, color + cls.BOLD)}")
        for item in items:
            print(f"  • {item}")
        print()

    @classmethod
    def print_summary_stats(cls, stats_dict: dict) -> None:
        """Print formatted summary statistics."""
        cls.print_subheader("Classification Summary")

        for key, value in stats_dict.items():
            # Color code based on key type
            if "error" in key.lower() or "fail" in key.lower():
                color = cls.RED
            elif "warning" in key.lower() or "excluded" in key.lower():
                color = cls.YELLOW
            elif "success" in key.lower() or "complete" in key.lower():
                color = cls.GREEN
            else:
                color = cls.WHITE

            formatted_key = key.replace("_", " ").title()
            print(f"  {cls._colorize('•', color)} {formatted_key}: {cls._colorize(str(value), cls.BOLD)}")
        print()

    @classmethod
    def _print_usage_hint(cls) -> None:
        """Print usage hint with proper formatting."""
        separator = "-" * 60
        print(f"{cls._colorize(separator, cls.GRAY)}")
        print(
            f"{cls._colorize('Usage:', cls.WHITE + cls.BOLD)} "
            "autocleaneeg-icvision raw_data_path [ica_data_path] [options]"
        )
        print(f"{cls._colorize('Help:', cls.WHITE + cls.BOLD)}  autocleaneeg-icvision --help")
        print(f"{cls._colorize(separator, cls.GRAY)}\n")

    @classmethod
    def print_welcome(cls, version: str) -> None:
        """Print welcome message with version."""
        cls.print_header(f"ICVision v{version}")
        print(f"{cls._colorize('Automated ICA component classification using OpenAI Vision API', cls.GRAY)}")
        print()


# Convenience functions for common use cases
def print_error(message: str, details: Optional[str] = None, suggestion: Optional[str] = None) -> None:
    """Print a formatted error message."""
    CLIFormatter.print_error(message, details, suggestion)


def print_warning(message: str, details: Optional[str] = None) -> None:
    """Print a formatted warning message."""
    CLIFormatter.print_warning(message, details)


def print_success(message: str, details: Optional[str] = None) -> None:
    """Print a formatted success message."""
    CLIFormatter.print_success(message, details)


def print_info(message: str, details: Optional[str] = None) -> None:
    """Print a formatted info message."""
    CLIFormatter.print_info(message, details)
