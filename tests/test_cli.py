"""Tests for the CLI."""

import pytest

from sotopia_verifiable.cli.app import app


def test_cli_help():
    """Test that the CLI help command works."""
    exit_code = app(["--help"])
    assert exit_code == 0
    
    exit_code = app(["-h"])
    assert exit_code == 0
    
    exit_code = app(["help"])
    assert exit_code == 0


def test_cli_no_args():
    """Test that the CLI works with no arguments."""
    exit_code = app([])
    assert exit_code == 0


def test_cli_unknown_command():
    """Test that the CLI handles unknown commands."""
    exit_code = app(["unknown_command"])
    assert exit_code == 1


def test_cli_verify_command():
    """Test that the verify command works."""
    exit_code = app(["verify"])
    assert exit_code == 0


def test_cli_list_command():
    """Test that the list command works."""
    exit_code = app(["list"])
    assert exit_code == 0