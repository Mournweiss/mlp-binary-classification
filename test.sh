#!/usr/bin/env bash

# =============================================================================
# Test Orchestration Script
# =============================================================================
# This script handles test execution with configurable options:
#   - Smoke tests
#   - Unit tests
#   - Integration tests
#   - Coverage reporting
#   - Specific test selection
#
# Usage:
#   ./test.sh [COMMAND] [OPTIONS]
#
# Commands:
#   all           Run all tests (default)
#   smoke         Run smoke tests only
#   unit          Run unit tests only
#   integration   Run integration tests only
#   coverage      Run tests with coverage report
#   help          Show this help message
#
# Options:
#   --path <PATH>           Run tests in specific directory
#   --pattern <PATTERN>     Match test file names (default: test_*.py)
#   --verbose, -v           Enable verbose output
#   --parallel, -p          Run tests in parallel
#   --help, -h              Show this help message
#
# Examples:
#   ./test.sh all
#   ./test.sh unit --verbose
#   ./test.sh smoke
#   ./test.sh coverage --parallel
#   ./test.sh integration --path tests/integration
# =============================================================================

set -euo pipefail

# ANSI color codes
COLOR_INFO="\033[0m"
COLOR_WARN="\033[1;33m"
COLOR_ERROR="\033[1;31m"
COLOR_SUCCESS="\033[1;32m"
COLOR_RESET="\033[0m"

info()    { echo -e "${COLOR_INFO}[INFO]${COLOR_RESET} $1" >&2; }
warn()    { echo -e "${COLOR_WARN}[WARN]${COLOR_RESET} $1" >&2; }
error()   { echo -e "${COLOR_ERROR}[ERROR]${COLOR_RESET} $1" >&2; exit 1; }
success() { echo -e "${COLOR_SUCCESS}[OK]${COLOR_RESET} $1" >&2; }

# Runtime variables
TEST_MODE="all"
TEST_PATH=""
TEST_PATTERN="test_*.py"
VERBOSE=false
PARALLEL=false
COVERAGE=false
SMOKE=false
EXTRA_ARGS=()

# Shows help message for the test script.
#
# Parameters:
# - None
#
# Returns:
# - None (prints help and exits)
show_help() {
    cat <<'EOF'
Usage: ./test.sh [COMMAND] [OPTIONS]

Commands:
  all           Run all tests (default)
  unit          Run unit tests only
  integration   Run integration tests only
  smoke         Run smoke tests only
  coverage      Run tests with coverage report
  help          Show this help message

Options:
  --path <PATH>           Run tests in specific directory
  --pattern <PATTERN>     Match test file names (default: test_*.py)
  --verbose, -v           Enable verbose output
  --parallel, -p          Run tests in parallel
  --help, -h              Show this help message

Examples:
  ./test.sh all
  ./test.sh unit --verbose
  ./test.sh smoke
  ./test.sh coverage --parallel
  ./test.sh integration --path tests/integration
EOF
}

# Parses command-line arguments and sets global variables for orchestrator and options.
#
# Parameters:
# - $@: array - command-line arguments
#
# Returns:
# - None (sets global shell vars)
parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            all|unit|integration|smoke|coverage|help)
                TEST_MODE="$1"
                shift
                ;;
            --path|-p)
                if [[ -n "${2:-}" && "$2" != "--"* ]]; then
                    TEST_PATH="$2"
                    shift 2
                else
                    error "Missing path argument for --path flag"
                fi
                ;;
            --pattern)
                if [[ -n "${2:-}" && "$2" != "--"* ]]; then
                    TEST_PATTERN="$2"
                    shift 2
                else
                    error "Missing pattern argument for --pattern flag"
                fi
                ;;
            --verbose|-v)
                VERBOSE=true
                shift
                ;;
            --parallel|-p)
                PARALLEL=true
                shift
                ;;
            --help|-h)
                show_help
                exit 0
                ;;
            *)
                warn "Unknown argument: $1 — ignoring"
                shift
                ;;
        esac
    done
}

# Checks and installs/updates project dev dependencies.
#
# Parameters:
# - None
#
# Returns:
# - None (exits on failure)
check_dependencies() {
    # Install/update dev dependencies
    info "Installing/updating dev dependencies..."
    if ! pixi install --environment dev; then
        error "Failed to install dev dependencies"
    fi
    success "Dev dependencies are up to date"
}

# Discovers test directories based on the current TEST_MODE and optional TEST_PATH.
#
# Parameters:
# - None (reads global TEST_MODE and TEST_PATH)
#
# Returns:
# - Outputs space-separated list of test directories to stdout
discover_tests() {
    local test_dirs=()

    case "$TEST_MODE" in
        unit)
            test_dirs+=("tests/unit")
            ;;
        integration)
            test_dirs+=("tests/integration")
            ;;
        smoke)
            test_dirs+=("tests/smoke")
            ;;
        all)
            test_dirs+=("tests/unit" "tests/integration" "tests/smoke")
            ;;
    esac

    # Override with custom path if specified
    if [[ -n "$TEST_PATH" ]]; then
        test_dirs=("$TEST_PATH")
    fi

    for d in "${test_dirs[@]}"; do
        echo "$d"
    done
}

# Runs tests with configured options (verbose, parallel, coverage).
#
# Parameters:
# - None (reads global TEST_MODE, TEST_PATH, TEST_PATTERN, VERBOSE, PARALLEL, COVERAGE)
#
# Returns:
# - None (exits on failure)
run_tests() {
    local test_dirs=()
    while IFS= read -r dir; do
        [[ -n "$dir" ]] && test_dirs+=("$dir")
    done < <(discover_tests)

    if [[ ${#test_dirs[@]} -eq 0 ]]; then
        error "No test directories found."
    fi

    info "============================================"
    info "  Titanic MLP — Test Execution"
    info "============================================"
    info "Mode: ${TEST_MODE}"
    info "Directories: ${test_dirs[*]}"
    info "Pattern: ${TEST_PATTERN}"
    info "Verbose: ${VERBOSE}"
    info "Parallel: ${PARALLEL}"
    info "Coverage: ${COVERAGE}"

    # Build pytest command
    local cmd="pixi run --environment dev python -m pytest"

    # Add smoke marker if smoke mode
    if [[ "$TEST_MODE" == "smoke" ]]; then
        cmd+=" -m smoke"
    fi

    # Add verbose flag
    if $VERBOSE; then
        cmd+=" -v"
    fi

    # Add parallel flag
    if $PARALLEL; then
        cmd+=" -n auto"
    fi

    # Add coverage if requested
    if $COVERAGE; then
        cmd+=" --cov=src/mlp_binary_classification --cov-report=term-missing --cov-report=html"
    fi

    # Add test path or directories
    if [[ -n "$TEST_PATH" ]]; then
        cmd+=" ${TEST_PATH}"
    else
        for dir in "${test_dirs[@]}"; do
            if [[ -d "$dir" ]]; then
                cmd+=" ${dir}"
            else
                warn "Directory not found: ${dir}"
            fi
        done
    fi

    info "Executing: ${cmd}"
    echo ""

    # Run tests
    eval "$cmd"
}

# Main orchestration entrypoint for the test script.
#
# Parameters:
# - $@: array - command-line invocation arguments
#
# Returns:
# - None
main() {
    parse_args "$@"

    case "$TEST_MODE" in
        help)
            show_help
            ;;
        *)
            check_dependencies
            run_tests
            ;;
    esac
}

main "$@"
